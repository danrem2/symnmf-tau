#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "symnmf.h"

/* Optimization constants. */
#define BETA 0.5
#define EPSILON 1e-4
#define MAX_ITER 300

/* Report whether a line holds only blanks, so it can be skipped rather than
 * counted as a datapoint. */
static int is_blank(const char *line) {
    while (*line != '\0') {
        if (*line != ' ' && *line != '\t' && *line != '\r') {
            return 0;
        }
        line++;
    }
    return 1;
}

/* Read one line, caller frees; NULL at EOF or, with *err set, on failure. */
static char *read_line(FILE *fp, int *err) {
    int capacity = 128, length = 0, c;
    char *buffer, *grown;

    *err = 0;
    buffer = (char *)malloc(capacity);
    if (buffer == NULL) {
        *err = 1;
        return NULL;
    }
    c = fgetc(fp);
    if (c == EOF) {
        free(buffer);
        return NULL;
    }
    while (c != EOF && c != '\n') {
        if (length + 1 >= capacity) {
            capacity *= 2;
            grown = (char *)realloc(buffer, capacity);
            if (grown == NULL) {
                free(buffer);
                *err = 1;
                return NULL;
            }
            buffer = grown;
        }
        buffer[length++] = (char)c;
        c = fgetc(fp);
    }
    buffer[length] = '\0';
    return buffer;
}

/* Parse a line of comma separated doubles, or -1 if malformed; blanks around a
 * value are accepted, a trailing comma is not. A NULL values only counts. */
static int parse_line(const char *line, double *values, int capacity) {
    const char *position = line;
    char *end;
    int count = 0;

    for (;;) {
        double value = strtod(position, &end);
        if (end == position) {
            return -1;
        }
        if (values != NULL) {
            if (count >= capacity) {
                return -1;
            }
            values[count] = value;
        }
        count++;
        while (*end == ' ' || *end == '\t' || *end == '\r') {
            end++;
        }
        if (*end == '\0') {
            return count;
        }
        if (*end != ',') {
            return -1;
        }
        position = end + 1;
    }
}

/* First pass: find n and d so load_points() allocates once; 1 if unusable. */
static int count_dimensions(FILE *fp, int *n, int *d) {
    char *line;
    int err, values;

    *n = 0;
    *d = 0;
    for (;;) {
        line = read_line(fp, &err);
        if (line == NULL) {
            return err ? 1 : (*n > 0 ? 0 : 1);
        }
        if (is_blank(line)) {
            free(line);
            continue;
        }
        values = parse_line(line, NULL, 0);
        free(line);
        if (values <= 0 || (*d != 0 && values != *d)) {
            return 1;
        }
        *d = values;
        (*n)++;
    }
}

/* Second pass: fill the already allocated matrix, skipping blanks as before. */
static int read_points(FILE *fp, double **points, int n, int d) {
    char *line;
    int err, row = 0;

    while (row < n) {
        line = read_line(fp, &err);
        if (line == NULL) {
            return 1;
        }
        if (is_blank(line)) {
            free(line);
            continue;
        }
        if (parse_line(line, points[row], d) != d) {
            free(line);
            return 1;
        }
        free(line);
        row++;
    }
    return 0;
}

/* Allocate a rows x cols matrix. */
double **alloc_matrix(int rows, int cols) {
    double **matrix;
    int i;

    matrix = (double **)malloc(rows * sizeof(double *));
    if (matrix == NULL) {
        return NULL;
    }
    for (i = 0; i < rows; i++) {
        matrix[i] = (double *)calloc(cols, sizeof(double));
        if (matrix[i] == NULL) {
            free_matrix(matrix, i); /* release the rows already allocated */
            return NULL;
        }
    }
    return matrix;
}

/* Free a matrix allocated with alloc_matrix(). */
void free_matrix(double **matrix, int rows) {
    int i;

    if (matrix == NULL) {
        return;
    }
    for (i = 0; i < rows; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

/* Print matrix */
void print_matrix(double **matrix, int rows, int cols) {
    int i, j;

    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            if (j > 0) {
                printf(",");
            }
            printf("%.4f", matrix[i][j]);
        }
        printf("\n");
    }
}

/* Read datapoints from a file, discovering n and d from the file itself. */
double **load_points(const char *path, int *n, int *d) {
    FILE *fp;
    double **points;

    fp = fopen(path, "r");
    if (fp == NULL) {
        return NULL;
    }
    if (count_dimensions(fp, n, d) != 0) {
        fclose(fp);
        return NULL;
    }
    rewind(fp); /* two passes: count exactly, then fill */
    points = alloc_matrix(*n, *d);
    if (points == NULL || read_points(fp, points, *n, *d) != 0) {
        free_matrix(points, *n);
        fclose(fp);
        return NULL;
    }
    fclose(fp);
    return points;
}

/* Squared Euclidean distance between two datapoints; no square root taken. */
static double squared_distance(const double *a, const double *b, int d) {
    double sum = 0.0, diff;
    int t;

    for (t = 0; t < d; t++) {
        diff = a[t] - b[t];
        sum += diff * diff;
    }
    return sum;
}

/* Compute the similarity matrix A. */
double **sym(double **points, int n, int d) {
    double **matrix;
    int i, j;

    matrix = alloc_matrix(n, n);
    if (matrix == NULL) {
        return NULL;
    }
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            /* a_ij = exp(-||x_i - x_j||^2 / 2), and a_ii = 0 by definition. */
            if (i != j) {
                matrix[i][j] = exp(-squared_distance(points[i], points[j], d)
                                   / 2.0);
            }
        }
    }
    return matrix;
}

/* Row sums of the similarity matrix: the diagonal of D, as a vector. */
static double *degrees(double **matrix, int n) {
    double *result;
    double sum;
    int i, j;

    result = (double *)malloc(n * sizeof(double));
    if (result == NULL) {
        return NULL;
    }
    for (i = 0; i < n; i++) {
        sum = 0.0;
        for (j = 0; j < n; j++) {
            sum += matrix[i][j];
        }
        result[i] = sum;
    }
    return result;
}

/* Compute the diagonal degree matrix D. */
double **ddg(double **matrix, int n) {
    double *degree;
    double **result;
    int i;

    degree = degrees(matrix, n);
    if (degree == NULL) {
        return NULL;
    }
    /* alloc_matrix() zero fills, so only the diagonal has to be written. */
    result = alloc_matrix(n, n);
    if (result == NULL) {
        free(degree); /* ours to release before bailing out */
        return NULL;
    }
    for (i = 0; i < n; i++) {
        result[i][i] = degree[i];
    }
    free(degree);
    return result;
}

/* Compute the normalized similarity matrix W. */
double **norm(double **matrix, int n) {
    double *degree;
    double **result;
    int i, j;

    degree = degrees(matrix, n);
    if (degree == NULL) {
        return NULL;
    }
    result = alloc_matrix(n, n);
    if (result == NULL) {
        free(degree); /* ours to release before bailing out */
        return NULL;
    }
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            /* W = D^(-1/2) A D^(-1/2) in closed form, O(n^2) and with no
             * D^(-1/2) built; d_i > 0 since off-diagonal a_ij are positive. */
            result[i][j] = matrix[i][j] / sqrt(degree[i] * degree[j]);
        }
    }
    free(degree);
    return result;
}

/* Check if the goal is supported here; symnmf runs only in Python. */
static int is_valid_goal(const char *goal) {
    return strcmp(goal, "sym") == 0 || strcmp(goal, "ddg") == 0
        || strcmp(goal, "norm") == 0;
}

/* Compute the matrix a goal asks for. */
double **compute_goal(const char *goal, double **points, int n, int d) {
    double **similarity;
    double **result;

    similarity = sym(points, n, d);
    if (similarity == NULL) {
        return NULL;
    }
    if (strcmp(goal, "sym") == 0) {
        return similarity;
    }
    result = strcmp(goal, "ddg") == 0 ? ddg(similarity, n)
                                      : norm(similarity, n);
    free_matrix(similarity, n);
    return result;
}

/* Multiply two matrices. */
static double **multiply(double **a, double **b, int n, int m, int p) {
    double **result;
    double sum;
    int i, j, t;

    result = alloc_matrix(n, p);
    if (result == NULL) {
        return NULL;
    }
    for (i = 0; i < n; i++) {
        for (j = 0; j < p; j++) {
            sum = 0.0;
            for (t = 0; t < m; t++) {
                sum += a[i][t] * b[t][j];
            }
            result[i][j] = sum;
        }
    }
    return result;
}

/* Compute H^T H without building the transpose. */
static double **transpose_product(double **h, int n, int k) {
    double **result;
    double sum;
    int i, j, r;

    result = alloc_matrix(k, k);
    if (result == NULL) {
        return NULL;
    }
    for (i = 0; i < k; i++) {
        for (j = 0; j < k; j++) {
            sum = 0.0;
            for (r = 0; r < n; r++) {
                sum += h[r][i] * h[r][j];
            }
            result[i][j] = sum;
        }
    }
    return result;
}

/* Squared Frobenius norm of a - b. */
static double squared_difference(double **a, double **b, int n, int k) {
    double sum = 0.0, difference;
    int i, j;

    for (i = 0; i < n; i++) {
        for (j = 0; j < k; j++) {
            difference = a[i][j] - b[i][j];
            sum += difference * difference;
        }
    }
    return sum;
}

/* Compute the next H from the current one. */
static double **update_h(double **h, double **w, int n, int k) {
    double **numerator, **square, **denominator, **result = NULL;
    int i, j;

    numerator = multiply(w, h, n, n, k);
    square = transpose_product(h, n, k); /* k x k, not the n x n of H H^T */
    denominator = (square == NULL) ? NULL : multiply(h, square, n, k, k);
    free_matrix(square, k);
    if (numerator != NULL && denominator != NULL) {
        result = alloc_matrix(n, k);
    }
    for (i = 0; result != NULL && i < n; i++) {
        for (j = 0; j < k; j++) {
            /* Never zero: entries stay positive, scaled by >= 1-beta. */
            result[i][j] = h[i][j] * (1.0 - BETA
                                      + BETA * numerator[i][j] / denominator[i][j]);
        }
    }
    free_matrix(numerator, n);
    free_matrix(denominator, n);
    return result;
}

/* Optimize H until it converges or MAX_ITER is reached. */
double **symnmf(double **h0, double **w, int n, int k) {
    double **current = h0;
    double **next;
    int iteration, converged = 0;

    for (iteration = 0; iteration < MAX_ITER && !converged; iteration++) {
        next = update_h(current, w, n, k);
        if (next == NULL) {
            /* h0 was allocated by the caller, which frees it. */
            if (current != h0) {
                free_matrix(current, n);
            }
            return NULL;
        }
        converged = squared_difference(next, current, n, k) < EPSILON;
        if (current != h0) {
            free_matrix(current, n);
        }
        current = next;
    }
    return current;
}

int main(int argc, char **argv) {
    double **points, **matrix;
    int n, d;

    if (argc != 3 || !is_valid_goal(argv[1]))
    { /* bad arg count or goal */
        printf("An Error Has Occurred\n");
        return 1;
    }
    points = load_points(argv[2], &n, &d);
    if (points == NULL)
    { /* unreadable or malformed file */
        printf("An Error Has Occurred\n");
        return 1;
    }
    matrix = compute_goal(argv[1], points, n, d);
    free_matrix(points, n); /* not needed once the goal is computed */
    if (matrix == NULL)
    { /* allocation failure */
        printf("An Error Has Occurred\n");
        return 1;
    }
    print_matrix(matrix, n, n);
    free_matrix(matrix, n);
    return 0;
}
