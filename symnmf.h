#ifndef SYMNMF_H_
#define SYMNMF_H_

/* Prototypes used by symnmfmodule.c, implemented in symnmf.c. */

/* Allocate a rows x cols matrix. */
double **alloc_matrix(int rows, int cols);

/* Free a matrix allocated with alloc_matrix(). */
void free_matrix(double **matrix, int rows);

/* Print matrix */
void print_matrix(double **matrix, int rows, int cols);

/* Read datapoints from a file, discovering n and d from the file itself. */
double **load_points(const char *path, int *n, int *d);

/* Compute the similarity matrix A. */
double **sym(double **points, int n, int d);

/* Compute the diagonal degree matrix D. */
double **ddg(double **matrix, int n);

/* Compute the normalized similarity matrix W. */
double **norm(double **matrix, int n);

/* Compute the matrix a goal asks for. */
double **compute_goal(const char *goal, double **points, int n, int d);

/* Optimize H until it converges or MAX_ITER is reached. */
double **symnmf(double **h0, double **w, int n, int k);

#endif
