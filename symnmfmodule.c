#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "symnmf.h"

/* Python C API wrapper: converts Python lists to the double** matrices
 * symnmf.c works with, calls into it, and converts the result back. */

/* Copy a Python row into an allocated C row; 1 and an exception on error. */
static int fill_row(PyObject *row, double *destination, int cols) {
    PyObject *item;
    int j;

    if (!PyList_Check(row) || (int)PyList_Size(row) != cols) {
        PyErr_SetString(PyExc_ValueError, "datapoints must be a rectangular list");
        return 1;
    }
    for (j = 0; j < cols; j++) {
        item = PyList_GetItem(row, j);
        destination[j] = PyFloat_AsDouble(item);
        /* -1.0 is both a valid value and the error return, check the flag. */
        if (PyErr_Occurred() != NULL) {
            return 1;
        }
    }
    return 0;
}

/* Convert a Python list of lists into a C matrix. */
static double **matrix_from_list(PyObject *object, int *rows, int *cols) {
    double **matrix;
    int i;

    if (!PyList_Check(object) || PyList_Size(object) <= 0) {
        PyErr_SetString(PyExc_ValueError, "datapoints must be a non-empty list");
        return NULL;
    }
    *rows = (int)PyList_Size(object);
    if (!PyList_Check(PyList_GetItem(object, 0))) {
        PyErr_SetString(PyExc_ValueError, "datapoints must be a list of lists");
        return NULL;
    }
    *cols = (int)PyList_Size(PyList_GetItem(object, 0));
    matrix = alloc_matrix(*rows, *cols);
    if (matrix == NULL) {
        PyErr_NoMemory();
        return NULL;
    }
    for (i = 0; i < *rows; i++) {
        if (fill_row(PyList_GetItem(object, i), matrix[i], *cols) != 0) {
            free_matrix(matrix, *rows);
            return NULL;
        }
    }
    return matrix;
}

/* Convert a C matrix into a new Python list of lists. */
static PyObject *list_from_matrix(double **matrix, int rows, int cols) {
    PyObject *result, *row;
    int i, j;

    result = PyList_New(rows);
    if (result == NULL) {
        return NULL;
    }
    for (i = 0; i < rows; i++) {
        row = PyList_New(cols);
        if (row == NULL) {
            Py_DECREF(result);
            return NULL;
        }
        for (j = 0; j < cols; j++) {
            PyList_SetItem(row, j, PyFloat_FromDouble(matrix[i][j]));
        }
        /* PyList_SetItem steals the reference, so row is not decremented. */
        PyList_SetItem(result, i, row);
    }
    return result;
}

/* Run one of the n x n goals on datapoints supplied from Python. */
static PyObject *run_goal(PyObject *args, const char *goal) {
    PyObject *points_object, *result;
    double **points, **matrix;
    int n, d;

    if (!PyArg_ParseTuple(args, "O", &points_object)) {
        return NULL;
    }
    points = matrix_from_list(points_object, &n, &d);   /* Python -> C */
    if (points == NULL) {
        return NULL;
    }
    matrix = compute_goal(goal, points, n, d);
    free_matrix(points, n);   /* not needed once the goal is computed */
    if (matrix == NULL) {
        return PyErr_NoMemory();
    }
    result = list_from_matrix(matrix, n, n);   /* C -> Python */
    free_matrix(matrix, n);
    return result;
}

/* sym(X) -> the similarity matrix of the datapoints X. */
static PyObject *api_sym(PyObject *self, PyObject *args) {
    (void)self;
    return run_goal(args, "sym");
}

/* ddg(X) -> the diagonal degree matrix of the datapoints X. */
static PyObject *api_ddg(PyObject *self, PyObject *args) {
    (void)self;
    return run_goal(args, "ddg");
}

/* norm(X) -> the normalized similarity matrix of the datapoints X. */
static PyObject *api_norm(PyObject *self, PyObject *args) {
    (void)self;
    return run_goal(args, "norm");
}

/* Check that W is n x n for an n x k H. */
static int check_w_shape(int rows, int cols, int n) {
    if (rows != n || cols != n) {
        PyErr_SetString(PyExc_ValueError, "W must be n x n for an n x k H");
        return 1;
    }
    return 0;
}

/* symnmf(H0, W) -> the optimized association matrix H.
 * H is initialized in Python, so both matrices arrive from the caller. */
static PyObject *api_symnmf(PyObject *self, PyObject *args) {
    PyObject *h_object, *w_object, *result;
    double **h0, **w, **h;
    int n, k, w_rows, w_cols;

    (void)self;
    if (!PyArg_ParseTuple(args, "OO", &h_object, &w_object)) {
        return NULL;
    }
    h0 = matrix_from_list(h_object, &n, &k);
    if (h0 == NULL) {
        return NULL;
    }
    w = matrix_from_list(w_object, &w_rows, &w_cols);
    if (w == NULL) {
        free_matrix(h0, n);
        return NULL;
    }
    if (check_w_shape(w_rows, w_cols, n) != 0) {
        free_matrix(h0, n);
        free_matrix(w, w_rows);
        return NULL;
    }
    h = symnmf(h0, w, n, k);
    free_matrix(h0, n);   /* ours: symnmf() never frees its inputs */
    free_matrix(w, w_rows);
    if (h == NULL) {
        PyErr_NoMemory();
        return NULL;
    }
    result = list_from_matrix(h, n, k);
    free_matrix(h, n);
    return result;
}

/* The functions Python can call. */
static PyMethodDef symnmfMethods[] = {
    {"symnmf", (PyCFunction)api_symnmf, METH_VARARGS,
     PyDoc_STR("symnmf(H0, W) -> optimized association matrix H")},
    {"sym", (PyCFunction)api_sym, METH_VARARGS,
     PyDoc_STR("sym(X) -> similarity matrix of the datapoints X")},
    {"ddg", (PyCFunction)api_ddg, METH_VARARGS,
     PyDoc_STR("ddg(X) -> diagonal degree matrix of the datapoints X")},
    {"norm", (PyCFunction)api_norm, METH_VARARGS,
     PyDoc_STR("norm(X) -> normalized similarity matrix of the datapoints X")},
    {NULL, NULL, 0, NULL}
};

/* The module itself. */
static struct PyModuleDef symnmfmodule = {
    PyModuleDef_HEAD_INIT,
    "symnmfmodule",
    "C implementation of the symNMF matrix goals",
    -1,
    symnmfMethods
};

/* Runs on import; the name has to match the module name. */
PyMODINIT_FUNC PyInit_symnmfmodule(void) {
    return PyModule_Create(&symnmfmodule);
}
