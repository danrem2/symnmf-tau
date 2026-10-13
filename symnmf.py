

import sys

import numpy as np

import symnmfmodule

np.random.seed(1234)

GOALS = ("symnmf", "sym", "ddg", "norm")
ERROR_MESSAGE = "An Error Has Occurred"
K_ERROR_MESSAGE = "Incorrect number of clusters!"


def fail(message):
    """Print an error message and terminate the program."""
    print(message)
    sys.exit(1)


def read_points(path):
    """Read the datapoints from a file, skipping blank lines, as symnmf.c does.
    """
    with open(path, "r") as handle:
        # Blank lines are dropped here, not by np.loadtxt, which rejects a
        # whitespace-only line and only warns on an empty file.
        lines = [line for line in handle if line.strip()]
    if not lines:
        raise ValueError("no datapoints")
    # ndmin=2 keeps a single row, or a single column, from collapsing to 1-D.
    return np.loadtxt(lines, delimiter=",", ndmin=2)


def parse_k(text):
    """Validate the k argument and return it as an int, or exit on a bad value."""
    try:
        value = float(text)
        # Parsed as a float because int("3.0") would raise; int() in turn rejects
        # the inf and nan that float() accepts, so the test sits inside the guard.
        if value != int(value) or int(value) <= 1:
            # only the lower bound is checked here; main() applies k < N once N is known.
            fail(K_ERROR_MESSAGE)
    except (ValueError, OverflowError):
        fail(K_ERROR_MESSAGE)
    return int(value)


def print_matrix(matrix):
    """Print a matrix as comma-separated rows with four decimals."""
    for row in matrix:
        print(",".join("%.4f" % value for value in row))


def initial_H(w, k):
    """Draw the initial H uniformly from [0, 2*sqrt(m/k)], where m is the mean
    of all n*n entries of W, the zero diagonal included."""
    mean = np.mean(w)
    high = 2.0 * np.sqrt(mean / k)
    return np.random.uniform(0, high, size=(len(w), k)).tolist()


def run_goal(goal, points, k):
    """Run the requested goal through the C extension and return its matrix.
    """
    # tolist() hands the extension real Python floats rather than numpy scalars.
    data = points.tolist()
    if goal != "symnmf":
        return getattr(symnmfmodule, goal)(data)
    w = symnmfmodule.norm(data)
    return symnmfmodule.symnmf(initial_H(w, k), w)


def main():
    """Validate the command line, run the requested goal and print the result.
    """
    if len(sys.argv) != 4:
        fail(ERROR_MESSAGE)
    k = parse_k(sys.argv[1])
    goal = sys.argv[2]
    if goal not in GOALS:
        fail(ERROR_MESSAGE)
    try:
        points = read_points(sys.argv[3])
    except (OSError, ValueError):
        fail(ERROR_MESSAGE)
    if k >= len(points):
        fail(K_ERROR_MESSAGE)
    try:
        matrix = run_goal(goal, points, k)
    except (TypeError, ValueError, MemoryError):
        fail(ERROR_MESSAGE)
    print_matrix(matrix)


if __name__ == "__main__":
    main()
