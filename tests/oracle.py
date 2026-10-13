"""Independent numpy reference implementation of SymNMF.

Written from the formulas alone, not from the C or Python code, so that it can
disagree with them when they are wrong.
"""

import numpy as np

EPS = 1e-4
MAX_ITER = 300
BETA = 0.5
SEED = 1234


def load_points(path):
    """Read datapoints, skipping blank lines; raise ValueError on bad input."""
    rows = []
    with open(path, "r") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = [float(token) for token in line.split(",")]
            if rows and len(row) != len(rows[0]):
                raise ValueError("ragged row")
            rows.append(row)
    if not rows:
        raise ValueError("no datapoints")
    return np.array(rows)


def sym(x):
    """Similarity matrix: exp(-squared distance / 2), with a zero diagonal."""
    sq_dist = ((x[:, None, :] - x[None, :, :]) ** 2).sum(-1)
    a = np.exp(-sq_dist / 2.0)
    np.fill_diagonal(a, 0.0)
    return a


def ddg(a):
    """Diagonal degree matrix: the row sums of A on the diagonal."""
    return np.diag(a.sum(axis=1))


def norm(a):
    """Normalized similarity D^(-1/2) A D^(-1/2), as a_ij / sqrt(d_i * d_j)."""
    d = a.sum(axis=1)
    return a / np.sqrt(np.outer(d, d))


def init_h(w, k):
    """Initial H, uniform on [0, 2*sqrt(m/k)] with m the mean of all of W."""
    np.random.seed(SEED)
    high = 2.0 * np.sqrt(w.mean() / k)
    return np.random.uniform(0, high, size=(w.shape[0], k))


def symnmf(w, h):
    """Multiplicative updates until the squared Frobenius change is < EPS."""
    for _ in range(MAX_ITER):
        h_next = h * (1 - BETA + BETA * (w @ h) / (h @ (h.T @ h)))
        converged = ((h_next - h) ** 2).sum() < EPS
        h = h_next
        if converged:
            break
    return h


def fmt(matrix):
    """Format a matrix as the programs print it: '%.4f', comma separated."""
    return "\n".join(",".join("%.4f" % v for v in row) for row in matrix)


def expected(path, goal, k):
    """Return the formatted output the given goal should produce."""
    a = sym(load_points(path))
    if goal == "sym":
        return fmt(a)
    if goal == "ddg":
        return fmt(ddg(a))
    w = norm(a)
    if goal == "norm":
        return fmt(w)
    return fmt(symnmf(w, init_h(w, k)))
