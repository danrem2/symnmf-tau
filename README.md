# SymNMF Clustering

Symmetric Non-negative Matrix Factorization (SymNMF) clustering, implemented in C with a
Python C-API extension, and compared against K-means.

Final project for the Software Project course at Tel Aviv University.

The numerical work is written in strict ANSI C (C89). Python provides the command-line
interface, the seeded random initialization and the evaluation metrics.

## How it works

Given `n` datapoints, SymNMF builds a similarity graph and factorizes it:

1. **Similarity matrix** `A`: `a_ij = exp(-‖xi − xj‖² / 2)` for `i ≠ j`, and `a_ii = 0`.
2. **Degree matrix** `D`: a diagonal matrix with `d_i = Σ_j a_ij`.
3. **Normalized similarity** `W = D^(-1/2) · A · D^(-1/2)`.
4. **Factorization**: find a non-negative `H` (n × k) that minimizes `‖W − HHᵀ‖²_F`, using
   the multiplicative update

   ```
   H_ij ← H_ij · (1 − β + β · (WH)_ij / (H HᵀH)_ij),   β = 0.5
   ```

   `H` starts from uniform random values in `[0, 2·√(m/k)]`, where `m` is the mean of `W`.
   The updates stop when `‖H(t+1) − H(t)‖²_F < 1e-4` or after 300 iterations.
5. **Clustering**: each point is assigned to the column of `H` with the highest value in
   its row.

## Project structure

| File | Purpose |
| ---- | ------- |
| `symnmf.c` | All the math (`sym`, `ddg`, `norm`, `symnmf`), the file reader and the C CLI |
| `symnmf.h` | Prototypes shared with the extension module |
| `symnmfmodule.c` | Python C-API wrapper that converts lists to C matrices and back |
| `symnmf.py` | Python CLI: argument validation, `H` initialization and output |
| `analysis.py` | Compares SymNMF with K-means |
| `kmeans.py` | K-means clustering, returns a label for each point |
| `setup.py` | Builds the `symnmfmodule` extension |
| `Makefile` | Builds the `symnmf` executable |

## Requirements

- gcc and make
- Python 3 with development headers (`Python.h`)
- numpy and scikit-learn

## Build

```bash
make                                    # C executable: ./symnmf
python3 setup.py build_ext --inplace    # Python extension: symnmfmodule
```

The C code compiles with `gcc -ansi -Wall -Wextra -Werror -pedantic-errors` and
produces no warnings.

## Usage

Input is a text file with one datapoint per line and comma-separated values:

```
0.8147,0.9058,0.1270
0.9134,0.6324,0.0975
...
```

```bash
# Print a matrix (C): goal is sym, ddg or norm
./symnmf norm data.txt

# Print a matrix, or run full SymNMF (Python): goal is symnmf, sym, ddg or norm
python3 symnmf.py 3 symnmf data.txt

# Compare SymNMF with K-means
python3 analysis.py 3 data.txt
```

Matrices are printed one row per line, with comma-separated values to 4 decimal places.
`analysis.py` prints the silhouette score of each method and the Adjusted Rand Index
between their clusterings. Example output for 10 points in 5 dimensions with `k = 5`:

```
nmf: 0.4681
kmeans: 0.2389
ari: 0.8148
```

Invalid input prints `Incorrect number of clusters!` (bad `k`) or `An Error Has Occurred`
(anything else) and exits with code 1.

## Testing

- An independent numpy implementation of every formula, compared against both the C and
  the Python outputs.
- Byte-identical output from the C executable and the Python interface.
- Edge cases: blank lines, Windows line endings, ragged rows, non-numeric values, empty
  files, and invalid `k` values (including `inf` and `nan`).
- Memory checked with valgrind on both successful and error paths: no leaks, no errors.
