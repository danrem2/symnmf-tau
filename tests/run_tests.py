"""Build the project and test it against the numpy reference.

Run from anywhere:  python3 tests/run_tests.py
"""

import os
import re
import subprocess
import sys

import oracle

TESTS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TESTS)
DATA = os.path.join(TESTS, "data")

# (file, k) pairs with valid input.
VALID = [("tiny_n3_d2.txt", 2), ("d1_n6.txt", 2), ("medium_n50_d7.txt", 5)]
MALFORMED = ["empty.txt", "ragged.txt", "nonnumeric.txt", "trailing_comma.txt",
             "missing.txt"]
GENERIC = "An Error Has Occurred"
CLUSTERS = "Incorrect number of clusters!"

results = []


def check(name, ok):
    """Record and print one test result."""
    results.append(ok)
    print("  [%s] %s" % ("PASS" if ok else "FAIL", name))


def run(*args):
    """Run a command in the project root and return (stdout, exit code)."""
    proc = subprocess.run(args, cwd=ROOT, capture_output=True, text=True)
    return proc.stdout.strip(), proc.returncode


def c_cli(goal, path):
    """Run the C executable."""
    return run("./symnmf", goal, path)


def py_cli(k, goal, path):
    """Run the Python CLI."""
    return run(sys.executable, "symnmf.py", str(k), goal, path)


def build():
    """Build both the executable and the extension; exit if either fails."""
    print("Build")
    commands = {"make": ["make"],
                "setup.py": [sys.executable, "setup.py", "build_ext", "--inplace"]}
    for label, cmd in commands.items():
        ok = subprocess.run(cmd, cwd=ROOT, capture_output=True).returncode == 0
        check(label, ok)
        if not ok:
            sys.exit(1)


def test_outputs():
    """C, Python and the reference must agree on every goal."""
    print("Outputs match the reference")
    for name, k in VALID:
        path = os.path.join(DATA, name)
        for goal in ("sym", "ddg", "norm"):
            want = oracle.expected(path, goal, k)
            check("C      %-6s %s" % (goal, name), c_cli(goal, path) == (want, 0))
            check("Python %-6s %s" % (goal, name),
                  py_cli(k, goal, path) == (want, 0))
        want = oracle.expected(path, "symnmf", k)
        check("Python symnmf %s" % name, py_cli(k, "symnmf", path) == (want, 0))


def test_parsing():
    """Windows line endings and blank lines must not change the result."""
    print("Parsing")
    clean = os.path.join(DATA, "tiny_n3_d2.txt")
    messy = os.path.join(DATA, "crlf_blank_lines.txt")
    check("C      CRLF + blank lines", c_cli("norm", messy) == c_cli("norm", clean))
    check("Python CRLF + blank lines",
          py_cli(2, "symnmf", messy) == py_cli(2, "symnmf", clean))


def test_errors():
    """Every invalid input prints the right message and exits with code 1."""
    print("Error handling")
    tiny = os.path.join(DATA, "tiny_n3_d2.txt")
    for name in MALFORMED:
        path = os.path.join(DATA, name)
        check("C      %s" % name, c_cli("sym", path) == (GENERIC, 1))
        check("Python %s" % name, py_cli(2, "symnmf", path) == (GENERIC, 1))
    check("C      bad goal", c_cli("symnmf", tiny) == (GENERIC, 1))
    check("C      wrong argument count", run("./symnmf", "sym") == (GENERIC, 1))
    check("Python bad goal", py_cli(2, "bad", tiny) == (GENERIC, 1))
    for k in ("1", "3", "2.5", "abc", "inf", "nan"):
        check("Python k=%s" % k, py_cli(k, "symnmf", tiny) == (CLUSTERS, 1))


def test_analysis():
    """analysis.py prints three scores, each within [-1, 1]."""
    print("Analysis")
    out, code = run(sys.executable, "analysis.py", "5",
                    os.path.join(DATA, "medium_n50_d7.txt"))
    lines = out.split("\n")
    keys = ("nmf", "kmeans", "ari")
    shape = code == 0 and len(lines) == 3 and all(
        re.fullmatch(r"%s: -?\d\.\d{4}" % key, line)
        for key, line in zip(keys, lines))
    check("three lines: nmf, kmeans, ari", shape)
    check("scores within [-1, 1]",
          shape and all(-1 <= float(l.split()[1]) <= 1 for l in lines))


def main():
    """Run every test group and exit with 0 only if all passed."""
    build()
    test_outputs()
    test_parsing()
    test_errors()
    test_analysis()
    passed = sum(results)
    print("\n%d/%d passed" % (passed, len(results)))
    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()
