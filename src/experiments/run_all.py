"""Regenerate every number and figure:  python -m experiments.run_all  (from the repo root,
with PYTHONPATH=src).  Owner: M4.  Writes CSV files to results/ (commit them).

Tuning protocol (Section 3):
  * alpha grid 10^-5, 10^-4.5, ..., 10^0 (11 values); beta grid {0.5, 0.8, 0.9, 0.95, 0.99}
  * pick the combination with the fewest iterations among runs that reach the stopping rule;
    runs that hit the cap / blow-up guard are reported as "-"
  * if the best alpha is at the upper edge, extend by half-decades up to 10^4 and say so
  * on Q1, Q2 add alpha* = 1/(1+c) as an extra GD candidate
"""
import numpy as np

ALPHA_GRID = 10.0 ** np.linspace(-5, 0, 11)
BETA_GRID = [0.5, 0.8, 0.9, 0.95, 0.99]
ALPHA_MAX_EXT = 1e4


def main():
    raise NotImplementedError("TODO M4: tuning + Table 1 + Table 2 -> results/*.csv")


if __name__ == "__main__":
    main()
