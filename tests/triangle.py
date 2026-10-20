"""The triangle problem of Section 6, as data for the hand-task tests.

It is not part of the reference model: it lives here, next to the tests that
reproduce the hand numbers H1 to H5.

    minimise 3 y1 + 4 y2 + 5 y3
    (T1)  y1 + y2 >= 1
    (T2)  y2 + y3 >= 1
    (T3)  y1 + y3 >= 1
          y in {0,1},  LP relaxation 0 <= y <= 1

Costs and dual prices are the ones in the team T03 task file.
"""
import itertools

import numpy as np

C = np.array([3.0, 4.0, 5.0])                 # c1, c2, c3 of the task file
A = np.array([[1.0, 1.0, 0.0],                # (T1)  y1 + y2
              [0.0, 1.0, 1.0],                # (T2)  y2 + y3
              [1.0, 0.0, 1.0]])               # (T3)  y1 + y3
B = np.array([1.0, 1.0, 1.0])
SENSE = np.array(["G", "G", "G"], dtype="<U1")
LO = np.zeros(3)
UP = np.ones(3)

PI_H3A = np.array([1.0, 3.0, 2.0])            # the dual prices of the task file
PI_H3B = np.array([2.0, 2.0, 1.0])            # the vector pi' of the task file

HALF = np.array([0.5, 0.5, 0.5])


def is_feasible(y):
    """Does y satisfy all three covering rows?"""
    return bool(np.all(A @ np.asarray(y, dtype=float) >= B - 1e-12))


def cost(y):
    return float(C @ np.asarray(y, dtype=float))


def enumerate_binary():
    """All eight binary points with feasibility flag and cost (used by H1 and H2)."""
    out = []
    for y in itertools.product([0, 1], repeat=3):
        v = np.array(y, dtype=float)
        out.append((tuple(int(t) for t in y), is_feasible(v), cost(v)))
    return out


def integer_optimum():
    """(point, value) of the best feasible binary point."""
    feasible = [(p, z) for p, ok, z in enumerate_binary() if ok]
    best = min(feasible, key=lambda t: t[1])
    return best[0], best[1]


def solve_lp():
    """Optimal value of the LP relaxation, with HiGHS."""
    from scipy.optimize import linprog

    res = linprog(c=C, A_ub=-A, b_ub=-B, bounds=list(zip(LO, UP)), method="highs")
    if not res.success:
        raise RuntimeError(res.message)
    # the solver sees "<=" rows, so its prices are the negatives of the ">=" prices
    return float(res.fun), np.asarray(res.x), -np.asarray(res.ineqlin.marginals)


def as_le_form():
    """H1(b): the same rows written as "<=" rows, by multiplying by -1."""
    return -A.copy(), -B.copy(), np.array(["L", "L", "L"], dtype="<U1")
