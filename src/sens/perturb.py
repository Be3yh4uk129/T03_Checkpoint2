"""Task 4: sensitivity of the designated constraint.

The team task file names row (A5), sum_k y_k = V, with right-hand side b = 13.
We solve the LP three times, with b - 0.01, b and b + 0.01, and compare the two
one-sided slopes of z(b) with the dual price the solver reports.

z(b) is convex and piecewise linear, so a valid dual price always lies between
the two slopes:

    left slope  <=  pi  <=  right slope.

If the two slopes differ, b sits at a kink of z(b): the LP is degenerate there
and the dual price is not unique.
"""
import numpy as np

from lp.solve import solve_highs

DEFAULT_STEP = 0.01


def perturb_row(mf, row, step=DEFAULT_STEP):
    """Solve the LP at b - step, b and b + step on the given global row.

    Returns a dictionary with the three optimal values, the two one-sided
    slopes and the dual price of the unperturbed solve.
    """
    b0 = float(mf.b[row])

    def solve_at(value):
        b = mf.b.copy()
        b[row] = value
        return solve_highs(mf, b=b)

    minus = solve_at(b0 - step)
    base = solve_at(b0)
    plus = solve_at(b0 + step)

    left = (base.z - minus.z) / step
    right = (plus.z - base.z) / step

    return {
        "row": int(row),
        "b": b0,
        "step": float(step),
        "z_minus": minus.z,
        "z_base": base.z,
        "z_plus": plus.z,
        "left_slope": float(left),
        "right_slope": float(right),
        "pi": float(base.pi[row]),
        "seconds": minus.seconds + base.seconds + plus.seconds,
        "solution": base,
    }


def interval_check(result, tol=1e-6):
    """Does left slope <= pi <= right slope hold, and is the point degenerate?"""
    left, right, pi = result["left_slope"], result["right_slope"], result["pi"]
    return {
        "inside": bool(left - tol <= pi <= right + tol),
        "degenerate": bool(abs(right - left) > tol),
        "slope_difference": float(right - left),
        "left_margin": float(pi - left),
        "right_margin": float(right - pi),
        "tol": tol,
    }
