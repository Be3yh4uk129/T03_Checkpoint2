"""How we established the sign convention of our solver (Task 3 asks us to say).

The handout warns that solvers disagree about the sign of a dual price, so we
fixed ours on a two-variable LP whose answer can be written down by hand:

    minimise   -v1 - 2 v2
    subject to  v1 + v2 <= 1.5                (one row, right-hand side b = 1.5)
                0 <= v1 <= 1,  0 <= v2 <= 1

v2 is the cheaper unit of objective, so v2 = 1 and the row leaves v1 = 0.5:
    v* = (0.5, 1),  z* = -0.5 - 2 = -2.5.
Raising b by a small delta lets v1 grow by delta, so z falls by delta and the
true dual price is

    pi = dz/db = -1.

A solver that returns -1 already uses the convention of the handout
(pi <= 0 on a "<=" row of a minimisation). A solver that returns +1 uses the
opposite one and its prices have to be negated before the checker sees them.
HiGHS through scipy returns -1, so we pass its prices through unchanged.
"""
import numpy as np
from scipy.optimize import linprog

EXPECTED_X = np.array([0.5, 1.0])
EXPECTED_Z = -2.5
EXPECTED_PI = -1.0


def sign_convention_test(tol=1e-9):
    """Solve the reference LP and report what the solver's price convention is."""
    res = linprog(
        c=np.array([-1.0, -2.0]),
        A_ub=np.array([[1.0, 1.0]]), b_ub=np.array([1.5]),
        bounds=[(0.0, 1.0), (0.0, 1.0)],
        method="highs",
    )
    if not res.success:
        raise RuntimeError(res.message)

    pi = float(res.ineqlin.marginals[0])
    return {
        "x": np.asarray(res.x, dtype=float),
        "z": float(res.fun),
        "pi": pi,
        "x_matches": bool(np.allclose(res.x, EXPECTED_X, atol=1e-9)),
        "z_matches": bool(abs(res.fun - EXPECTED_Z) <= 1e-9),
        "pi_matches_handout": bool(abs(pi - EXPECTED_PI) <= 1e-9),
        "needs_negation": bool(abs(pi + EXPECTED_PI) <= 1e-9),
        "tol": tol,
    }
