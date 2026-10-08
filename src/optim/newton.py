"""Newton's method: pure and damped.  Owner: M2 (Newton).

Pure   : solve H p = -grad f with np.linalg.solve; x <- x + p
Damped : p as above; if <grad f, p> >= 0 use p = -grad f;
         then backtracking with d = p (alpha0 = 1, c = 1e-4)
Iteration cap: 100.
"""
import numpy as np
from .common import MAX_ITER_NEWTON, BLOWUP, ARMIJO_C, ALPHA0_BT, converged  # noqa: F401


def newton_pure(prob, max_iter=MAX_ITER_NEWTON):
    """Returns (x, hist, k)."""
    raise NotImplementedError("TODO M2")


def newton_damped(prob, max_iter=MAX_ITER_NEWTON):
    """Returns (x, hist, k)."""
    raise NotImplementedError("TODO M2")
