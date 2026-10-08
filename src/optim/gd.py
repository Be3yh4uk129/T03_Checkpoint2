"""Gradient descent: fixed step and backtracking.  Owner: M1 (GD).

GD     : x <- x - alpha * grad f(x)
GD-BT  : d = -grad f; alpha = 1; while f(x + alpha d) > f(x) + 1e-4 * alpha * <grad f, d>:
             alpha /= 2
         then x <- x + alpha d
"""
import numpy as np
from .common import MAX_ITER_GRAD, BLOWUP, ARMIJO_C, ALPHA0_BT, converged  # noqa: F401


def gd(prob, alpha, max_iter=MAX_ITER_GRAD):
    """Fixed-step gradient descent. Returns (x, hist, k)."""
    raise NotImplementedError("TODO M1")


def gd_bt(prob, max_iter=MAX_ITER_GRAD):
    """Gradient descent with backtracking. Returns (x, hist, k).

    For analysis S1 it is useful to also record the accepted alphas and the
    number of halvings (e.g. as extra attributes / a separate helper).
    """
    raise NotImplementedError("TODO M1")
