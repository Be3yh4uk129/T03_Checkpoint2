"""Heavy-ball momentum.  Owner: M3 (Momentum).

v <- beta * v + grad f(x);   x <- x - alpha * v;   v^0 = 0
"""
import numpy as np
from .common import MAX_ITER_GRAD, BLOWUP, converged  # noqa: F401


def momentum(prob, alpha, beta, max_iter=MAX_ITER_GRAD):
    """Returns (x, hist, k)."""
    raise NotImplementedError("TODO M3")
