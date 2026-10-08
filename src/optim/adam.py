"""Adam.  Owner: M4 (Adam).

m <- b1*m + (1-b1)*g;   s <- b2*s + (1-b2)*g**2   (elementwise)
mhat = m / (1 - b1**(k+1));   shat = s / (1 - b2**(k+1))
x <- x - alpha * mhat / (sqrt(shat) + eps)
m^0 = s^0 = 0;  k = 0, 1, 2, ... counts updates.
Defaults: b1 = 0.9, b2 = 0.999, eps = 1e-8; only alpha is tuned.
"""
import numpy as np
from .common import MAX_ITER_GRAD, BLOWUP, converged  # noqa: F401


def adam(prob, alpha, beta1=0.9, beta2=0.999, eps=1e-8, max_iter=MAX_ITER_GRAD):
    """Returns (x, hist, k)."""
    raise NotImplementedError("TODO M4")
