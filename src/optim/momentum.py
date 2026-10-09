"""Heavy-ball momentum.  Owner: M3 (Momentum).

v <- beta * v + grad f(x);   x <- x - alpha * v;   v^0 = 0

Note the sign convention of the course: v accumulates the gradient itself
(not the negative gradient), and the update subtracts alpha * v.
Momentum is not a descent method: f may increase along the way (see S3).
"""
import numpy as np
from .common import MAX_ITER_GRAD, BLOWUP, converged


def _failed(x):
    """Blow-up guard: ||x|| > 1e12 or the iterate is not finite."""
    n = np.linalg.norm(x)
    return (not np.isfinite(n)) or n > BLOWUP


def momentum(prob, alpha, beta, max_iter=MAX_ITER_GRAD, return_info=False):
    """Heavy-ball momentum. Returns (x, hist, k).

    With return_info=True also returns the f value at every iterate, which S3
    uses to show that f is not monotone.
    """
    fvals = []
    with np.errstate(all="ignore"):
        x = np.array(prob.x0, dtype=float)
        g = prob.grad(x)
        g0_norm = np.linalg.norm(g)
        v = np.zeros_like(x)
        hist = [x.copy()]
        if return_info:
            fvals.append(prob.f(x))
        k = 0
        while k < max_iter:
            if converged(np.linalg.norm(g), g0_norm, prob.rule, prob.tol):
                break
            if _failed(x):
                break
            v = beta * v + g
            x = x - alpha * v
            g = prob.grad(x)
            hist.append(x.copy())
            if return_info:
                fvals.append(prob.f(x))
            k += 1
    if return_info:
        fvals = np.array(fvals)
        increased = bool(np.any(np.diff(fvals) > 0)) if fvals.size > 1 else False
        return x, np.array(hist), k, {"f": fvals, "f_increased": increased}
    return x, np.array(hist), k
