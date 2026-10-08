"""Gradient descent: fixed step and backtracking.  Owner: M1 (GD).

GD     : x <- x - alpha * grad f(x)
GD-BT  : d = -grad f; alpha = 1; while f(x + alpha d) > f(x) + 1e-4 * alpha * <grad f, d>:
             alpha /= 2
         then x <- x + alpha d
"""
import numpy as np
from .common import MAX_ITER_GRAD, BLOWUP, ARMIJO_C, ALPHA0_BT, converged


def _failed(x):
    """Blow-up guard: ||x|| > 1e12 or the iterate is not finite."""
    n = np.linalg.norm(x)
    return (not np.isfinite(n)) or n > BLOWUP


def gd(prob, alpha, max_iter=MAX_ITER_GRAD):
    """Fixed-step gradient descent. Returns (x, hist, k)."""
    with np.errstate(all="ignore"):
        x = np.array(prob.x0, dtype=float)
        g = prob.grad(x)
        g0_norm = np.linalg.norm(g)
        hist = [x.copy()]
        k = 0
        while k < max_iter:
            if converged(np.linalg.norm(g), g0_norm, prob.rule, prob.tol):
                break
            if _failed(x):
                break
            x = x - alpha * g
            g = prob.grad(x)
            hist.append(x.copy())
            k += 1
    return x, np.array(hist), k


def gd_bt(prob, max_iter=MAX_ITER_GRAD, return_info=False):
    """GD with backtracking (alpha0 = 1, c = 1e-4). Returns (x, hist, k).

    For analysis S1 it is useful to also record the accepted alphas and the
    number of halvings (e.g. as extra attributes / a separate helper).
    """
    alphas, halvings = [], []
    with np.errstate(all="ignore"):
        x = np.array(prob.x0, dtype=float)
        g = prob.grad(x)
        g0_norm = np.linalg.norm(g)
        hist = [x.copy()]
        k = 0
        while k < max_iter:
            if converged(np.linalg.norm(g), g0_norm, prob.rule, prob.tol):
                break
            if _failed(x):
                break
            d = -g
            fx = prob.f(x)
            slope = np.dot(g, d)            # <grad f, d> = -||g||^2
            alpha = ALPHA0_BT
            nh = 0
            while prob.f(x + alpha * d) > fx + ARMIJO_C * alpha * slope:
                alpha /= 2.0
                nh += 1
                if alpha < 1e-300:          # safeguard against an endless loop
                    break
            x = x + alpha * d
            g = prob.grad(x)
            hist.append(x.copy())
            alphas.append(alpha)
            halvings.append(nh)
            k += 1
    if return_info:
        return x, np.array(hist), k, {"alphas": np.array(alphas),
                                      "halvings": np.array(halvings)}
    return x, np.array(hist), k
