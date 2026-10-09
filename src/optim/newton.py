"""Newton's method: pure and damped.  Owner: M2 (Newton).

Pure   : solve H p = -grad f with np.linalg.solve; x <- x + p
Damped : p as above; if <grad f, p> >= 0 use p = -grad f;
         then backtracking with d = p (alpha0 = 1, c = 1e-4)
Iteration cap: 100.
"""
import numpy as np
from .common import MAX_ITER_NEWTON, BLOWUP, ARMIJO_C, ALPHA0_BT, converged


def _failed(x):
    """Blow-up guard: ||x|| > 1e12 or the iterate is not finite."""
    n = np.linalg.norm(x)
    return (not np.isfinite(n)) or n > BLOWUP


def _newton_step(prob, x, g):
    """Solve H p = -g.  Falls back to the steepest-descent direction if H is
    singular, so that a singular Hessian does not crash the run."""
    H = np.asarray(prob.hess(x), dtype=float)
    try:
        p = np.linalg.solve(H, -g)
    except np.linalg.LinAlgError:
        return -g
    if not np.all(np.isfinite(p)):
        return -g
    return p


def newton_pure(prob, max_iter=MAX_ITER_NEWTON):
    """Pure Newton: full step, no safeguard. Returns (x, hist, k)."""
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
            p = _newton_step(prob, x, g)
            x = x + p
            g = prob.grad(x)
            hist.append(x.copy())
            k += 1
    return x, np.array(hist), k


def newton_damped(prob, max_iter=MAX_ITER_NEWTON, return_info=False):
    """Damped Newton: descent-direction safeguard + Armijo backtracking.

    If <grad f, p> >= 0 the Newton direction is not a descent direction and is
    replaced by -grad f; the step length then comes from backtracking with
    alpha0 = 1, c = 1e-4.  Returns (x, hist, k).
    """
    alphas, halvings, fallbacks = [], [], 0
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
            d = _newton_step(prob, x, g)
            if np.dot(g, d) >= 0.0:         # not a descent direction
                d = -g
                fallbacks += 1
            fx = prob.f(x)
            slope = np.dot(g, d)
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
                                      "halvings": np.array(halvings),
                                      "fallbacks": fallbacks}
    return x, np.array(hist), k
