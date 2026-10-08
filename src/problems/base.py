"""Problem container shared by all solvers (role: Infra) + check_grad (M1)."""
from dataclasses import dataclass
from typing import Callable, Optional
import numpy as np


@dataclass
class Problem:
    name: str
    f: Callable[[np.ndarray], float]
    grad: Callable[[np.ndarray], np.ndarray]
    hess: Optional[Callable[[np.ndarray], np.ndarray]]
    x0: np.ndarray
    rule: str                      # "abs" (Rosenbrock) or "rel" (Q1, Q2, project)
    tol: float = 1e-6
    x_star: Optional[np.ndarray] = None   # known minimizer, if any


def check_grad(fun, grad, x, h=1e-6):
    """Central-difference check of grad against fun at x.

        Return the max abs error; it should be below ~1e-6. Owner: M1.
        (A Hessian check can be done the same way by differencing grad.)
        """
    x = np.asarray(x, dtype=float)
    g = np.asarray(grad(x), dtype=float)
    err = 0.0
    for i in range(x.size):
        e = np.zeros_like(x)
        e[i] = h
        num = (fun(x + e) - fun(x - e)) / (2.0 * h)
        err = max(err, abs(num - g[i]))
    return err

def check_hess(grad, hess, x, h=1e-6):
    """Same idea for the Hessian: differences of the gradient, column by column."""
    x = np.asarray(x, dtype=float)
    H = np.asarray(hess(x), dtype=float)
    err = 0.0
    for j in range(x.size):
        e = np.zeros_like(x)
        e[j] = h
        col = (np.asarray(grad(x + e)) - np.asarray(grad(x - e))) / (2.0 * h)
        err = max(err, np.max(np.abs(col - H[:, j])))
    return err
