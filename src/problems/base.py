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
    raise NotImplementedError("TODO M1")
