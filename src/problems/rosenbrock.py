"""Rosenbrock f(x) = (1-x1)^2 + 100 (x2 - x1^2)^2.  Owner: M1.

R1: start (-1.2, 1), absolute rule.  R2: same function, team start from the task file.
Derive gradient and Hessian by hand first (put the derivation in the report/hand/).
"""
import numpy as np
from .base import Problem

def make_r1() -> Problem:
    return _make("R1", [-1.2, 1.0])


def make_r2(x0=(-0.78, 0.38)) -> Problem:
    """x0 = the team start from the task file (T03: (-0.78, 0.38))."""
    return _make("R2", x0)

def rosenbrock_f(x):
    return (1.0 - x[0]) ** 2 + 100.0 * (x[1] - x[0] ** 2) ** 2


def rosenbrock_grad(x):
    r = x[1] - x[0] ** 2
    return np.array([-2.0 * (1.0 - x[0]) - 400.0 * x[0] * r,
                     200.0 * r])


def rosenbrock_hess(x):
    return np.array([[2.0 - 400.0 * x[1] + 1200.0 * x[0] ** 2, -400.0 * x[0]],
                     [-400.0 * x[0], 200.0]])


def _make(name, x0):
    return Problem(name=name, f=rosenbrock_f, grad=rosenbrock_grad,
                   hess=rosenbrock_hess, x0=np.array(x0, dtype=float),
                   rule="abs", tol=1e-6, x_star=np.array([1.0, 1.0]))

