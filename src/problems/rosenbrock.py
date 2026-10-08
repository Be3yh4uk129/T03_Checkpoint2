"""Rosenbrock f(x) = (1-x1)^2 + 100 (x2 - x1^2)^2.  Owner: M1.

R1: start (-1.2, 1), absolute rule.  R2: same function, team start from the task file.
Derive gradient and Hessian by hand first (put the derivation in the report/hand/).
"""
import numpy as np
from .base import Problem


def make_r1() -> Problem:
    raise NotImplementedError("TODO M1")


def make_r2(x0) -> Problem:
    raise NotImplementedError("TODO M1")
