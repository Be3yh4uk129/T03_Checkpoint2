"""Q1: f(x) = x1^2 + c*x2^2, H = diag(2, 2c).  Q2: g(x) = f(R^T x), R = rotation by theta.

Owner: M2.  Start Q1: (1.3, 0.7); start Q2: R @ (1.3, 0.7).  Relative stopping rule.
c and theta_deg come from checkpoint2_params(SEED) (checkpoint2_generator_PUBLIC.py).
"""
import numpy as np
from .base import Problem


def make_q1(c) -> Problem:
    raise NotImplementedError("TODO M2")


def make_q2(c, theta_deg) -> Problem:
    raise NotImplementedError("TODO M2")
