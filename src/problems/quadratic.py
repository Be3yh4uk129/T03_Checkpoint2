"""Q1: f(x) = x1^2 + c*x2^2, H = diag(2, 2c).  Q2: g(x) = f(R^T x), R = rotation by theta.

Owner: M2.  Start Q1: (1.3, 0.7); start Q2: R @ (1.3, 0.7).  Relative stopping rule.
c and theta_deg come from checkpoint2_params(SEED) (checkpoint2_generator_PUBLIC.py).

Derivatives (by hand).
  Q1:  f(x)  = x1^2 + c x2^2
       grad f = (2 x1, 2 c x2),           H = diag(2, 2c),  kappa = c
  Q2:  g(x)  = f(R^T x) with R orthogonal (R^T = R^-1)
       grad g = R grad f(R^T x),          Hg = R diag(2, 2c) R^T
       Same eigenvalues {2, 2c} as Q1, same minimum value 0 at x = 0, and the
       start R @ x0 is at the same distance from it, so any rotation-invariant
       method must produce exactly the same iteration count on Q1 and Q2.
"""
import numpy as np
from .base import Problem

X0_QUAD = np.array([1.3, 0.7])      # common start for the quadratic test problem


def rotation(theta_deg):
    """Counter-clockwise rotation matrix by theta_deg degrees."""
    t = np.deg2rad(theta_deg)
    return np.array([[np.cos(t), -np.sin(t)],
                     [np.sin(t), np.cos(t)]])


def make_q1(c) -> Problem:
    """f(x) = x1^2 + c x2^2, start (1.3, 0.7), relative rule."""
    c = float(c)

    def f(x):
        return x[0] ** 2 + c * x[1] ** 2

    def grad(x):
        return np.array([2.0 * x[0], 2.0 * c * x[1]])

    def hess(x):
        return np.diag([2.0, 2.0 * c])

    return Problem(name="Q1", f=f, grad=grad, hess=hess,
                   x0=X0_QUAD.copy(), rule="rel", tol=1e-6,
                   x_star=np.zeros(2))


def make_q2(c, theta_deg) -> Problem:
    """g(x) = f(R^T x): Q1 rotated by theta_deg, start R @ (1.3, 0.7)."""
    c = float(c)
    R = rotation(theta_deg)
    H = R @ np.diag([2.0, 2.0 * c]) @ R.T

    def f(x):
        y = R.T @ x
        return y[0] ** 2 + c * y[1] ** 2

    def grad(x):
        y = R.T @ x
        return R @ np.array([2.0 * y[0], 2.0 * c * y[1]])

    def hess(x):
        return H.copy()

    return Problem(name="Q2", f=f, grad=grad, hess=hess,
                   x0=R @ X0_QUAD, rule="rel", tol=1e-6,
                   x_star=np.zeros(2))


def alpha_star(c):
    """Theoretical optimal fixed GD step for Q1/Q2: 2/(lam_min+lam_max) = 1/(1+c)."""
    return 1.0 / (1.0 + float(c))


def heavy_ball_params(c):
    """Classical Polyak heavy-ball constants for lam_min = 2, lam_max = 2c.

    alpha = 4 / (sqrt(lam_max) + sqrt(lam_min))^2,  beta = ((sqrt(k)-1)/(sqrt(k)+1))^2.
    Quoted from Polyak (1964); used in S3.
    """
    lam_min, lam_max = 2.0, 2.0 * float(c)
    kappa = lam_max / lam_min
    alpha = 4.0 / (np.sqrt(lam_max) + np.sqrt(lam_min)) ** 2
    beta = ((np.sqrt(kappa) - 1.0) / (np.sqrt(kappa) + 1.0)) ** 2
    return alpha, beta
