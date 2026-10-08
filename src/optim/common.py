"""Shared constants and helpers for all solvers (role: Infra).

Spec: Checkpoint 2, Section 3 "Solvers and conventions".
"""
import numpy as np

MAX_ITER_GRAD = 100_000   # GD, GD-BT, momentum, Adam
MAX_ITER_NEWTON = 100     # Newton (pure and damped)
BLOWUP = 1e12             # stop as failed if ||x|| > BLOWUP
ARMIJO_C = 1e-4           # backtracking constant c
ALPHA0_BT = 1.0           # backtracking starting step


def converged(grad_norm, grad0_norm, rule, tol=1e-6):
    """Stopping rule. 'abs': ||g|| < tol.  'rel': ||g|| < tol * ||g0||."""
    if rule == "abs":
        return grad_norm < tol
    if rule == "rel":
        return grad_norm < tol * grad0_norm
    raise ValueError(f"unknown rule {rule!r}")


def reached_tol(prob, x):
    """True if the stopping rule holds at x (used to tell success from cap/blow-up)."""
    g0 = np.linalg.norm(prob.grad(prob.x0))
    return bool(converged(np.linalg.norm(prob.grad(x)), g0, prob.rule, prob.tol))
