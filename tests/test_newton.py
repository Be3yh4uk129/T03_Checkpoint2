import numpy as np
import pytest
from helpers import close_count
from optim.newton import newton_pure, newton_damped
from problems.base import check_grad, check_hess
from problems.quadratic import make_q1, make_q2, rotation
from problems.rosenbrock import make_r1

C_TEAM, THETA_DEG = 106, 49


# ---- reference iteration counts on R1 (same for all teams) ----
def test_newton_pure_r1():
    _, _, k = newton_pure(make_r1())
    assert close_count(k, 6)


def test_newton_damped_r1():
    _, _, k = newton_damped(make_r1())
    assert close_count(k, 21)


def test_newton_returns_all_iterates():
    x, hist, k = newton_pure(make_r1())
    assert len(hist) == k + 1
    assert np.allclose(hist[0], [-1.2, 1.0]) and np.allclose(hist[-1], x)


# ---- Q1/Q2 derivatives and structure (M2 owns problems/quadratic.py) ----
@pytest.mark.parametrize("x", [[1.3, 0.7], [-0.4, 2.0], [0.0, 1.0]])
def test_q1_derivatives(x):
    p = make_q1(C_TEAM)
    assert check_grad(p.f, p.grad, np.array(x)) < 1e-5
    assert check_hess(p.grad, p.hess, np.array(x)) < 1e-4


@pytest.mark.parametrize("x", [[1.3, 0.7], [-0.4, 2.0], [0.0, 1.0]])
def test_q2_derivatives(x):
    p = make_q2(C_TEAM, THETA_DEG)
    assert check_grad(p.f, p.grad, np.array(x)) < 1e-5
    assert check_hess(p.grad, p.hess, np.array(x)) < 1e-4


def test_q2_is_q1_rotated():
    """Same eigenvalues, same minimum value, same distance from the start."""
    q1, q2 = make_q1(C_TEAM), make_q2(C_TEAM, THETA_DEG)
    R = rotation(THETA_DEG)
    assert np.allclose(np.linalg.eigvalsh(q1.hess(q1.x0)),
                       np.linalg.eigvalsh(q2.hess(q2.x0)))
    assert np.allclose(q2.x0, R @ q1.x0)
    assert q1.f(q1.x0) == pytest.approx(q2.f(q2.x0), rel=1e-12)
    assert np.linalg.norm(q1.x0) == pytest.approx(np.linalg.norm(q2.x0), rel=1e-12)


def test_newton_one_step_on_quadratic():
    """Newton solves a quadratic in a single step, on Q1 and on Q2 alike."""
    for p in (make_q1(C_TEAM), make_q2(C_TEAM, THETA_DEG)):
        x, _, k = newton_pure(p)
        assert k == 1
        assert np.allclose(x, [0.0, 0.0], atol=1e-10)


# ---- H2 hand trace: f = 2x^2 + e^y - 5y, x0 = (3, 0) ----
def _h2_problem():
    from problems.base import Problem
    return Problem(
        name="H2",
        f=lambda z: 2 * z[0] ** 2 + np.exp(z[1]) - 5.0 * z[1],
        grad=lambda z: np.array([4.0 * z[0], np.exp(z[1]) - 5.0]),
        hess=lambda z: np.diag([4.0, np.exp(z[1])]),
        x0=np.array([3.0, 0.0]),
        rule="abs",
        tol=1e-6,
        x_star=np.array([0.0, np.log(5.0)]),
    )


def test_hand_trace_h2_minimizer():
    p = _h2_problem()
    assert np.allclose(p.x_star, [0.0, 1.6094], atol=1e-3)
    assert p.f(p.x_star) == pytest.approx(-3.0472, abs=1e-3)


def test_hand_trace_h2_two_newton_steps():
    p = _h2_problem()
    _, hist, k = newton_pure(p, max_iter=2)
    assert k == 2
    assert np.allclose(hist[0], [3.0, 0.0], atol=1e-3)
    assert np.allclose(hist[1], [0.0, 4.0], atol=1e-3)
    assert np.allclose(hist[2], [0.0, 3.0916], atol=1e-3)
    assert np.allclose([p.f(z) for z in hist], [19.0, 34.5982, 6.5539], atol=1e-3)


def test_hand_trace_h2_f_is_not_monotone():
    """f goes 19 -> 34.6 -> 6.55: the pure Newton step is not a descent step at k=0."""
    p = _h2_problem()
    _, hist, _ = newton_pure(p, max_iter=2)
    fv = np.array([p.f(z) for z in hist])
    assert fv[1] > fv[0]


def test_hand_trace_h2_armijo_rejects_full_step():
    """Damped Newton would NOT accept alpha = 1 at k = 0."""
    p = _h2_problem()
    x0 = p.x0
    g0 = p.grad(x0)
    p0 = np.linalg.solve(p.hess(x0), -g0)
    assert np.allclose(p0, [-3.0, 4.0], atol=1e-3)
    assert np.dot(g0, p0) == pytest.approx(-52.0, abs=1e-3)
    assert p.f(x0 + p0) > p.f(x0) + 1e-4 * np.dot(g0, p0)


def test_hand_trace_h2_quadratic_ratio():
    """|y2-y*| / |y1-y*|^2 at k = 1."""
    p = _h2_problem()
    _, hist, _ = newton_pure(p, max_iter=2)
    ys = np.log(5.0)
    ratio = abs(hist[2][1] - ys) / abs(hist[1][1] - ys) ** 2
    assert ratio == pytest.approx(0.2594, abs=1e-3)
