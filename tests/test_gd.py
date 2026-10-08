import numpy as np
import pytest
from helpers import close_count
from optim.gd import gd, gd_bt
from problems.base import check_grad, check_hess
from problems.rosenbrock import make_r1, make_r2, rosenbrock_f, rosenbrock_grad, rosenbrock_hess


# ---- derivatives and the check values from the task file (T03) ----
def test_check_values_r1():
    p = make_r1()
    assert p.f(p.x0) == pytest.approx(24.2, abs=1e-6)
    assert np.linalg.norm(p.grad(p.x0)) == pytest.approx(232.86, abs=1e-2)


def test_check_values_r2():
    p = make_r2()
    assert p.f(p.x0) == pytest.approx(8.38506, abs=1e-5)
    assert np.linalg.norm(p.grad(p.x0)) == pytest.approx(87.6631, abs=1e-3)


@pytest.mark.parametrize("x", [[-1.2, 1.0], [-0.78, 0.38], [0.5, -0.3], [2.0, 3.0]])
def test_derivatives(x):
    x = np.array(x)
    assert check_grad(rosenbrock_f, rosenbrock_grad, x) < 1e-5
    assert check_hess(rosenbrock_grad, rosenbrock_hess, x) < 1e-4


# ---- reference iteration counts on R1 (same for all teams) ----
def test_gd_r1():
    _, _, k = gd(make_r1(), alpha=1e-3)
    assert close_count(k, 32076)


def test_gd_bt_r1():
    _, _, k = gd_bt(make_r1())
    assert close_count(k, 13756)


def test_gd_returns_all_iterates():
    x, hist, k = gd(make_r1(), alpha=1e-3)
    assert len(hist) == k + 1
    assert np.allclose(hist[0], [-1.2, 1.0]) and np.allclose(hist[-1], x)


# ---- H1 hand trace: f = 2x^2 + 8y^2, x0 = (4, 2) ----
def _h1_problem():
    from problems.base import Problem
    return Problem(
        name="H1",
        f=lambda x: 2 * x[0] ** 2 + 8 * x[1] ** 2,
        grad=lambda x: np.array([4 * x[0], 16 * x[1]]),
        hess=lambda x: np.diag([4.0, 16.0]),
        x0=np.array([4.0, 2.0]),
        rule="abs",
        tol=1e-6,
        x_star=np.array([0.0, 0.0]),
    )


def test_hand_trace_h1_two_gd_steps():
    p = _h1_problem()
    _, hist, k = gd(p, alpha=0.05, max_iter=2)
    assert k == 2
    assert np.allclose(hist[0], [4.0, 2.0], atol=1e-3)
    assert np.allclose(hist[1], [3.2, 0.4], atol=1e-3)
    assert np.allclose(hist[2], [2.56, 0.08], atol=1e-3)
    assert np.allclose([p.f(x) for x in hist], [64.0, 21.76, 13.1584], atol=1e-3)
    assert np.allclose(p.grad(hist[2]), [10.24, 1.28], atol=1e-3)


def test_hand_trace_h1_backtracking_step():
    p = _h1_problem()
    _, hist, k, info = gd_bt(p, max_iter=1, return_info=True)
    assert k == 1
    assert info["alphas"][0] == pytest.approx(0.125, abs=1e-3)
    assert info["halvings"][0] == 3
    assert np.allclose(hist[1], [2.0, -2.0], atol=1e-3)
    assert p.f(hist[1]) == pytest.approx(40.0, abs=1e-3)
