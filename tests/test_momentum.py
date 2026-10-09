import numpy as np
import pytest
from helpers import close_count
from optim.momentum import momentum
from problems.quadratic import make_q1, make_q2
from problems.rosenbrock import make_r1

C_TEAM, THETA_DEG = 106, 49


def test_momentum_r1():
    _, _, k = momentum(make_r1(), alpha=1e-3, beta=0.9)
    assert close_count(k, 3020)


def test_momentum_beta_zero_is_gd():
    """beta = 0 reduces the update to plain gradient descent."""
    from optim.gd import gd
    _, h_mom, k_mom = momentum(make_r1(), alpha=1e-3, beta=0.0, max_iter=200)
    _, h_gd, k_gd = gd(make_r1(), alpha=1e-3, max_iter=200)
    assert k_mom == k_gd
    assert np.allclose(h_mom, h_gd, atol=1e-12)


def test_momentum_returns_all_iterates():
    x, hist, k = momentum(make_r1(), alpha=1e-3, beta=0.9)
    assert len(hist) == k + 1
    assert np.allclose(hist[0], [-1.2, 1.0]) and np.allclose(hist[-1], x)


def test_momentum_rotation_invariant():
    """Same (alpha, beta) must give the same count on Q1 and Q2 (consistency check)."""
    _, _, k1 = momentum(make_q1(C_TEAM), alpha=1e-3, beta=0.9)
    _, _, k2 = momentum(make_q2(C_TEAM, THETA_DEG), alpha=1e-3, beta=0.9)
    assert abs(k1 - k2) <= 1


def test_momentum_is_not_a_descent_method():
    """Evidence for S3: with a large beta, f increases at least once on R1."""
    _, _, _, info = momentum(make_r1(), alpha=1e-3, beta=0.95, return_info=True)
    assert info["f_increased"] is True


# ---- H3 hand trace: f = 2x^2 + 3y^2, x0 = (5, 1), alpha = 0.1, beta = 0.6 ----
def _h3_problem():
    from problems.base import Problem
    return Problem(
        name="H3",
        f=lambda z: 2 * z[0] ** 2 + 3 * z[1] ** 2,
        grad=lambda z: np.array([4.0 * z[0], 6.0 * z[1]]),
        hess=lambda z: np.diag([4.0, 6.0]),
        x0=np.array([5.0, 1.0]),
        rule="abs",
        tol=1e-6,
        x_star=np.zeros(2),
    )


def test_hand_trace_h3_two_momentum_steps():
    p = _h3_problem()
    _, hist, k = momentum(p, alpha=0.1, beta=0.6, max_iter=2)
    assert k == 2
    assert np.allclose(hist[0], [5.0, 1.0], atol=1e-3)
    assert np.allclose(hist[1], [3.0, 0.4], atol=1e-3)
    assert np.allclose(hist[2], [0.6, -0.2], atol=1e-3)
    assert np.allclose([p.f(z) for z in hist], [53.0, 18.48, 0.84], atol=1e-3)


def test_hand_trace_h3_against_plain_gd():
    """Same alpha, same start: GD reaches (1.8, 0.16); momentum overshoots y past 0."""
    from optim.gd import gd
    p = _h3_problem()
    _, hg, _ = gd(p, alpha=0.1, max_iter=2)
    _, hm, _ = momentum(p, alpha=0.1, beta=0.6, max_iter=2)
    assert np.allclose(hg[2], [1.8, 0.16], atol=1e-3)
    assert p.f(hg[2]) == pytest.approx(6.5568, abs=1e-3)
    assert abs(hm[2][0]) < abs(hg[2][0])      # momentum advances x faster
    assert hm[2][1] < 0 < hg[2][1]            # ... and overshoots y past zero


def test_hand_trace_h3_f_monotone_here():
    """With beta = 0.6 on this problem f happens to decrease every step."""
    p = _h3_problem()
    _, _, _, info = momentum(p, alpha=0.1, beta=0.6, max_iter=2, return_info=True)
    assert info["f_increased"] is False
