"""Role M4. The perturbation of the designated row (A5), and the hand numbers of H4."""
import numpy as np
import pytest
from scipy.optimize import linprog

from model import build_matrix_form, load_instance
from sens.perturb import interval_check, perturb_row

import triangle as T

TOL = 1e-6


@pytest.fixture(scope="module")
def sens():
    mf = build_matrix_form(load_instance())
    return mf, perturb_row(mf, mf.row_of("A5"))


def test_the_designated_row_is_a5_with_right_hand_side_13(sens):
    mf, out = sens
    assert mf.row_of("A5") == 70
    assert out["b"] == 13.0
    assert out["step"] == 0.01


def test_the_price_lies_between_the_two_slopes(sens):
    _, out = sens
    chk = interval_check(out, tol=TOL)
    assert chk["inside"]


def test_the_two_slopes_agree_so_the_point_is_not_degenerate(sens):
    _, out = sens
    chk = interval_check(out, tol=TOL)
    assert not chk["degenerate"]
    assert abs(out["left_slope"] - out["right_slope"]) < TOL
    assert abs(out["pi"] - 44.482543) < 1e-5


def test_one_more_center_costs_more_than_it_saves(sens):
    """A positive price on (A5): the LP would open fewer centers if it could."""
    _, out = sens
    assert out["pi"] > 0.0
    assert out["z_plus"] > out["z_base"] > out["z_minus"]


# --------------------------------------------------------------- hand task H4

def _triangle_z(rhs1):
    """LP value of the triangle problem when the right-hand side of (T1) is rhs1."""
    b = np.array([rhs1, 1.0, 1.0])
    res = linprog(c=T.C, A_ub=-T.A, b_ub=-b, bounds=list(zip(T.LO, T.UP)), method="highs")
    assert res.success
    return float(res.fun), np.asarray(res.x)


def test_h4a_the_perturbed_optimum_and_its_slope():
    """H4(a): with all three rows tight, y = ((1+e)/2, (1+e)/2, (1-e)/2) and z = 6 + e."""
    for eps in (-0.5, -0.1, 0.0, 0.1, 0.5):
        expected_y = np.array([(1 + eps) / 2, (1 + eps) / 2, (1 - eps) / 2])
        assert np.allclose(T.A @ expected_y, [1 + eps, 1.0, 1.0], atol=1e-12)
        assert abs(T.cost(expected_y) - (6.0 + eps)) < 1e-12
        z, _ = _triangle_z(1.0 + eps)
        assert abs(z - (6.0 + eps)) < 1e-6
    # the slope of z(1 + e) is 1, which is pi_1 of the task file
    z_plus, _ = _triangle_z(1.0 + 1e-4)
    z_minus, _ = _triangle_z(1.0 - 1e-4)
    slope = (z_plus - z_minus) / 2e-4
    assert abs(slope - 1.0) < 1e-6
    assert abs(slope - T.PI_H3A[0]) < 1e-6


def test_h4b_the_range_of_epsilon():
    """H4(b): the formula holds exactly while -1 <= e <= 1, the bounds 0 <= y <= 1 stop it."""
    for eps in (-1.0, 1.0):
        z, _ = _triangle_z(1.0 + eps)
        assert abs(z - (6.0 + eps)) < 1e-6
        y = np.array([(1 + eps) / 2, (1 + eps) / 2, (1 - eps) / 2])
        assert np.all(y >= -1e-12) and np.all(y <= 1.0 + 1e-12)
    # at e = 1 the point is (1,1,0) and y3 has reached its lower bound
    _, y = _triangle_z(2.0)
    assert abs(T.cost(y) - 7.0) < 1e-6
    # beyond e = 1 the row can no longer be met, the LP becomes infeasible
    res = linprog(c=T.C, A_ub=-T.A, b_ub=-np.array([2.5, 1.0, 1.0]),
                  bounds=list(zip(T.LO, T.UP)), method="highs")
    assert not res.success
    # below e = -1 the row goes slack and z stops falling, so the slope drops to 0
    z_below, _ = _triangle_z(0.0 - 0.5)
    assert abs(z_below - 5.0) < 1e-6
