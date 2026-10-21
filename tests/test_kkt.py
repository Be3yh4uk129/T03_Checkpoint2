"""Role M3. The checker of conditions (a) to (e), and the hand numbers of H3."""
import numpy as np
import pytest

from model import build_matrix_form, load_instance
from lp.solve import solve_highs
from kkt.checker import check_conditions, reduced_costs
from kkt.convention import sign_convention_test

import triangle as T

TOL = 1e-6


@pytest.fixture(scope="module")
def solved():
    mf = build_matrix_form(load_instance())
    return mf, solve_highs(mf)


def test_sign_convention_of_our_solver():
    """The two-variable LP of src/kkt/convention.py, whose answer we know by hand."""
    out = sign_convention_test()
    assert out["x_matches"] and out["z_matches"]
    assert abs(out["pi"] + 1.0) < 1e-9
    assert out["pi_matches_handout"] and not out["needs_negation"]


def test_all_five_conditions_hold_at_the_lp_optimum(solved):
    mf, sol = solved
    out = check_conditions(mf.c, mf.A, mf.b, mf.sense, mf.lo, mf.up, sol.x, sol.pi, tol=TOL)
    for key in "abcde":
        assert out[key] <= TOL, f"condition ({key}) = {out[key]:.3e}"
    assert out["all_satisfied"]


def test_the_checker_rejects_a_perturbed_solution(solved):
    """A checker that always answers "fine" is worthless, so break the solution."""
    mf, sol = solved
    bad = sol.x.copy()
    bad[0] += 0.5
    out = check_conditions(mf.c, mf.A, mf.b, mf.sense, mf.lo, mf.up, bad, sol.pi, tol=TOL)
    assert out["a"] > TOL
    assert not out["all_satisfied"]


def test_the_checker_accepts_a_foreign_solver(solved):
    """The checker takes arrays only, so any solver's output goes in unchanged."""
    mf, sol = solved
    out = check_conditions(T.C, T.A, T.B, T.SENSE, T.LO, T.UP, T.HALF, T.PI_H3A, tol=TOL)
    assert out["all_satisfied"]


# --------------------------------------------------------------- hand task H3

def test_h3a_the_five_numbers_for_the_task_file_prices():
    """H3(a): x = (1/2,1/2,1/2) with pi = (1,3,2) satisfies all five conditions."""
    out = check_conditions(T.C, T.A, T.B, T.SENSE, T.LO, T.UP, T.HALF, T.PI_H3A, tol=TOL)
    assert abs(out["a"] - 0.0) < 1e-12
    assert abs(out["b"] - 0.0) < 1e-12
    assert abs(out["c"] - 0.0) < 1e-12
    assert abs(out["d"] - 0.0) < 1e-12
    assert abs(out["e"] - 0.0) < 1e-12
    assert np.allclose(reduced_costs(T.C, T.A, T.PI_H3A), [0.0, 0.0, 0.0], atol=1e-12)
    assert abs(out["primal_objective"] - 6.0) < 1e-12
    assert abs(out["dual_objective"] - 6.0) < 1e-12


def test_h3b_which_conditions_fail_for_the_second_price_vector():
    """H3(b): with pi' = (2,2,1) only (d) and (e) fail, by 2 and by 1."""
    out = check_conditions(T.C, T.A, T.B, T.SENSE, T.LO, T.UP, T.HALF, T.PI_H3B, tol=TOL)
    r = reduced_costs(T.C, T.A, T.PI_H3B)
    assert np.allclose(r, [0.0, 0.0, 2.0], atol=1e-12)
    assert abs(out["a"] - 0.0) < 1e-12          # (a) still holds
    assert abs(out["b"] - 0.0) < 1e-12          # (b) still holds, pi' >= 0
    assert abs(out["c"] - 0.0) < 1e-12          # (c) still holds, every row is tight
    assert abs(out["d"] - 2.0) < 1e-12          # (d) fails by 2, on y3
    assert abs(out["e"] - 1.0) < 1e-12          # (e) fails by 1
    assert not out["all_satisfied"]


def test_h3c_the_second_vector_is_still_a_lower_bound():
    """H3(c): pi' is dual feasible, so b^T pi' = 5 <= z_LP = 6, short by exactly 1."""
    r = reduced_costs(T.C, T.A, T.PI_H3B)
    assert np.all(T.PI_H3B >= -1e-12)            # dual feasible: prices nonnegative
    assert np.all(r >= -1e-12)                   # dual feasible: reduced costs nonnegative
    bound = float(T.B @ T.PI_H3B)
    z_lp, _, _ = T.solve_lp()
    assert abs(bound - 5.0) < 1e-12
    assert abs(z_lp - 6.0) < 1e-6
    assert bound <= z_lp + 1e-9
    assert abs((z_lp - bound) - 1.0) < 1e-6
    # the best prices of the task file do close the gap
    assert abs(float(T.B @ T.PI_H3A) - z_lp) < 1e-6
