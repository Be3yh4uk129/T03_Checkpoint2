"""Role M2. The LP relaxation of the instance, and the hand numbers of H2."""
import numpy as np
import pytest

from model import build_matrix_form, load_instance
from lp.solve import fractional_report, solve_highs, solve_pulp_cbc

import triangle as T

Z_LP = 994.048826          # Task 2, six decimals


@pytest.fixture(scope="module")
def solved():
    mf = build_matrix_form(load_instance())
    return mf, solve_highs(mf)


def test_z_lp_to_six_decimals(solved):
    _, sol = solved
    assert round(sol.z, 6) == Z_LP


def test_pulp_cbc_agrees_with_highs(solved):
    """G2 asks for a relative difference below 1e-6 between the two solvers."""
    mf, sol = solved
    z_cbc, _, version = solve_pulp_cbc(mf)
    assert int(version.split(".")[0]) < 4, "the course requires PuLP below version 4"
    assert abs(z_cbc - sol.z) / abs(sol.z) < 1e-6


def test_the_relaxation_is_a_lower_bound_on_every_feasible_integer_cost(solved):
    """A greedy feasible integer solution must cost at least z_LP."""
    mf, sol = solved
    d = load_instance()
    _, y = mf.split_solution(sol.x)
    centers = list(np.argsort(-y)[: mf.V])          # the V most open centers
    q, dist, D = d["demand"], d["dist"], d["D"]
    load = {k: 0.0 for k in centers}
    for k in centers:
        load[k] += float(q[k])
    total = 2.0 * float(np.sum(D[centers]))
    for i in sorted(range(mf.N), key=lambda i: -q[i]):
        if i in load:
            continue
        options = [k for k in centers if load[k] + q[i] <= mf.Cap]
        assert options, "the greedy construction ran out of capacity"
        k = min(options, key=lambda k: dist[i, k])
        load[k] += float(q[i])
        total += float(dist[i, k])
    assert total >= sol.z - 1e-6


def test_fractional_entries(solved):
    """Two fractional centers, and they sum to exactly one."""
    mf, sol = solved
    fr = fractional_report(mf, sol.x)
    assert fr["n_fractional_y"] == 2
    assert fr["n_fractional_x"] == 8
    assert fr["n_points_split"] == 4
    assert fr["n_open_centers_integral"] == 12
    assert abs(sum(v for _, v in fr["fractional_y"]) - 1.0) < 1e-9


# --------------------------------------------------------------- hand task H2

def test_h2a_the_half_point_is_feasible_and_costs_six():
    """H2(a): y = (1/2, 1/2, 1/2) satisfies every row with equality, cost 6."""
    assert T.is_feasible(T.HALF)
    assert np.allclose(T.A @ T.HALF, T.B)
    assert abs(T.cost(T.HALF) - 6.0) < 1e-6
    z_lp, _, _ = T.solve_lp()
    assert abs(z_lp - 6.0) < 1e-6          # the half point is LP optimal


def test_h2b_rounding_the_half_point():
    """H2(b): rounding down is infeasible, rounding up is feasible and costs 12."""
    down = np.zeros(3)
    up = np.ones(3)
    assert not T.is_feasible(down)
    assert T.is_feasible(up)
    assert abs(T.cost(up) - 12.0) < 1e-6
    # the four mixed roundings: only those with at least two ones survive
    feasible_mixed = {p for p, ok, _ in T.enumerate_binary() if ok and sum(p) == 2}
    assert feasible_mixed == {(1, 1, 0), (1, 0, 1), (0, 1, 1)}


def test_h2c_the_lp_gap_and_why_no_integral_point_is_lp_optimal():
    """H2(c): (z* - z_LP)/z* = 1/7, and the cheapest integral point already costs 7."""
    z_lp, _, _ = T.solve_lp()
    _, z_star = T.integer_optimum()
    gap = (z_star - z_lp) / z_star
    assert abs(z_lp - 6.0) < 1e-6
    assert abs(z_star - 7.0) < 1e-6
    assert abs(gap - 1.0 / 7.0) < 1e-6
    assert round(gap, 3) == 0.143
    cheapest_integral = min(z for _, ok, z in T.enumerate_binary() if ok)
    assert cheapest_integral > z_lp + 1e-9
