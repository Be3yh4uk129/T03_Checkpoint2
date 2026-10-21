"""Group part (G4). The bound z_LP <= z*, the active-row table, and H5.

In a team of four, Task 5, section S5 and hand task H5 are joint work, so this
file belongs to all four members.
"""
import numpy as np
import pytest

from model import build_matrix_form, load_instance
from lp.solve import solve_highs
from kkt.checker import active_rows

import triangle as T

TOL = 1e-6

# H5(b): the three relative gaps, to six decimals
GAP_OPTIMUM = 1.0 / 7.0          # 0.142857, at the integer optimum (1,1,0), z = 7
GAP_101 = 2.0 / 8.0              # 0.250000, at (1,0,1), z = 8
GAP_111 = 6.0 / 12.0             # 0.500000, at (1,1,1), z = 12


@pytest.fixture(scope="module")
def solved():
    mf = build_matrix_form(load_instance())
    return mf, solve_highs(mf)


def test_active_rows_per_family(solved):
    """Task 5: the table of tight rows. Every equality row is tight by definition."""
    mf, sol = solved
    mask = active_rows(mf.A, mf.b, mf.sense, sol.x, tol=TOL)
    counts = {}
    for fam, (start, count) in mf.row_span.items():
        counts[fam] = int(mask[start:start + count].sum())
    assert counts["A1"] == 35            # all equality rows
    assert counts["A4"] == 35
    assert counts["A5"] == 1
    assert counts["A3"] == 23            # capacity binds on 23 of the 35 columns
    assert counts["A2"] == 770
    assert sum(counts.values()) == 864


def test_capacity_is_the_family_that_binds(solved):
    """Of the two inequality families, (A3) is the one that is nearly saturated."""
    mf, sol = solved
    mask = active_rows(mf.A, mf.b, mf.sense, sol.x, tol=TOL)
    s3, c3 = mf.row_span["A3"]
    share_a3 = mask[s3:s3 + c3].mean()
    assert share_a3 > 0.6


# --------------------------------------------------------------- hand task H5

def test_h5a_every_binary_feasible_point_is_lp_feasible():
    """H5(a): the integer feasible set sits inside the LP feasible set, so z_LP <= z*."""
    z_lp, _, _ = T.solve_lp()
    for point, ok, _ in T.enumerate_binary():
        if ok:
            y = np.array(point, dtype=float)
            assert np.all(y >= 0.0) and np.all(y <= 1.0)     # inside the LP box
            assert np.all(T.A @ y >= T.B - 1e-12)            # satisfies the LP rows
    _, z_star = T.integer_optimum()
    assert z_lp <= z_star + 1e-9


def test_h5b_the_three_relative_gaps():
    """H5(b): (z - z_LP)/z at the optimum, at (1,0,1) and at (1,1,1)."""
    z_lp, _, _ = T.solve_lp()
    assert abs(z_lp - 6.0) < 1e-6

    for point, expected_z, expected_gap in (
        ((1, 1, 0), 7.0, GAP_OPTIMUM),
        ((1, 0, 1), 8.0, GAP_101),
        ((1, 1, 1), 12.0, GAP_111),
    ):
        y = np.array(point, dtype=float)
        assert T.is_feasible(y)
        z = T.cost(y)
        assert abs(z - expected_z) < 1e-9
        assert abs((z - z_lp) / z - expected_gap) < 1e-6

    assert round(GAP_OPTIMUM, 3) == 0.143
    assert round(GAP_101, 3) == 0.25
    assert round(GAP_111, 3) == 0.5


def test_h5c_only_the_first_gap_measures_the_bound():
    """H5(c): the gap at the integer optimum is the one that cannot be improved."""
    z_lp, _, _ = T.solve_lp()
    _, z_star = T.integer_optimum()
    gaps = {z: (z - z_lp) / z for _, ok, z in T.enumerate_binary() if ok}
    # the smallest gap over all feasible points is reached at the integer optimum
    assert min(gaps.values()) == pytest.approx(GAP_OPTIMUM, abs=1e-9)
    assert min(gaps, key=gaps.get) == z_star
