"""Role M1. The seven check values of the task file, and the hand numbers of H1."""
import numpy as np
import pytest

from model import build_matrix_form, instance_summary, load_instance
from model.instance import CHECK_VALUES

import triangle as T


# ----------------------------------------------------------------- the instance

@pytest.fixture(scope="module")
def mf():
    return build_matrix_form(load_instance())


def test_check_values_of_the_task_file():
    got = instance_summary()
    assert got["N"] == CHECK_VALUES["N"]
    assert got["V"] == CHECK_VALUES["V"]
    assert got["Cap"] == CHECK_VALUES["Cap"]
    assert got["sum_demand"] == CHECK_VALUES["sum_demand"]
    assert round(got["sum_dist"], 3) == CHECK_VALUES["sum_dist"]
    assert got["n_variables"] == CHECK_VALUES["n_variables"]
    assert got["n_rows"] == CHECK_VALUES["n_rows"]


def test_matrix_sizes_match_the_check_values(mf):
    assert mf.n_variables == 1260
    assert mf.n_rows == 1331
    assert mf.A.shape == (1331, 1260)
    assert int(np.sum(mf.sense == "E")) == 71      # (A1) 35 + (A4) 35 + (A5) 1
    assert int(np.sum(mf.sense == "L")) == 1260    # (A2) 1225 + (A3) 35


def test_row_families_partition_the_rows(mf):
    total = sum(count for _, count in mf.row_span.values())
    assert total == mf.n_rows
    assert mf.row_span["A1"][1] == 35
    assert mf.row_span["A2"][1] == 1225
    assert mf.row_span["A3"][1] == 35
    assert mf.row_span["A4"][1] == 35
    assert mf.row_span["A5"][1] == 1
    assert mf.row_of("A5") == 70
    assert mf.b[mf.row_of("A5")] == 13.0


def test_objective_coefficients(mf):
    d = load_instance()
    N = mf.N
    assert np.allclose(mf.c[: N * N].reshape(N, N), d["dist"])
    assert np.allclose(mf.c[N * N:], 2.0 * d["D"])


def test_a_small_hand_built_corner_of_the_matrix(mf):
    """Task 1 asks us to verify the assembly on the first three points by hand."""
    N = mf.N
    A = mf.A.toarray()
    # (A1) for point 0: a single 1 in every x_0k column
    row = A[mf.row_of("A1", 0)]
    assert row[0:N].tolist() == [1.0] * N
    assert row[N * N:].tolist() == [0.0] * N
    # (A2) for the pair (2, 1): x_21 - y_1 <= 0
    row = A[mf.row_of("A2", 2 * N + 1)]
    assert row[2 * N + 1] == 1.0 and row[N * N + 1] == -1.0 and row.sum() == 0.0
    # (A4) for k = 1: x_11 - y_1 = 0
    row = A[mf.row_of("A4", 1)]
    assert row[1 * N + 1] == 1.0 and row[N * N + 1] == -1.0
    # (A3) for k = 0: the demands in column 0 against -Cap on y_0
    row = A[mf.row_of("A3", 0)]
    assert np.allclose(row[[i * N for i in range(N)]], load_instance()["demand"])
    assert row[N * N] == -float(mf.Cap)


# --------------------------------------------------------------- hand task H1

def test_h1a_matrix_form_of_the_triangle_problem():
    """H1(a): c, A, b and the two counts."""
    assert T.C.tolist() == [3.0, 4.0, 5.0]
    assert T.A.tolist() == [[1, 1, 0], [0, 1, 1], [1, 0, 1]]
    assert T.B.tolist() == [1.0, 1.0, 1.0]
    assert T.C.size == 3            # three variables
    assert T.A.shape[0] == 3        # three rows, the bounds are not rows


def test_h1b_rows_as_le_rows_flip_the_sign_of_every_price():
    """H1(b): multiplying a row by -1 negates its dual price, the value is unchanged."""
    from scipy.optimize import linprog

    A_le, b_le, sense_le = T.as_le_form()
    assert np.allclose(A_le, -T.A)
    assert np.allclose(b_le, -T.B)
    assert list(sense_le) == ["L", "L", "L"]

    z_ge, _, pi_ge = T.solve_lp()                       # prices of the ">=" rows
    res = linprog(c=T.C, A_ub=A_le, b_ub=b_le, bounds=list(zip(T.LO, T.UP)), method="highs")
    pi_le = np.asarray(res.ineqlin.marginals)           # prices of the "<=" rows
    assert abs(float(res.fun) - z_ge) < 1e-9
    assert np.allclose(pi_le, -pi_ge, atol=1e-9)
    assert np.all(pi_ge >= -1e-9) and np.all(pi_le <= 1e-9)


def test_h1c_all_eight_binary_points_and_the_integer_optimum():
    """H1(c): four of the eight points are feasible, the best costs 7."""
    table = T.enumerate_binary()
    assert len(table) == 8
    feasible = {p: z for p, ok, z in table if ok}
    assert feasible == {(0, 1, 1): 9.0, (1, 0, 1): 8.0, (1, 1, 0): 7.0, (1, 1, 1): 12.0}
    infeasible = {p for p, ok, _ in table if not ok}
    assert infeasible == {(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)}
    point, z_star = T.integer_optimum()
    assert point == (1, 1, 0)
    assert abs(z_star - 7.0) < 1e-6
