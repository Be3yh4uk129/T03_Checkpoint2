"""Task 3: the optimality conditions (a) to (e) of a linear program.

For a minimisation problem

    min c^T v   subject to   A v (sense) b,   lo <= v <= up

the conditions below are necessary and sufficient for v to be optimal. The
checker takes the primal vector and the dual prices as plain arrays, so it
accepts the output of any solver, including another team's, as long as the
dual prices use the convention pi_i = d z / d b_i.

The five numbers it returns are the five entries of Table 2.
"""
import numpy as np
from scipy import sparse

CONDITION_NAMES = {
    "a": "primal feasibility",
    "b": "dual feasibility (signs)",
    "c": "complementary slackness, rows",
    "d": "reduced costs",
    "e": "strong duality",
}


def reduced_costs(c, A, pi):
    """r_j = c_j - sum_i pi_i A_ij."""
    c = np.asarray(c, dtype=float)
    pi = np.asarray(pi, dtype=float)
    At_pi = A.T @ pi if sparse.issparse(A) else np.asarray(A).T @ pi
    return c - np.asarray(At_pi).reshape(-1)


def check_conditions(c, A, b, sense, lo, up, v, pi, tol=1e-6):
    """Return the five numbers (a) to (e) plus the details behind them.

    Every number is a largest violation, so a value below `tol` means the
    condition holds. (e) is the absolute difference of the two objectives.
    """
    c = np.asarray(c, dtype=float)
    b = np.asarray(b, dtype=float)
    v = np.asarray(v, dtype=float)
    pi = np.asarray(pi, dtype=float)
    lo = np.asarray(lo, dtype=float)
    up = np.asarray(up, dtype=float)
    sense = np.asarray(sense, dtype="<U1")

    Av = np.asarray(A @ v).reshape(-1)
    slack = b - Av                      # the handout's b_i - a_i^T x

    is_eq = sense == "E"
    is_le = sense == "L"
    is_ge = sense == "G"

    # (a) primal feasibility: rows and bounds
    row_viol = np.zeros_like(b)
    row_viol[is_eq] = np.abs(slack[is_eq])
    row_viol[is_le] = np.maximum(0.0, -slack[is_le])    # A v <= b  violated if A v > b
    row_viol[is_ge] = np.maximum(0.0, slack[is_ge])     # A v >= b  violated if A v < b
    bound_viol = np.maximum(np.maximum(0.0, lo - v), np.maximum(0.0, v - up))
    a_val = max(0.0, float(max(row_viol.max(initial=0.0), bound_viol.max(initial=0.0))))

    # (b) dual feasibility: pi <= 0 on "<=", pi >= 0 on ">=", free on "="
    dual_viol = np.zeros_like(pi)
    dual_viol[is_le] = np.maximum(0.0, pi[is_le])
    dual_viol[is_ge] = np.maximum(0.0, -pi[is_ge])
    b_val = max(0.0, float(dual_viol.max(initial=0.0)))

    # (c) complementary slackness on the rows
    comp = np.abs(pi * slack)
    c_val = float(comp.max(initial=0.0))

    # (d) reduced costs against the position of each variable in its box
    r = reduced_costs(c, A, pi)
    at_lo = v <= lo + tol
    at_up = v >= up - tol
    interior = ~at_lo & ~at_up
    rc_viol = np.zeros_like(r)
    rc_viol[at_lo & ~at_up] = np.maximum(0.0, -r[at_lo & ~at_up])   # v_j = lo  =>  r_j >= 0
    rc_viol[at_up & ~at_lo] = np.maximum(0.0, r[at_up & ~at_lo])    # v_j = up  =>  r_j <= 0
    rc_viol[interior] = np.abs(r[interior])                         # lo < v_j < up  =>  r_j = 0
    d_val = float(rc_viol.max(initial=0.0))

    # (e) strong duality: c^T v  against  b^T pi + sum over bounded j of up_j * min(0, r_j)
    primal_obj = float(c @ v)
    finite_up = np.isfinite(up)
    dual_obj = float(b @ pi + np.sum(up[finite_up] * np.minimum(0.0, r[finite_up])))
    e_val = float(abs(primal_obj - dual_obj))

    return {
        "a": a_val,
        "b": b_val,
        "c": c_val,
        "d": d_val,
        "e": e_val,
        "all_satisfied": bool(max(a_val, b_val, c_val, d_val, e_val) <= tol),
        "primal_objective": primal_obj,
        "dual_objective": dual_obj,
        "reduced_costs": r,
        "slack": slack,
        "row_violation": row_viol,
        "bound_violation": bound_viol,
        "tol": tol,
    }


def active_rows(A, b, sense, v, tol=1e-6):
    """Boolean mask of the rows that are tight at v (every equality row is tight)."""
    Av = np.asarray(A @ v).reshape(-1)
    return np.abs(b - Av) <= tol
