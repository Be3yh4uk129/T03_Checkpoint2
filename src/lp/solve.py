"""Task 2: solve the LP relaxation and read the solution.

Primary solver: scipy.optimize.linprog(method="highs").
Cross-check: PuLP (version below 4) with the bundled CBC.

Both solvers get exactly the matrix form of src/model, so a disagreement in
z_LP would mean a bug in the assembly and not in the model.

Dual prices are returned in the sign convention of the handout,
pi_i = d z_LP / d b_i, so pi <= 0 on "<=" rows and pi is free on "=" rows.
The convention of the solver is verified in src/kkt/convention.py on a
two-variable LP whose answer is known by hand.
"""
import time
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog


@dataclass
class LPSolution:
    """A primal-dual pair together with the facts the report has to quote."""

    x: np.ndarray            # primal solution, length n
    pi: np.ndarray           # dual price per row, length m, pi_i = dz/db_i
    z: float                 # optimal value
    solver: str
    seconds: float
    status: str
    n_iterations: int = -1


def solve_highs(mf, b=None) -> LPSolution:
    """Solve the LP relaxation with HiGHS through scipy.

    `b` replaces the right-hand side vector (used by the perturbation runs of
    Task 4); the default is the right-hand side of the model itself.
    """
    rhs = mf.b if b is None else np.asarray(b, dtype=float)
    eq = mf.sense == "E"
    ub = mf.sense == "L"

    t0 = time.perf_counter()
    res = linprog(
        c=mf.c,
        A_ub=mf.A[ub], b_ub=rhs[ub],
        A_eq=mf.A[eq], b_eq=rhs[eq],
        bounds=list(zip(mf.lo, mf.up)),
        method="highs",
    )
    seconds = time.perf_counter() - t0

    if not res.success:
        raise RuntimeError(f"HiGHS did not solve the LP: {res.message}")

    pi = np.zeros(mf.n_rows, dtype=float)
    pi[eq] = res.eqlin.marginals
    pi[ub] = res.ineqlin.marginals

    return LPSolution(
        x=np.asarray(res.x, dtype=float),
        pi=pi,
        z=float(res.fun),
        solver="HiGHS (scipy.optimize.linprog)",
        seconds=seconds,
        status=res.message.strip(),
        n_iterations=int(getattr(res, "nit", -1) or -1),
    )


def solve_pulp_cbc(mf, b=None):
    """Cross-check of z_LP with PuLP below version 4 and CBC.

    Returns (z, seconds, version). PuLP is used only to confirm the value, the
    dual prices of the report come from HiGHS.
    """
    import pulp

    if int(pulp.__version__.split(".")[0]) >= 4:
        raise RuntimeError(
            f"PuLP {pulp.__version__} found; the course requires a version below 4 "
            '(pip install "pulp<4")'
        )

    rhs = mf.b if b is None else np.asarray(b, dtype=float)
    A = mf.A.tocsr()
    n = mf.n_variables

    prob = pulp.LpProblem("cp3_lp_relaxation", pulp.LpMinimize)
    v = pulp.LpVariable.dicts("v", range(n), lowBound=0.0, upBound=1.0, cat="Continuous")

    prob += pulp.lpSum(float(mf.c[j]) * v[j] for j in range(n) if mf.c[j] != 0.0)

    for i in range(mf.n_rows):
        lo, hi = A.indptr[i], A.indptr[i + 1]
        expr = pulp.lpSum(float(A.data[p]) * v[int(A.indices[p])] for p in range(lo, hi))
        if mf.sense[i] == "E":
            prob += expr == float(rhs[i])
        elif mf.sense[i] == "L":
            prob += expr <= float(rhs[i])
        else:
            prob += expr >= float(rhs[i])

    t0 = time.perf_counter()
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    seconds = time.perf_counter() - t0

    status = pulp.LpStatus[prob.status]
    if status != "Optimal":
        raise RuntimeError(f"CBC returned status {status}")
    return float(pulp.value(prob.objective)), seconds, pulp.__version__


def fractional_report(mf, x, tol=1e-6):
    """Describe the integer variables that came out strictly between 0 and 1."""
    X, y = mf.split_solution(x)

    def frac_idx(a):
        return np.where((a > tol) & (a < 1.0 - tol))[0]

    y_idx = frac_idx(y)
    x_flat = X.reshape(-1)
    x_idx = frac_idx(x_flat)

    served_from = {}
    for j in x_idx:
        i, k = divmod(int(j), mf.N)
        served_from.setdefault(i, []).append((k, float(x_flat[j])))

    return {
        "n_fractional_total": int(y_idx.size + x_idx.size),
        "n_fractional_y": int(y_idx.size),
        "n_fractional_x": int(x_idx.size),
        "fractional_y": [(int(k), float(y[k])) for k in y_idx],
        "n_points_split": len(served_from),
        "split_points": {int(i): val for i, val in sorted(served_from.items())},
        "n_open_centers_integral": int(np.sum(y > 1.0 - tol)),
    }
