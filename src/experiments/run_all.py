from __future__ import annotations

import csv
from pathlib import Path
import numpy as np

from optim.common import MAX_ITER_GRAD, MAX_ITER_NEWTON, BLOWUP, converged
from optim.gd import gd, gd_bt
from optim.momentum import momentum
from optim.newton import newton_pure, newton_damped
from optim.adam import adam
from problems.rosenbrock import make_r1, make_r2
from problems.quadratic import make_q1, make_q2, alpha_star
from problems.project import make_project

ALPHA_GRID = 10.0 ** np.linspace(-5, 0, 11)
BETA_GRID = [0.5, 0.8, 0.9, 0.95, 0.99]
ALPHA_MAX_EXT = 1e4
RESULTS = Path("results")


def _success(prob, x, k, cap):
    """A run is successful only if the stopping rule is met before the cap."""
    x = np.asarray(x, dtype=float)
    norm = np.linalg.norm(x)
    if not np.isfinite(norm) or norm > BLOWUP or k >= cap:
        return False
    g = np.asarray(prob.grad(x), dtype=float)
    g0 = np.linalg.norm(prob.grad(prob.x0))
    return bool(converged(np.linalg.norm(g), g0, prob.rule, prob.tol))


def _safe_run(solver, prob, *args, cap, **kwargs):
    try:
        x, hist, k = solver(prob, *args, max_iter=cap, **kwargs)
        ok = _success(prob, x, k, cap)
        return {"ok": ok, "iterations": int(k) if ok else None,
                "x": x, "hist": hist}
    except (ArithmeticError, FloatingPointError, ValueError, np.linalg.LinAlgError):
        return {"ok": False, "iterations": None, "x": None, "hist": None}


def _alpha_candidates(prob):
    values = list(ALPHA_GRID)
    if prob.name in ("Q1", "Q2"):
        # Q1/Q2 have Hessian eigenvalues 2 and 2c; c is recovered from the Hessian.
        c = float(np.asarray(prob.hess(prob.x0))[0, 0])
        # The rotated Hessian's [0,0] entry is not 2, so infer c from eigenvalues.
        eig = np.linalg.eigvalsh(prob.hess(prob.x0))
        c = float(max(eig) / min(eig))
        values.append(alpha_star(c))
    return sorted(set(float(a) for a in values))


def _tune(prob, method, detail_rows):
    """Tune GD, momentum, or Adam, extending alpha grid if best sits at top edge."""
    alphas = _alpha_candidates(prob)
    while True:
        runs = []
        if method == "GD":
            for a in alphas:
                result = _safe_run(gd, prob, a, cap=MAX_ITER_GRAD)
                detail_rows.append(_detail(prob, method, a, "", result))
                if result["ok"]:
                    runs.append((result["iterations"], a, None, result))
        elif method == "Momentum":
            for a in alphas:
                for b in BETA_GRID:
                    result = _safe_run(momentum, prob, a, b, cap=MAX_ITER_GRAD)
                    detail_rows.append(_detail(prob, method, a, b, result))
                    if result["ok"]:
                        runs.append((result["iterations"], a, b, result))
        elif method == "Adam":
            for a in alphas:
                result = _safe_run(adam, prob, a, cap=MAX_ITER_GRAD)
                detail_rows.append(_detail(prob, method, a, "", result))
                if result["ok"]:
                    runs.append((result["iterations"], a, None, result))
        if not runs:
            return {"solver": method, "iterations": "—", "alpha": "—", "beta": "—",
                    "converged": False, "extended_grid": len(alphas) > len(_alpha_candidates(prob))}
        best = min(runs, key=lambda item: item[0])
        # Extend only when the best is at the upper edge of the original/extended alpha grid.
        if np.isclose(best[1], max(alphas)) and max(alphas) < ALPHA_MAX_EXT:
            next_a = max(alphas) * (10.0 ** 0.5)
            if next_a <= ALPHA_MAX_EXT * (1 + 1e-12):
                alphas.append(float(next_a))
                alphas = sorted(set(alphas))
                continue
        _, a, b, _ = best
        return {"solver": method, "iterations": best[0], "alpha": a,
                "beta": "" if b is None else b, "converged": True,
                "extended_grid": max(alphas) > 1.0}


def _detail(prob, method, alpha, beta, result):
    return {"problem": prob.name, "solver": method, "alpha": alpha, "beta": beta,
            "iterations": result["iterations"] if result["ok"] else "—",
            "converged": result["ok"]}


def _fixed_result(prob, label, solver, *args, cap, **kwargs):
    result = _safe_run(solver, prob, *args, cap=cap, **kwargs)
    return {"problem": prob.name, "solver": label,
            "iterations": result["iterations"] if result["ok"] else "—",
            "alpha": "", "beta": "", "converged": result["ok"], "extended_grid": False}


def _problem_rows(prob, detail_rows):
    rows = []
    rows.append({"problem": prob.name, **_tune(prob, "GD", detail_rows)})
    rows.append(_fixed_result(prob, "GD-BT", gd_bt, cap=MAX_ITER_GRAD))
    rows.append(_fixed_result(prob, "Newton (pure)", newton_pure, cap=MAX_ITER_NEWTON))
    rows.append(_fixed_result(prob, "Newton (damped)", newton_damped, cap=MAX_ITER_NEWTON))
    rows.append({"problem": prob.name, **_tune(prob, "Momentum", detail_rows)})
    rows.append({"problem": prob.name, **_tune(prob, "Adam", detail_rows)})
    return rows


def _write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    # Team parameters from T03's repository task/test setup.
    c_team, theta_deg = 106, 49
    problems = [make_r1(), make_r2(), make_q1(c_team), make_q2(c_team, theta_deg)]
    detail_rows = []
    table1 = []
    for prob in problems:
        table1.extend(_problem_rows(prob, detail_rows))

    project = make_project()
    table2 = _problem_rows(project, detail_rows)

    _write_csv(RESULTS / "table1.csv", table1,
               ["problem", "solver", "iterations", "alpha", "beta", "converged", "extended_grid"])
    _write_csv(RESULTS / "table2.csv", table2,
               ["problem", "solver", "iterations", "alpha", "beta", "converged", "extended_grid"])
    _write_csv(RESULTS / "tuning_details.csv", detail_rows,
               ["problem", "solver", "alpha", "beta", "iterations", "converged"])
    print(f"Wrote {len(table1)} rows to {RESULTS / 'table1.csv'}")
    print(f"Wrote {len(table2)} rows to {RESULTS / 'table2.csv'}")
    print(f"Wrote {len(detail_rows)} tuning records to {RESULTS / 'tuning_details.csv'}")
    for row in table1 + table2:
        params = f"alpha={row.get('alpha', '')}"
        if row.get("beta", "") != "":
            params += f", beta={row['beta']}"
        print(f"{row['problem']:8s} | {row['solver']:14s} | k={str(row['iterations']):>6s} | {params}")


if __name__ == "__main__":
    main()
