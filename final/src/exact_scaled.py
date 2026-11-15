"""Exact method of Block 3 (and the exact row of Table 2): the Checkpoint 3 model solved by HiGHS, 120 s limit.

    python final/src/exact_scaled.py                 # own instance (N = 35) and N = 60, 100, 150
    python final/src/exact_scaled.py --sizes 60      # one size only

Writes results/final_exact.csv (one row per instance) and results/final_exact_<N>.json (the incumbent,
so that the run can be checked with evaluate_A). The status word of the solver is not used: the
optimum counts as proven only when the solver gap (z - best bound) / |z| is at most 1e-6.
"""
import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (RESULTS, ROOT, gen, own_instance, scaled, scipy_input,   # noqa: E402
                    solution_from_vector)

sys.path.insert(0, os.path.join(ROOT, "experiments"))
from exact_helper import _pack                                                # noqa: E402

COLS = ["instance", "N", "V", "Cap", "n_variables", "n_rows", "objective", "bound", "solver_gap",
        "proven", "seconds", "checker_cost", "feasible"]


def solve_one(d, limit):
    """Same call as solve_milp of exact_helper.py; the solution vector is kept to check it."""
    import time
    from scipy.optimize import milp

    mf, c, cons, bounds, integ = scipy_input(d)
    t0 = time.time()
    res = milp(c, constraints=cons, integrality=integ, bounds=bounds,
               options=dict(time_limit=limit, mip_rel_gap=1e-6, disp=False))
    sec = time.time() - t0
    obj = float(res.fun) if res.x is not None else None
    bound = getattr(res, "mip_dual_bound", None)
    out = _pack(obj, None if bound is None else float(bound), sec)
    out["mf"] = mf
    out["x"] = res.x
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="35,60,100,150", help="35 is the own instance")
    ap.add_argument("--limit", type=float, default=120.0)
    ap.add_argument("--out", default=os.path.join(RESULTS, "final_exact.csv"))
    a = ap.parse_args()

    new = not os.path.exists(a.out)
    with open(a.out, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(COLS)
        for n in [int(s) for s in a.sizes.split(",")]:
            d = own_instance() if n == 35 else scaled(n)
            name = "own" if n == 35 else f"scaled_{n}"
            r = solve_one(d, a.limit)
            mf = r["mf"]
            cost, feas = "", ""
            if r["x"] is not None:
                centers, assign = solution_from_vector(mf, r["x"])
                cost, problems = gen.evaluate_A(d, 1, centers, assign)
                feas = len(problems) == 0
                with open(os.path.join(RESULTS, f"final_exact_{n}.json"), "w") as fj:
                    json.dump({"centers": centers, "assign": assign, "vehicle": None}, fj)
            w.writerow([name, n, d["V"], d["Cap"], mf.n_variables, mf.n_rows,
                        r["objective"], r["bound"], r["solver_gap"], r["proven"], r["seconds"],
                        cost, feas])
            fh.flush()
            print(name, r["objective"], r["bound"], r["solver_gap"], r["proven"], r["seconds"])


if __name__ == "__main__":
    main()
