"""Nelder-Mead tuning of T0 and Tend for the SA (role SA-T, Zhamilya Kaniyeva).

    PYTHONPATH=. python experiments/tuning.py                     # everything, about 3 minutes
    PYTHONPATH=. python experiments/tuning.py --inner-budget 20000  # faster, must be stated

Steps
 1. p0 = (log10 T0, log10 Tend) from the rule of Section 4 (sa.rule_params: random walk of
    3000 proposals that accepts everything, on tuning seed 100).
 2. Nelder-Mead on x = (log10 T0, log10 Tend), simplex p0, p0 + 0.3 e1, p0 + 0.3 e2,
    coefficients 1, 2, 1/2, 1/2, at most 80 calls of the objective.
    objective(x) = mean over seeds 100..104 of the best feasible cost of sa.solve with
    T0 = 10**x[0], Tend = 10**x[1]; an infeasible run counts as 10 x its starting cost.
 3. The tuning table: rule-based values and tuned values, each run on seeds 0..9 (budget
    100000), mean / worst / std of cost and gap_ref.
 4. The random-search baseline on seeds 0..9, budget 100000 (configuration = random).

Writes
    results/Kaniyeva_tuning.csv          every run (tuning calls, comparison, baseline)
    results/Kaniyeva_tuned_params.json   p0, the tuned T0 and Tend, the objective values
    results/Kaniyeva_tuning_table.csv    the tuning table
    results/Kaniyeva_nm_trace.csv        best objective against the objective calls
    figures/F5_tuning_trace.png          the same as a figure
z_ref is used here only for reporting the gaps, never inside the solver.
"""
import argparse
import csv
import json
import os
import statistics
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.meta import sa, random_search_sa          # noqa: E402
from src.meta.nm import nelder_mead                # noqa: E402

TEAM, ROLE = "T03", "SA-T"
CONFIG = "SA, transfer+swap"
TUNING_SEEDS = [100, 101, 102, 103, 104]
FINAL_SEEDS = list(range(10))
Z_REF = 997.603759          # reporting only
Z_LP = 994.048826           # reporting only
COLUMNS = ["team", "role", "configuration", "seed", "budget", "cost", "evals", "feasible",
           "seconds", "params"]


def gap_ref(z):
    return 100.0 * (z - Z_REF) / Z_REF


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inner-budget", type=int, default=100000)
    ap.add_argument("--max-calls", type=int, default=80)
    ap.add_argument("--final-budget", type=int, default=100000)
    args = ap.parse_args()
    if args.inner_budget < 20000:
        raise SystemExit("the inner budget must be at least 20000")

    d = sa._generator().cp34_instance(3, "A", 1)
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)
    out_csv = os.path.join(ROOT, "results", "Kaniyeva_tuning.csv")
    fh = open(out_csv, "w", newline="")
    wr = csv.writer(fh)
    wr.writerow(COLUMNS)

    def run(module, seed, budget, params, configuration, tag):
        t = time.perf_counter()
        r = module.solve(d, 1, seed, budget, params=params)
        sec = time.perf_counter() - t
        p = dict(params or {})
        p.update(tag)
        wr.writerow([TEAM, ROLE, configuration, seed, budget, repr(r["cost"]), r["evals"],
                     r["feasible"], round(sec, 4), json.dumps(p)])
        fh.flush()
        return r

    # 1. the rule-based start p0
    rule = sa.rule_params(d)
    p0 = np.array([np.log10(rule["T0"]), np.log10(rule["Tend"])])
    print("rule: T0 = %.6g, Tend = %.6g, p0 = %s" % (rule["T0"], rule["Tend"], p0))

    # 2. Nelder-Mead
    history = []                    # (call, x0, x1, objective, best so far)

    def objective(x):
        call = len(history) + 1
        T0, Tend = 10.0 ** x[0], 10.0 ** x[1]
        vals = []
        for s in TUNING_SEEDS:
            r = run(sa, s, args.inner_budget, {"T0": T0, "Tend": Tend},
                    CONFIG + " (tuning)", {"call": call, "x": [float(x[0]), float(x[1])]})
            vals.append(r["cost"] if r["feasible"] else 10.0 * r["start_cost"])
        v = float(np.mean(vals))
        best = min([h[3] for h in history] + [v])
        history.append((call, float(x[0]), float(x[1]), v, best))
        print("call %2d  log10 T0 = %+.4f  log10 Tend = %+.4f  objective = %.6f  best = %.6f"
              % (call, x[0], x[1], v, best))
        return v

    simplex = [p0, p0 + np.array([0.3, 0.0]), p0 + np.array([0.0, 0.3])]
    res = nelder_mead(objective, simplex, max_evals=args.max_calls, xtol=1e-3, ftol=1e-6)
    tuned = {"T0": 10.0 ** res["x"][0], "Tend": 10.0 ** res["x"][1]}
    obj_rule = history[0][3]
    obj_tuned = res["fx"]
    chosen = "tuned" if obj_tuned < obj_rule else "rule"
    print("tuned: T0 = %.6g, Tend = %.6g, objective %.6f (rule %.6f) -> use %s"
          % (tuned["T0"], tuned["Tend"], obj_tuned, obj_rule, chosen))

    with open(os.path.join(ROOT, "results", "Kaniyeva_nm_trace.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["call", "log10_T0", "log10_Tend", "objective", "best_so_far"])
        w.writerows(history)

    # 3. the tuning table on seeds 0..9
    rows = []
    for name, prm in (("rule", {"T0": rule["T0"], "Tend": rule["Tend"]}), ("tuned", tuned)):
        costs, feas = [], 0
        for s in FINAL_SEEDS:
            r = run(sa, s, args.final_budget, prm, CONFIG + " (%s)" % name, {"set": name})
            if r["feasible"]:
                feas += 1
                costs.append(r["cost"])
        rows.append({
            "set": name, "T0": prm["T0"], "Tend": prm["Tend"],
            "objective_seeds_100_104": obj_rule if name == "rule" else obj_tuned,
            "feasible_of_10": feas,
            "mean_cost": statistics.mean(costs), "worst_cost": max(costs),
            "std_cost": statistics.stdev(costs) if len(costs) > 1 else 0.0,
            "mean_gap_ref": gap_ref(statistics.mean(costs)), "worst_gap_ref": gap_ref(max(costs)),
            "std_gap_ref": 100.0 * (statistics.stdev(costs) if len(costs) > 1 else 0.0) / Z_REF,
        })
    with open(os.path.join(ROOT, "results", "Kaniyeva_tuning_table.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    gain = rows[0]["mean_gap_ref"] - rows[1]["mean_gap_ref"]
    print("tuning table:")
    for r in rows:
        print("  %-5s mean gap_ref %.4f %%  worst %.4f %%  std %.4f %%"
              % (r["set"], r["mean_gap_ref"], r["worst_gap_ref"], r["std_gap_ref"]))
    print("gain of tuning in mean gap_ref: %.4f points; std across seeds (rule): %.4f points"
          % (gain, rows[0]["std_gap_ref"]))

    # 4. random-search baseline on seeds 0..9
    rnd = []
    for s in FINAL_SEEDS:
        r = run(random_search_sa, s, args.final_budget, None, "random", {})
        rnd.append(r)
    feas_rnd = [r["cost"] for r in rnd if r["feasible"]]
    print("random search: %d/10 feasible, mean %.4f, best %.4f"
          % (len(feas_rnd), statistics.mean(feas_rnd), min(feas_rnd)))
    fh.close()

    summary = {
        "rule": rule, "p0": [float(p0[0]), float(p0[1])], "tuned": tuned,
        "x_tuned": [float(res["x"][0]), float(res["x"][1])],
        "objective_rule_p0": obj_rule, "objective_tuned": obj_tuned,
        "objective_calls": res["evals"], "nm_iterations": res["iterations"],
        "inner_budget": args.inner_budget, "chosen_on_seeds_100_104": chosen,
        "table": rows, "gain_mean_gap_ref_points": gain,
        "random_search": {"feasible_of_10": len(feas_rnd), "mean_cost": statistics.mean(feas_rnd),
                          "best_cost": min(feas_rnd), "worst_cost": max(feas_rnd)},
    }
    with open(os.path.join(ROOT, "results", "Kaniyeva_tuned_params.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # the trace figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 3.4))
    calls = [h[0] for h in history]
    ax.plot(calls, [h[3] for h in history], "o", ms=3, color="0.6", label="objective of the call")
    ax.step(calls, [h[4] for h in history], where="post", color="C0", label="best so far")
    ax.axhline(obj_rule, color="C3", ls="--", lw=1, label="rule-based p0")
    ax.set_xlabel("objective call")
    ax.set_ylabel("mean best cost, seeds 100-104")
    ax.set_title("Nelder-Mead tuning of (log10 T0, log10 Tend)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "figures", "F5_tuning_trace.png"), dpi=150)
    print("written:", out_csv)


if __name__ == "__main__":
    main()
