"""Checkpoint 4, role GA-P: the required item of the analysis.

Author: Abdrashit Yersultan (GA-P).

The penalty configuration has to be shown with three values of the penalty weight rho,
reporting for each of them

  * the share of feasible offspring,
  * the final feasible cost,
  * the share of runs that end infeasible.

The runs use the tuning seeds 100 to 104, never the final seeds 0 to 9.

    PYTHONPATH=. python experiments/exp_rho.py

Writes results/Abdrashit_exp.csv and figures/F3_rho.png.
"""
import csv
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import checkpoint4_generator_PUBLIC as gen
from src.meta import ga

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(ROOT, "figures")

TEAM, ROLE = "T03", "GA-P"
TUNING_SEEDS = [100, 101, 102, 103, 104]
BUDGET = 100000
RHOS = [1.0, 10.0, 100.0]
COLS = ["team", "role", "configuration", "seed", "budget", "cost", "evals",
        "feasible", "seconds", "params"]


def main():
    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(FIGURES, exist_ok=True)
    d = gen.cp34_instance(3, "A", 1)

    rows, summary = [], {}
    for rho in RHOS:
        costs, shares, n_infeasible, secs = [], [], 0, []
        for seed in TUNING_SEEDS:
            t0 = time.time()
            r = ga.solve(d, 1, seed, BUDGET, params={"rho": rho})
            sec = time.time() - t0
            share = r["offspring_feasible"] / max(1, r["offspring"])
            costs.append(r["cost"])
            shares.append(share)
            secs.append(sec)
            n_infeasible += int(not r["feasible"])
            rows.append([TEAM, ROLE, f"GA, penalty, rho={rho:g}", seed, BUDGET,
                         f"{r['cost']:.9f}", r["evals"], r["feasible"], f"{sec:.2f}",
                         json.dumps({"rho": rho, "feasible_offspring_share": round(share, 6)})])
            print(f"  rho={rho:>6g} seed {seed}: cost {r['cost']:.6f} feasible {r['feasible']} "
                  f"feasible offspring {100*share:5.1f}%  {sec:.1f}s")
        summary[rho] = {
            "mean_cost": statistics.fmean(costs),
            "best_cost": min(costs),
            "worst_cost": max(costs),
            "mean_feasible_offspring": statistics.fmean(shares),
            "runs_infeasible": n_infeasible,
            "runs": len(TUNING_SEEDS),
            "seconds": statistics.fmean(secs),
        }

    with open(os.path.join(RESULTS, "Abdrashit_exp.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(COLS)
        w.writerows(rows)
    print("wrote results/Abdrashit_exp.csv")

    print("\npenalty weight, seeds 100-104, budget 100000")
    print(f"{'rho':>7} {'feas.offspr':>12} {'best':>10} {'mean':>10} {'worst':>10} {'infeas runs':>12}")
    for rho, s in summary.items():
        print(f"{rho:>7g} {100*s['mean_feasible_offspring']:>11.1f}% {s['best_cost']:>10.3f} "
              f"{s['mean_cost']:>10.3f} {s['worst_cost']:>10.3f} {s['runs_infeasible']:>8}/{s['runs']}")

    fig, ax1 = plt.subplots(figsize=(7.0, 4.0))
    xs = list(range(len(RHOS)))
    ax1.bar([x - 0.18 for x in xs],
            [100 * summary[r]["mean_feasible_offspring"] for r in RHOS],
            width=0.36, color="#4c78a8", label="feasible offspring (%)")
    ax1.set_ylabel("feasible offspring (%)")
    ax1.set_xticks(xs)
    ax1.set_xticklabels([f"rho = {r:g}" for r in RHOS])
    ax1.set_ylim(0, 100)
    ax2 = ax1.twinx()
    ax2.plot(xs, [summary[r]["mean_cost"] for r in RHOS], "o-", color="#e45756",
             label="mean final cost")
    ax2.set_ylabel("mean final feasible cost")
    ax1.set_title("F3: effect of the penalty weight")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=8, loc="lower center")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "F3_rho.png"), dpi=160)
    print("wrote figures/F3_rho.png")

    with open(os.path.join(RESULTS, "Abdrashit_exp_summary.json"), "w") as fh:
        json.dump({str(k): v for k, v in summary.items()}, fh, indent=1)
    print("wrote results/Abdrashit_exp_summary.json")


if __name__ == "__main__":
    main()
