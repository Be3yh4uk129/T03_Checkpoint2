"""Block 3: Table 3 and Figure F2 of the Final.

Inputs:
  results/final_exact.csv          exact solver (final/src/exact_scaled.py); if a size was run more than once,
                                   every run is kept and the table uses the last one
  results/final_scaling_runs.csv   GA and SA, seeds 0-9, own and LP-rounded start (final/run_block3_methods.sh)
  results/final_lp_rounding.csv    z_LP of each scaled instance and the cost of the rounded start

    python final/src/block3.py
Writes results/final_table3.csv and figures/final_F2_exact_time.png.
"""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import FIGURES, RESULTS, SIZES   # noqa: E402

METHODS = ("GA, own start", "GA, LP-rounded start", "SA, own start", "SA, LP-rounded start")


def main():
    exact = list(csv.DictReader(open(os.path.join(RESULTS, "final_exact.csv"))))
    runs = list(csv.DictReader(open(os.path.join(RESULTS, "final_scaling_runs.csv"))))
    lp = {int(r["N"]): r for r in csv.DictReader(open(os.path.join(RESULTS, "final_lp_rounding.csv")))}

    out = []
    for n in SIZES:
        ex = [r for r in exact if int(r["N"]) == n][-1]
        rs = {m: [r for r in runs if int(r["size"]) == n and r["configuration"] == m] for m in METHODS}
        allc = [float(r["cost"]) for m in METHODS for r in rs[m] if r["feasible"] == "True"]
        zex = float(ex["checker_cost"]) if ex["checker_cost"] else None
        best_known = min(allc + ([zex] if zex is not None else []))
        zlp = float(lp[n]["z_LP"])
        out.append(dict(N=n, method="exact (HiGHS, 120 s)", value_mean=zex, value_best=zex, value_worst=zex, std=0.0,
                        feasible=1 if zex is not None else 0, runs=1, best_known=best_known,
                        gap_best_known_mean=100 * (zex - best_known) / best_known,
                        gap_best_known_best=100 * (zex - best_known) / best_known,
                        solver_gap=100 * float(ex["solver_gap"]), proven=ex["proven"], seconds=float(ex["seconds"]),
                        z_LP=zlp, gap_lp_mean=100 * (zex - zlp) / zex,
                        n_better_than_exact=""))
        for m in METHODS:
            assert len(rs[m]) == 10 and all(int(r["evals"]) == 100000 for r in rs[m]), (n, m, len(rs[m]))
            # only feasible runs give a value; an infeasible run is counted in the feasible column only
            c = np.array([float(r["cost"]) for r in rs[m] if r["feasible"] == "True"])
            sec = float(np.mean([float(r["seconds"]) for r in rs[m]]))
            if c.size == 0:
                out.append(dict(N=n, method=m, value_mean=None, value_best=None, value_worst=None, std=None,
                                feasible=0, runs=len(rs[m]), best_known=best_known, gap_best_known_mean=None,
                                gap_best_known_best=None, solver_gap="", proven="", seconds=sec, z_LP=zlp,
                                gap_lp_mean=None, n_better_than_exact=0))
                continue
            out.append(dict(N=n, method=m, value_mean=c.mean(), value_best=c.min(), value_worst=c.max(),
                            std=c.std(ddof=1) if c.size > 1 else 0.0, feasible=int(c.size), runs=len(rs[m]),
                            best_known=best_known,
                            gap_best_known_mean=100 * (c.mean() - best_known) / best_known,
                            gap_best_known_best=100 * (c.min() - best_known) / best_known,
                            solver_gap="", proven="", seconds=sec,
                            z_LP=zlp, gap_lp_mean=100 * (c.mean() - zlp) / c.mean(),
                            n_better_than_exact=int((c < zex - 1e-9).sum())))

    keys = list(out[0].keys())
    with open(os.path.join(RESULTS, "final_table3.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in out:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})
    for r in out:
        print(r["N"], r["method"], "feasible", r["feasible"], "mean", r["value_mean"], "best", r["value_best"],
              "gap_bk_mean", r["gap_best_known_mean"], "solver_gap", r["solver_gap"], "s", round(r["seconds"], 2),
              "better_than_exact", r["n_better_than_exact"])

    # the penalty check (final/run_block3_extra.sh): GA with rho = 10 on N = 100, not part of Table 3
    chk = os.path.join(RESULTS, "final_scaling_check.csv")
    if os.path.exists(chk):
        rc = list(csv.DictReader(open(chk)))
        c = np.array([float(r["cost"]) for r in rc if r["feasible"] == "True"])
        bk = [r for r in out if r["N"] == 100][0]["best_known"]
        with open(os.path.join(RESULTS, "final_table3_check.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["N", "configuration", "feasible", "runs", "mean", "best", "worst", "std", "gap_best_known_mean"])
            w.writerow([100, rc[0]["configuration"], int(c.size), len(rc), f"{c.mean():.4f}", f"{c.min():.4f}",
                        f"{c.max():.4f}", f"{c.std(ddof=1):.4f}", f"{100 * (c.mean() - bk) / bk:.4f}"])
        print("check rho=10:", c.size, c.mean(), c.min(), 100 * (c.mean() - bk) / bk)

    # Figure F2: time of the exact solver against size, limit marked
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    for r in exact:
        n, t, proven = int(r["N"]), float(r["seconds"]), r["proven"] == "True"
        ax.scatter(n, t, marker="o" if proven else "x", s=46, color="#1f77b4" if proven else "#d62728", zorder=3)
    last = {}
    for r in exact:
        last[int(r["N"])] = float(r["seconds"])
    ns = sorted(last)
    ax.plot(ns, [last[n] for n in ns], color="gray", lw=1, zorder=2)
    ax.axhline(120, color="k", ls="--", lw=1)
    ax.text(36, 123, "time limit 120 s", fontsize=8)
    ax.scatter([], [], marker="o", color="#1f77b4", label="optimum proven (solver gap 0)")
    ax.scatter([], [], marker="x", color="#d62728", label="stopped by the limit, not proven")
    ax.set_xlabel("number of points N (35 = our own instance)")
    ax.set_ylabel("exact solver time, s")
    ax.set_xticks([35] + list(SIZES))
    ax.set_ylim(-5, 140)
    ax.legend(fontsize=8, loc="center left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "final_F2_exact_time.png"), dpi=200)


if __name__ == "__main__":
    main()
