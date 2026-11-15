"""Block 2: Table 2 and Figure F1 of the Final.

Inputs (all from Checkpoint 4 and the exact run of the Final):
  results/Ikhsanova_runs.csv       GA, penalty, seeds 0-9 (GA pair)
  results/Amanbay_runs.csv         SA, transfer+swap, tuned, seeds 0-9 (SA pair)
  results/Ikhsanova_analysis.json  random search of the GA pair (table1_random)
  results/Kaniyeva_tuning.csv      random search of the SA pair (configuration random)
  results/final_exact.csv          exact solver, own instance
  results/final_cp4_reruns.json    best-so-far traces of the 20 final runs (final/src/rerun_cp4.py)

    python final/src/block2.py
Writes results/final_table2.csv and figures/final_F1_methods.png.
"""
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import FIGURES, RESULTS, Z_LP, Z_REF, gaps   # noqa: E402


def stats(costs, feasible, seconds):
    c = np.asarray(costs, dtype=float)
    return dict(best=c.min(), mean=c.mean(), worst=c.max(), std=c.std(ddof=1) if c.size > 1 else 0.0,
                feasible=int(sum(feasible)), runs=int(c.size), seconds=float(np.mean(seconds)))


def from_runs(name, configuration=None):
    rows = [r for r in csv.DictReader(open(os.path.join(RESULTS, name)))
            if configuration is None or r["configuration"] == configuration]
    return stats([float(r["cost"]) for r in rows], [r["feasible"] == "True" for r in rows],
                 [float(r["seconds"]) for r in rows])


def main():
    table = []
    table.append(("GA, penalty (rho = 1)", from_runs("Ikhsanova_runs.csv", "GA, penalty")))
    table.append(("SA, transfer+swap (tuned)", from_runs("Amanbay_runs.csv", "SA, transfer+swap")))
    rnd = json.load(open(os.path.join(RESULTS, "Ikhsanova_analysis.json")))["table1_random"]
    table.append(("Random search (GA pair)", dict(best=rnd["best"], mean=rnd["mean"], worst=rnd["worst"], std=rnd["sd"],
                                                  feasible=rnd["feasible"], runs=rnd["runs"], seconds=rnd["seconds"])))
    rows = [r for r in csv.DictReader(open(os.path.join(RESULTS, "Kaniyeva_tuning.csv"))) if r["configuration"] == "random"]
    table.append(("Random search (SA pair)", stats([float(r["cost"]) for r in rows], [r["feasible"] == "True" for r in rows],
                                                   [float(r["seconds"]) for r in rows])))
    ex = [r for r in csv.DictReader(open(os.path.join(RESULTS, "final_exact.csv"))) if r["instance"] == "own"][-1]
    z = float(ex["checker_cost"])
    table.append(("Exact solver (HiGHS)", dict(best=z, mean=z, worst=z, std=0.0, feasible=1, runs=1,
                                               seconds=float(ex["seconds"]))))

    with open(os.path.join(RESULTS, "final_table2.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "best", "mean", "worst", "std", "feasible", "runs", "mean_seconds",
                    "gap_ref_best", "gap_ref_mean", "gap_ref_worst", "gap_lp_best", "gap_lp_mean", "gap_lp_worst"])
        for name, s in table:
            gb, lb = gaps(s["best"]); gm, lm = gaps(s["mean"]); gw, lw = gaps(s["worst"])
            w.writerow([name, f'{s["best"]:.6f}', f'{s["mean"]:.6f}', f'{s["worst"]:.6f}', f'{s["std"]:.6f}',
                        s["feasible"], s["runs"], f'{s["seconds"]:.3f}',
                        f"{gb:.4f}", f"{gm:.4f}", f"{gw:.4f}", f"{lb:.4f}", f"{lm:.4f}", f"{lw:.4f}"])
            print(f'{name:28s} {s["best"]:9.3f} {s["mean"]:9.3f} {s["worst"]:9.3f} {s["std"]:7.3f} '
                  f'{s["feasible"]}/{s["runs"]} {s["seconds"]:7.3f}s  gap_ref {gm:.3f}  gap_LP {lm:.3f}')

    # Figure F1: mean best-so-far with the min-max band, both methods, on a common grid of evaluations
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reruns = json.load(open(os.path.join(RESULTS, "final_cp4_reruns.json")))
    grid = np.arange(1000, 100001, 500)
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    summary = {}
    for label, color in (("GA, penalty", "#1f77b4"), ("SA, transfer+swap", "#d62728")):
        curves = []
        for r in (x for x in reruns if x["method"] == label):
            e = np.array([t[0] for t in r["trace"]], dtype=float)
            b = np.array([t[1] if t[1] is not None else np.inf for t in r["trace"]], dtype=float)
            idx = np.searchsorted(e, grid, side="right") - 1
            curves.append(np.where(idx >= 0, b[np.clip(idx, 0, None)], np.inf))
        C = np.array(curves)
        mean = C.mean(axis=0)
        ax.fill_between(grid, C.min(axis=0), C.max(axis=0), color=color, alpha=0.18, lw=0)
        ax.plot(grid, mean, color=color, lw=1.6, label=f"{label}: mean of 10 runs (band: min to max)")
        summary[label] = {str(g): float(mean[list(grid).index(g)]) for g in (1000, 10000, 50000, 100000)}
    ax.axhline(Z_REF, color="k", ls="--", lw=1, label=r"$z_{\mathrm{ref}}$ = 997.604")
    ax.axhline(Z_LP, color="gray", ls=":", lw=1.2, label=r"$z_{\mathrm{LP}}$ = 994.049")
    ax.set_ylim(990, 1100)
    ax.set_xlim(0, 100000)
    ax.set_xlabel("evaluations")
    ax.set_ylabel("best feasible cost so far")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(FIGURES, exist_ok=True)
    fig.savefig(os.path.join(FIGURES, "final_F1_methods.png"), dpi=200)
    json.dump(summary, open(os.path.join(RESULTS, "final_F1_points.json"), "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
