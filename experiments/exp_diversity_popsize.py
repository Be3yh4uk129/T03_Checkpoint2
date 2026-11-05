"""Checkpoint 4, role GA-A: the required item of the analysis.

Author: Ikhsanova Aziza (GA-A).

Two things the instructions ask GA-A to show:
  1. the population diversity against the generation;
  2. the effect of the population size N (two values) on the cost.

Diversity is the number of distinct individuals in the population, counted once per
generation by the loop itself. The population sizes are compared on the tuning seeds
100 to 104, never on the final seeds 0 to 9.

    PYTHONPATH=. python experiments/exp_diversity_popsize.py

Writes results/Ikhsanova_exp.csv and figures/F2_diversity.png.
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
import numpy as np

import checkpoint4_generator_PUBLIC as gen
from src.meta import ga

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(ROOT, "figures")

TEAM, ROLE = "T03", "GA-A"
TUNING_SEEDS = [100, 101, 102, 103, 104]
BUDGET = 100000
POP_SIZES = [30, 60]            # the rule of Section 4, and twice it
COLS = ["team", "role", "configuration", "seed", "budget", "cost", "evals",
        "feasible", "seconds", "params"]


def main():
    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(FIGURES, exist_ok=True)
    d = gen.cp34_instance(3, "A", 1)

    rows = []
    curves = {}
    summary = {}

    for pop_size in POP_SIZES:
        costs, feas, secs = [], 0, []
        for seed in TUNING_SEEDS:
            t0 = time.time()
            r = ga.solve(d, 1, seed, BUDGET, params={"pop_size": pop_size})
            sec = time.time() - t0
            costs.append(r["cost"])
            feas += int(r["feasible"])
            secs.append(sec)
            rows.append([TEAM, ROLE, f"GA, penalty, N={pop_size}", seed, BUDGET,
                         f"{r['cost']:.9f}", r["evals"], r["feasible"], f"{sec:.2f}",
                         json.dumps({"pop_size": pop_size, "rho": r["params"]["rho"]})])
            if seed == TUNING_SEEDS[0]:
                curves[pop_size] = {"individuals": r["diversity"],
                                    "center_sets": r["diversity_centers"],
                                    "fitness_spread": r["fitness_spread"]}
            print(f"  N={pop_size:3d} seed {seed}: cost {r['cost']:.6f} "
                  f"feasible {r['feasible']} gens {r['generations']} {sec:.1f}s")
        summary[pop_size] = {
            "mean": statistics.fmean(costs),
            "best": min(costs),
            "worst": max(costs),
            "sd": statistics.stdev(costs) if len(costs) > 1 else 0.0,
            "feasible": feas,
            "seconds": statistics.fmean(secs),
        }

    with open(os.path.join(RESULTS, "Ikhsanova_exp.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(COLS)
        w.writerows(rows)
    print("wrote results/Ikhsanova_exp.csv")

    print("\npopulation size, seeds 100-104, budget 100000")
    print(f"{'N':>4} {'best':>10} {'mean':>10} {'worst':>10} {'sd':>8} {'feas':>5} {'s/run':>7}")
    for n, s in summary.items():
        print(f"{n:>4} {s['best']:>10.3f} {s['mean']:>10.3f} {s['worst']:>10.3f} "
              f"{s['sd']:>8.3f} {s['feasible']:>3}/5 {s['seconds']:>7.1f}")

    def smooth(v, window=51):
        """Rolling mean: the per-generation counts jump every generation and the trend
        is what the question is about."""
        v = np.asarray(v, dtype=float)
        if v.size < window:
            return v
        k = np.ones(window) / window
        return np.convolve(v, k, mode="valid")

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
    colors = {30: "#4c78a8", 60: "#e45756"}
    for n, c in curves.items():
        ind, cs, fs = smooth(c["individuals"]), smooth(c["center_sets"]), smooth(c["fitness_spread"])
        x = np.arange(ind.size) / max(1, ind.size - 1)
        axes[0].plot(x, ind / n, lw=1.3, color=colors.get(n),
                     label=f"N = {n}: distinct individuals")
        axes[0].plot(x, cs / n, lw=1.3, ls="--", color=colors.get(n),
                     label=f"N = {n}: distinct center sets")
        axes[1].plot(np.arange(fs.size) / max(1, fs.size - 1), fs, lw=1.3,
                     color=colors.get(n), label=f"N = {n}")
    axes[0].set_xlabel("share of the generations")
    axes[0].set_ylabel("distinct / N")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title(f"F2a: diversity against the generation (seed {TUNING_SEEDS[0]}, rolling mean of 51)")
    axes[1].set_xlabel("share of the generations")
    axes[1].set_ylabel("worst minus best fitness")
    axes[1].set_title("F2b: spread of the fitness in the population")
    for ax in axes:
        ax.grid(True, ls=":", lw=0.4, alpha=0.6)
        ax.legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "F2_diversity.png"), dpi=160)
    print("wrote figures/F2_diversity.png")

    with open(os.path.join(RESULTS, "Ikhsanova_exp_summary.json"), "w") as fh:
        json.dump({"pop_size": {str(k): v for k, v in summary.items()},
                   "diversity_curves": {str(k): v for k, v in curves.items()}}, fh, indent=1)
    print("wrote results/Ikhsanova_exp_summary.json")


if __name__ == "__main__":
    main()
