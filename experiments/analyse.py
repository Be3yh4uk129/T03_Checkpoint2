"""Checkpoint 4, role GA-A: Table 1, Table 2 and Figure F1 from the results files.

Author: Ikhsanova Aziza (GA-A).

    PYTHONPATH=. python experiments/analyse.py

Reads results/Ikhsanova_runs.csv (the final runs and the random-search baseline) and
reruns solve on the same seeds to recover the traces, which the results file does not
store. The rerun is deterministic, so every recovered cost is compared with the cost in
the file and the script stops if they differ.

Writes figures/F1_bestsofar.png and results/Ikhsanova_analysis.json, which holds every
number quoted in the report.
"""
import csv
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import checkpoint4_generator_PUBLIC as gen
from src.meta import ga, ga_problem as gp, random_search as rs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(ROOT, "figures")

Z_LP = 994.048826        # Checkpoint 3, the LP relaxation
Z_REF = 997.603759       # the optimal value from the task file; reporting only
BUDGET = 100000
SEEDS = list(range(10))


def gaps(z):
    """The two gaps of Section 3 of the instructions, in percent."""
    return 100.0 * (z - Z_REF) / Z_REF, 100.0 * (z - Z_LP) / z


def read_runs(path):
    rows = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            r["cost"] = float(r["cost"])
            r["seed"] = int(r["seed"])
            r["seconds"] = float(r["seconds"])
            r["feasible"] = r["feasible"] == "True"
            rows.append(r)
    return rows


def summarise(costs, seconds, n_feasible, n_runs):
    best, worst = min(costs), max(costs)
    mean = statistics.fmean(costs)
    median = statistics.median(costs)
    sd = statistics.stdev(costs) if len(costs) > 1 else 0.0
    out = {"best": best, "median": median, "mean": mean, "worst": worst, "sd": sd,
           "feasible": n_feasible, "runs": n_runs, "seconds": statistics.fmean(seconds)}
    for name, z in (("best", best), ("median", median), ("mean", mean), ("worst", worst)):
        gr, gl = gaps(z)
        out[f"gap_ref_{name}"] = gr
        out[f"gap_lp_{name}"] = gl
    return out


def improving_moves(d, a):
    """Count the improving transfers and the improving swaps from a final solution.

    A move is improving when the result is feasible and strictly cheaper. Only the two
    moves of the task file are counted: moves that change the set of centers are not.
    """
    a = np.asarray(a)
    centers = gp.centers_of(a)
    non_centers = [i for i in range(d["N"]) if a[i] != i]
    base_cost = gp.cost_of(d, a)
    cap = float(d["Cap"])

    def feasible(x):
        return float(np.maximum(gp.load_vector(d, x) - cap, 0.0).sum()) <= 0.0

    n_transfer = 0
    for i in non_centers:
        for c in centers:
            if c == a[i]:
                continue
            b = a.copy()
            b[i] = c
            if feasible(b) and gp.cost_of(d, b) < base_cost - 1e-9:
                n_transfer += 1

    n_swap = 0
    for x in range(len(non_centers)):
        for y in range(x + 1, len(non_centers)):
            i, j = non_centers[x], non_centers[y]
            if a[i] == a[j]:
                continue
            b = a.copy()
            b[i], b[j] = a[j], a[i]
            if feasible(b) and gp.cost_of(d, b) < base_cost - 1e-9:
                n_swap += 1
    return n_transfer, n_swap


def main():
    os.makedirs(FIGURES, exist_ok=True)
    d = gen.cp34_instance(3, "A", 1)

    rows = read_runs(os.path.join(RESULTS, "Ikhsanova_runs.csv"))
    ga_rows = [r for r in rows if r["configuration"] != "random"]
    rnd_rows = [r for r in rows if r["configuration"] == "random"]
    if len(ga_rows) != 10:
        sys.exit(f"expected 10 final runs, found {len(ga_rows)}")

    # ---- rerun to recover the traces, checking every cost against the file ----
    traces, diversity, offspring = {}, {}, {}
    results = {}
    for r in sorted(ga_rows, key=lambda r: r["seed"]):
        res = ga.solve(d, 1, r["seed"], BUDGET)
        if abs(res["cost"] - r["cost"]) > 1e-6:
            sys.exit(f"seed {r['seed']}: rerun gives {res['cost']:.6f}, file says {r['cost']:.6f}")
        traces[r["seed"]] = res["trace"]
        diversity[r["seed"]] = res["diversity"]
        offspring[r["seed"]] = (res["offspring_feasible"], res["offspring"])
        results[r["seed"]] = res
        print(f"  seed {r['seed']}: {res['cost']:.6f} reproduced")

    costs = [r["cost"] for r in ga_rows]
    table1 = summarise(costs, [r["seconds"] for r in ga_rows],
                       sum(r["feasible"] for r in ga_rows), len(ga_rows))

    table1_rnd = None
    if rnd_rows:
        table1_rnd = summarise([r["cost"] for r in rnd_rows],
                               [r["seconds"] for r in rnd_rows],
                               sum(r["feasible"] for r in rnd_rows), len(rnd_rows))

    # ---- Table 2: the worst feasible run ------------------------------------
    feasible_rows = [r for r in ga_rows if r["feasible"]]
    worst_row = max(feasible_rows, key=lambda r: r["cost"]) if feasible_rows else None
    table2 = None
    if worst_row:
        s = worst_row["seed"]
        res = results[s]
        a = np.array(res["solution"][1], dtype=np.int64)
        n_t, n_s = improving_moves(d, a)
        div = diversity[s]
        table2 = {
            "seed": s,
            "final_cost": res["cost"],
            "improving_transfers": n_t,
            "improving_swaps": n_s,
            "distinct_first_generation": div[0],
            "distinct_last_generation": div[-1],
            "generations": len(div),
            "feasible_offspring": offspring[s][0],
            "offspring": offspring[s][1],
            "feasible_offspring_share": offspring[s][0] / max(1, offspring[s][1]),
        }
        print(f"\nworst feasible seed {s}: cost {res['cost']:.6f}, "
              f"improving transfers {n_t}, improving swaps {n_s}, "
              f"distinct {div[0]} -> {div[-1]}")

    # ---- convergence: where does the curve flatten? -------------------------
    flat = {}
    for s, tr in traces.items():
        ev = np.array([e for e, _ in tr], dtype=float)
        bc = np.array([c for _, c in tr], dtype=float)
        ok = np.isfinite(bc)
        ev, bc = ev[ok], bc[ok]
        final = bc[-1]
        within = np.where(bc <= final * 1.01)[0]
        flat[s] = float(ev[within[0]] / BUDGET) if within.size else 1.0
    last_tenth_drop = {}
    for s, tr in traces.items():
        bc = [c for _, c in tr if np.isfinite(c)]
        ev = [e for e, c in tr if np.isfinite(c)]
        cut = next((k for k, e in enumerate(ev) if e >= 0.9 * BUDGET), len(bc) - 1)
        last_tenth_drop[s] = bc[cut] - bc[-1]

    # ---- Figure F1 ----------------------------------------------------------
    grid = np.arange(0, BUDGET + 1, 1000, dtype=float)
    curves = []
    for s, tr in traces.items():
        ev = np.array([e for e, _ in tr], dtype=float)
        bc = np.array([c for _, c in tr], dtype=float)
        bc = np.where(np.isfinite(bc), bc, np.nan)
        curves.append(np.interp(grid, ev, bc))
    curves = np.vstack(curves)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.fill_between(grid, np.nanmin(curves, axis=0), np.nanmax(curves, axis=0),
                    color="#4c78a8", alpha=0.20, label="min to max over the 10 seeds")
    ax.plot(grid, np.nanmean(curves, axis=0), color="#4c78a8", lw=1.6,
            label="GA, penalty: mean best-so-far")
    if table1_rnd:
        # the random-search mean ends near 1555, far above the range that makes the GA
        # curve readable, so it is reported as a number instead of a line
        ax.annotate(f"random search, same budget and seeds: mean {table1_rnd['mean']:.1f}",
                    xy=(0.5, 0.955), xycoords="axes fraction", ha="center", fontsize=8,
                    color="#555555")
    ax.axhline(Z_REF, color="#e45756", ls="--", lw=1.1, label=f"$z_{{ref}}$ = {Z_REF:.3f}")
    ax.axhline(Z_LP, color="#54a24b", ls=":", lw=1.1, label=f"$z_{{LP}}$ = {Z_LP:.3f}")
    ax.set_xlabel("evaluations")
    ax.set_ylabel("best feasible cost so far")
    ax.set_title("F1: best-so-far against evaluations, 10 seeds, budget 100 000")
    ax.set_ylim(980, 1200)
    ax.grid(True, ls=":", lw=0.4, alpha=0.6)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "F1_bestsofar.png"), dpi=160)
    print("wrote figures/F1_bestsofar.png")

    out = {
        "z_lp": Z_LP, "z_ref": Z_REF, "budget": BUDGET,
        "table1_ga": table1, "table1_random": table1_rnd, "table2": table2,
        "per_seed": [{"seed": r["seed"], "cost": r["cost"], "feasible": r["feasible"],
                      "seconds": r["seconds"],
                      "gap_ref": gaps(r["cost"])[0], "gap_lp": gaps(r["cost"])[1],
                      "distinct_first": diversity[r["seed"]][0],
                      "distinct_last": diversity[r["seed"]][-1],
                      "feasible_offspring_share": offspring[r["seed"]][0] / max(1, offspring[r["seed"]][1])}
                     for r in sorted(ga_rows, key=lambda r: r["seed"])],
        "flat_fraction_of_budget": flat,
        "drop_in_last_tenth": last_tenth_drop,
    }
    with open(os.path.join(RESULTS, "Ikhsanova_analysis.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print("wrote results/Ikhsanova_analysis.json")

    print("\nTable 1")
    print(f"{'':>14} {'best':>10} {'median':>10} {'mean':>10} {'worst':>10} {'sd':>8} "
          f"{'feas':>6} {'s/run':>7} {'gapref%':>9} {'gapLP%':>8}")
    for name, t in (("GA, penalty", table1), ("random search", table1_rnd)):
        if t:
            print(f"{name:>14} {t['best']:>10.3f} {t['median']:>10.3f} {t['mean']:>10.3f} "
                  f"{t['worst']:>10.3f} {t['sd']:>8.3f} {t['feasible']:>4}/{t['runs']} "
                  f"{t['seconds']:>7.1f} {t['gap_ref_mean']:>9.3f} {t['gap_lp_mean']:>8.3f}")


if __name__ == "__main__":
    main()
