"""Analysis of the 10 final SA runs (role SA-I, Amir Amanbay).

    PYTHONPATH=. python experiments/sa_analysis.py

Reads results/Amanbay_runs.csv (written by experiments/run_final.py) and
results/Amanbay_chosen_params.json. Every seed is rerun with the same parameters (the solver
is deterministic per seed) to get the traces and the acceptance statistics, and the rerun cost
is asserted to equal the cost in the CSV, so every number below comes from the code.

Writes
    results/Amanbay_table1.csv       Table 1: best, median, mean, worst, std, feasible, time, gaps
    results/Amanbay_table2.json      Table 2 for the worst feasible seed
    results/Amanbay_acceptance.csv   acceptance share of worsening moves per temperature bucket
    results/Amanbay_analysis.json    everything else used in the six questions
    figures/F1_sa_bestsofar.png      Figure F1 (mean of 10 runs, min-max band, z_LP, z_ref)
    figures/F4_sa_acceptance.png     the required item: acceptance share against temperature
z_ref and z_LP are used for reporting only.
"""
import csv
import json
import os
import statistics
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.meta import sa  # noqa: E402

Z_REF = 997.603759
Z_LP = 994.048826


def gap_ref(z):
    return 100.0 * (z - Z_REF) / Z_REF


def gap_lp(z):
    return 100.0 * (z - Z_LP) / z


def read_rows(path, configuration=None):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if configuration is not None:
        rows = [r for r in rows if r["configuration"] == configuration]
    return rows


def is_true(x):
    return str(x).strip().lower() in ("true", "1", "yes")


def stats_block(costs, times, n_total):
    s = sorted(costs)
    out = {"feasible_of": "%d/%d" % (len(costs), n_total),
           "best": s[0], "median": statistics.median(s), "mean": statistics.mean(s),
           "worst": s[-1], "std": statistics.stdev(s) if len(s) > 1 else 0.0,
           "mean_seconds": statistics.mean(times)}
    for k in ("best", "median", "mean", "worst"):
        out["gap_ref_" + k] = gap_ref(out[k])
        out["gap_LP_" + k] = gap_lp(out[k])
    return out


def main():
    res_dir, fig_dir = os.path.join(ROOT, "results"), os.path.join(ROOT, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    with open(os.path.join(res_dir, "Amanbay_chosen_params.json")) as f:
        chosen = json.load(f)
    params = chosen["params"]
    rows = [r for r in read_rows(os.path.join(res_dir, "Amanbay_runs.csv"))
            if r["configuration"] != "random"]
    rows.sort(key=lambda r: int(r["seed"]))
    d = sa._generator().cp34_instance(3, "A", 1)

    runs = {}
    for r in rows:
        seed, budget = int(r["seed"]), int(r["budget"])
        out = sa.solve(d, 1, seed, budget, params=params)
        assert abs(out["cost"] - float(r["cost"])) < 1e-6, ("rerun differs", seed)
        assert out["feasible"] == is_true(r["feasible"])
        runs[seed] = out
        print("seed %d  cost %.6f  feasible %s  accept first 5%% %.4f  last 5%% %.4f"
              % (seed, out["cost"], out["feasible"], out["accept_first5"] or 0, out["accept_last5"] or 0))

    # Table 1
    feas_costs = [float(r["cost"]) for r in rows if is_true(r["feasible"])]
    times = [float(r["seconds"]) for r in rows]
    t1 = stats_block(feas_costs, times, len(rows))
    t1["configuration"] = "SA, transfer+swap"
    t1["params"] = json.dumps(params)

    # random-search baseline of the pair (Zhamilya's file)
    base = None
    base_path = os.path.join(res_dir, "Kaniyeva_tuning.csv")
    if os.path.exists(base_path):
        br = read_rows(base_path, "random")
        bc = [float(r["cost"]) for r in br if is_true(r["feasible"])]
        if bc:
            base = stats_block(bc, [float(r["seconds"]) for r in br], len(br))
            base["configuration"] = "random"
    with open(os.path.join(res_dir, "Amanbay_table1.csv"), "w", newline="") as f:
        keys = ["configuration", "feasible_of", "best", "median", "mean", "worst", "std",
                "mean_seconds", "gap_ref_best", "gap_ref_median", "gap_ref_mean", "gap_ref_worst",
                "gap_LP_best", "gap_LP_median", "gap_LP_mean", "gap_LP_worst"]
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerow(t1)
        if base:
            w.writerow(base)

    # Table 2: the worst feasible seed
    feas_seeds = [s for s, o in runs.items() if o["feasible"]]
    t2 = None
    if feas_seeds:
        ws = max(feas_seeds, key=lambda s: runs[s]["cost"])
        o = runs[ws]
        n_tr, n_sw = sa.count_improving(d, o["solution"])
        loads = {}
        for i, k in enumerate(o["solution"][1]):
            loads[k] = loads.get(k, 0) + int(d["demand"][i])
        t2 = {"seed": ws, "cost": o["cost"], "improving_transfer": n_tr, "improving_swap": n_sw,
              "accept_share_first5": o["accept_first5"], "accept_share_last5": o["accept_last5"],
              "worsening_first5": o["worsening_first5"], "worsening_last5": o["worsening_last5"],
              "centers": o["solution"][0], "loads": {str(k): v for k, v in sorted(loads.items())},
              "max_load": max(loads.values()), "Cap": int(d["Cap"]),
              "last_improvement_at": next(e for e, c in o["trace"] if c == o["trace"][-1][1]),
              "params": params}
        with open(os.path.join(res_dir, "Amanbay_table2.json"), "w") as f:
            json.dump(t2, f, indent=2)

    # acceptance against temperature, all runs
    with open(os.path.join(res_dir, "Amanbay_acceptance.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "bucket", "T_mean", "proposals", "worsening", "accepted", "share"])
        for s, o in runs.items():
            for b in o["acceptance"]:
                w.writerow([s, b["bucket"], b["T_mean"], b["proposals"], b["worsening"],
                            b["accepted"], b["share"]])

    analysis = {"params": params, "chosen": chosen, "table1": t1, "random": base, "table2": t2,
                "per_seed": {s: {"cost": o["cost"], "feasible": o["feasible"],
                                 "start_cost": o["start_cost"],
                                 "accept_first5": o["accept_first5"], "accept_last5": o["accept_last5"],
                                 "accepted_share_all": o["accepted_share_all"],
                                 "last_improvement_at": next(e for e, c in o["trace"]
                                                             if c == o["trace"][-1][1]),
                                 "cost_at_10k": dict(o["trace"]).get(10000),
                                 "cost_at_50k": dict(o["trace"]).get(50000)}
                             for s, o in runs.items()},
                "p_swap": sa.P_SWAP, "p_recenter": sa.P_RECENTER}
    with open(os.path.join(res_dir, "Amanbay_analysis.json"), "w") as f:
        json.dump(analysis, f, indent=2, default=float)

    # figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = sorted({e for o in runs.values() for e, _ in o["trace"]})
    M = []
    for o in runs.values():
        ev = [e for e, _ in o["trace"]]
        cv = [c for _, c in o["trace"]]
        idx = np.searchsorted(ev, grid, side="right") - 1
        M.append([cv[i] for i in idx])
    M = np.array(M, dtype=float)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.fill_between(grid, M.min(axis=0), M.max(axis=0), color="C0", alpha=0.25, label="min-max of 10 runs")
    ax.plot(grid, M.mean(axis=0), color="C0", lw=1.6, label="mean of 10 runs")
    ax.axhline(Z_REF, color="C2", ls="--", lw=1, label="z_ref = %.2f" % Z_REF)
    ax.axhline(Z_LP, color="C3", ls=":", lw=1, label="z_LP = %.2f" % Z_LP)
    ax.set_xlabel("evaluations")
    ax.set_ylabel("best feasible cost so far")
    ax.set_ylim(Z_LP - 5, min(M[:, 0].max(), Z_REF + 120))
    ax.set_title("F1: SA, transfer+swap, seeds 0-9, budget %d" % int(rows[0]["budget"]))
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "F1_sa_bestsofar.png"), dpi=150)
    plt.close(fig)

    def pooled_curve(run_dict):
        nb = len(next(iter(run_dict.values()))["acceptance"])
        Tm, sh = [], []
        for i in range(nb):
            wsum = sum(o["acceptance"][i]["worsening"] for o in run_dict.values())
            asum = sum(o["acceptance"][i]["accepted"] for o in run_dict.values())
            Tm.append(next(iter(run_dict.values()))["acceptance"][i]["T_mean"])
            sh.append(asum / wsum if wsum else np.nan)
        return Tm, sh

    # contrast: the rule-based temperatures on the tuning seeds 100-104 (never seeds 0-9)
    rule_set = chosen.get("sets", {}).get("rule")
    rule_runs = {}
    if rule_set and chosen["chosen"] != "rule":
        for s in (100, 101, 102, 103, 104):
            rule_runs[s] = sa.solve(d, 1, s, int(rows[0]["budget"]), params=rule_set)
        with open(os.path.join(res_dir, "Amanbay_acceptance_rule.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["seed", "bucket", "T_mean", "proposals", "worsening", "accepted", "share"])
            for s, o in rule_runs.items():
                for b in o["acceptance"]:
                    w.writerow([s, b["bucket"], b["T_mean"], b["proposals"], b["worsening"],
                                b["accepted"], b["share"]])
        analysis["rule_on_tuning_seeds"] = {
            str(s): {"cost": o["cost"], "accept_first5": o["accept_first5"],
                     "accept_last5": o["accept_last5"]} for s, o in rule_runs.items()}
        with open(os.path.join(res_dir, "Amanbay_analysis.json"), "w") as f:
            json.dump(analysis, f, indent=2, default=float)

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for s, o in runs.items():
        T = [b["T_mean"] for b in o["acceptance"] if b["share"] is not None]
        A = [b["share"] for b in o["acceptance"] if b["share"] is not None]
        ax.plot(T, A, color="0.75", lw=0.7)
    Tm, pooled = pooled_curve(runs)
    ax.plot(Tm, pooled, color="C1", lw=1.8,
            label="final runs, seeds 0-9 (T0=%.2f, Tend=%.2f)" % (params["T0"], params["Tend"]))
    ax.plot([], [], color="0.75", lw=0.7, label="single final runs")
    if rule_runs:
        Tr, pr = pooled_curve(rule_runs)
        ax.plot(Tr, pr, color="C0", lw=1.4, ls="--",
                label="rule-based, seeds 100-104 (T0=%.1f, Tend=%.2f)"
                % (rule_set["T0"], rule_set["Tend"]))
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("temperature T (log scale, cooling runs left to right)")
    ax.set_ylabel("accepted / worsening proposals")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Acceptance share of worsening moves against temperature")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "F4_sa_acceptance.png"), dpi=150)
    plt.close(fig)

    print("Table 1:", json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in t1.items()}))
    if base:
        print("random :", json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in base.items()}))
    print("Table 2:", json.dumps({k: v for k, v in (t2 or {}).items() if k not in ("loads",)}))


if __name__ == "__main__":
    main()