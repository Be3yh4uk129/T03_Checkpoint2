"""Rule-based T0, Tend and the choice between rule and tuned values (role SA-I, Amir Amanbay).

    PYTHONPATH=. python experiments/sa_params.py
    PYTHONPATH=. python experiments/sa_params.py --tuned '{"T0": 6.89, "Tend": 3.01}'

1. The rule of Section 4: a random walk of 3000 proposals that accepts everything (tuning
   seed 100); Delta = median of the positive deltas, Delta_min = 5th percentile,
   T0 = 4.5 Delta, Tend = 0.22 Delta_min.
2. Both sets, the rule-based one and the Nelder-Mead one of Zhamilya (read from
   results/Kaniyeva_tuned_params.json unless --tuned is given), run on seeds 100..104 with the
   full budget 100000. The set with the lower mean there is used for the final runs on 0..9.
   Seeds 0..9 are never looked at here.

Writes results/Amanbay_rule_params.json, results/Amanbay_choice.csv and
results/Amanbay_chosen_params.json, and prints the --params argument for run_final.py.
"""
import argparse
import csv
import json
import os
import statistics
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.meta import sa  # noqa: E402

TUNING_SEEDS = [100, 101, 102, 103, 104]
COLUMNS = ["team", "role", "configuration", "seed", "budget", "cost", "evals", "feasible",
           "seconds", "params"]


def load_tuned(arg):
    """Zhamilya's tuned T0, Tend: from --tuned, else results/Kaniyeva_tuned_params.json,
    else the rows 'SA, transfer+swap (tuned)' of results/Kaniyeva_tuning.csv."""
    if arg:
        return json.loads(arg)
    path_json = os.path.join(ROOT, "results", "Kaniyeva_tuned_params.json")
    if os.path.exists(path_json):
        with open(path_json) as f:
            return json.load(f)["tuned"]
    path_csv = os.path.join(ROOT, "results", "Kaniyeva_tuning.csv")
    if os.path.exists(path_csv):
        with open(path_csv, newline="") as f:
            for row in csv.DictReader(f):
                if row["configuration"].strip().endswith("(tuned)"):
                    p = json.loads(row["params"])
                    return {"T0": float(p["T0"]), "Tend": float(p["Tend"])}
    raise SystemExit("tuned T0, Tend not found: pass --tuned '{\"T0\": ..., \"Tend\": ...}'")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tuned", default=None, help='JSON like {"T0": ..., "Tend": ...}')
    ap.add_argument("--budget", type=int, default=100000)
    args = ap.parse_args()

    d = sa._generator().cp34_instance(3, "A", 1)
    rule = sa.rule_params(d)
    with open(os.path.join(ROOT, "results", "Amanbay_rule_params.json"), "w") as f:
        json.dump(rule, f, indent=2)
    print("rule: Delta = %.6f, Delta_min = %.6f -> T0 = %.6f, Tend = %.6f"
          % (rule["Delta"], rule["Delta_min"], rule["T0"], rule["Tend"]))

    tuned = load_tuned(args.tuned)
    print("tuned (Nelder-Mead): T0 = %.6f, Tend = %.6f" % (float(tuned["T0"]), float(tuned["Tend"])))
    sets = {"rule": {"T0": rule["T0"], "Tend": rule["Tend"]},
            "tuned": {"T0": float(tuned["T0"]), "Tend": float(tuned["Tend"])}}

    means = {}
    with open(os.path.join(ROOT, "results", "Amanbay_choice.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(COLUMNS)
        for name, prm in sets.items():
            vals = []
            for s in TUNING_SEEDS:
                t = time.perf_counter()
                r = sa.solve(d, 1, s, args.budget, params=prm)
                sec = time.perf_counter() - t
                w.writerow(["T03", "SA-I", "SA, transfer+swap (%s)" % name, s, args.budget,
                            repr(r["cost"]), r["evals"], r["feasible"], round(sec, 4),
                            json.dumps(prm)])
                vals.append(r["cost"] if r["feasible"] else 10.0 * r["start_cost"])
            means[name] = statistics.mean(vals)
            print("%-5s T0 = %.6f Tend = %.6f  mean on seeds 100-104 = %.6f"
                  % (name, prm["T0"], prm["Tend"], means[name]))

    chosen = min(means, key=means.get)
    out = {"chosen": chosen, "params": sets[chosen], "means_seeds_100_104": means,
           "budget": args.budget, "sets": sets}
    with open(os.path.join(ROOT, "results", "Amanbay_chosen_params.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("chosen on seeds 100-104:", chosen)
    print("--params '%s'" % json.dumps(sets[chosen]))


if __name__ == "__main__":
    main()