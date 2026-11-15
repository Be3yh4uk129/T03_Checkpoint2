"""Rerun the 10 final runs of both Checkpoint 4 methods to keep their solutions and best-so-far traces.

Same seeds (0-9), budget (100000) and parameters as Checkpoint 4: GA with its defaults (rho = 1),
SA with the tuned T0, Tend of results/Amanbay_chosen_params.json. Every cost is asserted to be equal
to the Checkpoint 4 results file, so nothing new is measured here; the solutions are needed by Block 1
and the traces by Figure F1.

    python final/src/rerun_cp4.py          # about 5 minutes, almost all of it the GA
Writes results/final_cp4_reruns.json.
"""
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS, ROOT, SA_PARAMS, own_instance   # noqa: E402

sys.path.insert(0, ROOT)
from src.meta import ga, sa                                 # noqa: E402


def read_costs(name, configuration):
    return {int(r["seed"]): float(r["cost"]) for r in csv.DictReader(open(os.path.join(RESULTS, name)))
            if r["configuration"] == configuration}


def main():
    d = own_instance()
    out = []
    for label, module, params, file in (("GA, penalty", ga, None, "Ikhsanova_runs.csv"),
                                        ("SA, transfer+swap", sa, SA_PARAMS, "Amanbay_runs.csv")):
        ref = read_costs(file, label)
        for seed in range(10):
            r = module.solve(d, 1, seed, 100000, params=params)
            assert abs(r["cost"] - ref[seed]) < 1e-6, (label, seed, r["cost"], ref[seed])
            c, a, _ = r["solution"]
            out.append(dict(method=label, seed=seed, cost=r["cost"], evals=r["evals"],
                            centers=[int(x) for x in c], assign=[int(x) for x in a],
                            trace=[[int(e), float(b)] for e, b in r["trace"]]))
            print(label, seed, r["cost"])
    json.dump(out, open(os.path.join(RESULTS, "final_cp4_reruns.json"), "w"))


if __name__ == "__main__":
    main()
