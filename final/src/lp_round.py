"""Start from the rounded LP solution (Block 3).

Rounding rule of the team:
  1. solve the LP relaxation of the Checkpoint 3 model (HiGHS, src/lp/solve.py);
  2. open the V points with the largest y_k (ties: the smaller index);
  3. every open center serves itself;
  4. the other points, largest demand first, go to the open center with the largest LP value x_ik
     that still has room; if all x_ik of the point are zero on centers with room, to the nearest
     center with room.
The result is checked with evaluate_A. It is written in the format of run_scaling.py --start-file.

    python final/src/lp_round.py            # N = 60, 100, 150
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS, ROOT, SIZES, gen, scaled   # noqa: E402

sys.path.insert(0, ROOT)
from src.lp.solve import solve_highs                   # noqa: E402
from src.model.matrices import build_matrix_form       # noqa: E402


def round_lp(d, X, y):
    N, V, Cap = int(d["N"]), int(d["V"]), float(d["Cap"])
    q = np.asarray(d["demand"], dtype=float)
    centers = sorted(np.argsort(-y, kind="stable")[:V].tolist())
    load = {c: q[c] for c in centers}
    assign = [-1] * N
    for c in centers:
        assign[c] = c
    others = [i for i in np.argsort(-q, kind="stable") if i not in load]
    for i in others:
        room = [c for c in centers if load[c] + q[i] <= Cap]
        if not room:
            room = centers                      # no room anywhere: the checker will report it
        lp = [X[i, c] for c in room]
        if max(lp) > 1e-9:
            c = room[int(np.argmax(lp))]
        else:
            c = room[int(np.argmin([d["dist"][i, k] for k in room]))]
        assign[i] = int(c)
        load[c] += q[i]
    return [int(c) for c in centers], assign


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default=",".join(str(s) for s in SIZES))
    a = ap.parse_args()
    out = os.path.join(RESULTS, "final_lp_rounding.csv")
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["N", "z_LP", "lp_seconds", "n_fractional_y", "rounded_cost", "rounded_feasible", "start_file"])
        for n in [int(s) for s in a.sizes.split(",")]:
            d = scaled(n)
            mf = build_matrix_form(d)
            t0 = time.time()
            lp = solve_highs(mf)
            sec = time.time() - t0
            X, y = mf.split_solution(lp.x)
            centers, assign = round_lp(d, X, y)
            cost, problems = gen.evaluate_A(d, 1, centers, assign)
            path = os.path.join(RESULTS, f"final_rounded_A_{n}.json")
            with open(path, "w") as fj:
                json.dump({"centers": centers, "assign": assign, "vehicle": None}, fj)
            nfy = int(np.sum((y > 1e-6) & (y < 1 - 1e-6)))
            w.writerow([n, f"{lp.z:.6f}", f"{sec:.2f}", nfy, f"{cost:.6f}", len(problems) == 0,
                        os.path.relpath(path, ROOT)])
            print(n, lp.z, nfy, cost, problems[:3])


if __name__ == "__main__":
    main()
