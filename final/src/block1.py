"""Block 1: reference solutions under our own Checkpoint 1 objective.

Checkpoint 1 objective (T03, Section 2.2): f(x, c) = sum_j sum_i q_i x_ij ||p_i - c_j||^2, minimised over
the assignment x and free centroids c. For a fixed assignment the best c_j is the demand-weighted centroid of
the cluster (Checkpoint 1, Section 2.4), so f of a solution is evaluated with those centroids.

The script
  1. reads the solutions of the 10 final GA runs and the 10 final SA runs of Checkpoint 4
     (results/final_cp4_reruns.json, written by final/src/rerun_cp4.py);
  2. reads the optimum of the reference model from results/final_exact_35.json (final/src/exact_scaled.py);
  3. evaluates every solution under both objectives;
  4. looks for a good solution of our own Checkpoint 1 model (local search with transfer moves and exact
     centroid updates, started from every solution above and from 200 random feasible starts), and
     evaluates it under the reference objective with the best center point inside each of its clusters.

    python final/src/block1.py
Writes results/final_block1_solutions.csv and results/final_block1_summary.json.
"""
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS, gen, own_instance   # noqa: E402


# ------------------------------------------------------------------ the two objectives
def cp1_objective(d, assign):
    """f of Checkpoint 1 for the clusters given by `assign` (cluster label = serving center)."""
    P = np.asarray(d["points"], dtype=float)
    q = np.asarray(d["demand"], dtype=float)
    labels = np.asarray(assign)
    f = 0.0
    for c in np.unique(labels):
        m = labels == c
        cen = (q[m, None] * P[m]).sum(axis=0) / q[m].sum()
        f += float((q[m] * ((P[m] - cen) ** 2).sum(axis=1)).sum())
    return f


def ref_objective(d, centers, assign):
    cost, problems = gen.evaluate_A(d, 1, centers, assign)
    return cost, len(problems) == 0


def best_center_points(d, clusters):
    """Reference value of a clustering: inside every cluster the member k minimising sum_i d_ik + 2 D_k."""
    centers, assign = [], [0] * int(d["N"])
    for members in clusters:
        members = list(members)
        cost = [sum(d["dist"][i, k] for i in members) + 2 * d["D"][k] for k in members]
        k = members[int(np.argmin(cost))]
        centers.append(int(k))
        for i in members:
            assign[i] = int(k)
    return centers, assign


# ------------------------------------------------------------------ local search for the CP1 model
def cp1_local_search(d, labels, cap):
    """Transfer moves between clusters with exact incremental centroid updates, first improvement."""
    P = np.asarray(d["points"], dtype=float)
    q = np.asarray(d["demand"], dtype=float)
    labels = np.asarray(labels).copy()
    K = int(labels.max()) + 1
    W = np.array([q[labels == j].sum() for j in range(K)])
    S = np.array([(q[labels == j, None] * P[labels == j]).sum(axis=0) for j in range(K)])
    improved = True
    while improved:
        improved = False
        for i in range(len(q)):
            a = labels[i]
            if W[a] - q[i] <= 0:
                continue                       # keep every vehicle used (more clusters never hurt f)
            ca = S[a] / W[a]
            gain = q[i] * W[a] / (W[a] - q[i]) * float(((P[i] - ca) ** 2).sum())
            best, best_b = 1e-12, -1
            for b in range(K):
                if b == a or W[b] + q[i] > cap:
                    continue
                cb = S[b] / W[b]
                cost = q[i] * W[b] / (W[b] + q[i]) * float(((P[i] - cb) ** 2).sum())
                if gain - cost > best:
                    best, best_b = gain - cost, b
            if best_b >= 0:
                b = best_b
                W[a] -= q[i]; S[a] -= q[i] * P[i]
                W[b] += q[i]; S[b] += q[i] * P[i]
                labels[i] = b
                improved = True
    return labels


def random_labels(d, rng):
    """Random feasible start: V seeds, then every point (random order) to the nearest seed cluster with room."""
    N, V, cap = int(d["N"]), int(d["V"]), float(d["Cap"])
    q = np.asarray(d["demand"], dtype=float)
    seeds = rng.choice(N, size=V, replace=False)
    labels = -np.ones(N, dtype=int)
    load = np.zeros(V)
    for j, s in enumerate(seeds):
        labels[s] = j; load[j] = q[s]
    for i in rng.permutation(N):
        if labels[i] >= 0:
            continue
        order = np.argsort([d["dist"][i, s] for s in seeds])
        for j in order:
            if load[j] + q[i] <= cap:
                labels[i] = j; load[j] += q[i]; break
    if (labels < 0).any():
        return None
    return labels


def to_labels(assign):
    u = {c: j for j, c in enumerate(sorted(set(assign)))}
    return np.array([u[c] for c in assign])


# ------------------------------------------------------------------ main
def main():
    d = own_instance()
    rows = []

    ex = json.load(open(os.path.join(RESULTS, "final_exact_35.json")))
    rows.append(("exact (reference optimum)", "", ex["centers"], ex["assign"]))

    for r in json.load(open(os.path.join(RESULTS, "final_cp4_reruns.json"))):
        rows.append((r["method"], r["seed"], r["centers"], r["assign"]))

    out = []
    ex_key = (tuple(sorted(ex["centers"])), tuple(ex["assign"]))
    for method, seed, c, a in rows:
        z, feas = ref_objective(d, c, a)
        out.append(dict(method=method, seed=seed, ref_cost=z, cp1_cost=cp1_objective(d, a), feasible=feas,
                        same_as_optimum=(tuple(sorted(c)), tuple(a)) == ex_key,
                        centers=" ".join(str(x) for x in sorted(c))))

    # our own model: local search from every solution above and from random starts
    cap = float(d["Cap"])
    rng = np.random.default_rng(2026)
    starts = [to_labels(a) for _, _, _, a in rows]
    while len(starts) < len(rows) + 200:
        lab = random_labels(d, rng)
        if lab is not None:
            starts.append(lab)
    best = None
    for lab in starts:
        lab = cp1_local_search(d, lab, cap)
        f = cp1_objective(d, lab)
        if best is None or f < best[0] - 1e-9:
            best = (f, lab)
    f1, lab = best
    clusters = [np.where(lab == j)[0] for j in np.unique(lab)]
    c1, a1 = best_center_points(d, clusters)
    z1, feas1 = ref_objective(d, c1, a1)
    out.append(dict(method="own CP1 model (local search)", seed="", ref_cost=z1, cp1_cost=cp1_objective(d, a1),
                    feasible=feas1, same_as_optimum=False, centers=" ".join(str(x) for x in sorted(c1))))

    with open(os.path.join(RESULTS, "final_block1_solutions.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader()
        for r in out:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()})

    # summary: ranks under both objectives over the 21 reference-model solutions
    ref = np.array([r["ref_cost"] for r in out[:-1]])
    cp1 = np.array([r["cp1_cost"] for r in out[:-1]])
    rr, rc = ref.argsort().argsort(), cp1.argsort().argsort()
    spearman = float(np.corrcoef(rr, rc)[0, 1])
    best_cp1 = int(np.argmin(cp1))
    distinct = [r for r in out[1:-1] if not r["same_as_optimum"]]
    best_other = min(distinct, key=lambda r: r["ref_cost"])
    summary = dict(
        optimum=dict(ref=out[0]["ref_cost"], cp1=out[0]["cp1_cost"]),
        approx_runs_equal_to_optimum=[f'{r["method"]} seed {r["seed"]}' for r in out if r["same_as_optimum"] and r["seed"] != ""],
        best_other_approx=dict(label=f'{best_other["method"]} seed {best_other["seed"]}', ref=best_other["ref_cost"], cp1=best_other["cp1_cost"]),
        lowest_cp1_among_reference_solutions=dict(label=f'{out[best_cp1]["method"]} seed {out[best_cp1]["seed"]}',
                                                  ref=out[best_cp1]["ref_cost"], cp1=out[best_cp1]["cp1_cost"]),
        spearman_rank_ref_vs_cp1_21_solutions=spearman,
        own_model=dict(cp1=f1, ref=z1, feasible=feas1, n_starts=len(starts),
                       gap_ref_percent=100 * (z1 - out[0]["ref_cost"]) / out[0]["ref_cost"]),
    )
    # why the order changes: the parts of the reference cost, and whether two solutions have the same clusters
    def parts(c, a):
        assign_part = sum(d["dist"][i, a[i]] for i in range(len(a)))
        return dict(assignment=float(assign_part), warehouse_trips=float(2 * sum(d["D"][k] for k in c)))

    def partition(a):
        return sorted(tuple(sorted(i for i in range(len(a)) if a[i] == c)) for c in set(a))

    sa5 = next(r for r in rows if r[0] == "SA, transfer+swap" and r[1] == 5)
    summary["same_clusters_optimum_and_sa5"] = partition(ex["assign"]) == partition(sa5[3])
    summary["centers_only_in_optimum_vs_sa5"] = [sorted(set(ex["centers"]) - set(sa5[2])), sorted(set(sa5[2]) - set(ex["centers"]))]
    summary["parts_optimum"] = parts(ex["centers"], ex["assign"])
    summary["parts_own_model"] = parts(c1, a1)
    summary["own_model_cluster_loads"] = sorted(int(sum(d["demand"][i] for i in cl)) for cl in clusters)
    summary["optimum_cluster_loads"] = sorted(int(sum(d["demand"][i] for i in range(len(ex["assign"])) if ex["assign"][i] == c)) for c in ex["centers"])
    json.dump(summary, open(os.path.join(RESULTS, "final_block1_summary.json"), "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
