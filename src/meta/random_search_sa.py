"""Random-search baseline for the SA pair (role SA-T, Zhamilya Kaniyeva).

Each draw is an independent random solution: V distinct centers chosen uniformly, every
center serves itself, every other point goes to a uniformly random center. No neighbourhood,
no memory between draws. Each draw costs one evaluation; the best FEASIBLE draw is kept.
The reported cost and feasibility come from the checker evaluate_A.

Same interface as solve of sa.py: solve(d, variant, seed, budget, params=None, start=None).
"""
import importlib.util
import inspect
import math
import os
import random

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_GEN_PATH = os.path.join(_ROOT, "checkpoint4_generator_PUBLIC.py")
_GEN = None
TRACE_EVERY = 1000


def _generator():
    global _GEN
    if _GEN is None:
        spec = importlib.util.spec_from_file_location("checkpoint4_generator_PUBLIC", _GEN_PATH)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _GEN = mod
    return _GEN


def _check(d, solution):
    """evaluate_A on (centers, assign, vehicle) -> (cost, feasible). Adapt here if needed."""
    ev = _generator().evaluate_A
    centers, assign, vehicle = solution
    n_args = len(inspect.signature(ev).parameters)
    # The public Checkpoint 4 generator API is evaluate_A(d, variant, centers, assign, vehicle).
    # This solver is restricted to Storyline A, variant 1, just like sa.py.
    if n_args >= 5:
        out = ev(d, 1, centers, assign, vehicle)
    elif n_args == 4:
        out = ev(d, 1, centers, assign)
    elif n_args == 3:
        out = ev(d, centers, assign)
    else:
        out = ev(d, (centers, assign, vehicle))
    if isinstance(out, dict):
        problems = out.get("problems", out.get("violations", []))
        feasible = bool(out["feasible"]) if "feasible" in out else not problems
        return float(out["cost"]), feasible and not problems
    if len(out) == 2:
        return float(out[0]), not out[1]
    return float(out[0]), bool(out[1]) and not out[2]


def solve(d, variant, seed, budget, params=None, start=None):
    if variant != 1:
        raise NotImplementedError("storyline A, variant 1 only")
    rng = random.Random(seed)
    N, V, Cap = int(d["N"]), int(d["V"]), float(d["Cap"])
    q = [float(x) for x in d["demand"]]
    dist = [[float(x) for x in row] for row in d["dist"]]
    D = [float(x) for x in d["D"]]

    best_cost, best_sol, n_feasible = math.inf, None, 0
    first_sol = None
    trace = []
    for e in range(1, int(budget) + 1):
        centers = rng.sample(range(N), V)
        is_c = [False] * N
        for k in centers:
            is_c[k] = True
        assign = list(range(N))
        load = {k: q[k] for k in centers}
        cost = 2.0 * sum(D[k] for k in centers)
        for i in range(N):
            if not is_c[i]:
                k = centers[rng.randrange(V)]
                assign[i] = k
                load[k] += q[i]
                cost += dist[i][k]
        if first_sol is None:
            first_sol = (sorted(centers), assign, None)
        if max(load.values()) <= Cap:
            n_feasible += 1
            if cost < best_cost:
                best_cost, best_sol = cost, (sorted(centers), assign, None)
        if e % TRACE_EVERY == 0 or e == 1:
            trace.append((e, best_cost))
    if trace[-1][0] != budget:
        trace.append((int(budget), best_cost))

    sol = best_sol if best_sol is not None else first_sol
    cost, feasible = _check(d, sol)
    return {"solution": sol, "cost": cost, "evals": int(budget), "feasible": feasible,
            "trace": trace, "feasible_draws": n_feasible}
