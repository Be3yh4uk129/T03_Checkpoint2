"""Checkpoint 4, role GA-A: the random-search baseline.

Author: Ikhsanova Aziza (GA-A).

The baseline answers question 6, "is the method better than random search?", so it must
spend exactly the same budget on exactly the same seeds and be measured by the same
checker. It draws independent random individuals with the same `random_individual` of
ga_problem.py that seeds the GA population, keeps the best feasible one, and does nothing
else: no neighbourhood, no selection, no memory between draws.
"""
import numpy as np

from . import ga_problem as gp
from .interface import check

DEFAULTS = {"rho": gp.RHO_DEFAULT, "trace_every": 1000}


def solve(d, variant, seed, budget, params=None, start=None):
    """Draw `budget` random individuals and report the best feasible one."""
    p = dict(DEFAULTS)
    if params:
        p.update({k: v for k, v in params.items() if v is not None})
    rho = float(p["rho"])
    trace_every = int(p["trace_every"])

    rng = np.random.default_rng(seed)

    evals = 0
    trace = []
    next_trace = 0
    best_feasible = None
    best_feasible_cost = float("inf")
    best_any = None
    best_any_fitness = float("inf")

    if start is not None:
        start = np.asarray(start, dtype=np.int64)

    while evals < budget:
        ind = start.copy() if (start is not None and evals == 0) else gp.random_individual(d, rng)
        f, c, _, ok = gp.evaluate(d, ind, rho)
        evals += 1
        if ok and c < best_feasible_cost:
            best_feasible_cost = c
            best_feasible = np.array(ind, copy=True)
        if f < best_any_fitness:
            best_any_fitness = f
            best_any = np.array(ind, copy=True)
        while evals >= next_trace:
            trace.append((evals, best_feasible_cost))
            next_trace += trace_every
    trace.append((evals, best_feasible_cost))

    reported = best_feasible if best_feasible is not None else best_any
    solution = gp.to_solution(d, reported)
    cost, problems = check(d, variant, solution)

    return {
        "solution": solution,
        "cost": float(cost),
        "evals": int(evals),
        "feasible": len(problems) == 0,
        "trace": trace,
        "params": {"rho": rho},
    }
