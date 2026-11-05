"""Checkpoint 4, role GA-A: the genetic algorithm loop and solve(...).

Author: Ikhsanova Aziza (GA-A).  Configuration: GA, penalty.
The representation and the operators are ga_problem.py, authored by Abdrashit Yersultan (GA-P).

Loop
----
Generational GA with elitism of one individual:

    initial population of `pop_size` random individuals        (one evaluation each)
    repeat
        the best individual of the population is copied over   (free, it is not an offspring)
        while the new population is not full:
            two parents by tournament of size `tournament`
            crossover with probability `pc`, otherwise a copy of the first parent
            mutation of 1 or 2 components
            one evaluation of the offspring
    until exactly `budget` evaluations have been spent

Counting follows Section 3 of the instructions: every offspring is one evaluation, even
when it is an unchanged copy of a parent; the members of the initial population count;
the copy of the elite costs nothing. The last generation is cut short so that `evals`
equals `budget` exactly.

Selection is by fitness, which is cost + rho * violation, so an infeasible offspring can
survive and carry useful structure. Only feasible solutions are ever reported: the best
feasible individual seen is kept separately from the population and is what solve returns.
"""
import numpy as np

from . import ga_problem as gp
from .interface import check

DEFAULTS = {
    "pop_size": 30,        # Section 4 of the instructions
    "pc": 0.9,             # crossover probability
    "tournament": 2,       # tournament size
    "elitism": 1,          # individuals copied unchanged
    "rho": gp.RHO_DEFAULT,  # penalty weight
    "trace_every": 1000,   # entries in the trace
}


def _tournament(rng, fitness, k):
    """Index of the best of k entrants drawn with replacement."""
    idx = rng.integers(0, fitness.size, size=k)
    return int(idx[int(np.argmin(fitness[idx]))])


def solve(d, variant, seed, budget, params=None, start=None):
    """Run the GA on instance `d` and return the dictionary of the interface.

    `start` is accepted and, when given, replaces one member of the initial population.
    Nothing here reads z_ref or z_LP, and the only source of randomness is the generator
    created from `seed`.
    """
    p = dict(DEFAULTS)
    if params:
        p.update({k: v for k, v in params.items() if v is not None})
    pop_size = int(p["pop_size"])
    pc = float(p["pc"])
    k_tour = int(p["tournament"])
    n_elite = int(p["elitism"])
    rho = float(p["rho"])
    trace_every = int(p["trace_every"])

    rng = np.random.default_rng(seed)

    evals = 0
    trace = []
    next_trace = 0
    diversity = []           # distinct individuals per generation, for the GA-A item
    diversity_centers = []   # distinct center sets per generation
    fitness_spread = []      # worst minus best fitness in the population
    best_fitness = []        # best fitness of each population, for the elitism test
    n_offspring = 0          # offspring produced, for the feasible-offspring share
    n_offspring_feasible = 0

    best_feasible = None
    best_feasible_cost = float("inf")
    best_any = None
    best_any_fitness = float("inf")

    def note(ind, fit, cost, feasible):
        nonlocal best_feasible, best_feasible_cost, best_any, best_any_fitness
        if feasible and cost < best_feasible_cost:
            best_feasible_cost = cost
            best_feasible = np.array(ind, copy=True)
        if fit < best_any_fitness:
            best_any_fitness = fit
            best_any = np.array(ind, copy=True)

    def _record(population, fitness):
        """Three measures of one generation, for the diversity item of the analysis.

        `diversity` is the count the instructions ask for, distinct individuals. On this
        representation two individuals differ as soon as one point is served differently,
        so that count stays near the population size even when the search has converged;
        the other two measures are the ones that can actually fall.
        """
        diversity.append(len({tuple(int(x) for x in ind) for ind in population}))
        diversity_centers.append(len({tuple(sorted(int(c) for c in gp.centers_of(ind)))
                                      for ind in population}))
        fitness_spread.append(float(np.max(fitness) - np.min(fitness)))
        best_fitness.append(float(np.min(fitness)))

    def tick():
        nonlocal next_trace
        while evals >= next_trace:
            trace.append((evals, best_feasible_cost))
            next_trace += trace_every

    # ---- initial population -------------------------------------------------
    pop, fit = [], []
    n_init = min(pop_size, budget)
    for j in range(n_init):
        ind = np.asarray(start, dtype=np.int64).copy() if (start is not None and j == 0) \
            else gp.random_individual(d, rng)
        f, c, _, ok = gp.evaluate(d, ind, rho)
        evals += 1
        note(ind, f, c, ok)
        pop.append(ind)
        fit.append(f)
        tick()
    fit = np.array(fit, dtype=float)

    # ---- generations --------------------------------------------------------
    while evals < budget and pop:
        _record(pop, fit)
        order = np.argsort(fit)
        new_pop = [np.array(pop[int(i)], copy=True) for i in order[:n_elite]]
        new_fit = [float(fit[int(i)]) for i in order[:n_elite]]

        while len(new_pop) < pop_size and evals < budget:
            i1 = _tournament(rng, fit, k_tour)
            i2 = _tournament(rng, fit, k_tour)
            if rng.random() < pc:
                child = gp.crossover(d, pop[i1], pop[i2], rng)
            else:
                child = np.array(pop[i1], copy=True)
            child = gp.mutate(d, child, rng, n_components=int(rng.integers(1, 3)))

            f, c, _, ok = gp.evaluate(d, child, rho)
            evals += 1
            n_offspring += 1
            if ok:
                n_offspring_feasible += 1
            note(child, f, c, ok)
            new_pop.append(child)
            new_fit.append(f)
            tick()

        pop = new_pop
        fit = np.array(new_fit, dtype=float)

    # The last generation is cut short so that the budget is spent exactly, so the final
    # population is usually smaller than pop_size. Counting its distinct individuals would
    # measure the truncation, not the diversity, so only a full population is recorded.
    if pop and len(pop) == pop_size:
        _record(pop, fit)
    final_population = len(pop)
    trace.append((evals, best_feasible_cost))

    # ---- report -------------------------------------------------------------
    reported = best_feasible if best_feasible is not None else best_any
    solution = gp.to_solution(d, reported)
    cost, problems = check(d, variant, solution)

    return {
        "solution": solution,
        "cost": float(cost),
        "evals": int(evals),
        "feasible": len(problems) == 0,
        "trace": trace,
        "generations": len(diversity),
        "final_population": final_population,
        "diversity": diversity,
        "diversity_centers": diversity_centers,
        "fitness_spread": fitness_spread,
        "best_fitness": best_fitness,
        "offspring": n_offspring,
        "offspring_feasible": n_offspring_feasible,
        "params": {"pop_size": pop_size, "pc": pc, "tournament": k_tour,
                   "elitism": n_elite, "rho": rho},
    }
