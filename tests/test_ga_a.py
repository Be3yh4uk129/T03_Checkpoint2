"""Role test of GA-A (Ikhsanova Aziza): the loop, the budget, elitism, the tournament.

The four properties the instructions ask this test to guard:
  same seed gives the same run; exactly the budget is spent;
  elitism keeps the best individual; the tournament never returns the worst of its entrants.
"""
import numpy as np
import pytest

import checkpoint4_generator_PUBLIC as gen
from src.meta import ga
from src.meta.ga import _tournament, solve
from src.meta.interface import check

VARIANT = 1


@pytest.fixture(scope="module")
def d():
    return gen.cp34_instance(3, "A", VARIANT)


# --------------------------------------------------------------- reproducibility

def test_same_seed_gives_the_same_run(d):
    a = solve(d, VARIANT, 4, 3000)
    b = solve(d, VARIANT, 4, 3000)
    assert a["cost"] == b["cost"]
    assert a["evals"] == b["evals"]
    assert a["feasible"] == b["feasible"]
    assert a["solution"][0] == b["solution"][0]
    assert a["solution"][1] == b["solution"][1]
    assert a["trace"] == b["trace"]


def test_different_seeds_do_not_give_the_same_run(d):
    """Guards against a solver that ignores its seed, which would make the first test vacuous."""
    costs = {solve(d, VARIANT, s, 3000)["cost"] for s in (0, 1, 2, 3)}
    assert len(costs) > 1


def test_the_global_random_state_is_not_used(d):
    """Seeding numpy globally must not change the run."""
    np.random.seed(12345)
    a = solve(d, VARIANT, 1, 2000)
    np.random.seed(999)
    b = solve(d, VARIANT, 1, 2000)
    assert a["cost"] == b["cost"]
    assert a["trace"] == b["trace"]


# --------------------------------------------------------------- the budget

@pytest.mark.parametrize("budget", [1, 7, 30, 31, 137, 1000, 5000])
def test_exactly_the_budget_is_spent(d, budget):
    """Including budgets smaller than the population and budgets that cut a generation."""
    r = solve(d, VARIANT, 0, budget)
    assert r["evals"] == budget


def test_the_elite_copy_is_not_charged(d):
    """One evaluation per offspring plus the initial population, and nothing else."""
    r = solve(d, VARIANT, 0, 5000)
    pop_size = r["params"]["pop_size"]
    assert r["offspring"] + min(pop_size, 5000) == r["evals"]


# --------------------------------------------------------------- elitism

def test_elitism_keeps_the_best_individual(d):
    """The best fitness of the population may never get worse from one generation to the next."""
    r = solve(d, VARIANT, 2, 20000)
    bf = r["best_fitness"]
    assert len(bf) > 10
    for k in range(len(bf) - 1):
        assert bf[k + 1] <= bf[k] + 1e-12, f"generation {k + 1} lost the elite: {bf[k]} -> {bf[k+1]}"


def test_the_best_so_far_trace_never_increases(d):
    r = solve(d, VARIANT, 3, 20000)
    costs = [c for _, c in r["trace"]]
    for k in range(len(costs) - 1):
        assert costs[k + 1] <= costs[k] + 1e-12
    assert r["trace"][-1][0] == r["evals"]


# --------------------------------------------------------------- the tournament

def test_the_tournament_never_returns_the_worst_of_its_entrants():
    """A second generator in lockstep reproduces the entrants the tournament drew."""
    fitness = np.array([5.0, 1.0, 9.0, 3.0, 7.0, 2.0, 8.0, 4.0, 6.0, 0.0])
    rng = np.random.default_rng(7)
    mirror = np.random.default_rng(7)
    for _ in range(1000):
        idx = _tournament(rng, fitness, 2)
        entrants = mirror.integers(0, fitness.size, size=2)
        assert fitness[idx] == fitness[entrants].min()
        assert fitness[idx] <= fitness[entrants].max()


def test_the_tournament_with_a_larger_size_still_picks_the_best_entrant():
    fitness = np.arange(20.0)
    rng = np.random.default_rng(11)
    mirror = np.random.default_rng(11)
    for _ in range(500):
        idx = _tournament(rng, fitness, 5)
        entrants = mirror.integers(0, fitness.size, size=5)
        assert fitness[idx] == fitness[entrants].min()


# --------------------------------------------------------------- honest reporting

def test_the_reported_cost_and_flag_come_from_the_checker(d):
    r = solve(d, VARIANT, 5, 10000)
    cost, problems = check(d, VARIANT, r["solution"])
    assert abs(r["cost"] - cost) < 1e-9
    assert r["feasible"] == (len(problems) == 0)


def test_solve_accepts_a_start_and_still_spends_the_budget(d):
    """`start` is unused in this checkpoint but the signature must accept it."""
    from src.meta import ga_problem as gp
    warm = gp.random_individual(d, np.random.default_rng(0))
    r = solve(d, VARIANT, 0, 3000, start=warm)
    assert r["evals"] == 3000


def test_params_override_the_defaults(d):
    r = solve(d, VARIANT, 0, 2000, params={"pop_size": 12, "rho": 25.0})
    assert r["params"]["pop_size"] == 12
    assert r["params"]["rho"] == 25.0
    assert r["evals"] == 2000
