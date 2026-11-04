"""Role test of GA-P (Abdrashit Yersultan): representation, evaluation and the operators.

The three properties the instructions ask this test to guard, for the penalty configuration:
  evaluate agrees with the checker on 1000 random individuals;
  crossover and mutation return valid individuals on 1000 random pairs;
  the penalty formula is verified on a hand example.
"""
import numpy as np
import pytest

import checkpoint4_generator_PUBLIC as gen
from src.meta import ga_problem as gp

VARIANT = 1
N_TRIALS = 1000


@pytest.fixture(scope="module")
def d():
    return gen.cp34_instance(3, "A", VARIANT)


# --------------------------------------------------------------- representation

def test_random_individuals_are_structurally_valid(d):
    rng = np.random.default_rng(0)
    for _ in range(N_TRIALS):
        a = gp.random_individual(d, rng)
        assert gp.is_valid(d, a)
        centers = gp.centers_of(a)
        assert centers.size == d["V"]
        assert all(a[c] == c for c in centers)


def test_to_solution_passes_the_checker_structurally(d):
    """A random individual may be overloaded, but it is never structurally broken."""
    rng = np.random.default_rng(1)
    for _ in range(200):
        a = gp.random_individual(d, rng)
        centers, assign, vehicle = gp.to_solution(d, a)
        _, problems = gen.evaluate_A(d, VARIANT, centers, assign, vehicle)
        for p in problems:
            assert "load" in p, f"structural problem reported: {p}"


# --------------------------------------------------------------- evaluation

def test_evaluate_agrees_with_the_checker_on_1000_individuals(d):
    rng = np.random.default_rng(2)
    worst = 0.0
    for _ in range(N_TRIALS):
        a = gp.random_individual(d, rng)
        _, cost, _, _ = gp.evaluate(d, a, gp.RHO_DEFAULT)
        centers, assign, vehicle = gp.to_solution(d, a)
        ref, _ = gen.evaluate_A(d, VARIANT, centers, assign, vehicle)
        worst = max(worst, abs(cost - ref))
    assert worst < 1e-9, f"largest disagreement with the checker: {worst:.3e}"


def test_the_feasible_flag_agrees_with_the_checker(d):
    rng = np.random.default_rng(3)
    for _ in range(300):
        a = gp.random_individual(d, rng)
        _, _, _, feasible = gp.evaluate(d, a, gp.RHO_DEFAULT)
        centers, assign, vehicle = gp.to_solution(d, a)
        _, problems = gen.evaluate_A(d, VARIANT, centers, assign, vehicle)
        assert feasible == (len(problems) == 0)


# --------------------------------------------------------------- the penalty, by hand

def test_the_penalty_formula_on_a_hand_example(d):
    """A case whose loads can be written down by hand.

    Points 0 to 12 are the centers, and every one of the remaining points 13 to 34 is
    served by center 0. Then center 0 carries its own demand plus the demand of all 22
    outside points, while centers 1 to 12 carry only their own demand, which is at most
    29 and so never exceeds Cap = 105. The whole violation therefore sits on center 0
    and equals load(0) - Cap, and the fitness must be exactly cost + rho * violation.
    """
    q = d["demand"]
    cap = float(d["Cap"])
    centers = np.arange(13)
    a = np.zeros(d["N"], dtype=np.int64)      # send everybody to center 0
    a[centers] = centers                      # but the centers serve themselves

    load0_by_hand = float(q[0]) + float(sum(q[i] for i in range(13, d["N"])))
    assert abs(float(gp.load_vector(d, a)[0]) - load0_by_hand) < 1e-9
    for c in range(1, 13):
        assert float(gp.load_vector(d, a)[c]) == float(q[c]) <= cap

    expected_violation = load0_by_hand - cap
    assert expected_violation > 0, "the hand example must actually overload center 0"

    rho = 7.5
    fitness, cost, violation, feasible = gp.evaluate(d, a, rho)
    assert abs(violation - expected_violation) < 1e-9
    assert not feasible
    assert abs(fitness - (cost + rho * expected_violation)) < 1e-9

    # the same overload is what the checker complains about, and about nothing else
    _, problems = gen.evaluate_A(d, VARIANT, *gp.to_solution(d, a)[:2], None)
    assert len(problems) == 1 and f"load {load0_by_hand:.0f}" in problems[0], problems


def test_the_fitness_arithmetic_of_the_guide_example():
    """The shape of the formula, on the three numbers of the CP4 Guide, Section 2.

    There an offspring of cost -37 has overload 3, so with rho = 12 the fitness is
    -37 + 12 * 3 = -1 and with rho = 2 it is -37 + 2 * 3 = -31. Our fitness is the same
    expression cost + rho * violation, only the cost of our problem is positive.
    """
    for rho, expected in ((12.0, -1.0), (2.0, -31.0)):
        assert (-37.0) + rho * 3.0 == expected


def test_the_penalty_is_zero_exactly_when_the_solution_is_feasible(d):
    rng = np.random.default_rng(4)
    for _ in range(300):
        a = gp.random_individual(d, rng)
        _, _, violation, feasible = gp.evaluate(d, a, 3.0)
        assert (violation == 0.0) == feasible


def test_a_larger_rho_never_lowers_the_fitness_of_an_infeasible_individual(d):
    rng = np.random.default_rng(5)
    seen = 0
    for _ in range(500):
        a = gp.random_individual(d, rng)
        f1, _, viol, _ = gp.evaluate(d, a, 1.0)
        f2, _, _, _ = gp.evaluate(d, a, 100.0)
        if viol > 0:
            seen += 1
            assert f2 > f1
        else:
            assert abs(f2 - f1) < 1e-9
    assert seen > 0, "no infeasible individual was drawn, the test proved nothing"


# --------------------------------------------------------------- the operators

def test_crossover_returns_valid_individuals_on_1000_pairs(d):
    rng = np.random.default_rng(6)
    for _ in range(N_TRIALS):
        p1 = gp.random_individual(d, rng)
        p2 = gp.random_individual(d, rng)
        child = gp.crossover(d, p1, p2, rng)
        assert gp.is_valid(d, child)


def test_crossover_centers_come_from_the_parents(d):
    rng = np.random.default_rng(7)
    for _ in range(300):
        p1 = gp.random_individual(d, rng)
        p2 = gp.random_individual(d, rng)
        child = gp.crossover(d, p1, p2, rng)
        union = set(int(c) for c in gp.centers_of(p1)) | set(int(c) for c in gp.centers_of(p2))
        assert set(int(c) for c in gp.centers_of(child)) <= union


def test_mutation_returns_valid_individuals_on_1000_individuals(d):
    rng = np.random.default_rng(8)
    for _ in range(N_TRIALS):
        a = gp.random_individual(d, rng)
        for n in (1, 2):
            m = gp.mutate(d, a, rng, n_components=n)
            assert gp.is_valid(d, m)


def test_mutation_does_not_change_its_argument(d):
    rng = np.random.default_rng(9)
    a = gp.random_individual(d, rng)
    before = a.copy()
    gp.mutate(d, a, rng, n_components=2)
    assert np.array_equal(a, before)


def test_mutation_usually_changes_something(d):
    """A mutation operator that silently does nothing would make the GA a random restart."""
    rng = np.random.default_rng(10)
    changed = 0
    for _ in range(200):
        a = gp.random_individual(d, rng)
        m = gp.mutate(d, a, rng, n_components=1)
        if not np.array_equal(a, m):
            changed += 1
    assert changed > 150, f"only {changed} of 200 mutations changed the individual"


def test_swap_keeps_the_individual_valid(d):
    rng = np.random.default_rng(11)
    for _ in range(300):
        a = gp.random_individual(d, rng)
        s = gp.swap_mutate(d, a, rng)
        assert gp.is_valid(d, s)
        assert set(int(c) for c in gp.centers_of(s)) == set(int(c) for c in gp.centers_of(a))
