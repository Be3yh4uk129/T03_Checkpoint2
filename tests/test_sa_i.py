"""Role test SA-I (Amir Amanbay): src/meta/sa.py.

1. the incremental cost equals the checker cost on 1000 random moves (to 1e-9)
2. the same seed gives the same run; different seeds give different runs
3. exactly the budget is spent
4. the best-so-far trace never increases
plus: seeding numpy / random globally does not change a run.
"""
import os
import random
import sys

import numpy as np
import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.meta import sa  # noqa: E402

T = {"T0": 100.0, "Tend": 0.3}       # any positive pair works for these tests


@pytest.fixture(scope="module")
def d():
    return sa._generator().cp34_instance(3, "A", 1)


# ------------------------------------------------------------------ 1. incremental cost
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_incremental_cost_equals_checker_on_1000_moves(d, seed):
    rng = random.Random(seed)
    st = sa.State(d, *sa.random_start(d, rng))
    cost0, feas0, _ = sa.check(d, st.solution())
    assert feas0
    assert abs(st.cost - cost0) <= 1e-9
    seen = set()
    for _ in range(1000):
        mv = st.propose(rng)
        seen.add(mv[0])
        st.apply(mv)
        cost, feasible, problems = sa.check(d, st.solution())
        assert abs(st.cost - cost) <= 1e-9, (mv, st.cost, cost)
        assert feasible, problems                 # only feasible moves are proposed
        assert feasible == st.feasible
    assert seen == {sa.TRANSFER, sa.SWAP, sa.RECENTER}


def test_each_move_type_delta_by_hand(d):
    """The delta of each move type against a recomputation from scratch, one by one."""
    rng = random.Random(7)
    st = sa.State(d, *sa.random_start(d, rng))
    for propose in (st._propose_transfer, st._propose_swap, st._propose_recenter):
        for _ in range(200):
            mv = propose(rng)
            before = sa.full_cost(d, st.centers, st.assign)
            st.apply(mv)
            after = sa.full_cost(d, st.centers, st.assign)
            assert abs((after - before) - mv[3]) <= 1e-9


# ------------------------------------------------------------------ 2. reproducibility
def test_same_seed_same_run(d):
    a = sa.solve(d, 1, 5, 20000, params=T)
    b = sa.solve(d, 1, 5, 20000, params=T)
    assert a["cost"] == b["cost"]
    assert a["solution"] == b["solution"]
    assert a["trace"] == b["trace"]
    assert a["evals"] == b["evals"]


def test_different_seeds_differ(d):
    runs = [sa.solve(d, 1, s, 5000, params=T) for s in range(4)]
    assert len({tuple(r["trace"]) for r in runs}) > 1
    assert len({(tuple(r["solution"][0]), tuple(r["solution"][1])) for r in runs}) > 1


def test_global_random_state_is_not_used(d):
    np.random.seed(1)
    random.seed(1)
    a = sa.solve(d, 1, 3, 10000, params=T)
    np.random.seed(999)
    random.seed(999)
    for _ in range(17):
        np.random.rand()
        random.random()
    b = sa.solve(d, 1, 3, 10000, params=T)
    assert a["trace"] == b["trace"] and a["solution"] == b["solution"]


# ------------------------------------------------------------------ 3. budget
@pytest.mark.parametrize("budget", [1, 2, 3, 99, 100, 101, 999, 1000, 1001, 1234, 7777, 20000])
def test_exact_budget_with_params(d, budget):
    r = sa.solve(d, 1, 0, budget, params=T)
    assert r["evals"] == budget


@pytest.mark.parametrize("budget", [1, 5, 57, 299, 2001, 30001, 31234])
def test_exact_budget_rule_inside_run(d, budget):
    r = sa.solve(d, 1, 0, budget)                 # rule-based walk is charged to the budget
    assert r["evals"] == budget


# ------------------------------------------------------------------ 4. trace
@pytest.mark.parametrize("seed", [0, 1])
def test_trace_never_increases_and_is_dense(d, seed):
    r = sa.solve(d, 1, seed, 23456, params=T)
    tr = r["trace"]
    for (e0, c0), (e1, c1) in zip(tr, tr[1:]):
        assert c1 <= c0
        assert e1 > e0
        assert e1 - e0 <= 1000                   # at least every 1000 evaluations
    assert tr[-1][0] == r["evals"] == 23456
    assert tr[-1][1] == pytest.approx(r["cost"], abs=1e-9)


# ------------------------------------------------------------------ interface and checker
def test_interface_and_honest_feasible_flag(d):
    r = sa.solve(d, 1, 2, 5000, params=T, start=None)
    for key in ("solution", "cost", "evals", "feasible", "trace"):
        assert key in r
    centers, assign, vehicle = r["solution"]
    assert vehicle is None
    cost, feasible, _ = sa.check(d, r["solution"])
    assert r["cost"] == cost and r["feasible"] == feasible


def test_start_is_accepted(d):
    rng = random.Random(11)
    start = sa.random_start(d, rng)
    r = sa.solve(d, 1, 0, 3000, params=T, start=(start[0], start[1], None))
    assert r["evals"] == 3000
    assert r["start_cost"] == pytest.approx(sa.full_cost(d, *start), abs=1e-9)
    assert r["cost"] <= r["start_cost"] + 1e-9


def test_count_improving_is_zero_after_greedy_descent(d):
    """After a transfer/swap descent to a local optimum the helper finds no improving move."""
    rng = random.Random(3)
    st = sa.State(d, *sa.random_start(d, rng))
    improved = True
    while improved:
        improved = False
        for i in st.nc:
            for c in st.centers:
                a = st.assign[i]
                if c != a and st.load[c] + st.q[i] <= st.Cap and st.dist[i][c] - st.dist[i][a] < -1e-9:
                    st.apply((sa.TRANSFER, i, c, st.dist[i][c] - st.dist[i][a]))
                    improved = True
    n_tr, _ = sa.count_improving(d, st.solution())
    assert n_tr == 0
