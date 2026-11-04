"""Role test SA-T (Zhamilya Kaniyeva): src/meta/nm.py.

1. the worked example of Lecture 9: simplex (0,0), (1,0), (2,1); best values 5, 3.25, 1.953,
   1.771 after three iterations, using exactly 9 evaluations (the three vertices included)
2. convergence on f(x1, x2) = (x1 - 3)^2 + 4 (x2 - 2)^2 to within 1e-3 of (3, 2) in at most
   100 evaluations (the initial vertices included)
"""
import os
import sys

import numpy as np
import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.meta.nm import nelder_mead  # noqa: E402


class Counted:
    def __init__(self):
        self.n = 0

    def __call__(self, x):
        self.n += 1
        return (x[0] - 3.0) ** 2 + 4.0 * (x[1] - 2.0) ** 2


SIMPLEX = [(0.0, 0.0), (1.0, 0.0), (2.0, 1.0)]


def test_lecture9_worked_example_values_and_nine_evaluations():
    f = Counted()
    r = nelder_mead(f, SIMPLEX, max_iter=3, max_evals=1000)
    assert r["iterations"] == 3
    assert [round(v, 3) for v in r["best_history"]] == [5.0, 3.25, 1.953, 1.771]
    assert r["best_history"][0] == 5.0
    assert r["best_history"][1] == 3.25
    assert abs(r["best_history"][2] - 1.953125) < 1e-12
    assert abs(r["best_history"][3] - 1.7705078125) < 1e-12
    assert r["evals"] == 9
    assert f.n == 9                      # f really called 9 times, nothing evaluated twice
    assert r["steps"] == ["expand", "contract_out", "contract_in"]


def test_convergence_to_minimiser_within_100_evaluations():
    f = Counted()
    r = nelder_mead(f, SIMPLEX, max_evals=100, xtol=1e-5, ftol=1e-12)
    assert r["evals"] <= 100
    assert f.n <= 100
    assert np.linalg.norm(r["x"] - np.array([3.0, 2.0])) < 1e-3


def test_hard_budget_is_never_exceeded():
    for budget in (3, 4, 5, 10, 17):
        f = Counted()
        r = nelder_mead(f, SIMPLEX, max_evals=budget, xtol=0.0, ftol=0.0)
        assert r["evals"] == f.n <= budget


def test_bad_simplex_is_rejected():
    with pytest.raises(ValueError):
        nelder_mead(Counted(), [(0.0, 0.0), (1.0, 0.0)], max_evals=10)
