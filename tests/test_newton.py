import pytest
from helpers import close_count
from optim.newton import newton_pure, newton_damped
from problems.rosenbrock import make_r1

SKEL = pytest.mark.xfail(raises=NotImplementedError, reason="not implemented yet; remove this mark")


@SKEL
def test_newton_pure_r1():
    _, _, k = newton_pure(make_r1())
    assert close_count(k, 6)


@SKEL
def test_newton_damped_r1():
    _, _, k = newton_damped(make_r1())
    assert close_count(k, 21)


@pytest.mark.skip(reason="H2: fill in from the task file, reproduce hand numbers to 1e-3")
def test_hand_trace_h2():
    pass
