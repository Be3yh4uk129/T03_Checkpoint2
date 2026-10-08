import pytest
from helpers import close_count
from optim.momentum import momentum
from problems.rosenbrock import make_r1

SKEL = pytest.mark.xfail(raises=NotImplementedError, reason="not implemented yet; remove this mark")


@SKEL
def test_momentum_r1():
    _, _, k = momentum(make_r1(), alpha=1e-3, beta=0.9)
    assert close_count(k, 3020)


@pytest.mark.skip(reason="H3: fill in from the task file, reproduce hand numbers to 1e-3")
def test_hand_trace_h3():
    pass
