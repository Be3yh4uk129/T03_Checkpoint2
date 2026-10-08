import pytest
from helpers import close_count
from optim.gd import gd, gd_bt
from problems.rosenbrock import make_r1

SKEL = pytest.mark.xfail(raises=NotImplementedError, reason="not implemented yet; remove this mark")


@SKEL
def test_gd_r1():
    _, _, k = gd(make_r1(), alpha=1e-3)
    assert close_count(k, 32076)


@SKEL
def test_gd_bt_r1():
    _, _, k = gd_bt(make_r1())
    assert close_count(k, 13756)


@pytest.mark.skip(reason="H1: fill in from the task file, reproduce hand numbers to 1e-3")
def test_hand_trace_h1():
    pass
