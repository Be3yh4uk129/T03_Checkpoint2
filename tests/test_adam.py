import pytest
from helpers import close_count
from optim.adam import adam
from problems.rosenbrock import make_r1


def test_adam_r1():
    _, _, k = adam(make_r1(), alpha=0.1)
    assert close_count(k, 1706)

@pytest.mark.skip(reason="H4: fill in from the task file, reproduce hand numbers to 1e-3")
def test_hand_trace_h4():
    pass
