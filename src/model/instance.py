"""Team T03 instance: storyline A, variant 1, seed 3.

The call is fixed by the team task file and must not change: the same call gives
the same instance in Checkpoint 4.

    d = cp34_instance(3, 'A', 1)

Check values from the task file (all seven are verified in tests/test_model.py):
    N = 35, V = 13, Cap = 105, sum of demand = 755,
    sum_i sum_k d_ik = 59554.520,
    variables = 1260, constraint rows = 1331.
"""
import importlib.util
import os

import numpy as np

SEED = 3
STORY = "A"
VARIANT = 1

# the seven numbers of the team task file
CHECK_VALUES = {
    "N": 35,
    "V": 13,
    "Cap": 105,
    "sum_demand": 755,
    "sum_dist": 59554.520,
    "n_variables": 1260,
    "n_rows": 1331,
}

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_GENERATOR = os.path.join(_ROOT, "checkpoint3_generator_PUBLIC.py")


def _generator_module():
    """Import the public generator from the repository root by file path.

    It is kept at the root, unchanged, exactly as the course published it.
    """
    spec = importlib.util.spec_from_file_location("checkpoint3_generator_PUBLIC", _GENERATOR)
    if spec is None or spec.loader is None:
        raise ImportError("cannot locate checkpoint3_generator_PUBLIC.py at " + _GENERATOR)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_instance():
    """Return the team instance dictionary d = cp34_instance(3, 'A', 1)."""
    return _generator_module().cp34_instance(SEED, STORY, VARIANT)


def instance_summary(d=None):
    """The check values recomputed from the instance, as a plain dictionary."""
    if d is None:
        d = load_instance()
    N = int(d["N"])
    return {
        "N": N,
        "V": int(d["V"]),
        "Cap": int(d["Cap"]),
        "sum_demand": int(np.sum(d["demand"])),
        "sum_dist": float(np.sum(d["dist"])),
        "n_variables": N * N + N,
        "n_rows": N + N * N + N + N + 1,
    }
