"""Small shared pieces of the Final: paths, the instance calls and the CP3 model as scipy input.

The model itself is the Checkpoint 3 model of src/model/matrices.py; build_matrix_form(d) already
takes the instance as a parameter, so the same function builds the scaled instances of Block 3.
"""
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import checkpoint4_generator_PUBLIC as gen          # noqa: E402
from src.model.matrices import build_matrix_form    # noqa: E402

RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(ROOT, "figures")

Z_LP = 994.048826        # task file, reporting only
Z_REF = 997.603759       # task file, reporting only
SA_PARAMS = {"T0": 6.892341137379357, "Tend": 3.0081420522057414}   # results/Amanbay_chosen_params.json
SIZES = (60, 100, 150)


def own_instance():
    return gen.cp34_instance(3, "A", 1)


def scaled(n):
    return gen.scaled_instance("A", n, 1)


def scipy_input(d):
    """(c, constraints, bounds, integrality) of the CP3 model for scipy.optimize.milp."""
    from scipy.optimize import Bounds, LinearConstraint

    mf = build_matrix_form(d)
    lo_rows = np.where(mf.sense == "E", mf.b, -np.inf)
    hi_rows = mf.b.copy()
    constraints = LinearConstraint(mf.A, lo_rows, hi_rows)
    bounds = Bounds(mf.lo, mf.up)
    integrality = np.ones(mf.n_variables)
    return mf, mf.c, constraints, bounds, integrality


def solution_from_vector(mf, v):
    """Read (centers, assign) from a 0/1 solution vector of the CP3 model."""
    X, y = mf.split_solution(v)
    centers = [int(k) for k in np.where(y > 0.5)[0]]
    assign = [int(np.argmax(X[i])) for i in range(mf.N)]
    return centers, assign


def gaps(z):
    return 100.0 * (z - Z_REF) / Z_REF, 100.0 * (z - Z_LP) / z
