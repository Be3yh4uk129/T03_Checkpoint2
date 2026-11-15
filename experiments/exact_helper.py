"""Exact solver for Block 3 of the Final: HiGHS, one thread (PuLP) / default (scipy), wall-clock limit 120 s.

Use the route that matches how your Checkpoint 3 model was built; both return the same dictionary:
    objective, bound, solver_gap, proven, seconds

(a) Your model is a set of matrices solved with scipy.optimize.milp (you pass c, constraints, bounds, integrality):
        from exact_helper import solve_milp
        res = solve_milp(c, constraints, bounds, integrality, limit=120)
    Pass exactly what you pass to scipy.optimize.milp (a minimization). Needs scipy >= 1.9.

(b) Your model is a PuLP problem `prob`:
        from exact_helper import solve_pulp
        res = solve_pulp(prob, limit=120)
    Install once:  pip install pulp highspy

In both cases the model must be built FOR THE SCALED INSTANCE: your function that builds the model has to take the
instance (for example build_model(d)) and not create it inside.

`solver_gap` = (objective - bound) / |objective|, where `bound` is the best bound found by the solver during the search.
`proven` is True when the gap is zero up to 1e-6. The word 'Optimal' of PuLP is not a proof: the time limit may have stopped the solver.
`solver_gap` is NOT the gap_LP of the report: gap_LP is measured from the root LP relaxation and is positive even at the optimum.
"""
import time


def _pack(obj, bound, seconds):
    gap = None
    if obj is not None and bound is not None and obj:
        gap = max(0.0, (obj - bound) / abs(obj))
    return dict(objective=obj, bound=bound, solver_gap=gap,
                proven=bool(gap is not None and gap <= 1e-6), seconds=round(seconds, 1))


def solve_milp(c, constraints, bounds, integrality, limit=120.0):
    from scipy.optimize import milp
    t0 = time.time()
    res = milp(c, constraints=constraints, integrality=integrality, bounds=bounds,
               options=dict(time_limit=limit, mip_rel_gap=1e-6, disp=False))
    sec = time.time() - t0
    obj = float(res.fun) if res.x is not None else None
    bound = getattr(res, 'mip_dual_bound', None)
    return _pack(obj, None if bound is None else float(bound), sec)


def solve_pulp(prob, limit=120.0):
    import pulp
    solver = pulp.HiGHS(msg=False, timeLimit=limit, threads=1, gapRel=1e-6)
    t0 = time.time()
    prob.solve(solver)
    sec = time.time() - t0
    obj = pulp.value(prob.objective)
    has_solution = obj is not None and all(v.varValue is not None for v in prob.variables())
    bound = None
    try:
        bound = float(prob.solverModel.getInfo().mip_dual_bound)
    except Exception:
        pass
    return _pack(obj if has_solution else None, bound, sec)


solve_exact = solve_pulp   # old name
