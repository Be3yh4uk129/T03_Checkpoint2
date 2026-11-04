"""Nelder-Mead from scratch (role SA-T, Zhamilya Kaniyeva).

Coefficients of Lecture 9: reflection 1, expansion 2, contraction 1/2, shrinkage 1/2.

One iteration
    1. sort the vertices, best x_1 ... worst x_{n+1}; centroid c of all but the worst
    2. reflect   x_r = c + (c - x_worst)
    3. f_r < f_best          -> expand x_e = c + 2 (c - x_worst); keep the better of x_e, x_r
       f_best <= f_r < f_2nd -> keep x_r
       otherwise contract:
           outside x_c = c + 1/2 (x_r - c)     if f_r <  f_worst
           inside  x_c = c + 1/2 (x_worst - c) if f_r >= f_worst
           keep x_c if it is better than min(f_r, f_worst), else shrink every vertex
           halfway towards the best one

Every value of f is computed once and counted: the initial vertices count, nothing is
evaluated twice. Only numpy and the standard library.
"""
import numpy as np

ALPHA, GAMMA, RHO, SIGMA = 1.0, 2.0, 0.5, 0.5


class _BudgetExhausted(Exception):
    pass


def nelder_mead(f, simplex, max_evals=200, max_iter=None, xtol=1e-6, ftol=1e-10, callback=None):
    """Minimise f starting from the given simplex (n+1 points in R^n).

    Stops when max_evals evaluations are spent (never more), after max_iter iterations, or
    when the simplex is smaller than xtol and the spread of the values is below ftol.

    Returns a dict:
        x, fx          best vertex and its value
        evals          number of evaluations of f (initial vertices included)
        iterations     number of completed iterations
        best_history   best value after the initial simplex and after every iteration
        calls          list of (eval_number, x, f(x)) for every evaluation, in order
        steps          the kind of each iteration (reflect, expand, contract_out, ...)
    """
    calls = []

    def F(x):
        if len(calls) >= max_evals:
            raise _BudgetExhausted
        x = np.array(x, dtype=float)
        v = float(f(x))
        calls.append((len(calls) + 1, x.copy(), v))
        if callback is not None:
            callback(len(calls), x, v)
        return v

    pts = [np.array(p, dtype=float) for p in simplex]
    n = len(pts) - 1
    if any(p.shape != (n,) for p in pts):
        raise ValueError("the simplex needs n+1 points of dimension n")

    vals = []
    best_history, steps = [], []
    it = 0
    try:
        for p in pts:
            vals.append(F(p))
        best_history.append(min(vals))
        while max_iter is None or it < max_iter:
            order = np.argsort(vals, kind="stable")
            pts = [pts[i] for i in order]
            vals = [vals[i] for i in order]
            size = max(np.linalg.norm(p - pts[0]) for p in pts[1:])
            if size < xtol and vals[-1] - vals[0] < ftol:
                break
            c = np.mean(pts[:-1], axis=0)
            xw, fw = pts[-1], vals[-1]
            xr = c + ALPHA * (c - xw)
            fr = F(xr)
            if fr < vals[0]:
                xe = c + GAMMA * (c - xw)
                fe = F(xe)
                if fe < fr:
                    pts[-1], vals[-1], kind = xe, fe, "expand"
                else:
                    pts[-1], vals[-1], kind = xr, fr, "reflect"
            elif fr < vals[-2]:
                pts[-1], vals[-1], kind = xr, fr, "reflect"
            else:
                if fr < fw:
                    xc, kind = c + RHO * (xr - c), "contract_out"
                else:
                    xc, kind = c + RHO * (xw - c), "contract_in"
                fc = F(xc)
                if fc < min(fr, fw):
                    pts[-1], vals[-1] = xc, fc
                else:
                    kind = "shrink"
                    for i in range(1, n + 1):
                        pts[i] = pts[0] + SIGMA * (pts[i] - pts[0])
                        vals[i] = F(pts[i])
            it += 1
            steps.append(kind)
            best_history.append(min(vals))
    except _BudgetExhausted:
        pass

    if not vals:
        raise ValueError("max_evals is too small for the initial simplex")
    # best vertex; a partly evaluated shrink can only leave already evaluated vertices behind
    best_call = min(calls, key=lambda t: t[2])
    return {
        "x": best_call[1],
        "fx": best_call[2],
        "evals": len(calls),
        "iterations": it,
        "best_history": best_history,
        "calls": calls,
        "steps": steps,
    }
