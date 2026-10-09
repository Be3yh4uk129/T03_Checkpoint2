"""Extra experiment runs behind the analysis sections S1-S4.

Separate from run_all.py so that the tuning tables and the analysis runs can be
regenerated independently:

    PYTHONPATH=src python -m experiments.analysis

Writes to results/:
  s1_stability.csv  S1: GD at alpha = t * 2/lam_max on Q1, measured vs predicted decay
  s1_backtrack.csv  S1: mean accepted alpha and mean halvings for GD-BT
  s2_kappa.csv      S2: iterations against the condition number c
  s2_newton.csv     S2: pure vs damped Newton, f monotonicity, eigenvalues at x*
  s3_beta.csv       S3: beta sweep on R1 at alpha = 1e-3, and whether f ever increased
  s3_heavyball.csv  S3: tuned (alpha, beta) against the Polyak heavy-ball choice on Q1
  s4_rotation.csv   S4: Adam on Q1 vs Q2, next to the rotation-invariant methods
  s4_alpha.csv      S4: Adam iterations against alpha on R1, Q1 and the project block
"""
import csv
import os

import numpy as np

from optim.adam import adam
from optim.gd import gd, gd_bt
from optim.momentum import momentum
from optim.newton import newton_damped, newton_pure
from optim.common import reached_tol
from problems.quadratic import alpha_star, heavy_ball_params, make_q1, make_q2
from problems.rosenbrock import make_r1, make_r2

SEED = 3
C_TEAM = 106
THETA_DEG = 49

ALPHA_GRID = 10.0 ** np.linspace(-5, 0, 11)
BETA_GRID = [0.5, 0.8, 0.9, 0.95, 0.99]
ALPHA_MAX_EXT = 1e4

RESULTS = os.path.join(os.path.dirname(__file__), "..", "..", "results")
DASH = "-"


def _run(solver, prob, **kw):
    """Run one configuration. Returns iterations, or None if it did not converge."""
    x, _, k = solver(prob, **kw)
    return k if reached_tol(prob, x) else None


def _extended_alphas(base):
    top = np.log10(base[-1])
    return 10.0 ** np.arange(top + 0.5, np.log10(ALPHA_MAX_EXT) + 1e-9, 0.5)


def tune_alpha(solver, prob, extra_alphas=(), **kw):
    """Fewest iterations over the alpha grid, extending upward if the best alpha
    sits on the upper edge. Returns (k, alpha, extended)."""
    alphas = list(ALPHA_GRID) + list(extra_alphas)
    best = (None, None)
    for a in alphas:
        k = _run(solver, prob, alpha=a, **kw)
        if k is not None and (best[0] is None or k < best[0]):
            best = (k, a)
    extended = False
    if best[1] is not None and np.isclose(best[1], ALPHA_GRID[-1]):
        extended = True
        for a in _extended_alphas(ALPHA_GRID):
            k = _run(solver, prob, alpha=a, **kw)
            if k is not None and (best[0] is None or k < best[0]):
                best = (k, a)
    return best[0], best[1], extended


def tune_alpha_beta(prob):
    """Momentum: fewest iterations over the alpha x beta grid.

    The grid is the eleven protocol values only; alpha* is an extra candidate for
    GD on Q1/Q2, not for momentum.
    """
    best = (None, None, None)
    for a in ALPHA_GRID:
        for b in BETA_GRID:
            k = _run(momentum, prob, alpha=a, beta=b)
            if k is not None and (best[0] is None or k < best[0]):
                best = (k, a, b)
    extended = False
    if best[1] is not None and np.isclose(best[1], ALPHA_GRID[-1]):
        extended = True
        for a in _extended_alphas(ALPHA_GRID):
            for b in BETA_GRID:
                k = _run(momentum, prob, alpha=a, beta=b)
                if k is not None and (best[0] is None or k < best[0]):
                    best = (k, a, b)
    return best[0], best[1], best[2], extended


def _fmt(k):
    return DASH if k is None else str(k)


def _a(alpha):
    return DASH if alpha is None else f"{alpha:.3e}"


def _write(name, header, rows):
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, name)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    print(f"  wrote results/{name}")


def _project():
    """The project block, or None if problems/project.py is not available."""
    try:
        from problems.project import make_project
        return make_project()
    except (ImportError, NotImplementedError):
        return None


# --------------------------------------------------------------------------- S1 (GD owner)
def s1_stability():
    """GD on Q1 at alpha = t * 2/lam_max, t in {0.5, 0.9, 0.99, 1.01, 1.1}.

    Also records the measured per-step decay of ||grad f|| and the predicted
    factor rho(alpha) = max_i |1 - alpha*lam_i|.
    """
    prob = make_q1(C_TEAM)
    lam = np.array([2.0, 2.0 * C_TEAM])
    rows = []
    for t in (0.5, 0.9, 0.99, 1.01, 1.1):
        a = t * 2.0 / lam.max()
        x, hist, k = gd(prob, alpha=a)
        ok = reached_tol(prob, x)
        rho = float(np.max(np.abs(1.0 - a * lam)))
        gn = np.array([np.linalg.norm(prob.grad(z)) for z in hist[: min(len(hist), 200)]])
        if gn.size > 10 and np.all(gn[:-1] > 0):
            measured = float(np.exp(np.mean(np.log(gn[1:] / gn[:-1]))))
        else:
            measured = float("nan")
        rows.append([f"{t:g}", f"{a:.6e}", "yes" if ok else "no", _fmt(k if ok else None),
                     f"{rho:.6f}", f"{measured:.6f}"])
    _write("s1_stability.csv",
           ["t", "alpha", "converged", "iters", "rho_predicted", "decay_measured"], rows)


def s1_backtracking():
    """Mean accepted alpha and mean halvings for GD-BT, against 1/lam_max(H) at x*."""
    rows = []
    targets = [("R1", make_r1()), ("Q1", make_q1(C_TEAM))]
    proj = _project()
    if proj is not None:
        targets.append(("project", proj))
    for name, prob in targets:
        x, _, k, info = gd_bt(prob, return_info=True)
        lam_max = float(np.max(np.linalg.eigvalsh(prob.hess(x))))
        rows.append([name, str(k),
                     f"{info['alphas'].mean():.6e}" if info["alphas"].size else DASH,
                     f"{info['halvings'].mean():.3f}" if info["halvings"].size else DASH,
                     f"{1.0 / lam_max:.6e}"])
    _write("s1_backtrack.csv",
           ["problem", "iters", "mean_alpha", "mean_halvings", "one_over_lam_max_at_x"], rows)


# --------------------------------------------------------------------------- S2 (Newton owner)
def s2_kappa():
    """Iterations against the condition number: GD(alpha*), momentum, Adam (tuned), Newton."""
    rows = []
    for c in (10, 100, 1000, C_TEAM):
        prob = make_q1(c)
        k_gd = _run(gd, prob, alpha=alpha_star(c))
        k_mo, a_mo, b_mo, _ = tune_alpha_beta(prob)
        k_ad, a_ad, _ = tune_alpha(adam, prob)
        k_np = _run(newton_pure, prob)
        rows.append([str(c), _fmt(k_gd), f"{alpha_star(c):.6e}",
                     _fmt(k_mo), _a(a_mo), DASH if b_mo is None else f"{b_mo:g}",
                     _fmt(k_ad), _a(a_ad), _fmt(k_np)])
    _write("s2_kappa.csv",
           ["c", "GD_alphastar_iters", "alpha_star", "Momentum_iters", "Momentum_alpha",
            "Momentum_beta", "Adam_iters", "Adam_alpha", "Newton_pure_iters"], rows)


def s2_newton():
    """Pure vs damped Newton on R2 and the project block: monotonicity of f and the
    second-order sufficient condition (eigenvalues of H) at the returned point."""
    rows = []
    targets = [("R2", make_r2())]
    proj = _project()
    if proj is not None:
        targets.append(("project", proj))
    for name, prob in targets:
        for label, solver in (("pure", newton_pure), ("damped", newton_damped)):
            x, hist, k = solver(prob)
            ok = reached_tol(prob, x)
            fvals = np.array([prob.f(z) for z in hist])
            monotone = bool(np.all(np.diff(fvals) <= 1e-12)) if fvals.size > 1 else True
            ev = np.linalg.eigvalsh(prob.hess(x))
            rows.append([name, label, _fmt(k if ok else None), "yes" if ok else "no",
                         "yes" if monotone else "no",
                         f"{ev.min():.6e}", f"{ev.max():.6e}",
                         "yes" if ev.min() > 0 else "no",
                         f"{np.linalg.norm(prob.grad(x)):.3e}"])
    _write("s2_newton.csv",
           ["problem", "variant", "iters", "converged", "f_monotone",
            "lam_min_at_x", "lam_max_at_x", "second_order_sufficient", "grad_norm"], rows)


# --------------------------------------------------------------------------- S3 (Momentum owner)
def s3_beta():
    """R1, alpha = 1e-3, beta sweep; records whether f ever increased."""
    rows = []
    for b in (0.0, 0.5, 0.8, 0.9, 0.95, 0.99):
        prob = make_r1()
        x, _, k, info = momentum(prob, alpha=1e-3, beta=b, return_info=True)
        ok = reached_tol(prob, x)
        rows.append([f"{b:g}", _fmt(k if ok else None), "yes" if ok else "no",
                     "yes" if info["f_increased"] else "no",
                     f"{info['f'].max():.6e}", f"{info['f'][0]:.6e}"])
    _write("s3_beta.csv",
           ["beta", "iters", "converged", "f_increased", "f_max", "f_x0"], rows)


def s3_heavyball():
    """Tuned (alpha, beta) against the Polyak heavy-ball constants on Q1, plus GD at alpha*."""
    prob = make_q1(C_TEAM)
    k_t, a_t, b_t, _ = tune_alpha_beta(prob)
    a_hb, b_hb = heavy_ball_params(C_TEAM)
    k_hb = _run(momentum, prob, alpha=a_hb, beta=b_hb)
    k_gd = _run(gd, prob, alpha=alpha_star(C_TEAM))
    kappa = float(C_TEAM)
    rows = [["tuned_grid", _a(a_t), DASH if b_t is None else f"{b_t:g}", _fmt(k_t)],
            ["polyak_heavy_ball", _a(a_hb), f"{b_hb:g}", _fmt(k_hb)],
            ["gd_alpha_star", _a(alpha_star(C_TEAM)), DASH, _fmt(k_gd)]]
    rows.append(["kappa", f"{kappa:g}", "sqrt_kappa", f"{np.sqrt(kappa):.3f}"])
    _write("s3_heavyball.csv", ["setting", "alpha", "beta", "iters"], rows)


# --------------------------------------------------------------------------- S4 (Adam owner)
def s4_rotation():
    """Adam on Q1 vs Q2 at a few alphas, next to the rotation-invariant methods."""
    q1, q2 = make_q1(C_TEAM), make_q2(C_TEAM, THETA_DEG)
    rows = []
    for a in (1e-3, 1e-2, 1e-1, 1.0):
        rows.append(["adam", f"{a:.3e}", _fmt(_run(adam, q1, alpha=a)),
                     _fmt(_run(adam, q2, alpha=a))])
    rows.append(["gd_alpha_star", _a(alpha_star(C_TEAM)),
                 _fmt(_run(gd, q1, alpha=alpha_star(C_TEAM))),
                 _fmt(_run(gd, q2, alpha=alpha_star(C_TEAM)))])
    rows.append(["gd_bt", DASH, _fmt(_run(gd_bt, q1)), _fmt(_run(gd_bt, q2))])
    rows.append(["newton_pure", DASH, _fmt(_run(newton_pure, q1)), _fmt(_run(newton_pure, q2))])
    rows.append(["momentum_1e-3_0.9", "1.000e-03",
                 _fmt(_run(momentum, q1, alpha=1e-3, beta=0.9)),
                 _fmt(_run(momentum, q2, alpha=1e-3, beta=0.9))])
    _write("s4_rotation.csv", ["solver", "alpha", "Q1_iters", "Q2_iters"], rows)


def s4_alpha():
    """Adam iterations against alpha on R1, Q1 and the project block (default 1e-3 included)."""
    targets = [("R1", make_r1()), ("Q1", make_q1(C_TEAM))]
    proj = _project()
    if proj is not None:
        targets.append(("project", proj))
    alphas = sorted(set(list(ALPHA_GRID) + [1e-3]))
    rows = []
    for name, prob in targets:
        for a in alphas:
            rows.append([name, f"{a:.3e}", _fmt(_run(adam, prob, alpha=a))])
    _write("s4_alpha.csv", ["problem", "alpha", "iters"], rows)



def main():
    os.makedirs(RESULTS, exist_ok=True)
    print(f"Team T03 - seed {SEED}: c = {C_TEAM}, theta = {THETA_DEG} deg\n")
    print("S1 ...");  s1_stability();  s1_backtracking()
    print("S2 ...");  s2_kappa();      s2_newton()
    print("S3 ...");  s3_beta();       s3_heavyball()
    print("S4 ...");  s4_rotation();   s4_alpha()
    print("\ndone.")


if __name__ == "__main__":
    main()
