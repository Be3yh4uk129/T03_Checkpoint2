"""Extra runs behind analysis section S1 (step size and stability).  Owner: M1 (GD).

Complements experiments.analysis (s1_stability.csv, s1_backtrack.csv):

    PYTHONPATH=src python -m experiments.s1_extra

Writes to results/:
  s1_modes.csv     Q1, alpha = t * 2/lam_max: per-mode factors |1 - alpha*lam_i|, the decay
                   of ||grad f|| over the last 50 steps, and the count predicted by the
                   dominant mode, k = ln(tol*||g0|| / |g0_i|) / ln|1 - alpha*lam_i|
  s1_sweep.csv     Q1: fine alpha sweep t in [0.5, 0.999] (500 values) vs alpha* = 1/(1+c)
  s1_project.csv   project block: eigenvalues of H at x*, GD at a few fixed alphas vs GD-BT
"""
import csv
from pathlib import Path

import numpy as np

from optim.gd import gd, gd_bt
from optim.common import reached_tol
from problems.quadratic import make_q1, alpha_star
from problems.project import make_project

C_TEAM = 106
RESULTS = Path("results")
DASH = "-"


def _write(name, header, rows):
    RESULTS.mkdir(parents=True, exist_ok=True)
    with (RESULTS / name).open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    print(f"  wrote results/{name}")


def s1_modes():
    prob = make_q1(C_TEAM)
    lam = np.array([2.0, 2.0 * C_TEAM])
    g0 = prob.grad(prob.x0)                  # gradient components = mode amplitudes on Q1
    target = prob.tol * np.linalg.norm(g0)
    rows = []
    for t in (0.5, 0.9, 0.99, 1.01, 1.1):
        a = t * 2.0 / lam.max()
        x, hist, k = gd(prob, a, max_iter=2000 if t > 1 else 100_000)
        ok = reached_tol(prob, x)
        gn = np.array([np.linalg.norm(prob.grad(z)) for z in hist[-51:]])
        tail = float(np.exp(np.mean(np.log(gn[1:] / gn[:-1]))))
        fac = np.abs(1.0 - a * lam)
        # each mode alone needs ln(target/|g0_i|)/ln(fac_i) steps; the slower one decides
        pred = [np.log(target / abs(g0[i])) / np.log(fac[i]) if 0 < fac[i] < 1 else 0.0
                for i in range(2)]
        k_pred = int(np.ceil(max(pred))) if np.all(fac < 1) else None
        rows.append([f"{t:g}", f"{a:.6e}", f"{fac[0]:.5f}", f"{fac[1]:.5f}",
                     f"{fac.max():.5f}", f"{tail:.5f}",
                     str(k) if ok else DASH, DASH if k_pred is None else str(k_pred)])
    _write("s1_modes.csv",
           ["t", "alpha", "factor_lam_min", "factor_lam_max", "rho", "decay_last50",
            "iters", "iters_predicted"], rows)


def s1_sweep():
    prob = make_q1(C_TEAM)
    a_max = 2.0 / (2.0 * C_TEAM)
    best = (None, None)
    for t in np.linspace(0.5, 0.999, 500):
        x, _, k = gd(prob, t * a_max)
        if reached_tol(prob, x) and (best[0] is None or k < best[0]):
            best = (k, t)
    a_s = alpha_star(C_TEAM)
    _, _, k_s = gd(prob, a_s)
    _write("s1_sweep.csv", ["setting", "t", "alpha", "iters"],
           [["alpha_star", f"{a_s / a_max:.4f}", f"{a_s:.6e}", str(k_s)],
            ["best_of_sweep", f"{best[1]:.4f}", f"{best[1] * a_max:.6e}", str(best[0])]])


def s1_project():
    prob = make_project()
    x, _, k_bt = gd_bt(prob)
    ev = np.linalg.eigvalsh(prob.hess(x))
    rows = [["lam_min_at_x", f"{ev.min():.6e}", DASH, DASH],
            ["lam_max_at_x", f"{ev.max():.6e}", DASH, DASH],
            ["alpha_opt=2/(lmin+lmax)", f"{2.0 / ev.sum():.4f}", DASH, DASH],
            ["gd_bt (alpha0=1)", "1", str(k_bt), f"{np.max(np.abs(1 - ev)):.5f}"]]
    for a in (1.0, 10.0, 10 ** 1.5, 50.0, 100.0):
        xx, _, kk = gd(prob, a)
        rows.append(["gd", f"{a:.4g}", str(kk) if reached_tol(prob, xx) else DASH,
                     f"{np.max(np.abs(1 - a * ev)):.5f}"])
    _write("s1_project.csv", ["setting", "alpha", "iters", "rho_at_x"], rows)


def main():
    s1_modes()
    s1_sweep()
    s1_project()


if __name__ == "__main__":
    main()
