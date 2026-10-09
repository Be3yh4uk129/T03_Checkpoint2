"""Figures F1-F3 (matplotlib).  Owner: M1 (GD).

F1: trajectories of all six solvers on R1 over contours, levels=np.logspace(-1, 3.5, 20)
F2: ||grad f|| vs k (semilogy) on Q1 and on Q2, all solvers
F3: ||grad f|| vs k on the project block

The tuned alpha (and beta) are read from results/table1.csv and results/table2.csv,
so run experiments.run_all first; the figures then use exactly the Table 1/2 settings.

    PYTHONPATH=src python -m experiments.plots
"""
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from optim.common import MAX_ITER_GRAD, MAX_ITER_NEWTON
from optim.gd import gd, gd_bt
from optim.newton import newton_pure, newton_damped
from optim.momentum import momentum
from optim.adam import adam
from problems.rosenbrock import make_r1, rosenbrock_f
from problems.quadratic import make_q1, make_q2
from problems.project import make_project

C_TEAM, THETA_DEG = 106, 49
RESULTS = Path("results")
FIGURES = Path("figures")

SOLVERS = ["GD", "GD-BT", "Newton (pure)", "Newton (damped)", "Momentum", "Adam"]
STYLE = {
    "GD":              dict(color="tab:blue"),
    "GD-BT":           dict(color="tab:orange"),
    "Newton (pure)":   dict(color="tab:red", marker="o", ms=4),
    "Newton (damped)": dict(color="tab:purple", marker="s", ms=3),
    "Momentum":        dict(color="tab:green"),
    "Adam":            dict(color="tab:brown"),
}


def _read_table(name):
    """{(problem, solver): (alpha, beta)} from a results table; None where not tuned."""
    out = {}
    with (RESULTS / name).open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            a = float(row["alpha"]) if row["alpha"] not in ("", "—", "-") else None
            b = float(row["beta"]) if row["beta"] not in ("", "—", "-") else None
            out[(row["problem"], row["solver"])] = (a, b)
    return out


def _run_all_solvers(prob, params):
    """Run the six configurations with the Table 1/2 settings. Returns {label: hist}."""
    runs = {}
    for label in SOLVERS:
        a, b = params.get((prob.name, label), (None, None))
        if label == "GD":
            if a is None:
                continue
            _, hist, _ = gd(prob, a, max_iter=MAX_ITER_GRAD)
        elif label == "GD-BT":
            _, hist, _ = gd_bt(prob, max_iter=MAX_ITER_GRAD)
        elif label == "Newton (pure)":
            _, hist, _ = newton_pure(prob, max_iter=MAX_ITER_NEWTON)
        elif label == "Newton (damped)":
            _, hist, _ = newton_damped(prob, max_iter=MAX_ITER_NEWTON)
        elif label == "Momentum":
            if a is None or b is None:
                continue
            _, hist, _ = momentum(prob, a, b, max_iter=MAX_ITER_GRAD)
        else:
            if a is None:
                continue
            _, hist, _ = adam(prob, a, max_iter=MAX_ITER_GRAD)
        runs[label] = hist
    return runs


def _label(label, prob_name, params, hist):
    """Legend entry: solver, iteration count and the tuned parameters."""
    a, b = params.get((prob_name, label), (None, None))
    text = f"{label}, k={len(hist) - 1}"
    if a is not None:
        text += f", α={a:.3g}"
    if b is not None:
        text += f", β={b:g}"
    return text


def _grad_norms(prob, hist):
    return np.array([np.linalg.norm(prob.grad(z)) for z in hist])


def plot_f1(params=None):
    """Trajectories on R1: full view (pure Newton leaves the valley) and a zoom on the valley."""
    params = _read_table("table1.csv") if params is None else params
    prob = make_r1()
    runs = _run_all_solvers(prob, params)
    levels = np.logspace(-1, 3.5, 20)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    views = [((-1.6, 1.6), (-3.5, 2.0), "full view"),
             ((-1.4, 1.3), (-0.2, 1.5), "zoom on the valley")]
    for ax, (xl, yl, title) in zip(axes, views):
        X, Y = np.meshgrid(np.linspace(*xl, 400), np.linspace(*yl, 400))
        Z = rosenbrock_f(np.array([X, Y]))
        ax.contour(X, Y, Z, levels=levels, cmap="Greys", linewidths=0.6)
        for label, hist in runs.items():
            st = STYLE[label]
            ax.plot(hist[:, 0], hist[:, 1], lw=1.2, label=_label(label, prob.name, params, hist),
                    color=st["color"], marker=st.get("marker"), ms=st.get("ms", 0))
        ax.plot(*prob.x0, "k^", ms=8)
        ax.plot(1.0, 1.0, "k*", ms=12)
        ax.set_xlim(*xl)
        ax.set_ylim(*yl)
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_title(f"R1, start (−1.2, 1): {title}")
    axes[0].legend(loc="lower left", fontsize=8)
    fig.suptitle("F1. Trajectories of the six solvers on Rosenbrock R1 (▲ start, ★ minimizer)")
    fig.tight_layout()
    return _save(fig, "F1_R1_trajectories.png")


def _convergence_axes(ax, prob, params, runs, title):
    for label, hist in runs.items():
        gn = _grad_norms(prob, hist)
        st = STYLE[label]
        ax.semilogy(np.arange(gn.size), gn, lw=1.2, color=st["color"],
                    marker=st.get("marker"), ms=st.get("ms", 0),
                    label=_label(label, prob.name, params, hist))
    g0 = np.linalg.norm(prob.grad(prob.x0))
    ax.axhline(prob.tol * g0, color="k", ls="--", lw=0.8, label="stopping level $10^{-6}\\|\\nabla f(x^0)\\|$")
    ax.set_xscale("symlog", linthresh=10)
    ax.set_xlim(left=0)
    ax.set_xlabel("k (iterations, symlog)")
    ax.set_ylabel(r"$\|\nabla f(x^k)\|$")
    ax.set_title(title)
    ax.grid(True, which="major", alpha=0.3)
    ax.legend(fontsize=8)


def plot_f2(params=None):
    """||grad f|| against k on Q1 and Q2 (team c = 106, theta = 49 deg)."""
    params = _read_table("table1.csv") if params is None else params
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, prob, title in ((axes[0], make_q1(C_TEAM), f"Q1, c = {C_TEAM}"),
                            (axes[1], make_q2(C_TEAM, THETA_DEG), f"Q2, c = {C_TEAM}, θ = {THETA_DEG}°")):
        _convergence_axes(ax, prob, params, _run_all_solvers(prob, params), title)
    fig.suptitle("F2. Convergence on Q1 and Q2 (relative stopping rule)")
    fig.tight_layout()
    return _save(fig, "F2_Q1_Q2_convergence.png")


def plot_f3(params=None):
    """||grad F|| against k on the project block (single hub, delta = 1)."""
    params = _read_table("table2.csv") if params is None else params
    prob = make_project()
    fig, ax = plt.subplots(figsize=(8, 5))
    _convergence_axes(ax, prob, params, _run_all_solvers(prob, params),
                      "Project block: single hub, δ = 1, start at the warehouse")
    fig.suptitle("F3. Convergence on the project block")
    fig.tight_layout()
    return _save(fig, "F3_project_convergence.png")


def _save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / name
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  wrote {path}")
    return path


def main():
    plot_f1()
    plot_f2()
    plot_f3()


if __name__ == "__main__":
    main()
