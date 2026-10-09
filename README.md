# Checkpoint 2 — Unconstrained Component: optimizers from scratch

Team: T03 · Course: Introduction to Optimization · Deadline: 09.10.2026, 23:59 · Tag: `checkpoint2`

## Roles

Roles are fixed in the first commit and are not changed without the instructor's approval.

| Role | Member (name as in roster) | GitHub       | Code authored | Analysis | Hand trace |
|------|----------------------------|--------------|---------------|----------|-----------|
| M1 — GD       | Amanbay Amir        | @Be3yh4uk129 | `optim/gd.py`, `problems/rosenbrock.py`, `check_grad` (`problems/base.py`), `experiments/plots.py`* | S1 | H1 |
| M2 — Newton   | Ikhsanova Aziza     | @aziza200767593059     | `optim/newton.py`, `problems/quadratic.py` | S2 | H2 |
| M3 — Momentum | Abdrashit Yersultan | @Sunaizun    | `optim/momentum.py`, `problems/project.py` | S3 | H3 |
| M4 — Adam     | Kaniyeva Zhamilya   | @kzhamilya     | `optim/adam.py`, `experiments/run_all.py`, tables | S4 | H4 |

\* The task does not assign `experiments/plots.py` in a team of four; the team agreed that M1 writes it.
S5 (project block) and H5 are done jointly by all four members and graded as group work;
`tests/test_project.py` belongs to that joint part.

Note on attribution: M3's commits carry the git author name `Sunaizun`, which is the GitHub
account @Sunaizun belonging to Abdrashit Yersultan. The `[Momentum]` commit tag and this table
identify the author.

## Layout

```
src/optim/         gd.py  newton.py  momentum.py  adam.py  common.py (shared constants)
src/problems/      rosenbrock.py  quadratic.py  project.py  base.py (Problem, check_grad)
src/experiments/   run_all.py  analysis.py  plots.py
tests/             test_gd.py  test_newton.py  test_momentum.py  test_adam.py  test_project.py
results/           *.csv  (all numbers behind Tables 1 and 2)
figures/           F1-F3 (png), produced by experiments/plots.py
hand/              hand traces H1-H5 (PDF or scans)
```

## Team parameters

`checkpoint2_params(3)` (same seed as Checkpoint 1) gives `c = 106`, `theta = 49` degrees and the
Rosenbrock start `(-0.78, 0.38)`. The project block (Part C) places one hub by minimising the
demand-weighted smoothed distance to the 35 Checkpoint 1 delivery points; its data is embedded in
`src/problems/project.py`.

## Common interface

Every solver: `solver(prob, <params>, max_iter=...) -> (x, hist, k)` where `hist` holds all
iterates (including `x0`) and `k` is the number of updates. The stopping rule is checked at the
top of the loop, before the update. Failure: `||x|| > 1e12` or the iteration cap
(100 000 for gradient methods, 100 for Newton). See `src/optim/__init__.py` and
`src/optim/common.py`. A problem is a `problems.base.Problem` (`f`, `grad`, `hess`, `x0`,
`rule` = `"abs"`/`"rel"`, `tol`).

## How to run

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest                                                  # unit tests (R1 reference counts + hand traces)
PYTHONPATH=src python -m experiments.run_all            # Tables 1-2 and the full tuning grid
PYTHONPATH=src python -m experiments.analysis           # the extra runs behind sections S1-S4
PYTHONPATH=src python -m experiments.plots              # figures F1-F3
```

Windows PowerShell: `$env:PYTHONPATH="src"; python -m experiments.run_all`.

`pytest` must report **54 passed**. `run_all` takes about 4-5 minutes (the tuning grid is a few
hundred runs) and writes `table1.csv`, `table2.csv` and `tuning_details.csv`; `analysis` takes
about 45 seconds and writes the eight `s1`-`s4` CSVs; `plots` writes the three figures.

Environment: Python 3.10+, numpy, matplotlib (plots), pytest, sympy (tests only).
Inside `src/optim/` only numpy and the standard library are allowed.

## References

- Lectures 1-5, Nurseitova A.T., Introduction to Optimization (ItO2025).
- B. T. Polyak, "Some methods of speeding up the convergence of iteration methods",
  USSR Computational Mathematics and Mathematical Physics, 4(5):1-17, 1964.
  (Heavy-ball constants quoted in S3.)
- D. P. Kingma and J. Ba, "Adam: A Method for Stochastic Optimization", ICLR 2015.
  (Update rule and default beta1, beta2, eps.)
