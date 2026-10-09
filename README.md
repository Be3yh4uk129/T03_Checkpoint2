# Checkpoint 2 — Unconstrained Component: optimizers from scratch

Team: T03 · Course: Introduction to Optimization · Deadline: 09.10.2026, 23:59 · Tag: `checkpoint2`

## Roles

Roles are fixed in the first commit and are not changed without the instructor's approval.

| Role | Member (name as in roster) | GitHub       | Code authored | Analysis | Hand trace |
|------|----------------------------|--------------|---------------|----------|-----------|
| M1 — GD       | Amanbay Amir               | @Be3yh4uk129 | `optim/gd.py`, `problems/rosenbrock.py`, `check_grad` (`problems/base.py`), `experiments/plots.py`* | S1 | H1 |
| M2 — Newton   | <Name Surname>             | @<github>    | `optim/newton.py`, `problems/quadratic.py` | S2 | H2 |
| M3 — Momentum | <Name Surname>             | @<github>    | `optim/momentum.py`, `problems/project.py` | S3 | H3 |
| M4 — Adam     | <Name Surname>             | @<github>    | `optim/adam.py`, `experiments/run_all.py`, tables | S4 | H4 |

\* The task does not assign `experiments/plots.py` in a team of four; the team agreed that M1 writes it.
S5 (project block) and H5 are done jointly by all four members and graded as group work.

## Layout

```
src/optim/         gd.py  newton.py  momentum.py  adam.py  common.py (shared constants)
src/problems/      rosenbrock.py  quadratic.py  project.py  base.py (Problem, check_grad)
src/experiments/   run_all.py  plots.py
tests/             test_gd.py  test_newton.py  test_momentum.py  test_adam.py
results/           *.csv  (all numbers behind Tables 1 and 2)
hand/              hand traces H1-H5 (PDF or scans)
```

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
PYTHONPATH=src python -m experiments.run_all            # regenerates all tables (CSV) and figures
PYTHONPATH=src python -m experiments.plots              # figures F1-F3 -> figures/ (reads results/table1.csv, table2.csv)
PYTHONPATH=src python -m experiments.s1_extra           # extra runs behind S1 -> results/s1_*.csv
```

Windows PowerShell: `$env:PYTHONPATH="src"; python -m experiments.run_all`.

Environment: Python 3.10+, numpy, matplotlib (plots), pytest, sympy (tests only).
Inside `src/optim/` only numpy and the standard library are allowed.

## Git rules (team agreement)

- `git config user.name` = real name as in the roster (Latin letters); `user.email` = an address
  added to your GitHub account.
- Commit message starts with a tag: `[GD]`, `[Newton]`, `[Momentum]`, `[Adam]`, `[Infra]`, `[Report]`.
- Small commits over several days; no force-push, no squash, no history rewriting after the first push.
- You may review or fix someone else's module, but the commit author is the credited person.
- Final commit: `git tag checkpoint2 && git push --tags`. Commits after the deadline are ignored.

## Declarations

-  [Infra] and M1 | Repository skeleton folder structure, `README.md` template, `src/optim/common.py`, `src/problems/base.py` (`Problem`), 
stubs, test stubs | GD code `src/optim/gd.py`, `src/problems/rosenbrock.py`, `check_grad` / `check_hess` in `src/problems/base.py`, 
`tests/test_gd.py` | drafted by AI, then read, run against the reference iteration counts and committed by the author 
-  [GD] M1 | `src/experiments/plots.py` (F1-F3), `src/experiments/s1_extra.py`, draft of S1 (`report/S1_AmirAmanbay.md`) |
drafted with an AI assistant (Claude), then read, run, and every number checked against `results/*.csv` by the author

## References
- Lecture 1-5 (Nurseitova A.T.)

