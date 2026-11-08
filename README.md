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
## Declarations

External code: none. All solvers in `src/optim/` were implemented from scratch using only numpy and Python's standard library.

AI assistance: Claude was used to help with some parts of this checkpoint. However, all code was reviewed, tested, and checked by the author before committing.

## References

- Lectures 1-5, Nurseitova A.T., Introduction to Optimization (ItO2025).
- B. T. Polyak, "Some methods of speeding up the convergence of iteration methods",
  USSR Computational Mathematics and Mathematical Physics, 4(5):1-17, 1964.
  (Heavy-ball constants quoted in S3.)
- D. P. Kingma and J. Ba, "Adam: A Method for Stochastic Optimization", ICLR 2015.
  (Update rule and default beta1, beta2, eps.)


---------------------------------------------


# Checkpoint 3 section for README.md

Team: T03 · Deadline: 23.10.2026, 23:59 · Tag: `checkpoint3`
Instance: `cp34_instance(3, 'A', 1)` (storyline A, variant 1, same seed as Checkpoints 1 and 2).

## Roles

Roles are fixed in the first commit of this checkpoint and are not changed afterwards.

| Role | Member (name as in roster) | GitHub | Code authored | Task | Section | Hand task |
|------|----------------------------|--------|---------------|------|---------|-----------|
| M1 | Amanbay Amir        | @Be3yh4uk129          | `src/model/` | Task 1 | S1 | H1 |
| M2 | Ikhsanova Aziza     | @aziza200767593059    | `src/lp/`    | Task 2 | S2 | H2 |
| M3 | Abdrashit Yersultan | @Sunaizun             | `src/kkt/`   | Task 3 | S3 | H3 |
| M4 | Kaniyeva Zhamilya   | @kzhamilya            | `src/sens/`  | Task 4 | S4 | H4 |

In a team of four, Task 5, section S5, hand task H5, `src/experiments/run_all_cp3.py` and
`tests/test_group.py` are joint work of all four members and are graded as group work (G4).

Note on attribution: M3's commits carry the git author name `Sunaizun`, which is the GitHub
account @Sunaizun belonging to Abdrashit Yersultan. The `[KKT]` commit tag and this table
identify the author.

## Layout added in Checkpoint 3

```
checkpoint3_generator_PUBLIC.py   the public generator, kept at the root, unchanged
src/model/      instance.py (the team call and the seven check values)
                matrices.py (c, A, b, row senses, bounds, row and column bookkeeping)
src/lp/         solve.py    (HiGHS through scipy, PuLP 3.3.2 with CBC, fractionality report)
src/kkt/        checker.py  (conditions (a) to (e)), convention.py (the sign-convention test)
src/sens/       perturb.py  (the three solves at b-0.01, b, b+0.01 and the interval check)
src/experiments/run_all_cp3.py    regenerates Tables 1 to 4
tests/          conftest.py  triangle.py  test_model.py  test_lp.py  test_kkt.py
                test_sens.py  test_group.py
results/        cp3_table1.csv  cp3_table2.csv  cp3_table3.csv
                cp3_active_rows.csv  cp3_fractional.csv
hand/           H1 to H5 (PDF or scans)
```

The runner is called `run_all_cp3.py` so that the Checkpoint 2 runner `run_all.py` keeps
working and both checkpoints stay reproducible. For the same reason every Checkpoint 3 CSV
starts with `cp3_`.

## Reference model

Storyline A, variant 1, exactly as in the handout, with no changes:

```
minimise  sum_i sum_k d_ik x_ik + 2 sum_k D_k y_k
(A1)  sum_k x_ik = 1             for every point i        35 rows,  equality
(A2)  x_ik <= y_k                for all i, k           1225 rows,  <=
(A3)  sum_i q_i x_ik <= Cap y_k  for every k              35 rows,  <=
(A4)  x_kk = y_k                 for every k              35 rows,  equality
(A5)  sum_k y_k = V                                        1 row,   equality
      0 <= x, y <= 1  (LP relaxation; bounds are not rows)
```

Column order: `x_ik` in column `i*N + k`, `y_k` in column `N*N + k`. Row order: the equality
rows (A1), (A4), (A5) first, then (A2) and (A3), so the designated row of Task 4 is row 70.

## Check values

All seven match the team task file and are asserted in `tests/test_model.py`.

| Quantity | Value |
|---|---|
| N (points) | 35 |
| V (centers) | 13 |
| Cap | 105 |
| sum of demand | 755 |
| sum_i sum_k d_ik | 59554.520 |
| number of variables | 1260 |
| number of constraint rows | 1331 |

Main results: `z_LP = 994.048826` (HiGHS, 0.028 s, 104 simplex iterations; PuLP 3.3.2 with CBC
gives the same value to a relative difference of 1.3e-11). All five optimality conditions are
satisfied. The dual price of (A5) is 44.482543 and lies exactly between the two one-sided
slopes, so the LP is not degenerate at b = 13.

## How to run

```bash
pip install "pulp<4"                                   # PuLP 4 changed the interface
pip install -r requirements.txt
pytest tests                                           # 33 passed
PYTHONPATH=src python -m experiments.run_all_cp3       # Tables 1 to 4 into results/
```

Windows PowerShell: `$env:PYTHONPATH="src"; python -m experiments.run_all_cp3`.

The whole run takes a few seconds. `pytest tests` must report **33 passed** for Checkpoint 3
(the Checkpoint 2 tests still report their own 54).

Environment: Python 3.10+, numpy, scipy, PuLP below version 4. The LP is solved with HiGHS
through `scipy.optimize.linprog`; PuLP with CBC is used only as a second opinion on `z_LP`.

## Git rules for this checkpoint

- Commit message starts with a tag: `[Model]`, `[LP]`, `[KKT]`, `[Sens]`, `[Report]`.
- `git config user.name` = real name as in the roster; `user.email` = an address on your
  GitHub account.
- No force-push, no squash, no history rewriting.
- You may review or fix someone else's module, but the commit author is the credited person.
- Final commit: `git tag checkpoint3 && git push --tags`. Commits after the deadline are ignored.

## Declarations (Checkpoint 3)

**External code.** None. The solvers themselves (HiGHS through `scipy.optimize.linprog`, CBC
through PuLP 3.3.2) are third-party components that the handout explicitly allows. The public
generator `checkpoint3_generator_PUBLIC.py` is kept at the repository root unchanged. The model,
the matrix assembly, the checker of the optimality conditions and the perturbation runs are our
own code.

**AI assistance.** In every case the author read the code, ran it against the check
values of the team task file and the hand numbers of their own hand task, and committed it
themselves.

| Part | Files | How AI was used |
|------|-------|-----------------|
| [Model] | `src/model/instance.py`, `src/model/matrices.py`, `tests/test_model.py`, `tests/conftest.py`, `tests/triangle.py` | drafted by AI, then read, checked against the seven check values of the task file and the hand-built corner of the matrix, and committed by the author |
| [LP] | `src/lp/solve.py`, `tests/test_lp.py` | drafted by AI, then read, verified by the HiGHS and CBC cross-check and against the H2 hand numbers, and committed by the author |
| [KKT] | `src/kkt/checker.py`, `src/kkt/convention.py`, `tests/test_kkt.py` | drafted by AI, then read, verified on the two-variable sign-convention LP and against the H3 hand numbers, and committed by the author |
| [Sens] | `src/sens/perturb.py`, `tests/test_sens.py` | drafted by AI, then read, verified against the H4 hand numbers and the interval check, and committed by the author |

The hand tasks H1 to H5 and the signed analysis sections S1 to S5 are the authors' own work.

## References

- Lectures 6 and 7, Nurseitova A.T., Introduction to Optimization (ItO2025):
  linear programming, duality, optimality conditions.
- HiGHS, used through `scipy.optimize.linprog(method="highs")`.
- PuLP 3.3.2 with the bundled CBC, used as a second solver for `z_LP` only.
# Checkpoint 4 - Your Own Metaheuristic

Team: T03 · Deadline: 08.11.2026, 23:59 · Tag: `checkpoint4`
Instance: `cp34_instance(3, 'A', 1)`, the same data as Checkpoints 1 to 3.
Reference values (task file, reporting only, never read by any solver):
`z_LP = 994.048826`, `z_ref = 997.603759`.

## Roles and configurations

| Student | Group | Configuration | Role | Code authored | Test | Results file |
|---|---|---|---|---|---|---|
| Ikhsanova Aziza | GA pair | GA, penalty | GA-A | `src/meta/ga.py`, `src/meta/random_search.py`, `experiments/exp_diversity_popsize.py`, `experiments/analyse.py` | `tests/test_ga_a.py` | `results/Ikhsanova_runs.csv`, `results/Ikhsanova_exp.csv` |
| Abdrashit Yersultan | GA pair | GA, penalty | GA-P | `src/meta/ga_problem.py`, `experiments/exp_rho.py` | `tests/test_ga_p.py` | `results/Abdrashit_exp.csv` |
| Amanbay Amir | SA pair | SA, transfer+swap | SA-I | `src/meta/sa.py` | `tests/test_sa_i.py` | `results/Amanbay_runs.csv` |
| Kaniyeva Zhamilya | SA pair | SA, transfer+swap | SA-T | `src/meta/nm.py`, `experiments/tuning.py` | `tests/test_sa_t.py` | `results/Kaniyeva_tuning.csv` |

The final runs of the GA pair are in `results/Ikhsanova_runs.csv`, written by GA-A; the
random-search baseline is in the same file with `configuration = random`. GA-P takes Table 1
and Figure F1 from that file.

Code is not shared between the two pairs. The only common code is the starter kit and the
`solve(...)` interface.

## The GA pair: method and design choices

**Representation.** An individual is one integer array `a` of length `N = 35`, where `a[i]`
is the point index of the center serving point `i`. A point is a center exactly when it
serves itself, `a[c] == c`, so the centers are the fixed points of the array. An individual
is structurally valid when it has exactly `V = 13` fixed points and every entry points at
one of them.

**Penalty (the pair's configuration).**

```
violation(a) = sum over centers of max(0, load(c) - Cap)      in demand units
fitness(a)   = cost(a) + rho * violation(a),      rho = 10 by default
```

The violation is measured in demand units because that is the unit the capacity is written
in, so one unit of overload is comparable across centers. `rho = 10` is chosen so that
removing one unit of overload is worth more than any single reassignment can save: the
largest distance in the instance is about 131 and the smallest demand is 15, so a transfer
that repairs an overload can save at most about `131 / 15 = 8.7` cost units per unit of
overload. `experiments/exp_rho.py` runs `rho` in `{1, 10, 100}` and reports what happens.
Only feasible solutions are ever reported.

**Moves used by the operators.** Besides crossover, mutation applies one or two components:

| move | what it does | counted in question 4 |
|---|---|---|
| transfer | a non-center point is reassigned to another center | yes |
| center move | a center is replaced by a non-center point, and the affected points go to their nearest surviving center | no, it changes the set of centers |

The center move is the additional move required to be listed here. Without it the set of
centers could only shrink towards the intersection of the initial population's center sets,
because crossover never introduces a center that neither parent had. Swap is implemented in
`ga_problem.swap_mutate` but the GA loop does not use it; it exists because question 4 of the
analysis has to count improving swaps as well as improving transfers.

**Counting of evaluations.** One evaluation is one call of `evaluate` on one candidate.
Every member of the initial population counts, every offspring counts, and the copy of the
elite individual costs nothing. The last generation is cut short so that `evals` equals
`budget` exactly.

**Trace.** `trace` holds `(evaluations, best feasible cost so far)` every 1000 evaluations
and once more at the end. Before the first feasible individual is found the second entry is
`inf`; the plotting code drops those points.

## How to reproduce every table and figure

Run from the repository root, with the generator `checkpoint4_generator_PUBLIC.py` in it.

```bash
pip install -r requirements.txt
python check_env.py --instance-seed 3 --story A --variant 1      # the five check values

pytest tests/test_ga_a.py tests/test_ga_p.py                     # 32 passed

# the 10 final runs of the GA pair, seeds 0-9, budget 100000       (GA-A)
PYTHONPATH=. python experiments/run_final.py --module src.meta.ga \
    --instance-seed 3 --story A --variant 1 --team T03 --role GA-A \
    --configuration "GA, penalty" --seeds 0-9 --budget 100000 \
    --out results/Ikhsanova_runs.csv

# the random-search baseline, same seeds and budget                (GA-A)
PYTHONPATH=. python experiments/run_final.py --module src.meta.random_search \
    --instance-seed 3 --story A --variant 1 --team T03 --role GA-A \
    --configuration "random" --seeds 0-9 --budget 100000 \
    --out results/Ikhsanova_runs.csv --append

# Table 1, Table 2 and Figure F1                                   (GA-A)
PYTHONPATH=. python experiments/analyse.py

# diversity per generation and the effect of the population size   (GA-A required item)
PYTHONPATH=. python experiments/exp_diversity_popsize.py

# three values of rho                                              (GA-P required item)
PYTHONPATH=. python experiments/exp_rho.py
```

`analyse.py` reruns `solve` on seeds 0 to 9 to recover the traces, which the results file
does not store, and stops if any recovered cost differs from the cost in the file. Every
number in `Ikhsanova_CP4.pdf` comes from `results/Ikhsanova_runs.csv`,
`results/Ikhsanova_analysis.json` and `results/Ikhsanova_exp.csv`.

## The SA pair (Amir Amanbay SA-I, Zhamilya Kaniyeva SA-T)

Configuration: **SA, transfer and swap**. Instance `cp34_instance(3, 'A', 1)`.

Agreed interface: `solve(d, variant, seed, budget, params={"T0": ..., "Tend": ...}, start=None)`.

**Moves and shares.** transfer 60 %, swap 30 %, and one **additional move, recenter** 10 %: a
non-center point j takes over the role of its own center k, the whole cluster of k is served
by j afterwards (loads unchanged). It is the only move that changes the set of centers.

**Constraint handling.** Only feasible moves are proposed: a draw that would push a load above
Cap = 105 is redrawn from the data (loads, demands) before its cost is computed, so it is not
an evaluation. Every proposal whose cost is computed is one evaluation, accepted or not; the
starting solution is one evaluation. Start: 13 random centers, every other point (random
order) to the nearest center with room.

**Incremental cost.** transfer: one distance difference; swap: two; recenter: the distance
changes over the cluster plus 2 (D_j - D_k). Checked against `evaluate_A` on 1000 random
moves in `tests/test_sa_i.py`.

**Temperatures.** Rule of Section 4 in `sa.rule_params` (walk of 3000 proposals, tuning seed
100). Nelder-Mead tuning in `experiments/tuning.py` with full inner budget 100000 (no reduction),
at most 80 objective calls. The set with the lower mean on seeds 100-104 is used for 0-9.

**Baseline.** `src/meta/random_search_sa.py`: independent draws (13 random centers, every
other point to a uniformly random center), best feasible kept, budget 100000, seeds 0-9;
rows with `configuration = random` in `results/Kaniyeva_tuning.csv`.

### Commands (in this order)

```bash
pytest tests/test_sa_i.py tests/test_sa_t.py
PYTHONPATH=. python experiments/tuning.py           # Zhamilya, about 2 minutes
PYTHONPATH=. python experiments/sa_params.py        # Amir: rule vs tuned on seeds 100-104
PYTHONPATH=. python experiments/run_final.py --module src.meta.sa \
    --instance-seed 3 --story A --variant 1 --team T03 --role SA-I \
    --configuration "SA, transfer+swap" --seeds 0-9 --budget 100000 \
    --params '<the line printed by sa_params.py>' --out results/Amanbay_runs.csv
PYTHONPATH=. python experiments/sa_analysis.py      # Table 1, Table 2, F1, acceptance curve
```

Table 1 and Figure F1 of both partners: `results/Amanbay_runs.csv` -> `results/Amanbay_table1.csv`,
`figures/F1_sa_bestsofar.png`. Baseline: `results/Kaniyeva_tuning.csv` (configuration `random`).

## Git rules for this checkpoint

- Commit tags: `[SA]`, `[GA]`, `[NM]`, `[Exp]`, `[Report]`.
- `git config user.name` = your roster name, `user.email` = an address on your GitHub account.
- Each file of your role is authored by you; small commits; no history rewriting.
- Final commit: `git tag checkpoint4 && git push --tags`. Commits after the deadline are ignored.

## Declarations

### Ikhsanova Aziza (GA-A)

| Tool or source | Used for | Files or sections | What I checked myself |
|---|---|---|---|
|  |  | `src/meta/ga.py`, `src/meta/random_search.py`, `experiments/exp_diversity_popsize.py`, `experiments/analyse.py`, `tests/test_ga_a.py` | ran the whole test file; checked the budget by hand on seven budgets including 1, 7 and 137; confirmed elitism by checking that the best fitness of the population never rises across 3449 generations; reproduced every cost in `Ikhsanova_runs.csv` by rerunning `solve` with the same seeds; compared the reported cost and feasibility against `evaluate_A` of the generator |
| numpy, matplotlib | arrays and figures | all of the above | versions pinned in `requirements.txt` |

### Abdrashit Yersultan (GA-P)

| Tool or source | Used for | Files or sections | What I checked myself |
|---|---|---|---|
| numpy, matplotlib | arrays, figure F3, data analysis | src/meta/ga_problem.py, tests/test_ga_p.py, experiments/exp_rho.py, results/Abdrashit_exp.csv | ran the GA_P tests (all 14 passed), ran exp_rho.py and confirmed the output matches results/Abdrashit_exp.csv, checked the 10 columns and evals = 100000 |
| AI assistant | draft of the code, help with the report text | src/meta/ga_problem.py, tests/test_ga_p.py, experiments/exp_rho.py, Abdrashit_CP4.pdf | read and understood the code, chose rho = 1 from my own runs on seeds 100-104, checked every number in the report against the CSV files |

### Amanbay Amir (SA-I)

| Tool or source | Used for | Files or sections | What I checked myself |
| Amir Amanbay | AI | drafting the SA, the moves, the incremental cost, the test and the analysis scripts | `src/meta/sa.py`, `tests/test_sa_i.py`, `experiments/sa_params.py`, `experiments/sa_analysis.py` | read the code; made a predict; run the tests; reran `experiments/sa_analysis.py`, which reruns every seed and asserts the same cost as in `results/Amanbay_runs.csv`; checked that the CSV has 10 rows with evals = 100000;|

### Kaniyeva Zhamilya (SA-T)

| Tool or source | Used for | Files or sections | What I checked myself |
|---|---|---|---|
| numpy, matplotlib | figures, data analysis, numerical tasks | src/meta/nm.py, src/meta/random_search_sa.py, tests/test_sa_t.py, experiments/tuning.py, results/Kaniyeva_tuning.csv | ran the SA_T tests (all 4 of them passed), executed tuning experiment and confirmed the output, handled Nelder-Mead parameter updates, objective evaluation, stopping conditions, tuning the results |
| AI assistant | code draft and debugging, report redaction | src/meta/nm.py, src/meta/random_search_sa.py, tests/test_sa_t.py, experiments/tuning.py, Kaniyeva_CP4.pdf | SA-T implementation, tuning experiment, analysis, code and results validation and verification|

