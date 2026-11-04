"""Runs `solve` on your team's instance for several seeds and writes the results file.

Example (team T05, storyline A, variant 1, the pair's SA with transfer and swap, role SA-I):
  python experiments/run_final.py --module src.meta.sa --instance-seed 5 --story A --variant 1 \
      --team T05 --role SA-I --configuration "SA, transfer+swap" --out results/Surname_runs.csv

For team T11 add --cap 108. Use --seeds 0-9 and --budget 100000 for the final runs
(the defaults); use --seeds 100-104 for tuning runs. Parameters of solve: --params '{"T0": 12.0, "Tend": 0.05}'.
The checker is applied to every returned solution: the cost and the feasibility in the file are the checker's.
"""
import argparse, csv, importlib, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import checkpoint4_generator_PUBLIC as gen
from src.meta.interface import check

COLS = ['team', 'role', 'configuration', 'seed', 'budget', 'cost', 'evals', 'feasible', 'seconds', 'params']


def parse_seeds(text):
    if '-' in text:
        a, b = text.split('-'); return list(range(int(a), int(b) + 1))
    return [int(x) for x in text.split(',')]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--module', required=True, help='module that defines solve, for example src.meta.sa')
    ap.add_argument('--instance-seed', type=int, required=True)
    ap.add_argument('--story', required=True, choices=['A', 'B'])
    ap.add_argument('--variant', type=int, required=True, choices=[1, 2, 3])
    ap.add_argument('--cap', type=int, default=None, help='only if your task file says so')
    ap.add_argument('--team', required=True); ap.add_argument('--role', required=True)
    ap.add_argument('--configuration', required=True)
    ap.add_argument('--seeds', default='0-9'); ap.add_argument('--budget', type=int, default=100000)
    ap.add_argument('--params', default=None, help='JSON dictionary passed to solve as params')
    ap.add_argument('--out', required=True)
    ap.add_argument('--append', action='store_true')
    a = ap.parse_args()
    d = gen.cp34_instance(a.instance_seed, a.story, a.variant, cap=a.cap)
    solve = importlib.import_module(a.module).solve
    params = json.loads(a.params) if a.params else None
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    new = not (a.append and os.path.exists(a.out))
    with open(a.out, 'a' if a.append else 'w', newline='') as fh:
        w = csv.writer(fh)
        if new: w.writerow(COLS)
        for seed in parse_seeds(a.seeds):
            t0 = time.time()
            res = solve(d, a.variant, seed, a.budget, params=params, start=None)
            sec = time.time() - t0
            cost, problems = check(d, a.variant, res['solution'])
            feasible = len(problems) == 0
            notes = []
            if res.get('evals') != a.budget: notes.append(f"evals {res.get('evals')} != budget {a.budget}")
            if abs(float(res.get('cost', float('nan'))) - cost) > 1e-6 * max(1.0, abs(cost)): notes.append('reported cost differs from the checker')
            if bool(res.get('feasible')) != feasible: notes.append('reported feasible flag differs from the checker')
            for n in notes: print(f'  seed {seed}: WARNING: {n}')
            w.writerow([a.team, a.role, a.configuration, seed, a.budget, f'{cost:.9f}', res.get('evals'), feasible, f'{sec:.2f}',
                        json.dumps(params) if params else ''])
            fh.flush()
            print(f'seed {seed}: cost {cost:.6f}, feasible {feasible}, {sec:.1f} s')


if __name__ == '__main__':
    main()
