"""Block 3 of the Final: run `solve` of your team on a larger instance for several seeds and write the results file.

Example (storyline A, variant 2, size 100, SA of the team, start from your own start):
  python experiments/run_scaling.py --module src.meta.sa --story A --variant 2 --size 100 \
      --team T05 --configuration "SA, own start" --out results/scaling_runs.csv --append

Run from the LP-rounded start: put your rounded solution into a JSON file and add  --start-file rounded_A_100.json
  storyline A: {"centers": [...], "assign": [...], "vehicle": [...] or null}      storyline B: {"assign": [...]}
Sizes: storyline A  60, 100, 150;  storyline B  100, 200, 400.  Seeds 0-9 and budget 100000 as in Checkpoint 4,
the same parameters that you used in Checkpoint 4 (say so in the report).
"""
import argparse, csv, importlib, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import checkpoint4_generator_PUBLIC as gen
from src.meta.interface import check

COLS = ['team', 'configuration', 'story', 'variant', 'size', 'seed', 'budget', 'cost', 'evals', 'feasible', 'seconds', 'start', 'params']


def parse_seeds(text):
    if '-' in text:
        a, b = text.split('-'); return list(range(int(a), int(b) + 1))
    return [int(x) for x in text.split(',')]


def load_start(path, story):
    if not path:
        return None
    j = json.load(open(path))
    if story == 'A':
        return (list(j['centers']), list(j['assign']), j.get('vehicle'))
    return list(j['assign'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--module', required=True)
    ap.add_argument('--story', required=True, choices=['A', 'B'])
    ap.add_argument('--variant', type=int, required=True, choices=[1, 2, 3])
    ap.add_argument('--size', type=int, required=True)
    ap.add_argument('--team', required=True); ap.add_argument('--configuration', required=True)
    ap.add_argument('--seeds', default='0-9'); ap.add_argument('--budget', type=int, default=100000)
    ap.add_argument('--params', default=None); ap.add_argument('--start-file', default=None)
    ap.add_argument('--out', required=True); ap.add_argument('--append', action='store_true')
    a = ap.parse_args()
    d = gen.scaled_instance(a.story, a.size, a.variant)
    solve = importlib.import_module(a.module).solve
    params = json.loads(a.params) if a.params else None
    start = load_start(a.start_file, a.story)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    new = not (a.append and os.path.exists(a.out))
    with open(a.out, 'a' if a.append else 'w', newline='') as fh:
        w = csv.writer(fh)
        if new: w.writerow(COLS)
        for seed in parse_seeds(a.seeds):
            t0 = time.time()
            res = solve(d, a.variant, seed, a.budget, params=params, start=start)
            sec = time.time() - t0
            cost, problems = check(d, a.variant, res['solution'])
            feasible = len(problems) == 0
            w.writerow([a.team, a.configuration, a.story, a.variant, a.size, seed, a.budget, f'{cost:.9f}', res.get('evals'),
                        feasible, f'{sec:.2f}', 'file' if start is not None else 'own', json.dumps(params) if params else ''])
            fh.flush()
            print(f'size {a.size} seed {seed}: cost {cost:.6f}, feasible {feasible}, {sec:.1f} s')


if __name__ == '__main__':
    main()
