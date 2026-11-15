#!/usr/bin/env bash
# Block 3: the two methods of the team on N = 60, 100, 150, seeds 0-9, budget 100000,
# with the parameters of Checkpoint 4 (GA: rho = 1 and the defaults of src/meta/ga.py;
# SA: the tuned T0, Tend of results/Amanbay_chosen_params.json), from the own start and
# from the rounded LP start (final/src/lp_round.py). Run from the repository root.
set -e
SA='{"T0": 6.892341137379357, "Tend": 3.0081420522057414}'
for N in 60 100 150; do
  python experiments/run_scaling.py --module src.meta.sa --story A --variant 1 --size $N --team T03 \
    --configuration "SA, own start" --params "$SA" --out results/final_scaling_runs.csv --append
  python experiments/run_scaling.py --module src.meta.sa --story A --variant 1 --size $N --team T03 \
    --configuration "SA, LP-rounded start" --params "$SA" --start-file results/final_rounded_A_$N.json \
    --out results/final_scaling_runs.csv --append
  python experiments/run_scaling.py --module src.meta.ga --story A --variant 1 --size $N --team T03 \
    --configuration "GA, own start" --out results/final_scaling_runs.csv --append
  python experiments/run_scaling.py --module src.meta.ga --story A --variant 1 --size $N --team T03 \
    --configuration "GA, LP-rounded start" --start-file results/final_rounded_A_$N.json \
    --out results/final_scaling_runs.csv --append
done
