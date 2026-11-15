#!/usr/bin/env bash
# Two extra runs of Block 3, after run_block3_methods.sh:
#  (a) a second run of the exact solver on every size, with nothing else running, to see how much the
#      time and the incumbent move between runs (both runs stay in results/final_exact.csv);
#  (b) a check of the penalty weight: the GA with rho = 10 instead of 1 on N = 100 (own start), because
#      with rho = 1 no GA run on N = 100 ends feasible. This is not a Checkpoint 4 parameter and is
#      reported only as a check, outside Table 3.
set -e
python final/src/exact_scaled.py --sizes 35,60,100,150
python experiments/run_scaling.py --module src.meta.ga --story A --variant 1 --size 100 --team T03 \
  --configuration "GA rho=10, own start (check)" --params '{"rho": 10}' --out results/final_scaling_check.csv --append
