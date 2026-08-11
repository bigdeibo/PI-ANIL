#!/bin/bash
# TRIP head-to-head full run: train all configurations + evaluate test7 × {5,10,25} × 10 reps.
# Each step is independent (set +e; a single-step failure does not abort), log appended sequentially to _run_trip_full.log.
# Usage: bash scripts/run_trip_full.sh  (in background: run_in_background)
set +e
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
LOG=results/garzon_headtohead/_run_trip_full.log
mkdir -p results/garzon_headtohead
echo "=== TRIP full run start $(date) ===" > "$LOG"

EP=2500
run() { echo; echo ">>> $*" | tee -a "$LOG"; "$@" 2>&1 | tee -a "$LOG"; }

# ---------- Training: anil / proto × {rand,mae} ----------
for init in rand mae; do
  run python -m src.fsl_regression.trip_metatrain --method anil  --init $init --episodes $EP --time-budget 300 --seed 0
  run python -m src.fsl_regression.trip_metatrain --method proto --init $init --episodes $EP --time-budget 300 --seed 0
done
# ---------- Training: pi/both × {rand,mae} (band is slow, so give it a generous budget) ----------
for init in rand mae; do
  run python -m src.fsl_regression.trip_metatrain --method pi --init $init --variant both --episodes $EP --time-budget 600 --seed 0
done
# ---------- Training: maml/mae (second-order, train-mode BN for stability) ----------
run python -m src.fsl_regression.trip_metatrain --method maml --init mae --episodes 1500 --lr 1e-3 --time-budget 700 --seed 0
# ---------- FT pretraining × {rand,mae} ----------
run python -c "from src.fsl_regression.trip_metatrain import trip_pretrain_ft as f; [f(i, epochs=200, time_budget=400, seed=0) for i in ('rand','mae')]"

# ---------- Evaluation: test7 × {5,10,25} × 10 reps ----------
for init in rand mae; do
  run python scripts/run_trip_headtohead.py --method anil  --init $init --tasks test --shots 5,10,25 --reps 10
  run python scripts/run_trip_headtohead.py --method proto --init $init --tasks test --shots 5,10,25 --reps 10
  run python scripts/run_trip_headtohead.py --method pi    --init $init --variant both --tasks test --shots 5,10,25 --reps 10
  run python scripts/run_trip_headtohead.py --method ft    --init $init --tasks test --shots 5,10,25 --reps 10
done
run python scripts/run_trip_headtohead.py --method maml --init mae --tasks test --shots 5,10,25 --reps 10
run python scripts/run_trip_headtohead.py --method base --init rand --tasks test --shots 5,10,25 --reps 10

# ---------- Table merge ----------
run python scripts/merge_trip_headtohead.py
echo "=== TRIP full run DONE $(date) ===" | tee -a "$LOG"
