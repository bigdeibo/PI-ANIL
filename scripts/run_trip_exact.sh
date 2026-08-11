#!/bin/bash
# G-P1b-exact Phase 1 full run: 512-dim Garzón backbone + GPU + meta-batch=25 + __exact isolation.
# ProtoNet was already trained in Phase 0 (meta-batch=1, aligned with Garzón ProtoNet semantics); only evaluation here.
# ANIL/PI/MAML: 50000ep (aligned with Garzón MAML) + meta-batch=25 (25 tasks per update) + early stopping patience 8.
# Artifacts are written incrementally to trip_exact_parts/ (does not overwrite the legacy 128-dim trip_parts/), deduplicated by (tag,task,shot,rep) for resumability.
set +e
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=0
export PYTHONIOENCODING=utf-8
LOG=results/garzon_headtohead/_run_trip_exact.log
mkdir -p results/garzon_headtohead
echo "=== TRIP exact Phase1 start $(date) ===" > "$LOG"
A="--arch resnet1d_garzon --emb-dim 512 --device cuda --tag-suffix __exact"
E="--device cuda --arch resnet1d_garzon --emb-dim 512 --tag-suffix __exact --parts-dir trip_exact_parts"
MAML_LRIN=0.001  # MAML inner-loop lr: Garzón uses 0.1, but for the 512-dim full-parameter second-order 5-step setting—0.1 skips every episode (>50), 0.01 still skips many (loss 50-644), 0.001 is stable (smoke: upd8 loss1.09, val2.80 finite). Second-order MAML ~1s/ep; time_budget 2400 trains ~2400ep/96 updates, somewhat undertrained but honestly reportable
run() { echo; echo ">>> $*" | tee -a "$LOG"; "$@" 2>&1 | tee -a "$LOG"; }

# ---------- Training ----------
# ANIL: meta-batch 25, lr_in 0.001 (512-dim head, to prevent divergence at lr_in=0.01)
for init in rand mae; do
  run python -m src.fsl_regression.trip_metatrain --method anil --init $init --variant both \
    --episodes 15000 --meta-batch 25 --lr 1e-3 --lr-in 0.001 --inner-steps 5 \
    --val-every 500 --patience 5 --time-budget 1800 $A --seed 0
done
# PI-ANIL: meta-batch 25, variant both, lr_in 0.001
for init in rand mae; do
  run python -m src.fsl_regression.trip_metatrain --method pi --init $init --variant both \
    --episodes 15000 --meta-batch 25 --lr 1e-3 --lr-in 0.001 --inner-steps 5 \
    --val-every 500 --patience 5 --time-budget 2400 $A --seed 0
done
# MAML: meta-batch 25, lr_in 0.1 (Garzón), second-order functional_call + train-mode BN + clip guardrail
for init in rand mae; do
  run python -m src.fsl_regression.trip_metatrain --method maml --init $init \
    --episodes 50000 --meta-batch 25 --lr 1e-3 --lr-in $MAML_LRIN --inner-steps 5 \
    --val-every 500 --patience 5 --time-budget 2400 $A --seed 0
done
# FT pretraining × {rand,mae} (pretraining lr 1e-4; note: not Garzón's finetune lr 0.09, different stage semantics, FT is a weak method)
run python -c "import sys; sys.path.insert(0,'.'); from src.fsl_regression.trip_metatrain import trip_pretrain_ft as f; [f(i, epochs=500, time_budget=1800, seed=0, arch='resnet1d_garzon', emb_dim=512, device='cuda', tag_suffix='__exact') for i in ('rand','mae')]"

# ---------- Evaluation test7 × {5,10,25} × 10rep ----------
for init in rand mae; do
  run python scripts/run_trip_headtohead.py --method anil  --init $init --variant both --tasks test --shots 5,10,25 --reps 10 $E
  run python scripts/run_trip_headtohead.py --method pi     --init $init --variant both --tasks test --shots 5,10,25 --reps 10 $E
  run python scripts/run_trip_headtohead.py --method proto  --init $init             --tasks test --shots 5,10,25 --reps 10 $E
  run python scripts/run_trip_headtohead.py --method maml   --init $init             --tasks test --shots 5,10,25 --reps 10 --lr-in $MAML_LRIN $E
  run python scripts/run_trip_headtohead.py --method ft     --init $init             --tasks test --shots 5,10,25 --reps 10 $E
done
run python scripts/run_trip_headtohead.py --method base --init rand --tasks test --shots 5,10,25 --reps 10 $E

# ---------- Table merge ----------
run python scripts/merge_trip_exact.py
echo "=== TRIP exact Phase1 DONE $(date) ===" | tee -a "$LOG"
