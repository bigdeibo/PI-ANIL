#!/usr/bin/env bash
# W4 refinement queue (G3 fired 2026-09-16: K=5 max-min = 0.168 > 0.05).
# Sim-count response curve at the winning corner (lib=111, perturb=on),
# plus an intermediate library size.
set -u
cd "$(dirname "$0")/.."

run_recipe () {
  local corpus_tag="$1"; local suffix="$2"
  local corpus="data/pretrain_corpus_ablation/${corpus_tag}/spectra.npy"
  if [ ! -f "results/pretraining/checkpoints/encoder_mae${suffix}.pt" ]; then
    echo "=== train ${suffix} ==="
    python src/ssl/train.py --track mae --epochs 60 --suffix "${suffix}" \
      --scratch --device cuda --corpus "${corpus}"
  fi
  python scripts/run_ablation_probe.py --suffix "${suffix}" --task all --reps 10
}

run_recipe "s1500_l111_p1" "_s1500_l111_p1"
run_recipe "s3000_l111_p1" "_s3000_l111_p1"
run_recipe "s6000_l56_p1"  "_s6000_l56_p1"
echo "W4_REFINEMENT_DONE"
