#!/usr/bin/env bash
# W4 recipe ablation training+probe queue (design doc §5.2, G3 gate).
# Sequential GPU usage. Reference recipe (s6000_l111_p1) is already trained
# (encoder_mae.pt); only its probe (shared code path) is run here.
set -u
cd "$(dirname "$0")/.."

run_recipe () {
  local corpus_tag="$1"   # directory under data/pretrain_corpus_ablation
  local suffix="$2"       # checkpoint suffix after encoder_mae
  local corpus="data/pretrain_corpus_ablation/${corpus_tag}/spectra.npy"
  if [ ! -f "results/pretraining/checkpoints/encoder_mae${suffix}.pt" ]; then
    echo "=== train ${suffix} on ${corpus} ==="
    python src/ssl/train.py --track mae --epochs 60 --suffix "${suffix}" \
      --scratch --device cuda --corpus "${corpus}"
  else
    echo "=== skip train ${suffix} (checkpoint exists) ==="
  fi
  python scripts/run_ablation_probe.py --suffix "${suffix}" --task all --reps 10
}

# reference recipe probe (shared code path; checkpoint already exists)
python scripts/run_ablation_probe.py --suffix "" --task all --reps 10

run_recipe "s0_l28_p0"    "_s0"
run_recipe "s6000_l28_p0" "_s6000_l28_p0"
run_recipe "s6000_l28_p1" "_s6000_l28_p1"
run_recipe "s6000_l111_p0" "_s6000_l111_p0"

echo "W4_QUEUE_DONE"
