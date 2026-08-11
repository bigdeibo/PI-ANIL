# Checkpoints

**No trained checkpoints are bundled in this repository.** Every reported
number in the paper can be verified directly from the per-repetition CSVs under
`results/` (a few megabytes total) without loading any model weights, and every
checkpoint is reproducible from the code and data in this repository.

## Why no checkpoints

Trained weights are large (~260 MB for the 128-dimensional models, ~640 MB for
the 512-dimensional benchmark backbone) and are fully derived from the data and
code already released. Bundling them would inflate the repository without
adding verifiability, since the per-repetition CSVs already let a reader or
reviewer recompute every mean, effect size, and p value in the paper.

## Reproducing a checkpoint

All training is deterministic given the data and the fixed seeds.

- Self-supervised encoders (MAE / SimCLR), written to
  `results/pretraining/checkpoints/`:
  ```bash
  python src/ssl/train.py --track mae    --epochs 60
  python src/ssl/train.py --track simclr --epochs 60
  ```
- 13-task meta-learned models (ANIL / DKL / FOMAML / ProtoNet), written to
  `results/meta_routes/checkpoints/`:
  ```bash
  python src/fsl_regression/metatrain.py --method anil --init mae --source multi --episodes 2500
  ```
- Physics-informed (PI-ANIL) variants, written to `results/physics/checkpoints/`:
  ```bash
  python src/pifsl/metatrain.py --variant both --init mae --source multi
  ```
- TRIP benchmark models, written to `results/trip_benchmark/checkpoints/`:
  see `scripts/run_trip_exact.sh` (512-d) and `scripts/run_trip_full.sh` (128-d).

Each evaluation script (`scripts/run_*.py`) loads the checkpoint it needs, so
run the corresponding training command first, then the evaluation. The exact
commands behind every table and figure are listed in `docs/REPRODUCE.md`.

## Archive upon acceptance

A complete checkpoint archive (all 128-dimensional models and the
512-dimensional benchmark backbone) will be deposited on Zenodo under a
persistent DOI upon acceptance, for readers who want to skip retraining.
