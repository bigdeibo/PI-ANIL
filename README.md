# PI-ANIL: Physics-informed meta-learning for few-shot near-infrared quantitation

Companion repository for the PI-ANIL manuscript
("Physics-informed meta-learning for few-shot near-infrared quantitation
across unseen substance families"). It contains the complete training and evaluation
pipeline, the paired train/test split indices, the simulated pretraining
corpus, and all per-repetition results behind every table and figure in the
paper. Trained checkpoints are not bundled; every number is verifiable from the
per-repetition CSVs, and every checkpoint is reproducible from the code
(see `docs/CHECKPOINTS.md`).

**Third-party datasets are not mirrored.** Automated scripts and per-dataset
instructions obtain each dataset from its original source under its original
terms; see [`docs/DATA_SOURCES.md`](docs/DATA_SOURCES.md).

## Layout

```
src/         model, meta-training, physics-constraint, and evaluation library
scripts/     one driver script per experiment, plus table-merging scripts
splits/      paired split indices (CSV) for bit-identical reproduction
data/        bundled license-clean data (GPL-2 pls arrays; our own FT-NIR set;
             simulated corpus compositions); third-party data goes here too
results/     per-repetition CSVs and merged tables behind every reported
             number, organized by experiment (no checkpoints; see docs/CHECKPOINTS.md)
docs/        data sources, checkpoints, reproduction guide
```

`results/` is organized by the experiments in the paper:

```
results/baselines/             classical + deep baselines, variable selection (Sec. 3.1, 3.4)
results/pretraining/           SSL pretraining probes and encoders (Sec. 2.2, 3.1)
results/meta_routes/           ANIL / DKL / FOMAML route comparison (Sec. 3.1)
results/physics/               physics-informed variants, lambda sweep, LOMO (Sec. 3.2-3.3)
results/protonet_headtohead/   ProtoNet regression head-to-head (Sec. 3.4)
results/trip_benchmark/        TRIP benchmark head-to-head (Sec. 3.4)
results/fingerprint/           band localization on MLNIR and PHC (Sec. 3.5)
results/probes/                boundary probes cited in the Discussion
results/uq/tabpfn/             TabPFN foundation-head comparison on the frozen
                               embeddings (Sec. 3.5)
results/pretraining/ablation/  pretraining-corpus recipe ablation (SI S4)
```

## Quick start

```bash
pip install -r requirements.txt
# 1. obtain the public datasets (scripts + instructions)
python scripts/fetch_nist_ir.py            # NIST IR reference spectra
#    ... then place the main-benchmark datasets as in docs/DATA_SOURCES.md
# 2. rebuild the pretraining corpus (deterministic, fixed seeds)
python scripts/build_pretrain_corpus.py
# 3. reproduce a headline result, e.g. the PI-ANIL main ablation
python scripts/run_physics_variants.py --help
```

The exact command behind every table and figure is listed in
[`docs/REPRODUCE.md`](docs/REPRODUCE.md). Every number in the paper can also be
verified directly from the per-repetition CSVs under `results/` without
re-running any training.

## Paired evaluation protocol

All methods share bit-identical splits through the seed formula
`s = seed*1000 + K*100 + rep` with `seed = 42` (`src/fsl_regression/core.py`).
Deep and meta-learning methods use `rep = 0..9`; classical baselines use
`rep = 0..29`. The olive-oil task (and the two bundled replicate-structured
sets) are split group-aware by physical sample identifier to prevent replicate
spectra from crossing the support/test boundary. The exported indices under
`splits/` allow independent verification of this pairing.

## License

Code is released under the MIT License (`LICENSE`). The simulated corpus, split
indices, and result tables are released under CC BY 4.0 (`LICENSE-DATA`).
Third-party datasets remain under their original licenses; this repository only
automates their retrieval.
