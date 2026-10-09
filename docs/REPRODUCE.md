# Reproduction guide

Every table and figure in the paper traces to the commands below. Two paths:

- **Verify the numbers** — every mean, effect size, and p value in the paper
  can be recomputed directly from the per-repetition CSVs under `results/`
  (a few megabytes), with no training and no checkpoints. The `scripts/merge_*.py`
  scripts read exactly these CSVs and produce the summary tables.
- **Re-run an evaluation or training** — no trained checkpoints are bundled
  (see `docs/CHECKPOINTS.md`); run the training command in the relevant section
  first to produce the checkpoint, then the evaluation script.

All commands run from the repository root. Datasets must be in place first
(`docs/DATA_SOURCES.md`). Splits are paired and deterministic
(`splits/README.md`).

## 0. Data and corpus

```bash
python scripts/fetch_nist_ir.py              # NIST IR reference spectra
python scripts/build_pretrain_corpus.py      # corpus, deterministic (seed 42)
python scripts/export_splits.py              # regenerate splits/ (optional check)
```

## 1. Self-supervised pretraining (Sec. 2.2, 3.1; Fig. 3)

```bash
python src/ssl/train.py --track mae    --epochs 60            # encoder_mae.pt
python src/ssl/train.py --track simclr --epochs 60            # equal-budget control
```

Probe evaluation (frozen encoder + ridge, 30 reps):

```bash
python scripts/run_pretraining_probe.py --model mae_ridge --task all --reps 30
python scripts/merge_pretraining.py
python scripts/merge_equal_budget_simclr.py
```

Results: `results/pretraining/probe_results.csv`.

## 2. Classical and deep baselines (Sec. 3.1; Fig. 2a)

```bash
python scripts/run_fullcv_baselines.py                        # full-CV anchor
python scripts/run_fewshot_baselines.py --task diesel-CN --model PLS --reps 30
#   --model in {PLS, SVR, CNN, CNN+Aug}; --task per 13-task list
python scripts/merge_baselines.py
```

Variable-selection strong baselines (Sec. 3.4; Fig. 6a):

```bash
python scripts/run_variable_selection.py --model CARSPLS --task all --reps 30
python scripts/run_variable_selection.py --model UVEPLS  --task all --reps 30
python scripts/merge_variable_selection.py
```

## 3. Meta-learning routes (Sec. 3.1; Table 2)

Meta-training (writes checkpoints under `results/meta_routes/checkpoints/`):

```bash
python src/fsl_regression/metatrain.py --method anil --init mae --source multi --episodes 2500
#   --method in {anil, dkl, fomaml, proto}; --init in {rand, mae}
```

Evaluation (10 reps, K = 5/10/20):

```bash
python scripts/run_meta_routes.py --route anil --init mae --task diesel-CN
python scripts/merge_meta_routes.py
```

Results: `results/meta_routes/per_rep.csv`.

## 4. Physics-informed variants (Sec. 3.2-3.3; Table 3, Fig. 4)

Meta-training with the physical constraints:

```bash
python src/pifsl/metatrain.py --variant both --init mae --source multi
#   --variant in {vanilla, add, band, both}; --lam-add/--lam-band for the
#   lambda sweep (0.01 / 0.1 / 1.0); --source controls the LOMO pool
```

Evaluation and LOMO:

```bash
python scripts/run_physics_variants.py --tag anilpi_both_mae_multi --task diesel-CN
python scripts/merge_physics.py
python scripts/merge_lambda_lomo.py
```

Mechanism analyses:

```bash
python scripts/run_linearity_tracking.py     # embedding-linearity tracking
python scripts/run_mechanism_audit.py        # saliency / additivity audit
python scripts/make_figures.py               # paper figures
```

Results: `results/physics/per_rep.csv`.

## 5. ProtoNet head-to-head (Sec. 3.4)

```bash
python src/fsl_regression/metatrain.py --method proto --init mae --source multi
python scripts/run_protonet_headtohead.py --mode meta   --task all --reps 10
python scripts/run_protonet_headtohead.py --mode frozen --task all --reps 10
python scripts/merge_protonet_headtohead.py
```

Results: `results/protonet_headtohead/paired_per_rep.csv`.

## 6. TRIP head-to-head with the concurrent benchmark (Sec. 3.4; Table 4, Fig. 5b)

Requires the benchmark repository under
`data/spectroscopy-fewshot-benchmark-main/` (`docs/DATA_SOURCES.md`).

```bash
# 512-dimensional backbone (architecture-matched, main report)
python src/fsl_regression/trip_metatrain.py --method pi --init mae \
    --arch resnet1d_garzon --emb-dim 512 --device cuda
python scripts/run_trip_headtohead.py --method pi --init mae \
    --arch resnet1d_garzon --emb-dim 512 --tag-suffix __exact \
    --parts-dir trip_exact_parts
python scripts/merge_trip_exact.py
# 128-dimensional backbone (secondary): scripts/run_trip_full.sh
```

The exact driver scripts are `scripts/run_trip_exact.sh` (512-d) and
`scripts/run_trip_full.sh` (128-d). Results:
`results/trip_benchmark/trip_exact_all.csv`,
`results/trip_benchmark/trip_headtohead_all.csv`.

## 7. Band-localization analyses (Sec. 3.5; Fig. 7)

```bash
python scripts/band_localization_mlnir.py    # hydrocarbon mixtures, density
python scripts/band_localization_phc.py      # crude oils, API gravity
```

Results: `results/fingerprint/mlnir/`, `results/fingerprint/phc/`.

## 8. Boundary probes (cited in the Discussion)

```bash
python scripts/probe_oxidation_property.py          # edible-oil peroxide value null
python scripts/probe_corn_instruments.py            # corn m5/mp5/mp6 transfer
python scripts/probe_corn_instruments_optionB.py
python scripts/probe_mango_instruments.py           # multi-instrument mango
python scripts/probe_within_instrument.py           # bundled pentane/CCl4 data
python scripts/probe_within_instrument_harden.py    # 100-rep x 3-seed hardening
```

Results: `results/probes/`.

## 9. TabPFN foundation-head comparison (Sec. 3.5)

TabPFN v2/v2.5/v3 weights are downloaded separately (see docs/CHECKPOINTS.md);
all three run on the frozen MAE embeddings under the paired splits.

```bash
python scripts/run_tabpfn_baseline.py --model v2    # plus v2.5, v3
python scripts/run_tabpfn_raw.py --model v3         # raw-spectra arm (no embeddings)
```

Results: `results/uq/tabpfn/parts/` (on the frozen embeddings) and
`results/uq/tabpfn/parts_raw/` (raw 512-point spectra). The EVOO raw-arm CSV
shares the bit-identical paired protocol with the representation-comparison run
of the companion training-free study; all other raw-arm CSVs were produced by
`run_tabpfn_raw.py` in this repository.

## 10. Pretraining-corpus recipe ablation (SI S4)

```bash
python scripts/build_pretrain_corpus_ablation.py    # build the recipe corpora
python scripts/run_ablation_probe.py                # frozen-encoder ridge probe
```

Results: `results/pretraining/ablation/`.

## Statistical testing

All headline comparisons use the paired Wilcoxon signed-rank test on
rep-level differences plus a task-level sign test, exactly as implemented in
the `merge_*.py` scripts; no correction for multiplicity is applied and exact
p values are reported.
