# Pretraining corpus artifacts

- `sim_compositions.npz` — the 6000 simulated mixture spectra (`X`),
  their Beer–Lambert mixing compositions (`C`), and the component names
  (`lib_names`) of the 111-component reference library. CC BY 4.0.
- `sources.csv` — per-spectrum provenance labels for the full 9339-spectrum
  corpus.

The full corpus matrix `spectra.npy` (9339 x 512) is **not** bundled because
one third of it derives from third-party datasets that must not be
redistributed. Rebuild it deterministically (fixed seed 42) with:

```bash
python scripts/build_pretrain_corpus.py
```

after placing the input datasets (`docs/DATA_SOURCES.md`). The rebuild is
bit-identical (verified by SHA256).
