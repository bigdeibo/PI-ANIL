# Self-collected FT-NIR mixture data (pentane / CCl4)

`processed/pentane_ccl4_FTIR.npz` — 79 FT-NIR absorbance spectra
(ABB MB3600, 800–2500 nm, 17631 points per spectrum) of n-pentane diluted in
carbon tetrachloride. The prediction target is the n-pentane volume fraction
with values {0.0, 0.12, 0.16, 0.20, 0.24, 0.28, 0.32, 0.36, 0.40, 1.0}. Each
mixture was measured in replicate (3 repeats at 25/35/50 °C); `sample_id`
carries the physical-sample (bottle) identifier used for group-aware
evaluation splits.

Contents of the npz:

- `X` (79, 17631) float32 — absorbance spectra, columns ascending in nm
- `wavelength_nm` (17631,) float32 — nm grid
- `wavenumber` (17631,) float32 — cm^-1 grid
- `y` (79,) float32 — n-pentane volume fraction (prepared parts/250)
- `sample_id` (79,) — bottle identifier (group-aware split key)
- `rep` (79,) int32, `temp_c` (79,) int32 — replicate index, temperature
- `kind`, `name`, `source_file`, `entry` — provenance labels
- `parts_pentane`, `parts_ccl4` — prepared recipe parts
- `meta` — JSON string with dataset metadata

The loader `src/datasets.py::load_selfmix` shows the intended usage; the
analysis appears in `scripts/probe_within_instrument.py` and is cited in the
Discussion as the within-instrument boundary probe.

License: CC BY 4.0 (this work).
