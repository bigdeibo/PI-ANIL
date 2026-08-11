# Paired split indices

These files make the paired evaluation protocol independently verifiable.
All methods in the paper were evaluated on exactly these partitions.

- `{task}_support.csv` — support-set row indices for each task,
  `K in {5, 10, 20}`, `rep in 0..29` (deep and meta-learning methods used
  reps 0–9; classical baselines used reps 0–29). `index` is the row position
  in that task's NaN-filtered array as returned by
  `src/fsl_regression/core.py::load_task`. Note that `diesel-CN` has n = 381
  while the other six diesel attributes have n = 395 (different NaN masks).
- `{task}_test.csv` — test-set row indices, exported only for the three
  group-aware tasks (evoo-adulteration, selfmix-phi, edibleoil-pv), where the
  test set excludes every replicate of the K support groups and is therefore
  not the plain complement of the support rows. For all other tasks,
  `test = setdiff(arange(n), support)`.
- `trip_support.csv` — support indices for the TRIP head-to-head testbed
  (7 held-out tasks, `K in {5, 10, 25}`, `rep in 0..9`); the query set is
  always the full query pool of the task.

Regeneration (requires the datasets, see docs/DATA_SOURCES.md):

```bash
python scripts/export_splits.py
```

The splits derive deterministically from the seed formula
`s = 42*1000 + K*100 + rep` implemented in `eval_split`
(`src/fsl_regression/core.py`) and `trip_eval_support_query`
(`src/fsl_regression/trip_data.py`).
