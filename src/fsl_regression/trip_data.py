"""TRIP benchmark data loading — Garzón 2026 EAAI head-to-head (Option B: in-house unified implementation).

Reads data/spectroscopy-fewshot-benchmark-main/data/TRIP/{task}/{X_supp,X_query,y_supp,y_query}.csv
+ splits.csv. Each task's spectra are **interpolated to 512 points + SG first
derivative** (Garzón-style preprocessing; SG is per-sample and stateless ->
leakage-free), returning separate support/query pools (Garzón
MixedTask.sample_fixed semantics: the support set draws K from the X_supp pool;
the query evaluates the full X_query pool).

Task pools are defined by splits.csv: TRIP_TRAIN(11) / TRIP_VAL(6) / TRIP_TEST(7).
TRIP task directory names carry trailing-underscore padding (e.g. Corn_Starch__),
while splits.csv uses clean names -> _resolve_dir matches on rstrip("_").

Isomorphic to src/fsl_regression/core.py (L=512, EMB_DIM=128, BASE_SEED=42,
eval_split seed formula), so all meta-learning/evaluation code can be reused.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.preprocess import sg_deriv  # noqa: E402

L = 512
EMB_DIM = 128
BASE_SEED = 42

BENCH = ROOT / "data" / "spectroscopy-fewshot-benchmark-main"
TRIP_DIR = BENCH / "data" / "TRIP"


def _load_splits():
    """splits.csv: ,task,split -> {clean_task_name: split}."""
    out = {}
    with open(TRIP_DIR / "splits.csv") as f:
        for row in csv.DictReader(f):
            out[row["task"].strip()] = row["split"].strip()
    return out


SPLITS = _load_splits()
TRIP_TRAIN = [t for t, s in SPLITS.items() if s == "train"]
TRIP_VAL = [t for t, s in SPLITS.items() if s == "val"]
TRIP_TEST = [t for t, s in SPLITS.items() if s == "test"]
TRIP_TASKS = list(SPLITS.keys())

assert len(TRIP_TRAIN) == 11, f"expected 11 train tasks, got {len(TRIP_TRAIN)}"
assert len(TRIP_VAL) == 6, f"expected 6 val tasks, got {len(TRIP_VAL)}"
assert len(TRIP_TEST) == 7, f"expected 7 test tasks, got {len(TRIP_TEST)}"

# reverse-lookup cache: directory name -> clean task name
_DIR_INDEX: dict[str, str] | None = None


def _build_dir_index():
    """clean_name -> Path (handles trailing-underscore padding)."""
    global _DIR_INDEX
    if _DIR_INDEX is not None:
        return _DIR_INDEX
    idx = {}
    for d in TRIP_DIR.iterdir():
        if not d.is_dir():
            continue
        idx[d.name.rstrip("_")] = d
    _DIR_INDEX = idx
    return idx


def _resolve_dir(task: str) -> Path:
    idx = _build_dir_index()
    if task in idx:
        return idx[task]
    # fallback: case/whitespace-insensitive
    for k, p in idx.items():
        if k.strip().lower() == task.strip().lower():
            return p
    raise KeyError(f"TRIP task dir not found for '{task}'")


_cache: dict = {}


def _interp_to_L(M: np.ndarray) -> np.ndarray:
    """Position-normalized linear interpolation to L=512 points (isomorphic to core.load_task)."""
    go = np.linspace(0, 1, M.shape[1])
    gn = np.linspace(0, 1, L)
    return np.stack([np.interp(gn, go, r) for r in M])


def load_trip_task(task: str) -> dict:
    """Return dict:
      X_supp (n_s,512) float32 SG1 spectra; X_query (n_q,512) float32 SG1 spectra;
      y_supp (n_s,) float64 raw scale; y_query (n_q,) float64 raw scale;
      wavelengths (n_native,) float64 original wavelength axis (for band-constraint mapping);
      split str; name str.
    SG1 is per-sample and stateless -> leakage-free; target z-scoring is deferred
    to the episode using support-set statistics.
    """
    if task in _cache:
        return _cache[task]
    d = _resolve_dir(task)
    xs_df = pd.read_csv(d / "X_supp.csv", index_col=0)
    xq_df = pd.read_csv(d / "X_query.csv", index_col=0)
    ys_df = pd.read_csv(d / "y_supp.csv", index_col=0)
    yq_df = pd.read_csv(d / "y_query.csv", index_col=0)

    # wavelength axis (X column names); fall back to a synthetic linspace on parse failure
    try:
        wlv = np.asarray([float(c) for c in xs_df.columns], dtype=float)
    except ValueError:
        wlv = np.linspace(0, 1, xs_df.shape[1])

    Xs = _interp_to_L(xs_df.values.astype(float))
    Xq = _interp_to_L(xq_df.values.astype(float))
    Xs = sg_deriv(Xs, deriv=1, window=15, poly=2)
    Xq = sg_deriv(Xq, deriv=1, window=15, poly=2)

    ys = ys_df.values.astype(float).reshape(-1)
    yq = yq_df.values.astype(float).reshape(-1)

    out = dict(
        X_supp=Xs.astype(np.float32),
        X_query=Xq.astype(np.float32),
        y_supp=ys,
        y_query=yq,
        wavelengths=wlv,
        split=SPLITS[task],
        name=task,
    )
    _cache[task] = out
    return out


def load_trip_pool(tasks: list[str]) -> dict:
    """Aggregate multiple tasks into a meta-training pool: returns dict(X (N,512), y (N,), task_id (N,)).
    For episode sampling (stratified K+Q draws per task)."""
    Xs, ys, ti = [], [], []
    for i, t in enumerate(tasks):
        d = load_trip_task(t)
        # the meta-training pool uses the support pool (consistent with Garzón:
        # episodes are drawn from the support pool)
        Xs.append(d["X_supp"])
        ys.append(d["y_supp"])
        ti.append(np.full(len(d["y_supp"]), i))
    return dict(
        X=np.concatenate(Xs, axis=0),
        y=np.concatenate(ys, axis=0),
        task_id=np.concatenate(ti, axis=0),
        task_names=list(tasks),
    )


def trip_eval_support_query(task: str, K: int, rep: int, seed: int = BASE_SEED):
    """TRIP K-shot evaluation split (same seed formula as core.eval_split, bit-identical across methods).

    Returns (X_supp_K, y_supp_K, X_query_full, y_query_full, ym, ysd):
      draws K support spectra from the task's X_supp pool; the query uses the
      full X_query pool (stable evaluation); ym/ysd are computed from the
      support set (leakage-free z-score).
    """
    from src.fsl_regression.core import zscore_fit
    d = load_trip_task(task)
    n_s = len(d["y_supp"])
    rng = np.random.default_rng(seed * 1000 + K * 100 + rep)
    sup = rng.choice(n_s, K, replace=False)
    Xs = d["X_supp"][sup]
    ys = d["y_supp"][sup]
    Xq = d["X_query"]
    yq = d["y_query"]
    ym, ysd = zscore_fit(ys)
    return Xs, ys, Xq, yq, ym, ysd


if __name__ == "__main__":
    # self-check: print shapes of all 24 tasks + the test pool
    print(f"TRIP: train={len(TRIP_TRAIN)} val={len(TRIP_VAL)} test={len(TRIP_TEST)}")
    print("TRAIN:", TRIP_TRAIN)
    print("VAL  :", TRIP_VAL)
    print("TEST :", TRIP_TEST)
    for t in TRIP_TEST:
        d = load_trip_task(t)
        print(f"  {t:18s} X_supp{d['X_supp'].shape} X_query{d['X_query'].shape} "
              f"wl={d['wavelengths'][0]:.0f}..{d['wavelengths'][-1]:.0f}nm "
              f"y_supp[{d['y_supp'].min():.2f}..{d['y_supp'].max():.2f}]")
