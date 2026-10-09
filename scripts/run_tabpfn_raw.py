"""TabPFN on RAW 512-point spectra (no MAE embeddings) — the raw-input control arm
for manuscript Section 3.5 (reviewer-simulation P0, 2026-10-09).

Same 13-task paired protocol as run_tabpfn_baseline.py (W1): seed formula
s = 42*1000 + K*100 + rep, rep 0..9, K in {5,10,20}, group-aware split for EVOO.
The only difference from the W1 arm: features are the raw SNV+resampled spectra
(load_task X) instead of the frozen MAE embeddings. Features are standardized by
support-set statistics exactly as in the W1 arm.

Weights: read-only from the phase-2 frozen area (SpecAI4ds .../models/).
Output: results/stage5/tabpfn_raw/parts/tabpfn-{v}__raw__{task}.csv (resumable).
"""
import sys
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.fsl_regression.core import TASKS, load_task, eval_split, zscore_fit, r2_score
from src.fewshot import rpd

# the 13 benchmark tasks: exclude the two phase-2 probe tasks
BENCH = [t for t in TASKS if t not in ("selfmix-phi", "edibleoil-pv")]

# read-only weights from the frozen phase-2 repo
MODEL_DIR = Path(r"D:\AI4Coding\SpectralAlgorithm\SpecAI4ds\release-pi\PI-ANIL\models")
MODELS = {"v2": MODEL_DIR / "tabpfn-v2-regressor.ckpt",
          "v2.5": MODEL_DIR / "tabpfn-v2.5-regressor-v2.5_default.ckpt",
          "v3": MODEL_DIR / "tabpfn-v3-regressor-v3_default.ckpt"}

OUT = ROOT / "results" / "stage5" / "tabpfn_raw"
PARTS = OUT / "parts"
PARTS.mkdir(parents=True, exist_ok=True)


def _write_rows(out_file, rows, retries=5):
    df = pd.DataFrame(rows)
    for attempt in range(retries):
        try:
            df.to_csv(out_file, mode="a", header=not out_file.exists(), index=False)
            return
        except PermissionError:
            time.sleep(2 * (attempt + 1))
    raise PermissionError(f"still locked after {retries} retries: {out_file}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="all")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--model", default="v3", choices=["v2", "v2.5", "v3"])
    args = ap.parse_args()
    tasks = BENCH if args.task == "all" else [args.task]

    from tabpfn import TabPFNRegressor  # deferred: heavy import
    import torch as _torch
    DEVICE = "cuda" if _torch.cuda.is_available() else "cpu"
    print(f"[tabpfn-raw] model={args.model} device={DEVICE}", flush=True)

    for task in tasks:
        d = load_task(task)
        X, y = np.asarray(d["X"], dtype=np.float64), d["y"]
        n = len(y)
        out_file = PARTS / f"tabpfn-{args.model}__raw__{task}.csv"
        done = set()
        if out_file.exists():
            prev = pd.read_csv(out_file)
            done = set(zip(prev.shot, prev.rep))
        rows = []
        for K in (5, 10, 20):
            if n <= K + 5:
                continue
            for rep in range(args.reps):
                if (K, rep) in done:
                    continue
                tr, te = eval_split(n, K, rep, groups=d.get("groups"))
                if len(te) == 0:
                    continue
                t0 = time.time()
                ym, ysd = zscore_fit(y[tr])
                # same feature handling as the W1 embedding arm:
                # standardize by support stats
                mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
                Xs = (X[tr] - mu) / sd
                Xq = (X[te] - mu) / sd
                ys_z = (y[tr] - ym) / ysd
                m = TabPFNRegressor(model_path=str(MODELS[args.model]),
                                    device=DEVICE,
                                    ignore_pretraining_limits=True)
                m.fit(Xs, ys_z)
                pred = m.predict(Xq) * ysd + ym
                yt = y[te]
                rmse = float(np.sqrt(np.mean((pred - yt) ** 2)))
                row = dict(task=task, shot=K, rep=rep,
                           r2=r2_score(pred, yt), rmse=rmse,
                           rpd=rpd(yt, rmse), secs=round(time.time() - t0, 2))
                rows.append(row)
                print(f"[raw:{args.model}] {task} K={K} rep={rep}: "
                      f"r2={row['r2']:.3f} ({row['secs']}s)", flush=True)
        if rows:
            _write_rows(out_file, rows)


if __name__ == "__main__":
    main()
