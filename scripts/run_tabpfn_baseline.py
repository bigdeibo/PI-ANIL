"""Phase-2 W1 defense baseline: training-free in-context regression (TabPFN) on frozen MAE embeddings.

Design doc: PI-ANIL二期研究设计文档.md §5.1. Answers "is meta-training necessary?" by running
TabPFN directly on the cached 128-d MAE embeddings under the *same* paired splits
(s = 42*1000 + K*100 + rep, rep 0..9) as every other method.

Embedding cache: results/pretraining/emb_cache/mae__{task}.npy (raw encoder embeddings,
NOT support-standardized). Covers 11 of 13 tasks; corn_m5-moisture / corn_m5-starch are
regenerated below with encoder_mae.pt if absent (seconds each).

Output: results/uq/tabpfn/parts/tabpfn__{backbone}__{task}.csv
        one row per (task, shot, rep) with r2/rmse/rpd, resumable.
"""
import sys
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.fsl_regression.core import (TASKS, load_task, eval_split, zscore_fit,
                                     r2_score, EMB_DIM, L)
from src.fewshot import rpd

OUT = ROOT / "results" / "uq" / "tabpfn"
PARTS = OUT / "parts"
PARTS.mkdir(parents=True, exist_ok=True)
EMB_CACHE = ROOT / "results" / "pretraining" / "emb_cache"


def get_embeddings(task, backbone="mae"):
    """Cached embeddings, else compute with the released encoder checkpoint."""
    f = EMB_CACHE / f"{backbone}__{task}.npy"
    if f.exists():
        return np.load(f)
    import torch
    from src.fsl.models import ResNet1Encoder
    ck = torch.load(ROOT / "results" / "pretraining" / "checkpoints"
                    / f"encoder_{backbone}.pt", map_location="cpu",
                    weights_only=False)
    enc = ResNet1Encoder(L, emb_dim=EMB_DIM)
    enc.load_state_dict(ck["enc"] if "enc" in ck else ck)
    enc.eval()
    d = load_task(task)
    X = torch.from_numpy(d["X"])[:, None]
    with torch.no_grad():
        Z = enc(X).numpy()
    np.save(f, Z)
    return Z


def _write_rows(out_file, rows, retries=5):
    """Write a task's rows in one go, with backoff against transient file locks
    (AV real-time scanning has caused Errno 13 on append-per-row writes)."""
    import time as _t
    df = pd.DataFrame(rows)
    for attempt in range(retries):
        try:
            df.to_csv(out_file, mode="a", header=not out_file.exists(),
                      index=False)
            return
        except PermissionError:
            _t.sleep(2 * (attempt + 1))
    raise PermissionError(f"still locked after {retries} retries: {out_file}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(TASKS) + ["all"])
    ap.add_argument("--backbone", default="mae",
                    choices=["mae", "rand", "simclr"])
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--model", default="v2", choices=["v2", "v2.5", "v3"],
                    help="v2 = Nature 2025 (GCS mirror copy); v2.5 / v3 = HF gated, "
                         "downloaded with user token (models/)")
    args = ap.parse_args()
    tasks = list(TASKS) if args.task == "all" else [args.task]

    from tabpfn import TabPFNRegressor  # deferred: heavy import
    import torch as _torch
    MODELS = {"v2": ROOT / "models" / "tabpfn-v2-regressor.ckpt",
              "v2.5": ROOT / "models" / "tabpfn-v2.5-regressor-v2.5_default.ckpt",
              "v3": ROOT / "models" / "tabpfn-v3-regressor-v3_default.ckpt"}
    MODEL = MODELS[args.model]
    DEVICE = "cuda" if _torch.cuda.is_available() else "cpu"
    print(f"[tabpfn] model={args.model} device={DEVICE}", flush=True)

    for task in tasks:
        d = load_task(task)
        n, y = len(d["y"]), d["y"]
        Z = get_embeddings(task, args.backbone)
        out_file = PARTS / f"tabpfn-{args.model}__{args.backbone}__{task}.csv"
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
                    # infeasible under group-aware splitting (e.g. selfmix-phi
                    # has only 10 physical groups: K>=10 leaves no test group)
                    print(f"[skip] {task} K={K} rep={rep}: empty test set "
                          f"(group-aware split)", flush=True)
                    continue
                t0 = time.time()
                # same label handling as the eval protocol: z-score by support stats
                ym, ysd = zscore_fit(y[tr])
                # same embedding handling: standardize by support stats, clamp ±5
                mu, sd = Z[tr].mean(0), Z[tr].std(0) + 1e-8
                Zs = np.clip((Z[tr] - mu) / sd, -5, 5)
                Zq = np.clip((Z[te] - mu) / sd, -5, 5)
                ys_z = (y[tr] - ym) / ysd
                m = TabPFNRegressor(model_path=str(MODEL), device=DEVICE)
                m.fit(Zs, ys_z)
                pred = m.predict(Zq) * ysd + ym
                yt = y[te]
                rmse = float(np.sqrt(np.mean((pred - yt) ** 2)))
                row = dict(task=task, shot=K, rep=rep,
                           r2=r2_score(pred, yt), rmse=rmse,
                           rpd=rpd(yt, rmse), secs=round(time.time() - t0, 2))
                rows.append(row)
                print(f"[tabpfn:{args.backbone}] {task} K={K} rep={rep}: "
                      f"r2={row['r2']:.3f} ({row['secs']}s)", flush=True)
        if rows:
            _write_rows(out_file, rows)


if __name__ == "__main__":
    main()
