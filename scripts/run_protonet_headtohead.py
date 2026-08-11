"""P1c evaluation: ProtoNet regression readout (frozen SSL encoder / meta-trained encoder).

Purpose: close the loop on Garzón 2025's "ProtoNet family for few-shot spectroscopic
quantification" — a head-to-head comparison against PI-ANIL / vanilla ANIL /
traditional baselines under our 13-task paired protocol.

Two modes (same kernel-regression readout; the only difference is the encoder source):
  --mode frozen : frozen MAE encoder (cached embeddings emb_cache/mae__{task}.npy),
                  the MAE counterpart of Garzón's "SimCLR+PR" (SSL encoder +
                  prototype readout).
  --mode meta   : meta-trained ProtoNet encoder (product of the proto branch of
                  metatrain.py, results/meta_routes/checkpoints/proto__mae__multi.pt),
                  the regression counterpart of Garzón's "meta-trained Prototypical
                  Networks".

ProtoNet regression = Nadaraya-Watson kernel regression in embedding space:
  pred(q) = sum_s softmax(-||z_q - z_s||^2 / tau) * y_s
- Embeddings are standardized by support-set statistics; tau is selected on an
  adaptive grid by support-set leave-one-out (LOO) MSE (leakage prevention).
- The readout is bit-identical to scripts/p1c_protonet_pilot.py; frozen and meta
  share the same readout, isolating the single variable "encoder source (SSL vs
  meta-trained)".

Protocol: bit-identical eval_split (seed*1000+K*100+rep), paired with Stages 0/2/3/4.
Default 30 reps (frozen-readout convention, same as kNN/GP/PLS); when pairing with
meta-learning (10 reps), use rep 0-9.

Usage:
  python3 scripts/run_p1c_part.py --mode frozen --task headline --reps 30
  python3 scripts/run_p1c_part.py --mode meta   --task headline --reps 30
"""
import sys
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.fsl_regression.core import (TASKS, HEADLINE_TASKS, DIESEL_TASKS,
                                     load_task, eval_split, EMB_DIM, L)
from src.fewshot import rpd

OUT = ROOT / "results" / "protonet_headtohead"
PARTS = OUT / "parts"
PARTS.mkdir(parents=True, exist_ok=True)
EMB_CACHE = OUT / "emb_cache"
S3_CKPT = ROOT / "results" / "meta_routes" / "checkpoints"

TASK_GROUPS = dict(
    headline=HEADLINE_TASKS,
    diesel=list(DIESEL_TASKS),
    corn=[t for t in TASKS if t.startswith("corn")],
    **{"headline+evoo": HEADLINE_TASKS + ["evoo-adulteration"],
       "all": list(TASKS)})


def protonet_predict(Es, ys, Ete):
    """ProtoNet regression: support-set standardization -> LOO selection of tau ->
    distance-weighted prediction.

    Bit-identical to the pilot scripts/p1c_protonet_pilot.py::protonet_predict.
    """
    mu, sd = Es.mean(0), Es.std(0) + 1e-8
    Es_n = (Es - mu) / sd
    Ete_n = (Ete - mu) / sd
    Ds = ((Es_n[:, None, :] - Es_n[None, :, :]) ** 2).sum(-1)  # K×K
    med = np.median(Ds[Ds > 0]) if (Ds > 0).any() else 1.0
    taus = med * np.logspace(-2, 2, 25)
    best_tau, best_loo = taus[0], np.inf
    for tau in taus:
        W = np.exp(-Ds / tau)
        np.fill_diagonal(W, 0.0)          # LOO: exclude the sample itself
        s = W.sum(1, keepdims=True)
        s[s == 0] = 1.0
        pred = (W / s) @ ys
        loo = float(((pred - ys) ** 2).mean())
        if loo < best_loo:
            best_loo, best_tau = loo, tau
    D = ((Ete_n[:, None, :] - Es_n[None, :, :]) ** 2).sum(-1)  # Nte×K
    W = np.exp(-D / best_tau)
    s = W.sum(1, keepdims=True)
    s[s == 0] = 1.0
    return (W / s) @ ys


def r2_score(pred, yt):
    ss_res = float(np.sum((pred - yt) ** 2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    return 1 - ss_res / ss_tot if ss_tot > 1e-12 else np.nan


def embeddings_frozen(task):
    """Frozen MAE cached embeddings (row order matches the non-NaN rows of load_task)."""
    f = EMB_CACHE / f"mae__{task}.npy"
    if not f.exists():
        raise FileNotFoundError(f"no frozen embedding cache: {f} (run run_stage2_probe first to build the cache)")
    return np.load(f), load_task(task)["y"]


def embeddings_meta(task, ck):
    """Meta-trained ProtoNet encoder embeddings (same X path as run_stage4_part)."""
    import torch
    from src.fsl.models import ResNet1Encoder
    enc = ResNet1Encoder(L, emb_dim=EMB_DIM)
    enc.load_state_dict(ck["enc"])
    enc.eval()
    d = load_task(task)
    X = torch.from_numpy(d["X"])[:, None]
    zs = []
    with torch.no_grad():
        for i in range(0, len(X), 256):
            zs.append(enc(X[i:i + 256]).numpy())
    return np.vstack(zs).astype(np.float32), d["y"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["frozen", "meta"])
    ap.add_argument("--task", required=True,
                    choices=list(TASKS) + list(TASK_GROUPS))
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--ckpt", default=str(S3_CKPT / "proto__mae__multi.pt"),
                    help="encoder checkpoint for meta mode")
    args = ap.parse_args()
    tasks = TASK_GROUPS.get(args.task, [args.task])

    ck = None
    if args.mode == "meta":
        import torch
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        print(f"[meta] loaded {args.ckpt} (ep {ck.get('epoch')}, "
              f"best_val {ck.get('best_val'):.4f})", flush=True)

    for task in tasks:
        E, y = (embeddings_meta(task, ck) if args.mode == "meta"
                else embeddings_frozen(task))
        groups = load_task(task)["groups"]  # EVOO: grouped by sample_id; None otherwise
        n = len(y)
        assert E.shape[0] == n, f"{task}: E{E.shape[0]} != y{n}"
        out_file = PARTS / f"protonet_{args.mode}__{task}.csv"
        done = set()
        if out_file.exists():
            done = set(zip(pd.read_csv(out_file).shot,
                           pd.read_csv(out_file).rep))
        n_done0 = len(done)
        for K in (5, 10, 20):
            if n <= K + 5:
                continue
            for rep in range(args.reps):
                if (K, rep) in done:
                    continue
                tr, te = eval_split(n, K, rep, groups=groups)
                try:
                    pred = protonet_predict(E[tr], y[tr], E[te])
                except Exception as ex:
                    print(f"[fail] {task} K={K} rep={rep}: {ex}", flush=True)
                    continue
                yt = y[te]
                rmse = float(np.sqrt(np.mean((pred - yt) ** 2)))
                row = dict(task=task, model=f"protonet_{args.mode}", shot=K,
                           rep=rep, r2=r2_score(pred, yt), rmse=rmse,
                           rpd=rpd(yt, rmse))
                pd.DataFrame([row]).to_csv(out_file, mode="a",
                                           header=not out_file.exists(),
                                           index=False)
        print(f"[done] {task} ({len(set(zip(pd.read_csv(out_file).shot, pd.read_csv(out_file).rep))) - n_done0} new rows) "
              f"-> {out_file}", flush=True)


if __name__ == "__main__":
    main()
