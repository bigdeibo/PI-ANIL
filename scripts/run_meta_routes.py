"""Meta-route evaluation: Route B (kNN / frozen GP / DKL) and Route C (ANIL / FOMAML) few-shot quantitative evaluation.

The pairing protocol is bit-identical to Stages 0/2 (core.eval_split). Incremental checkpointing supports resumable runs.

Usage:
  python3 scripts/run_stage3_part.py --route knn --task all --reps 30
  python3 scripts/run_stage3_part.py --route gp --task all --reps 30
  python3 scripts/run_stage3_part.py --route dkl --init mae --source multi --task headline --reps 10
  python3 scripts/run_stage3_part.py --route anil --init mae --source multi --task headline --reps 10
"""
import sys
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.fsl_regression.core import (TASKS, HEADLINE_TASKS, load_task, eval_split,
                                     zscore_fit, z_standardize, r2_score,
                                     knn_predict, gp_frozen_predict, EMB_DIM)
from src.fewshot import rpd

OUT = ROOT / "results" / "meta_routes"
PARTS = OUT / "parts"
PARTS.mkdir(parents=True, exist_ok=True)
CKPT_DIR = OUT / "checkpoints"
EMB_CACHE = OUT / "emb_cache"
EMB_CACHE.mkdir(exist_ok=True)

EVAL_INNER_STEPS = 50  # number of head adaptation steps during ANIL evaluation


def _stable_lr(zs, safety=0.8):
    """Compute a stable full-batch GD learning rate from the support-set Hessian
    spectral radius: lr = safety / lambda_max(H), H = 2 Z^T Z / K (least squares).
    Adapted per episode to prevent divergence."""
    import torch
    H = 2.0 * (zs.T @ zs) / len(zs)
    lam = float(torch.linalg.eigvalsh(H).max())
    return safety / max(lam, 1e-6)


def get_mae_embeddings(task):
    f = EMB_CACHE / f"mae__{task}.npy"
    if f.exists():
        return np.load(f)
    import torch
    from src.ssl import load_encoder
    enc = load_encoder("mae")
    d = load_task(task)
    X = torch.from_numpy(d["X"])[:, None]
    zs = []
    with torch.no_grad():
        for i in range(0, len(X), 256):
            zs.append(enc(X[i:i + 256]).numpy())
    E = np.vstack(zs).astype(np.float32)
    np.save(f, E)
    return E


def eval_episode_knn(E, y, tr, te):
    zs, zq, _, _ = z_standardize(E[tr], E[te])
    return knn_predict(zs, y[tr], zq)


def eval_episode_gp(E, y, tr, te):
    zs, zq, _, _ = z_standardize(E[tr], E[te])
    ym, ysd = zscore_fit(y[tr])
    pred_z = gp_frozen_predict(zs, (y[tr] - ym) / ysd, zq)
    return pred_z * ysd + ym


def eval_episode_dkl(ck, d, tr, te):
    import torch
    from src.fsl.models import ResNet1Encoder
    from src.fsl_regression.core import gp_predict, L
    enc = ResNet1Encoder(L, emb_dim=EMB_DIM)
    enc.load_state_dict(ck["enc"]); enc.eval()
    X = torch.from_numpy(d["X"])[:, None]
    y = torch.from_numpy(d["y"]).float()
    with torch.no_grad():
        zs, zq = enc(X[tr]), enc(X[te])
        mu, sd = zs.mean(0, keepdim=True), zs.std(0, keepdim=True) + 1e-8
        zs = ((zs - mu) / sd).clamp(-5, 5)  # guard against division-by-zero blow-up when embeddings collapse
        zq = ((zq - mu) / sd).clamp(-5, 5)
        ym, ysd = zscore_fit(y[tr].numpy())
        pred = gp_predict(zs, (y[tr] - ym) / ysd, zq, ck["log_hyp"])
    return pred.numpy() * ysd + ym


def eval_episode_anil(ck, d, tr, te):
    import torch
    from src.fsl.models import ResNet1Encoder
    from src.fsl_regression.core import L
    enc = ResNet1Encoder(L, emb_dim=EMB_DIM)
    enc.load_state_dict(ck["enc"]); enc.eval()
    head = torch.nn.Linear(EMB_DIM, 1)
    head.load_state_dict(ck["head"])
    X = torch.from_numpy(d["X"])[:, None]
    y = torch.from_numpy(d["y"]).float()
    ym, ysd = zscore_fit(y[tr].numpy())
    ys_z = (y[tr] - ym) / ysd
    with torch.no_grad():
        zs = enc(X[tr])
        mu, sd = zs.mean(0, keepdim=True), zs.std(0, keepdim=True) + 1e-8
        zs = ((zs - mu) / sd).clamp(-5, 5)  # guard against division-by-zero blow-up when embeddings collapse
    lr_in = _stable_lr(zs)
    prev_loss = np.inf
    for _ in range(EVAL_INNER_STEPS):  # full-batch head adaptation (step size from spectral radius + divergence guard)
        loss = torch.mean((head(zs).squeeze(1) - ys_z) ** 2)
        lv = float(loss.item())
        if not np.isfinite(lv) or lv > 50.0 or lv > prev_loss * 3:
            break
        prev_loss = lv
        g = torch.autograd.grad(loss, list(head.parameters()))
        with torch.no_grad():
            for p, gr in zip(head.parameters(), g):
                p.sub_(lr_in * gr)
    with torch.no_grad():
        zq = ((enc(X[te]) - mu) / sd).clamp(-5, 5)
        pred = head(zq).squeeze(1)
    return pred.numpy() * ysd + ym


def eval_episode_fomaml(ck, d, tr, te):
    import copy
    import torch
    from src.fsl.models import ResNet1Encoder
    from src.fsl_regression.core import L
    enc = ResNet1Encoder(L, emb_dim=EMB_DIM)
    enc.load_state_dict(ck["enc"])
    head = torch.nn.Linear(EMB_DIM, 1)
    head.load_state_dict(ck["head"])
    X = torch.from_numpy(d["X"])[:, None]
    y = torch.from_numpy(d["y"]).float()
    ym, ysd = zscore_fit(y[tr].numpy())
    ys_z = (y[tr] - ym) / ysd
    params = list(enc.parameters()) + list(head.parameters())
    steps = int(ck.get("inner_steps", 5))
    lr_in = 0.0
    prev_loss = np.inf
    for it in range(steps):
        enc.train()
        zs = enc(X[tr])
        mu, sd = zs.mean(0, keepdim=True), zs.std(0, keepdim=True) + 1e-8
        zsc = ((zs - mu) / sd).clamp(-5, 5)
        lr_in = _stable_lr(zsc, safety=0.4) if it == 0 else lr_in  # more conservative for full-parameter adaptation
        loss = torch.mean((head(zsc).squeeze(1) - ys_z) ** 2)
        lv = float(loss.item())
        if not np.isfinite(lv) or lv > 50.0 or lv > prev_loss * 3:
            break
        prev_loss = lv
        g = torch.autograd.grad(loss, params)
        with torch.no_grad():
            for p, gr in zip(params, g):
                p.sub_(lr_in * gr)
    enc.eval()
    with torch.no_grad():
        zs = enc(X[tr])
        mu, sd = zs.mean(0, keepdim=True), zs.std(0, keepdim=True) + 1e-8
        zq = ((enc(X[te]) - mu) / sd).clamp(-5, 5)
        pred = head(zq).squeeze(1)
    return pred.numpy() * ysd + ym


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--route", required=True,
                    choices=["knn", "gp", "dkl", "anil", "fomaml"])
    ap.add_argument("--init", default="mae", choices=["rand", "mae"])
    ap.add_argument("--source", default="multi", choices=["multi", "diesel"])
    ap.add_argument("--task", required=True,
                    choices=list(TASKS) + ["headline", "all"])
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--shots", default="5,10,20")
    args = ap.parse_args()

    tasks = (list(TASKS) if args.task == "all"
             else HEADLINE_TASKS if args.task == "headline" else [args.task])
    shots = [int(s) for s in args.shots.split(",")]

    ck = None
    if args.route in ("dkl", "anil", "fomaml"):
        import torch
        ck = torch.load(CKPT_DIR / f"{args.route}__{args.init}__{args.source}.pt",
                        map_location="cpu", weights_only=False)

    tag = (f"{args.route}" if args.route in ("knn", "gp")
           else f"{args.route}__{args.init}__{args.source}")
    for task in tasks:
        d = load_task(task)
        n, y = len(d["y"]), d["y"]
        E = get_mae_embeddings(task) if args.route in ("knn", "gp") else None
        out_file = PARTS / f"{tag}__{task}.csv"
        done = set()
        if out_file.exists():
            prev = pd.read_csv(out_file)
            done = set(zip(prev.shot, prev.rep))
        for K in shots:
            if n <= K + 5:
                continue
            for rep in range(args.reps):
                if (K, rep) in done:
                    continue
                tr, te = eval_split(n, K, rep, groups=d.get("groups"))
                t0 = time.time()
                try:
                    if args.route == "knn":
                        pred = eval_episode_knn(E, y, tr, te)
                    elif args.route == "gp":
                        pred = eval_episode_gp(E, y, tr, te)
                    elif args.route == "dkl":
                        pred = eval_episode_dkl(ck, d, tr, te)
                    elif args.route == "anil":
                        pred = eval_episode_anil(ck, d, tr, te)
                    else:
                        pred = eval_episode_fomaml(ck, d, tr, te)
                except Exception as ex:
                    print(f"[fail] {task} K={K} rep={rep}: {ex}", flush=True)
                    continue
                yt = y[te]
                rmse = float(np.sqrt(np.mean((pred - yt) ** 2)))
                row = dict(task=task, route=args.route, init=args.init,
                           source=args.source, shot=K, rep=rep,
                           r2=r2_score(pred, yt), rmse=rmse, rpd=rpd(yt, rmse))
                pd.DataFrame([row]).to_csv(out_file, mode="a",
                                           header=not out_file.exists(), index=False)
                print(f"[eval] {tag} {task} K={K} rep={rep}: "
                      f"r2={row['r2']:.3f} ({time.time() - t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
