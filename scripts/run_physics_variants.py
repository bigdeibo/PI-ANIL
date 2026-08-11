"""Physics-variant evaluation: PI-ANIL variant ablation and LOMO cross-material extrapolation.

The evaluation-time adaptation protocol is identical to the meta-route adaptation (adaptive step size from
the Hessian spectral radius + divergence guard + ±5 clipping).
Usage:
  python3 scripts/run_stage4_part.py --tag anilpi__both__mae__multi --task headline+evoo
  python3 scripts/run_stage4_part.py --tag anilpi__both__mae__no_gasoline --task gasoline-octane
  python3 scripts/run_stage4_part.py --tag anilpi__both__mae__no_diesel --task diesel
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
                                     load_task, eval_split, zscore_fit,
                                     r2_score, EMB_DIM, L)
from src.fewshot import rpd

OUT = ROOT / "results" / "physics"
PARTS = OUT / "parts"
PARTS.mkdir(parents=True, exist_ok=True)
CKPT_DIR = OUT / "checkpoints"

EVAL_INNER_STEPS = 50
TASK_GROUPS = dict(headline=HEADLINE_TASKS, diesel=list(DIESEL_TASKS),
                   corn=[t for t in TASKS if t.startswith("corn")],
                   **{"headline+evoo": HEADLINE_TASKS + ["evoo-adulteration"]})


def _stable_lr(zs, safety=0.8):
    import torch
    H = 2.0 * (zs.T @ zs) / len(zs)
    lam = float(torch.linalg.eigvalsh(H).max())
    return safety / max(lam, 1e-6)


def eval_episode_anil(ck, d, tr, te):
    """Adaptation protocol bit-identical to eval_episode_anil in the meta-route script."""
    import torch
    from src.fsl.models import ResNet1Encoder
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
        zs = ((zs - mu) / sd).clamp(-5, 5)
    lr_in = _stable_lr(zs)
    prev_loss = np.inf
    for _ in range(EVAL_INNER_STEPS):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--task", required=True,
                    choices=list(TASKS) + list(TASK_GROUPS))
    ap.add_argument("--reps", type=int, default=10)
    args = ap.parse_args()
    tasks = TASK_GROUPS.get(args.task, [args.task])

    import torch
    ck = torch.load(CKPT_DIR / f"{args.tag}.pt", map_location="cpu",
                    weights_only=False)
    for task in tasks:
        d = load_task(task)
        n, y = len(d["y"]), d["y"]
        out_file = PARTS / f"{args.tag}__{task}.csv"
        done = set()
        if out_file.exists():
            prev = pd.read_csv(out_file)
            done = set(zip(prev.shot, prev.rep))
        for K in (5, 10, 20):
            if n <= K + 5:
                continue
            for rep in range(args.reps):
                if (K, rep) in done:
                    continue
                tr, te = eval_split(n, K, rep, groups=d.get("groups"))
                t0 = time.time()
                try:
                    pred = eval_episode_anil(ck, d, tr, te)
                except Exception as ex:
                    print(f"[fail] {task} K={K} rep={rep}: {ex}", flush=True)
                    continue
                yt = y[te]
                rmse = float(np.sqrt(np.mean((pred - yt) ** 2)))
                row = dict(task=task, tag=args.tag, shot=K, rep=rep,
                           r2=r2_score(pred, yt), rmse=rmse, rpd=rpd(yt, rmse))
                pd.DataFrame([row]).to_csv(out_file, mode="a",
                                           header=not out_file.exists(), index=False)
                print(f"[eval] {args.tag} {task} K={K} rep={rep}: "
                      f"r2={row['r2']:.3f} ({time.time() - t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
