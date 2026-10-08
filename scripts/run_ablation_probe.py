"""W4: frozen-encoder ridge probe for corpus-recipe ablation (design doc §5.2).

For each recipe checkpoint results/pretraining/checkpoints/encoder_mae{SUFFIX}.pt,
embed all 13 benchmark tasks and evaluate a closed-form ridge head (channel-B
protocol: scale-only + unpenalized intercept + LOO-CV lambda) under the paired
splits (s = 42*1000 + K*100 + rep, rep 0..9). The reference recipe (paper corpus)
is probed via --suffix "" so all recipes share one evaluation code path.

Output: results/pretraining/ablation/probe{suffix}__{task}.csv (resumable).
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
from src.uq.ridge_uq import loo_lambda, ridge_solve

OUT = ROOT / "results" / "pretraining" / "ablation"
OUT.mkdir(parents=True, exist_ok=True)
CKPT_DIR = ROOT / "results" / "pretraining" / "checkpoints"
BENCH = [t for t in TASKS if t not in ("selfmix-phi", "edibleoil-pv")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="",
                    help="checkpoint suffix after encoder_mae (e.g. _s6000_l28_p0; "
                         "empty = reference recipe encoder_mae.pt)")
    ap.add_argument("--task", default="all")
    ap.add_argument("--reps", type=int, default=10)
    args = ap.parse_args()
    tasks = BENCH if args.task == "all" else [args.task]

    import torch
    from src.fsl.models import ResNet1Encoder
    ck = torch.load(CKPT_DIR / f"encoder_mae{args.suffix}.pt", map_location="cpu",
                    weights_only=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc = ResNet1Encoder(L, emb_dim=EMB_DIM)
    enc.load_state_dict(ck["enc"])
    enc.eval().to(device)
    tag = f"mae{args.suffix}" if args.suffix else "mae_ref"
    print(f"[ablation-probe] {tag} device={device}", flush=True)

    for task in tasks:
        d = load_task(task)
        n, y_all = len(d["y"]), d["y"].astype(np.float64)
        X = torch.from_numpy(d["X"])[:, None].to(device)
        with torch.no_grad():
            Z = enc(X).cpu().numpy().astype(np.float64)
        out_file = OUT / f"probe_{tag}__{task}.csv"
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
                if len(te) == 0:
                    continue
                t0 = time.time()
                ym, ysd = zscore_fit(y_all[tr])
                sd = Z[tr].std(0) + 1e-8
                Zs = np.clip(Z[tr] / sd, -5, 5)
                Zq = np.clip(Z[te] / sd, -5, 5)
                ys_z = (y_all[tr] - ym) / ysd
                lam = loo_lambda(Zs, ys_z)
                w, _, _ = ridge_solve(Zs, ys_z, lam)
                Zqa = np.column_stack([np.ones(len(Zq)), Zq])
                pred = Zqa @ w * ysd + ym
                yt = y_all[te]
                row = dict(task=task, encoder=tag, shot=K, rep=rep,
                           r2=r2_score(pred, yt),
                           rmse=float(np.sqrt(np.mean((pred - yt) ** 2))),
                           lam=lam)
                pd.DataFrame([row]).to_csv(out_file, mode="a",
                                           header=not out_file.exists(),
                                           index=False)
                print(f"[{tag}] {task} K={K} rep={rep}: r2={row['r2']:.3f} "
                      f"({time.time()-t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
