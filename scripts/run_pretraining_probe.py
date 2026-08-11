"""Pretraining linear-probe / fine-tuning few-shot validation — strictly paired comparison against the baselines.

Replicates the seeding scheme of src/fewshot.run_fewshot_regression
(seed*1000+K*100+rep, rng.choice(n,K) + setdiff), guaranteeing that the
support/test splits are exactly identical to the baseline protocol (paired).

Models:
  mae_ridge / simclr_ridge / rand_ridge — frozen encoders (MAE / SimCLR / random init)
      → 128-dim embedding → support-set z-score standardization → RidgeCV (LOO)
  mae_ft — MAE encoder + linear head fine-tuned end-to-end (150 epochs, capturing the initialization gain)

Usage (granular runs):
  python3 scripts/run_stage2_probe.py --model mae_ridge --task diesel-CN
  python3 scripts/run_stage2_probe.py --model mae_ft --task gasoline-octane --reps 10
"""
import sys
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.datasets import LOADERS
from src.ssl import load_encoder, snv
from src.fewshot import rpd

OUT = ROOT / "results" / "pretraining"
PARTS = OUT / "parts"
PARTS.mkdir(parents=True, exist_ok=True)
L = 512
EMB_CACHE = OUT / "emb_cache"
EMB_CACHE.mkdir(exist_ok=True)

TASKS = dict(
    **{f"diesel-{a}": ("diesel", a) for a in
       ("CN", "BP50", "D4052", "FLASH", "FREEZE", "TOTAL", "VISC")},
    **{"gasoline-octane": ("gasoline", "octane")},
    **{"corn_m5-protein": ("corn", "protein"), "corn_m5-oil": ("corn", "oil")},
    **{"evoo-adulteration": ("evoo", "adulteration_level")},
)


def get_task_data(task):
    ds, attr = TASKS[task]
    d = LOADERS[ds]()
    if ds == "corn":
        X = d["instruments"]["m5"][0]
        y = d["targets"][attr]
    else:
        X, y = d["X"], d["targets"][attr]
    m = ~np.isnan(y)
    return np.asarray(X, float)[m], np.asarray(y, float)[m]


def resample512(X):
    n, li = X.shape
    go, gn = np.linspace(0, 1, li), np.linspace(0, 1, L)
    return np.stack([np.interp(gn, go, r) for r in X]).astype(np.float32)


def get_embeddings(task, which):
    """Cached embedding matrix (n,128). which: mae|simclr|rand."""
    key = f"{which}__{task}.npy"
    f = EMB_CACHE / key
    if f.exists():
        return np.load(f)
    import torch
    from src.fsl.models import ResNet1Encoder
    if which == "rand":
        torch.manual_seed(1234)
        enc = ResNet1Encoder(L, emb_dim=128).eval()
    else:
        enc = load_encoder(which)
    X, _y = get_task_data(task)
    Xr = torch.from_numpy(snv(resample512(X)))[:, None]
    zs = []
    with torch.no_grad():
        for i in range(0, len(Xr), 256):
            zs.append(enc(Xr[i:i + 256]).numpy())
    E = np.vstack(zs).astype(np.float32)
    np.save(f, E)
    return E


def fit_predict_ridge(Es, ys, Ete, alphas=np.logspace(-4, 4, 17)):
    from sklearn.linear_model import RidgeCV
    mu, sd = Es.mean(0), Es.std(0) + 1e-8
    m = RidgeCV(alphas=alphas)
    m.fit((Es - mu) / sd, ys)
    return m.predict((Ete - mu) / sd)


def fit_predict_ft(which, Xs_raw, ys, Xte_raw, rep, epochs=150):
    """Fine-tune the encoder + linear head end-to-end. Input spectra are at their original length; SNV + resampling are applied internally."""
    import torch
    enc = load_encoder(which)
    head = torch.nn.Linear(128, 1)
    torch.manual_seed(42 * 1000 + rep)
    opt = torch.optim.Adam(list(enc.parameters()) + list(head.parameters()),
                           lr=1e-3, weight_decay=1e-4)
    Xt = torch.from_numpy(snv(resample512(Xs_raw)))[:, None]
    ym, ysd = float(ys.mean()), float(ys.std() + 1e-8)  # support-set target standardization (same as the baseline CNN)
    yt = torch.tensor((ys - ym) / ysd, dtype=torch.float32)[:, None]
    bs = min(16, len(yt))
    enc.train(); head.train()
    lossf = torch.nn.MSELoss()
    for _ in range(epochs):
        perm = torch.randperm(len(yt))
        for i in range(0, len(yt), bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = lossf(head(enc(Xt[idx])), yt[idx])
            loss.backward()
            opt.step()
    enc.eval(); head.eval()
    with torch.no_grad():
        Xq = torch.from_numpy(snv(resample512(Xte_raw)))[:, None]
        return head(enc(Xq)).numpy().ravel() * ysd + ym


def run(task, model, reps, seed=42, shots=(5, 10, 20)):
    X, y = get_task_data(task)
    n = len(y)
    E = None
    if model.endswith("ridge"):
        E = get_embeddings(task, model.rsplit("_", 1)[0])
    out_file = PARTS / f"probe__{model}__{task}.csv"
    done = set()
    if out_file.exists():
        prev = pd.read_csv(out_file)
        done = set(zip(prev.shot, prev.rep))
    for K in shots:
        if n <= K + 5:
            continue
        for rep in range(reps):
            if (K, rep) in done:
                continue
            rng = np.random.default_rng(seed * 1000 + K * 100 + rep)
            tr = rng.choice(n, K, replace=False)
            te = np.setdiff1d(np.arange(n), tr)
            t0 = time.time()
            try:
                if model.endswith("ridge"):
                    pred = fit_predict_ridge(E[tr], y[tr], E[te])
                else:
                    pred = fit_predict_ft(model.rsplit("_", 1)[0], X[tr], y[tr], X[te], rep)
            except Exception as ex:
                print(f"[fail] {task} {model} K={K} rep={rep}: {ex}", flush=True)
                continue
            yt = y[te]
            rmse = float(np.sqrt(np.mean((pred - yt) ** 2)))
            ss_res = float(np.sum((pred - yt) ** 2))
            ss_tot = float(np.sum((yt - yt.mean()) ** 2))
            r2 = 1 - ss_res / ss_tot if ss_tot > 1e-12 else np.nan
            row = dict(task=task, model=model, shot=K, rep=rep, r2=r2,
                       rmse=rmse, rpd=rpd(yt, rmse))
            pd.DataFrame([row]).to_csv(out_file, mode="a",
                                       header=not out_file.exists(), index=False)
            print(f"[probe] {task} {model} K={K} rep={rep}: r2={r2:.3f} "
                  f"({time.time() - t0:.1f}s)", flush=True)
    print("appended ->", out_file)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    choices=["mae_ridge", "simclr_ridge", "rand_ridge", "mae_ft",
                             "simclr_ft", "simclr_cont60_ridge",
                             "simclr_scratch60_ridge"])
    ap.add_argument("--task", required=True, choices=list(TASKS) + ["all"])
    ap.add_argument("--reps", type=int, default=30)
    args = ap.parse_args()
    tasks = list(TASKS) if args.task == "all" else [args.task]
    for t in tasks:
        run(t, args.model, args.reps)
