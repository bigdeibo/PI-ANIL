"""A1: persist the mechanism-verification numbers — freeze the saliency ratio and
additivity-geometry deviation from Section 3.6 of the paper, previously only printed
to the console by stage4_visualize.py, into an auditable CSV.

The computation is bit-identical to _saliency_curve / fig_additivity in
stage4_visualize.py (evaluation-protocol head adaptation: step size from the Hessian
spectral radius + 50 steps + loss guard + ±5 clipping), but this script is
self-contained and does not import visualize (to avoid daimon_runtime side effects).

Output: results/physics/mechanism_audit.csv (long table: variant, task, metric, value, n)
Closes review item A1 (mechanism numbers not persisted).

Usage: python scripts/mechanism_audit.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
from src.fsl.models import ResNet1Encoder
from src.fsl_regression.core import (load_task, eval_split, zscore_fit, EMB_DIM, L)
from src.pifsl.constraints import band_mask
from src.ssl import snv

OUT = ROOT / "results" / "physics"
CK3 = ROOT / "results" / "meta_routes" / "checkpoints" / "anil__mae__multi.pt"
CK4 = ROOT / "results" / "physics" / "checkpoints"

VARIANTS = [
    ("vanilla", CK3),
    ("add", CK4 / "anilpi__add__mae__multi.pt"),
    ("band", CK4 / "anilpi__band__mae__multi.pt"),
    ("both", CK4 / "anilpi__both__mae__multi.pt"),
]

SALIENCY_TASKS = ["diesel-CN", "gasoline-octane"]  # consistent with Section 3.6 / Fig. 2 of the paper


def _adapt_head(ck, d, tr):
    """Evaluation-protocol head adaptation (identical to run_stage4_part.eval_episode_anil)."""
    enc = ResNet1Encoder(L, emb_dim=EMB_DIM)
    enc.load_state_dict(ck["enc"])
    enc.eval()
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
    H = 2.0 * (zs.T @ zs) / len(zs)
    lr_in = 0.8 / max(float(torch.linalg.eigvalsh(H).max()), 1e-6)
    prev = np.inf
    for _ in range(50):
        loss = torch.mean((head(zs).squeeze(1) - ys_z) ** 2)
        lv = float(loss.item())
        if not np.isfinite(lv) or lv > 50.0 or lv > prev * 3:
            break
        prev = lv
        g = torch.autograd.grad(loss, list(head.parameters()))
        with torch.no_grad():
            for p, gr in zip(head.parameters(), g):
                p.sub_(lr_in * gr)
    return enc, head, mu, sd


def saliency_in_out(ckpt_path, task, K=10, reps=5, n_test=48):
    """Mean |d yhat / dx| over test spectra from several episodes -> in-band / out-of-band
    sensitivity ratio."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    d = load_task(task)
    n = len(d["y"])
    acc = np.zeros(L)
    for rep in range(reps):
        tr, te = eval_split(n, K, rep)
        rng = np.random.default_rng(1234 + rep)
        te_sub = rng.choice(te, min(n_test, len(te)), replace=False)
        enc, head, mu, sd = _adapt_head(ck, d, tr)
        Xq = torch.from_numpy(d["X"][te_sub])[:, None].requires_grad_(True)
        zq = ((enc(Xq) - mu) / sd).clamp(-5, 5)
        pred = head(zq).squeeze(1)
        s = torch.autograd.grad(pred.sum(), Xq)[0].squeeze(1)  # (n,L)
        acc += s.detach().abs().mean(0).numpy()
    acc /= reps
    acc = acc / (acc.max() + 1e-12)
    mask = band_mask(task)
    inside = float(acc[mask > 0.5].sum())
    outside = float(acc[mask <= 0.5].sum())
    return inside / (outside + 1e-12), inside, outside


def additivity_dev(ckpt_path):
    """Embedding linear-geometry deviation on real EVOO+HZO blend spectra (as in Fig. 3).

    target = (1-c) * z_EVOO + c * z_HZO (linear interpolation of class means),
    c = HZO concentration.
    dev = ||z_mix - target|| / ||z_mix|| (relative deviation, scale-free).
    """
    from src.datasets import load_evoo
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    e = load_evoo()
    X = np.asarray(e["X"], float)
    go, gn = np.linspace(0, 1, X.shape[1]), np.linspace(0, 1, L)
    X = np.stack([np.interp(gn, go, r) for r in X])
    X = snv(X).astype(np.float32)
    labels = np.asarray(e["labels"])
    levels = np.asarray(e["targets"]["adulteration_level"], float)

    enc = ResNet1Encoder(L, emb_dim=EMB_DIM)
    enc.load_state_dict(ck["enc"])
    enc.eval()
    with torch.no_grad():
        Z = enc(torch.from_numpy(X)[:, None]).numpy()

    m_ev = labels == "EVOO"
    m_hz = labels == "Hazelnut oil"
    m_mix = labels == "EVOO+HZO"
    z_ev, z_hz = Z[m_ev].mean(0), Z[m_hz].mean(0)
    cs = levels[m_mix] / 100.0
    dev = []
    for ci, zi in zip(cs, Z[m_mix]):
        target = (1 - ci) * z_ev + ci * z_hz
        dev.append(np.linalg.norm(zi - target) / (np.linalg.norm(zi) + 1e-12))
    dev = np.asarray(dev)
    return float(dev.mean()), float(dev.max()), int(len(dev))


def main():
    rows = []
    print("=== saliency in-band / out-of-band sensitivity ratio (larger = more concentrated on the C-H prior bands) ===")
    for variant, path in VARIANTS:
        for task in SALIENCY_TASKS:
            rin, inside, outside = saliency_in_out(path, task)
            rows.append(dict(variant=variant, task=task,
                             metric="saliency_in_out", value=rin, n=5))
            print(f"  {variant:8s} {task:16s} in/out={rin:.3f} "
                  f"(in={inside:.2f} out={outside:.2f})")

    print("\n=== additivity embedding linear-geometry deviation (real EVOO+HZO blends) ===")
    for variant, path in VARIANTS:
        mean_dev, max_dev, n_b = additivity_dev(path)
        rows.append(dict(variant=variant, task="EVOO+HZO",
                         metric="additivity_dev_mean", value=mean_dev, n=n_b))
        rows.append(dict(variant=variant, task="EVOO+HZO",
                         metric="additivity_dev_max", value=max_dev, n=n_b))
        print(f"  {variant:8s} mean_dev={mean_dev:.4f}  max_dev={max_dev:.4f}  "
              f"n_blends={n_b}")

    df = pd.DataFrame(rows)
    out_csv = OUT / "mechanism_audit.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nsaved -> {out_csv}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
