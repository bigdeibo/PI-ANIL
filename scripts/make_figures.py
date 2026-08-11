"""Mechanism visualization: produces the four main-paper figures.

Fig1 framework.png        PI-ANIL framework schematic (matplotlib text boxes)
Fig2 saliency_vs_bands.png prediction sensitivity vs physics-prior bands (vanilla / band / both)
Fig3 additivity_check.png  embedding linear-geometry deviation on real EVOO+HZO blends
Fig4 lomo_extrapolation.png LOMO cross-material extrapolation comparison (vanilla-LOMO / PI-LOMO / seen-task reference)

Run: python3 scripts/make_figures.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(sys.executable).parent.parent.parent))

from daimon_runtime import setup_plot
setup_plot()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from src.fsl_regression.core import (TASKS, DIESEL_TASKS, load_task, eval_split,
                                     zscore_fit, EMB_DIM, L)
from src.pifsl.constraints import band_mask, BANDS_CH, task_wavelengths

FIG = ROOT / "results" / "physics" / "figs"
FIG.mkdir(parents=True, exist_ok=True)
CK3 = ROOT / "results" / "meta_routes" / "checkpoints" / "anil__mae__multi.pt"
CK4 = ROOT / "results" / "physics" / "checkpoints"

COLORS = dict(vanilla="#7f7f7f", add="#1f77b4", band="#ff7f0e", both="#d62728")


# ------------------------------------------------- Fig1 framework schematic
def fig_framework():
    fig, ax = plt.subplots(figsize=(11, 6.2))
    ax.set_xlim(0, 100); ax.set_ylim(0, 62); ax.axis("off")

    def box(x, y, w, h, text, fc="#eef3fb", ec="#3465a4", fs=10, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6",
                                    fc=fc, ec=ec, lw=1.4))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold else "normal")

    def arrow(x1, y1, x2, y2, style="-|>", color="#333", lw=1.6, cs=None):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                     mutation_scale=14, color=color, lw=lw,
                     connectionstyle=cs or "arc3,rad=0"))

    box(1, 40, 20, 14, "Episode sampler\nsubstance × attribute task\n"
        "(diesel · gasoline · corn · EVOO)", fc="#fdf6e3", ec="#b58900", bold=True)
    box(1, 12, 20, 12, "SSL pretraining\nMAE on 9339 unlabelled\nspectra (multi-source corpus)",
        fc="#f3f3f3", ec="#666")
    box(28, 44, 16, 10, "Support set\nK spectra + y", fs=10)
    box(28, 14, 16, 10, "Query set\nQ spectra", fs=10)
    box(50, 28, 17, 13, "Encoder E(·)\n1D-ResNet\n(MAE initialised)", fc="#e8f5e9",
        ec="#2e7d32", bold=True)
    box(73, 40, 25, 12, "ANIL head adaptation\nridge-stable LR, 50 steps\n(support only)", fc="#e8f5e9",
        ec="#2e7d32")
    box(73, 12, 25, 12, "K-shot prediction\n$\\hat{y}$ = head(E(x))", fc="#e8f5e9",
        ec="#2e7d32", bold=True)
    box(20, 0.5, 62, 9, "Meta-objective:  query MSE  +  λ_add · Beer–Lambert additivity"
        "  +  λ_band · C–H band saliency\n(meta-gradient back-propagates to E(·))",
        fc="#fce4ec", ec="#ad1457", fs=10, bold=True)

    arrow(21, 47, 28, 48.5)
    arrow(21, 44, 28, 20, cs="arc3,rad=0.15")
    arrow(44, 49, 52, 41.2)
    arrow(44, 19, 52, 28.2)
    arrow(67, 36, 73, 45)
    arrow(67, 33, 73, 19)
    arrow(85.5, 40, 85.5, 24.5)
    arrow(11, 40, 11, 24.5)
    arrow(21, 16, 50, 28.5, cs="arc3,rad=-0.12")
    arrow(49, 9.8, 56, 28, cs="arc3,rad=-0.1")
    ax.text(3, 56.5, "Physics-informed few-shot NIR quantitative analysis (PI-ANIL)",
            fontsize=13, fontweight="bold")
    fig.savefig(FIG / "fig1_framework.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("fig1 ok", flush=True)


# ------------------------------------------------- Fig2 band sensitivity
def _adapt_head(ck, d, tr):
    """ANIL head adaptation consistent with the evaluation protocol; returns
    (enc, head, mu, sd, ym, ysd)."""
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
    return enc, head, mu, sd, ym, ysd


def _saliency_curve(ckpt_path, task, K=10, reps=5, n_test=48):
    """Mean |d yhat / dx| over test spectra from several episodes (512 positions)."""
    import torch
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    d = load_task(task)
    n = len(d["y"])
    acc = np.zeros(L)
    for rep in range(reps):
        tr, te = eval_split(n, K, rep)
        rng = np.random.default_rng(1234 + rep)
        te_sub = rng.choice(te, min(n_test, len(te)), replace=False)
        enc, head, mu, sd, ym, ysd = _adapt_head(ck, d, tr)
        Xq = torch.from_numpy(d["X"][te_sub])[:, None].requires_grad_(True)
        zq = ((enc(Xq) - mu) / sd).clamp(-5, 5)
        pred = head(zq).squeeze(1)
        s = torch.autograd.grad(pred.sum(), Xq)[0].squeeze(1)  # (n,L)
        acc += s.detach().abs().mean(0).numpy()
    acc /= reps
    return acc / (acc.max() + 1e-12)


def fig_saliency():
    ckpts = [("vanilla (route-comparison ANIL)", CK3, COLORS["vanilla"]),
             ("PI: band prior", CK4 / "anilpi__band__mae__multi.pt", COLORS["band"]),
             ("PI: dual constraint", CK4 / "anilpi__both__mae__multi.pt", COLORS["both"])]
    tasks = [("diesel-CN", "diesel"), ("gasoline-octane", "gasoline")]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.9), sharey=True)
    for ax, (task, short) in zip(axes, tasks):
        wl = np.linspace(*_wl_range(short), L)
        for a, b in BANDS_CH:
            ax.axvspan(max(a, wl[0]), min(b, wl[-1]), color="#ffd54f", alpha=0.28,
                       lw=0)
        for name, path, c in ckpts:
            t0 = time.time()
            curve = _saliency_curve(path, task)
            mask = band_mask(task)
            rin = float(curve[mask > 0.5].sum() / (curve[mask <= 0.5].sum() + 1e-12))
            from scipy.ndimage import uniform_filter1d
            ax.plot(wl, uniform_filter1d(curve, 9), color=c, lw=1.5,
                    label=f"{name} (in/out={rin:.2f})")
            print(f"  saliency {task} {name}: in/out={rin:.2f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        ax.set_title(task, fontsize=11)
        ax.set_xlabel("Wavelength (nm)")
        ax.set_xlim(wl[0], wl[-1])
    axes[0].set_ylabel(r"Normalised |$\partial\hat{y}/\partial x$|")
    axes[0].legend(fontsize=8.5, loc="upper right")
    axes[1].text(0.02, 0.04, "shaded: C–H prior bands", transform=axes[1].transAxes,
                 fontsize=9, color="#8d6e00")
    fig.suptitle("Prediction sensitivity concentrates on physical C–H bands", y=1.02,
                 fontsize=12, fontweight="bold")
    fig.savefig(FIG / "fig2_saliency_vs_bands.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("fig2 ok", flush=True)


def _wl_range(ds):
    wl = task_wavelengths(next(t for t in TASKS if TASKS[t][0] == ds))
    return float(wl[0]), float(wl[-1])


# ------------------------------------------------- Fig3 additivity-geometry check
def fig_additivity():
    import torch
    from src.datasets import load_evoo
    from src.fsl.models import ResNet1Encoder
    from src.ssl import snv

    e = load_evoo()
    X, wl = np.asarray(e["X"], float), np.asarray(e["wavelengths"], float)
    go, gn = np.linspace(0, 1, X.shape[1]), np.linspace(0, 1, L)
    X = np.stack([np.interp(gn, go, r) for r in X])
    X = snv(X).astype(np.float32)
    labels = np.asarray(e["labels"])
    levels = np.asarray(e["targets"]["adulteration_level"], float)
    Xt = torch.from_numpy(X)[:, None]

    m_ev = labels == "EVOO"
    m_hz = labels == "Hazelnut oil"
    m_mix = labels == "EVOO+HZO"

    ckpts = [("vanilla (route-comparison ANIL)", CK3, COLORS["vanilla"], "o"),
             ("PI: +additivity", CK4 / "anilpi__add__mae__multi.pt", COLORS["add"], "s"),
             ("PI: dual constraint", CK4 / "anilpi__both__mae__multi.pt", COLORS["both"], "^")]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for name, path, c, mk in ckpts:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        enc = ResNet1Encoder(L, emb_dim=EMB_DIM)
        enc.load_state_dict(ck["enc"]); enc.eval()
        with torch.no_grad():
            Z = enc(Xt).numpy()
        z_ev, z_hz = Z[m_ev].mean(0), Z[m_hz].mean(0)
        cs = levels[m_mix] / 100.0
        Zm = Z[m_mix]
        dev = []
        for ci, zi in zip(cs, Zm):
            target = (1 - ci) * z_ev + ci * z_hz
            dev.append(np.linalg.norm(zi - target) / (np.linalg.norm(zi) + 1e-12))
        dev = np.asarray(dev)
        ax.scatter(cs * 100, dev, s=9, alpha=0.25, color=c, marker=mk)
        # moving average (by concentration level)
        ux = sorted(set((cs * 100).round(1)))
        ax.plot(ux, [dev[np.isclose(cs * 100, u)].mean() for u in ux],
                color=c, lw=2, label=name)
        print(f"  additivity {name}: mean dev {dev.mean():.4f}", flush=True)
    ax.set_xlabel("Hazelnut concentration in EVOO blend (%)")
    ax.set_ylabel("Relative deviation from linear\nembedding geometry")
    ax.set_title("Learned embeddings are near-linear in mixture concentration\n"
                 "(EVOO + hazelnut oil, true blends; deviation < 1% for all variants)",
                 fontsize=11)
    ax.legend(fontsize=9)
    fig.savefig(FIG / "fig3_additivity_check.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("fig3 ok", flush=True)


# ------------------------------------------------- Fig4 LOMO extrapolation
def fig_lomo():
    df4 = pd.read_csv(ROOT / "results" / "physics" / "per_rep.csv")
    df3 = pd.read_csv(ROOT / "results" / "meta_routes" / "per_rep.csv")
    seen = df3[df3.mid == "anil__mae__multi"]
    van = df4[df4.tag == "anilpi__vanilla__mae__no_diesel"]
    pi = df4[df4.tag == "anilpi__both__mae__no_diesel"]
    van_g = df4[df4.tag == "anilpi__vanilla__mae__no_gasoline"]
    pi_g = df4[df4.tag == "anilpi__both__mae__no_gasoline"]

    shots = [5, 10, 20]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.2))
    labels3 = ["vanilla\n(LOMO)", "PI dual\n(LOMO)", "seen-task\ncontrol"]
    colors3 = [COLORS["vanilla"], COLORS["both"], "#2e7d32"]
    for ax, K in zip(axes.flat[:3], shots):
        rows = []
        for t in DIESEL_TASKS:
            rows.append([van[(van.task == t) & (van.shot == K)].r2.mean(),
                         pi[(pi.task == t) & (pi.shot == K)].r2.mean(),
                         seen[(seen.task == t) & (seen.shot == K)].r2.mean()])
        rows = np.array(rows)
        x = np.arange(len(DIESEL_TASKS)); w = 0.26
        for i in range(3):
            ax.bar(x + (i - 1) * w, rows[:, i], w, color=colors3[i], label=labels3[i])
        ax.axhline(0, color="#333", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([t.replace("diesel-", "") for t in DIESEL_TASKS], fontsize=9)
        ax.set_title(f"Diesel extrapolation, K={K}", fontsize=11)
        ax.set_ylabel("R² (mean of 10 reps)")
        ax.set_ylim(-2.6, 1.05)
    ax = axes.flat[3]
    rows = np.array([[van_g[van_g.shot == K].r2.mean(),
                      pi_g[pi_g.shot == K].r2.mean(),
                      seen[(seen.task == "gasoline-octane") & (seen.shot == K)].r2.mean()]
                     for K in shots])
    x = np.arange(3); w = 0.26
    for i in range(3):
        ax.bar(x + (i - 1) * w, rows[:, i], w, color=colors3[i], label=labels3[i])
    ax.axhline(0, color="#333", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([f"K={K}" for K in shots])
    ax.set_title("Gasoline extrapolation (octane)", fontsize=11)
    ax.set_ylabel("R² (mean of 10 reps)")
    ax.set_ylim(-2.6, 1.05)
    axes.flat[0].legend(fontsize=8.5, loc="lower left")
    fig.suptitle("Leave-one-material-out extrapolation: physics constraints close the domain gap",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIG / "fig4_lomo_extrapolation.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("fig4 ok", flush=True)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "fig1"):
        fig_framework()
    if which in ("all", "fig4"):
        fig_lomo()
    if which in ("all", "fig3"):
        fig_additivity()
    if which in ("all", "fig2"):
        fig_saliency()
    print("done ->", FIG)
