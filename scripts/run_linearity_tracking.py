"""P1a: track the embedding-linearity curve over the full course of vanilla and
both-constraints (both) meta-training (closes R1-M2).

Adjudicates the "drift prevention" narrative:
  - If the deviation of vanilla during training is systematically higher than that of
    both, drift exists and is suppressed by the constraints -> narrative holds.
  - If the vanilla deviation stays flat throughout and is comparable to / lower than
    that of both, the narrative must be revised to "accelerates convergence to a
    linear geometry".

Probe: every probe_every episodes, compute the embedding-linearity deviation of the
current encoder on real EVOO+HZO blend spectra:
dev = ||z_mix - ((1-c) z_EVOO + c z_HZO)|| / ||z_mix||.
Bit-identical to the terminal-state dev computation in mechanism_audit.py / Fig. 3,
so the endpoints are cross-checkable.

Retrains to new checkpoint paths (__probe suffix); the original checkpoints are kept
untouched.
Outputs: results/physics/probe_linearity.csv (variant, epoch, dev) + figs/probe_linearity.png

Usage: python scripts/probe_linearity_train.py [--max-chunks 15]
"""
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from src.datasets import load_evoo  # noqa: E402
from src.ssl import snv  # noqa: E402
from src.fsl_regression.core import L  # noqa: E402
from src.fsl_regression.metatrain import meta_train  # noqa: E402
from src.pifsl.metatrain import meta_train_pi  # noqa: E402

OUT = ROOT / "results" / "physics"
FIG = OUT / "figs"
CK3 = ROOT / "results" / "meta_routes" / "checkpoints"
CK4 = ROOT / "results" / "physics" / "checkpoints"
PATIENCE = 4


def prepare_probe_data():
    """Full EVOO spectra (512 points + SNV) + sample indices for pure EVOO / pure HZO /
    blends + HZO concentrations."""
    e = load_evoo()
    X = np.asarray(e["X"], float)
    go, gn = np.linspace(0, 1, X.shape[1]), np.linspace(0, 1, L)
    X = np.stack([np.interp(gn, go, r) for r in X])
    X = snv(X).astype(np.float32)
    labels = np.asarray(e["labels"])
    levels = np.asarray(e["targets"]["adulteration_level"], float)
    Xt = torch.from_numpy(X)[:, None]
    ev_idx = np.where(labels == "EVOO")[0]
    hz_idx = np.where(labels == "Hazelnut oil")[0]
    mix_idx = np.where(labels == "EVOO+HZO")[0]
    cs = levels[mix_idx] / 100.0  # HZO concentration
    return Xt, ev_idx, hz_idx, mix_idx, cs


def make_probe(Xt, ev_idx, hz_idx, mix_idx, cs):
    """Return (probe_fn, records). probe_fn(enc, ep) records the mean dev of the
    current encoder."""
    rec = []

    def probe(enc, ep):
        was_training = enc.training
        enc.eval()
        with torch.no_grad():
            Z = enc(Xt).numpy()
        if was_training:
            enc.train()
        z_ev = Z[ev_idx].mean(0)
        z_hz = Z[hz_idx].mean(0)
        Zm = Z[mix_idx]
        target = (1 - cs)[:, None] * z_ev + cs[:, None] * z_hz
        dev = np.linalg.norm(Zm - target, axis=1) / (np.linalg.norm(Zm, axis=1) + 1e-12)
        rec.append((ep, float(dev.mean())))
        print(f"    [probe] ep {ep}: mean_dev={dev.mean():.4f}", flush=True)

    return probe, rec


def train_until_done(train_fn, ckpt_file, max_chunks):
    """Resume training in chunks until early stopping (bad >= patience) or max_chunks
    is exhausted. Returns the final epoch."""
    if ckpt_file.exists():
        ckpt_file.unlink()  # train from scratch to keep the curve complete
    prev_ep = 0
    for chunk in range(max_chunks):
        train_fn()
        ck = torch.load(ckpt_file, map_location="cpu", weights_only=False)
        ep, bad = ck["epoch"], ck.get("bad", 0)
        print(f"  [chunk {chunk}] ep={ep} bad={bad} best_val={ck['best_val']:.4f}",
              flush=True)
        if bad >= PATIENCE:
            print(f"  early stopping confirmed at ep {ep}", flush=True)
            return ep
        if ep <= prev_ep:  # no-progress guard
            return ep
        prev_ep = ep
    print(f"  reached max_chunks, stopped at ep {prev_ep}", flush=True)
    return prev_ep


def run_variant(name, train_fn, ckpt_file, probe, max_chunks):
    print(f"\n=== retraining {name} with probe ===", flush=True)
    final_ep = train_until_done(train_fn, ckpt_file, max_chunks)
    print(f"=== {name} done, final ep={final_ep}, probe points {len(probe[1])} ===",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-chunks", type=int, default=15)
    ap.add_argument("--variants", default="vanilla,both")
    args = ap.parse_args()
    variants = args.variants.split(",")

    Xt, ev_idx, hz_idx, mix_idx, cs = prepare_probe_data()
    print(f"probe data: {Xt.shape[0]} full EVOO spectra; "
          f"pure EVOO {len(ev_idx)} / pure HZO {len(hz_idx)} / blends {len(mix_idx)}",
          flush=True)

    rows = []

    if "vanilla" in variants:
        probe_v, rec_v = make_probe(Xt, ev_idx, hz_idx, mix_idx, cs)
        ckpt_v = CK3 / "anil__mae__multi__probe.pt"
        run_variant(
            "vanilla(anil)",
            lambda: meta_train("anil", "mae", "multi", n_episodes=2500,
                               tag_suffix="__probe", probe_fn=probe_v,
                               probe_every=100),
            ckpt_v, (probe_v, rec_v), args.max_chunks)
        rows += [dict(variant="vanilla", epoch=ep, dev=d) for ep, d in rec_v]

    if "both" in variants:
        probe_b, rec_b = make_probe(Xt, ev_idx, hz_idx, mix_idx, cs)
        ckpt_b = CK4 / "anilpi__both__probe__mae__multi.pt"
        run_variant(
            "both(anilpi)",
            lambda: meta_train_pi("both", "mae", "multi", n_episodes=2500,
                                  tag_suffix="__probe", probe_fn=probe_b,
                                  probe_every=100, time_budget=240),
            ckpt_b, (probe_b, rec_b), args.max_chunks)
        rows += [dict(variant="both", epoch=ep, dev=d) for ep, d in rec_b]

    df = pd.DataFrame(rows)
    out_csv = OUT / "probe_linearity.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nsaved -> {out_csv} ({len(df)} points)")

    # plot the curve + terminal-state reference
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        colors = dict(vanilla="#7f7f7f", both="#d62728")
        for v in df.variant.unique():
            sub = df[df.variant == v].sort_values("epoch")
            ax.plot(sub.epoch, sub.dev, color=colors.get(v, "b"), lw=1.6,
                    marker="o", ms=3, label=f"{v} (training dev)")
        # terminal-state reference (final-checkpoint dev from mechanism_audit.csv)
        audit = OUT / "mechanism_audit.csv"
        if audit.exists():
            au = pd.read_csv(audit)
            term = au[au.metric == "additivity_dev_mean"].set_index("variant")["value"]
            for v, c in [("vanilla", "#7f7f7f"), ("both", "#d62728")]:
                if v in term.index:
                    ax.axhline(term[v], color=c, ls="--", lw=1, alpha=0.6,
                               label=f"{v} final ckpt={term[v]:.4f}")
        ax.set_xlabel("Meta-training episode")
        ax.set_ylabel("Embedding linearity deviation (mean)")
        ax.set_title("Does vanilla drift? Embedding linearity over training")
        ax.legend(fontsize=8.5)
        fig.tight_layout()
        fig.savefig(FIG / "probe_linearity.png", dpi=200)
        plt.close(fig)
        print(f"fig -> {FIG / 'probe_linearity.png'}")
    except Exception as ex:
        print(f"[warn] plotting failed: {ex}")


if __name__ == "__main__":
    main()
