"""T1 corn cross-instrument breadth (calibration-transfer few-shot; Option A reuses checkpoints).

Research question: can an encoder meta-trained on the source instrument m5
(+ the 13-task cross-substance pool) transfer to the target instruments mp5/mp6
(never seen in meta-training)? Does the C-H/O-H band prior (same wavelength grid
across the three instruments -> instrument-invariant mask) help cross-instrument
transfer (a falsifiable test of "physics = instrument-invariance inductive bias")?

Leakage control (verified; not an EVOO-style leak):
  - Meta-training corn uses only m5 (the load_task corn branch) -> spectra of the
    target instruments mp5/mp6 never enter meta-training.
  - MAE initialization saw unlabeled corn spectra from all three instruments =
    transductive SSL (declared in Sec. 2.2); leakage concerns label / test-sample
    identity, not unlabeled spectra.
  - Paired samples (one physical sample measured once per instrument) are the
    standard premise of calibration transfer, not a leak.
  - eval_split guarantees no support/test overlap within a target instrument
    (same protocol as the main benchmark).

Methods (rep 0..reps-1, K=5/10/20, 4 attributes):
  - ANIL(mae)    = anilpi__vanilla__mae__multi (same code path as vanilla, physics constraints removed)
  - PI-ANIL(mae) = anilpi__both__mae__multi    (both = additivity + band saliency)
  - PLS-target   = PLS trained K-shot on the target instrument (per-instrument upper bound)
  - PLS-m5->tgt  = PLS trained on all m5 samples, predicting the target directly
    (zero-shot transfer lower bound; quantifies instrument offset)
Evaluation-time adaptation: 50-step head-only GD + Hessian-spectral-radius adaptive step
size + +/-5 clamp (bit-identical to the main adaptation protocol).
Outputs: results/probes/corn_instruments/{per_rep.csv, summary.txt}.
"""
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.datasets import load_corn
from src.ssl import snv
from src.fsl_regression.core import eval_split, zscore_fit, r2_score, EMB_DIM, L
from src.baselines import PLSBaseline

OUT = ROOT / "results" / "probes" / "corn_instruments"
OUT.mkdir(parents=True, exist_ok=True)
CKPT4 = ROOT / "results" / "physics" / "checkpoints"


def corn_inst_Xy(inst, attr, idx=None):
    """Load one corn attribute for one instrument -> (X(n,512) SNV+resampled, y(n,)).

    Preprocessing is bit-identical to core.load_task: per-row linear interpolation
    to 512 (position normalization) followed by SNV.
    idx (optional): subset by original sample index (Option B holdout/train subset);
    None = all 80 samples.
    """
    c = load_corn()
    X = c["instruments"][inst][0]
    y = c["targets"][attr]
    m = ~np.isnan(y)
    X, y = np.asarray(X, float)[m], np.asarray(y, float)[m]
    if idx is not None:
        X, y = X[np.asarray(idx)], y[np.asarray(idx)]
    go = np.linspace(0, 1, X.shape[1])
    gn = np.linspace(0, 1, L)
    X = np.stack([np.interp(gn, go, r) for r in X])
    return snv(X).astype(np.float32), y


# ---- Evaluation-time head adaptation (copied bit-for-bit from scripts/run_stage4_part.py, matching the main adaptation protocol) ----
EVAL_INNER_STEPS = 50


def _stable_lr(zs, safety=0.8):
    import torch
    H = 2.0 * (zs.T @ zs) / len(zs)
    lam = float(torch.linalg.eigvalsh(H).max())
    return safety / max(lam, 1e-6)


def build_enc(ck, device="cpu"):
    """Build the encoder once from a checkpoint (avoid rebuilding per evaluation)."""
    import torch
    from src.fsl.models import ResNet1Encoder
    enc = ResNet1Encoder(L, emb_dim=EMB_DIM)
    enc.load_state_dict(ck["enc"])
    enc.eval()
    return enc.to(device)


def eval_episode_anil(enc, head_state, X, y, tr, te, device="cpu"):
    """50-step head-only adaptation on the support set, then predict the test set
    (original scale). The head is reset every call."""
    import torch
    head = torch.nn.Linear(EMB_DIM, 1).to(device)
    head.load_state_dict(head_state)
    Xt = torch.from_numpy(X)[:, None].to(device)
    yt = torch.from_numpy(y).float().to(device)
    ym, ysd = zscore_fit(yt[tr].cpu().numpy())
    ys_z = (yt[tr] - ym) / ysd
    with torch.no_grad():
        zs = enc(Xt[tr])
        mu, sd = zs.mean(0, keepdim=True), zs.std(0, keepdim=True) + 1e-8
        zs = ((zs - mu) / sd).clamp(-5, 5)
    lr_in = _stable_lr(zs)
    prev = np.inf
    for _ in range(EVAL_INNER_STEPS):
        loss = torch.mean((head(zs).squeeze(1) - ys_z) ** 2)
        lv = float(loss.item())
        if not np.isfinite(lv) or lv > 50.0 or lv > prev * 3:
            break
        prev = lv
        g = torch.autograd.grad(loss, list(head.parameters()))
        with torch.no_grad():
            for p, gr in zip(head.parameters(), g):
                p.sub_(lr_in * gr)
    with torch.no_grad():
        zq = ((enc(Xt[te]) - mu) / sd).clamp(-5, 5)
        pred = head(zq).squeeze(1)
    return pred.cpu().numpy() * ysd + ym


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--attr", default="moisture,oil,protein,starch",
                    help="comma-separated attributes")
    ap.add_argument("--shots", default="5,10,20")
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--targets", default="m5,mp5,mp6")
    ap.add_argument("--source", default="m5")
    ap.add_argument("--device", default="cpu", help="cpu / cuda")
    ap.add_argument("--ck-suffix", default="",
                    help="checkpoint tag suffix (A-gpu control uses __xinstAgpu); default empty = canonical")
    args = ap.parse_args()
    attrs = [a.strip() for a in args.attr.split(",")]
    shots = [int(s) for s in args.shots.split(",")]
    targets = args.targets.split(",")
    source = args.source

    ck_anil = torch.load(CKPT4 / f"anilpi__vanilla{args.ck_suffix}__mae__multi.pt",
                         map_location="cpu", weights_only=False)
    ck_pi = torch.load(CKPT4 / f"anilpi__both{args.ck_suffix}__mae__multi.pt",
                       map_location="cpu", weights_only=False)
    enc_anil = build_enc(ck_anil, device=args.device)
    enc_pi = build_enc(ck_pi, device=args.device)
    print(f"[ckpt] ANIL={ck_anil.get('variant')}/{ck_anil.get('epoch')}ep  "
          f"PI={ck_pi.get('variant')}/{ck_pi.get('epoch')}ep", flush=True)

    rows = []
    for attr in attrs:
        X_src, y_src = corn_inst_Xy(source, attr)
        pls_src = PLSBaseline().fit(X_src, y_src)
        data = {t: corn_inst_Xy(t, attr) for t in set(targets) | {source}}
        for tgt in targets:
            X_t, y_t = data[tgt]
            n = len(y_t)
            for K in shots:
                if n <= K + 5:
                    continue
                for rep in range(args.reps):
                    tr, te = eval_split(n, K, rep)
                    yt = y_t[te]
                    rec = dict(source=source, target=tgt, attr=attr, shot=K, rep=rep)
                    for name, enc, hs in (("ANIL(mae)", enc_anil, ck_anil["head"]),
                                          ("PI-ANIL(mae)", enc_pi, ck_pi["head"])):
                        try:
                            p = eval_episode_anil(enc, hs, X_t, y_t, tr, te, device=args.device)
                            rec[name] = r2_score(p, yt)
                        except Exception as ex:
                            rec[name] = np.nan
                            print(f"[fail] {name} {attr} {tgt} K={K} rep={rep}: {ex}",
                                  flush=True)
                    try:
                        p = PLSBaseline().fit(X_t[tr], y_t[tr]).predict(X_t[te])
                        rec["PLS-target"] = r2_score(p, yt)
                    except Exception:
                        rec["PLS-target"] = np.nan
                    if tgt != source:
                        try:
                            p = pls_src.predict(X_t[te])
                            rec["PLS-m5->tgt(0shot)"] = r2_score(p, yt)
                        except Exception:
                            rec["PLS-m5->tgt(0shot)"] = np.nan
                    else:
                        rec["PLS-m5->tgt(0shot)"] = np.nan
                    rows.append(rec)
                print(f"  [{attr} {source}->{tgt} K={K}] done", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "per_rep.csv", index=False)
    methods = ["ANIL(mae)", "PI-ANIL(mae)", "PLS-target", "PLS-m5->tgt(0shot)"]

    lines = [f"T1 corn cross-instrument breadth (attrs={attrs}, reps={args.reps}, source={source})",
             "=" * 92]
    for attr in attrs:
        lines.append(f"\n===== attr={attr} (mean R2 +/- std) =====")
        lines.append(f"{'tgt':>4} {'K':>3} | " + " ".join(f"{m:>14}" for m in methods))
        lines.append("-" * 92)
        for tgt in targets:
            for K in shots:
                s = df[(df.attr == attr) & (df.target == tgt) & (df.shot == K)]
                if len(s) == 0:
                    continue
                cells = []
                for m in methods:
                    v = s[m].dropna()
                    cells.append(f"{v.mean():+8.3f}±{v.std():4.2f}" if len(v) else "   N/A      ")
                lines.append(f"{tgt:>4} {K:>3} | " + " ".join(f"{c:>14}" for c in cells))
    lines.append("\n===== PI-ANIL vs ANIL paired Wilcoxon (one-sided PI>ANIL) =====")
    lines.append(f"{'attr':>9} {'tgt':>4} {'K':>3} | {'med d':>8} {'PI>ANIL':>8} {'p':>8}")
    lines.append("-" * 60)
    for attr in attrs:
        for tgt in targets:
            for K in shots:
                s = df[(df.attr == attr) & (df.target == tgt) & (df.shot == K)]
                if len(s) < 3:
                    continue
                d = (s["PI-ANIL(mae)"] - s["ANIL(mae)"]).dropna()
                if len(d) < 3:
                    continue
                try:
                    p = wilcoxon(s["PI-ANIL(mae)"], s["ANIL(mae)"],
                                 alternative="greater").pvalue
                except Exception:
                    p = float("nan")
                lines.append(f"{attr:>9} {tgt:>4} {K:>3} | {d.median():>+8.3f} "
                             f"{(d > 0).sum()}/{len(d):>3} {p:>8.4f}")
    summary = "\n".join(lines)
    (OUT / "summary.txt").write_text(summary, encoding="utf-8")
    print("\n" + summary, flush=True)


if __name__ == "__main__":
    main()
