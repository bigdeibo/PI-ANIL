"""T3 self-collected FT-NIR n-pentane/CCl4 few-shot quantitation -- Phase 0 probe.

Task: predict the n-pentane volume fraction phi in CCl4 from FT-NIR spectra
(the physically optimal C-H scenario).
Data: data/pentane_ccl4/processed/pentane_ccl4_FTIR.npz (79 spectra / 10 vials; see the README there).
Two protocols (paired 30 reps, same seed scheme as the whole project seed*1000+K*100+rep):
  A vial-level LOVO (leak-proof): eval_split(groups=vial id), K support vials / remaining test
    vials, with zero support/test vial overlap (no same-vial replicate crossing). K=3,5.
  B spectrum-level standard (comparable to the 13 tasks): eval_split(groups=None), K=5/10/20;
    same-vial spectra may cross the boundary (declared as mild replicate optimism, as EVOO before the fix).
Methods: PLS / frozen MAE + ridge / ANIL(mae,t3) / PI-ANIL(mae,t3).
  The only difference between ANIL and PI-ANIL = the C-H band + additivity physical
  constraints (band_saliency + additivity).
Evaluation-time head adaptation is bit-identical to the main adaptation protocol (50 head-only steps +
spectral-radius adaptive step size + +/-5 clamp).
Outputs: results/probes/within_instrument/{per_rep.csv, summary.txt}.

CLI:
  python scripts/probe_within_instrument.py --methods pls,mae_ridge --device cuda        # Step1 baselines
  python scripts/probe_within_instrument.py --methods pls,mae_ridge,anil,pi --device cuda # Step3 full
"""
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.fsl_regression.core import (load_task, eval_split, r2_score, EMB_DIM, L)
from src.baselines import PLSBaseline

OUT = ROOT / "results" / "probes" / "within_instrument"
OUT.mkdir(parents=True, exist_ok=True)
CKPT4 = ROOT / "results" / "physics" / "checkpoints"

# ---- Evaluation-time head adaptation (copied bit-for-bit from probe_corn_instruments.py, matching the main adaptation protocol) ----
EVAL_INNER_STEPS = 50


def _stable_lr(zs, safety=0.8):
    import torch
    H = 2.0 * (zs.T @ zs) / len(zs)
    lam = float(torch.linalg.eigvalsh(H).max())
    return safety / max(lam, 1e-6)


def build_enc(ck, device="cpu"):
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
    ym, ysd = float(yt[tr].cpu().numpy().mean()), float(yt[tr].cpu().numpy().std() + 1e-8)
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


def mae_embeddings(X, device="cpu"):
    """Frozen MAE encoder -> (n,128) embeddings (X already at 512+SNV)."""
    import torch
    from src.ssl import load_encoder
    enc = load_encoder("mae").to(device).eval()
    Xt = torch.from_numpy(X)[:, None].to(device)
    zs = []
    with torch.no_grad():
        for i in range(0, len(Xt), 256):
            zs.append(enc(Xt[i:i + 256]).cpu().numpy())
    return np.vstack(zs).astype(np.float32)


def fit_predict_ridge(Es, ys, Ete, alphas=np.logspace(-4, 4, 17)):
    from sklearn.linear_model import RidgeCV
    mu, sd = Es.mean(0), Es.std(0) + 1e-8
    m = RidgeCV(alphas=alphas)
    m.fit((Es - mu) / sd, ys)
    return m.predict((Ete - mu) / sd)


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="pls,mae_ridge,anil,pi")
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ck-source", default="t3",
                    help="source tag of the meta-learning checkpoint (t3 = the 14-task pool including selfmix)")
    args = ap.parse_args()
    methods = [m.strip() for m in args.methods.split(",")]
    dev = args.device

    d = load_task("selfmix-phi")
    X, y, groups = d["X"], d["y"], d["groups"]
    n = len(y)
    print(f"[task] selfmix-phi: n={n}, vials={len(set(groups))}, "
          f"φ∈{sorted(set(round(float(v),3) for v in y))}", flush=True)

    # Prepare each method's encoder / embeddings
    E_mae = mae_embeddings(X, device=dev) if "mae_ridge" in methods else None
    encs = {}
    if "anil" in methods:
        ck = torch.load(CKPT4 / f"anilpi__vanilla__mae__{args.ck_source}.pt",
                        map_location="cpu", weights_only=False)
        encs["anil"] = (build_enc(ck, dev), ck["head"])
        print(f"[ckpt] ANIL vanilla/{args.ck_source}: ep{ck.get('epoch')} best_val{ck.get('best_val'):.4f}",
              flush=True)
    if "pi" in methods:
        ck = torch.load(CKPT4 / f"anilpi__both__mae__{args.ck_source}.pt",
                        map_location="cpu", weights_only=False)
        encs["pi"] = (build_enc(ck, dev), ck["head"])
        print(f"[ckpt] PI-ANIL both/{args.ck_source}: ep{ck.get('epoch')} best_val{ck.get('best_val'):.4f}",
              flush=True)

    PROTO = [("A_vial", [3, 5], groups), ("B_spec", [5, 10, 20], None)]
    rows = []
    for pname, shots, g in PROTO:
        for K in shots:
            for rep in range(args.reps):
                tr, te = eval_split(n, K, rep, groups=g)
                yt = y[te]
                rec = dict(protocol=pname, shot=K, rep=rep)
                if "pls" in methods:
                    try:
                        rec["PLS"] = r2_score(PLSBaseline().fit(X[tr], y[tr]).predict(X[te]), yt)
                    except Exception:
                        rec["PLS"] = np.nan
                if "mae_ridge" in methods:
                    try:
                        rec["MAE+ridge"] = r2_score(fit_predict_ridge(E_mae[tr], y[tr], E_mae[te]), yt)
                    except Exception:
                        rec["MAE+ridge"] = np.nan
                for key in ("anil", "pi"):
                    if key in encs:
                        enc, hs = encs[key]
                        label = "ANIL(mae)" if key == "anil" else "PI-ANIL(mae)"
                        try:
                            p = eval_episode_anil(enc, hs, X, y, tr, te, device=dev)
                            rec[label] = r2_score(p, yt)
                        except Exception as ex:
                            rec[label] = np.nan
                            print(f"[fail] {label} {pname} K={K} rep={rep}: {ex}", flush=True)
                rows.append(rec)
            print(f"  [{pname} K={K}] done", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "per_rep.csv", index=False)
    cols = [c for c in ["PLS", "MAE+ridge", "ANIL(mae)", "PI-ANIL(mae)"] if c in df.columns]

    lines = [f"T3 selfmix n-pentane/CCl4 few-shot quantitation Phase 0 (reps={args.reps}, device={dev})",
             f"Task: n={n} spectra / {len(set(groups))} vials, phi in {sorted(set(round(float(v),3) for v in y))}",
             "=" * 92]
    for pname, _, _ in PROTO:
        lines.append(f"\n===== Protocol {pname} (mean / median R2 / crash rate) =====")
        lines.append(f"{'K':>3} | " + " ".join(f"{c:>18}" for c in cols))
        lines.append("-" * 92)
        for K in sorted(df[df.protocol == pname].shot.unique()):
            s = df[(df.protocol == pname) & (df.shot == K)]
            cells = []
            for c in cols:
                v = s[c].dropna()
                if len(v):
                    # The mean is dragged down by PLS blow-ups; median R2 + crash rate
                    # (fraction with R2<0) are the robust signals.
                    coll = float((v < 0).mean())
                    cells.append(f"{v.mean():+6.2f}/{v.median():+6.2f}/{coll*100:2.0f}%")
                else:
                    cells.append("      N/A      ")
            lines.append(f"{K:>3} | " + " ".join(f"{c:>18}" for c in cells))
        if "ANIL(mae)" in cols and "PI-ANIL(mae)" in cols:
            lines.append(f"\n  PI-ANIL vs ANIL paired Wilcoxon (one-sided PI>ANIL):")
            lines.append(f"  {'K':>3} | {'med d':>8} {'PI>ANIL':>8} {'p':>9}")
            for K in sorted(df[df.protocol == pname].shot.unique()):
                s = df[(df.protocol == pname) & (df.shot == K)]
                dd = (s["PI-ANIL(mae)"] - s["ANIL(mae)"]).dropna()
                if len(dd) < 3:
                    continue
                try:
                    p = wilcoxon(s["PI-ANIL(mae)"], s["ANIL(mae)"], alternative="greater").pvalue
                except Exception:
                    p = float("nan")
                lines.append(f"  {K:>3} | {dd.median():>+8.3f} {(dd > 0).sum()}/{len(dd):>3} {p:>9.4g}")
    summary = "\n".join(lines)
    (OUT / "summary.txt").write_text(summary, encoding="utf-8")
    print("\n" + summary, flush=True)


if __name__ == "__main__":
    main()
