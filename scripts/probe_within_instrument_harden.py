"""T3 self-collected FT-NIR n-pentane/CCl4 few-shot quantitation -- hardening
(100 reps x 3 training seeds).

Phase 0 (30 reps x single seed) gave PI>ANIL p=0.018 (K=5) under protocol A. This
script hardens it:
  (1) evaluation reps 30 -> 100 (covers evaluation-split randomness);
  (2) meta-training seeds 1 -> 3 (seeds 0/1/2, covering training randomness --
     the one flagship weakness of Phase 0).
For each (rep, seed) pair it computes the paired PI/ANIL R2, then runs:
  - per-seed paired Wilcoxon (training-robust);
  - paired Wilcoxon + bootstrap 95% CI on the seed-mean R2 (the training-robust headline statistic).

Data / protocol / evaluation-time head adaptation are bit-identical to probe_within_instrument.py
(50 head-only steps + spectral-radius adaptive step size + +/-5 clamp). The only extension =
multi-seed checkpoint loading + aggregation.

Checkpoint naming: seed0 = anilpi__{variant}__mae__t3.pt (the original Phase 0 file);
seed1/2 = ..._s1/_s2... (_t3_harden_run.sh first copies the original checkpoint to _s0,
then trains s1/s2).

Outputs: results/probes/within_instrument/{harden_per_rep.csv, harden_summary.txt}.
"""
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.fsl_regression.core import load_task, eval_split, r2_score, EMB_DIM, L
from src.baselines import PLSBaseline

OUT = ROOT / "results" / "probes" / "within_instrument"
OUT.mkdir(parents=True, exist_ok=True)
CKPT4 = ROOT / "results" / "physics" / "checkpoints"

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
    (original scale). The head is reset every call. Bit-identical to probe_within_instrument.py."""
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


def ck_path(variant, seed):
    sfx = "" if seed == 0 else f"_s{seed}"
    return CKPT4 / f"anilpi__{variant}{sfx}__mae__t3.pt"


def boot_ci(deltas, B=10000, seed0=12345):
    """Bootstrap 95% CI (percentile method) of the median of deltas."""
    deltas = np.asarray(deltas, float)
    deltas = deltas[np.isfinite(deltas)]
    if len(deltas) < 3:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed0)
    n = len(deltas)
    meds = np.empty(B)
    for i in range(B):
        idx = rng.integers(0, n, n)
        meds[i] = np.median(deltas[idx])
    return (float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5)))


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--pls", action="store_true", help="include PLS/MAE+ridge baselines (included by default)")
    ap.add_argument("--no-baseline", action="store_true")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    dev = args.device
    with_base = not args.no_baseline
    # The Windows terminal (GBK) cannot encode 2/+-/CI characters; force stdout utf-8
    # (summary.txt is already written to disk as utf-8).
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    d = load_task("selfmix-phi")
    X, y, groups = d["X"], d["y"], d["groups"]
    n = len(y)
    print(f"[task] selfmix-phi: n={n}, vials={len(set(groups))}, "
          f"φ∈{sorted(set(round(float(v),3) for v in y))}", flush=True)

    E_mae = mae_embeddings(X, device=dev) if with_base else None
    encs = {}  # f"{anil|pi}_s{seed}" -> (enc, head_state)
    for variant, label in [("vanilla", "anil"), ("both", "pi")]:
        for s in seeds:
            p = ck_path(variant, s)
            if not p.exists():
                print(f"[warn] missing {p}, skipping {label}_s{s}", flush=True)
                continue
            ck = torch.load(p, map_location="cpu", weights_only=False)
            encs[f"{label}_s{s}"] = (build_enc(ck, dev), ck["head"])
            print(f"[ckpt] {label}_s{s}: ep{ck.get('epoch')} best_val{ck.get('best_val'):.4f}",
                  flush=True)

    PROTO = [("A_vial", [3, 5], groups), ("B_spec", [5, 10, 20], None)]
    rows = []
    for pname, shots, g in PROTO:
        for K in shots:
            for rep in range(args.reps):
                tr, te = eval_split(n, K, rep, groups=g)
                yt = y[te]
                rec = dict(protocol=pname, shot=K, rep=rep)
                if with_base:
                    try:
                        rec["PLS"] = r2_score(PLSBaseline().fit(X[tr], y[tr]).predict(X[te]), yt)
                    except Exception:
                        rec["PLS"] = np.nan
                    try:
                        rec["MAE_ridge"] = r2_score(fit_predict_ridge(E_mae[tr], y[tr], E_mae[te]), yt)
                    except Exception:
                        rec["MAE_ridge"] = np.nan
                for label in ("anil", "pi"):
                    for s in seeds:
                        key = f"{label}_s{s}"
                        if key not in encs:
                            continue
                        enc, hs = encs[key]
                        col = f"{'ANIL' if label=='anil' else 'PI'}_s{s}"
                        try:
                            p = eval_episode_anil(enc, hs, X, y, tr, te, device=dev)
                            rec[col] = r2_score(p, yt)
                        except Exception as ex:
                            rec[col] = np.nan
                            print(f"[fail] {col} {pname} K={K} rep={rep}: {ex}", flush=True)
                rows.append(rec)
            print(f"  [{pname} K={K}] {args.reps} reps done", flush=True)

    df = pd.DataFrame(rows)
    # Seed mean (training-robust estimate)
    anil_cols = [f"ANIL_s{s}" for s in seeds if f"ANIL_s{s}" in df.columns]
    pi_cols = [f"PI_s{s}" for s in seeds if f"PI_s{s}" in df.columns]
    if anil_cols:
        df["ANIL_mean"] = df[anil_cols].mean(axis=1)
    if pi_cols:
        df["PI_mean"] = df[pi_cols].mean(axis=1)
    df.to_csv(OUT / "harden_per_rep.csv", index=False)

    # ---- Summary ----
    lines = [f"T3 selfmix hardening (reps={args.reps}, seeds={seeds}, device={dev})",
             f"Task: n={n} spectra / {len(set(groups))} vials, phi in {sorted(set(round(float(v),3) for v in y))}",
             "=" * 92]
    base_cols = [c for c in ["PLS", "MAE_ridge"] if c in df.columns]
    for pname, _, _ in PROTO:
        lines.append(f"\n===== Protocol {pname} (median R2 / crash rate; ANIL/PI are seed means) =====")
        show = base_cols + (["ANIL_mean", "PI_mean"] if anil_cols else [])
        lines.append(f"{'K':>3} | " + " ".join(f"{c:>18}" for c in show))
        lines.append("-" * 92)
        for K in sorted(df[df.protocol == pname].shot.unique()):
            s = df[(df.protocol == pname) & (df.shot == K)]
            cells = []
            for c in show:
                v = s[c].dropna()
                coll = float((v < 0).mean()) if len(v) else float("nan")
                cells.append(f"{v.median():+7.3f}/{coll*100:3.0f}%" if len(v) else "      N/A      ")
            lines.append(f"{K:>3} | " + " ".join(f"{c:>18}" for c in cells))

        # Per-seed + seed-mean PI vs ANIL
        if anil_cols and pi_cols:
            lines.append(f"\n  PI-ANIL vs ANIL (per-seed paired Wilcoxon, one-sided PI>ANIL):")
            lines.append(f"  {'K':>3} | " + " ".join(f"{'seed'+str(s):>18}" for s in seeds) +
                         f"   {'mean(3seed)':>18}")
            for K in sorted(df[df.protocol == pname].shot.unique()):
                s = df[(df.protocol == pname) & (df.shot == K)]
                cells = []
                for sd_ in seeds:
                    ac, pc = f"ANIL_s{sd_}", f"PI_s{sd_}"
                    if ac in s.columns and pc in s.columns:
                        sub = s[[ac, pc]].dropna()  # row-aligned before the paired test
                        if len(sub) >= 3:
                            dd = sub[pc] - sub[ac]
                            try:
                                pv = wilcoxon(sub[pc], sub[ac],
                                              alternative="greater").pvalue
                            except Exception:
                                pv = float("nan")
                            cells.append(f"{dd.median():+.3f},p={pv:.3g}")
                        else:
                            cells.append("       N/A        ")
                    else:
                        cells.append("       N/A        ")
                # Seed mean
                sub = s[["ANIL_mean", "PI_mean"]].dropna()
                dd = sub["PI_mean"] - sub["ANIL_mean"]
                try:
                    pmean = wilcoxon(sub["PI_mean"], sub["ANIL_mean"],
                                     alternative="greater").pvalue
                except Exception:
                    pmean = float("nan")
                cells.append(f"{dd.median():+.3f},p={pmean:.3g}")
                lines.append(f"  {K:>3} | " + " ".join(f"{c:>18}" for c in cells))
            # Bootstrap CI of the seed-mean on its own line
            lines.append(f"\n  Seed-mean delta(PI-ANIL) median + bootstrap 95% CI:")
            for K in sorted(df[df.protocol == pname].shot.unique()):
                s = df[(df.protocol == pname) & (df.shot == K)]
                dd = (s["PI_mean"] - s["ANIL_mean"]).dropna().values
                lo, hi = boot_ci(dd)
                pos = int((dd > 0).sum())
                lines.append(f"  K={K}: med delta={np.median(dd):+.3f}  95%CI=[{lo:+.3f},{hi:+.3f}]  "
                             f"PI>ANIL {pos}/{len(dd)}")
    summary = "\n".join(lines)
    (OUT / "harden_summary.txt").write_text(summary, encoding="utf-8")
    print("\n" + summary, flush=True)


if __name__ == "__main__":
    main()
