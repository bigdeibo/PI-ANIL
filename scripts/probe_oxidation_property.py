"""Edible-oil peroxide value -- cross-substance LOMO physics-extrapolation probe
(the oil-type analogue of the diesel LOMO).

Data: edibleoil-pv (Lavine/Booksh Mendeley; FT-NIR 667-2632 nm, full C-H coverage
band=0.279; 100 oils x 3 replicates; PV 1.52-165). The strongest band-prior domain.

LOMO protocol (cross-substance extrapolation, leak-proof): meta-train ANIL/PI-ANIL
on the no_edibleoil pool (diesel/gasoline/corn/EVOO/self-collected; **without edible
oil**) -- the model never sees edible oil -- then few-shot-evaluate on edible-oil PV.
PI vs ANIL tests whether the C-H physics-constrained encoder still extrapolates
positively to a **new oil category** (strengthening the cross-substance physics axis).

Methods: PLS / frozen MAE+ridge / ANIL(mae, head-adapt) / PI-ANIL(mae, head-adapt) +
  encoder-isolated ANIL+ridge / PI+ridge (same ridge readout, isolating the encoder,
  avoiding the readout confound of T2).
Evaluation-time head adaptation is bit-identical to the main adaptation protocol; support/test split
group-aware by oil (100 oils, leakage-safe).
Outputs: results/probes/oxidation_property/{per_rep.csv, summary.txt}; encoders reuse the physics-variant checkpoints.
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

OUT = ROOT / "results" / "probes" / "oxidation_property"
OUT.mkdir(parents=True, exist_ok=True)
CKPT4 = ROOT / "results" / "physics" / "checkpoints"
POOL = "no_edibleoil"   # LOMO: training pool excludes edible oil
TASK = "edibleoil-pv"

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
    import torch
    head = torch.nn.Linear(EMB_DIM, 1).to(device)
    head.load_state_dict(head_state)
    Xt = torch.from_numpy(X)[:, None].to(device)
    yt = torch.from_numpy(y).float().to(device)
    ym = float(yt[tr].cpu().numpy().mean())
    ysd = float(yt[tr].cpu().numpy().std() + 1e-8)
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


def _enc_embeddings(enc, X, device="cpu"):
    import torch
    enc = enc.to(device).eval()
    Xt = torch.from_numpy(X)[:, None].to(device)
    zs = []
    with torch.no_grad():
        for i in range(0, len(Xt), 256):
            zs.append(enc(Xt[i:i + 256]).cpu().numpy())
    return np.vstack(zs).astype(np.float32)


def mae_embeddings(X, device="cpu"):
    from src.ssl import load_encoder
    return _enc_embeddings(load_encoder("mae"), X, device)


def fit_predict_ridge(Es, ys, Ete, alphas=np.logspace(-4, 4, 17)):
    from sklearn.linear_model import RidgeCV
    mu, sd = Es.mean(0), Es.std(0) + 1e-8
    m = RidgeCV(alphas=alphas)
    m.fit((Es - mu) / sd, ys)
    return m.predict((Ete - mu) / sd)


def main():
    import torch
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", help="train|eval|all")
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--episodes", type=int, default=2500)
    ap.add_argument("--time-budget", type=int, default=600)
    args = ap.parse_args()
    dev = args.device

    if args.stage in ("all", "train"):
        from src.pifsl.metatrain import meta_train_pi
        for variant in ("vanilla", "both"):
            print(f"\n=== LOMO meta-training {variant} on {POOL} (without edible oil) ===", flush=True)
            meta_train_pi(variant, "mae", POOL, n_episodes=args.episodes,
                          time_budget=args.time_budget, device=dev, seed=0)

    if args.stage not in ("all", "eval"):
        return

    d = load_task(TASK)
    X, y, groups = d["X"], d["y"], d["groups"]
    n = len(y)
    print(f"[task] {TASK}: n={n}, oils={len(set(groups))}, "
          f"PV {y.min():.1f}-{y.max():.1f}", flush=True)

    ck_anil = torch.load(CKPT4 / f"anilpi__vanilla__mae__{POOL}.pt",
                         map_location="cpu", weights_only=False)
    ck_pi = torch.load(CKPT4 / f"anilpi__both__mae__{POOL}.pt",
                       map_location="cpu", weights_only=False)
    print(f"[ckpt] ANIL ep{ck_anil.get('epoch')} best{ck_anil.get('best_val'):.4f} | "
          f"PI ep{ck_pi.get('epoch')} best{ck_pi.get('best_val'):.4f}", flush=True)
    encs = {"anil": (build_enc(ck_anil, dev), ck_anil["head"]),
            "pi": (build_enc(ck_pi, dev), ck_pi["head"])}
    E_mae = mae_embeddings(X, device=dev)
    E_anil = _enc_embeddings(encs["anil"][0], X, device=dev)
    E_pi = _enc_embeddings(encs["pi"][0], X, device=dev)

    rows = []
    for K in [5, 10, 20]:
        for rep in range(args.reps):
            tr, te = eval_split(n, K, rep, groups=groups)
            yte = y[te]
            rec = dict(shot=K, rep=rep)
            try:
                rec["PLS"] = r2_score(PLSBaseline().fit(X[tr], y[tr]).predict(X[te]), yte)
            except Exception:
                rec["PLS"] = np.nan
            try:
                rec["MAE_ridge"] = r2_score(fit_predict_ridge(E_mae[tr], y[tr], E_mae[te]), yte)
            except Exception:
                rec["MAE_ridge"] = np.nan
            for key, label in (("anil", "ANIL(mae)"), ("pi", "PI-ANIL(mae)")):
                enc, hs = encs[key]
                try:
                    rec[label] = r2_score(eval_episode_anil(enc, hs, X, y, tr, te, dev), yte)
                except Exception as ex:
                    rec[label] = np.nan
                    print(f"[fail] {label} K={K} rep={rep}: {ex}", flush=True)
            try:
                rec["ANIL+ridge"] = r2_score(fit_predict_ridge(E_anil[tr], y[tr], E_anil[te]), yte)
            except Exception:
                rec["ANIL+ridge"] = np.nan
            try:
                rec["PI+ridge"] = r2_score(fit_predict_ridge(E_pi[tr], y[tr], E_pi[te]), yte)
            except Exception:
                rec["PI+ridge"] = np.nan
            rows.append(rec)
        print(f"  [K={K}] {args.reps} reps done", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "per_rep.csv", index=False)
    cols = ["PLS", "MAE_ridge", "ANIL+ridge", "PI+ridge", "ANIL(mae)", "PI-ANIL(mae)"]
    lines = [f"Edible-oil peroxide value cross-substance LOMO probe (reps={args.reps}, device={dev})",
             f"Training pool: {POOL} (without edible oil) -> evaluate {TASK} (C-H band=0.279, strongest domain)",
             "=" * 96]
    lines.append("\nMedian R2 / crash rate:")
    lines.append(f"{'K':>3} | " + " ".join(f"{c:>18}" for c in cols))
    lines.append("-" * 96)
    for K in [5, 10, 20]:
        s = df[df.shot == K]
        cells = []
        for c in cols:
            v = s[c].dropna()
            coll = float((v < 0).mean()) if len(v) else float("nan")
            cells.append(f"{v.median():+7.3f}/{coll*100:3.0f}%" if len(v) else "      N/A      ")
        lines.append(f"{K:>3} | " + " ".join(f"{c:>18}" for c in cells))
    # PI vs ANIL (head-adapt readout)
    lines.append("\nPI-ANIL vs ANIL (head-adapt readout, paired Wilcoxon one-sided PI>ANIL):")
    lines.append(f"  {'K':>3} | {'med d':>8} {'PI>ANIL':>8} {'p':>9}")
    for K in [5, 10, 20]:
        s = df[df.shot == K]
        dd = (s["PI-ANIL(mae)"] - s["ANIL(mae)"]).dropna()
        if len(dd) < 3:
            continue
        try:
            pv = wilcoxon(s["PI-ANIL(mae)"].dropna(), s["ANIL(mae)"].dropna(),
                          alternative="greater").pvalue
        except Exception:
            pv = float("nan")
        lines.append(f"  {K:>3} | {dd.median():>+8.3f} {(dd>0).sum()}/{len(dd):>3} {pv:>9.4g}")
    # Encoder-isolated comparison (same ridge readout)
    lines.append("\nEncoder isolation (same ridge readout, median R2): MAE_ridge / ANIL+ridge / PI+ridge")
    for K in [5, 10, 20]:
        s = df[df.shot == K]
        cells = []
        for c in ["MAE_ridge", "ANIL+ridge", "PI+ridge"]:
            v = s[c].dropna()
            cells.append(f"{c}={v.median():+.3f}" if len(v) else f"{c}=NA")
        lines.append(f"  K={K}: " + "  ".join(cells))
    lines.append("Physics (encoder) PI+ridge vs ANIL+ridge (one-sided PI>ANIL):")
    for K in [5, 10, 20]:
        s = df[df.shot == K]
        dd = (s["PI+ridge"] - s["ANIL+ridge"]).dropna()
        if len(dd) < 3:
            continue
        try:
            pv = wilcoxon(s["PI+ridge"].dropna(), s["ANIL+ridge"].dropna(),
                          alternative="greater").pvalue
        except Exception:
            pv = float("nan")
        lines.append(f"  K={K}: d={dd.median():+.3f} {(dd>0).sum()}/{len(dd)} p={pv:.3g}")
    summary = "\n".join(lines)
    (OUT / "summary.txt").write_text(summary, encoding="utf-8")
    print("\n" + summary, flush=True)


if __name__ == "__main__":
    main()
