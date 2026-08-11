"""T2 mango cross-instrument Phase 0 -- leave-instrument-out + fruit-disjoint
(LOIO + fruit-disjoint), cross-domain transfer from the existing MAE encoder.

Data: data/spectroscopy-fewshot-benchmark-main/data_tmp/MangoDMC_NIR_Data_v4.csv
  85401 spectra / 31 instruments / dry_matter (7.99-27.81) / 285-1200 nm Vis-NIR / 306 points.
  Structural caveat: 67% of fruits (reference_no) span multiple instruments, so a per-instrument
  fruit-disjoint LOIO is infeasible.
  -> This script splits fruits 80/20 by reference_no (source/target disjoint), holds out 2 target
    instruments, meta-trains on the source-instrument pool (one dry_matter meta-task per instrument),
    and few-shot-evaluates on target instrument x target fruit.
    The model never sees the target instrument nor the target fruit (double leakage control).

Methods: PLS (per-target-instrument baseline) / frozen MAE+ridge (cross-domain SSL transfer) /
  ANIL(mae) / PI-ANIL(mae).
  The only difference between ANIL and PI-ANIL = the physical constraints (additivity + C-H band).
  Mango 285-1200 nm overlaps the C-H second overtone (1100-1250 nm) by ~11%, so PI is not a pure
  null (band_coverage reported honestly); the T2 headline is cross-instrument robustness, and
  PI vs ANIL is a secondary comparison (the marginal effect of the prior in this domain).

Evaluation-time head adaptation is bit-identical to the main adaptation protocol / t3 (50 head-only steps +
spectral-radius adaptive step size + +/-5 clamp).
Outputs: results/probes/mango_instruments/{per_rep.csv, summary.txt} + checkpoints/.
"""
import sys
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.fsl_regression.core import (EMB_DIM, L, sample_train_episode,
                                     zscore_fit, eval_split, r2_score)
from src.fsl_regression.metatrain import build_encoder
from src.pifsl.metatrain import episode_loss_pi
from src.pifsl.constraints import BANDS_CH
from src.ssl import snv, load_encoder
from src.baselines import PLSBaseline

OUT = ROOT / "results" / "probes" / "mango_instruments"
OUT.mkdir(parents=True, exist_ok=True)
CKPT = OUT / "checkpoints"
CKPT.mkdir(parents=True, exist_ok=True)

F = ROOT / "data" / "spectroscopy-fewshot-benchmark-main" / "data_tmp" / "MangoDMC_NIR_Data_v4.csv"
CACHE = OUT / "mango_cache.npz"

SEED = 42
TARGET_INS = [16041, 17033]          # held-out target instruments (mid-large, ample fruits)
N_SOURCE_INS = 8                      # source-instrument pool size (top by sample count, excluding targets)
FRUIT_TARGET_FRAC = 0.20              # target-fruit fraction

EVAL_INNER_STEPS = 50


# ---------------- Data preparation ----------------
def build_cache(rebuild=False):
    if CACHE.exists() and not rebuild:
        return None
    cols = pd.read_csv(F, nrows=0).columns.tolist()
    spec_cols = [c for c in cols if c.isdigit()]
    use = ["instrument", "dry_matter", "reference_no"] + spec_cols
    dt = {c: np.float32 for c in spec_cols}
    print(f"[cache] reading mango CSV ({len(spec_cols)} wavelength columns)...", flush=True)
    df = pd.read_csv(F, usecols=use, dtype=dt, low_memory=False)
    X = df[spec_cols].values.astype(np.float32)
    wl = np.array([float(c) for c in spec_cols], dtype=np.float32)
    y = df["dry_matter"].values.astype(np.float32)
    fruit = df["reference_no"].values.astype(np.int64)
    instr = df["instrument"].values.astype(np.int64)
    np.savez(CACHE, X=X, wl=wl, y=y, fruit=fruit, instr=instr)
    print(f"[cache] {X.shape} spectra, {len(set(instr))} instruments, "
          f"{len(set(fruit))} fruits -> {CACHE.name}", flush=True)


def load_cache():
    z = np.load(CACHE, allow_pickle=True)
    return (z["X"], z["wl"], z["y"], z["fruit"], z["instr"])


def mango_band_mask(wl, n_out=L):
    """Mango C-H mask (285-1200 nm overlaps only the 1100-1250 second overtone)."""
    bands = list(BANDS_CH)
    m = np.zeros(len(wl), dtype=np.float32)
    for a, b in bands:
        m[(wl >= a) & (wl <= b)] = 1.0
    pos_old = np.linspace(0, 1, len(wl))
    pos_new = np.linspace(0, 1, n_out)
    return (np.interp(pos_new, pos_old, m) > 0.5).astype(np.float32)


def prep(X):
    """306 -> 512 position-normalized interpolation + per-spectrum SNV
    (matching the pipeline; the MAE encoder was trained on SNV-512)."""
    go = np.linspace(0, 1, X.shape[1])
    gn = np.linspace(0, 1, L)
    Xi = np.stack([np.interp(gn, go, r) for r in X])
    return snv(Xi).astype(np.float32)


# ---------------- Evaluation-time head adaptation (copied bit-for-bit from t3) ----------------
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


def mae_embeddings(X, device="cpu"):
    import torch
    enc = load_encoder("mae").to(device).eval()
    return _enc_embeddings(enc, X, device)


def _enc_embeddings(enc, X, device="cpu"):
    import torch
    enc = enc.to(device).eval()
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


# ---------------- Mango meta-training (mirrors pifsl.meta_train_pi with custom task data) ----------------
def mango_meta_train(tasks, variant, init="mae", n_episodes=2500, K=10, Q=16,
                     lr=1e-3, lr_in=0.01, inner_steps=5, lam_add=0.1, lam_band=0.1,
                     seed=0, val_every=250, patience=4, time_budget=600,
                     mask=None, device="cuda", tag="mango"):
    """tasks: dict[name] -> dict(X=(n,512) SNV, y, groups). variant in {vanilla (=ANIL), both (=PI)}."""
    import torch
    from src.fsl_regression.metatrain import episode_adapt_anil
    rng = np.random.default_rng(seed)
    pool = list(tasks)
    mask_t = (torch.tensor(mask).to(device) if mask is not None
              else torch.zeros(L, device=device))
    enc = build_encoder(init, device=device)
    head = torch.nn.Linear(EMB_DIM, 1).to(device)
    params = list(enc.parameters()) + list(head.parameters())
    opt = torch.optim.Adam(params, lr=lr)

    ckpt_file = CKPT / f"{tag}.pt"
    log_file = CKPT / f"trainlog__{tag}.csv"
    ep0, best_val, best_state, bad = 0, np.inf, None, 0
    if ckpt_file.exists():
        ck = torch.load(ckpt_file, map_location=device, weights_only=False)
        enc.load_state_dict(ck["enc"]); head.load_state_dict(ck["head"])
        opt.load_state_dict(ck["opt"])
        ep0, best_val = ck["epoch"], ck["best_val"]
        best_state, bad = ck.get("best_state"), ck.get("bad", 0)
        print(f"[resume] {tag} from ep {ep0} (best {best_val:.4f})", flush=True)

    Xt = {t: (torch.from_numpy(tasks[t]["X"])[:, None].to(device),
              torch.from_numpy(tasks[t]["y"]).float().to(device))
          for t in pool}

    def run_episode(t, rep):
        X, y = Xt[t]
        rr = np.random.default_rng(seed * 7919 + rep)
        si, qi = sample_train_episode(rr, X, y, K, Q)
        ym, ysd = zscore_fit(y[si].cpu().numpy())
        return episode_loss_pi(enc, head, X[si], (y[si] - ym) / ysd,
                               X[qi], (y[qi] - ym) / ysd, lr_in, inner_steps,
                               variant, mask_t, lam_add, lam_band, rr)

    t0 = time.time()
    ep = ep0
    while ep < ep0 + n_episodes:
        enc.train()
        t = pool[int(rng.integers(len(pool)))]
        ep += 1
        opt.zero_grad()
        loss, lv_task = run_episode(t, rep=ep)
        lv = float(loss.item())
        task_ref = lv if not np.isfinite(lv_task) else lv_task
        if not np.isfinite(lv) or task_ref > 50:
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        opt.step()
        if ep % 100 == 0:
            print(f"[{tag}] ep {ep}: task_mse={lv_task:.4f} ({time.time() - t0:.0f}s)",
                  flush=True)
            with open(log_file, "a") as f:
                f.write(f"{ep},{lv_task:.6f}\n")
        if ep % val_every == 0 or ep >= ep0 + n_episodes:
            enc.eval()
            vals = []
            with torch.enable_grad():
                for vr in range(40):
                    tt = pool[int(rng.integers(len(pool)))]
                    _l, vlt = run_episode(tt, rep=10 ** 6 + ep * 100 + vr)
                    vals.append(vlt if np.isfinite(vlt) else float(_l.item()))
            va = float(np.nanmean(vals))
            print(f"  [val] ep {ep}: query_mse={va:.4f} (best {best_val:.4f})", flush=True)
            if va < best_val:
                best_val, bad = va, 0
                best_state = dict(enc={k: v.clone() for k, v in enc.state_dict().items()},
                                  head={k: v.clone() for k, v in head.state_dict().items()})
            else:
                bad += 1
                if bad >= patience:
                    print(f"  early stop at ep {ep}", flush=True)
                    break
        if time.time() - t0 > time_budget:
            print(f"  time budget reached, saving at ep {ep}", flush=True)
            break
    if best_state is not None:
        enc.load_state_dict(best_state["enc"]); head.load_state_dict(best_state["head"])
    torch.save(dict(enc=enc.state_dict(), head=head.state_dict(), opt=opt.state_dict(),
                    epoch=ep, best_val=best_val, best_state=best_state, bad=bad,
                    variant=variant, init=init, K=K, Q=Q, lr_in=lr_in,
                    inner_steps=inner_steps, lam_add=lam_add, lam_band=lam_band), ckpt_file)
    print(f"saved -> {ckpt_file.name} (ep {ep}, best_val {best_val:.4f})", flush=True)


def main():
    import torch
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    help="prep|train|eval|all (default all)")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--episodes", type=int, default=2500)
    ap.add_argument("--time-budget", type=int, default=600)
    ap.add_argument("--rebuild", action="store_true", help="rebuild the mango cache")
    args = ap.parse_args()
    dev = args.device

    # ---- prep ----
    build_cache(rebuild=args.rebuild)
    X_raw, wl, y, fruit, instr = load_cache()
    mask = mango_band_mask(wl)
    print(f"[mask] mango C-H band_coverage={float(mask.mean()):.3f} "
          f"(285-1200 nm overlaps only the 1100-1250 second overtone)", flush=True)

    # Fruit 80/20 split (source/target disjoint, fixed seed)
    rng = np.random.default_rng(SEED)
    allf = np.unique(fruit)
    rng.shuffle(allf)
    nt = int(len(allf) * FRUIT_TARGET_FRAC)
    target_fruits = set(int(f) for f in allf[:nt])
    source_fruits = set(int(f) for f in allf[nt:])
    fruitset = np.array([int(f) in target_fruits for f in fruit])  # True = target fruit
    print(f"[split] fruits: source {len(source_fruits)} / target {len(target_fruits)} (disjoint)",
          flush=True)

    # Instrument split: targets = TARGET_INS; source = top N_SOURCE_INS (excluding targets)
    counts = pd.Series(instr).value_counts()
    src_ins = [int(i) for i in counts.index if int(i) not in TARGET_INS][:N_SOURCE_INS]
    print(f"[split] target instruments {TARGET_INS}; source pool {src_ins}", flush=True)

    # Source tasks: each source instrument x source fruits
    tasks = {}
    for ins in src_ins:
        m = (instr == ins) & (~fruitset)
        if m.sum() < 30:
            print(f"  [warn] source instrument {ins} has too few source-fruit spectra ({m.sum()}), skipping",
                  flush=True)
            continue
        tasks[f"mango_{ins}"] = dict(X=prep(X_raw[m]), y=y[m], groups=fruit[m])
        print(f"  source task mango_{ins}: {m.sum()} spectra / {len(set(fruit[m]))} fruits", flush=True)

    # ---- train ----
    if args.stage in ("all", "train"):
        print("\n=== Meta-training ANIL(vanilla,mae) ===", flush=True)
        mango_meta_train(tasks, "vanilla", init="mae", n_episodes=args.episodes,
                         time_budget=args.time_budget, mask=mask, device=dev,
                         tag="mango_anil_mae")
        print("\n=== Meta-training PI-ANIL(both,mae) ===", flush=True)
        mango_meta_train(tasks, "both", init="mae", n_episodes=args.episodes,
                         time_budget=args.time_budget, mask=mask, device=dev,
                         tag="mango_pi_mae")

    if args.stage not in ("all", "eval"):
        return

    # ---- eval ----
    print("\n=== Evaluation ===", flush=True)
    ck_anil = torch.load(CKPT / "mango_anil_mae.pt", map_location="cpu", weights_only=False)
    ck_pi = torch.load(CKPT / "mango_pi_mae.pt", map_location="cpu", weights_only=False)
    print(f"[ckpt] ANIL ep{ck_anil.get('epoch')} best{ck_anil.get('best_val'):.4f} | "
          f"PI ep{ck_pi.get('epoch')} best{ck_pi.get('best_val'):.4f}", flush=True)
    encs = {"anil": (build_enc(ck_anil, dev), ck_anil["head"]),
            "pi": (build_enc(ck_pi, dev), ck_pi["head"])}

    rows = []
    for tins in TARGET_INS:
        m = (instr == tins) & fruitset
        Xt = prep(X_raw[m])
        yt = y[m]
        gt = fruit[m]
        n = len(yt)
        if n < 30:
            print(f"  [warn] target instrument {tins} has too few target-fruit spectra ({n}), skipping",
                  flush=True)
            continue
        E_mae = mae_embeddings(Xt, device=dev)
        # Encoder/readout isolation: meta-encoder + ridge readout (same readout as MAE_ridge,
        # isolating encoder quality)
        E_anil = _enc_embeddings(encs["anil"][0], Xt, device=dev)
        E_pi = _enc_embeddings(encs["pi"][0], Xt, device=dev)
        print(f"[target] instrument {tins}: {n} spectra / {len(set(gt))} fruits, "
              f"DM {yt.min():.1f}-{yt.max():.1f}", flush=True)
        for K in [5, 10, 20]:
            for rep in range(args.reps):
                tr, te = eval_split(n, K, rep, groups=gt)
                yte = yt[te]
                rec = dict(target=tins, shot=K, rep=rep)
                try:
                    rec["PLS"] = r2_score(PLSBaseline().fit(Xt[tr], yt[tr]).predict(Xt[te]), yte)
                except Exception:
                    rec["PLS"] = np.nan
                try:
                    rec["MAE_ridge"] = r2_score(fit_predict_ridge(E_mae[tr], yt[tr], E_mae[te]), yte)
                except Exception:
                    rec["MAE_ridge"] = np.nan
                for key, label in (("anil", "ANIL(mae)"), ("pi", "PI-ANIL(mae)")):
                    enc, hs = encs[key]
                    try:
                        rec[label] = r2_score(eval_episode_anil(enc, hs, Xt, yt, tr, te, dev), yte)
                    except Exception as ex:
                        rec[label] = np.nan
                        print(f"[fail] {label} {tins} K={K} rep={rep}: {ex}", flush=True)
                # Meta-encoder + ridge (encoder-isolated)
                try:
                    rec["ANIL+ridge"] = r2_score(fit_predict_ridge(E_anil[tr], yt[tr], E_anil[te]), yte)
                except Exception:
                    rec["ANIL+ridge"] = np.nan
                try:
                    rec["PI+ridge"] = r2_score(fit_predict_ridge(E_pi[tr], yt[tr], E_pi[te]), yte)
                except Exception:
                    rec["PI+ridge"] = np.nan
                rows.append(rec)
            print(f"  [{tins} K={K}] {args.reps} reps done", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "per_rep.csv", index=False)
    cols = ["PLS", "MAE_ridge", "ANIL+ridge", "PI+ridge", "ANIL(mae)", "PI-ANIL(mae)"]
    lines = [f"T2 mango cross-instrument Phase 0 (reps={args.reps}, device={dev}, "
             f"MAE = cross-domain transfer)",
             f"Setup: LOIO + fruit-disjoint (target instruments {TARGET_INS}, source pool {src_ins}, "
             f"fruit 80/20)",
             f"Task: dry_matter, mango C-H band_coverage={float(mask.mean()):.3f}",
             "=" * 96]
    for tins in TARGET_INS:
        sub = df[df.target == tins]
        if not len(sub):
            continue
        lines.append(f"\n===== Target instrument {tins} (median R2 / crash rate) =====")
        lines.append(f"{'K':>3} | " + " ".join(f"{c:>18}" for c in cols))
        lines.append("-" * 96)
        for K in [5, 10, 20]:
            s = sub[sub.shot == K]
            cells = []
            for c in cols:
                v = s[c].dropna()
                coll = float((v < 0).mean()) if len(v) else float("nan")
                cells.append(f"{v.median():+7.3f}/{coll*100:3.0f}%" if len(v) else "      N/A      ")
            lines.append(f"{K:>3} | " + " ".join(f"{c:>18}" for c in cells))
        if "ANIL(mae)" in cols and "PI-ANIL(mae)" in cols:
            lines.append(f"\n  PI-ANIL vs ANIL (paired Wilcoxon, one-sided PI>ANIL):")
            lines.append(f"  {'K':>3} | {'med d':>8} {'PI>ANIL':>8} {'p':>9}")
            for K in [5, 10, 20]:
                s = sub[sub.shot == K]
                dd = (s["PI-ANIL(mae)"] - s["ANIL(mae)"]).dropna()
                if len(dd) < 3:
                    continue
                try:
                    pv = wilcoxon(s["PI-ANIL(mae)"].dropna(), s["ANIL(mae)"].dropna(),
                                  alternative="greater").pvalue
                except Exception:
                    pv = float("nan")
                lines.append(f"  {K:>3} | {dd.median():>+8.3f} {(dd>0).sum()}/{len(dd):>3} {pv:>9.4g}")
    # Encoder-isolated comparison (same ridge readout, isolating cross-instrument encoder quality)
    for tins in TARGET_INS:
        sub = df[df.target == tins]
        if not len(sub):
            continue
        lines.append(f"\n  Instrument {tins} encoder isolation (same ridge readout, median R2):")
        for K in [5, 10, 20]:
            s = sub[sub.shot == K]
            cells = []
            for c in ["MAE_ridge", "ANIL+ridge", "PI+ridge"]:
                v = s[c].dropna()
                cells.append(f"{c}={v.median():+.3f}" if len(v) else f"{c}=NA")
            lines.append(f"    K={K}: " + "  ".join(cells))
        lines.append(f"  Physics (encoder) PI+ridge vs ANIL+ridge (one-sided PI>ANIL):")
        for K in [5, 10, 20]:
            s = sub[sub.shot == K]
            dd = (s["PI+ridge"] - s["ANIL+ridge"]).dropna()
            if len(dd) < 3:
                continue
            try:
                pv = wilcoxon(s["PI+ridge"].dropna(), s["ANIL+ridge"].dropna(),
                              alternative="greater").pvalue
            except Exception:
                pv = float("nan")
            lines.append(f"    K={K}: d={dd.median():+.3f} {(dd>0).sum()}/{len(dd)} p={pv:.3g}")
    summary = "\n".join(lines)
    (OUT / "summary.txt").write_text(summary, encoding="utf-8")
    print("\n" + summary, flush=True)


if __name__ == "__main__":
    main()
