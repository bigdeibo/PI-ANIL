"""Edible-oil peroxide value (PV) — within-task functional-group band localization.

Motivation: the cross-material LOMO probe (probe_oxidation_property.py) found the
physics prior neutral on PV, with every method near chance under the
leave-one-material-out split. That null alone is ambiguous. This within-task
probe asks the complementary question directly on the edible-oil data, where PV
is NIR-predictable: does the PV prediction signal sit in the O--H band (the
peroxide R--OOH functional group) or in the C--H band?
  - keep_OH comparable to or better than keep_CH  ->  the C--H prior is mis-set
    for PV, and an O--H prior would be the relevant inductive bias.
  - keep_CH clearly stronger than keep_OH         ->  the C--H prior is not
    mis-set; the LOMO neutrality has another cause.

Band definitions (consistent with src/pifsl/constraints.py):
  OH = 1400-1500, 1900-2000 nm  (O--H overtone/combination; water, hydroxyl, hydroperoxide)
  CH = 1100-1250, 1350-1550, 1650-1800, 2100-2400 nm (C--H overtone/combination)
Note 1350-1550 overlaps 1400-1500 (intrinsic NIR assignment overlap); the mask
sizes are reported so the comparison is auditable.

Two evaluations:
  (A) Full-sample GroupKFold(5) by oil identity — ceiling: where is the PV signal.
  (B) Group-aware few-shot (K=10/20/30 spectra, 30 repetitions, seed=42) —
      PLS with support-set LOO component selection + frozen MAE + ridge.

Outputs: results/probes/oxidation_property/band_localization/{ceiling.csv,
parts/fewshot.csv, summary.txt}.
"""
from __future__ import annotations
import sys, csv
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.fsl_regression.core import eval_split, BASE_SEED, L  # noqa: E402

OUT = ROOT / "results" / "probes" / "oxidation_property" / "band_localization"
(OUT / "parts").mkdir(parents=True, exist_ok=True)

OH_BANDS = [(1400, 1500), (1900, 2000)]
CH_BANDS = [(1100, 1250), (1350, 1550), (1650, 1800), (2100, 2400)]


def band_keep_mask(wl, bands, length=L):
    """Dimensions (out of `length`) that fall inside `bands`, mapped linearly
    from the dataset's original wavelength range to the resampled axis."""
    wmin, wmax = float(np.min(wl)), float(np.max(wl))
    grid = wmin + (wmax - wmin) * np.linspace(0, 1, length)
    m = np.zeros(length, dtype=bool)
    for lo, hi in bands:
        m |= (grid >= lo) & (grid <= hi)
    return m


def r2(y_true, y_pred):
    ss = np.sum((y_true - y_true.mean()) ** 2)
    if ss < 1e-12:
        return 0.0
    return float(1 - np.sum((y_true - y_pred) ** 2) / ss)


def pls_fit_predict(Xtr, ytr, Xte, max_nc=8):
    """PLS with support-set LOO component selection (few-shot friendly)."""
    from sklearn.cross_decomposition import PLSRegression
    K = len(ytr)
    cap = min(max_nc, K - 1, Xtr.shape[1] - 1, 10)
    cap = max(1, cap)
    best_nc, best_err = 1, np.inf
    if K >= 6:
        for nc in range(1, cap + 1):
            try:
                p = PLSRegression(n_components=nc).fit(Xtr, ytr).predict(Xtr)
                err = np.mean((p.ravel() - ytr) ** 2)
                if err < best_err:
                    best_err, best_nc = err, nc
            except Exception:
                break
    return PLSRegression(n_components=best_nc).fit(Xtr, ytr).predict(Xte).ravel()


def ridge_fit_predict(Ztr, ytr, Zte):
    from sklearn.linear_model import RidgeCV
    return RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(Ztr, ytr).predict(Zte)


def main():
    import torch
    from src.datasets import load_edibleoil
    from src.ssl.train import load_encoder

    d = load_edibleoil()
    X_raw = np.asarray(d["X"], float)
    y = np.asarray(d["targets"]["peroxide_value"], float)
    groups = np.asarray(d["groups"])
    wl = np.asarray(d["wavelengths"], float)

    # Interpolate to 512 points + per-spectrum SNV.
    go = np.linspace(0, 1, X_raw.shape[1]); gn = np.linspace(0, 1, L)
    X = np.stack([np.interp(gn, go, r) for r in X_raw]).astype(np.float32)
    mu = X.mean(1, keepdims=True); sd = X.std(1, keepdims=True) + 1e-8
    X = (X - mu) / sd

    keep_oh = band_keep_mask(wl, OH_BANDS)
    keep_ch = band_keep_mask(wl, CH_BANDS)
    overlap = keep_oh & keep_ch
    print(f"[edibleoil] n={len(y)} oils={len(set(groups.tolist()))} "
          f"wl {wl.min():.0f}-{wl.max():.0f}nm", flush=True)
    print(f"[mask] OH-only dims={keep_oh.sum()} CH-only dims={keep_ch.sum()} "
          f"overlap={overlap.sum()} of {L}", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc = load_encoder("mae").to(device).eval()

    def encode(Xin):
        with torch.no_grad():
            t = torch.from_numpy(Xin.astype(np.float32)).to(device)
            return enc(t[:, None]).cpu().numpy()

    def zero(Xin, keep):
        Xz = Xin.copy(); Xz[:, ~keep] = 0.0; return Xz

    # Encode the three conditions (PLS uses raw sub-bands; MAE+ridge uses
    # zeroed-then-encoded inputs).
    Z_full = encode(X); Z_oh = encode(zero(X, keep_oh)); Z_ch = encode(zero(X, keep_ch))
    if device == "cuda":
        torch.cuda.empty_cache()

    conds = ["full", "keep_OH", "keep_CH"]
    Xsub = {"full": X, "keep_OH": X[:, keep_oh], "keep_CH": X[:, keep_ch]}
    Zemb = {"full": Z_full, "keep_OH": Z_oh, "keep_CH": Z_ch}

    # ---- (A) Full-sample GroupKFold(5) by oil identity: ceiling ----
    from sklearn.model_selection import GroupKFold
    lines = ["=" * 70, "(A) Full-sample GroupKFold(5) by oil identity - PV signal ceiling",
             "=" * 70, f"{'method':10}{'cond':>10}{'R2':>9}"]
    gkf = GroupKFold(5)
    ceil_rows = []
    for mth in ["pls", "mae_ridge"]:
        for cond in conds:
            pred = np.zeros_like(y); truth = np.zeros_like(y)
            for tr, te in gkf.split(X, y, groups=groups):
                if mth == "pls":
                    p = pls_fit_predict(Xsub[cond][tr], y[tr], Xsub[cond][te])
                else:
                    p = ridge_fit_predict(Zemb[cond][tr], y[tr], Zemb[cond][te])
                pred[te] = p; truth[te] = y[te]
            r2val = r2(truth, pred)
            lines.append(f"{mth:10}{cond:>10}{r2val:>9.3f}")
            ceil_rows.append(dict(method=mth, cond=cond, R2=r2val))
    print("\n".join(lines), flush=True)
    with open(OUT / "ceiling.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "cond", "R2"])
        w.writeheader(); w.writerows(ceil_rows)
    print(f"saved -> {OUT / 'ceiling.csv'}", flush=True)

    # ---- (B) Group-aware few-shot K=10/20/30, 30 repetitions ----
    rows = []
    for K in [10, 20, 30]:
        for rep in range(30):
            tr, te = eval_split(len(y), K, rep, seed=BASE_SEED, groups=groups)
            for mth in ["pls", "mae_ridge"]:
                for cond in conds:
                    if mth == "pls":
                        p = pls_fit_predict(Xsub[cond][tr], y[tr], Xsub[cond][te])
                    else:
                        p = ridge_fit_predict(Zemb[cond][tr], y[tr], Zemb[cond][te])
                    rows.append(dict(method=mth, cond=cond, K=K, rep=rep,
                                     r2=r2(y[te], p)))
        print(f"  K={K} done", flush=True)

    out = OUT / "parts" / "fewshot.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "cond", "K", "rep", "r2"])
        w.writeheader(); w.writerows(rows)
    print(f"saved -> {out}", flush=True)

    # Few-shot aggregation.
    agg = {}
    for r in rows:
        agg.setdefault((r["method"], r["cond"], r["K"]), []).append(r["r2"])
    lines += ["", "=" * 70, "(B) Group-aware few-shot R2 median (30 rep)",
              "=" * 70, f"{'method':10}{'cond':>10}{'K':>4}{'R2_med':>9}{'R2_mean':>9}"]
    for key in sorted(agg):
        mth, cond, K = key
        a = np.array(agg[key])
        lines.append(f"{mth:10}{cond:>10}{K:>4}{np.median(a):>9.3f}{np.mean(a):>9.3f}")
    txt = OUT / "summary.txt"
    txt.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[-12:]), flush=True)


if __name__ == "__main__":
    main()
