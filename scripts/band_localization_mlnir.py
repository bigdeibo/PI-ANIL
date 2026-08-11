"""MLNIRdata full-NIR hydrocarbon density — fine-grained molecular-fingerprint sub-band ablation (discriminative probe).

Core discriminating question: the earlier diesel (750-1550)/gasoline (900-1700) cuts made the
1st overtone (1620-1800) + combination band 2 (2100-2450) unreachable, and three falsifications concluded
"NIR molecular fingerprints are infeasible". MLNIRdata covers the full NIR 1106-2524nm with 208 hydrocarbon
mixtures + density, so all fine sub-bands are reachable → discriminate: is the density signal specifically
carried by the 1st overtone/combination band 2 regions that diesel cannot reach?
  - If yes → reversal: the earlier result was a bandwidth artifact; under full NIR, "molecular fingerprint → density" holds.
  - If it still concentrates in the broad band → strengthens the view that NIR overtones are intrinsically overlapped (hard to be specific even under full NIR).

Data: data/MLNIRdata/ (IFPEN Duval 2025; used for Dual-sPLS, Alsouki et al. Chemolab 2023).
  X = MLNIR_matrixX_NirSpectrumData.csv        (2635 wavelengths × 208 samples, rows = wavenumber axis)
  axis = MLNIR_matrixX_NirSpectrumDataAxis.csv (cm⁻¹, 3961-9041, ascending)
  y = MLNIR_matrixY_NirPropertyDensityNormalized.csv (208 densities, 0-1)
Preprocessing: cm⁻¹→nm (nm=1e7/cm⁻¹), reorder ascending by nm, transpose to (208 samples × 2635 wavelengths), interpolate to 512 + SNV.
Band definitions (nm, Workman & Weyer assignments); all reachable under full NIR:
  2nd:   CH3_2nd(1150-1210) / CH2_2nd(1190-1250) / ArH_2nd(1140-1180)
  combo: CH3_combo(1350-1410) / CH2_combo(1390-1450)
  1st*:  =CH_1st(1620-1660) / ArH_1st(1680-1720) / CH3_1st(1690-1760) / CH2_1st(1720-1800)  ← unreachable by diesel
  combo2*: ArH_combo2(2140-2180) / CH_combo2(2200-2350)                                    ← unreachable by diesel
Controls: full / keep_bigCH(1100-1250+1350-1550 = the "old broad C-H band" reachable by diesel)
Methods: PLS (LOO nc) + frozen MAE + ridge (trained on an NIR corpus; hydrocarbon NIR is close — adaptation is validated if the full R² is reasonable)
Evaluation: KFold5 ceiling (primary readout) + few-shot K=10/20 × 30 reps (groups=None random paired splits)
"""
from __future__ import annotations
import sys, csv
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.fsl_regression.core import eval_split, BASE_SEED, L  # noqa: E402

OUT = ROOT / "results" / "fingerprint" / "mlnir"
(OUT / "parts").mkdir(parents=True, exist_ok=True)
DATA = ROOT / "data" / "MLNIRdata"

SUBBANDS = [
    ("CH3_2nd", 1150, 1210), ("CH2_2nd", 1190, 1250), ("ArH_2nd", 1140, 1180),
    ("CH3_combo", 1350, 1410), ("CH2_combo", 1390, 1450),
    ("=CH_1st", 1620, 1660), ("ArH_1st", 1680, 1720), ("CH3_1st", 1690, 1760), ("CH2_1st", 1720, 1800),
    ("ArH_combo2", 2140, 2180), ("CH_combo2", 2200, 2350),
]
BIG_CH = [(1100, 1250), (1350, 1550)]


def band_keep_mask(wl, bands, length=L):
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


def pls_fit_predict(Xtr, ytr, Xte, max_nc=12):
    from sklearn.cross_decomposition import PLSRegression
    K = len(ytr)
    cap = min(max_nc, K - 1, Xtr.shape[1] - 1, 15)
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


def load_mlnir():
    """Returns (X(208,512) SNV, y(208,), wl_nm(2635,)). Raw X rows = wavenumber cm-1, columns = samples."""
    X = np.loadtxt(DATA / "MLNIR_matrixX_NirSpectrumData.csv", delimiter=",")   # (2635 wav, 208 samp)
    ax = np.loadtxt(DATA / "MLNIR_matrixX_NirSpectrumDataAxis.csv")             # (2635,) cm-1
    y = np.loadtxt(DATA / "MLNIR_matrixY_NirPropertyDensityNormalized.csv", delimiter=",")  # (208,)
    nm = 1e7 / ax                                                               # cm-1 → nm
    order = np.argsort(nm)                                                      # ascending by nm
    nm = nm[order]
    X = X[order, :]                                                             # rows reordered ascending by nm
    X = X.T.astype(float)                                                       # (208 samp, 2635 wav)
    go = np.linspace(0, 1, X.shape[1]); gn = np.linspace(0, 1, L)
    X = np.stack([np.interp(gn, go, r) for r in X]).astype(np.float32)
    mu = X.mean(1, keepdims=True); sd = X.std(1, keepdims=True) + 1e-8
    X = (X - mu) / sd
    return X, y, nm


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    import torch
    from sklearn.model_selection import KFold
    from src.ssl.train import load_encoder

    X, y, wl = load_mlnir()
    n = len(y)
    print(f"[MLNIR] n={n} wl {wl.min():.0f}-{wl.max():.0f}nm | "
          f"1st_overtone_reachable={wl.min()<=1620 and wl.max()>=1800} combo2_reachable={wl.min()<=2100 and wl.max()>=2450}",
          flush=True)

    keep_big = band_keep_mask(wl, BIG_CH)
    conds = [("full", None), ("keep_bigCH", keep_big)] + \
            [(nm_, band_keep_mask(wl, [(lo, hi)])) for nm_, lo, hi in SUBBANDS]
    print(f"[mask] " + " ".join(f"{c}={k.sum()}d" for c, k in conds[1:]), flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc = load_encoder("mae").to(device).eval()

    def encode(Xin):
        with torch.no_grad():
            t = torch.from_numpy(Xin.astype(np.float32)).to(device)
            return enc(t[:, None]).cpu().numpy()

    def zero(Xin, keep):
        Xz = Xin.copy(); Xz[:, ~keep] = 0.0; return Xz

    Zemb = {c: (encode(X) if k is None else encode(zero(X, k))) for c, k in conds}
    Xsub = {c: (X if k is None else X[:, k]) for c, k in conds}
    if device == "cuda":
        torch.cuda.empty_cache()

    kf = KFold(5, shuffle=True, random_state=42)
    ceiling = {}
    fewshot_rows = []

    for mth in ["pls", "mae_ridge"]:
        for c, _ in conds:
            pred = np.zeros_like(y); truth = np.zeros_like(y)
            for tr, te in kf.split(X):
                if mth == "pls":
                    p = pls_fit_predict(Xsub[c][tr], y[tr], Xsub[c][te])
                else:
                    p = ridge_fit_predict(Zemb[c][tr], y[tr], Zemb[c][te])
                pred[te] = p; truth[te] = y[te]
            ceiling[(mth, c)] = r2(truth, pred)

    for K in [10, 20]:
        for rep in range(30):
            tr, te = eval_split(n, K, rep, seed=BASE_SEED)
            for mth in ["pls", "mae_ridge"]:
                for c, _ in conds:
                    try:
                        if mth == "pls":
                            p = pls_fit_predict(Xsub[c][tr], y[tr], Xsub[c][te])
                        else:
                            p = ridge_fit_predict(Zemb[c][tr], y[tr], Zemb[c][te])
                        fewshot_rows.append(dict(method=mth, cond=c, K=K, rep=rep, r2=r2(y[te], p)))
                    except Exception:
                        fewshot_rows.append(dict(method=mth, cond=c, K=K, rep=rep, r2=np.nan))
    print("  eval done", flush=True)

    out = OUT / "parts" / "fewshot.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "cond", "K", "rep", "r2"])
        w.writeheader(); w.writerows(fewshot_rows)
    agg = {}
    for r in fewshot_rows:
        agg.setdefault((r["method"], r["cond"], r["K"]), []).append(r["r2"])

    cnames = [c for c, _ in conds]
    lines = ["=" * 110, "MLNIRdata full-NIR hydrocarbon density — fine-grained sub-band ablation (is the 'infeasibility' of NIR molecular fingerprints real or a bandwidth artifact?)",
             "=" * 110, f"wavelength {wl.min():.0f}-{wl.max():.0f}nm (full NIR), n={n} hydrocarbon mixtures, density regression",
             "", "(A) KFold5 full-sample ceiling R²", f"{'method':12}" + "".join(f"{c:>13}" for c in cnames),
             "-" * 110]
    for mth in ["pls", "mae_ridge"]:
        lines.append(f"{mth:12}" + "".join(f"{ceiling[(mth,c)]:>13.3f}" for c in cnames))

    lines += ["", "(B) MAE+ridge few-shot median R² (K=20)",
              f"{'K':>4}" + "".join(f"{c:>13}" for c in cnames)]
    for K in [10, 20]:
        cells = "".join(f"{np.nanmedian(agg.get(('mae_ridge',c,K),[np.nan])):>13.3f}" for c in cnames)
        lines.append(f"{K:>4}{cells}")

    # Key readout: do the 1st overtone + combination band 2 (diesel-unreachable regions) specifically carry the density signal
    lines += ["", "=" * 110, "Readout: PLS ceiling of the diesel-unreachable regions (1st overtone 1620-1800 / combination band 2 2140-2350)",
              "=" * 110]
    diesel_unreachable = {"=CH_1st", "ArH_1st", "CH3_1st", "CH2_1st", "ArH_combo2", "CH_combo2"}
    big = ceiling[("pls", "keep_bigCH")]
    lines.append(f"keep_bigCH (old broad C-H band, reachable by diesel) = {big:.3f}")
    for c, _ in conds:
        if c in ("full", "keep_bigCH"):
            continue
        v = ceiling[("pls", c)]
        tag = "  ★diesel-unreachable" if c in diesel_unreachable else ""
        lines.append(f"  {c:14} {v:.3f}  (Δ vs bigCH {v-big:+.3f}){tag}")

    txt = OUT / "summary.txt"
    txt.write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines), flush=True)
    print(f"\nsaved -> {txt} | {out}", flush=True)


if __name__ == "__main__":
    main()
