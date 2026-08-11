"""PHC real crude oil VNIR-SWIR API gravity — fine-grained molecular-fingerprint sub-band ablation (Phase 2a).

Validation probe: can the reversal finding from MLNIRdata (full-NIR synthetic hydrocarbon mixtures)—"the density
signal is specifically concentrated in the 1st overtone 1620-1800 + combination band 2 2140-2350"—be reproduced
on [real crude oils + API gravity]? API gravity belongs to the same family as density (API = 141.5/SG − 131.5;
higher API = lower density) and is the most important macroscopic property of crude oil.
If the API signal of real crude oils also falls in the 1st overtone/combination band 2 → the reversal finding
generalizes from synthetic hydrocarbons to real crude oils, nailing down the paper's physical-mechanism narrative
(bandwidth-dependent effective domain of molecular fingerprints).

Data: data/PHC/VNIR-SWIR/CRUDE OILS/ (Pabon & Souza Filho 2019 Fuel, 10.1016/j.fuel.2018.09.098;
Scientific Data 10.1038/s41597-024-03892-y).
  CRUDE OILS_VNIR-SWIR.sli  ENVI spectral library, 37 spectra × 2151 bands, float32, bsq, LE, header offset=0
  Wavelengths 350-2500nm (1nm spacing); full NIR fully reachable (including the MLNIR reversal zone)
  Instrument ASD FieldSpec 4, reflectance (DHR)—remote-sensing form, different from laboratory absorbance (honest risk point)
  API labels: spectrum name "OIL NNN" → API = NNN/10 (13.2-47.2, wide span)
Preprocessing: reflectance → interpolate to 512 (positional normalization) + per-spectrum SNV (isomorphic to the project's NIR convention).
Band definitions (nm, Workman & Weyer assignments); all reachable under full NIR:
  2nd:   CH3_2nd(1150-1210) / CH2_2nd(1190-1250) / ArH_2nd(1140-1180)
  combo: CH3_combo(1350-1410) / CH2_combo(1390-1450)
  1st*:  =CH_1st(1620-1660) / ArH_1st(1680-1720) / CH3_1st(1690-1760) / CH2_1st(1720-1800)  ← MLNIR reversal zone
  combo2*: ArH_combo2(2140-2180) / CH_combo2(2200-2350)                                     ← MLNIR reversal zone
  VIS:   keep_VIS(400-700) color control (crude color ↔ API is strongly correlated, heavy oils are dark; separates molecular-fingerprint vs color signal)
Controls: full(350-2500) / keep_bigCH(1100-1250+1350-1550 = old broad C-H band)
Methods: PLS (LOO nc, primary readout) + frozen MAE + ridge (tests how well the encoder adapts to DHR reflectance spectra)
Evaluation: KFold5 ceiling (37 samples) + few-shot K=10/20 × 30 reps (groups=None; 37 independent crude oils, no replicates)
"""
from __future__ import annotations
import sys, csv, re
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.fsl_regression.core import eval_split, BASE_SEED, L  # noqa: E402

OUT = ROOT / "results" / "fingerprint" / "phc"
(OUT / "parts").mkdir(parents=True, exist_ok=True)
DATA = ROOT / "data" / "PHC" / "VNIR-SWIR" / "CRUDE OILS"

SUBBANDS = [
    ("CH3_2nd", 1150, 1210), ("CH2_2nd", 1190, 1250), ("ArH_2nd", 1140, 1180),
    ("CH3_combo", 1350, 1410), ("CH2_combo", 1390, 1450),
    ("=CH_1st", 1620, 1660), ("ArH_1st", 1680, 1720), ("CH3_1st", 1690, 1760), ("CH2_1st", 1720, 1800),
    ("ArH_combo2", 2140, 2180), ("CH_combo2", 2200, 2350),
]
BIG_CH = [(1100, 1250), (1350, 1550)]
VIS_BAND = [(400, 700)]


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


def load_phc_vnir():
    """Returns (X(37,512) SNV, y(37,) API, wl(2151,)). Reads [37,2151] float32 from the ENVI .sli."""
    raw = np.fromfile(DATA / "CRUDE OILS_VNIR-SWIR.sli", dtype="<f4")  # little-endian float32
    # Parse spectra names (OIL NNN) from the hdr to determine the number of rows and API labels
    hdr = (DATA / "CRUDE OILS_VNIR-SWIR.hdr").read_text(encoding="utf-8", errors="replace")
    names_block = re.search(r"spectra names\s*=\s*\{([^}]*)\}", hdr, re.S).group(1)
    codes = [int(m) for m in re.findall(r"OIL\s*(\d+)", names_block)]
    n_spec = len(codes)
    raw = raw.reshape(n_spec, -1)               # (37, 2151)
    assert raw.shape[1] == 2151, f"expected 2151 bands, got {raw.shape[1]}"
    y = np.array([c / 10.0 for c in codes], float)   # API gravity
    wl = np.arange(350, 2501, 1.0)              # 2151 points, 350-2500nm
    X = raw.astype(float)
    # Interpolate to 512 + SNV
    go = np.linspace(0, 1, X.shape[1]); gn = np.linspace(0, 1, L)
    X = np.stack([np.interp(gn, go, r) for r in X]).astype(np.float32)
    mu = X.mean(1, keepdims=True); sd = X.std(1, keepdims=True) + 1e-8
    X = (X - mu) / sd
    return X, y, wl


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    import torch
    from sklearn.model_selection import KFold
    from src.ssl.train import load_encoder

    X, y, wl = load_phc_vnir()
    n = len(y)
    print(f"[PHC-VNIR] n={n} crude oils wl {wl.min():.0f}-{wl.max():.0f}nm | "
          f"API {y.min():.1f}-{y.max():.1f} | "
          f"1st_overtone_reachable={wl.min()<=1620 and wl.max()>=1800} combo2_reachable={wl.min()<=2100 and wl.max()>=2450}",
          flush=True)
    print(f"[API] labels (in spectrum order): {np.round(y,1).tolist()}", flush=True)

    keep_big = band_keep_mask(wl, BIG_CH)
    keep_vis = band_keep_mask(wl, VIS_BAND)
    conds = [("full", None), ("keep_bigCH", keep_big), ("keep_VIS", keep_vis)] + \
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
    lines = ["=" * 118,
             "PHC real crude oil VNIR-SWIR API gravity — fine-grained sub-band ablation (validating whether the MLNIR reversal finding reproduces on real crude oils)",
             "=" * 118,
             f"wavelength {wl.min():.0f}-{wl.max():.0f}nm (full NIR+VIS), n={n} real crude oils, API gravity regression (DHR reflectance spectra)",
             "", "(A) KFold5 full-sample ceiling R²",
             f"{'method':12}" + "".join(f"{c:>13}" for c in cnames),
             "-" * 118]
    for mth in ["pls", "mae_ridge"]:
        lines.append(f"{mth:12}" + "".join(f"{ceiling[(mth,c)]:>13.3f}" for c in cnames))

    lines += ["", "(B) MAE+ridge few-shot median R²",
              f"{'K':>4}" + "".join(f"{c:>13}" for c in cnames)]
    for K in [10, 20]:
        cells = "".join(f"{np.nanmedian(agg.get(('mae_ridge',c,K),[np.nan])):>13.3f}" for c in cnames)
        lines.append(f"{K:>4}{cells}")

    # Key readout: does the MLNIR reversal zone (1st overtone + combination band 2) specifically carry the API signal; how strong is the VIS color signal
    lines += ["", "=" * 118,
              "Readout 1: MLNIR reversal zone (1st overtone 1620-1800 / combination band 2 2140-2350) PLS ceiling vs bigCH vs VIS",
              "=" * 118]
    mlnir_zone = {"=CH_1st", "ArH_1st", "CH3_1st", "CH2_1st", "ArH_combo2", "CH_combo2"}
    big = ceiling[("pls", "keep_bigCH")]
    vis = ceiling[("pls", "keep_VIS")]
    full = ceiling[("pls", "full")]
    lines.append(f"full (350-2500 full spectrum) = {full:.3f}")
    lines.append(f"keep_VIS (400-700 color)     = {vis:.3f}   ← color-signal benchmark")
    lines.append(f"keep_bigCH (old broad C-H band) = {big:.3f}   ← diesel-reachable region")
    for c, _ in conds:
        if c in ("full", "keep_bigCH", "keep_VIS"):
            continue
        v = ceiling[("pls", c)]
        tag = "  ★MLNIR reversal zone" if c in mlnir_zone else ""
        lines.append(f"  {c:14} {v:.3f}  (Δ vs bigCH {v-big:+.3f}){tag}")

    lines += ["", "Readout 2: MAE encoder adaptation to DHR reflectance spectra (full R²; MLNIR .973 as reference)",
              f"  MAE+ridge full = {ceiling[('mae_ridge','full')]:.3f}   PLS full = {full:.3f}"]

    txt = OUT / "summary.txt"
    txt.write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines), flush=True)
    print(f"\nsaved -> {txt} | {out}", flush=True)


if __name__ == "__main__":
    main()
