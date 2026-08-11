"""Build the self-supervised pretraining corpus: local real spectra (labels removed) + NIST IR + simulated mixtures.

Strategy: every spectrum is resampled to a uniform length L=512 over its own
wavelength range (position normalization), so the encoder learns "shape
features"; downstream tasks are resampled the same way, and the loss of
absolute wavelength information is honestly noted in the summary. Outputs:
  data/pretrain_corpus/spectra.npy  (N, 512) float32
  data/pretrain_corpus/sources.csv  provenance label of each spectrum
  data/pretrain_corpus/sim_compositions.npz  mixture compositions of the simulated spectra (reused in the boundary probes)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.datasets import load_diesel, load_gasoline, load_corn, load_evoo, load_mayonnaise
from src.simulate import build_pure_library, simulate_mixtures

L = 512
N_SIM = 6000
OUT = ROOT / "data" / "pretrain_corpus"
OUT.mkdir(parents=True, exist_ok=True)


def resample(X):
    """(n, L_i) -> (n, L); linear interpolation over each spectrum's own range."""
    n, li = X.shape
    grid_old = np.linspace(0, 1, li)
    grid_new = np.linspace(0, 1, L)
    return np.stack([np.interp(grid_new, grid_old, row) for row in X]).astype(np.float32)


def main():
    spectra, sources = [], []

    d = load_diesel(); spectra.append(resample(d["X"])); sources += ["diesel"] * len(d["X"])
    g = load_gasoline(); spectra.append(resample(g["X"])); sources += ["gasoline"] * len(g["X"])
    c = load_corn()
    for inst, (Xi, _wl) in c["instruments"].items():
        spectra.append(resample(Xi)); sources += [f"corn_{inst}"] * len(Xi)
    e = load_evoo(); spectra.append(resample(e["X"])); sources += ["evoo"] * len(e["X"])
    m = load_mayonnaise(); spectra.append(resample(m["X"])); sources += ["mayonnaise"] * len(m["X"])

    # NIST IR
    z = np.load(ROOT / "data" / "NIST_IR" / "nist_ir.npz", allow_pickle=True)
    Xn = np.stack([np.interp(np.linspace(0, 1, L), np.linspace(0, 1, len(y)), y)
                   for y in z["y"]]).astype(np.float32)
    spectra.append(Xn); sources += ["nist_ir"] * len(Xn)

    # Simulated mixtures: pure-component library = NIST 98 + EVOO 7-class means + mayonnaise 6-class means
    class_spec = {}
    for cls in sorted(set(e["labels"])):
        class_spec[f"evoo_{cls}"] = e["X"][e["labels"] == cls]
    for cls in sorted(set(m["labels"].astype(str))):
        class_spec[f"mayo_{cls}"] = m["X"][m["labels"].astype(str) == cls]
    lib = build_pure_library(nist_npz=ROOT / "data" / "NIST_IR" / "nist_ir.npz",
                             class_spectra=class_spec)
    Xs, C, lib_names = simulate_mixtures(lib, N_SIM, L=L, seed=42)
    spectra.append(Xs.astype(np.float32)); sources += ["simulated"] * N_SIM

    X = np.vstack(spectra)
    np.save(OUT / "spectra.npy", X)
    pd.DataFrame(dict(source=sources)).to_csv(OUT / "sources.csv", index=False)
    np.savez(OUT / "sim_compositions.npz", X=Xs, C=C, lib_names=np.array(lib_names))
    print(f"corpus: {X.shape}, sources:")
    print(pd.Series(sources).value_counts().to_string())
    print(f"pure library: {len(lib)} components -> sim_compositions.npz")


if __name__ == "__main__":
    main()
