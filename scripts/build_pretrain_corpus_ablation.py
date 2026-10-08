"""W4 corpus-recipe ablation: build recipe-variant pretraining corpora (design doc §5.2).

Axes (2^3 corner design; the reference recipe s6000_l111_p1 is the paper's corpus,
already trained as encoder_mae.pt — do not rebuild):
  --sim {0, 6000}     simulated mixture count (real spectra always all 3339)
  --lib {28, 111}     pure-component library size (28 = deterministic even
                      subsample of the sorted 111 names, recorded in recipe.json)
  --perturb {0, 1}    physical perturbation chain on/off (simulate_mixtures
                      do_perturb flag)

Output: data/pretrain_corpus_ablation/{tag}/  spectra.npy, sources.csv,
        sim_compositions.npz, recipe.json   (tag = s{sim}_l{lib}_p{perturb})

Leakage red line (design doc §5.2): library membership is recorded per recipe;
the exclusion audit vs downstream-task molecules lives in recipe.json for the
molecule-level overlap check.
"""
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.datasets import load_diesel, load_gasoline, load_corn, load_evoo, load_mayonnaise
from src.simulate import build_pure_library, simulate_mixtures

L = 512
OUT = ROOT / "data" / "pretrain_corpus_ablation"


def resample(X):
    n, li = X.shape
    grid_old = np.linspace(0, 1, li)
    grid_new = np.linspace(0, 1, L)
    return np.stack([np.interp(grid_new, grid_old, row) for row in X]).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", type=int, required=True, choices=[0, 1500, 3000, 6000])
    ap.add_argument("--lib", type=int, required=True, choices=[28, 56, 111])
    ap.add_argument("--perturb", type=int, required=True, choices=[0, 1])
    args = ap.parse_args()
    tag = f"s{args.sim}_l{args.lib}_p{args.perturb}"
    out = OUT / tag
    out.mkdir(parents=True, exist_ok=True)

    spectra, sources = [], []
    d = load_diesel(); spectra.append(resample(d["X"])); sources += ["diesel"] * len(d["X"])
    g = load_gasoline(); spectra.append(resample(g["X"])); sources += ["gasoline"] * len(g["X"])
    c = load_corn()
    for inst, (Xi, _wl) in c["instruments"].items():
        spectra.append(resample(Xi)); sources += [f"corn_{inst}"] * len(Xi)
    e = load_evoo(); spectra.append(resample(e["X"])); sources += ["evoo"] * len(e["X"])
    m = load_mayonnaise(); spectra.append(resample(m["X"])); sources += ["mayonnaise"] * len(m["X"])
    z = np.load(ROOT / "data" / "NIST_IR" / "nist_ir.npz", allow_pickle=True)
    Xn = np.stack([np.interp(np.linspace(0, 1, L), np.linspace(0, 1, len(y)), y)
                   for y in z["y"]]).astype(np.float32)
    spectra.append(Xn); sources += ["nist_ir"] * len(Xn)

    lib = {}
    lib_names_full = []
    if args.sim > 0:
        class_spec = {}
        for cls in sorted(set(e["labels"])):
            class_spec[f"evoo_{cls}"] = e["X"][e["labels"] == cls]
        for cls in sorted(set(m["labels"].astype(str))):
            class_spec[f"mayo_{cls}"] = m["X"][m["labels"].astype(str) == cls]
        lib = build_pure_library(nist_npz=ROOT / "data" / "NIST_IR" / "nist_ir.npz",
                                 class_spectra=class_spec)
        lib_names_full = sorted(lib)
        if args.lib < len(lib_names_full):
            keep = set(np.array(lib_names_full)[
                np.linspace(0, len(lib_names_full) - 1, args.lib).round().astype(int)])
            lib = {k: v for k, v in lib.items() if k in keep}
        Xs, C, lib_names = simulate_mixtures(lib, args.sim, L=L, seed=42,
                                             do_perturb=bool(args.perturb))
        spectra.append(Xs.astype(np.float32)); sources += ["simulated"] * args.sim
        np.savez(out / "sim_compositions.npz", X=Xs, C=C,
                 lib_names=np.array(lib_names))

    X = np.vstack(spectra)
    np.save(out / "spectra.npy", X)
    pd.DataFrame(dict(source=sources)).to_csv(out / "sources.csv", index=False)
    (out / "recipe.json").write_text(json.dumps(dict(
        tag=tag, sim=args.sim, lib=args.lib, perturb=args.perturb,
        n_total=int(len(X)), library_members=sorted(lib),
        library_full=sorted(lib_names_full), seed=42,
        note="28-member library = even subsample of the sorted 111 names"),
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[{tag}] corpus {X.shape}; sources:")
    print(pd.Series(sources).value_counts().to_string())


if __name__ == "__main__":
    main()
