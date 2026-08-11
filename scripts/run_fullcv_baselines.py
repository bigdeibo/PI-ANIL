"""Baseline main script: baseline few-shot evaluation + full-CV reference + table and figure output.
Outputs to results/baselines/
"""
import sys, json, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(sys.executable).parent.parent.parent))

from src.datasets import load_diesel, load_gasoline, load_corn, load_evoo
from src.baselines import BASELINES
from src.fewshot import run_fewshot_regression, run_full_cv

OUT = ROOT / "results" / "baselines"
OUT.mkdir(parents=True, exist_ok=True)

# Task configuration: (dataset display name, X, y, property name)
def build_tasks():
    tasks = []
    d = load_diesel()
    for t in ["CN", "BP50", "D4052", "FREEZE", "TOTAL", "VISC", "FLASH"]:
        tasks.append((f"diesel-{t}", d["X"], d["targets"][t], d["wavelengths"]))
    g = load_gasoline()
    tasks.append(("gasoline-octane", g["X"], g["targets"]["octane"], g["wavelengths"]))
    c = load_corn()
    Xm5, ax5 = c["instruments"]["m5"]
    for t in ["oil", "protein", "moisture", "starch"]:
        tasks.append((f"corn_m5-{t}", Xm5, c["targets"][t], ax5))
    e = load_evoo()
    tasks.append(("evoo-adulteration", e["X"], e["targets"]["adulteration_level"], e["wavelengths"]))
    return tasks

FAST_MODELS = ["PLS", "SVR"]
DEEP_MODELS = ["CNN", "CNN+Aug"]
# Deep baselines run only on representative tasks (to bound CPU runtime)
DEEP_TASKS = {"diesel-CN", "gasoline-octane", "corn_m5-oil", "evoo-adulteration"}
SHOTS = (5, 10, 20)


def main():
    tasks = build_tasks()
    rows, full_rows = [], []
    spectra_store = {}
    for name, X, y, wl in tasks:
        spectra_store[name] = (X, y, wl)
        # ---- Full 10-fold CV (PLS/SVR, literature reference) ----
        for mname in FAST_MODELS:
            r = run_full_cv(X, y, BASELINES[mname], preprocess="snv", folds=10)
            full_rows.append(dict(task=name, model=mname, **r))
            print(f"[full-cv] {name} {mname}: R2={r['r2']:.4f} RMSE={r['rmse']:.4f}", flush=True)
        # ---- Few-shot protocol ----
        for mname in FAST_MODELS:
            t0 = time.time()
            res = run_fewshot_regression(X, y, BASELINES[mname], shots=SHOTS,
                                         repeats=30, preprocess="snv", seed=42)
            for K, r in res.items():
                rows.append(dict(task=name, model=mname, shot=K, **r))
            print(f"[fewshot] {name} {mname} done {time.time()-t0:.0f}s", flush=True)
        if name in DEEP_TASKS:
            for mname in DEEP_MODELS:
                t0 = time.time()
                res = run_fewshot_regression(X, y, BASELINES[mname], shots=SHOTS,
                                             repeats=10, preprocess="snv", seed=42)
                for K, r in res.items():
                    rows.append(dict(task=name, model=mname, shot=K, **r))
                print(f"[fewshot] {name} {mname} done {time.time()-t0:.0f}s", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "baseline_fewshot.csv", index=False)
    dff = pd.DataFrame(full_rows)
    dff.to_csv(OUT / "baseline_fullcv.csv", index=False)

    # Markdown summary
    with open(OUT / "baseline_table.md", "w", encoding="utf-8") as f:
        f.write("# Baseline Performance Summary\n\n## Full 10-fold cross-validation (literature reference)\n\n")
        f.write(dff.round(4).to_markdown(index=False))
        f.write("\n\n## Few-shot protocol (5/10/20-shot, mean±std R² / mean RPD)\n\n")
        view = df.assign(
            val=lambda d: d.apply(lambda r: f"{r.r2_mean:.3f}±{r.r2_std:.3f} / {r.rpd_mean:.2f}", axis=1))
        piv = view.pivot_table(index=["task", "model"], columns="shot", values="val",
                               aggfunc="first").reset_index()
        f.write(piv.to_markdown(index=False))
    np.savez(OUT / "spectra_cache.npz",
             **{f"{k}__X": v[0] for k, v in spectra_store.items()},
             **{f"{k}__wl": v[2] for k, v in spectra_store.items()})
    print("saved:", OUT)


if __name__ == "__main__":
    main()
