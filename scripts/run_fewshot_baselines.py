"""Few-shot baseline execution script: runs a single (task × model) and appends results to results/baselines/parts/
Usage:
  python scripts/run_stage0_part.py --task diesel-CN --model PLS
  python scripts/run_stage0_part.py --task all --model PLS --reps 30
  python scripts/run_stage0_part.py --fullcv --task all --model PLS
"""
import sys, argparse, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.datasets import load_diesel, load_gasoline, load_corn, load_evoo
from src.baselines import BASELINES
from src.fewshot import run_fewshot_regression, run_full_cv

PARTS = ROOT / "results" / "baselines" / "parts"
PARTS.mkdir(parents=True, exist_ok=True)
SHOTS = (5, 10, 20)


def build_tasks(only=None):
    """Build the task dictionary. only=None loads everything (identical behavior to
    the original version); when a specific task name is given, only its own
    dataset is loaded — so single-task reproduction does not depend on
    unrelated datasets being present (numerics unchanged; only the loading
    scope is narrowed; release-repo robustness change, 2026-08-10)."""
    def need(prefix):
        return only is None or only == "all" or only.startswith(prefix)
    tasks = {}
    if need("diesel"):
        d = load_diesel()
        for t in ["CN", "BP50", "D4052", "FREEZE", "TOTAL", "VISC", "FLASH"]:
            tasks[f"diesel-{t}"] = (d["X"], d["targets"][t], d["wavelengths"])
    if need("gasoline"):
        g = load_gasoline()
        tasks["gasoline-octane"] = (g["X"], g["targets"]["octane"], g["wavelengths"])
    if need("corn"):
        c = load_corn()
        Xm5, ax5 = c["instruments"]["m5"]
        for t in ["oil", "protein", "moisture", "starch"]:
            tasks[f"corn_m5-{t}"] = (Xm5, c["targets"][t], ax5)
    if need("evoo"):
        e = load_evoo()
        tasks["evoo-adulteration"] = (e["X"], e["targets"]["adulteration_level"], e["wavelengths"])
    return tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--fullcv", action="store_true")
    args = ap.parse_args()

    tasks = build_tasks(only=args.task)
    sel = list(tasks) if args.task == "all" else [args.task]
    out_file = PARTS / f"{'fullcv' if args.fullcv else 'fewshot'}__{args.model.replace('+','')}.csv"

    for name in sel:
        X, y, wl = tasks[name]
        t0 = time.time()
        rows = []
        if args.fullcv:
            r = run_full_cv(X, y, BASELINES[args.model], preprocess="snv", folds=10)
            rows.append(dict(task=name, model=args.model, **r))
            print(f"[fullcv] {name}: R2={r['r2']:.4f} RMSE={r['rmse']:.4f} RPD={r['rpd']:.2f} ({time.time()-t0:.0f}s)", flush=True)
        else:
            res = run_fewshot_regression(X, y, BASELINES[args.model], shots=SHOTS,
                                         repeats=args.reps, preprocess="snv", seed=42)
            for K, r in res.items():
                rows.append(dict(task=name, model=args.model, shot=K, **r))
                print(f"[fewshot] {name} K={K}: R2={r['r2_mean']:.4f}±{r['r2_std']:.3f} RMSE={r['rmse_mean']:.4f} RPD={r['rpd_mean']:.2f} ({time.time()-t0:.0f}s)", flush=True)
        df = pd.DataFrame(rows)
        header = not out_file.exists()
        df.to_csv(out_file, mode="a", header=header, index=False)
    print("appended ->", out_file)


if __name__ == "__main__":
    main()
