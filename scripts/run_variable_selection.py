"""P1b: variable-selection strong baselines (UVE-PLS / CARS-PLS) + PLS, per-rep few-shot evaluation.

Reuses the unified evaluation protocol's seed formula seed*1000+K*100+rep (seed=42),
bit-identical to src/fsl_regression/core.eval_split and
src/fewshot.run_fewshot_regression, so per-rep results can be paired with
stage3 (ANIL) / stage4 (PI-ANIL) on rep 0-9 for paired statistics
(two-level Wilcoxon + sign test).

Leakage-prevention rule: both SNV and variable selection are fitted on the support
set X[tr] only; the test set is only transformed/predicted.
Output: results/baselines/variable_selection_parts/{model}__{task}.csv
        columns: task, model, shot, rep, r2, rmse, rpd, n_var

Usage:
  python scripts/run_p1b_baselines.py --model PLS --task all --reps 30
  python scripts/run_p1b_baselines.py --model UVEPLS,CARSPLS --task all
  python scripts/run_p1b_baselines.py --model all --task all
"""
import sys
import time
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.datasets import load_diesel, load_gasoline, load_corn, load_evoo
from src.baselines import BASELINES
from src.preprocess import Preprocess
from src.fsl_regression.core import eval_split  # reuse grouped splits (EVOO: prevent sample_id replicate cross-boundary leakage)

OUT = ROOT / "results" / "baselines" / "variable_selection_parts"
OUT.mkdir(parents=True, exist_ok=True)
SHOTS = (5, 10, 20)
SEED = 42


def build_tasks():
    """The 13 'material x property' tasks, consistent with run_stage0_part.py / core.TASKS.

    Returns (X, y, groups): groups carries sample_id only for EVOO (to prevent HSI
    replicate-spectra cross-boundary leakage); None otherwise.
    """
    tasks = {}
    d = load_diesel()
    for t in ["CN", "BP50", "D4052", "FREEZE", "TOTAL", "VISC", "FLASH"]:
        tasks[f"diesel-{t}"] = (d["X"], d["targets"][t], None)
    g = load_gasoline()
    tasks["gasoline-octane"] = (g["X"], g["targets"]["octane"], None)
    c = load_corn()
    Xm, _ = c["instruments"]["m5"]
    for t in ["oil", "protein", "moisture", "starch"]:
        tasks[f"corn_m5-{t}"] = (Xm, c["targets"][t], None)
    e = load_evoo()
    tasks["evoo-adulteration"] = (e["X"], e["targets"]["adulteration_level"],
                                  np.asarray(e["ids"]))
    return tasks


def rpd(y, rmse):
    s = np.std(y, ddof=1)
    return s / rmse if rmse > 1e-12 else np.inf


def run_task_model(name, X, y, groups, model_name, reps=30):
    """Per-rep few-shot evaluation reusing the seed formula; SNV is fitted on the
    support set (leakage prevention).

    When groups is not None (EVOO), uses the grouped split of core.eval_split: the K
    support spectra come from K distinct physical samples, and the test set excludes
    all members of the support groups, eliminating sample_id replicate
    cross-boundary leakage.
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    mask = ~np.isnan(y)
    X, y = X[mask], y[mask]
    if groups is not None:
        groups = np.asarray(groups)[mask]
    n = len(y)
    rows = []
    for K in SHOTS:
        if n <= K + 5:
            continue
        for rep in range(reps):
            tr, te = eval_split(n, K, rep, groups=groups)  # groups=None -> bit-identical to the original protocol
            pp = Preprocess("snv").fit(X[tr])
            Xs, Xte = pp.transform(X[tr]), pp.transform(X[te])
            m = BASELINES[model_name](seed=rep)
            try:
                m.fit(Xs, y[tr])
                pred = m.predict(Xte)
                n_var = int(getattr(m, "n_var_", X.shape[1]))
            except Exception:
                continue
            yt = y[te]
            rmse = float(np.sqrt(np.mean((pred - yt) ** 2)))
            ss_res = float(np.sum((pred - yt) ** 2))
            ss_tot = float(np.sum((yt - yt.mean()) ** 2))
            r2 = 1 - ss_res / ss_tot if ss_tot > 1e-12 else np.nan
            rows.append(dict(task=name, model=model_name, shot=K, rep=rep,
                             r2=r2, rmse=rmse, rpd=rpd(yt, rmse), n_var=n_var))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="PLS|UVEPLS|CARSPLS, comma-separated, or all")
    ap.add_argument("--task", required=True, help="task name or all")
    ap.add_argument("--reps", type=int, default=30)
    args = ap.parse_args()

    models = (["PLS", "UVEPLS", "CARSPLS"] if args.model == "all"
              else [m.strip() for m in args.model.split(",")])
    tasks = build_tasks()
    sel = (list(tasks) if args.task == "all"
           else [t.strip() for t in args.task.split(",")])

    for model_name in models:
        for name in sel:
            if name not in tasks:
                print(f"[skip] unknown task {name}")
                continue
            X, y, groups = tasks[name]
            t0 = time.time()
            rows = run_task_model(name, X, y, groups, model_name, args.reps)
            df = pd.DataFrame(rows)
            out_file = OUT / f"{model_name}__{name}.csv"
            df.to_csv(out_file, index=False)
            by_shot = df.groupby("shot").agg(
                r2_med=("r2", "median"), r2_mean=("r2", "mean"),
                crash=("r2", lambda s: (s < -1).mean() * 100),
                nvar_med=("n_var", "median")).reset_index()
            print(f"[{model_name}] {name} ({time.time() - t0:.0f}s)", flush=True)
            for _, r in by_shot.iterrows():
                print(f"    K={int(r.shot)}: R2med={r.r2_med:.3f} "
                      f"crash={r.crash:.0f}% nvar={r.nvar_med:.0f}", flush=True)
    print("done.")


if __name__ == "__main__":
    main()
