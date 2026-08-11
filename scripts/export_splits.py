"""Export paired evaluation split indices (for bit-identical independent re-verification).

Main protocol: 15 tasks (diesel×7 / gasoline / corn_m5×4 / evoo / selfmix / edibleoil) ×
K∈{5,10,20} × rep 0–29 (deep/meta-learning methods use reps 0–9, classical baselines
use 0–29; a superset is exported here).
Exported per task (diesel-CN has a different NaN mask from the other 6 properties,
n=381 vs 395, so they are not merged).

Outputs (splits/):
  {task}_support.csv    task, K, rep, position, index (index = row number in the task's NaN-filtered array)
  {task}_test.csv       grouped tasks only (evoo/selfmix/edibleoil): test excludes all members of support groups
  trip_support.csv      TRIP test7 × K∈{5,10,25} × rep 0–9 (query = full query pool)
For non-grouped tasks, test = setdiff(arange(n), support); see splits/README.md.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.fsl_regression.core import load_task, eval_split

OUT = ROOT / "splits"
OUT.mkdir(parents=True, exist_ok=True)

MAIN_TASKS = ([f"diesel-{a}" for a in ("CN", "BP50", "D4052", "FLASH", "FREEZE", "TOTAL", "VISC")]
              + ["gasoline-octane"]
              + [f"corn_m5-{a}" for a in ("moisture", "oil", "protein", "starch")]
              + ["evoo-adulteration", "selfmix-phi", "edibleoil-pv"])
GROUPED = {"evoo-adulteration", "selfmix-phi", "edibleoil-pv"}
KS_MAIN = (5, 10, 20)
REPS_MAIN = range(30)


def main():
    for task in MAIN_TASKS:
        d = load_task(task)
        n = len(d["y"])
        groups = d["groups"]
        grouped = task in GROUPED
        assert (groups is not None) == grouped, f"{task}: groups flag does not match the data"
        sup_rows, te_rows = [], []
        for K in KS_MAIN:
            for rep in REPS_MAIN:
                tr, te = eval_split(n, K, rep, groups=groups)
                for pos, idx in enumerate(tr):
                    sup_rows.append((task, K, rep, pos, int(idx)))
                if grouped:
                    te_rows.extend((task, K, rep, int(i)) for i in te)
        safe = task.replace("/", "_")
        pd.DataFrame(sup_rows, columns=["task", "K", "rep", "position", "index"]
                     ).to_csv(OUT / f"{safe}_support.csv", index=False)
        if grouped:
            pd.DataFrame(te_rows, columns=["task", "K", "rep", "index"]
                         ).to_csv(OUT / f"{safe}_test.csv", index=False)
        print(f"{task}: n={n}, grouped={grouped}, "
              f"support_rows={len(sup_rows)}, test_rows={len(te_rows)}")

    # TRIP test7 support-set indices (query is always the full query pool)
    from src.fsl_regression.trip_data import TRIP_TEST, load_trip_task
    rows = []
    for task in TRIP_TEST:
        d = load_trip_task(task)
        n_s = len(d["y_supp"])
        for K in (5, 10, 25):
            for rep in range(10):
                rng = np.random.default_rng(42 * 1000 + K * 100 + rep)
                sup = rng.choice(n_s, K, replace=False)
                for pos, idx in enumerate(sup):
                    rows.append((task, K, rep, pos, int(idx)))
        print(f"trip/{task}: n_support_pool={n_s}")
    pd.DataFrame(rows, columns=["task", "K", "rep", "position", "index"]
                 ).to_csv(OUT / "trip_support.csv", index=False)
    print("done ->", OUT)


if __name__ == "__main__":
    main()
