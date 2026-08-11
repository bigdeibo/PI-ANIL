"""P0 supplementary-experiment summary: lambda sensitivity + LOMO in four directions.

Produces results/physics/lambda_lomo_table.md:
1. Lambda sensitivity (headline 8 tasks: mean R² + failure rate + Wilcoxon vs vanilla)
2. LOMO corn/EVOO directions (vanilla vs PI both constraints + seen-task reference + statistical tests)
3. LOMO four-direction overview (diesel/gasoline/corn/EVOO)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "physics"

HEADLINE = ["diesel-CN", "diesel-BP50", "diesel-D4052", "diesel-FLASH",
            "diesel-FREEZE", "diesel-TOTAL", "diesel-VISC", "gasoline-octane"]
CORN = ["corn_m5-moisture", "corn_m5-oil", "corn_m5-protein", "corn_m5-starch"]


def load_parts():
    rows = []
    for f in (OUT / "parts").glob("*.csv"):
        rows.append(pd.read_csv(f))
    df = pd.concat(rows, ignore_index=True).drop_duplicates(
        ["tag", "task", "shot", "rep"])
    df3 = pd.read_csv(ROOT / "results" / "meta_routes" / "per_rep.csv")
    df3 = df3.rename(columns={"mid": "tag"})
    return pd.concat([df, df3[df3.tag == "anil__mae__multi"]],
                     ignore_index=True)


def collapse(x):
    return float(np.mean(np.asarray(x) < -1))


def wilc(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if len(a) != len(b) or len(a) == 0:
        return np.nan, np.nan, np.nan
    d = b - a
    try:
        p = stats.wilcoxon(d).pvalue
    except Exception:
        p = np.nan
    return float(np.median(d)), float(np.mean(d)), p


def fmt(m, s):
    return f"{m:.3f}±{s:.3f}"


def main():
    df = load_parts()
    VAN = "anil__mae__multi"  # vanilla (multi pool, lambda=0 reference)
    lines = ["# P0 supplementary experiments: lambda sensitivity and LOMO in four directions\n",
             "R² mean±std, paired splits (rep 0-9), MAE initialization.\n"]

    # ---------------- 1. Lambda sensitivity ----------------
    lines += ["\n## 1. Lambda sensitivity sweep (headline 8 tasks)\n",
              "| variant | lambda | K=5 mean | K=5 failure rate | K=10 mean | K=20 mean |",
              "|---|---|---|---|---|---|"]
    sweep = [("vanilla (lambda=0)", VAN), ("PI additivity", "anilpi__add_la0.01__mae__multi"),
             ("PI additivity", "anilpi__add__mae__multi"),
             ("PI additivity", "anilpi__add_la1.0__mae__multi"),
             ("PI band", "anilpi__band_lb0.01__mae__multi"),
             ("PI band", "anilpi__band__mae__multi"),
             ("PI band", "anilpi__band_lb1.0__mae__multi"),
             ("PI both constraints", "anilpi__both_la0.01_lb0.01__mae__multi"),
             ("PI both constraints", "anilpi__both__mae__multi")]
    lam_lab = {VAN: "0", "anilpi__add_la0.01__mae__multi": "0.01",
               "anilpi__add__mae__multi": "0.1",
               "anilpi__add_la1.0__mae__multi": "1.0",
               "anilpi__band_lb0.01__mae__multi": "0.01",
               "anilpi__band__mae__multi": "0.1",
               "anilpi__band_lb1.0__mae__multi": "1.0",
               "anilpi__both_la0.01_lb0.01__mae__multi": "0.01/0.01",
               "anilpi__both__mae__multi": "0.1/0.1"}
    for name, tag in sweep:
        d = df[(df.tag == tag) & (df.task.isin(HEADLINE))]
        cells = []
        for K in (5, 10, 20):
            x = d[d.shot == K].r2.to_numpy()
            cells.append((x.mean(), collapse(x)))
        lines.append(f"| {name} | {lam_lab[tag]} | {cells[0][0]:.3f} | "
                     f"{cells[0][1]:.0%} | {cells[1][0]:.3f} | {cells[2][0]:.3f} |")
    # Paired Wilcoxon for each lambda vs vanilla (5-shot, cell level + task level)
    lines += ["\n### Paired Wilcoxon vs vanilla (headline 8 tasks, 5-shot only)\n",
              "| variant lambda | median diff | mean diff | p (cell level) | task-level median diff | p (task level) |",
              "|---|---|---|---|---|---|"]
    dv = df[(df.tag == VAN) & (df.task.isin(HEADLINE)) & (df.shot == 5)]
    for name, tag in sweep[1:]:
        d = df[(df.tag == tag) & (df.task.isin(HEADLINE)) & (df.shot == 5)]
        a = dv.sort_values(["task", "rep"]).r2.to_numpy()
        b = d.sort_values(["task", "rep"]).r2.to_numpy()
        med, mean, p = wilc(a, b)
        td = []
        for t in HEADLINE:
            at = dv[dv.task == t].sort_values("rep").r2.to_numpy()
            bt = d[d.task == t].sort_values("rep").r2.to_numpy()
            td.append(bt.mean() - at.mean())
        _, _, pt = wilc(np.zeros(len(td)), np.array(td))
        lines.append(f"| {name} lambda={lam_lab[tag]} | {med:+.3f} | {mean:+.3f} | "
                     f"{p:.4f} | {np.median(td):+.3f} | {pt:.4f} |")

    # ---------------- 2. LOMO corn/EVOO ----------------
    lines += ["\n## 2. New LOMO directions: corn and EVOO\n"]
    lomo_cfg = [("no_corn", CORN), ("no_evoo", ["evoo-adulteration"])]
    for pool, tasks in lomo_cfg:
        lines.append(f"\n### Pool = {pool} (test tasks: {', '.join(tasks)})\n")
        lines.append("| task | model | 5 | 10 | 20 |")
        lines.append("|---|---|---|---|---|")
        for t in tasks:
            for lab, tag in [("vanilla", f"anilpi__vanilla__mae__{pool}"),
                             ("PI both constraints", f"anilpi__both__mae__{pool}"),
                             ("seen-task reference (route-comparison multi pool)", VAN)]:
                d = df[(df.tag == tag) & (df.task == t)]
                if len(d) == 0:
                    continue
                cells = []
                for K in (5, 10, 20):
                    x = d[d.shot == K].r2.to_numpy()
                    cells.append(fmt(x.mean(), x.std()) if len(x) else "-")
                lines.append(f"| {t} | {lab} | {cells[0]} | {cells[1]} | {cells[2]} |")
        # tests
        a = df[df.tag == f"anilpi__vanilla__mae__{pool}"]
        b = df[df.tag == f"anilpi__both__mae__{pool}"]
        key = ["task", "shot", "rep"]
        m = a.merge(b, on=key, suffixes=("_v", "_p"))
        med, mean, p = wilc(m.r2_v, m.r2_p)
        td = []
        for t in tasks:
            mm = m[m.task == t]
            td.append((mm.r2_p - mm.r2_v).mean())
        _, _, pt = wilc(np.zeros(len(td)), np.array(td))
        lines.append(f"\nPaired Wilcoxon (PI - vanilla): n={len(m)}, median diff {med:+.3f}, "
                     f"mean diff {mean:+.3f}, p={p:.4g}; "
                     f"task level: {sum(np.array(td) > 0)}/{len(td)} positive, "
                     f"median {np.median(td):+.3f}, p={pt:.4f}\n")

    # ---------------- 3. Four-direction overview ----------------
    lines += ["\n## 3. LOMO four-direction overview (PI both constraints - vanilla)\n",
              "| direction (test material) | n (cells) | median diff | mean diff | p (cell level) | task-level positive | p (task level) |",
              "|---|---|---|---|---|---|---|"]
    for lab, pool, tasks in [
            ("diesel", "no_diesel", [t for t in HEADLINE if t.startswith("diesel")]),
            ("gasoline", "no_gasoline", ["gasoline-octane"]),
            ("corn", "no_corn", CORN),
            ("EVOO", "no_evoo", ["evoo-adulteration"])]:
        a = df[(df.tag == f"anilpi__vanilla__mae__{pool}") & (df.task.isin(tasks))]
        b = df[(df.tag == f"anilpi__both__mae__{pool}") & (df.task.isin(tasks))]
        m = a.merge(b, on=["task", "shot", "rep"], suffixes=("_v", "_p"))
        med, mean, p = wilc(m.r2_v, m.r2_p)
        td = [(m[m.task == t].r2_p - m[m.task == t].r2_v).mean() for t in tasks]
        _, _, pt = wilc(np.zeros(len(td)), np.array(td))
        lines.append(f"| {lab} | {len(m)} | {med:+.3f} | {mean:+.3f} | {p:.4g} | "
                     f"{sum(np.array(td) > 0)}/{len(td)} | {pt:.4g} |")

    # LOMO diesel comparison for both-constraints lambda=0.01 (vanilla / both 0.1 / both 0.01)
    lines += ["\n### LOMO diesel: both constraints lambda=0.01 vs lambda=0.1 vs vanilla\n",
              "| model | K=5 | K=10 | K=20 | vs vanilla p (cell level) | task-level positive |",
              "|---|---|---|---|---|---|"]
    dtasks = [t for t in HEADLINE if t.startswith("diesel")]
    av = df[(df.tag == "anilpi__vanilla__mae__no_diesel") & (df.task.isin(dtasks))]
    for lab, tag in [("vanilla", "anilpi__vanilla__mae__no_diesel"),
                     ("PI both lambda=0.1", "anilpi__both__mae__no_diesel"),
                     ("PI both lambda=0.01", "anilpi__both_la0.01_lb0.01__mae__no_diesel")]:
        d = df[(df.tag == tag) & (df.task.isin(dtasks))]
        cells = [d[d.shot == K].r2.mean() for K in (5, 10, 20)]
        m = av.merge(d, on=["task", "shot", "rep"], suffixes=("_v", "_p"))
        med, mean, p = wilc(m.r2_v, m.r2_p)
        td = [(m[m.task == t].r2_p - m[m.task == t].r2_v).mean() for t in dtasks]
        lines.append(f"| {lab} | {cells[0]:.3f} | {cells[1]:.3f} | {cells[2]:.3f} | "
                     f"{p:.4g} | {sum(np.array(td) > 0)}/7 |")

    text = "\n".join(lines) + "\n"
    (OUT / "lambda_lomo_table.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
