"""P2a merge: SimCLR equal-budget (60ep) retraining vs MAE-60, closing R2-M2.

Four-encoder comparison: MAE-60 / SimCLR-32 / SimCLR-cont60 (continued to 60ep) /
SimCLR-scratch60 (trained from scratch for 60ep).
Determines whether the §3.1 conclusion "reconstruction-based SSL outperforms
contrastive SSL (quantitatively)" still holds under an equal budget (60ep).

Protocol: frozen encoder + ridge-regression probe, 11 tasks × {5,10,20}-shot × 30 reps,
seed formula 42*1000+K*100+rep paired with the baselines (bit-identical).
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, binomtest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT = ROOT / "results" / "pretraining"
PARTS = OUT / "parts"

TASKS11 = ([f"diesel-{a}" for a in
            ("CN", "BP50", "D4052", "FLASH", "FREEZE", "TOTAL", "VISC")]
           + ["gasoline-octane", "corn_m5-protein", "corn_m5-oil", "evoo-adulteration"])
HEADLINE = [t for t in TASKS11 if t not in
            ("corn_m5-protein", "corn_m5-oil", "evoo-adulteration")]  # diesel7 + gasoline = 8

MODELS = [("MAE-60", "mae_ridge"),
          ("SimCLR-32", "simclr_ridge"),
          ("SimCLR-cont60", "simclr_cont60_ridge"),
          ("SimCLR-scratch60", "simclr_scratch60_ridge")]


def _md_table(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(str(c) for c in cols) + " |",
           "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in df.iterrows():
        out.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(out)


def load_parts():
    df = pd.concat([pd.read_csv(p) for p in PARTS.glob("probe__*.csv")],
                   ignore_index=True)
    return df[df.model.isin([m for _, m in MODELS])].copy()


def main():
    df = load_parts()

    lines = ["# P2a: SimCLR equal-budget (60ep) retraining vs MAE-60 (closing R2-M2)\n",
             "R2-M2 concern: §3.1 \"SimCLR pretraining was systematically weaker than MAE\" "
             "rested on unequal training budgets (MAE-60ep vs SimCLR-32ep).\n",
             "This experiment retrains SimCLR to 60ep under an equal budget (two variants "
             "cross-validated: cont60 continued from 32ep, scratch60 retrained from scratch "
             "with the same seed) and tests whether the conclusion still holds.\n\n",
             "Protocol: frozen encoder + ridge-regression probe, 11 tasks × {5,10,20}-shot × 30 reps, "
             "seed formula paired with the baselines (bit-identical)."]

    # Table 1: per-task median R²
    lines.append("\n## 1. Per-task median R² (30 reps)\n")
    for K in (5, 10, 20):
        lines.append(f"\n### K={K}-shot\n")
        rows = []
        for task in TASKS11:
            row = dict(task=task)
            for name, m in MODELS:
                sub = df[(df.task == task) & (df.model == m) & (df.shot == K)]
                row[name] = f"{sub.r2.median():.3f}" if len(sub) else "—"
            rows.append(row)
        lines.append(_md_table(pd.DataFrame(rows)))

    # Table 2: headline-8 task summary
    lines.append("\n\n## 2. Headline-8 task summary (mean R² / failure rate = fraction with R²<-1)\n")
    rows = []
    for name, m in MODELS:
        sub = df[(df.model == m) & (df.task.isin(HEADLINE))]
        row = dict(encoder=name)
        for K in (5, 10, 20):
            g = sub[sub.shot == K].r2
            row[f"K={K} R²"] = f"{g.mean():.3f}" if len(g) else "—"
            row[f"K={K} fail"] = f"{(g < -1).mean() * 100:.0f}%" if len(g) else "—"
        rows.append(row)
    lines.append(_md_table(pd.DataFrame(rows)))

    # Table 3: paired two-level tests (MAE-60 vs each SimCLR, 30 paired reps)
    lines.append("\n\n## 3. Paired two-level tests (30 bit-identical reps)\n")
    lines.append("Rep-level Wilcoxon (pooled across task×rep) + task-level sign test (binomial).\n")
    pairs = [
        ("MAE-60 vs SimCLR-32 (original unequal 60 vs 32)", "mae_ridge", "simclr_ridge"),
        ("MAE-60 vs SimCLR-cont60 (equal budget, continued)", "mae_ridge", "simclr_cont60_ridge"),
        ("MAE-60 vs SimCLR-scratch60 (equal budget, from scratch)", "mae_ridge", "simclr_scratch60_ridge"),
        ("cont60 vs scratch60 (consistency of the two equal-budget variants)", "simclr_cont60_ridge",
         "simclr_scratch60_ridge"),
    ]
    rows = []
    for K in (5, 10, 20):
        for name, a, b in pairs:
            da = df[(df.model == a) & (df.shot == K)]
            db = df[(df.model == b) & (df.shot == K)]
            j = da.merge(db, on=["task", "rep"], suffixes=("_a", "_b"))
            j = j.dropna(subset=["r2_a", "r2_b"])
            if len(j) < 8:
                continue
            diff = j.r2_a - j.r2_b
            p_w = float(wilcoxon(diff).pvalue) if (diff != 0).any() else 1.0
            tag = j.groupby("task").agg(da=("r2_a", "median"), db=("r2_b", "median"))
            tdiff = tag["da"] - tag["db"]
            npos = int((tdiff > 0).sum())
            ntask = len(tdiff)
            p_sign = float(binomtest(npos, ntask, 0.5).pvalue) if ntask else 1.0
            rows.append(dict(shot=K, comparison=name, n=len(j),
                             median_diff=f"{diff.median():+.3f}",
                             Wilcoxon_p=f"{p_w:.2e}",
                             task_level=f"{npos}/{ntask} pos",
                             sign_p=f"{p_sign:.3f}"))
    lines.append(_md_table(pd.DataFrame(rows)))
    lines.append("\n(positive sign = the former's median is better than the latter's; no multiple-comparison correction.)\n")

    (OUT / "equal_budget_table.md").write_text("\n".join(lines), encoding="utf-8")
    print("merged ->", OUT / "equal_budget_table.md")


if __name__ == "__main__":
    main()
