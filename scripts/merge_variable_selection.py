"""P1b merge: variable-selection strong baselines (PLS / UVE-PLS / CARS-PLS) vs
meta-learning (ANIL / PI-ANIL), paired statistics + table generation. Closes R1-M3
(the chemometrics reviewers' top objection: why not CARS/UVE-PLS?).

Protocol: 30 reps for variable-selection baselines, 10 reps for meta-learning
(each method's standard); paired statistics use rep 0-9 (the same splits,
bit-identical to eval_split). Two-level tests: rep-level Wilcoxon (pooled across
tasks x reps) + task-level sign test (sign of the median difference across the 13
tasks, binomial; mitigates pseudoreplication).
Leakage prevention: both SNV and variable selection are fitted on the support set.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, binomtest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT = ROOT / "results" / "baselines"
P1B = OUT / "variable_selection_parts"
S3 = ROOT / "results" / "meta_routes"
S4 = ROOT / "results" / "physics"

HEADLINE = ["diesel-CN", "diesel-BP50", "diesel-D4052", "diesel-FLASH",
            "diesel-FREEZE", "diesel-TOTAL", "diesel-VISC", "gasoline-octane"]
ALL13 = HEADLINE + ["corn_m5-oil", "corn_m5-protein", "corn_m5-moisture",
                    "corn_m5-starch", "evoo-adulteration"]

MODELS = ["PLS", "UVEPLS", "CARSPLS", "ANIL", "PI-ANIL"]
MODEL_NAME = {"PLS": "PLS", "UVEPLS": "UVE-PLS", "CARSPLS": "CARS-PLS",
              "ANIL": "ANIL(MAE)", "PI-ANIL": "PI-ANIL"}


def _md_table(df):
    """Dependency-free markdown table (replaces pandas.to_markdown; no tabulate needed)."""
    cols = list(df.columns)
    out = ["| " + " | ".join(str(c) for c in cols) + " |",
           "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in df.iterrows():
        out.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(out)


def load_p1b():
    df = pd.concat([pd.read_csv(p) for p in P1B.glob("*.csv")
                    if not p.name.startswith("log_")], ignore_index=True)
    return df  # task,model,shot,rep,r2,rmse,rpd,n_var


def load_meta():
    """Per-rep results for ANIL (stage3 vanilla, mid=anil__mae__multi) + PI-ANIL both (stage4)."""
    s3 = pd.read_csv(S3 / "per_rep.csv")
    anil = s3[s3.mid == "anil__mae__multi"][["task", "shot", "rep", "r2"]].copy()
    anil["model"] = "ANIL"
    s4 = pd.concat([pd.read_csv(p) for p in (S4 / "parts").glob("*.csv")],
                   ignore_index=True)
    pi = s4[s4.tag == "anilpi__both__mae__multi"][["task", "shot", "rep", "r2"]].copy()
    pi["model"] = "PI-ANIL"
    return pd.concat([anil, pi], ignore_index=True)


def main():
    p1b = load_p1b()
    meta = load_meta()
    # pairing: rep 0-9 (meta-learning has only 10 reps; bit-identical splits)
    p1b_p = p1b[p1b.rep < 10][["task", "model", "shot", "rep", "r2", "n_var"]].copy()
    meta_p = meta[meta.rep < 10][["task", "model", "shot", "rep", "r2"]].copy()
    meta_p["n_var"] = np.nan
    paired = pd.concat([p1b_p, meta_p], ignore_index=True)
    paired.to_csv(OUT / "variable_selection_paired_per_rep.csv", index=False)

    lines = ["# P1b strong baselines: variable-selection PLS (UVE/CARS) vs meta-learning (ANIL/PI-ANIL)\n",
             "Closes R1-M3 (the chemometrics reviewers' top objection: why not CARS/UVE-PLS?).\n\n",
             "Protocol: 30 reps for variable-selection baselines, 10 reps for meta-learning (each method's standard); ",
             "paired statistics use rep 0-9 (the same splits, bit-identical to eval_split). ",
             "Both SNV and variable selection are fitted on the support set (leakage prevention).\n"]

    # ---- Table 1: per-task R² median (each method's standard rep count) ----
    lines.append("\n## 1. Per-task R² median\n")
    lines.append("PLS/UVE/CARS based on 30 reps; ANIL/PI-ANIL based on 10 reps (each method's standard protocol).\n")
    for K in (5, 10, 20):
        lines.append(f"\n### K={K}-shot\n")
        rows = []
        for task in ALL13:
            row = dict(task=task)
            for m in MODELS:
                src = p1b if m in ("PLS", "UVEPLS", "CARSPLS") else meta
                sub = src[(src.task == task) & (src.model == m) & (src.shot == K)]
                row[MODEL_NAME[m]] = f"{sub.r2.median():.3f}" if len(sub) else "—"
            rows.append(row)
        lines.append(_md_table(pd.DataFrame(rows)))

    # ---- Table 2: headline 8-task summary (mean + failure rate) ----
    lines.append("\n\n## 2. Headline 8-task summary (R² mean / failure rate = fraction with R²<-1)\n")
    rows = []
    for m in MODELS:
        src = p1b if m in ("PLS", "UVEPLS", "CARSPLS") else meta
        sub = src[(src.model == m) & (src.task.isin(HEADLINE))]
        row = dict(method=MODEL_NAME[m])
        for K in (5, 10, 20):
            g = sub[sub.shot == K].r2
            row[f"K={K} R²"] = f"{g.mean():.3f}" if len(g) else "—"
            row[f"K={K} failure"] = f"{(g < -1).mean() * 100:.0f}%" if len(g) else "—"
        rows.append(row)
    lines.append(_md_table(pd.DataFrame(rows)))

    # ---- Table 3: variable-selection statistics (selected variable count + fallback rate) ----
    lines.append("\n\n## 3. Variable-selection statistics (median selected variable count for UVE/CARS / fallback rate)\n")
    lines.append("Fallback = fewer than 2 variables selected, falling back to full-spectrum plain PLS (n_var ~ full variable count).\n")
    rows = []
    for m in ["UVEPLS", "CARSPLS"]:
        for K in (5, 10, 20):
            for task in ALL13:
                st = p1b[(p1b.model == m) & (p1b.shot == K) & (p1b.task == task)]
                p_tot = p1b[(p1b.task == task) & (p1b.shot == K) & (p1b.model == "PLS")]
                if not len(st) or not len(p_tot):
                    continue
                full = int(p_tot.n_var.iloc[0])
                fbk = (st.n_var >= full - 1).mean() * 100
                rows.append(dict(model=MODEL_NAME[m], shot=K, task=task,
                                 nvar_med=int(st.n_var.median()),
                                 fallback=f"{fbk:.0f}%"))
    lines.append(_md_table(pd.DataFrame(rows)))

    # ---- Table 4: paired two-level tests (rep 0-9, bit-identical) ----
    lines.append("\n\n## 4. Paired two-level tests (rep 0-9 strict pairing)\n")
    lines.append("Rep-level Wilcoxon (pooled across tasks x reps) + task-level sign test "
                 "(sign of the median difference across the 13 tasks, binomial).\n")
    pairs = [
        ("PI-ANIL vs PLS", "PI-ANIL", "PLS"),
        ("PI-ANIL vs UVE-PLS", "PI-ANIL", "UVEPLS"),
        ("PI-ANIL vs CARS-PLS", "PI-ANIL", "CARSPLS"),
        ("ANIL vs PLS", "ANIL", "PLS"),
        ("ANIL vs UVE-PLS", "ANIL", "UVEPLS"),
        ("ANIL vs CARS-PLS", "ANIL", "CARSPLS"),
    ]
    rows = []
    for K in (5, 10, 20):
        for name, a, b in pairs:
            da = paired[(paired.model == a) & (paired.shot == K)]
            db = paired[(paired.model == b) & (paired.shot == K)]
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
            rows.append(dict(shot=K, comparison=name, n_rep=len(j),
                             median_diff=f"{diff.median():+.3f}",
                             Wilcoxon_p=f"{p_w:.2e}",
                             task_level=f"{npos}/{ntask} positive",
                             sign_p=f"{p_sign:.3f}"))
    lines.append(_md_table(pd.DataFrame(rows)))
    lines.append("\n(No multiple-comparison correction; task level = on how many tasks "
                 "PI-ANIL/ANIL beats the baseline by the median.)\n")

    (OUT / "variable_selection_table.md").write_text("\n".join(lines), encoding="utf-8")
    print("merged ->", OUT / "variable_selection_table.md")


if __name__ == "__main__":
    main()
