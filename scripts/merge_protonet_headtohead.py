"""P1c merge: ProtoNet regression (frozen SSL / meta-trained) vs meta-learning
(PI-ANIL / ANIL) and traditional baselines, paired statistics + table generation.
Closes the loop on Garzón 2025's "ProtoNet family for few-shot spectroscopic
quantification".

Core question (the Garzón gap): Garzón claims the ProtoNet family
(MAML/ProtoNet/SNAIL) is suitable for few-shot spectroscopic quantification, and
that ProtoNet often outperforms MAML across tasks. This project advocates ANIL
(which, like ProtoNet, belongs to the frozen-encoder family) + physics constraints.
P1c is a direct head-to-head: ProtoNet regression vs PI-ANIL / vanilla ANIL.

Two ProtoNet variants (the only difference = encoder source):
  protonet_frozen : frozen MAE encoder (= the MAE counterpart of Garzón's "SSL+PR")
  protonet_meta   : meta-trained ProtoNet encoder (= the regression counterpart of
                    Garzón's "meta-trained ProtoNet")

Protocol: 30 reps for ProtoNet (frozen-readout convention, same as kNN/GP/PLS);
10 reps for PI-ANIL/ANIL (meta-learning convention).
Pairing: bit-identical eval_split (seed*1000+K*100+rep); merge on (task,rep)
automatically takes the common reps (rep 0-9 vs meta-learning, rep 0-29 vs
traditional). Two-level tests: rep-level Wilcoxon + task-level sign test.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, binomtest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT = ROOT / "results" / "protonet_headtohead"
P1C = OUT / "parts"
P1B = ROOT / "results" / "baselines" / "variable_selection_parts"
S3 = ROOT / "results" / "meta_routes"
S3P = S3 / "parts"
S4 = ROOT / "results" / "physics"
S4P = S4 / "parts"

HEADLINE = ["diesel-CN", "diesel-BP50", "diesel-D4052", "diesel-FLASH",
            "diesel-FREEZE", "diesel-TOTAL", "diesel-VISC", "gasoline-octane"]
# tasks covered by ProtoNet and shared with PI-ANIL multi (headline 8 + evoo)
COMMON9 = HEADLINE + ["evoo-adulteration"]

NAME = {"protonet_frozen": "ProtoNet (frozen MAE)", "protonet_meta": "ProtoNet (meta-trained)",
        "PI-ANIL": "PI-ANIL", "ANIL": "ANIL (MAE)", "PLS": "PLS",
        "UVEPLS": "UVE-PLS", "CARSPLS": "CARS-PLS", "kNN": "kNN (MAE)", "GP": "GP (MAE)"}


def _md(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(str(c) for c in cols) + " |",
           "| " + " | ".join("---" for _ in cols) + " |"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out)


def load_protonet():
    out = []
    for mode in ("frozen", "meta"):
        fs = list(P1C.glob(f"protonet_{mode}__*.csv"))
        if not fs:
            continue
        df = pd.concat([pd.read_csv(p) for p in fs], ignore_index=True)
        df = df[["task", "shot", "rep", "r2"]].copy()
        df["model"] = f"protonet_{mode}"
        out.append(df)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def load_meta():
    """Per-rep results for PI-ANIL (stage4 both) + ANIL (stage3 multi).

    Note: stage4 trained/evaluated only the both/add/band variants and the
    LOMO-form vanilla on the multi pool — vanilla ANIL on the multi pool was not
    rerun. Therefore vanilla ANIL under the multi setting uses stage3
    `anil__mae__multi` (same meta_train configuration as PI-ANIL: MAE
    initialization, seed=0, K=10/Q=16, 1250 episodes, same episode sequence);
    this is also the standing practice in P1b. Their best_val values
    1.3431 (ANIL) vs 1.1413 (PI-ANIL) are comparable.
    """
    out = []
    s4 = pd.concat([pd.read_csv(p) for p in S4P.glob("*.csv")], ignore_index=True)
    pi = s4[s4.tag == "anilpi__both__mae__multi"][["task", "shot", "rep", "r2"]].copy()
    pi["model"] = "PI-ANIL"
    out.append(pi)
    s3 = pd.concat([pd.read_csv(p) for p in S3P.glob("anil__mae__multi__*.csv")],
                   ignore_index=True)
    anil = s3[["task", "shot", "rep", "r2"]].copy()
    anil["model"] = "ANIL"
    out.append(anil)
    return pd.concat(out, ignore_index=True)


def load_traditional():
    out = []
    p1b = pd.concat([pd.read_csv(p) for p in P1B.glob("*.csv")
                     if not p.name.startswith("log_")], ignore_index=True)
    for m in ("PLS", "UVEPLS", "CARSPLS"):
        s = p1b[p1b.model == m][["task", "shot", "rep", "r2"]].copy()
        s["model"] = m
        out.append(s)
    for route, m in (("knn", "kNN"), ("gp", "GP")):
        s = pd.concat([pd.read_csv(p) for p in S3P.glob(f"{route}__*.csv")],
                      ignore_index=True)
        s = s[["task", "shot", "rep", "r2"]].copy()
        s["model"] = m
        out.append(s)
    return pd.concat(out, ignore_index=True)


def paired_test(paired, K, a, b, tasks):
    da = paired[(paired.model == a) & (paired.shot == K) & (paired.task.isin(tasks))]
    db = paired[(paired.model == b) & (paired.shot == K) & (paired.task.isin(tasks))]
    j = da.merge(db, on=["task", "rep"], suffixes=("_a", "_b")).dropna(subset=["r2_a", "r2_b"])
    if len(j) < 8:
        return None
    diff = j.r2_a - j.r2_b
    p_w = float(wilcoxon(diff).pvalue) if (diff != 0).any() else 1.0
    tag = j.groupby("task").agg(da=("r2_a", "median"), db=("r2_b", "median"))
    tdiff = tag["da"] - tag["db"]
    npos = int((tdiff > 0).sum())
    ntask = len(tdiff)
    p_sign = float(binomtest(npos, ntask, 0.5).pvalue) if ntask else 1.0
    return dict(comparison=f"{NAME.get(a,a)} vs {NAME.get(b,b)}", n_rep=len(j),
                median_diff=f"{diff.median():+.3f}", Wilcoxon_p=f"{p_w:.2e}",
                task_level=f"{npos}/{ntask} positive", sign_p=f"{p_sign:.3f}")


def main():
    proto = load_protonet()
    meta = load_meta()
    trad = load_traditional()
    all_df = pd.concat([proto, meta, trad], ignore_index=True)
    all_df.to_csv(OUT / "paired_per_rep.csv", index=False)

    tasks_avail = sorted(all_df.task.unique())
    pro_tasks = sorted(proto.task.unique()) if len(proto) else []
    pi_tasks = sorted(meta[meta.model == "PI-ANIL"].task.unique())
    common = [t for t in COMMON9 if t in pro_tasks and t in pi_tasks]
    print(f"[info] ProtoNet tasks: {pro_tasks}")
    print(f"[info] PI-ANIL multi tasks: {pi_tasks}")
    print(f"[info] paired common tasks: {common}")

    models = ["protonet_meta", "protonet_frozen", "PI-ANIL", "ANIL",
              "PLS", "CARSPLS", "UVEPLS", "kNN", "GP"]

    L = ["# P1c: ProtoNet regression (frozen / meta-trained) vs PI-ANIL/ANIL and traditional baselines\n",
         "Closing the loop on Garzón 2025 (*Eng. Appl. Artif. Intell.*) and its claim that "
         "the \"ProtoNet family is suitable for few-shot spectroscopic quantification\" — "
         "ANIL, the method advocated in this project, belongs to the same frozen-encoder "
         "family as ProtoNet, and P1c is a direct head-to-head: ProtoNet regression (a "
         "Garzón-isomorphic method) vs PI-ANIL/ANIL.\n",
         "\n**Two ProtoNet variants** (the only difference = encoder source; same LOO-tau "
         "kernel-regression readout):",
         "- `protonet_frozen`: frozen MAE encoder = the MAE counterpart of Garzón's \"SSL+PR\";",
         "- `protonet_meta`: meta-trained ProtoNet encoder (the proto branch of metatrain.py; "
         "same MAE initialization / same episodes / same budget as ANIL/PI-ANIL, only the "
         "objective changed to query kernel-regression MSE) = the regression counterpart of "
         "Garzón's \"meta-trained Prototypical Networks\".\n",
         "Protocol: 30 reps for ProtoNet (frozen-readout convention, same as kNN/GP/PLS); "
         "10 reps for PI-ANIL/ANIL. Bit-identical splits; merge on (task,rep) automatically "
         "takes the common reps. Two-level tests: rep-level Wilcoxon + task-level sign test.\n"]

    # Table 1: per-task R² median
    L.append("\n## 1. Per-task R² median\n")
    for K in (5, 10, 20):
        L.append(f"\n### K={K}-shot\n")
        rows = []
        for task in COMMON9:
            if task not in tasks_avail:
                continue
            row = dict(task=task)
            for m in models:
                sub = all_df[(all_df.task == task) & (all_df.model == m) & (all_df.shot == K)]
                row[NAME[m]] = f"{sub.r2.median():.3f}" if len(sub) else "—"
            rows.append(row)
        L.append(_md(pd.DataFrame(rows)))

    # Table 2: headline 8 summary (mean + failure rate)
    L.append("\n\n## 2. Headline 8-task summary (R² mean / failure rate = fraction with R²<-1)\n")
    rows = []
    for m in models:
        sub = all_df[(all_df.model == m) & (all_df.task.isin(HEADLINE))]
        if not len(sub):
            continue
        row = dict(method=NAME[m])
        for K in (5, 10, 20):
            g = sub[sub.shot == K].r2
            row[f"K={K} R²"] = f"{g.mean():.3f}" if len(g) else "—"
            row[f"K={K} failure"] = f"{(g < -1).mean() * 100:.0f}%" if len(g) else "—"
        rows.append(row)
    L.append(_md(pd.DataFrame(rows)))

    # Table 3: paired two-level tests
    L.append("\n\n## 3. Paired two-level tests\n")
    L.append("(a) **ProtoNet vs the meta-learning family** (rep 0-9, paired with PI-ANIL/ANIL) — the P1c core.\n")
    pairs_meta = [("protonet_meta", "PI-ANIL"), ("protonet_meta", "ANIL"),
                  ("protonet_frozen", "PI-ANIL"), ("protonet_frozen", "ANIL"),
                  ("protonet_meta", "protonet_frozen")]
    rows = []
    for K in (5, 10, 20):
        for a, b in pairs_meta:
            r = paired_test(all_df, K, a, b, common)
            if r:
                rows.append(dict(shot=K, **r))
    L.append(_md(pd.DataFrame(rows)))
    L.append("\n(Positive sign = the former is better by the median. ProtoNet(meta) vs "
             "ProtoNet(frozen) measures the gain of meta-training on the ProtoNet encoder; "
             "both use 30 reps.)\n")

    L.append("\n(b) **ProtoNet (meta-trained) vs traditional baselines** (rep 0-29, paired with PLS/CARS/UVE/kNN/GP) — context.\n")
    pairs_trad = [("protonet_meta", "PLS"), ("protonet_meta", "CARSPLS"),
                  ("protonet_meta", "UVEPLS"), ("protonet_meta", "kNN"),
                  ("protonet_meta", "GP")]
    rows = []
    for K in (5, 10, 20):
        for a, b in pairs_trad:
            r = paired_test(all_df, K, a, b, common)
            if r:
                rows.append(dict(shot=K, **r))
    L.append(_md(pd.DataFrame(rows)))
    L.append("\n(No multiple-comparison correction.)\n")

    (OUT / "table.md").write_text("\n".join(L), encoding="utf-8")
    print("merged ->", OUT / "table.md")


if __name__ == "__main__":
    main()
