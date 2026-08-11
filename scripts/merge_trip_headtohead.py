"""TRIP head-to-head table merge + init-controlled paired statistics + Garzón soft alignment.

Outputs:
  A. Full-configuration mean table (method × init × shot, per-task means averaged)
  B. Soft alignment with Garzón Table 4 (order-of-magnitude check, not bit-match)
  C. init-controlled paired tests (PI-ANIL(mae) vs each method at the same init + vs ProtoNet(rand), the headline)
  D. MAE ablation (rand vs mae for the same method)

Honesty statement: in-house unified implementation (same backbone, same protocol), not a bit-match of Garzón;
absolute values differ from Garzón due to protocol differences (50-step evaluation adaptation, full query pool).
The core is internal relative comparison + order-of-magnitude alignment.
"""
import sys
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT = ROOT / "results" / "trip_benchmark"
PARTS = OUT / "trip_parts"

# Garzón Table 4 TRIP R² (values reported in the paper)
GARZON = {  # method -> (5,10,25)
    "Individual": (-0.310, 0.122, 0.179),
    "Fine-tuning": (-2.200, -2.190, -2.975),
    "MAML": (-0.095, 0.249, 0.344),
    "Prototypical": (-0.115, 0.251, 0.429),
}


def tag_of(method, init, variant="both"):
    if method == "pi":
        return f"trip__pi__{variant}__{init}"
    return f"trip__{method}__{init}"


def mean_r2(big, tag, shot):
    sub = big[(big.tag == tag) & (big.shot == shot)]
    if len(sub) == 0:
        return np.nan
    return sub.groupby("task").r2.mean().mean()  # average of per-task means


def per_taskrep(big, tag, shot):
    """Returns {task: [r2 per rep]}."""
    sub = big[(big.tag == tag) & (big.shot == shot)]
    return sub.groupby("task").r2.apply(list).to_dict()


def paired(big, tag_a, tag_b, shot):
    """Paired difference a-b over the same (task,rep); returns (n, median, wilcoxon_p, npos_task, nneg_task)."""
    a = big[big.tag == tag_a].set_index(["task", "rep"]).query("shot==@shot").r2
    b = big[big.tag == tag_b].set_index(["task", "rep"]).query("shot==@shot").r2
    common = a.index.intersection(b.index)
    if len(common) < 5:
        return (0, np.nan, np.nan, 0, 0)
    d = (a.loc[common] - b.loc[common])
    med = float(d.median())
    diffs = d.values
    try:
        p = stats.wilcoxon(diffs).pvalue if np.all(diffs != 0) else 1.0
    except Exception:
        p = np.nan
    per_task = d.groupby("task").mean()
    return (len(common), med, p, int((per_task > 0).sum()), int((per_task < 0).sum()))


def main():
    files = sorted(glob.glob(str(PARTS / "*.csv")))
    if not files:
        print("no parts files"); return
    big = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    print(f"loaded {len(files)} parts, {len(big)} rows\n")

    SHOTS = (5, 10, 25)

    # ---- A. Full-configuration mean table ----
    print("=" * 72)
    print("A. TRIP test7 mean R² (average of per-task means)")
    print("=" * 72)
    print(f"{'method(init)':24s} {'5-shot':>8s} {'10-shot':>8s} {'25-shot':>8s}")
    rows = []
    for method in ["base", "ft", "protonet", "maml", "anil", "pi"]:
        for init in ["rand", "mae"]:
            tag = tag_of(method, init) if method != "protonet" else tag_of("proto", init)
            name = {"base": "Base", "ft": "FT", "protonet": "ProtoNet",
                    "maml": "MAML", "anil": "ANIL", "pi": "PI-ANIL"}[method]
            if tag not in big.tag.values:
                continue
            r2s = {K: mean_r2(big, tag, K) for K in SHOTS}
            rows.append((f"{name}({init})", tag, r2s))
            print(f"{name+'('+init+')':24s} " + " ".join(f"{r2s[K]:+8.3f}" for K in SHOTS))

    # ---- B. Soft alignment with Garzón ----
    print("\n" + "=" * 72)
    print("B. Soft alignment with Garzón Table 4 (TRIP R²; order-of-magnitude check, not bit-match)")
    print("=" * 72)
    print(f"{'comparison':26s} {'shot':>5s} {'ours':>8s} {'Garzón':>8s}")
    align = [("Base", "base", "rand", "Individual"), ("FT", "ft", "rand", "Fine-tuning"),
             ("FT", "ft", "mae", "Fine-tuning"), ("ProtoNet", "proto", "mae", "Prototypical"),
             ("ProtoNet", "proto", "rand", "Prototypical"), ("MAML", "maml", "mae", "MAML")]
    for disp, m, init, g in align:
        tag = tag_of(m, init)
        if tag not in big.tag.values:
            continue
        for i, K in enumerate(SHOTS):
            print(f"{disp+'('+init+')':26s} {K:>5d} {mean_r2(big,tag,K):+8.3f} "
                  f"{GARZON[g][i]:+8.3f}")

    # ---- C. init-controlled paired tests ----
    print("\n" + "=" * 72)
    print("C. PI-ANIL(mae) vs each method, paired tests (rep-level Wilcoxon + task-level sign)")
    print("=" * 72)
    pi_mae = tag_of("pi", "mae")
    if pi_mae not in big.tag.values:
        print("no PI-ANIL(mae)"); return
    print(f"{'vs':24s} {'shot':>5s} {'n':>4s} {'med_Δ':>8s} {'Wilcoxon_p':>11s} {'tasks':>7s}")
    # same-init comparisons
    same_init = [("ANIL(mae)", tag_of("anil", "mae")),
                 ("ProtoNet(mae)", tag_of("proto", "mae")),
                 ("MAML(mae)", tag_of("maml", "mae")),
                 ("FT(mae)", tag_of("ft", "mae")),
                 ("Base(rand)", tag_of("base", "rand")),
                 ("ProtoNet(rand)·headline", tag_of("proto", "rand"))]
    for nm, tb in same_init:
        if tb not in big.tag.values:
            continue
        for K in SHOTS:
            n, med, p, np_, nn = paired(big, pi_mae, tb, K)
            if n == 0:
                continue
            print(f"{nm:24s} {K:>5d} {n:>4d} {med:+8.3f} {p:11.3e} {np_}+/{nn}-")

    # ---- D. MAE ablation (rand vs mae for the same method, paired at 25-shot) ----
    print("\n" + "=" * 72)
    print("D. MAE initialization ablation (mae vs rand for the same method, Δ=mae-rand)")
    print("=" * 72)
    print(f"{'method':16s} {'shot':>5s} {'rand':>8s} {'mae':>8s} {'Δ':>8s} {'p':>10s}")
    for method, mtag in [("ANIL", "anil"), ("ProtoNet", "proto"), ("PI-ANIL", "pi"), ("FT", "ft")]:
        tr, tm = tag_of(mtag, "rand"), tag_of(mtag, "mae")
        if tr not in big.tag.values or tm not in big.tag.values:
            continue
        for K in SHOTS:
            n, med, p, np_, nn = paired(big, tm, tr, K)
            print(f"{method:16s} {K:>5d} {mean_r2(big,tr,K):+8.3f} {mean_r2(big,tm,K):+8.3f} "
                  f"{med:+8.3f} {p:10.3e}")

    # ---- Write to disk ----
    pd.DataFrame([(n, t, K, r[K]) for n, t, r in rows for K in SHOTS],
                 columns=["method", "tag", "shot", "r2_mean"]).to_csv(
        OUT / "trip_headtohead_summary.csv", index=False)
    big.to_csv(OUT / "trip_headtohead_all.csv", index=False)
    print(f"\nsaved -> trip_headtohead_summary.csv / _all.csv")


if __name__ == "__main__":
    main()
