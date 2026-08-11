"""TRIP exact-backbone head-to-head table merge (G-P1b-exact, 512-dim Garzón backbone) + comparison against the old G-P1b (128-dim).

Reads trip_exact_parts/ (__exact tag) + trip_parts/ (legacy 128-dim, for comparison).
Outputs:
  A. Full-configuration mean table (method × init × shot, per-task means averaged)
  B. Soft alignment with Garzón Table 4 (order-of-magnitude check, not bit-match)
  C. PI-ANIL(mae) init-controlled paired tests + vs ProtoNet(rand) headline
  D. MAE ablation (rand vs mae for the same method)
  E. Comparison against the old G-P1b (128-dim): how the numbers change after backbone exactification, whether conclusions flip or strengthen

Honesty statement: architecture/hyperparameters are bit-match grade (512-dim ResNet1D-18 + Garzón Table B.2), but absolute values
still are not a Garzón bit-match due to protocol differences (reps 10 vs 3, episode semantics, seeds); the core is internal relative comparison
+ order-of-magnitude alignment (which should be tighter than the old 128-dim run).
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
EXACT_PARTS = OUT / "trip_exact_parts"
OLD_PARTS = OUT / "trip_parts"
SUF = "__exact"

# Garzón Table 4 TRIP R² (values reported in the paper, shots 5/10/25)
GARZON = {
    "Individual": (-0.310, 0.122, 0.179),
    "Fine-tuning": (-2.200, -2.190, -2.975),
    "MAML": (-0.095, 0.249, 0.344),
    "Prototypical": (-0.115, 0.251, 0.429),
}
SHOTS = (5, 10, 25)


def tag_of(method, init, variant="both", suffix=SUF):
    base = f"trip__pi__{variant}__{init}" if method == "pi" else f"trip__{method}__{init}"
    return base + suffix


def mean_r2(big, tag, shot):
    sub = big[(big.tag == tag) & (big.shot == shot)]
    if len(sub) == 0:
        return np.nan
    return sub.groupby("task").r2.mean().mean()


def paired(big, tag_a, tag_b, shot):
    a = big[big.tag == tag_a].set_index(["task", "rep"]).query("shot==@shot").r2
    b = big[big.tag == tag_b].set_index(["task", "rep"]).query("shot==@shot").r2
    common = a.index.intersection(b.index)
    if len(common) < 5:
        return (0, np.nan, np.nan, 0, 0)
    d = a.loc[common] - b.loc[common]
    med = float(d.median())
    try:
        p = stats.wilcoxon(d.values).pvalue if np.all(d.values != 0) else 1.0
    except Exception:
        p = np.nan
    pt = d.groupby("task").mean()
    return (len(common), med, p, int((pt > 0).sum()), int((pt < 0).sum()))


def load_parts(parts_dir):
    files = sorted(glob.glob(str(parts_dir / "*.csv")))
    if not files:
        return None
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def main():
    big = load_parts(EXACT_PARTS)
    if big is None:
        print(f"no exact parts files ({EXACT_PARTS})"); return
    print(f"loaded {len(big)} rows of exact parts\n")

    # ---- A. Full-configuration mean table ----
    print("=" * 72)
    print("A. TRIP test7 mean R² (512-dim exact backbone, average of per-task means)")
    print("=" * 72)
    print(f"{'method(init)':24s} {'5-shot':>8s} {'10-shot':>8s} {'25-shot':>8s}")
    rows = []
    for method in ["base", "ft", "protonet", "maml", "anil", "pi"]:
        mkey = "proto" if method == "protonet" else method
        for init in ["rand", "mae"]:
            tag = tag_of(mkey, init)
            name = {"base": "Base", "ft": "FT", "protonet": "ProtoNet",
                    "maml": "MAML", "anil": "ANIL", "pi": "PI-ANIL"}[method]
            if tag not in big.tag.values:
                continue
            r2s = {K: mean_r2(big, tag, K) for K in SHOTS}
            rows.append((f"{name}({init})", tag, r2s))
            print(f"{name+'('+init+')':24s} " + " ".join(f"{r2s[K]:+8.3f}" for K in SHOTS))

    # ---- B. Soft alignment with Garzón ----
    print("\n" + "=" * 72)
    print("B. Soft alignment with Garzón Table 4 (order-of-magnitude check, not bit-match)")
    print("=" * 72)
    print(f"{'comparison':26s} {'shot':>5s} {'ours':>8s} {'Garzón':>8s}")
    align = [("Base", "base", "rand", "Individual"),
             ("ProtoNet", "proto", "rand", "Prototypical"),
             ("ProtoNet", "proto", "mae", "Prototypical"),
             ("MAML", "maml", "mae", "MAML"),
             ("MAML", "maml", "rand", "MAML")]
    for disp, m, init, g in align:
        tag = tag_of(m, init)
        if tag not in big.tag.values:
            continue
        for i, K in enumerate(SHOTS):
            print(f"{disp+'('+init+')':26s} {K:>5d} {mean_r2(big,tag,K):+8.3f} {GARZON[g][i]:+8.3f}")

    # ---- C. init-controlled paired tests ----
    print("\n" + "=" * 72)
    print("C. PI-ANIL(mae) vs each method, paired tests (rep-level Wilcoxon + task-level sign)")
    print("=" * 72)
    pi_mae = tag_of("pi", "mae")
    if pi_mae not in big.tag.values:
        print("no PI-ANIL(mae)")
    else:
        print(f"{'vs':24s} {'shot':>5s} {'n':>4s} {'med_Δ':>8s} {'Wilcoxon_p':>11s} {'tasks':>7s}")
        for nm, tb in [("ANIL(mae)", tag_of("anil", "mae")),
                       ("ProtoNet(mae)", tag_of("proto", "mae")),
                       ("MAML(mae)", tag_of("maml", "mae")),
                       ("FT(mae)", tag_of("ft", "mae")),
                       ("ProtoNet(rand)·headline", tag_of("proto", "rand")),
                       ("Base(rand)", tag_of("base", "rand"))]:
            if tb not in big.tag.values:
                continue
            for K in SHOTS:
                n, med, p, np_, nn = paired(big, pi_mae, tb, K)
                if n == 0:
                    continue
                print(f"{nm:24s} {K:>5d} {n:>4d} {med:+8.3f} {p:11.3e} {np_}+/{nn}-")

    # ---- D. MAE ablation ----
    print("\n" + "=" * 72)
    print("D. MAE initialization ablation (mae vs rand for the same method, Δ=mae-rand)")
    print("=" * 72)
    print(f"{'method':16s} {'shot':>5s} {'rand':>8s} {'mae':>8s} {'Δ':>8s} {'p':>10s}")
    for method, mtag in [("ANIL", "anil"), ("ProtoNet", "proto"), ("PI-ANIL", "pi"), ("MAML", "maml")]:
        tr, tm = tag_of(mtag, "rand"), tag_of(mtag, "mae")
        if tr not in big.tag.values or tm not in big.tag.values:
            continue
        for K in SHOTS:
            n, med, p, np_, nn = paired(big, tm, tr, K)
            print(f"{method:16s} {K:>5d} {mean_r2(big,tr,K):+8.3f} {mean_r2(big,tm,K):+8.3f} "
                  f"{med:+8.3f} {p:10.3e}")

    # ---- E. Comparison against the old G-P1b (128-dim) ----
    old = load_parts(OLD_PARTS)
    print("\n" + "=" * 72)
    print("E. Comparison vs the old G-P1b (128-dim resnet1): change after backbone exactification (Δ=exact−old)")
    print("=" * 72)
    if old is None:
        print("no old trip_parts, skipping comparison")
    else:
        print(f"{'method(init)':24s} {'shot':>5s} {'128-dim':>8s} {'512-dim':>8s} {'Δ':>8s}")
        for method in ["base", "ft", "protonet", "maml", "anil", "pi"]:
            mkey = "proto" if method == "protonet" else method
            name = {"base": "Base", "ft": "FT", "protonet": "ProtoNet",
                    "maml": "MAML", "anil": "ANIL", "pi": "PI-ANIL"}[method]
            for init in ["rand", "mae"]:
                te, to = tag_of(mkey, init), tag_of(mkey, init, suffix="")
                if to not in old.tag.values or te not in big.tag.values:
                    continue
                for K in SHOTS:
                    ro, re = mean_r2(old, to, K), mean_r2(big, te, K)
                    if np.isnan(ro) or np.isnan(re):
                        continue
                    print(f"{name+'('+init+')':24s} {K:>5d} {ro:+8.3f} {re:+8.3f} {re-ro:+8.3f}")

    # ---- Write to disk ----
    pd.DataFrame([(n, t, K, r[K]) for n, t, r in rows for K in SHOTS],
                 columns=["method", "tag", "shot", "r2_mean"]).to_csv(
        OUT / "trip_exact_summary.csv", index=False)
    big.to_csv(OUT / "trip_exact_all.csv", index=False)
    print(f"\nsaved -> trip_exact_summary.csv / _all.csv")


if __name__ == "__main__":
    main()
