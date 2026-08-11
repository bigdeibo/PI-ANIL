"""Physics wrap-up: variant ablation table + LOMO extrapolation table + paired tests."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(sys.executable).parent.parent.parent))

OUT = ROOT / "results" / "physics"
HEADLINE = ["diesel-CN", "diesel-BP50", "diesel-D4052", "diesel-FLASH",
            "diesel-FREEZE", "diesel-TOTAL", "diesel-VISC", "gasoline-octane"]
DIESEL = HEADLINE[:7]

VARIANTS = [("vanilla (route-comparison ANIL)", "anil__mae__multi"),
            ("PI: +additivity constraint", "anilpi__add__mae__multi"),
            ("PI: +band prior", "anilpi__band__mae__multi"),
            ("PI: both constraints", "anilpi__both__mae__multi")]


def load_perrep():
    s4 = pd.concat([pd.read_csv(p) for p in (OUT / "parts").glob("*.csv")],
                   ignore_index=True)[["task", "tag", "shot", "rep", "r2"]]
    s3 = pd.read_csv(ROOT / "results" / "meta_routes" / "per_rep.csv")
    s3 = s3[s3.mid == "anil__mae__multi"][["task", "mid", "shot", "rep", "r2"]]
    s3.columns = ["task", "tag", "shot", "rep", "r2"]
    return pd.concat([s4, s3], ignore_index=True)


def ms(df, task, shot):
    sub = df[(df.task == task) & (df.shot == shot)]
    if not len(sub):
        return "—", np.nan
    return f"{sub.r2.mean():.3f}±{sub.r2.std():.3f}", sub.r2.mean()


def main():
    df = load_perrep()
    df.to_csv(OUT / "per_rep.csv", index=False)
    lines = ["# Physics-informed meta-learning: variant ablation and cross-material extrapolation\n",
             "R² mean±std, paired splits (rep 0-9), all MAE-initialized.\n",
             "\n## 1. Variant ablation (multi task pool, headline 8 tasks + EVOO)\n"]

    tasks9 = HEADLINE + ["evoo-adulteration"]
    for task in tasks9:
        rows = []
        for name, tag in VARIANTS:
            row = dict(variant=name)
            for K in (5, 10, 20):
                row[str(K)] = ms(df[df.tag == tag], task, K)[0]
            rows.append(row)
        dfm = pd.DataFrame(rows)
        for K in ("5", "10", "20"):
            means = [float(v.split("±")[0]) if v != "—" else -np.inf
                     for v in dfm[K]]
            bi = int(np.argmax(means))
            dfm.loc[bi, K] = f"**{dfm.loc[bi, K]}**"
        lines.append(f"\n### {task}\n\n{dfm.to_markdown(index=False)}")

    # headline summary means
    lines.append("\n\n### Headline 8-task means (stability metric: fraction of episodes with R²<-1 per shot level)\n")
    rows = []
    for name, tag in VARIANTS:
        sub = df[(df.tag == tag) & (df.task.isin(HEADLINE))]
        row = dict(variant=name)
        for K in (5, 10, 20):
            g = sub[sub.shot == K].r2
            row[f"K={K} mean"] = f"{g.mean():.3f}"
            row[f"K={K} failure rate"] = f"{(g < -1).mean() * 100:.0f}%"
        rows.append(row)
    lines.append(pd.DataFrame(rows).to_markdown(index=False))

    # LOMO
    lines.append("\n\n## 2. Cross-material extrapolation (LOMO: target material removed from the meta-training pool)\n")
    lines.append("\n### Test = gasoline (pool: diesel + corn + EVOO, 12 tasks)\n")
    rows = []
    for name, tag in [("vanilla ANIL", "anilpi__vanilla__mae__no_gasoline"),
                      ("PI both constraints", "anilpi__both__mae__no_gasoline"),
                      ("reference: route-comparison multi pool (incl. gasoline, seen task)", "anil__mae__multi")]:
        row = dict(model=name)
        for K in (5, 10, 20):
            row[str(K)] = ms(df[df.tag == tag], "gasoline-octane", K)[0]
        rows.append(row)
    lines.append(pd.DataFrame(rows).to_markdown(index=False))
    lines.append("\n### Test = 7 diesel properties (pool: gasoline + corn + EVOO, 6 tasks)\n")
    for task in DIESEL:
        rows = []
        for name, tag in [("vanilla ANIL", "anilpi__vanilla__mae__no_diesel"),
                          ("PI both constraints", "anilpi__both__mae__no_diesel"),
                          ("reference: route-comparison multi pool (incl. diesel, seen task)", "anil__mae__multi")]:
            row = dict(model=name)
            for K in (5, 10, 20):
                row[str(K)] = ms(df[df.tag == tag], task, K)[0]
            rows.append(row)
        lines.append(f"\n#### {task}\n\n{pd.DataFrame(rows).to_markdown(index=False)}")

    # Wilcoxon
    from scipy.stats import wilcoxon
    lines.append("\n\n## 3. Paired Wilcoxon signed-rank tests\n")
    pairs = [
        ("PI both vs vanilla (headline 8 tasks)", "anilpi__both__mae__multi",
         "anil__mae__multi", HEADLINE),
        ("PI additivity vs vanilla (headline 8 tasks)", "anilpi__add__mae__multi",
         "anil__mae__multi", HEADLINE),
        ("PI band vs vanilla (headline 8 tasks)", "anilpi__band__mae__multi",
         "anil__mae__multi", HEADLINE),
        ("PI both vs vanilla (headline, 5-shot only)", "anilpi__both__mae__multi",
         "anil__mae__multi", HEADLINE, 5),
        ("LOMO gasoline: PI vs vanilla", "anilpi__both__mae__no_gasoline",
         "anilpi__vanilla__mae__no_gasoline", ["gasoline-octane"]),
        ("LOMO diesel: PI vs vanilla", "anilpi__both__mae__no_diesel",
         "anilpi__vanilla__mae__no_diesel", DIESEL),
    ]
    rows = []
    for item in pairs:
        name, t1, t2, tasks = item[0], item[1], item[2], item[3]
        shot_filter = item[4] if len(item) > 4 else None
        d1 = df[(df.tag == t1) & (df.task.isin(tasks))]
        d2 = df[(df.tag == t2) & (df.task.isin(tasks))]
        if shot_filter:
            d1, d2 = d1[d1.shot == shot_filter], d2[d2.shot == shot_filter]
        j = d1.merge(d2, on=["task", "shot", "rep"], suffixes=("_1", "_2"))
        j = j.dropna(subset=["r2_1", "r2_2"])
        if len(j) < 8:
            continue
        diff = j.r2_1 - j.r2_2
        p = float(wilcoxon(diff).pvalue) if (diff != 0).any() else 1.0
        rows.append(dict(comparison=name, n=len(j), median_diff=f"{diff.median():+.3f}",
                         mean_diff=f"{diff.mean():+.3f}", p_value=f"{p:.2e}",
                         significant="yes" if p < 0.05 else "no"))
    lines.append(pd.DataFrame(rows).to_markdown(index=False))
    lines.append("\n(No multiple-comparison correction; the route-comparison vanilla ANIL results and the PI "
                 "variants share the same MAE initialization and evaluation protocol.)\n")

    (OUT / "table.md").write_text("\n".join(lines), encoding="utf-8")
    print("merged ->", OUT / "table.md")


if __name__ == "__main__":
    main()
