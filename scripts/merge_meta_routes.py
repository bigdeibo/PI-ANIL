"""Meta-route wrap-up: three-route comparison master table + ablation table + paired statistical tests + figure.

Data sources:
- results/meta_routes/parts/ (knn/gp 30 reps; dkl/anil/fomaml 10 reps, per-rep)
- results/pretraining/probe_results.csv (Route A: mae_ridge/simclr_ridge/mae_ft, per-rep)
- results/baselines/baseline_fewshot.csv (PLS/SVR/CNN/CNNAug anchors, mean±std only)
Paired tests: Wilcoxon signed-rank, paired by rep 0-9 (splits are bit-identical across methods).
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(sys.executable).parent.parent.parent))

OUT = ROOT / "results" / "meta_routes"
HEADLINE = ["diesel-CN", "diesel-BP50", "diesel-D4052", "diesel-FLASH",
            "diesel-FREEZE", "diesel-TOTAL", "diesel-VISC", "gasoline-octane"]

MAIN_METHODS = [  # (internal id, display name, source)
    ("PLS", "PLS", "s0"), ("SVR", "SVR", "s0"),
    ("CNN", "scratch CNN", "s0"), ("CNN+Aug", "scratch CNN+Aug", "s0"),
    ("mae_ridge", "A: frozen MAE + ridge", "s2"),
    ("simclr_ridge", "A: frozen SimCLR + ridge", "s2"),
    ("mae_ft", "A: MAE fine-tune", "s2"),
    ("knn", "B: frozen MAE + kNN", "s3"),
    ("gp", "B: frozen MAE + GP", "s3"),
    ("dkl__mae__multi", "B: DKL (meta-trained)", "s3"),
    ("anil__mae__multi", "C: ANIL (meta-trained)", "s3"),
    ("fomaml__mae__multi", "C: FOMAML (meta-trained)", "s3"),
]
ABLATION_METHODS = [
    ("dkl__mae__multi", "DKL MAE + multi-dataset"), ("dkl__rand__multi", "DKL random + multi-dataset"),
    ("dkl__mae__diesel", "DKL MAE + diesel-only"),
    ("anil__mae__multi", "ANIL MAE + multi-dataset"), ("anil__rand__multi", "ANIL random + multi-dataset"),
    ("anil__mae__diesel", "ANIL MAE + diesel-only"),
    ("fomaml__mae__multi", "FOMAML MAE + multi-dataset"), ("fomaml__rand__multi", "FOMAML random + multi-dataset"),
]


def load_all():
    s3 = pd.concat([pd.read_csv(p) for p in (OUT / "parts").glob("*.csv")],
                   ignore_index=True)
    s3["mid"] = np.where(s3.route.isin(["knn", "gp"]), s3.route,
                         s3.route + "__" + s3["init"] + "__" + s3.source)
    s2 = pd.read_csv(ROOT / "results" / "pretraining" / "probe_results.csv")
    s2["mid"] = s2.model
    s0 = pd.read_csv(ROOT / "results" / "baselines" / "baseline_fewshot.csv")
    s0["mid"] = s0.model
    return s3, s2, s0


def cell_mean_std(df, task, shot):
    sub = df[(df.task == task) & (df.shot == shot)]
    if not len(sub):
        return None
    return f"{sub.r2.mean():.3f}±{sub.r2.std():.3f}", sub.r2.mean()


def main():
    s3, s2, s0 = load_all()
    per_rep = pd.concat([s3[["task", "mid", "shot", "rep", "r2"]],
                         s2[["task", "mid", "shot", "rep", "r2"]]], ignore_index=True)
    per_rep.to_csv(OUT / "per_rep.csv", index=False)

    # ---------- Main comparison table (headline 8 tasks) ----------
    lines = ["# Meta-learning route comparison (headline tasks: 7 diesel properties + gasoline)\n",
             "R² mean±std; paired splits; A/B/C are the three routes (frozen/fine-tune, metric-based regression, gradient-based meta-learning).\n"]
    win_count = {}
    for task in HEADLINE:
        lines.append(f"\n## {task}\n")
        rows = []
        for mid, name, src in MAIN_METHODS:
            if src == "s0":
                sub = s0[(s0.mid == mid) & (s0.task == task)]
                if not len(sub):
                    continue
                row = dict(model=name)
                for K in (5, 10, 20):
                    ss = sub[sub.shot == K]
                    row[str(K)] = (f"{ss.r2_mean.iloc[0]:.3f}±{ss.r2_std.iloc[0]:.3f}"
                                   if len(ss) else "—")
                rows.append(row)
            else:
                df = per_rep[per_rep.mid == mid]
                row = dict(model=name)
                got = False
                for K in (5, 10, 20):
                    r = cell_mean_std(df, task, K)
                    row[str(K)] = r[0] if r else "—"
                    got = got or bool(r)
                if got:
                    rows.append(row)
        # bold the best entry in each column
        dfm = pd.DataFrame(rows)
        for K in ("5", "10", "20"):
            vals = [r[K] for r in rows]
            means = [float(v.split("±")[0]) if v != "—" else -np.inf for v in vals]
            bi = int(np.argmax(means))
            dfm.loc[bi, K] = f"**{vals[bi]}**"
            win_count.setdefault(K, {})
            win_count[K][rows[bi]["model"]] = win_count[K].get(rows[bi]["model"], 0) + 1
        lines.append(dfm.to_markdown(index=False))
    lines.append("\n## Headline cell win counts (8 tasks x 3 shot levels, counted by largest mean)\n")
    wc = pd.DataFrame(win_count).fillna(0).astype(int)
    wc["total"] = wc.sum(axis=1)
    lines.append(wc.sort_values("total", ascending=False).to_markdown())

    # ---------- Ablation table ----------
    lines.append("\n\n# Ablation: initialization x task source (headline tasks, R² mean)\n")
    ab_rows = []
    for mid, name in ABLATION_METHODS:
        df = per_rep[per_rep.mid == mid]
        row = dict(config=name)
        for K in (5, 10, 20):
            vals = [cell_mean_std(df, t, K) for t in HEADLINE]
            vals = [v[1] for v in vals if v]
            row[f"K={K} headline mean"] = f"{np.mean(vals):.3f}" if vals else "—"
        ab_rows.append(row)
    lines.append(pd.DataFrame(ab_rows).to_markdown(index=False))
    lines.append("\nNote: evaluating source=diesel-only models on the gasoline task is an unseen-task "
                 "generalization probe (gasoline is absent from their meta-training pool); for "
                 "multi-dataset models all tasks are seen tasks (sample-level protocol).\n")

    # ---------- Paired Wilcoxon ----------
    from scipy.stats import wilcoxon
    lines.append("\n# Paired Wilcoxon signed-rank test (headline tasks, paired by rep 0-9)\n")
    pairs = [("C: ANIL (meta-trained)", "anil__mae__multi", "A: frozen MAE + ridge", "mae_ridge"),
             ("C: ANIL (meta-trained)", "anil__mae__multi", "B: frozen MAE + GP", "gp"),
             ("C: ANIL (meta-trained)", "anil__mae__multi", "B: DKL (meta-trained)", "dkl__mae__multi"),
             ("ANIL MAE init", "anil__mae__multi", "ANIL random init", "anil__rand__multi"),
             ("ANIL multi-dataset", "anil__mae__multi", "ANIL diesel-only", "anil__mae__diesel"),
             ("DKL MAE init", "dkl__mae__multi", "DKL random init", "dkl__rand__multi")]
    st_rows = []
    for n1, m1, n2, m2 in pairs:
        d1 = per_rep[(per_rep.mid == m1) & (per_rep.task.isin(HEADLINE))]
        d2 = per_rep[(per_rep.mid == m2) & (per_rep.task.isin(HEADLINE))]
        j = d1.merge(d2, on=["task", "shot", "rep"], suffixes=("_1", "_2"))
        j = j.dropna(subset=["r2_1", "r2_2"])
        if len(j) < 8:
            continue
        diff = j.r2_1 - j.r2_2
        try:
            p = float(wilcoxon(diff).pvalue)
        except Exception:
            p = np.nan
        st_rows.append(dict(comparison=f"{n1} vs {n2}", n=len(j),
                            median_diff=f"{diff.median():.3f}", mean_diff=f"{diff.mean():.3f}",
                            p_value=f"{p:.2e}",
                            significant="yes (p<0.05)" if p < 0.05 else "no"))
    lines.append(pd.DataFrame(st_rows).to_markdown(index=False))
    lines.append("\n(No multiple-comparison correction; p values are for reference. the baseline "
                 "traditional baselines have no per-rep records and are excluded from paired tests.)\n")

    (OUT / "table.md").write_text("\n".join(lines), encoding="utf-8")

    # ---------- Figure ----------
    try:
        from daimon_runtime import setup_plot
        setup_plot()
    except Exception:
        pass
    import matplotlib.pyplot as plt
    show_tasks = ["gasoline-octane", "diesel-CN", "diesel-TOTAL", "evoo-adulteration"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), sharex=True)
    style = [("mae_ridge", "o-", "A: frozen+ridge"), ("mae_ft", "s-", "A: fine-tune"),
             ("gp", "v-", "B: frozen+GP"), ("dkl__mae__multi", "d-", "B: DKL"),
             ("anil__mae__multi", "p-", "C: ANIL"), ("fomaml__mae__multi", "x:", "C: FOMAML")]
    s0style = [("PLS", "^-", "PLS"), ("CNN", "<-", "scratch CNN")]
    for ax, t in zip(axes, show_tasks):
        for mid, mk, name in style:
            df = per_rep[(per_rep.mid == mid) & (per_rep.task == t)]
            if not len(df):
                continue
            g = df.groupby("shot").r2.agg(["mean", "std"]).reset_index()
            ax.errorbar(g.shot, g["mean"], yerr=g["std"], fmt=mk, ms=4, lw=1.2,
                        capsize=2, label=name)
        for mid, mk, name in s0style:
            ss = s0[(s0.mid == mid) & (s0.task == t)].sort_values("shot")
            if len(ss):
                ax.errorbar(ss.shot, ss.r2_mean, yerr=ss.r2_std, fmt=mk, ms=4,
                            lw=1.2, capsize=2, label=name)
        ax.axhline(0, color="gray", lw=0.5, ls="--")
        ax.set_title(t, fontsize=11)
        ax.set_xticks([5, 10, 20])
        ax.set_xlabel("support-set size K")
        ax.set_ylim(-1.2, 1.05)
    axes[0].set_ylabel("R²")
    axes[-1].legend(fontsize=7, loc="lower right")
    fig.suptitle("Three-route few-shot quantitative comparison (mean±std, paired splits)", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "routes.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("merged ->", OUT / "table.md")


if __name__ == "__main__":
    main()
