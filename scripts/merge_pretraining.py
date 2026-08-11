"""Pretraining wrap-up: merge linear-probe / fine-tuning results, and build the comparison table and figure against the baselines."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(sys.executable).parent.parent.parent))

OUT = ROOT / "results" / "pretraining"
PARTS = OUT / "parts"
STAGE0 = ROOT / "results" / "baselines" / "baseline_fewshot.csv"

MODEL_LABEL = dict(mae_ridge="MAE+ridge (frozen)", simclr_ridge="SimCLR+ridge (frozen)",
                   rand_ridge="random init+ridge", mae_ft="MAE+fine-tune",
                   PLS="PLS", SVR="SVR", CNN="CNN (from scratch)", **{"CNN+Aug": "CNN (from scratch)+aug"})
TASK_ORDER = ["diesel-CN", "diesel-BP50", "diesel-D4052", "diesel-FLASH",
              "diesel-FREEZE", "diesel-TOTAL", "diesel-VISC",
              "gasoline-octane", "corn_m5-protein", "corn_m5-oil",
              "evoo-adulteration"]


def main():
    df = pd.concat([pd.read_csv(p) for p in PARTS.glob("probe__*.csv")],
                   ignore_index=True)
    df.to_csv(OUT / "probe_results.csv", index=False)
    agg = (df.groupby(["task", "model", "shot"])
             .agg(r2_mean=("r2", "mean"), r2_std=("r2", "std"),
                  r2_median=("r2", "median"), n=("r2", "size"))
             .reset_index())

    # Baseline anchors
    b0 = pd.read_csv(STAGE0)
    b0 = b0[["task", "model", "shot", "r2_mean", "r2_std"]]
    b0["r2_median"] = np.nan
    b0["n"] = 30
    allr = pd.concat([agg, b0], ignore_index=True)

    with open(OUT / "table.md", "w", encoding="utf-8") as f:
        f.write("# Pretraining-Gain Comparison (few-shot mean±std R²)\n\n")
        f.write("Support/test splits exactly identical to the baseline protocol (paired). New here: "
                "frozen encoder + ridge regression (30 repeats), MAE-initialized end-to-end fine-tuning (10 repeats).\n\n")
        for t in TASK_ORDER:
            sub = allr[allr.task == t]
            if not len(sub):
                continue
            view = sub.assign(val=sub.apply(
                lambda r: f"{r.r2_mean:.3f}±{r.r2_std:.3f}", axis=1))
            view["model"] = view.model.map(lambda m: MODEL_LABEL.get(m, m))
            piv = view.pivot_table(index="model", columns="shot", values="val",
                                   aggfunc="first").reset_index()
            f.write(f"## {t}\n\n{piv.to_markdown(index=False)}\n\n")
        f.write("Note: rand_ridge is the random-init encoder ablation, included to show the gain comes from pretraining rather than the architecture.\n")

    # ---- Figure: 4 representative tasks, key models ----
    try:
        from daimon_runtime import setup_plot
        setup_plot()
    except Exception:
        pass
    import matplotlib.pyplot as plt
    show_tasks = ["gasoline-octane", "diesel-CN", "diesel-TOTAL", "evoo-adulteration"]
    show_models = [("mae_ridge", "o-"), ("simclr_ridge", "v-"), ("mae_ft", "d-"),
                   ("rand_ridge", "x:"), ("PLS", "s-"), ("CNN", "^-"),
                   ("CNN+Aug", "P-")]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4), sharex=True)
    for ax, t in zip(axes, show_tasks):
        sub = allr[allr.task == t]
        for m, mk in show_models:
            ss = sub[sub.model == m].sort_values("shot")
            if len(ss):
                ax.errorbar(ss.shot, ss.r2_mean, yerr=ss.r2_std, fmt=mk, ms=4,
                            lw=1.2, capsize=2, label=MODEL_LABEL[m])
        ax.axhline(0, color="gray", lw=0.5, ls="--")
        ax.set_title(t, fontsize=11)
        ax.set_xticks([5, 10, 20])
        ax.set_xlabel("Support-set size K")
    axes[0].set_ylabel("R²")
    axes[-1].legend(fontsize=7, loc="lower right")
    fig.suptitle("Self-supervised pretraining gain: few-shot R² comparison (mean±std, paired splits)", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "pretraining_gain.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("merged ->", OUT)


if __name__ == "__main__":
    main()
