"""Baseline wrap-up: merge parts and generate the summary table and figures."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(sys.executable).parent.parent.parent))

OUT = ROOT / "results" / "baselines"
PARTS = OUT / "parts"

def main():
    fs = pd.concat([pd.read_csv(p) for p in PARTS.glob("fewshot__*.csv")], ignore_index=True)
    fc = pd.concat([pd.read_csv(p) for p in PARTS.glob("fullcv__*.csv")], ignore_index=True)
    fs.to_csv(OUT / "baseline_fewshot.csv", index=False)
    fc.to_csv(OUT / "baseline_fullcv.csv", index=False)

    # Markdown summary
    with open(OUT / "table.md", "w", encoding="utf-8") as f:
        f.write("# Baseline Performance Summary\n\n")
        f.write("Evaluation protocol: 5/10/20-shot random support sets × 30 repeats (10 for CNN-family), SNV preprocessing, metrics R²/RMSE/RPD.\n\n")
        f.write("## Full 10-fold cross-validation (literature reference)\n\n")
        f.write(fc.round(4).to_markdown(index=False))
        f.write("\n\n## Few-shot protocol (mean±std R² / mean RPD)\n\n")
        view = fs.assign(val=fs.apply(lambda r: f"{r.r2_mean:.3f}±{r.r2_std:.3f} / {r.rpd_mean:.2f}", axis=1))
        piv = view.pivot_table(index=["task", "model"], columns="shot", values="val", aggfunc="first").reset_index()
        f.write(piv.to_markdown(index=False))
        f.write("\n")

    # ---- Figures ----
    # Figures are optional and require the raw datasets; skip gracefully if absent.
    try:
        try:
            from daimon_runtime import setup_plot
            setup_plot()
        except Exception:
            pass
        import matplotlib.pyplot as plt
        sys.path.insert(0, str(ROOT))
        from src.datasets import load_diesel, load_gasoline, load_corn, load_evoo

        fig, axes = plt.subplots(2, 2, figsize=(11, 7))
        panels = []
        d = load_diesel(); panels.append(("Diesel NIR (SWRI, n=784)", d["X"], d["wavelengths"]))
        g = load_gasoline(); panels.append(("Gasoline NIR (Kalivas, n=60)", g["X"], g["wavelengths"]))
        c = load_corn(); Xc, axc = c["instruments"]["m5"]; panels.append(("Corn NIR (Cargill-m5, n=80)", Xc, axc))
        e = load_evoo(); panels.append(("EVOO adulteration NIR-HSI (n=1995)", e["X"], e["wavelengths"]))
        for ax, (title, X, wl) in zip(axes.ravel(), panels):
            idx = np.linspace(0, len(X) - 1, min(25, len(X))).astype(int)
            for i in idx:
                ax.plot(wl, X[i], lw=0.6, alpha=0.6)
            ax.set_title(title, fontsize=11)
            ax.set_xlabel("Wavelength (nm)")
            ax.set_ylabel("Absorbance / reflectance")
        fig.suptitle("Dataset spectra overview", fontsize=13)
        fig.tight_layout()
        fig.savefig(OUT / "spectra_overview.png", bbox_inches="tight", dpi=150)
        plt.close(fig)

        # Few-shot learning curves
        tasks_order = ["diesel-CN", "diesel-BP50", "diesel-D4052", "diesel-TOTAL",
                       "diesel-VISC", "diesel-FREEZE", "gasoline-octane", "corn_m5-oil",
                       "corn_m5-protein", "corn_m5-moisture", "corn_m5-starch", "evoo-adulteration"]
        tasks_order = [t for t in tasks_order if t in set(fs.task)]
        n = len(tasks_order)
        ncol = 4
        nrow = int(np.ceil(n / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(13, 3.2 * nrow), sharex=True)
        axes = np.atleast_1d(axes).ravel()
        for ax, t in zip(axes, tasks_order):
            sub = fs[fs.task == t]
            for m, mk in [("PLS", "o-"), ("SVR", "s-"), ("CNN", "^-"), ("CNN+Aug", "d-")]:
                ss = sub[sub.model == m].sort_values("shot")
                if len(ss):
                    ax.errorbar(ss.shot, ss.r2_mean, yerr=ss.r2_std, fmt=mk, ms=4, lw=1.2,
                                capsize=2, label=m)
            ax.axhline(0, color="gray", lw=0.5, ls="--")
            ax.set_title(t, fontsize=10)
            ax.set_xticks([5, 10, 20])
            ax.set_xlabel("Support-set size K")
            ax.set_ylabel("R²")
        for ax in axes[n:]:
            ax.axis("off")
        axes[0].legend(fontsize=8, loc="lower right")
        fig.suptitle("Few-shot learning curves: R² vs support-set size (mean±std)", fontsize=13)
        fig.tight_layout()
        fig.savefig(OUT / "fewshot_curves.png", bbox_inches="tight", dpi=150)
        plt.close(fig)
    except Exception as e:
        print("tables merged; figures skipped (optional, need datasets):", e)
    print("merged tables + figures ->", OUT)

if __name__ == "__main__":
    main()
