"""T1 corn cross-instrument Option B (leak-proof held-out retrain, the main protocol).

Differences from Option A (stricter):
  - 30 corn physical samples are held out (corn_holdout_split, seed=42) and never
    enter meta-training in any form.
  - Meta-training corn tasks use only the remaining 50 samples (source instrument
    m5) -> checkpoint tag __xinstB.
  - Evaluation runs only on the 30 held-out samples (target instruments mp5/mp6,
    plus the m5 same-instrument holdout as a sanity check).
  -> Test samples are entirely unseen during meta-training (not even their m5
     version appeared) = strictly leak-proof.
  -> Because train(50)/holdout(30) are disjoint sample sets, zero-shot PLS
     transfer (m5-train50 -> target-holdout) is valid for every target instrument
     including m5 (Option A leaks when source==target; Option B does not).

Methods (rep 0..29, K=5/10/20, 4 attributes):
  - ANIL(mae)    = anilpi__vanilla__xinstB__mae__multi (corn-holdout retrain)
  - PI-ANIL(mae) = anilpi__both__xinstB__mae__multi    (corn-holdout retrain)
  - PLS-target   = PLS trained K-shot on the target-instrument holdout (per-instrument upper bound)
  - PLS-m5->tgt  = PLS trained on source m5 train-50 -> predicts target holdout (zero-shot transfer lower bound)
Evaluation-time adaptation is bit-identical to the main adaptation protocol (50 head-only steps +
spectral-radius adaptive step size + +/-5 clamp).
Outputs: results/probes/corn_instruments_optionB/{per_rep.csv, summary.txt}.
"""
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from t1_corn_xinst_probe import (corn_inst_Xy, build_enc, eval_episode_anil,
                                 CKPT4)
from src.fsl_regression.core import (corn_holdout_split, eval_split, r2_score)
from src.baselines import PLSBaseline

OUT = ROOT / "results" / "probes" / "corn_instruments_optionB"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--attr", default="moisture,oil,protein,starch")
    ap.add_argument("--shots", default="5,10,20")
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--targets", default="m5,mp5,mp6")
    ap.add_argument("--n-holdout", type=int, default=30)
    ap.add_argument("--device", default="cpu", help="cpu / cuda")
    args = ap.parse_args()
    attrs = [a.strip() for a in args.attr.split(",")]
    shots = [int(s) for s in args.shots.split(",")]
    targets = args.targets.split(",")

    ck_anil = torch.load(CKPT4 / "anilpi__vanilla__xinstB__mae__multi.pt",
                         map_location="cpu", weights_only=False)
    ck_pi = torch.load(CKPT4 / "anilpi__both__xinstB__mae__multi.pt",
                       map_location="cpu", weights_only=False)
    enc_anil = build_enc(ck_anil, device=args.device)
    enc_pi = build_enc(ck_pi, device=args.device)
    print(f"[ckpt] ANIL={ck_anil.get('variant')}/ho{ck_anil.get('corn_holdout')}/"
          f"{ck_anil.get('epoch')}ep  PI={ck_pi.get('variant')}/ho{ck_pi.get('corn_holdout')}/"
          f"{ck_pi.get('epoch')}ep", flush=True)

    tr_idx, ho_idx = corn_holdout_split(80, args.n_holdout)
    print(f"[holdout] train {len(tr_idx)} samples / held out {len(ho_idx)} samples (excluded from meta-training)",
          flush=True)

    rows = []
    for attr in attrs:
        X_src_tr, y_src_tr = corn_inst_Xy("m5", attr, idx=tr_idx)
        pls_src = PLSBaseline().fit(X_src_tr, y_src_tr)
        ho = {t: corn_inst_Xy(t, attr, idx=ho_idx) for t in targets}
        for tgt in targets:
            X_t, y_t = ho[tgt]
            n = len(y_t)
            for K in shots:
                if n <= K + 2:
                    continue
                for rep in range(args.reps):
                    tr, te = eval_split(n, K, rep)
                    yt = y_t[te]
                    rec = dict(target=tgt, attr=attr, shot=K, rep=rep)
                    for name, enc, hs in (("ANIL(mae)", enc_anil, ck_anil["head"]),
                                          ("PI-ANIL(mae)", enc_pi, ck_pi["head"])):
                        try:
                            p = eval_episode_anil(enc, hs, X_t, y_t, tr, te, device=args.device)
                            rec[name] = r2_score(p, yt)
                        except Exception as ex:
                            rec[name] = np.nan
                            print(f"[fail] {name} {attr} {tgt} K={K} rep={rep}: {ex}",
                                  flush=True)
                    try:
                        p = PLSBaseline().fit(X_t[tr], y_t[tr]).predict(X_t[te])
                        rec["PLS-target"] = r2_score(p, yt)
                    except Exception:
                        rec["PLS-target"] = np.nan
                    try:
                        p = pls_src.predict(X_t[te])
                        rec["PLS-m5->tgt(0shot)"] = r2_score(p, yt)
                    except Exception:
                        rec["PLS-m5->tgt(0shot)"] = np.nan
                    rows.append(rec)
                print(f"  [{attr} {tgt} K={K}] done", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "per_rep.csv", index=False)
    methods = ["ANIL(mae)", "PI-ANIL(mae)", "PLS-target", "PLS-m5->tgt(0shot)"]

    lines = [f"T1 corn cross-instrument Option B (leak-proof holdout {args.n_holdout}, attrs={attrs}, "
             f"reps={args.reps})", "=" * 92]
    for attr in attrs:
        lines.append(f"\n===== attr={attr} (mean R2 +/- std) =====")
        lines.append(f"{'tgt':>4} {'K':>3} | " + " ".join(f"{m:>14}" for m in methods))
        lines.append("-" * 92)
        for tgt in targets:
            for K in shots:
                s = df[(df.attr == attr) & (df.target == tgt) & (df.shot == K)]
                if len(s) == 0:
                    continue
                cells = []
                for m in methods:
                    v = s[m].dropna()
                    cells.append(f"{v.mean():+8.3f}±{v.std():4.2f}" if len(v) else "   N/A      ")
                lines.append(f"{tgt:>4} {K:>3} | " + " ".join(f"{c:>14}" for c in cells))
    lines.append("\n===== PI-ANIL vs ANIL paired Wilcoxon (one-sided PI>ANIL) =====")
    lines.append(f"{'attr':>9} {'tgt':>4} {'K':>3} | {'med d':>8} {'PI>ANIL':>8} {'p':>8}")
    lines.append("-" * 60)
    for attr in attrs:
        for tgt in targets:
            for K in shots:
                s = df[(df.attr == attr) & (df.target == tgt) & (df.shot == K)]
                if len(s) < 3:
                    continue
                d = (s["PI-ANIL(mae)"] - s["ANIL(mae)"]).dropna()
                if len(d) < 3:
                    continue
                try:
                    p = wilcoxon(s["PI-ANIL(mae)"], s["ANIL(mae)"],
                                 alternative="greater").pvalue
                except Exception:
                    p = float("nan")
                lines.append(f"{attr:>9} {tgt:>4} {K:>3} | {d.median():>+8.3f} "
                             f"{(d > 0).sum()}/{len(d):>3} {p:>8.4f}")
    summary = "\n".join(lines)
    (OUT / "summary.txt").write_text(summary, encoding="utf-8")
    print("\n" + summary, flush=True)


if __name__ == "__main__":
    main()
