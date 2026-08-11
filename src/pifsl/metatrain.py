"""Physics-informed meta-learning (PI-ANIL) meta-training.

On top of ANIL, physics constraints are embedded into the
meta-objective:
  loss = query MSE + lam_add * additivity constraint + lam_band * band-saliency constraint

Variants: vanilla (no constraints, for LOMO control) / add / band / both
Task pools: multi(13) / no_gasoline(12) / no_diesel(6) / diesel(7)
Early stopping and validation use query MSE only (consistent with the ANIL
protocol; constraint terms do not participate in model selection).

CLI:
  python3 -m src.pifsl.metatrain --variant both --init mae --source multi --episodes 2500
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.fsl_regression.core import (TASKS, DIESEL_TASKS, load_task,
                                     sample_train_episode, zscore_fit, EMB_DIM,
                                     corn_holdout_split)
from src.fsl_regression.metatrain import (build_encoder, episode_adapt_anil)
from .constraints import band_mask, additivity_loss, band_saliency_loss

CKPT_DIR = ROOT / "results" / "physics" / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

POOLS = dict(
    multi=list(TASKS),
    t3=list(TASKS),  # 14-task pool including selfmix (same as multi; tag_suffix distinguishes the ckpt)
    no_gasoline=[t for t in TASKS if t != "gasoline-octane"],
    no_diesel=[t for t in TASKS if not t.startswith("diesel-")],
    no_corn=[t for t in TASKS if not t.startswith("corn")],
    no_evoo=[t for t in TASKS if not t.startswith("evoo")],
    no_edibleoil=[t for t in TASKS if t != "edibleoil-pv"],  # LOMO: hold out edible oil as the cross-material target
    diesel=list(DIESEL_TASKS),
)


def episode_loss_pi(enc, head, Xs, ys_z, Xq, yq_z, lr_in, steps,
                    variant, mask_t, lam_add, lam_band, rng):
    import torch
    w, b, mu, sd = episode_adapt_anil(enc, head, Xs, ys_z, lr_in, steps)
    use_band = variant in ("band", "both")
    if use_band:
        Xq = Xq.clone().requires_grad_(True)
    zq = enc(Xq)
    pred = (((zq - mu) / sd).clamp(-5, 5) @ w.T + b).squeeze(1)
    loss_task = torch.mean((pred - yq_z) ** 2)
    loss = loss_task
    if variant in ("add", "both"):
        loss = loss + lam_add * additivity_loss(enc, torch.cat([Xs, Xq]), rng)
    if use_band:
        loss = loss + lam_band * band_saliency_loss(pred, Xq, mask_t)
    return loss, float(loss_task.item())


def meta_train_pi(variant, init, source, n_episodes=2500, K=10, Q=16, lr=1e-3,
                  lr_in=0.01, inner_steps=5, lam_add=0.1, lam_band=0.1,
                  seed=0, val_every=250, patience=4, time_budget=240,
                  tag_suffix="", probe_fn=None, probe_every=100, corn_holdout=0,
                  device="cpu"):
    import torch

    rng = np.random.default_rng(seed)
    pool = POOLS[source]
    data = {t: load_task(t) for t in pool}
    # Option B (cross-instrument holdout): corn tasks use only the train
    # subset (held-out samples never enter meta-training in any form)
    corn_tr_idx = None
    if corn_holdout and corn_holdout > 0:
        corn_tr_idx, _ho = corn_holdout_split(80, int(corn_holdout))
        for t in list(data):
            if t.startswith("corn"):
                d = data[t]
                data[t] = dict(X=d["X"][corn_tr_idx], y=d["y"][corn_tr_idx],
                               groups=d.get("groups"))
        print(f"[xinstB] corn_holdout={corn_holdout}: corn tasks use only {len(corn_tr_idx)} "
              f"training samples ({len(_ho)} held out from meta-training)", flush=True)
    masks = {t: torch.tensor(band_mask(t)).to(device) for t in pool}
    enc = build_encoder(init, device=device)
    head = torch.nn.Linear(EMB_DIM, 1).to(device)
    params = list(enc.parameters()) + list(head.parameters())
    opt = torch.optim.Adam(params, lr=lr)

    tag = f"anilpi__{variant}{tag_suffix}__{init}__{source}"
    ckpt_file = CKPT_DIR / f"{tag}.pt"
    log_file = CKPT_DIR / f"trainlog__{tag}.csv"
    ep0, best_val, best_state, bad = 0, np.inf, None, 0
    if ckpt_file.exists():
        ck = torch.load(ckpt_file, map_location=device, weights_only=False)
        enc.load_state_dict(ck["enc"]); head.load_state_dict(ck["head"])
        opt.load_state_dict(ck["opt"])
        ep0, best_val = ck["epoch"], ck["best_val"]
        best_state, bad = ck.get("best_state"), ck.get("bad", 0)
        print(f"[resume] {tag} from ep {ep0} (best {best_val:.4f})", flush=True)

    Xt = {t: (torch.from_numpy(d["X"])[:, None].to(device),
              torch.from_numpy(d["y"]).float().to(device))
          for t, d in data.items()}

    def run_episode(t, rep):
        X, y = Xt[t]
        rr = np.random.default_rng(seed * 7919 + rep)
        si, qi = sample_train_episode(rr, X, y, K, Q)
        ym, ysd = zscore_fit(y[si].cpu().numpy())
        return episode_loss_pi(enc, head, X[si], (y[si] - ym) / ysd,
                               X[qi], (y[qi] - ym) / ysd, lr_in, inner_steps,
                               variant, masks[t], lam_add, lam_band, rr)

    t0 = time.time()
    for ep in range(ep0 + 1, ep0 + n_episodes + 1):
        enc.train()
        t = pool[int(rng.integers(len(pool)))]
        opt.zero_grad()
        loss, lv_task = run_episode(t, rep=ep)
        lv = float(loss.item())
        if not np.isfinite(lv) or lv_task > 50:
            print(f"  [skip] ep {ep}: anomalous episode (loss {lv:.1f})", flush=True)
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        opt.step()
        if ep % 100 == 0:
            print(f"[{tag}] ep {ep}: task_mse={lv_task:.4f} ({time.time() - t0:.0f}s)",
                  flush=True)
            with open(log_file, "a") as f:
                f.write(f"{ep},{lv_task:.6f}\n")
        if probe_fn is not None and ep % probe_every == 0:
            probe_fn(enc, ep)
        if ep % val_every == 0 or ep == ep0 + n_episodes:
            enc.eval()
            vals = []
            with torch.enable_grad():  # band-variant episodes contain autograd.grad
                for vr in range(40):
                    t = pool[int(rng.integers(len(pool)))]
                    _l, vlt = run_episode(t, rep=10 ** 6 + ep * 100 + vr)
                    vals.append(vlt)
            va = float(np.nanmean(vals))
            print(f"  [val] ep {ep}: query_mse={va:.4f} (best {best_val:.4f})",
                  flush=True)
            if va < best_val:
                best_val, bad = va, 0
                best_state = dict(enc={k: v.clone() for k, v in enc.state_dict().items()},
                                  head={k: v.clone() for k, v in head.state_dict().items()})
            else:
                bad += 1
                if bad >= patience:
                    print(f"  early stopping at ep {ep}", flush=True)
                    break
        if time.time() - t0 > time_budget:
            print(f"  time budget reached; checkpointed at ep {ep}", flush=True)
            break
    if best_state is not None:
        enc.load_state_dict(best_state["enc"])
        head.load_state_dict(best_state["head"])
    torch.save(dict(enc=enc.state_dict(), head=head.state_dict(),
                    opt=opt.state_dict(), epoch=ep, best_val=best_val,
                    best_state=best_state, bad=bad,
                    variant=variant, init=init, source=source, K=K, Q=Q,
                    lr_in=lr_in, inner_steps=inner_steps,
                    lam_add=lam_add, lam_band=lam_band,
                    corn_holdout=corn_holdout), ckpt_file)
    print(f"saved -> {ckpt_file} (ep {ep}, best_val {best_val:.4f})", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True,
                    choices=["vanilla", "add", "band", "both"])
    ap.add_argument("--init", default="mae", choices=["mae", "rand"])
    ap.add_argument("--source", default="multi", choices=list(POOLS))
    ap.add_argument("--episodes", type=int, default=2500)
    ap.add_argument("--lam-add", type=float, default=0.1)
    ap.add_argument("--lam-band", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag-suffix", default="")
    ap.add_argument("--corn-holdout", type=int, default=0,
                    help="Option B: hold out N corn physical samples from meta-training (0=off, canonical)")
    ap.add_argument("--device", default="cpu", help="cpu / cuda (use cuda for GPU)")
    ap.add_argument("--time-budget", type=int, default=240,
                    help="budget in seconds (GPU 2500ep usually finishes well below this; increase if needed)")
    args = ap.parse_args()
    meta_train_pi(args.variant, args.init, args.source, n_episodes=args.episodes,
                  lam_add=args.lam_add, lam_band=args.lam_band, seed=args.seed,
                  tag_suffix=args.tag_suffix, corn_holdout=args.corn_holdout,
                  time_budget=args.time_budget, device=args.device)
