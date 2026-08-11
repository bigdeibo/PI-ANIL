"""TRIP head-to-head evaluation (Garzón 2026, in-house unified implementation; G-P1b-exact adds exact backbone + GPU support).

Evaluates each method on TRIP test7 at 5/10/25-shot × 10 reps, incrementally writing
results per (method,task,shot,rep) to results/trip_benchmark/{parts_dir}/{tag}__{task}.csv.

The backbone is determined by the arch/emb_dim fields stored in the checkpoint (legacy 128-dim resnet1 or exact 512-dim resnet1d_garzon);
base (no checkpoint) is set by --arch/--emb-dim. Device migration (cpu/cuda).
--tag-suffix isolates exact runs (e.g., __exact) so checkpoints/artifacts do not overwrite the old G-P1b; --parts-dir isolates the artifact directory.

Evaluation protocol is unchanged: trip_eval_support_query (seed=42*1000+K*100+rep, bit-identical across methods),
_stable_lr spectral-radius-adaptive step size, ±5 clamping, 50-step ANIL adaptation—bit-identical to the old G-P1b.
"""
import sys
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.fsl.models import ENCODERS, ResNet1Encoder
from src.fsl_regression.core import r2_score, EMB_DIM
from src.fsl_regression.trip_data import (TRIP_TEST, TRIP_TRAIN, TRIP_VAL, TRIP_TASKS,
                                          trip_eval_support_query)
from src.fewshot import rpd

OUT = ROOT / "results" / "trip_benchmark"
CKPT_DIR = OUT / "checkpoints"

L = 512
EVAL_INNER_STEPS = 50
TASK_GROUPS = dict(test=TRIP_TEST, train=TRIP_TRAIN, val=TRIP_VAL, all=TRIP_TASKS)


def _stable_lr(zs, safety=0.8):
    H = 2.0 * (zs.T @ zs) / len(zs)
    lam = float(torch.linalg.eigvalsh(H).max())
    return safety / max(lam, 1e-6)


def _enc_from_ckpt(ck, device):
    """Build the encoder from the ckpt arch/emb_dim fields (legacy ckpts without these fields fall back to resnet1/EMB_DIM). Returns (enc, emb_dim)."""
    arch = ck.get("arch", "resnet1")
    emb_dim = ck.get("emb_dim", EMB_DIM)
    cls = ENCODERS.get(arch, ResNet1Encoder)
    enc = cls(L, emb_dim=emb_dim).to(device)
    return enc, emb_dim


def _t(arr, device):
    return torch.from_numpy(arr).float().to(device)


def ckpt_tag(method, init, variant, tag_suffix=""):
    base = f"trip__pi__{variant}__{init}" if method == "pi" else f"trip__{method}__{init}"
    return base + tag_suffix


def load_ckpt(method, init, variant, tag_suffix="", device="cpu"):
    tag = ckpt_tag(method, init, variant, tag_suffix)
    f = CKPT_DIR / f"{tag}.pt"
    if not f.exists():
        raise FileNotFoundError(f"missing checkpoint: {f} (run trip_metatrain training first)")
    return torch.load(f, map_location=device, weights_only=False), tag


# ---------------- Per-method evaluation (returns predictions on the raw-y scale) ----------------
def eval_anil(ck, Xs, ys, Xq, yq, ym, ysd, device="cpu"):
    """ANIL/PI: load enc+head, adapt the head for 50 steps on the support embeddings, predict query."""
    enc, emb_dim = _enc_from_ckpt(ck, device); enc.load_state_dict(ck["enc"]); enc.eval()
    head = nn.Linear(emb_dim, 1).to(device); head.load_state_dict(ck["head"])
    Xs_t = _t(Xs[:, None], device); Xq_t = _t(Xq[:, None], device)
    ys_z = _t(((ys - ym) / ysd).astype(np.float32), device)
    with torch.no_grad():
        zs = enc(Xs_t)
        mu, sd = zs.mean(0, keepdim=True), zs.std(0, keepdim=True) + 1e-8
        zs = ((zs - mu) / sd).clamp(-5, 5)
    lr_in = _stable_lr(zs)
    prev = np.inf
    for _ in range(EVAL_INNER_STEPS):
        loss = torch.mean((head(zs).squeeze(1) - ys_z) ** 2)
        lv = float(loss.item())
        if not np.isfinite(lv) or lv > 50.0 or lv > prev * 3:
            break
        prev = lv
        g = torch.autograd.grad(loss, list(head.parameters()))
        with torch.no_grad():
            for p, gr in zip(head.parameters(), g):
                p.sub_(lr_in * gr)
    with torch.no_grad():
        zq = ((enc(Xq_t) - mu) / sd).clamp(-5, 5)
        pred = head(zq).squeeze(1).cpu().numpy()
    return pred * ysd + ym


def _protonet_predict(Es, ys, Ete):
    """ProtoNet regression: standardize support set → LOO selection of τ → weighted-distance prediction (bit-identical to p1c)."""
    mu, sd = Es.mean(0), Es.std(0) + 1e-8
    Es_n = (Es - mu) / sd
    Ete_n = (Ete - mu) / sd
    Ds = ((Es_n[:, None, :] - Es_n[None, :, :]) ** 2).sum(-1)
    med = np.median(Ds[Ds > 0]) if (Ds > 0).any() else 1.0
    taus = med * np.logspace(-2, 2, 25)
    best_tau, best_loo = taus[0], np.inf
    for tau in taus:
        W = np.exp(-Ds / tau)
        np.fill_diagonal(W, 0.0)
        s = W.sum(1, keepdims=True); s[s == 0] = 1.0
        loo = float(((W / s @ ys - ys) ** 2).mean())
        if loo < best_loo:
            best_loo, best_tau = loo, tau
    D = ((Ete_n[:, None, :] - Es_n[None, :, :]) ** 2).sum(-1)
    W = np.exp(-D / best_tau)
    s = W.sum(1, keepdims=True); s[s == 0] = 1.0
    return (W / s) @ ys


def eval_proto(ck, Xs, ys, Xq, yq, ym, ysd, device="cpu"):
    enc, _ = _enc_from_ckpt(ck, device); enc.load_state_dict(ck["enc"]); enc.eval()
    with torch.no_grad():
        zs = enc(_t(Xs[:, None], device)).cpu().numpy().astype(np.float32)
        zq = enc(_t(Xq[:, None], device)).cpu().numpy().astype(np.float32)
    pred = _protonet_predict(zs, ys.astype(np.float32), zq)
    # ±5σ prediction clamping + nan fallback to the support-set mean—same protocol as eval_base/ft/maml.
    # With 512-dim embeddings, distance concentration can underflow kernel-regression weights,
    # causing out-of-range predictions (extreme negative R² on Shootout/Corn);
    # the legacy 128-dim distances were small enough to avoid this, so no clamp was needed;
    # with clamping, ProtoNet is evaluated on the same footing as the other methods.
    pred = np.nan_to_num(pred, nan=ym)
    return np.clip(pred, ym - 5 * ysd, ym + 5 * ysd)


def eval_base(Xs, ys, Xq, yq, ym, ysd, epochs=10, lr=1e-5, seed=0,
              arch="resnet1", emb_dim=None, device="cpu"):
    """Individual: train a randomly initialized backbone+head on the support set (=Garzón Individual; lr=1e-5/ep=10)."""
    emb_dim = EMB_DIM if emb_dim is None else emb_dim
    torch.manual_seed(seed * 7 + seed)
    cls = ENCODERS.get(arch, ResNet1Encoder)
    enc = cls(L, emb_dim).to(device); head = nn.Linear(emb_dim, 1).to(device)
    opt = torch.optim.Adam(list(enc.parameters()) + list(head.parameters()), lr=lr)
    Xs_t = _t(Xs[:, None], device)
    ys_z = _t(((ys - ym) / ysd).astype(np.float32).reshape(-1, 1), device)
    Xq_t = _t(Xq[:, None], device)
    for _ in range(epochs):
        enc.train(); head.train(); opt.zero_grad()
        loss = nn.functional.mse_loss(head(enc(Xs_t)), ys_z)
        loss.backward(); opt.step()
    enc.eval(); head.eval()
    with torch.no_grad():
        pred = head(enc(Xq_t)).cpu().numpy().reshape(-1) * ysd + ym
    return np.clip(pred, ym - 5 * ysd, ym + 5 * ysd)


def eval_maml(ck, Xs, ys, Xq, yq, ym, ysd, lr_in=0.01, steps=5, device="cpu"):
    """Full MAML evaluation: full-parameter inner-loop adaptation on the support set (create_graph=False), predict query."""
    from torch.func import functional_call
    from src.fsl_regression.trip_metatrain import maml_adapt
    enc, emb_dim = _enc_from_ckpt(ck, device); enc.load_state_dict(ck["enc"])
    head = nn.Linear(emb_dim, 1).to(device); head.load_state_dict(ck["head"])
    Xs_t = _t(Xs[:, None], device); Xq_t = _t(Xq[:, None], device)
    ys_z = _t(((ys - ym) / ysd).astype(np.float32), device)
    enc.train()  # consistent with training: BN normalizes using support-set batch statistics
    fast_enc, fast_head, mu, sd = maml_adapt(enc, head, Xs_t, ys_z, lr_in, steps,
                                             create_graph=False)
    if mu is None:
        with torch.no_grad():
            zs = functional_call(enc, fast_enc, (Xs_t,))
            mu, sd = zs.mean(0, keepdim=True), zs.std(0, keepdim=True) + 1e-8
    with torch.no_grad():
        zq = functional_call(enc, fast_enc, (Xq_t,))
        pred = functional_call(head, fast_head, ((zq - mu) / sd,)).squeeze(1).cpu().numpy()
    return np.clip(pred * ysd + ym, ym - 5 * ysd, ym + 5 * ysd)


def eval_ft(ck, Xs, ys, Xq, yq, ym, ysd, epochs=10, lr=1e-4, device="cpu"):
    """FT evaluation: full-parameter fine-tuning (Adam) on the support set starting from the train11 pretrained init, predict query."""
    enc, emb_dim = _enc_from_ckpt(ck, device); enc.load_state_dict(ck["enc"])
    head = nn.Linear(emb_dim, 1).to(device); head.load_state_dict(ck["head"])
    opt = torch.optim.Adam(list(enc.parameters()) + list(head.parameters()), lr=lr)
    Xs_t = _t(Xs[:, None], device)
    ys_z = _t(((ys - ym) / ysd).astype(np.float32).reshape(-1, 1), device)
    Xq_t = _t(Xq[:, None], device)
    for _ in range(epochs):
        enc.train(); head.train(); opt.zero_grad()
        loss = nn.functional.mse_loss(head(enc(Xs_t)), ys_z)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(head.parameters()), 5.0)
        opt.step()
    enc.eval(); head.eval()
    with torch.no_grad():
        pred = head(enc(Xq_t)).cpu().numpy().reshape(-1) * ysd + ym
    return np.clip(pred, ym - 5 * ysd, ym + 5 * ysd)


# ---------------- Main loop ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True,
                    choices=["anil", "proto", "pi", "maml", "ft", "base"])
    ap.add_argument("--init", default="rand", choices=["rand", "mae"])
    ap.add_argument("--variant", default="both", choices=["vanilla", "add", "band", "both"])
    ap.add_argument("--tasks", default="test")
    ap.add_argument("--shots", default="5,10,25")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=10, help="base/ft only")
    ap.add_argument("--lr", type=float, default=1e-5, help="base/ft only")
    ap.add_argument("--lr-in", type=float, default=0.01, help="inner-loop lr for maml evaluation only (exact aligns with Garzón=0.1)")
    ap.add_argument("--arch", default="resnet1", help="base only (all others read the arch field from the ckpt)")
    ap.add_argument("--emb-dim", type=int, default=None, help="base only")
    ap.add_argument("--tag-suffix", default="", help="ckpt/artifact tag suffix (use __exact for exact runs)")
    ap.add_argument("--parts-dir", default="trip_parts",
                    help="artifact directory name (use trip_exact_parts for exact runs; does not overwrite the old one)")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    device = args.device
    parts = OUT / args.parts_dir
    parts.mkdir(parents=True, exist_ok=True)

    tasks = TASK_GROUPS.get(args.tasks, args.tasks.split(","))
    shots = [int(s) for s in args.shots.split(",")]
    method, init, variant = args.method, args.init, args.variant

    ck = None
    if method in ("anil", "pi"):
        ck, tag = load_ckpt(method, init, variant, args.tag_suffix, device)
        evalfn = eval_anil
    elif method == "proto":
        ck, tag = load_ckpt(method, init, variant, args.tag_suffix, device)
        evalfn = eval_proto
    elif method == "maml":
        ck, tag = load_ckpt(method, init, variant, args.tag_suffix, device)
        evalfn = eval_maml
    elif method == "ft":
        ck, tag = load_ckpt(method, init, variant, args.tag_suffix, device)
        evalfn = eval_ft
    else:  # base
        tag = ckpt_tag("base", init, "", args.tag_suffix)
        evalfn = None

    print(f"[run] method={method} init={init} tag={tag} tasks={len(tasks)} "
          f"shots={shots} reps={args.reps} device={device} parts={args.parts_dir}", flush=True)

    for task in tasks:
        out_file = parts / f"{tag}__{task}.csv"
        done = set()
        if out_file.exists():
            done = set(zip(pd.read_csv(out_file).shot, pd.read_csv(out_file).rep))
        for K in shots:
            for rep in range(args.reps):
                if (K, rep) in done:
                    continue
                Xs, ys, Xq, yq, ym, ysd = trip_eval_support_query(task, K, rep)
                t0 = time.time()
                try:
                    if method == "base":
                        pred = eval_base(Xs, ys, Xq, yq, ym, ysd, epochs=args.epochs,
                                         lr=args.lr, seed=42 + rep, arch=args.arch,
                                         emb_dim=args.emb_dim, device=device)
                    elif method == "ft":
                        pred = eval_ft(ck, Xs, ys, Xq, yq, ym, ysd,
                                       epochs=args.epochs, lr=1e-4, device=device)
                    elif method == "maml":
                        pred = evalfn(ck, Xs, ys, Xq, yq, ym, ysd,
                                      lr_in=args.lr_in, device=device)
                    else:
                        pred = evalfn(ck, Xs, ys, Xq, yq, ym, ysd, device=device)
                except Exception as ex:
                    print(f"[fail] {task} K={K} rep={rep}: {ex}", flush=True)
                    continue
                if pred.shape[0] != len(yq):
                    print(f"[skip] {task} K={K} rep={rep}: pred{pred.shape}!=yq{len(yq)}",
                          flush=True)
                    continue
                rmse = float(np.sqrt(np.mean((pred - yq) ** 2)))
                row = dict(task=task, method=method, init=init, variant=variant,
                           tag=tag, shot=K, rep=rep,
                           r2=r2_score(pred, yq), rmse=rmse, rpd=rpd(yq, rmse))
                pd.DataFrame([row]).to_csv(out_file, mode="a",
                                           header=not out_file.exists(), index=False)
                print(f"[eval] {tag} {task} K={K} rep={rep}: "
                      f"r2={row['r2']:+.3f} ({time.time() - t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
