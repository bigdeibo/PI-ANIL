"""TRIP benchmark meta-training (Garzón 2026 head-to-head, Option B in-house unified implementation).

Meta-trains the encoder on TRIP train11, reusing the method-level episode loss
functions from Stages 3/4:
  - anil : src.fsl_regression.metatrain.episode_loss_anil
  - proto: src.fsl_regression.metatrain.episode_loss_proto
  - pi   : src.pifsl.metatrain.episode_loss_pi (ANIL + additivity + band saliency)

episode = K+Q sampling within a single TRIP task (from that task's X_supp
pool), y z-scored with support-set statistics, embeddings standardized with
support-set statistics — bit-identical to our main adaptation protocol, with only the
data source switched to TRIP.

Checkpoints: results/trip_benchmark/checkpoints/{tag}.pt (enc/head/log_tau/config).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.fsl_regression.core import EMB_DIM, sample_train_episode, zscore_fit
from src.fsl_regression.metatrain import (
    build_encoder, episode_loss_anil, episode_loss_proto,
)
from src.pifsl.metatrain import episode_loss_pi
from src.pifsl.constraints import BANDS_CH, BANDS_OH

from .trip_data import TRIP_TRAIN, load_trip_task


# ---------------- full second-order MAML (functional_call all-parameter inner loop) ----------------
def maml_adapt(enc, head, Xs, ys_z, lr_in, steps, create_graph=True, clip_inner=1.0):
    """MAML inner loop: functional_call-based GD adaptation of **all** encoder+head
    parameters for `steps` steps.

    create_graph=True -> second order (the meta-gradient flows through the
    inner-loop updates); in **train mode**, BN normalizes by support-set batch
    statistics (constraining activation magnitudes to prevent divergence — in
    eval mode BN uses frozen running stats = identity, no normalization, so
    activations inflate exponentially during training and explode). Inner-loop
    gradients are element-wise clipped + zs/pred clamped + divergence guard.
    Returns (fast_enc, fast_head, mu, sd).
    fast_enc/fast_head are {name: tensor} dicts (functional_call parameter dicts).
    """
    import torch
    from torch.func import functional_call
    enc.train()  # critical: BN normalizes with batch statistics, constraining activation magnitudes
    fast_enc = dict(enc.named_parameters())
    fast_head = dict(head.named_parameters())
    enc_names = list(fast_enc.keys())
    head_names = list(fast_head.keys())
    mu = sd = None
    broke = False
    for _ in range(steps):
        zs = functional_call(enc, fast_enc, (Xs,))
        mu, sd = zs.mean(0, keepdim=True), zs.std(0, keepdim=True) + 1e-8
        zs_n = ((zs - mu) / sd).clamp(-5, 5)
        pred = functional_call(head, fast_head, (zs_n,)).squeeze(1).clamp(-5, 5)
        loss = torch.mean((pred - ys_z) ** 2)
        if not torch.isfinite(loss) or float(loss.item()) > 50.0:
            broke = True
            break
        g = torch.autograd.grad(loss, list(fast_enc.values()) + list(fast_head.values()),
                                create_graph=create_graph)
        if clip_inner:
            g = [torch.clamp(gr, -clip_inner, clip_inner) for gr in g]
        ge, gh = g[:len(enc_names)], g[len(enc_names):]
        fast_enc = {n: p - lr_in * gr for (n, p), gr in zip(fast_enc.items(), ge)}
        fast_head = {n: p - lr_in * gr for (n, p), gr in zip(fast_head.items(), gh)}
    return fast_enc, fast_head, mu, sd


def episode_loss_maml(enc, head, Xs, ys_z, Xq, yq_z, lr_in, steps):
    """Second-order MAML query MSE: inner-loop all-parameter adaptation -> query
    prediction MSE (meta-gradient flows through the inner loop)."""
    import torch
    from torch.func import functional_call
    fast_enc, fast_head, mu, sd = maml_adapt(enc, head, Xs, ys_z, lr_in, steps)
    if mu is None:  # degenerate case steps==0
        zs = functional_call(enc, fast_enc, (Xs,))
        mu, sd = zs.mean(0, keepdim=True), zs.std(0, keepdim=True) + 1e-8
    zq = functional_call(enc, fast_enc, (Xq,))
    pred_q = functional_call(head, fast_head, ((zq - mu) / sd,)).squeeze(1)
    return torch.mean((pred_q - yq_z) ** 2)

CKPT_DIR = ROOT / "results" / "trip_benchmark" / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

_mask_cache: dict = {}


def trip_band_mask(task: str, n_out: int = 512) -> np.ndarray:
    """C-H band mask for a TRIP task (plus O-H for corn), as a 0/1 vector on the
    512 resampled positions.

    Built from the original wavelength axis of load_trip_task; if the wavelength
    axis contains no C-H band (e.g. Eggs 740-1070, Wheat 850-1050) or is an
    index axis (Wheat stored as 0..99) -> returns all zeros (band prior
    automatically disabled).
    """
    key = (task, n_out)
    if key in _mask_cache:
        return _mask_cache[key]
    d = load_trip_task(task)
    wl = np.asarray(d["wavelengths"], float)
    # index-axis detection (Wheat stored as 0..99): a too-small maximum means no real wavelength coordinates
    if wl.max() < 300:
        out = np.zeros(n_out, dtype=np.float32)
        _mask_cache[key] = out
        return out
    bands = list(BANDS_CH) + (BANDS_OH if task.startswith("Corn") else [])
    m = np.zeros(len(wl), dtype=np.float32)
    for a, b in bands:
        m[(wl >= a) & (wl <= b)] = 1.0
    pos_old = np.linspace(0, 1, len(wl))
    pos_new = np.linspace(0, 1, n_out)
    mi = np.interp(pos_new, pos_old, m)
    out = (mi > 0.5).astype(np.float32)
    _mask_cache[key] = out
    return out


def trip_meta_train(method, init, variant="both", n_episodes=2500, K=10, Q=16,
                    lr=1e-3, lr_in=0.01, inner_steps=5, lam_add=0.1, lam_band=0.1,
                    seed=0, val_every=250, patience=4, time_budget=240,
                    tag_suffix="", arch="resnet1", emb_dim=None, device="cpu",
                    meta_batch=1):
    """Meta-train on TRIP train11.

    method in {anil, proto, pi, maml}; variant is used by pi only
    (add/band/both/vanilla).
    arch/emb_dim select the backbone (default resnet1/EMB_DIM=128; exact uses
    resnet1d_garzon/512). `device` moves the modules ("cpu"/"cuda").
    meta_batch = number of tasks aggregated per meta-update (implemented via
    gradient accumulation; meta_batch=1 reproduces the legacy G-P1b behavior;
    Garzón uses 25). n_episodes is counted in task-episodes (literally aligned
    with Garzón Table B.2: MAML 50000, ProtoNet 5000).
    time_budget: seconds before checkpoint-and-exit (resumable).
    """
    import torch

    emb_dim = EMB_DIM if emb_dim is None else emb_dim
    rng = np.random.default_rng(seed)
    pool = list(TRIP_TRAIN)
    data = {t: load_trip_task(t) for t in pool}
    masks = {t: torch.tensor(trip_band_mask(t)).to(device) for t in pool}
    enc = build_encoder(init, arch=arch, emb_dim=emb_dim, device=device)
    head = torch.nn.Linear(emb_dim, 1).to(device)
    log_tau = torch.nn.Parameter(torch.tensor(0.0, device=device))  # proto only

    if method == "proto":
        params = list(enc.parameters()) + [log_tau]
    else:
        params = list(enc.parameters()) + list(head.parameters())
    opt = torch.optim.Adam(params, lr=lr)

    tag = (f"trip__pi__{variant}__{init}{tag_suffix}" if method == "pi"
           else f"trip__{method}__{init}{tag_suffix}")
    ckpt_file = CKPT_DIR / f"{tag}.pt"
    log_file = CKPT_DIR / f"trainlog__{tag}.csv"
    ep0, best_val, best_state, bad = 0, np.inf, None, 0
    if ckpt_file.exists():
        ck = torch.load(ckpt_file, map_location=device, weights_only=False)
        enc.load_state_dict(ck["enc"])
        head.load_state_dict(ck["head"])
        if "log_tau" in ck:
            log_tau.data = ck["log_tau"].to(device)
        opt.load_state_dict(ck["opt"])
        ep0, best_val = ck["epoch"], ck["best_val"]
        best_state, bad = ck.get("best_state"), ck.get("bad", 0)
        print(f"[resume] {tag} from ep {ep0} (best_val {best_val:.4f})", flush=True)

    Xt = {t: (torch.from_numpy(d["X_supp"])[:, None].to(device),
              torch.from_numpy(d["y_supp"]).float().to(device))
          for t, d in data.items()}

    def run_episode(t, rep):
        X, y = Xt[t]
        rr = np.random.default_rng(seed * 7919 + rep)
        si, qi = sample_train_episode(rr, X, y, K, Q)
        ym, ysd = zscore_fit(y[si].cpu().numpy())
        ys_z = (y[si] - ym) / ysd
        yq_z = (y[qi] - ym) / ysd
        if method == "proto":
            return episode_loss_proto(enc, log_tau, X[si], ys_z, X[qi], yq_z), float("nan")
        if method == "anil":
            l = episode_loss_anil(enc, head, X[si], ys_z, X[qi], yq_z, lr_in, inner_steps)
            return l, float(l.item())
        if method == "maml":
            # with BN-train (batch normalization) constraining activations, the faithful 5 steps + lr 0.01 can be used
            l = episode_loss_maml(enc, head, X[si], ys_z, X[qi], yq_z, lr_in, inner_steps)
            return l, float(l.item())
        # pi: tasks with an empty band mask are automatically downgraded (both->add, band->vanilla)
        eff = variant
        if variant in ("band", "both") and float(masks[t].sum()) == 0:
            eff = "add" if variant == "both" else "vanilla"
        return episode_loss_pi(enc, head, X[si], ys_z, X[qi], yq_z, lr_in, inner_steps,
                               eff, masks[t], lam_add, lam_band, rr)

    t0 = time.time()
    mb = max(1, int(meta_batch))
    log_every_upd = max(1, 100 // mb)
    val_every_upd = max(1, val_every // mb)
    upd = 0
    ep = ep0
    while ep < ep0 + n_episodes:
        enc.train()
        opt.zero_grad()
        accum = 0
        last_lv = float("nan")
        for _ in range(mb):
            if ep >= ep0 + n_episodes:
                break
            ep += 1
            t = pool[int(rng.integers(len(pool)))]
            loss, lv_task = run_episode(t, rep=ep)
            lv = float(loss.item())
            task_ref = lv if not np.isfinite(lv_task) else lv_task
            if not np.isfinite(lv) or task_ref > 50:
                print(f"  [skip] ep {ep}: anomalous episode (loss {lv:.1f})", flush=True)
                continue
            loss.backward()  # accumulate gradients per task (graph released immediately, saving memory)
            accum += 1
            last_lv = lv
        if accum == 0:
            continue
        for p in params:
            if p.grad is not None:
                p.grad.mul_(1.0 / accum)  # normalize by the number of valid tasks = meta-batch mean gradient
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        opt.step()
        upd += 1
        if upd % log_every_upd == 0:
            print(f"[{tag}] upd {upd} (ep {ep}): loss={last_lv:.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
            with open(log_file, "a") as f:
                f.write(f"{ep},{last_lv:.6f}\n")
        if upd % val_every_upd == 0 or ep >= ep0 + n_episodes:
            enc.eval()
            vals = []
            ctx = torch.no_grad() if method in ("proto",) else torch.enable_grad()
            with ctx:
                for vr in range(40):
                    t = pool[int(rng.integers(len(pool)))]
                    _l, vlt = run_episode(t, rep=10 ** 6 + ep * 100 + vr)
                    vals.append(float(_l.item()) if np.isnan(vlt) else vlt)
            va = float(np.nanmean(vals))
            print(f"  [val] upd {upd} (ep {ep}): query_mse={va:.4f} "
                  f"(best {best_val:.4f})", flush=True)
            if va < best_val:
                best_val, bad = va, 0
                best_state = dict(enc={k: v.clone() for k, v in enc.state_dict().items()},
                                  head={k: v.clone() for k, v in head.state_dict().items()},
                                  log_tau=log_tau.data.clone())
            else:
                bad += 1
                if bad >= patience:
                    print(f"  early stopping at upd {upd} (ep {ep})", flush=True)
                    break
        if time.time() - t0 > time_budget:
            print(f"  time budget reached; checkpointed at upd {upd} (ep {ep})", flush=True)
            break
    if best_state is not None:
        enc.load_state_dict(best_state["enc"])
        head.load_state_dict(best_state["head"])
        log_tau.data = best_state["log_tau"]
    torch.save(dict(enc=enc.state_dict(), head=head.state_dict(),
                    log_tau=log_tau.data, opt=opt.state_dict(),
                    epoch=ep, best_val=best_val, best_state=best_state, bad=bad,
                    method=method, variant=variant, init=init, K=K, Q=Q,
                    lr_in=lr_in, inner_steps=inner_steps,
                    lam_add=lam_add, lam_band=lam_band,
                    arch=arch, emb_dim=emb_dim, meta_batch=meta_batch), ckpt_file)
    print(f"saved -> {ckpt_file} (ep {ep}, best_val {best_val:.4f})", flush=True)


def trip_pretrain_ft(init, epochs=200, lr=1e-4, batch_size=32, seed=0,
                     time_budget=400, tag_suffix="", arch="resnet1",
                     emb_dim=None, device="cpu"):
    """FT pretraining: supervised training of enc+head on TRIP train11 (each
    task's y is z-scored per task before pooling).

    Produces trip__ft__{init}.pt (enc+head), consumed by
    run_trip_headtohead.eval_ft for full-parameter finetuning at test time.
    arch/emb_dim select the backbone; `device` moves the modules. lr plus a
    divergence guard (skip batches with loss>50) prevents pretraining blow-ups.
    """
    import torch
    from .trip_data import load_trip_task
    emb_dim = EMB_DIM if emb_dim is None else emb_dim
    torch.manual_seed(seed)
    pool = list(TRIP_TRAIN)
    Xs, ys = [], []
    for t in pool:
        d = load_trip_task(t)
        y = d["y_supp"]
        ym, ysd = float(y.mean()), float(y.std() + 1e-8)
        Xs.append(d["X_supp"])
        ys.append((y - ym) / ysd)
    X = torch.from_numpy(np.concatenate(Xs)[:, None]).float().to(device)
    y = torch.from_numpy(np.concatenate(ys).astype(np.float32).reshape(-1, 1)).to(device)
    N = len(y)

    enc = build_encoder(init, arch=arch, emb_dim=emb_dim, device=device)
    head = torch.nn.Linear(emb_dim, 1).to(device)
    opt = torch.optim.Adam(list(enc.parameters()) + list(head.parameters()), lr=lr)
    tag = f"trip__ft__{init}{tag_suffix}"
    ckpt_file = CKPT_DIR / f"{tag}.pt"
    log_file = CKPT_DIR / f"trainlog__{tag}.csv"

    rng = np.random.default_rng(seed)
    t0 = time.time()
    best_loss, best_state = np.inf, None
    for ep in range(1, epochs + 1):
        enc.train(); head.train()
        perm = rng.permutation(N)
        tot, cnt = 0.0, 0
        for i in range(0, N, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            pred = head(enc(X[idx]))
            loss = torch.nn.functional.mse_loss(pred, y[idx])
            lv = float(loss.item())
            if not np.isfinite(lv) or lv > 50.0:
                continue  # skip diverged batch
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(head.parameters()), 5.0)
            opt.step()
            tot += lv * len(idx); cnt += len(idx)
        lv = tot / max(cnt, 1)
        if ep % 10 == 0 or ep == epochs:
            with open(log_file, "a") as f:
                f.write(f"{ep},{lv:.6f}\n")
            if lv < best_loss:
                best_loss = lv
                best_state = dict(enc={k: v.clone() for k, v in enc.state_dict().items()},
                                  head={k: v.clone() for k, v in head.state_dict().items()})
            if ep % 25 == 0:
                print(f"[{tag}] ep {ep}: train_mse={lv:.4f} ({time.time() - t0:.0f}s)", flush=True)
        if time.time() - t0 > time_budget:
            print(f"  time budget reached; checkpointed at ep {ep}", flush=True)
            break
    if best_state is not None:
        enc.load_state_dict(best_state["enc"]); head.load_state_dict(best_state["head"])
    torch.save(dict(enc=enc.state_dict(), head=head.state_dict(), epoch=ep,
                    best_val=best_loss, method="ft", init=init,
                    arch=arch, emb_dim=emb_dim), ckpt_file)
    print(f"saved -> {ckpt_file} (ep {ep}, train_mse {best_loss:.4f})", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["anil", "proto", "pi", "maml"])
    ap.add_argument("--init", required=True, choices=["rand", "mae"])
    ap.add_argument("--variant", default="both",
                    choices=["vanilla", "add", "band", "both"])
    ap.add_argument("--episodes", type=int, default=2500)
    ap.add_argument("--time-budget", type=int, default=240)
    ap.add_argument("--lr", type=float, default=1e-3, help="meta (outer-loop) Adam lr")
    ap.add_argument("--lr-in", type=float, default=0.01,
                    help="inner-loop adaptation lr (Garzón MAML=0.1)")
    ap.add_argument("--inner-steps", type=int, default=5)
    ap.add_argument("--meta-batch", type=int, default=1,
                    help="number of tasks aggregated per meta-update (Garzón=25)")
    ap.add_argument("--val-every", type=int, default=250,
                    help="validation interval (in task-episodes; divided by meta-batch internally)")
    ap.add_argument("--patience", type=int, default=4, help="validation early-stopping patience")
    ap.add_argument("--arch", default="resnet1",
                    help="backbone registry key (use resnet1d_garzon for exact)")
    ap.add_argument("--emb-dim", type=int, default=None, help="embedding dimension (exact=512)")
    ap.add_argument("--device", default="cpu", help="cpu / cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag-suffix", default="",
                    help="ckpt/log tag suffix (use __exact for exact runs; isolates from and does not overwrite the legacy 128-d runs)")
    args = ap.parse_args()
    trip_meta_train(args.method, args.init, variant=args.variant,
                    n_episodes=args.episodes, time_budget=args.time_budget,
                    lr=args.lr, lr_in=args.lr_in, inner_steps=args.inner_steps,
                    meta_batch=args.meta_batch, arch=args.arch, emb_dim=args.emb_dim,
                    device=args.device, seed=args.seed, tag_suffix=args.tag_suffix,
                    val_every=args.val_every, patience=args.patience)
