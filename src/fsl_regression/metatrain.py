"""Meta-training loops: DKL (deep kernel learning) and ANIL/FOMAML (gradient-based meta-learning).

Shared by all three methods: ResNet1 encoder (optionally MAE-initialized),
episode = K+Q sampling within a "material x property" task, y z-scored with
support-set statistics, embeddings standardized with support-set statistics.

- DKL: encoder + RBF-GP head; hyperparameters [lengthscale, outputscale,
  noise] are meta-parameters trained jointly with the encoder; loss = query-set
  GP posterior-mean MSE.
- ANIL: meta-learned encoder + shared head initialization; within an episode
  only the head is adapted (full-batch SGD, inner_steps steps, first-order
  gradients); the query loss is backpropagated to the encoder (via the zq path).
- FOMAML: within an episode all parameters are adapted (first-order); the
  gradient of the query loss at the adapted point serves as the meta-gradient.

Checkpoints: results/meta_routes/checkpoints/{method}__{init}__{source}.pt
(encoder/head/hyperparameters/config/training log), resumable.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from .core import (L, EMB_DIM, TASKS, DIESEL_TASKS, load_task,
                   sample_train_episode, zscore_fit, z_standardize, gp_predict)

CKPT_DIR = ROOT / "results" / "meta_routes" / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)


def build_encoder(init, arch="resnet1", emb_dim=None, device="cpu"):
    """Build the encoder. `arch` selects the backbone (ENCODERS registry key);
    emb_dim defaults to EMB_DIM (128); `device` moves the module. With init='mae',
    the SSL checkpoint is chosen by (arch, emb_dim):
    legacy resnet1/128 -> encoder_mae.pt; new garzon/512 -> encoder_mae_512.pt."""
    import torch
    from src.fsl.models import ENCODERS
    emb_dim = EMB_DIM if emb_dim is None else emb_dim
    cls = ENCODERS[arch]
    enc = cls(L, emb_dim=emb_dim)
    if init == "mae":
        from src.ssl import load_encoder
        track = "mae" if (arch == "resnet1" and emb_dim == EMB_DIM) else "mae_512"
        enc.load_state_dict(load_encoder(track).state_dict())
    elif init == "rand":
        torch.manual_seed(1234)
    return enc.to(device)


def task_pool(source):
    return list(TASKS) if source == "multi" else list(DIESEL_TASKS)


# ---------------- single-episode forward (three methods) ----------------
def episode_loss_dkl(enc, log_hyp, Xs, ys_z, Xq, yq_z):
    import torch
    zs, zq, _, _ = z_standardize_t(enc(Xs), enc(Xq))
    pred = gp_predict(zs, ys_z, zq, log_hyp)
    return torch.mean((pred - yq_z) ** 2)


def z_standardize_t(zs, zq):
    mu = zs.mean(0, keepdim=True)
    sd = zs.std(0, keepdim=True) + 1e-8
    return (zs - mu) / sd, (zq - mu) / sd, mu, sd


def episode_adapt_anil(enc, head_meta, Xs, ys_z, lr_in, steps):
    """ANIL: functionally adapt the head starting from the meta-head
    (create_graph=True, so the meta-head remains learnable).

    Support-set embeddings are detached (first-order encoder approximation:
    encoder gradients flow back only through the query path).
    Returns (w, b, mu, sd): adapted head parameters and support-set embedding
    statistics.
    """
    import torch
    with torch.no_grad():
        zs = enc(Xs)
        mu, sd = zs.mean(0, keepdim=True), zs.std(0, keepdim=True) + 1e-8
        zs = ((zs - mu) / sd).clamp(-5, 5)  # clamp to +/-5, consistent with eval_anil (prevents val explosion from eval-mode BN collapse in deep backbones)
    w, b = head_meta.weight, head_meta.bias
    for _ in range(steps):
        pred = (zs @ w.T + b).squeeze(1)
        loss = torch.mean((pred - ys_z) ** 2)
        gw, gb = torch.autograd.grad(loss, [w, b], create_graph=True)
        w = w - lr_in * gw
        b = b - lr_in * gb
    return w, b, mu, sd


def episode_loss_anil(enc, head_meta, Xs, ys_z, Xq, yq_z, lr_in, steps):
    import torch
    w, b, mu, sd = episode_adapt_anil(enc, head_meta, Xs, ys_z, lr_in, steps)
    zq = enc(Xq)
    pred = (((zq - mu) / sd).clamp(-5, 5) @ w.T + b).squeeze(1)
    return torch.mean((pred - yq_z) ** 2)


def episode_loss_fomaml(enc, head, Xs, ys_z, Xq, yq_z, lr_in, steps):
    """FOMAML: deep-copy, adapt all parameters (first-order), then take the
    gradient of the query loss at the adapted point."""
    import copy
    import torch
    enc_c, head_c = copy.deepcopy(enc), copy.deepcopy(head)
    params = list(enc_c.parameters()) + list(head_c.parameters())
    for _ in range(steps):
        zs, _, _, _ = z_standardize_t(enc_c(Xs), enc_c(Xs[:1]))
        loss = torch.mean((head_c(zs).squeeze(1) - ys_z) ** 2)
        g = torch.autograd.grad(loss, params)
        with torch.no_grad():
            for p, gr in zip(params, g):
                p.sub_(lr_in * gr)
    zq = enc_c(Xq)
    with torch.no_grad():
        zs = enc_c(Xs)
        mu, sd = zs.mean(0, keepdim=True), zs.std(0, keepdim=True) + 1e-8
    loss_q = torch.mean((head_c((zq - mu) / sd).squeeze(1) - yq_z) ** 2)
    gq = torch.autograd.grad(loss_q, params)
    return float(loss_q.item()), gq  # (query loss value, meta-gradient)


def episode_loss_proto(enc, log_tau, Xs, ys_z, Xq, yq_z):
    """ProtoNet regression (meta-trained): the encoder is trained via query-set
    kernel-regression MSE, **without an inner loop**.

    Same "frozen-encoder family" as ANIL — the only difference is the head:
    ANIL uses a learnable linear head (inner-loop adaptation), whereas ProtoNet
    uses a nonparametric kernel-regression readout (no head parameters to
    adapt, hence no inner loop). Embeddings are standardized with support-set
    statistics (consistent with DKL/ANIL); the kernel bandwidth tau is a
    learnable meta-parameter (log_tau, analogous to DKL's log_hyp), jointly
    optimized with the encoder during training.
    At evaluation, tau is instead selected by support-set LOO (see
    run_p1c_part.eval_episode_proto), bit-identical to the frozen ProtoNet
    readout — the only difference is the encoder (SSL-initialized vs
    meta-trained).

    This is the regression counterpart of Garzón 2025 "Prototypical Networks"
    (classification prototypes): episodic encoder training with query-loss
    backpropagation and no per-task inner loop.
    """
    import torch
    zs, zq, _, _ = z_standardize_t(enc(Xs), enc(Xq))
    D = ((zq[:, None, :] - zs[None, :, :]) ** 2).sum(-1)  # (Q,K) squared distances
    W = torch.softmax(-D / log_tau.exp().clamp(min=1e-3), dim=1)
    pred = W @ ys_z
    return torch.mean((pred - yq_z) ** 2)


# ---------------- meta-training main loop ----------------
def meta_train(method, init, source, n_episodes=2500, K=10, Q=16, lr=1e-3,
               lr_in=0.01, inner_steps=5, seed=0, val_every=250, patience=4,
               tag_suffix="", probe_fn=None, probe_every=100):
    import torch

    rng = np.random.default_rng(seed)
    pool = task_pool(source)
    data = {t: load_task(t) for t in pool}
    enc = build_encoder(init)
    head = torch.nn.Linear(EMB_DIM, 1)
    log_hyp = torch.nn.Parameter(torch.log(torch.tensor([1.0, 1.0, 0.1])))
    log_tau = torch.nn.Parameter(torch.tensor(0.0))  # ProtoNet regression kernel bandwidth (proto only)

    _suf = tag_suffix or ""
    ckpt_file = CKPT_DIR / f"{method}__{init}__{source}{_suf}.pt"
    log_file = CKPT_DIR / f"trainlog__{method}__{init}__{source}{_suf}.csv"
    ep0, best_val, best_state, bad = 0, np.inf, None, 0
    if method == "proto":
        params = list(enc.parameters()) + [log_tau]
    elif method == "dkl":
        params = list(enc.parameters()) + [log_hyp]
    else:
        params = list(enc.parameters()) + list(head.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    if ckpt_file.exists():
        ck = torch.load(ckpt_file, map_location="cpu", weights_only=False)
        enc.load_state_dict(ck["enc"])
        if method == "dkl":
            log_hyp.data = ck["log_hyp"]
        elif method == "proto":
            log_tau.data = ck["log_tau"]
        else:
            head.load_state_dict(ck["head"])
        opt.load_state_dict(ck["opt"])
        ep0, best_val = ck["epoch"], ck["best_val"]
        best_state, bad = ck.get("best_state"), ck.get("bad", 0)
        print(f"[resume] {method}/{init}/{source} from ep {ep0} "
              f"(best_val {best_val:.4f})", flush=True)

    Xt = {t: (torch.from_numpy(d["X"])[:, None], torch.from_numpy(d["y"]).float())
          for t, d in data.items()}

    def run_episode(t, rep, shot=None, train_mode=True):
        X, y = Xt[t]
        rr = np.random.default_rng(seed * 7919 + rep)
        K_ = shot or K
        si, qi = sample_train_episode(rr, X, y, K_, Q)
        ym, ysd = zscore_fit(y[si].numpy())
        ys_z = (y[si] - ym) / ysd
        yq_z = (y[qi] - ym) / ysd
        if method == "proto":
            return episode_loss_proto(enc, log_tau, X[si], ys_z, X[qi], yq_z)
        if method == "dkl":
            return episode_loss_dkl(enc, log_hyp, X[si], ys_z, X[qi], yq_z)
        if method == "anil":
            return episode_loss_anil(enc, head, X[si], ys_z, X[qi], yq_z,
                                     lr_in, inner_steps)
        return episode_loss_fomaml(enc, head, X[si], ys_z, X[qi], yq_z,
                                   lr_in, inner_steps)

    t0 = time.time()
    for ep in range(ep0 + 1, ep0 + n_episodes + 1):
        enc.train()
        t = pool[int(rng.integers(len(pool)))]
        if method == "fomaml":
            loss_val, gq = run_episode(t, rep=ep)
            opt.zero_grad()
            meta_params = list(enc.parameters()) + list(head.parameters())
            if (not np.isfinite(loss_val) or loss_val > 50
                    or any((not torch.isfinite(gr).all()) for gr in gq)):
                print(f"  [skip] ep {ep}: anomalous episode (loss {loss_val}); skipped",
                      flush=True)
                continue
            for p, gr in zip(meta_params, gq):
                p.grad = gr.clone()
            torch.nn.utils.clip_grad_norm_(meta_params, 5.0)
            opt.step()
        else:
            opt.zero_grad()
            loss = run_episode(t, rep=ep)
            loss_val = float(loss.item())
            if not np.isfinite(loss_val) or loss_val > 50:
                print(f"  [skip] ep {ep}: anomalous episode loss {loss_val:.1f}; skipped",
                      flush=True)
                opt.zero_grad()
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
        if ep % 100 == 0:
            print(f"[{method}/{init}/{source}] ep {ep}: loss={loss_val:.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
            with open(log_file, "a") as f:
                f.write(f"{ep},{loss_val:.6f}\n")
        if probe_fn is not None and ep % probe_every == 0:
            probe_fn(enc, ep)
        if ep % val_every == 0 or ep == ep0 + n_episodes:
            enc.eval()
            vals = []
            # anil/fomaml episodes contain autograd.grad and cannot run under no_grad
            ctx = torch.no_grad() if method in ("dkl", "proto") else torch.enable_grad()
            with ctx:
                for vr in range(40):
                    t = pool[int(rng.integers(len(pool)))]
                    r = run_episode(t, rep=10 ** 6 + ep * 100 + vr)
                    vals.append(float(r[0] if method == "fomaml" else r))
            va = float(np.nanmean(vals))
            print(f"  [val] ep {ep}: query_mse={va:.4f} (best {best_val:.4f})",
                  flush=True)
            if va < best_val:
                best_val, bad = va, 0
                best_state = dict(enc={k: v.clone() for k, v in enc.state_dict().items()},
                                  head={k: v.clone() for k, v in head.state_dict().items()},
                                  log_hyp=log_hyp.data.clone(),
                                  log_tau=log_tau.data.clone())
            else:
                bad += 1
                if bad >= patience:
                    print(f"  early stopping at ep {ep}", flush=True)
                    break
        if time.time() - t0 > 240:  # per-call safety bound: checkpoint and exit (resumable)
            print(f"  time budget reached; checkpointed at ep {ep}", flush=True)
            break
    if best_state is not None:
        enc.load_state_dict(best_state["enc"])
        head.load_state_dict(best_state["head"])
        log_hyp.data = best_state["log_hyp"]
        log_tau.data = best_state["log_tau"]
    torch.save(dict(enc=enc.state_dict(), head=head.state_dict(),
                    log_hyp=log_hyp.data, log_tau=log_tau.data, opt=opt.state_dict(),
                    epoch=ep, best_val=best_val, best_state=best_state, bad=bad,
                    method=method, init=init, source=source, K=K, Q=Q,
                    lr_in=lr_in, inner_steps=inner_steps), ckpt_file)
    print(f"saved -> {ckpt_file} (ep {ep}, best_val {best_val:.4f})", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True,
                    choices=["dkl", "anil", "fomaml", "proto"])
    ap.add_argument("--init", required=True, choices=["rand", "mae"])
    ap.add_argument("--source", required=True, choices=["multi", "diesel"])
    ap.add_argument("--episodes", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    meta_train(args.method, args.init, args.source, n_episodes=args.episodes,
               seed=args.seed)
