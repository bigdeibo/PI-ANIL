"""Self-supervised spectral pretraining: MAE-style masked band reconstruction
+ SimCLR contrastive learning (dual tracks).

- The backbone reuses src/fsl/models.ResNet1Encoder (in_len=512, emb_dim=128)
- MAE track: random contiguous band masks (25-40% total coverage); an encoder
  plus a lightweight decoder reconstructs the normalized spectrum, with the
  loss computed only at masked positions (MSE)
- SimCLR track: two physically perturbed views of the same spectrum
  (mild parameters of src/simulate.perturb) form a positive pair;
  NT-Xent (tau=0.1)
- Pretraining corpus: data/pretrain_corpus/spectra.npy (per-spectrum SNV
  normalization)
- checkpoint: results/pretraining/checkpoints/encoder_{track}.pt (includes
  optimizer state; training can be resumed)

CLI (chunked training; keep each invocation within a few minutes):
  python3 -m src.ssl.train --track mae --epochs 15
  python3 -m src.ssl.train --track simclr --epochs 15
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

L = 512
EMB_DIM = 128
CKPT_DIR = ROOT / "results" / "pretraining" / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------- Data ----------------
def load_corpus():
    """Load the corpus and apply per-spectrum SNV normalization.
    Returns (N, L) float32."""
    X = np.load(ROOT / "data" / "pretrain_corpus" / "spectra.npy").astype(np.float32)
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True) + 1e-8
    return (X - mu) / sd


def snv(x):
    x = np.asarray(x, dtype=np.float32)
    return (x - x.mean(axis=-1, keepdims=True)) / (x.std(axis=-1, keepdims=True) + 1e-8)


def band_mask(rng, batch, L, total=(0.25, 0.40), n_seg=(3, 6)):
    """Random contiguous band mask. Returns (B, L) bool, True=masked."""
    B = batch
    m = np.zeros((B, L), dtype=bool)
    for b in range(B):
        cover = rng.uniform(*total)
        k = int(rng.integers(n_seg[0], n_seg[1] + 1))
        seg = int(L * cover / k)
        for _ in range(k):
            s = int(rng.integers(0, max(1, L - seg)))
            m[b, s:s + seg] = True
    return m


# ---------------- Models ----------------
def build_models(track, arch="resnet1", emb_dim=None):
    """Return (encoder, aux_net). mae: aux=decoder; simclr: aux=projection head.
    arch selects the backbone (resnet1 legacy 128-dim / resnet1d_garzon
    512-dim); emb_dim defaults to EMB_DIM; the aux dimensions follow emb_dim."""
    import torch.nn as nn
    from src.fsl.models import ENCODERS, ResNet1Encoder
    emb_dim = EMB_DIM if emb_dim is None else emb_dim
    cls = ENCODERS.get(arch, ResNet1Encoder)
    enc = cls(L, emb_dim=emb_dim)
    if track == "mae":
        aux = nn.Sequential(nn.Linear(emb_dim, 256), nn.ReLU(), nn.Linear(256, L))
    else:
        aux = nn.Sequential(nn.Linear(emb_dim, 128), nn.ReLU(), nn.Linear(128, 64))
    return enc, aux


# ---------------- Training ----------------
def train(track, epochs, batch=128, lr=1e-3, seed=0, log_every=5,
          suffix="", scratch=False, arch="resnet1", emb_dim=None, device="cpu"):
    import torch
    from src.simulate import perturb

    if scratch:
        torch.manual_seed(seed)  # train from scratch: fix the initialization seed (not needed when resuming, since weights come from the ckpt)
    rng = np.random.default_rng(seed)
    X = load_corpus()
    emb_dim = EMB_DIM if emb_dim is None else emb_dim
    enc, aux = build_models(track, arch=arch, emb_dim=emb_dim)
    enc = enc.to(device); aux = aux.to(device)
    ckpt_file = CKPT_DIR / f"encoder_{track}{suffix}.pt"
    ep0 = 0
    opt = torch.optim.Adam(list(enc.parameters()) + list(aux.parameters()), lr=lr)
    if not scratch:
        # Resume: prefer continuing from the save file; otherwise pick up from
        # the original encoder_{track}.pt (keeping the original ckpt untouched)
        resume_file = (ckpt_file if ckpt_file.exists()
                       else CKPT_DIR / f"encoder_{track}.pt")
        if resume_file.exists():
            ck = torch.load(resume_file, map_location=device, weights_only=False)
            enc.load_state_dict(ck["enc"]); aux.load_state_dict(ck["aux"])
            opt.load_state_dict(ck["opt"]); ep0 = ck["epoch"]
            print(f"[resume] from {resume_file.name} epoch {ep0}", flush=True)
    Xt = torch.from_numpy(X).to(device)
    N = len(X)
    log_file = CKPT_DIR / f"trainlog_{track}{suffix}.csv"
    for ep in range(ep0 + 1, ep0 + epochs + 1):
        t0 = time.time()
        perm = torch.randperm(N)
        tot, nb = 0.0, 0
        enc.train(); aux.train()
        for i in range(0, N, batch):
            idx = perm[i:i + batch]
            xb = Xt[idx][:, None]  # (B,1,L)
            opt.zero_grad()
            if track == "mae":
                m = torch.from_numpy(band_mask(rng, len(idx), L)).to(device)
                xm = xb.clone()
                xm = xm * (~m)[:, None].float()  # zero out masked positions
                pred = aux(enc(xm))  # (B,L)
                loss = torch.mean(((pred - xb[:, 0]) ** 2) * m.float()) \
                    / (m.float().mean() + 1e-8)
            else:
                from src.simulate import perturb_batch
                xn = Xt[idx].cpu().numpy()
                v1 = torch.from_numpy(snv(perturb_batch(rng, xn))).to(device)[:, None]
                v2 = torch.from_numpy(snv(perturb_batch(rng, xn))).to(device)[:, None]
                z = torch.cat([aux(enc(v1)), aux(enc(v2))])
                z = torch.nn.functional.normalize(z, dim=1)
                B = len(idx)
                sim = z @ z.T / 0.1
                sim.fill_diagonal_(-1e9)
                pos = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)]).to(device)
                loss = torch.nn.functional.cross_entropy(sim, pos)
            loss.backward()
            opt.step()
            tot += float(loss.item()); nb += 1
        msg = f"[{track}] ep {ep}: loss={tot / nb:.4f} ({time.time() - t0:.1f}s)"
        print(msg, flush=True)
        with open(log_file, "a") as f:
            f.write(f"{ep},{tot / nb:.6f},{time.time() - t0:.1f}\n")
        torch.save(dict(enc=enc.state_dict(), aux=aux.state_dict(),
                        opt=opt.state_dict(), epoch=ep, track=track,
                        arch=arch, emb_dim=emb_dim, in_len=L), ckpt_file)
    print(f"saved -> {ckpt_file} (epoch {ep0 + epochs})")


def load_encoder(track, arch="resnet1"):
    """Load the pretrained encoder (eval mode). Reused for linear
    probing/fine-tuning.

    If the ckpt contains an arch field, rebuild the corresponding backbone
    accordingly (compatible with the newer garzon/512 ckpts); otherwise use
    the passed arch (default resnet1, backward compatible with legacy 128-dim
    ckpts).
    """
    import torch
    from src.fsl.models import ENCODERS, ResNet1Encoder
    ck = torch.load(CKPT_DIR / f"encoder_{track}.pt", map_location="cpu",
                    weights_only=False)
    a = ck.get("arch", arch)
    cls = ENCODERS.get(a, ResNet1Encoder)
    enc = cls(ck["in_len"], emb_dim=ck["emb_dim"])
    enc.load_state_dict(ck["enc"])
    return enc.eval()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", required=True, choices=["mae", "simclr"])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--suffix", default="",
                    help="checkpoint/log suffix (e.g. _cont60); keeps the original encoder_{track}.pt")
    ap.add_argument("--scratch", action="store_true",
                    help="train from scratch (no resume; fixed-seed initialization)")
    ap.add_argument("--arch", default="resnet1",
                    help="backbone registry key (use resnet1d_garzon for exact)")
    ap.add_argument("--emb-dim", type=int, default=None, help="embedding dimension (exact=512)")
    ap.add_argument("--device", default="cpu", help="cpu / cuda")
    args = ap.parse_args()
    train(args.track, args.epochs, batch=args.batch, seed=args.seed,
          suffix=args.suffix, scratch=args.scratch, arch=args.arch,
          emb_dim=args.emb_dim, device=args.device)
