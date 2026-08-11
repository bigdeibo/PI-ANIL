"""1D spectral encoders and metric-learning few-shot models (encoders reused for quantitative meta-learning).

- Conv1Encoder:   4-layer 1D-CNN + global average pooling (~0.2M params, CPU-friendly)
- ResNet1Encoder: 1D ResNet (stem + 3 residual stages, ~0.35M params)
- ProtoNet:       Euclidean-distance prototypical classification head (Snell et al., 2017)
- train_protonet: episode meta-training loop (Adam + early stopping on validation episodes)

Constraints: encoder parameter count <= 2M; everything is trainable on CPU.
Reuse: encoders output embeddings z=E(x); the ProtoNet head can be
replaced with a support-set regression readout.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- encoders ----------
class Conv1Encoder(nn.Module):
    """4-layer 1D-CNN encoder. Input (B, 1, L), output (B, emb_dim)."""

    def __init__(self, in_len, emb_dim=64, width=32):
        super().__init__()
        w = width
        self.features = nn.Sequential(
            nn.Conv1d(1, w, 25, stride=2, padding=12), nn.BatchNorm1d(w), nn.ReLU(),
            nn.Conv1d(w, 2 * w, 15, stride=2, padding=7), nn.BatchNorm1d(2 * w), nn.ReLU(),
            nn.Conv1d(2 * w, 4 * w, 9, stride=2, padding=4), nn.BatchNorm1d(4 * w), nn.ReLU(),
            nn.Conv1d(4 * w, 4 * w, 5, padding=2), nn.BatchNorm1d(4 * w), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
        )
        self.proj = nn.Linear(4 * w, emb_dim)
        self.out_dim = emb_dim

    def forward(self, x):
        return self.proj(self.features(x))


class ResBlock1D(nn.Module):
    """Residual block with two 9-kernel convolutions (optional downsampling)."""

    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(cin, cout, 9, stride=stride, padding=4, bias=False)
        self.bn1 = nn.BatchNorm1d(cout)
        self.conv2 = nn.Conv1d(cout, cout, 9, padding=4, bias=False)
        self.bn2 = nn.BatchNorm1d(cout)
        self.short = None
        if stride != 1 or cin != cout:
            self.short = nn.Sequential(
                nn.Conv1d(cin, cout, 1, stride=stride, bias=False),
                nn.BatchNorm1d(cout))

    def forward(self, x):
        idt = x if self.short is None else self.short(x)
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return F.relu(h + idt)


class ResNet1Encoder(nn.Module):
    """1D ResNet encoder. Input (B, 1, L), output (B, emb_dim)."""

    def __init__(self, in_len, emb_dim=64, width=32):
        super().__init__()
        w = width
        self.stem = nn.Sequential(
            nn.Conv1d(1, w, 25, stride=2, padding=12, bias=False),
            nn.BatchNorm1d(w), nn.ReLU())
        self.stage1 = ResBlock1D(w, w)
        self.stage2 = ResBlock1D(w, 2 * w, stride=2)
        self.stage3 = ResBlock1D(2 * w, 4 * w, stride=2)
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten())
        self.proj = nn.Linear(4 * w, emb_dim)
        self.out_dim = emb_dim

    def forward(self, x):
        h = self.stem(x)
        h = self.stage3(self.stage2(self.stage1(h)))
        return self.proj(self.head(h))


# ---------- Garzón exact backbone (Appendix Table B.1, 1D ResNet-18, ~11M params) ----------
class BasicBlock1D(nn.Module):
    """Garzón ResNet1D basic residual block (Appendix B.1): Conv1d(k3)+BN+ReLU+Conv1d(k3)+BN
    + identity/1x1-conv shortcut. Isomorphic to the torchvision BasicBlock (1D version)."""

    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(cin, cout, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(cout)
        self.conv2 = nn.Conv1d(cout, cout, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(cout)
        self.short = None
        if stride != 1 or cin != cout:
            self.short = nn.Sequential(
                nn.Conv1d(cin, cout, 1, stride=stride, bias=False),
                nn.BatchNorm1d(cout))

    def forward(self, x):
        idt = x if self.short is None else self.short(x)
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return F.relu(h + idt)


class ResNet1DGarzon(nn.Module):
    """Garzón 1D ResNet-18 encoder (Appendix Table B.1, bit-match level).

    stem: Conv1d(1->64, k7, s2, pad3)+BN+ReLU -> MaxPool1d(k3, s2, pad1)
    4 residual stages [2,2,2,2], channels 64/128/256/512, first block stride=2
    (except stage1)
    AdaptiveAvgPool1d(1) -> (B, 512).

    Input (B, 1, L), output (B, emb_dim). With emb_dim=512, proj=Identity
    (strictly following Garzón: the regression head Linear(512->1) is built
    separately in metatrain as `head`; ProtoNet uses the 512-d embedding
    directly). With emb_dim!=512, a Linear(512->emb_dim) is appended.
    """

    def __init__(self, in_len, emb_dim=512):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(3, stride=2, padding=1))
        self.stage1 = self._make_stage(64, 64, 2, stride=1)
        self.stage2 = self._make_stage(64, 128, 2, stride=2)
        self.stage3 = self._make_stage(128, 256, 2, stride=2)
        self.stage4 = self._make_stage(256, 512, 2, stride=2)
        self.pool = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten())
        self.proj = nn.Identity() if emb_dim == 512 else nn.Linear(512, emb_dim)
        self.out_dim = emb_dim

    @staticmethod
    def _make_stage(cin, cout, n_blocks, stride):
        blocks = [BasicBlock1D(cin, cout, stride=stride)]
        for _ in range(1, n_blocks):
            blocks.append(BasicBlock1D(cout, cout, stride=1))
        return nn.Sequential(*blocks)

    def forward(self, x):
        h = self.stem(x)
        h = self.stage4(self.stage3(self.stage2(self.stage1(h))))
        return self.proj(self.pool(h))


ENCODERS = dict(conv1=Conv1Encoder, resnet1=ResNet1Encoder,
                resnet1d_garzon=ResNet1DGarzon)


def count_params(net):
    """Number of trainable parameters."""
    return int(sum(p.numel() for p in net.parameters() if p.requires_grad))


# ---------- ProtoNet ----------
class ProtoNet(nn.Module):
    """ProtoNet (Snell 2017): support-set class means as prototypes; query samples
    are classified by negative squared Euclidean distance."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, xs, ys, xq):
        """
        xs: (N*K, 1, L) support set; ys: (N*K,) within-episode labels 0..N-1;
        xq: (N*Q, 1, L) query set.
        Returns logits: (N*Q, N); larger means more likely to belong to the class.
        """
        zs = self.encoder(xs)
        classes = torch.unique(ys)
        protos = torch.stack([zs[ys == c].mean(dim=0) for c in classes])
        zq = self.encoder(xq)
        d2 = torch.cdist(zq, protos) ** 2
        return -d2


# ---------- utilities ----------
def support_standardize(Xs, Xq):
    """Feature standardization using support-set statistics only (leakage-free).
    Returns (Xs', Xq')."""
    m = Xs.mean(axis=0)
    s = Xs.std(axis=0) + 1e-8
    return (Xs - m) / s, (Xq - m) / s


# ---------- meta-training ----------
@torch.no_grad()
def protonet_episode_acc(model, Xs, ys, Xq, yq):
    """Query-set accuracy of a single episode (model in eval mode)."""
    logits = model(torch.from_numpy(Xs[:, None]).float(),
                   torch.from_numpy(ys).long(),
                   torch.from_numpy(Xq[:, None]).float())
    return float((logits.argmax(dim=1).numpy() == yq).mean())


def train_protonet(model, sampler, X_snv, n_episodes=3000, lr=1e-3,
                   weight_decay=1e-4, eval_every=200, eval_episodes=100,
                   patience=5, seed=0, verbose=True):
    """ProtoNet episode meta-training loop (CPU).

    Parameters
    ----------
    model : ProtoNet
    sampler : EpisodeSampler (samples within the train-side class set; k_shot is
        the training shot)
    X_snv : SNV-preprocessed spectral matrix (per-sample, leakage-free)
    n_episodes : maximum number of episodes (recommended <=5000)
    eval_every / eval_episodes : every eval_every episodes, evaluate accuracy on
        eval_episodes validation episodes (rep offset 10**6, disjoint from
        training episodes)
    patience : early stopping after `patience` consecutive evaluations without
        validation-accuracy improvement

    Returns history: dict(train_loss=[(ep, loss)], val=[(ep, acc)], best_ep, best_acc).
    After training, the model is rolled back to the best-validation weights.
    """
    torch.manual_seed(seed)
    X = np.asarray(X_snv, dtype=np.float32)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    hist = dict(train_loss=[], val=[], best_ep=0, best_acc=-1.0)
    best_state, bad = None, 0
    K, Q = sampler.k_shot, sampler.q_query
    for ep in range(1, n_episodes + 1):
        sup, qry, ep_cls = sampler.sample(split="train", rep=ep)
        ys = np.repeat(np.arange(len(ep_cls)), K)
        yq = np.repeat(np.arange(len(ep_cls)), Q)
        Xs, Xq = support_standardize(X[sup], X[qry])
        model.train()
        opt.zero_grad()
        logits = model(torch.from_numpy(Xs[:, None]),
                       torch.from_numpy(ys).long(),
                       torch.from_numpy(Xq[:, None]))
        loss = F.cross_entropy(logits, torch.from_numpy(yq).long())
        loss.backward()
        opt.step()
        hist["train_loss"].append((ep, float(loss.item())))
        if ep % eval_every == 0 or ep == n_episodes:
            model.eval()
            accs = []
            for sup, qry, ep_cls in sampler.iter_episodes(
                    eval_episodes, split="train", start_rep=10 ** 6 + ep):
                ys = np.repeat(np.arange(len(ep_cls)), K)
                yq = np.repeat(np.arange(len(ep_cls)), Q)
                Xs, Xq = support_standardize(X[sup], X[qry])
                accs.append(protonet_episode_acc(model, Xs, ys, Xq, yq))
            acc = float(np.mean(accs))
            hist["val"].append((ep, acc))
            if verbose:
                print(f"  [metatrain] ep {ep}: loss={loss.item():.4f} "
                      f"val_acc={acc:.4f}", flush=True)
            if acc > hist["best_acc"]:
                hist["best_acc"], hist["best_ep"] = acc, ep
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    print(f"  [metatrain] early stopping at ep {ep} (best ep {hist['best_ep']}, "
                          f"acc {hist['best_acc']:.4f})", flush=True)
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return hist
