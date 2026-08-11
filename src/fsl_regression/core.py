"""Shared components: regression task library, paired episode protocol, GP/kNN regression heads.

13 "material x property" quantitative tasks: diesel 7 properties + gasoline
octane + corn 4 properties (m5 instrument) + EVOO adulteration. All tasks are
unified as (SNV spectra, resampled to 512, raw y).

Paired protocol: exactly the same seeding scheme as Stages 0/2
(seed*1000+K*100+rep, rng.choice(n,K) + setdiff), ensuring that all methods
across stages are compared on identical support/test splits.

Meta-learning episode conventions: y is z-scored with support-set statistics
(stabilizes training); embeddings z are standardized with support-set
statistics (makes the GP length scale / inner-loop learning rate comparable
across tasks).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

L = 512
EMB_DIM = 128
BASE_SEED = 42

DIESEL_ATTRS = ("CN", "BP50", "D4052", "FLASH", "FREEZE", "TOTAL", "VISC")
CORN_ATTRS = ("moisture", "oil", "protein", "starch")
TASKS = dict(
    **{f"diesel-{a}": ("diesel", a) for a in DIESEL_ATTRS},
    **{"gasoline-octane": ("gasoline", "octane")},
    **{f"corn_m5-{a}": ("corn", a) for a in CORN_ATTRS},
    **{"evoo-adulteration": ("evoo", "adulteration_level")},
    **{"selfmix-phi": ("selfmix", "phi_pentane")},  # self-collected FT-NIR pentane/CCl4
    **{"edibleoil-pv": ("edibleoil", "peroxide_value")},  # edible-oil FT-NIR peroxide value (Lavine/Booksh)
)
DIESEL_TASKS = [f"diesel-{a}" for a in DIESEL_ATTRS]
HEADLINE_TASKS = DIESEL_TASKS + ["gasoline-octane"]  # headline tasks defined by the technical roadmap

_cache: dict = {}


def load_task(task):
    """Return dict(X=(n,512) float32 SNV+resampled, y=(n,) float64 raw scale,
    groups=(n,) or None).

    groups: only provided for EVOO as sample_id (aligned to non-NaN rows) — EVOO
    consists of NIR-HSI replicate spectra (641 physical samples x ~3 spectra);
    a group-aware split by sample_id prevents replicate cross-boundary leakage
    between support and test (see the groups mode of eval_split). All other
    tasks use groups=None -> the original random split.
    """
    if task in _cache:
        return _cache[task]
    from src.datasets import LOADERS
    from src.ssl import snv
    ds, attr = TASKS[task]
    d = LOADERS[ds]()
    groups = None
    if ds == "corn":
        X = d["instruments"]["m5"][0]
        y = d["targets"][attr]
    else:
        X, y = d["X"], d["targets"][attr]
    m = ~np.isnan(y)
    X, y = np.asarray(X, float)[m], np.asarray(y, float)[m]
    if ds == "evoo":
        groups = np.asarray(d["ids"])[m]  # sample_id aligned to non-NaN rows
    elif ds == "selfmix":
        groups = np.asarray(d["groups"])[m]  # bottle-id sample_id (9 spectra per bottle = replicates; group-aware against leakage)
    elif ds == "edibleoil":
        groups = np.asarray(d["groups"])[m]  # oil identity (class, PV): 100 oils x 3 replicates; group-aware against leakage
    go = np.linspace(0, 1, X.shape[1])
    gn = np.linspace(0, 1, L)
    X = np.stack([np.interp(gn, go, r) for r in X])
    if ds == "selfmix":
        # selfmix uses raw absorbance (512-point interpolation only, no SNV):
        # Beer-Lambert additivity is the generative model of this task (mixture
        # fit R^2~0.99), whereas per-spectrum SNV demeaning would erase the
        # additive-offset signal.
        # Note: the distribution differs from the 13 tasks (SNV) and the MAE
        # encoder (trained on SNV) — during meta-training the encoder is
        # fine-tuned within the pool containing raw selfmix; the C-H band_mask
        # still applies by wavelength.
        out = dict(X=X.astype(np.float32), y=y, groups=groups)
    else:
        out = dict(X=snv(X).astype(np.float32), y=y, groups=groups)
    _cache[task] = out
    return out


def eval_split(n, K, rep, seed=BASE_SEED, groups=None):
    """Paired evaluation split.

    groups=None (default) -> random split, bit-identical to Stages 0/2.
    groups given (EVOO sample_id) -> **group-aware split**: the K support
    spectra are drawn from K distinct physical samples (one spectrum per
    group), and the test set excludes all members of those K support groups
    (including their replicate spectra), preventing replicate cross-boundary
    leakage between support and test. The group permutation and within-group
    selection reuse the same seed formula (seed*1000+K*100+rep) -> splits
    remain paired bit-identical across methods/stages for EVOO; the path for
    non-EVOO tasks (groups=None) is completely unchanged.
    """
    rng = np.random.default_rng(seed * 1000 + K * 100 + rep)
    if groups is None:
        tr = rng.choice(n, K, replace=False)
        te = np.setdiff1d(np.arange(n), tr)
        return tr, te
    groups = np.asarray(groups)
    _, inv = np.unique(groups, return_inverse=True)  # inv[i] = group id (0..G-1) of row i
    n_groups = int(inv.max()) + 1
    gperm = rng.permutation(n_groups)
    sup_gids = gperm[:K]
    tr = []
    for g in sup_gids:
        members = np.where(inv == g)[0]
        rng.shuffle(members)
        tr.append(int(members[0]))          # take one spectrum per support group
    tr = np.array(tr, dtype=int)
    sup_set = set(int(g) for g in sup_gids)
    # test = spectra of all non-support groups (the remaining replicate spectra of
    # support groups are excluded as well, preventing cross-boundary leakage)
    te_mask = np.array([int(i) not in sup_set for i in inv])
    te = np.where(te_mask)[0]
    return tr, te


def corn_holdout_split(n=80, n_holdout=30, seed=42):
    """Corn cross-instrument holdout split (cross-instrument Option B): deterministic
    holdout by **physical-sample original index**.

    Corn is a fully balanced design (80 samples x 3 instruments x 4 properties,
    0 NaN), so the original [0,n) indices align consistently across properties
    and instruments — the holdout selects the same physical samples.
    Returns (train_idx, holdout_idx):
      - train_idx: corn samples entering meta-training (corn tasks use only
        these, on the source instrument m5);
      - holdout_idx: **never enters meta-training in any form**, used only for
        target-instrument evaluation (leak-proof).
    Fixed seed -> meta-training and evaluation share the same split; non-Option-B
    paths never call this function (canonical behavior unchanged).
    """
    rng = np.random.default_rng(seed)
    ho = np.sort(rng.choice(n, n_holdout, replace=False))
    tr = np.setdiff1d(np.arange(n), ho)
    return tr, ho


def sample_train_episode(rng, X, y, K, Q):
    """Meta-training episode: K+Q random samples; first K as support, last Q as query."""
    idx = rng.choice(len(y), K + Q, replace=False)
    return idx[:K], idx[K:]


def zscore_fit(ys):
    ym, ysd = float(np.mean(ys)), float(np.std(ys) + 1e-8)
    return ym, ysd


def z_standardize(zs, zq=None):
    """Standardize embeddings with support-set statistics. Returns (zs', zq' or None, mu, sd)."""
    mu = zs.mean(0)
    sd = zs.std(0) + 1e-8
    if zq is None:
        return (zs - mu) / sd, None, mu, sd
    return (zs - mu) / sd, (zq - mu) / sd, mu, sd


def r2_score(pred, yt):
    ss_res = float(np.sum((pred - yt) ** 2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    return 1 - ss_res / ss_tot if ss_tot > 1e-12 else np.nan


# ---------------- torch GP regression head ----------------
def gp_predict(zs, ys, zq, log_hyp, jitter=1e-4):
    """RBF-kernel GP posterior mean (differentiable). log_hyp = log([lengthscale, outputscale, noise]).

    zs: (K,d) support embeddings (standardized); ys: (K,) z-scored targets; zq: (Q,d).
    Returns (Q,) predictions (z-score scale).
    """
    import torch
    ls, os_, noise = log_hyp.exp()
    K = os_ * torch.exp(-0.5 * torch.cdist(zs, zs) ** 2 / ls ** 2)
    K = K + (noise + jitter) * torch.eye(len(zs), dtype=zs.dtype)
    Lc = torch.linalg.cholesky(K)
    alpha = torch.cholesky_solve(ys.unsqueeze(1), Lc)
    Kq = os_ * torch.exp(-0.5 * torch.cdist(zq, zs) ** 2 / ls ** 2)
    return (Kq @ alpha).squeeze(1)


def gp_marginal_likelihood(zs, ys, log_hyp, jitter=1e-4):
    """GP log marginal likelihood (differentiable, for support-set hyperparameter optimization)."""
    import torch
    ls, os_, noise = log_hyp.exp()
    K = os_ * torch.exp(-0.5 * torch.cdist(zs, zs) ** 2 / ls ** 2)
    K = K + (noise + jitter) * torch.eye(len(zs), dtype=zs.dtype)
    Lc = torch.linalg.cholesky(K)
    alpha = torch.cholesky_solve(ys.unsqueeze(1), Lc)
    n = len(zs)
    return (-0.5 * ys @ alpha.squeeze(1)
            - torch.log(torch.diagonal(Lc)).sum()
            - 0.5 * n * np.log(2 * np.pi))


# ---------------- frozen embedding + kNN / GP (lightweight Route B) ----------------
def knn_predict(zs, ys, zq, k_grid=(1, 3, 5, 7)):
    """Distance-weighted kNN regression; k selected internally by support-set LOO."""
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.model_selection import LeaveOneOut, cross_val_score
    best_k, best_s = 3, -np.inf
    for k in k_grid:
        if k >= len(zs):
            continue
        m = KNeighborsRegressor(n_neighbors=k, weights="distance")
        s = cross_val_score(m, zs, ys, cv=LeaveOneOut(),
                            scoring="neg_mean_squared_error").mean()
        if s > best_s:
            best_s, best_k = s, k
    return KNeighborsRegressor(n_neighbors=best_k, weights="distance").fit(zs, ys).predict(zq)


def gp_frozen_predict(zs, ys, zq, opt_steps=30):
    """Frozen embedding + GP: optimize hyperparameters on the support set
    (marginal likelihood, 30 Adam steps), then predict."""
    import torch
    zs_t = torch.tensor(zs, dtype=torch.float32)
    ys_t = torch.tensor(ys, dtype=torch.float32)
    zq_t = torch.tensor(zq, dtype=torch.float32)
    log_hyp = torch.nn.Parameter(torch.log(torch.tensor([1.0, 1.0, 0.1])))
    opt = torch.optim.Adam([log_hyp], lr=0.05)
    for _ in range(opt_steps):
        opt.zero_grad()
        (-gp_marginal_likelihood(zs_t, ys_t, log_hyp)).backward()
        opt.step()
    with torch.no_grad():
        return gp_predict(zs_t, ys_t, zq_t, log_hyp).numpy()
