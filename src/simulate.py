"""Simulated spectrum generator: linear mixing of pure spectra + a physical
perturbation chain.

Purpose: augmenting the pretraining corpus; directly reused for the Sim-to-Real boundary probes (the mixing ratios are also returned and can serve as
labels for simulated calibration tasks).

Physical model
--------------
1. Beer-Lambert additive mixing: S_mix = sum_i c_i * P_i, c ~ Dirichlet (2-4 components)
2. Baseline drift: low-order polynomial (order 2-3, random coefficients)
3. Multiplicative scatter: a + b * u, where u is the normalized abscissa
   (simulating particle-size / path-length differences)
4. Wavelength shift/stretch: affine perturbation of the abscissa
   x' = x + d + s*(x - x_mid) (simulating instrument wavelength-calibration error)
5. Resolution broadening: optional Gaussian smoothing (random sigma)
6. Additive noise: Gaussian noise added to meet a target SNR (dB)

All perturbation parameters are sampled from a seed-driven rng and are
reproducible.
Pure-component library sources: NIST IR pure compounds
(data/NIST_IR/nist_ir.npz) + class-mean spectra from local datasets
(7 EVOO classes, 6 mayonnaise classes, etc., built by build_pure_library).
"""
from __future__ import annotations

import numpy as np


def _poly_baseline(rng, n, deg=3, scale=0.3):
    """Random low-order polynomial baseline (on a [-1,1] normalized abscissa)."""
    u = np.linspace(-1, 1, n)
    coefs = rng.normal(0, scale, deg + 1)
    coefs[0] = rng.normal(0, scale * 0.5)  # keep the constant term smaller
    return np.polyval(coefs, u)


def perturb(rng, x, shift_pts=3.0, stretch=0.01, smooth_p=0.3,
            baseline_scale=0.15, scatter=(0.9, 1.1, 0.05), snr_db=(30, 60)):
    """Apply the physical perturbation chain to a single spectrum.
    x: (L,) spectrum already resampled to a common length."""
    L = len(x)
    y = x.astype(float).copy()
    # 4. Wavelength shift/stretch (linear-interpolation shift of the spectrum itself)
    d = rng.uniform(-shift_pts, shift_pts)
    s = rng.uniform(-stretch, stretch)
    idx = np.arange(L)
    new_idx = idx + d + s * (idx - L / 2)
    y = np.interp(idx, new_idx, y, left=y[0], right=y[-1])
    # 5. Resolution broadening
    if rng.random() < smooth_p:
        sigma = rng.uniform(0.5, 2.0)
        k = int(6 * sigma) | 1
        g = np.exp(-0.5 * ((np.arange(k) - k // 2) / sigma) ** 2)
        g /= g.sum()
        y = np.convolve(y, g, mode="same")
    # 3. Multiplicative scatter
    a = rng.uniform(*scatter[:2])
    b = rng.uniform(-scatter[2], scatter[2])
    u = np.linspace(-1, 1, L)
    y = y * (a + b * u)
    # 2. Baseline drift
    y = y + _poly_baseline(rng, L, deg=rng.integers(2, 4),
                           scale=baseline_scale) * (np.abs(y).max() + 1e-8)
    # 6. Additive noise
    snr = rng.uniform(*snr_db)
    sig = np.sqrt(np.mean(y ** 2)) / (10 ** (snr / 20))
    y = y + rng.normal(0, sig, L)
    return y


def build_pure_library(nist_npz=None, class_spectra=None):
    """Build the pure-component spectral library (all spectra resampled to a
    common length L before being returned).

    Parameters
    ----------
    nist_npz : Path | None, NIST IR npz (object arrays x/y)
    class_spectra : dict[str, np.ndarray] | None, {name: (n_i, L_i) spectral
        groups}; the mean is taken as the pure spectrum of that class
        (e.g., the 7 EVOO class means)

    Returns dict{name: (L,) spectrum}; individual spectra may have different
    lengths — they are resampled onto a common grid at mixing time.
    """
    lib = {}
    if nist_npz is not None:
        z = np.load(nist_npz, allow_pickle=True)
        for name, x, y in zip(z["names"], z["x"], z["y"]):
            if len(y) >= 100 and np.all(np.isfinite(y)):
                lib[f"nist:{name}"] = np.asarray(y, dtype=float)
    if class_spectra:
        for name, X in class_spectra.items():
            if len(X) >= 3:
                lib[f"cls:{name}"] = np.asarray(X, dtype=float).mean(axis=0)
    return lib


def perturb_batch(rng, X, shift_pts=2.0, scale=(0.9, 1.1), offset=0.05,
                  snr_db=(35, 60), poly_deg=3, poly_scale=0.08):
    """Vectorized batch perturbation (for SimCLR views; millisecond-level).
    X: (B, L) normalized spectra."""
    B, L = X.shape
    Y = X.astype(np.float32).copy()
    u = np.linspace(-1, 1, L, dtype=np.float32)
    for b in range(B):
        d = int(round(rng.uniform(-shift_pts, shift_pts)))
        if d:
            Y[b] = np.roll(Y[b], d)
            if d > 0:
                Y[b, :d] = Y[b, d]
            else:
                Y[b, d:] = Y[b, d - 1]
    Y = Y * rng.uniform(*scale, (B, 1)).astype(np.float32)
    coefs = rng.normal(0, poly_scale, (B, poly_deg + 1)).astype(np.float32)
    basis = np.stack([u ** k for k in range(poly_deg + 1)])  # (deg+1, L)
    Y = Y + coefs @ basis
    Y = Y + rng.uniform(-offset, offset, (B, 1)).astype(np.float32)
    snr = rng.uniform(*snr_db, B).astype(np.float32)
    sig = np.sqrt((Y ** 2).mean(axis=1)) / (10 ** (snr / 20))
    Y = Y + rng.normal(0, 1, (B, L)).astype(np.float32) * sig[:, None]
    return Y


def simulate_mixtures(lib, n_samples, L=512, seed=0, k_range=(2, 4),
                      do_perturb=True):
    """Generate simulated mixture spectra from the pure-spectrum library.

    Parameters
    ----------
    lib : dict{name: spectrum} (output of build_pure_library)
    n_samples : number of spectra to generate
    L : common output length (each pure spectrum is resampled to L before mixing)
    seed : random seed
    k_range : range of the number of components per sample
    do_perturb : whether to apply the physical perturbation chain
        (The boundary probes may disable this for clean mixtures)

    Returns (X_sim (n,L), comps (n,n_lib) mixing ratios, lib_names)
    """
    rng = np.random.default_rng(seed)
    names = sorted(lib)
    P = np.stack([np.interp(np.linspace(0, 1, L),
                            np.linspace(0, 1, len(lib[n])), lib[n])
                  for n in names])  # (n_lib, L)
    # Normalize each component to [0,1] so amplitude differences do not dominate the mixing ratios
    P = P - P.min(axis=1, keepdims=True)
    P = P / (P.max(axis=1, keepdims=True) + 1e-8)
    n_lib = len(names)
    X = np.zeros((n_samples, L))
    C = np.zeros((n_samples, n_lib))
    for i in range(n_samples):
        k = int(rng.integers(k_range[0], k_range[1] + 1))
        sel = rng.choice(n_lib, size=min(k, n_lib), replace=False)
        w = rng.dirichlet(np.ones(len(sel)))
        C[i, sel] = w
        y = w @ P[sel]
        if do_perturb:
            y = perturb(rng, y)
        X[i] = y
    return X, C, names
