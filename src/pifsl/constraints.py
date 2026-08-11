"""Physics-informed constraints module.

Two physics constraints, both designed as regularizers embeddable into the
ANIL meta-objective:

1. **Beer-Lambert additivity constraint (additivity_loss)**
   Mixture absorbance is approximately the concentration-weighted linear
   combination of component absorbances. The encoder embedding space is
   required to satisfy the same linear geometry:
   E(c*xa+(1-c)*xb) ~ c*E(xa)+(1-c)*E(xb).
   The relative error (scale-free) serves as the regularizer. Components are
   drawn from real spectra of the same task (mixing within a material family is
   the most physically meaningful).

2. **C-H band prior (band_saliency_loss)**
   Quantitative NIR information concentrates in known C-H overtone/combination
   bands (literature values):
   - second overtone C-H stretch:            1100-1250 nm
   - combination C-H:                        1350-1550 nm
   - first overtone C-H stretch:             1650-1800 nm
   - combination C-H (bend+stretch):         2100-2400 nm
   Corn tasks additionally include O-H water bands: 1400-1500 / 1900-2000 nm.
   Constraint form: the sensitivity of the model prediction to the input
   (saliency, dy/dx) should concentrate inside the prior bands, using the
   "out-of-band / in-band sensitivity ratio" (scale-free) as the regularizer.
   Masks are constructed on each dataset's **original wavelength coordinates**
   and then mapped to the 512 resampled positions (consistent with the corpus
   resampling convention: position-normalized interpolation within each
   dataset's own range).

Note: the 512-point resampling loses absolute wavelengths;
this module re-injects physical coordinate information via the
task->original-wavelength mapping, consistent with that statement.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.fsl_regression.core import TASKS, L

BANDS_CH = [(1100, 1250), (1350, 1550), (1650, 1800), (2100, 2400)]
BANDS_OH = [(1400, 1500), (1900, 2000)]

_mask_cache: dict = {}


def task_wavelengths(task):
    """Original wavelength axis corresponding to the task."""
    from src.datasets import LOADERS
    ds, _attr = TASKS[task]
    d = LOADERS[ds]()
    if ds == "corn":
        return np.asarray(d["instruments"]["m5"][1], float)
    return np.asarray(d["wavelengths"], float)


def band_mask(task, n_out=L):
    """Prior band mask (0/1 vector on the 512 resampled positions), cached."""
    if task in _mask_cache:
        return _mask_cache[task]
    wl = task_wavelengths(task)
    bands = list(BANDS_CH) + (BANDS_OH if task.startswith("corn") else [])
    m = np.zeros(len(wl), dtype=np.float32)
    for a, b in bands:
        m[(wl >= a) & (wl <= b)] = 1.0
    pos_old = np.linspace(0, 1, len(wl))
    pos_new = np.linspace(0, 1, n_out)
    mi = np.interp(pos_new, pos_old, m)
    out = (mi > 0.5).astype(np.float32)
    _mask_cache[task] = out
    return out


def band_coverage(task):
    """Mask coverage (fraction of in-band points), for reporting."""
    return float(band_mask(task).mean())


# ---------------- constraint losses ----------------
def additivity_loss(enc, Xpool, rng, n_pairs=8):
    """Embedding linear-geometry regularizer. Xpool: (N,1,L) SNV spectra tensor
    of a single task.

    Returns the scale-free mean relative deviation. The embeddings of both the
    component spectra and the mixture spectra participate in the gradient
    (pushing the whole embedding space toward a linear structure).
    """
    import torch
    n = rng.integers(0, len(Xpool), size=n_pairs * 2)
    xa, xb = Xpool[n[:n_pairs]], Xpool[n[n_pairs:]]
    c = torch.tensor(rng.uniform(0.2, 0.8, size=(n_pairs, 1)), dtype=torch.float32,
                     device=Xpool.device)  # follow Xpool's device (GPU path: prevents device mismatch)
    c3 = c.unsqueeze(1)  # (P,1,1)
    xmix = c3 * xa + (1 - c3) * xb
    za, zb = enc(xa), enc(xb)
    zm = enc(xmix)
    target = c * za + (1 - c) * zb
    return (((zm - target) ** 2).mean()) / ((zm ** 2).mean() + 1e-8)


def band_saliency_loss(pred, Xq, mask_t):
    """Out-of-band / in-band sensitivity ratio (smaller = sensitivity more
    concentrated inside the prior bands).

    pred: (Q,) query predictions (graph connected to Xq); Xq: (Q,1,L) must have
    requires_grad; mask_t: (L,) prior band mask tensor.
    """
    import torch
    s = torch.autograd.grad(pred.sum(), Xq, create_graph=True)[0]  # (Q,1,L)
    s2 = (s.squeeze(1) ** 2)
    inside = (s2 * mask_t).mean()
    outside = (s2 * (1 - mask_t)).mean()
    return outside / (inside + 1e-8)
