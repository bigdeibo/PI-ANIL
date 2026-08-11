"""Spectral preprocessing module 
All transforms follow the "fit on the support set, apply to all samples" rule
to avoid information leakage.
"""
import numpy as np
from scipy.signal import savgol_filter


def snv(X):
    """Standard normal variate transform (per sample)."""
    m = X.mean(axis=1, keepdims=True)
    s = X.std(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return (X - m) / s


class MSC:
    """Multiplicative scatter correction: fit against the support-set mean spectrum as reference."""
    def fit(self, X_ref):
        self.ref_ = X_ref.mean(axis=0)
        return self

    def transform(self, X):
        out = np.empty_like(X, dtype=float)
        A = np.column_stack([np.ones_like(self.ref_), self.ref_])
        pinv = np.linalg.pinv(A)
        for i, x in enumerate(X):
            a0, b1 = pinv @ x
            denom = b1 if abs(b1) > 1e-12 else 1e-12
            out[i] = (x - a0) / denom
        return out


def sg_deriv(X, deriv=1, window=15, poly=2):
    """Savitzky-Golay derivative (per sample)."""
    window = min(window, X.shape[1] // 2 * 2 - 1)
    if window < poly + 2:
        window = poly + 3 if (poly + 3) % 2 == 1 else poly + 4
    return savgol_filter(X, window, poly, deriv=deriv, axis=1)


class Preprocess:
    """Composable preprocessing pipeline.
    kind: 'none' | 'snv' | 'msc' | 'sg1' | 'sg2' | 'snv_sg1' | 'msc_sg1'
    """
    def __init__(self, kind="snv"):
        self.kind = kind
        self._msc = None

    def fit(self, X_ref):
        if "msc" in self.kind:
            self._msc = MSC().fit(X_ref)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        parts = self.kind.split("_")
        if self.kind == "none":
            return X.copy()
        out = X.copy()
        if "msc" in parts:
            out = self._msc.transform(out)
        if "snv" in parts:
            out = snv(out)
        if "sg1" in parts:
            out = sg_deriv(out, 1)
        if "sg2" in parts:
            out = sg_deriv(out, 2)
        return out
