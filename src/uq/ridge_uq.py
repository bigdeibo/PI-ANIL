"""Closed-form ridge regression + jackknife+ uncertainty for ANIL's frozen-encoder linear head.

Design doc: PI-ANIL二期研究设计文档.md §3.2–§3.3. Channel B (calibration) replaces the
50-step GD head adaptation with the closed-form ridge solution of the *same* objective,
which makes leave-one-out quantities analytic at K<=20.

Key properties:
  - LOO residuals and LOO query predictions are closed-form (hat matrix), no resampling.
  - jackknife+ intervals (Barber et al., Ann. Statist. 2021, DOI 10.1214/20-aos1965):
    distribution-free coverage >= 1-2*alpha; at K=5 the interval endpoints live on a
    5-point grid (reported honestly, design doc Q2).
  - ridge posterior-style variance sigma^2(x) = s2 * (1 + phi' A^{-1} phi) is a
    *diagnostic*, not a guarantee; at K < p it depends on lambda (design doc R5/Q7).
"""
import numpy as np


def _augment(Phi, lam, fit_intercept):
    """Optionally prepend an UNPENALIZED intercept column.

    Critical at K <= 20: support-mean centering of the embeddings (the channel-A
    protocol) makes the columns sum to zero, so every row satisfies
    phi_i = -sum_{j!=i} phi_j — an exact linear dependency. Combined with
    mean-zero (z-scored) labels and no intercept, the leave-one-out prediction
    is then *identically exact* (yhat_{-i}(x_i) = y_i in the limit), which makes
    jackknife+ vacuous. The calibration channel therefore uses scale-only
    normalization plus an explicit unpenalized intercept, which breaks the
    dependency while staying closest to channel-A geometry (design doc, D-log
    2026-09-15)."""
    K, p = Phi.shape
    if not fit_intercept:
        return Phi, lam * np.eye(p)
    Phi_aug = np.column_stack([np.ones(K), Phi])
    pen = np.eye(p + 1) * lam
    pen[0, 0] = 0.0
    return Phi_aug, pen


def ridge_solve(Phi, y, lam, fit_intercept=True):
    """Return w, Ainv, hat-diagonal h for min ||Phi w - y||^2 + lam ||w||^2.

    Phi: (K, p) support embeddings (scale-only normalized per protocol — do NOT
    support-mean-center, see _augment); y: (K,) z-scored support labels.
    """
    Phi_aug, pen = _augment(Phi, lam, fit_intercept)
    K, p1 = Phi_aug.shape
    A = Phi_aug.T @ Phi_aug + pen
    Ainv = np.linalg.inv(A)
    w = Ainv @ Phi_aug.T @ y
    H = Phi_aug @ Ainv @ Phi_aug.T
    return w, Ainv, np.diag(H)


def gcv_lambda(Phi, y, grid=None):
    """Pick lambda by GCV. KNOWN PITFALL at K < p: as lambda -> 0 the fit
    interpolates (tr(H) -> K), the dof denominator vanishes, and GCV picks the
    grid minimum, collapsing all intervals to zero width. Kept for reference;
    use loo_lambda instead."""
    if grid is None:
        grid = np.logspace(-4, 1, 26)
    K = len(y)
    best, best_lam = np.inf, grid[0]
    for lam in grid:
        w, _, h = ridge_solve(Phi, y, lam)
        Phi_aug = np.column_stack([np.ones(K), Phi])
        r = y - Phi_aug @ w
        denom = max(1.0 - h.sum() / K, 1e-3)
        gcv = float(np.mean(r ** 2)) / denom ** 2
        if gcv < best:
            best, best_lam = gcv, lam
    return best_lam


def loo_lambda(Phi, y, grid=None):
    """Pick lambda by leave-one-out CV on the closed-form LOO residuals.

    At lambda -> 0 (interpolation regime, K < p) the LOO residuals stay finite
    (they converge to the minimum-norm interpolator's LOO residuals), so the
    criterion has a genuine interior minimum. This is the default selector."""
    if grid is None:
        grid = np.logspace(-4, 1, 26)
    best, best_lam = np.inf, grid[-1]
    for lam in grid:
        _, _, _, r_loo = loo_quantities(Phi, y, lam)
        score = float(np.mean(r_loo ** 2))
        if score < best:
            best, best_lam = score, lam
    return best_lam


def loo_quantities(Phi, y, lam, fit_intercept=True):
    """Closed-form LOO: r_loo[i] = residual of point i under the fit without it."""
    w, Ainv, h = ridge_solve(Phi, y, lam, fit_intercept)
    Phi_aug = np.column_stack([np.ones(len(Phi)), Phi]) if fit_intercept else Phi
    r = y - Phi_aug @ w
    r_loo = r / np.clip(1.0 - h, 1e-8, None)
    return w, Ainv, h, r_loo


def jackknife_plus_interval(Phi, y, lam, Phi_q, alpha, fit_intercept=True):
    """Jackknife+ prediction intervals for query embeddings Phi_q (nq, p).

    Returns (lo, hi, info). lo/hi are in the z-scored label scale; callers un-z-score.
    Quantile levels follow Barber et al. 2021: coverage guarantee is 1-2*alpha.
    """
    w, Ainv, h, r_loo = loo_quantities(Phi, y, lam, fit_intercept)
    Phi_qa = np.column_stack([np.ones(len(Phi_q)), Phi_q]) if fit_intercept else Phi_q
    G = Phi_qa @ Ainv @ (np.column_stack([np.ones(len(Phi)), Phi])
                         if fit_intercept else Phi).T   # (nq, K)
    yhat_q = Phi_qa @ w                                   # (nq,)
    # beta_{-i} = w - Ainv phi_i * r_loo[i]  =>  yhat_{-i}(x) = yhat(x) - G[:,i] r_loo[i]
    Y_loo = yhat_q[:, None] - G * r_loo[None, :]          # (nq, K)
    lo = np.quantile(Y_loo - r_loo[None, :], alpha, axis=1, method="inverted_cdf")
    hi = np.quantile(Y_loo + r_loo[None, :], 1.0 - alpha, axis=1, method="inverted_cdf")
    info = dict(yhat=yhat_q, r_loo=r_loo, Y_loo=Y_loo)
    return lo, hi, info


def ridge_variance(Phi, y, lam, Phi_q, fit_intercept=True):
    """sigma^2(x) = s2 * (1 + phi' Ainv phi); s2 uses effective degrees of freedom.

    Diagnostic only (no coverage guarantee). At K < p the estimate is lambda-dependent.
    """
    w, Ainv, h = ridge_solve(Phi, y, lam, fit_intercept)
    Phi_aug = np.column_stack([np.ones(len(Phi)), Phi]) if fit_intercept else Phi
    Phi_qa = np.column_stack([np.ones(len(Phi_q)), Phi_q]) if fit_intercept else Phi_q
    r = y - Phi_aug @ w
    dof = max(len(y) - h.sum(), 1.0)
    s2 = float(r @ r) / dof
    lev_q = np.einsum("ij,jk,ik->i", Phi_qa, Ainv, Phi_qa)
    return s2 * (1.0 + lev_q), s2
