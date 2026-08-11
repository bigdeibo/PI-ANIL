"""Few-shot quantitative evaluation protocol 
Protocol: each property is evaluated independently; the support set is formed
by randomly sampling K samples (5/10/20) and the remainder serves as the test
set; the procedure is repeated N times and the mean +/- std of R^2 / RMSE /
RPD is reported.
"""
import numpy as np
from .preprocess import Preprocess


def rpd(y, rmse):
    s = np.std(y, ddof=1)
    return s / rmse if rmse > 1e-12 else np.inf


def run_fewshot_regression(X, y, model_fn, shots=(5, 10, 20), repeats=30,
                           preprocess="snv", seed=42, test_frac=None):
    """Returns {K: dict(r2_mean, r2_std, rmse_mean, rmse_std, rpd_mean, rpd_std, n_test_mean)}"""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    mask = ~np.isnan(y)
    X, y = X[mask], y[mask]
    n = len(y)
    results = {}
    for K in shots:
        if n <= K + 5:
            continue
        r2s, rmses, rpds, nts = [], [], [], []
        for rep in range(repeats):
            rng = np.random.default_rng(seed * 1000 + K * 100 + rep)
            if test_frac:
                n_test = max(int(n * test_frac), 10)
                perm = rng.permutation(n)
                te_idx = perm[:n_test]
                sup_pool = perm[n_test:]
                if len(sup_pool) < K:
                    continue
                tr_idx = rng.choice(sup_pool, K, replace=False)
            else:
                tr_idx = rng.choice(n, K, replace=False)
                te_idx = np.setdiff1d(np.arange(n), tr_idx)
            pp = Preprocess(preprocess).fit(X[tr_idx])
            Xs, Xs_te = pp.transform(X[tr_idx]), pp.transform(X[te_idx])
            m = model_fn(seed=rep)
            try:
                m.fit(Xs, y[tr_idx])
                pred = m.predict(Xs_te)
            except Exception:
                continue
            yt = y[te_idx]
            rmse = float(np.sqrt(np.mean((pred - yt) ** 2)))
            ss_res = float(np.sum((pred - yt) ** 2))
            ss_tot = float(np.sum((yt - yt.mean()) ** 2))
            r2 = 1 - ss_res / ss_tot if ss_tot > 1e-12 else np.nan
            r2s.append(r2); rmses.append(rmse); rpds.append(rpd(yt, rmse))
            nts.append(len(te_idx))
        if r2s:
            results[K] = dict(
                r2_mean=float(np.nanmean(r2s)), r2_std=float(np.nanstd(r2s)),
                rmse_mean=float(np.nanmean(rmses)), rmse_std=float(np.nanstd(rmses)),
                rpd_mean=float(np.nanmean(rpds)), rpd_std=float(np.nanstd(rpds)),
                n_test_mean=float(np.mean(nts)), n_reps=len(r2s),
            )
    return results


def run_full_cv(X, y, model_fn, preprocess="snv", folds=10, seed=42):
    """Full-data K-fold CV, serving as the literature-reference baseline."""
    from sklearn.model_selection import KFold
    X = np.asarray(X, float); y = np.asarray(y, float)
    mask = ~np.isnan(y)
    X, y = X[mask], y[mask]
    kf = KFold(folds, shuffle=True, random_state=seed)
    preds = np.full(len(y), np.nan)
    for tr, te in kf.split(X):
        pp = Preprocess(preprocess).fit(X[tr])
        m = model_fn(seed=0)
        m.fit(pp.transform(X[tr]), y[tr])
        preds[te] = m.predict(pp.transform(X[te]))
    rmse = float(np.sqrt(np.nanmean((preds - y) ** 2)))
    ss_res = float(np.nansum((preds - y) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return dict(r2=1 - ss_res / ss_tot, rmse=rmse, rpd=rpd(y, rmse), n=len(y))
