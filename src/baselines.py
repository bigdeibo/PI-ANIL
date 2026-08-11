"""Baseline models .
Unified model factory interface: model = make_*(**kw); model.fit(Xs, ys); model.predict(Xt)
The support set is extremely small (5-20), so all hyperparameter tuning uses
only inner LOO-CV within the support set.
"""
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import make_pipeline


# ---------- PLS ----------
def _pls_best_nc(X, y, max_nc=15):
    n = len(y)
    if n > 40:  # for larger samples use 5-fold inner CV to avoid the LOO overhead
        from sklearn.model_selection import KFold
        cv = KFold(5, shuffle=True, random_state=0)
    else:
        cv = LeaveOneOut()
    best_nc, best_rmse = 1, np.inf
    for nc in range(1, min(max_nc, n - 1, X.shape[1]) + 1):
        try:
            pred = cross_val_predict(PLSRegression(n_components=nc), X, y,
                                     cv=cv).ravel()
            rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
            if rmse < best_rmse:
                best_rmse, best_nc = rmse, nc
        except Exception:
            break
    return best_nc


class PLSBaseline:
    name = "PLS"

    def __init__(self, max_nc=15):
        self.max_nc = max_nc

    def fit(self, X, y):
        self.nc_ = _pls_best_nc(X, y, self.max_nc)
        self.model_ = PLSRegression(n_components=self.nc_).fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X).ravel()


# ---------- Variable-selection PLS (P1b: strong chemometric baselines, closes R1-M3) ----------
# UVE-PLS (Centner 1996) and CARS-PLS (Jiang 2010).
# Anti-leakage rule: variable selection is performed only on the support set X
# at fit time (inner LOO/5-fold CV); test-set information is never used.
# Fallback: if fewer than 2 variables are selected, revert to plain
# full-variable PLS, so the worst case is ~PLS and no exception is raised
# (avoiding skipped reps that would distort the statistics).
def _uve_select(X, y, max_nc=15, seed=42):
    """UVE variable selection (Centner 1996): append p noise variables to the
    right of X, estimate the stability c=|mean(b)/std(b)| of each variable's
    regression coefficients via LOO, and keep the real variables whose
    stability exceeds the maximum stability of the noise variables.
    Returns a boolean keep mask."""
    n, p = X.shape
    if n < 4:
        return np.ones(p, dtype=bool)
    rng = np.random.default_rng(seed)
    # Noise variables: column-wise permutations of the real variables,
    # preserving the marginal distributions (as in libpls)
    Xnoise = np.empty((n, p))
    for j in range(p):
        Xnoise[:, j] = rng.permutation(X[:, j])
    Xaug = np.hstack([X, Xnoise])  # (n, 2p)
    nc = max(1, min(_pls_best_nc(X, y, max_nc), n - 1))
    nc = min(nc, Xaug.shape[1])
    # LOO estimate of each variable's regression coefficient sequence
    Bs = np.zeros((n, Xaug.shape[1]))
    for k, (tr, _) in enumerate(LeaveOneOut().split(Xaug)):
        nc_k = max(1, min(nc, len(tr) - 1))
        try:
            pls = PLSRegression(n_components=nc_k).fit(Xaug[tr], y[tr])
            Bs[k] = pls.coef_.ravel()
        except Exception:
            Bs[k] = 0.0
    mean_b = Bs.mean(0)
    std_b = Bs.std(0) + 1e-12
    c = np.abs(mean_b / std_b)  # stability
    c_real, c_noise = c[:p], c[p:]
    cutoff = float(c_noise.max()) if c_noise.size else 0.0
    keep = c_real > cutoff
    if keep.sum() < 2:
        keep = np.ones(p, dtype=bool)  # fallback
    return keep


def _cars_select(X, y, max_nc=15, n_runs=50, seed=0):
    """CARS variable selection (Jiang 2010): Monte Carlo sampling fits PLS to
    obtain |b| weights, combined with an exponentially decreasing function
    (EDF) retention ratio r(0)=1 -> r(N-1)=2/p and adaptive reweighted
    sampling (ARS); each round is evaluated by RMSECV and the best subset is
    selected. Returns a boolean keep mask."""
    from sklearn.model_selection import KFold
    n, p = X.shape
    if n < 4 or n_runs < 2:
        return np.ones(p, dtype=bool)
    rng = np.random.default_rng(seed)
    # EDF: retention ratio decays exponentially from 1 down to 2/p
    k_edf = np.log(max(p / 2.0, 1.0)) / (n_runs - 1)
    ratio = np.exp(-k_edf * np.arange(n_runs))
    cv = LeaveOneOut() if n <= 40 else KFold(5, shuffle=True, random_state=0)

    # Fixed number of latent components A (reused throughout CARS, as in the
    # original Jiang 2010 paper; this also avoids the LOO scan cost of
    # re-selecting nc each round, which would otherwise require tens of
    # thousands of PLS fits per episode on multivariate tasks)
    A = max(1, min(_pls_best_nc(X, y, max_nc), n - 1))
    sel = np.ones(p, dtype=bool)
    best_rmsecv, best_sel = np.inf, np.ones(p, dtype=bool)
    for i in range(n_runs):
        # MC sampling: draw ~90% of samples, fit PLS on the current sel to
        # obtain |b| weights
        m = max(int(round(0.9 * n)), 3)
        idx = rng.choice(n, m, replace=False)
        Xs, ys = X[idx][:, sel], y[idx]
        nc = max(1, min(A, m - 1, int(sel.sum())))
        try:
            pls = PLSRegression(n_components=nc).fit(Xs, ys)
            w_sel = np.abs(pls.coef_.ravel())
        except Exception:
            w_sel = np.ones(int(sel.sum()))
        # EDF target retention count (relative to the original p) + ARS
        # weighted resampling
        n_keep = max(2, int(round(ratio[i] * p)))
        sel_idx = np.where(sel)[0]
        if len(sel_idx) <= n_keep:
            chosen = sel_idx
        else:
            ws_sum = w_sel.sum()
            prob = w_sel / ws_sum if ws_sum > 0 else None
            chosen = rng.choice(sel_idx, size=n_keep, replace=False, p=prob)
        sel = np.zeros(p, dtype=bool)
        sel[chosen] = True
        # RMSECV (full support set x current sel, fixed A)
        if sel.sum() >= 2:
            Xcv = X[:, sel]
            nc_cv = max(1, min(A, n - 1, int(sel.sum())))
            try:
                pred = cross_val_predict(PLSRegression(n_components=nc_cv),
                                         Xcv, y, cv=cv).ravel()
                rmsecv = float(np.sqrt(np.mean((pred - y) ** 2)))
            except Exception:
                rmsecv = np.inf
        else:
            rmsecv = np.inf
        if rmsecv < best_rmsecv:
            best_rmsecv, best_sel = rmsecv, sel.copy()
    if best_sel.sum() < 2:
        best_sel = np.ones(p, dtype=bool)  # fallback
    return best_sel


class UVEPLSBaseline:
    name = "UVEPLS"

    def __init__(self, max_nc=15):
        self.max_nc = max_nc

    def fit(self, X, y):
        self.sel_ = _uve_select(X, y, self.max_nc, seed=42)
        self.n_var_ = int(self.sel_.sum())
        self.nc_ = _pls_best_nc(X[:, self.sel_], y, self.max_nc)
        self.model_ = PLSRegression(n_components=self.nc_).fit(X[:, self.sel_], y)
        return self

    def predict(self, X):
        return self.model_.predict(X[:, self.sel_]).ravel()


class CARSPLSBaseline:
    name = "CARSPLS"

    def __init__(self, max_nc=15, n_runs=50):
        self.max_nc = max_nc
        self.n_runs = n_runs

    def fit(self, X, y):
        self.sel_ = _cars_select(X, y, self.max_nc, self.n_runs, seed=0)
        self.n_var_ = int(self.sel_.sum())
        self.nc_ = _pls_best_nc(X[:, self.sel_], y, self.max_nc)
        self.model_ = PLSRegression(n_components=self.nc_).fit(X[:, self.sel_], y)
        return self

    def predict(self, X):
        return self.model_.predict(X[:, self.sel_]).ravel()


# ---------- SVR ----------
class SVRBaseline:
    name = "SVR"

    def __init__(self):
        self.grid_C = [1, 10, 100]
        self.grid_g = ["scale", 0.01, 0.001]

    def fit(self, X, y):
        from sklearn.model_selection import KFold
        n_splits = min(5, len(y))
        cv = KFold(n_splits, shuffle=True, random_state=0) if n_splits >= 2 else LeaveOneOut()
        best, best_rmse = None, np.inf
        for C in self.grid_C:
            for g in self.grid_g:
                pipe = make_pipeline(StandardScaler(), SVR(C=C, gamma=g))
                try:
                    pred = cross_val_predict(pipe, X, y, cv=cv)
                    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
                    if rmse < best_rmse:
                        best_rmse, best = rmse, (C, g)
                except Exception:
                    continue
        C, g = best if best else (10, "scale")
        self.model_ = make_pipeline(StandardScaler(), SVR(C=C, gamma=g)).fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X)


# ---------- 1D-CNN (PyTorch, CPU) ----------
def _cnn_module(n_wl):
    import torch
    return torch.nn.Sequential(
        torch.nn.Conv1d(1, 16, 15, padding=7), torch.nn.BatchNorm1d(16), torch.nn.ReLU(),
        torch.nn.MaxPool1d(2),
        torch.nn.Conv1d(16, 32, 15, padding=7), torch.nn.BatchNorm1d(32), torch.nn.ReLU(),
        torch.nn.MaxPool1d(2),
        torch.nn.Conv1d(32, 64, 11, padding=5), torch.nn.BatchNorm1d(64), torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool1d(8), torch.nn.Flatten(),
        torch.nn.Linear(64 * 8, 64), torch.nn.ReLU(), torch.nn.Dropout(0.3),
        torch.nn.Linear(64, 1),
    )


class CNNBaseline:
    name = "CNN"

    def __init__(self, epochs=300, lr=1e-3, augment=0, seed=0):
        self.epochs, self.lr, self.augment, self.seed = epochs, lr, augment, seed

    @staticmethod
    def _augment(X, y, n_aug, rng):
        """Slope/intensity offset + noise augmentation (Bjerrum 2017 style)."""
        Xs, ys = [X], [y]
        n, p = X.shape
        for _ in range(n_aug):
            slope = rng.uniform(0.9, 1.1, (n, 1))
            offset = rng.uniform(-0.05, 0.05, (n, 1))
            noise = rng.normal(0, 0.005, (n, p))
            Xs.append(X * slope + offset + noise)
            ys.append(y)
        return np.vstack(Xs), np.concatenate(ys)

    def fit(self, X, y):
        import torch
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        Xa, ya = (self._augment(X, y, self.augment, rng) if self.augment
                  else (X, y))
        self.x_mean_, self.x_std_ = Xa.mean(0), Xa.std(0) + 1e-8
        self.y_mean_, self.y_std_ = ya.mean(), ya.std() + 1e-8
        Xn = (Xa - self.x_mean_) / self.x_std_
        yn = (ya - self.y_mean_) / self.y_std_
        net = _cnn_module(X.shape[1])
        opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=1e-4)
        lossf = torch.nn.MSELoss()
        Xs = torch.tensor(Xn[:, None, :], dtype=torch.float32)
        ys = torch.tensor(yn[:, None], dtype=torch.float32)
        bs = min(len(yn), 32)
        net.train()
        for ep in range(self.epochs):
            perm = torch.randperm(len(yn))
            for i in range(0, len(yn), bs):
                idx = perm[i:i + bs]
                opt.zero_grad()
                loss = lossf(net(Xs[idx]), ys[idx])
                loss.backward()
                opt.step()
        self.net_ = net.eval()
        return self

    def predict(self, X):
        import torch
        Xn = (X - self.x_mean_) / self.x_std_
        with torch.no_grad():
            pred = self.net_(torch.tensor(Xn[:, None, :], dtype=torch.float32))
        return pred.numpy().ravel() * self.y_std_ + self.y_mean_


class CNNAugBaseline(CNNBaseline):
    name = "CNN+Aug"

    def __init__(self, **kw):
        # 5x augmented data with a comparable total number of update steps to
        # the plain CNN, ensuring fairness and speed
        kw.setdefault("augment", 4)
        kw.setdefault("epochs", 100)
        super().__init__(**kw)


BASELINES = {
    "PLS": lambda seed=0: PLSBaseline(),
    "SVR": lambda seed=0: SVRBaseline(),
    "CNN": lambda seed=0: CNNBaseline(seed=seed),
    "CNN+Aug": lambda seed=0: CNNAugBaseline(seed=seed),
    "UVEPLS": lambda seed=0: UVEPLSBaseline(),
    "CARSPLS": lambda seed=0: CARSPLSBaseline(),
}
