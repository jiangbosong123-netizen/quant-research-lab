"""Histogram gradient-boosted regression trees (the LightGBM/XGBoost idea, small and slow) plus a ridge baseline.

1. Binning: each feature is cut at training-set quantiles into <= 32 bins; splits are only tried at bin edges.
2. A tree: at each node, try every bin edge of every feature and keep the largest gain
       gain = G_L^2/(n_L+lam) + G_R^2/(n_R+lam) - G^2/(n+lam)
   (G: sum of gradients, n: count, lam: L2 penalty). For squared loss the gradient is the residual, and with lam = 0 the gain equals the
   reduction in the sum of squared errors. Leaf value = G/(n+lam). Stop at max depth, min leaf size, or non-positive gain.
3. Boosting: start from the mean; each round fits a small tree to the current residuals and adds learning_rate x tree; rows (and columns)
   can be subsampled per round.
"""
import numpy as np

from ._util import Checker


class Binner:
    """Quantile bins from training data; values beyond the training range fall into the edge bins."""

    def __init__(self, X, n_bins=32):
        qs = np.linspace(0, 1, n_bins + 1)[1:-1]
        self.cuts = [np.unique(np.quantile(X[:, j], qs)) for j in range(X.shape[1])]
        self.n_bins = n_bins

    def transform(self, X):
        out = np.empty(X.shape, dtype=np.int64)
        for j, c in enumerate(self.cuts):
            out[:, j] = np.searchsorted(c, X[:, j], side="right")
        return out


def build_tree(B, g, rows, depth, min_leaf, lam, n_bins, rng=None, colsample=1.0):
    """B: (n, F) binned features; g: gradients (residuals); rows: indices in this node -> node dict.
    One bincount builds the histograms of all features at once (feature j, bin k at position j*n_bins + k)."""
    n = len(rows)
    G = float(g[rows].sum())
    leaf = dict(value=G / (n + lam))
    if depth == 0 or n < 2 * min_leaf:
        return leaf
    F = B.shape[1]
    cols = np.arange(F) if colsample >= 1 or rng is None else np.sort(rng.choice(F, max(1, int(round(F * colsample))), replace=False))
    off = (B[np.ix_(rows, cols)] + np.arange(len(cols)) * n_bins).ravel()
    Gh = np.bincount(off, weights=np.repeat(g[rows], len(cols)), minlength=len(cols) * n_bins).reshape(len(cols), n_bins)
    nh = np.bincount(off, minlength=len(cols) * n_bins).reshape(len(cols), n_bins)
    GL, nL = np.cumsum(Gh, 1), np.cumsum(nh, 1)
    GR, nR = G - GL, n - nL
    with np.errstate(divide="ignore", invalid="ignore"):
        gain = GL ** 2 / (nL + lam) + GR ** 2 / (nR + lam) - G ** 2 / (n + lam)
    gain[(nL < min_leaf) | (nR < min_leaf)] = -np.inf
    i, k = np.unravel_index(np.argmax(gain), gain.shape)
    if not np.isfinite(gain[i, k]) or gain[i, k] <= 1e-12:
        return leaf
    j = int(cols[i])
    left = rows[B[rows, j] <= k]
    right = rows[B[rows, j] > k]
    return dict(feature=j, bin=int(k), gain=float(gain[i, k]),
                left=build_tree(B, g, left, depth - 1, min_leaf, lam, n_bins, rng, colsample),
                right=build_tree(B, g, right, depth - 1, min_leaf, lam, n_bins, rng, colsample))


def predict_tree(node, B):
    out = np.empty(len(B))
    stack = [(node, np.arange(len(B)))]
    while stack:
        nd, idx = stack.pop()
        if "feature" not in nd:
            out[idx] = nd["value"]
            continue
        m = B[idx, nd["feature"]] <= nd["bin"]
        stack.append((nd["left"], idx[m]))
        stack.append((nd["right"], idx[~m]))
    return out


def split_counts(node, out=None):
    """How many times each feature was used to split (a simple importance measure)."""
    out = {} if out is None else out
    if "feature" in node:
        out[node["feature"]] = out.get(node["feature"], 0) + 1
        split_counts(node["left"], out)
        split_counts(node["right"], out)
    return out


class GBDT:
    def __init__(self, n_rounds=100, learning_rate=0.05, max_depth=3, min_leaf=50, lam=1.0, subsample=0.8, colsample=1.0, n_bins=32, seed=0):
        self.n_rounds, self.learning_rate, self.max_depth, self.min_leaf, self.lam = n_rounds, learning_rate, max_depth, min_leaf, lam
        self.subsample, self.colsample, self.n_bins, self.seed = subsample, colsample, n_bins, seed

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        self.binner = Binner(X, self.n_bins)
        B = self.binner.transform(X)
        self.base = float(y.mean())
        Fx = np.full(len(y), self.base)
        self.trees, self.train_mse = [], []
        for _ in range(self.n_rounds):
            g = y - Fx                                                # negative gradient of 0.5 (y - F)^2
            rows = np.arange(len(y)) if self.subsample >= 1 else np.sort(rng.choice(len(y), int(len(y) * self.subsample), replace=False))
            t = build_tree(B, g, rows, self.max_depth, self.min_leaf, self.lam, self.n_bins, rng, self.colsample)
            Fx = Fx + self.learning_rate * predict_tree(t, B)
            self.trees.append(t)
            self.train_mse.append(float(np.mean((y - Fx) ** 2)))
        return self

    def predict(self, X):
        B = self.binner.transform(X)
        out = np.full(len(X), self.base)
        for t in self.trees:
            out += self.learning_rate * predict_tree(t, B)
        return out

    def importance(self):
        tot = {}
        for t in self.trees:
            for j, c in split_counts(t).items():
                tot[j] = tot.get(j, 0) + c
        return tot


class Ridge:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        self.mx, self.my = X.mean(0), y.mean()
        Z = X - self.mx
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            self.w = np.linalg.solve(Z.T @ Z + self.alpha * np.eye(X.shape[1]), Z.T @ (y - self.my))
        return self

    def predict(self, X):
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            return (X - self.mx) @ self.w + self.my


def selftest():
    np.seterr(divide="ignore", over="ignore", invalid="ignore")
    check = Checker()
    rng = np.random.default_rng(0)
    X = rng.standard_normal((2000, 5))
    bn = Binner(X, 16)
    B = bn.transform(X)
    edge = bn.transform(np.array([[-99.0] * 5, [99.0] * 5]))
    check("binning: bins 0..15, monotone in the feature, out-of-range values go to the edge bins",
          B.min() == 0 and B.max() == 15 and np.all(np.diff(B[np.argsort(X[:, 0]), 0]) >= 0) and np.all(edge[0] == 0) and np.all(edge[1] == 15))
    y = (X[:, 2] > 0.3).astype(float) + 0.1 * rng.standard_normal(2000)
    g = y - y.mean()
    root = build_tree(bn.transform(X), g, np.arange(2000), 1, 20, 0.0, 16)
    thr = bn.cuts[2][root["bin"]] if root["bin"] < len(bn.cuts[2]) else np.inf
    left = X[:, 2] < thr
    sse = lambda v: float(np.sum((v - v.mean()) ** 2))
    red = sse(y) - sse(y[left]) - sse(y[~left])
    check(f"one split: picks the informative feature x2, threshold near 0.3 ({thr:.2f}); with lam = 0 gain = SSE reduction ({root['gain']:.4f} vs {red:.4f})",
          root["feature"] == 2 and abs(thr - 0.3) < 0.15 and abs(root["gain"] - red) < 1e-8 * max(1, red))
    best = max((sse(y) - sse(y[X[:, j] < c]) - sse(y[X[:, j] >= c]), j) for j in range(5) for c in bn.cuts[j]
               if 20 <= np.sum(X[:, j] < c) <= 1980)
    check("one split: same as brute force over every feature and cut point", best[1] == root["feature"] and abs(best[0] - root["gain"]) < 1e-8 * max(1, best[0]))
    m = GBDT(n_rounds=60, learning_rate=0.1, max_depth=3, min_leaf=20, lam=1.0, subsample=1.0).fit(X, y)
    check("boosting without subsampling: training MSE never increases", all(b <= a + 1e-12 for a, b in zip(m.train_mse, m.train_mse[1:])))
    Xt = rng.standard_normal((2000, 5))
    f = lambda Z: Z[:, 0] ** 2 + np.sin(2 * Z[:, 1])
    yt = f(X) + 0.3 * rng.standard_normal(2000)
    gb = GBDT(n_rounds=200, learning_rate=0.1, max_depth=3, min_leaf=20, subsample=0.8).fit(X, yt)
    rr = Ridge(1.0).fit(X, yt)
    R2 = lambda p: 1 - np.mean((f(Xt) - p) ** 2) / np.var(f(Xt))
    check(f"nonlinear truth (x0^2 + sin 2x1): out-of-sample R^2 GBDT {R2(gb.predict(Xt)):.2f}, ridge {R2(rr.predict(Xt)):.2f}",
          R2(gb.predict(Xt)) > 0.8 and R2(rr.predict(Xt)) < 0.3)
    imp = gb.importance()
    check(f"importance: the two most-split features are the informative x0, x1 ({sorted(imp.items(), key=lambda t: -t[1])[:3]})",
          set(sorted(imp, key=lambda j: -imp[j])[:2]) == {0, 1})
    yl = X @ np.array([1.0, -0.5, 0, 0, 0]) + 0.1 * rng.standard_normal(2000)
    check("ridge with tiny alpha = OLS; recovers (1, -0.5, 0, 0, 0)", np.allclose(Ridge(1e-8).fit(X, yl).w, [1, -0.5, 0, 0, 0], atol=0.02))
    a = GBDT(n_rounds=10, seed=3).fit(X, yt).predict(Xt)
    b = GBDT(n_rounds=10, seed=3).fit(X, yt).predict(Xt)
    check("same seed -> identical predictions", np.array_equal(a, b))
    return check.summary()


if __name__ == "__main__":
    import sys
    sys.exit(0 if selftest() else 1)
