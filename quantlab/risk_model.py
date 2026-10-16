"""Fundamental factor risk model (Barra-style), tested on a synthetic market with known truth.

    r_t = X_t f_t + u_t          stock returns = exposures x factor returns + specific returns
    Sigma = X F X' + D           covariance = factor part + diagonal specific variance

1. Each day, a cross-sectional regression (weights proportional to sqrt(market cap)) estimates factor returns f_t.
   The market column equals the sum of the industry dummies, so the system is collinear; as is common, we add the constraint
   that cap-weighted industry factor returns sum to zero (solved with a Lagrange multiplier).
2. Factor covariance F: exponentially weighted covariance of the estimated f_t (half-life hl_f).
3. Specific variance D: exponentially weighted mean of squared residuals (half-life hl_u), shrunk toward the cross-sectional median.
4. Benchmarks: the sample covariance (singular when N exceeds the window) and Ledoit-Wolf (2004) shrinkage toward mu*I.
"""
import numpy as np

from ._util import Checker, mm

STYLE_NAMES = ("market", "size", "value", "momentum", "volatility")


def synthetic_market(N=500, T=750, n_industries=10, seed=0):
    """A market with a known answer -> dict.

    X: (T, N, K) exposures, K = 5 styles (market = 1; the others cross-sectionally standardised and slowly drifting) + industry dummies.
    cap: (N,) market caps, consistent with the size exposure.
    True daily factor returns f: annualised vols market 16 %, size 4 %, value 3 %, momentum 5 %, volatility 4 %, industries 5-8 %, correlated.
    True specific vols: 15-45 % annualised.
    """
    rng = np.random.default_rng(seed)
    K = len(STYLE_NAMES) + n_industries
    industry = rng.integers(0, n_industries, N)
    size = rng.standard_normal(N)
    cap = np.exp(1.2 * size + 22)                                     # roughly log-normal market caps
    style = rng.standard_normal((N, len(STYLE_NAMES) - 1))
    style[:, 0] = size
    X = np.zeros((T, N, K))
    for t in range(T):
        if t:
            style = 0.995 * style + np.sqrt(1 - 0.995 ** 2) * rng.standard_normal(style.shape)   # exposures drift (half-life ~140 days)
            style[:, 0] = size                                        # size follows market cap and does not drift
        z = (style - style.mean(0)) / style.std(0)
        X[t, :, 0] = 1.0
        X[t, :, 1:len(STYLE_NAMES)] = z
        X[t, np.arange(N), len(STYLE_NAMES) + industry] = 1.0
    ann_vol = np.r_[0.16, 0.04, 0.03, 0.05, 0.04, rng.uniform(0.05, 0.08, n_industries)]
    A = rng.standard_normal((K, K)) * 0.3 + np.eye(K)
    corr = A @ A.T
    d = np.sqrt(np.diag(corr))
    corr = corr / np.outer(d, d)
    F_true = corr * np.outer(ann_vol, ann_vol) / 252
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        f = rng.multivariate_normal(np.zeros(K), F_true, size=T)
    spec_vol = rng.uniform(0.15, 0.45, N)
    D_true = spec_vol ** 2 / 252
    u = rng.standard_normal((T, N)) * np.sqrt(D_true)
    r = np.einsum("tnk,tk->tn", X, f) + u
    if not (np.all(np.isfinite(f)) and np.all(np.isfinite(r))):
        raise FloatingPointError("synthetic market produced non-finite values")
    return dict(X=X, r=r, f=f, u=u, F_true=F_true, D_true=D_true, cap=cap, industry=industry, n_industries=n_industries,
                names=list(STYLE_NAMES) + [f"industry{i}" for i in range(n_industries)])


def cross_sectional_regression(r, X, cap, n_industries):
    """One day's regression, weights ~ sqrt(cap), with cap-weighted industry returns constrained to sum to zero -> (f, residuals).
    Solved via the KKT system [2X'WX c; c' 0][f; lambda] = [2X'Wr; 0]."""
    N, K = X.shape
    w = np.sqrt(cap)
    w = w / w.sum()
    XtW = X.T * w
    first_ind = K - n_industries
    c = np.zeros(K)
    c[first_ind:] = (X[:, first_ind:] * cap[:, None]).sum(0) / cap.sum()
    M = np.zeros((K + 1, K + 1))
    M[:K, :K] = 2 * mm(XtW, X)
    M[:K, K] = c
    M[K, :K] = c
    b = np.r_[2 * mm(XtW, r), 0.0]
    f = np.linalg.solve(M, b)[:K]
    return f, r - mm(X, f)


def exp_weights(n, half_life):
    """n weights summing to one, the most recent the largest."""
    a = 0.5 ** (1.0 / half_life)
    w = a ** np.arange(n)[::-1]
    return w / w.sum()


def weighted_cov(Y, w):
    """Y: (n, k) in time order; w: (n,) weights summing to one -> weighted covariance (weighted mean removed)."""
    m = (Y * w[:, None]).sum(0)
    Z = Y - m
    return mm((Z * w[:, None]).T, Z)


def factor_model(r, X, cap, n_industries, t, window=500, hl_f=90, hl_u=60, shrink=0.2):
    """Estimate the model from (at most `window`) days strictly before t -> dict(F, D, f_hist, u_hist). Uses the past only."""
    start = max(0, t - window)
    fs, us = [], []
    for s in range(start, t):
        f, u = cross_sectional_regression(r[s], X[s], cap, n_industries)
        fs.append(f)
        us.append(u)
    fs, us = np.array(fs), np.array(us)
    F = weighted_cov(fs, exp_weights(len(fs), hl_f))
    D = (us ** 2 * exp_weights(len(us), hl_u)[:, None]).sum(0)
    D = (1 - shrink) * D + shrink * np.median(D)
    return dict(F=F, D=D, f_hist=fs, u_hist=us)


def model_cov(X_t, F, D):
    """Sigma = X F X' + diag(D)."""
    S = mm(mm(X_t, F), X_t.T)
    S[np.diag_indices_from(S)] += D
    return S


def sample_cov(r, t, window=250):
    return np.cov(r[max(0, t - window):t].T)


def ledoit_wolf(r, t, window=250):
    """Ledoit & Wolf (2004) shrinkage toward mu*I: Sigma_hat = delta*mu*I + (1 - delta)*S, with the optimal delta -> (Sigma_hat, delta)."""
    Y = r[max(0, t - window):t]
    n, p = Y.shape
    Y = Y - Y.mean(0)
    S = mm(Y.T, Y) / n
    mu = np.trace(S) / p
    d2 = np.sum((S - mu * np.eye(p)) ** 2) / p
    # sum_k ||y_k y_k' - S||^2 = sum_k (||y_k||^4 - 2 y_k' S y_k + ||S||^2): avoids forming p x p outer products
    yy = np.sum(Y * Y, 1)
    ySy = np.sum(mm(Y, S) * Y, 1)
    b_bar2 = float(np.sum(yy ** 2 - 2 * ySy + np.sum(S * S))) / p / n ** 2
    b2 = min(b_bar2, d2)
    delta = b2 / d2 if d2 > 0 else 1.0
    return delta * mu * np.eye(p) + (1 - delta) * S, delta


def risk_decomposition(w, X_t, F, D):
    """Split portfolio risk -> dict(total variance, factor part, specific part, vol, exposures, factor contributions, asset risk contributions).
    Exposure x = X'w; factor k's variance contribution x_k (F x)_k; asset i's risk contribution w_i (Sigma w)_i / sigma (Euler: they sum to sigma)."""
    x = mm(X_t.T, w)
    Fx = mm(F, x)
    factor = float(np.sum(x * Fx))
    specific = float(np.sum(w * w * D))
    Sw = mm(X_t, Fx) + D * w
    sigma = np.sqrt(factor + specific)
    return dict(total_var=factor + specific, factor_var=factor, specific_var=specific, vol=sigma, exposures=x,
                factor_contrib=x * Fx, asset_risk_contrib=w * Sw / sigma)


def min_variance(S, budget=1.0):
    """Fully invested minimum-variance portfolio (weights sum to `budget`; shorting allowed): w ~ S^{-1} 1."""
    v = np.linalg.solve(S, np.ones(S.shape[0]))
    return budget * v / v.sum()


def evaluate(m, start=500, step=21, window=250, seed=1):
    """Out-of-sample comparison of covariance estimates (re-estimated every `step` days, past data only):
    1. bias statistic: daily returns of a random dollar-neutral portfolio / predicted daily vol, std over the next `step` days (~1 is accurate);
    2. the realised annualised vol and leverage (sum |w|) of the minimum-variance portfolio built from each estimate."""
    rng = np.random.default_rng(seed)
    r, X, cap, n_ind = m["r"], m["X"], m["cap"], m["n_industries"]
    T, N = r.shape
    keys = ("true covariance (benchmark)", "factor model", "Ledoit-Wolf", "sample covariance")
    out = {k: dict(z=[], realised=[], leverage=[]) for k in keys}
    deltas = []
    for t in range(start, T - step, step):
        fm = factor_model(r, X, cap, n_ind, t)
        est = {"true covariance (benchmark)": model_cov(X[t], m["F_true"], m["D_true"]), "factor model": model_cov(X[t], fm["F"], fm["D"]),
               "sample covariance": sample_cov(r, t, window)}
        est["Ledoit-Wolf"], delta = ledoit_wolf(r, t, window)
        deltas.append(delta)
        w = rng.standard_normal(N)
        w = w - w.mean()
        w = w / np.abs(w).sum()
        future = r[t:t + step]
        for k, S in est.items():
            sigma = np.sqrt(float(np.sum(w * mm(S, w))))
            out[k]["z"].extend(list(mm(future, w) / sigma))
            try:
                wm = min_variance(S)
                out[k]["realised"].append(mm(future, wm))
                out[k]["leverage"].append(float(np.abs(wm).sum()))
            except (np.linalg.LinAlgError, FloatingPointError):
                out[k]["realised"].append(None)
    res = {}
    for k, v in out.items():
        got = [x for x in v["realised"] if x is not None]
        res[k] = dict(bias=float(np.std(v["z"])),
                      min_var_realised_vol=(float(np.std(np.concatenate(got)) * np.sqrt(252)) if len(got) == len(v["realised"]) else None),
                      min_var_median_leverage=(float(np.median(v["leverage"])) if v["leverage"] else None))
    res["mean Ledoit-Wolf shrinkage"] = float(np.mean(deltas))
    return res


def selftest():
    np.seterr(divide="ignore", over="ignore", invalid="ignore")
    check = Checker()
    m = synthetic_market(N=200, T=320, n_industries=6, seed=3)
    t = 5
    f, u = cross_sectional_regression(m["r"][t], m["X"][t], m["cap"], m["n_industries"])
    first_ind = len(STYLE_NAMES)
    c = np.zeros(len(f))
    c[first_ind:] = (m["X"][t][:, first_ind:] * m["cap"][:, None]).sum(0) / m["cap"].sum()
    wq = np.sqrt(m["cap"]) / np.sqrt(m["cap"]).sum()
    ortho = (m["X"][t][:, :first_ind] * wq[:, None]).T @ u
    check("regression: fit + residual = returns; constraint holds; residuals orthogonal (sqrt-cap weighted) to the unconstrained style columns",
          np.allclose(m["X"][t] @ f + u, m["r"][t]) and abs(float(np.sum(c * f))) < 1e-12 and np.allclose(ortho, 0, atol=1e-12))
    est = np.array([cross_sectional_regression(m["r"][s], m["X"][s], m["cap"], m["n_industries"])[0] for s in range(300)])
    corr = [np.corrcoef(est[:, k], m["f"][:300, k])[0, 1] for k in range(1, len(STYLE_NAMES))]
    check(f"regression: estimated style factor returns correlate > 0.7 with the truth ({', '.join(f'{x:.3f}' for x in corr)}; error ~ specific vol / sqrt(N), N = 200)",
          min(corr) > 0.7)
    w = exp_weights(10, 3)
    check("exponential weights: sum to one, increasing, weight 3 days back is half of today's", abs(w.sum() - 1) < 1e-12 and np.all(np.diff(w) > 0)
          and abs(w[-4] / w[-1] - 0.5) < 1e-12)
    Y = np.random.default_rng(0).standard_normal((50, 3))
    check("weighted covariance equals np.cov(ddof=0) under equal weights", np.allclose(weighted_cov(Y, np.full(50, 1 / 50)), np.cov(Y.T, ddof=0)))
    fm = factor_model(m["r"], m["X"], m["cap"], m["n_industries"], t=300, window=300)
    S = model_cov(m["X"][300], fm["F"], fm["D"])
    S_true = model_cov(m["X"][300], m["F_true"], m["D_true"])
    rng = np.random.default_rng(5)
    ws = rng.standard_normal((200, 200))
    ws = ws - ws.mean(1, keepdims=True)
    ratio = np.sqrt(np.einsum("in,nm,im->i", ws, S, ws) / np.einsum("in,nm,im->i", ws, S_true, ws))
    check(f"factor model: estimated / true vol of 200 random portfolios, median {np.median(ratio):.3f} (0.85-1.15)", 0.85 < np.median(ratio) < 1.15)
    sr = np.sqrt(fm["D"] / m["D_true"])
    check(f"specific vol: estimated / true, median {np.median(sr):.3f} (0.85-1.15)", 0.85 < np.median(sr) < 1.15)
    wp = rng.standard_normal(200)
    wp -= wp.mean()
    wp /= np.abs(wp).sum()
    dec = risk_decomposition(wp, m["X"][300], fm["F"], fm["D"])
    check("risk decomposition: factor + specific = w'Sw; factor contributions sum to the factor part; asset contributions sum to the vol",
          abs(dec["total_var"] - float(wp @ S @ wp)) < 1e-12 * max(1, dec["total_var"]) + 1e-18 and abs(dec["factor_contrib"].sum() - dec["factor_var"]) < 1e-15
          and abs(dec["asset_risk_contrib"].sum() - dec["vol"]) < 1e-12)
    Yc = m["r"][200:300] - m["r"][200:300].mean(0)
    S0 = Yc.T @ Yc / 100
    slow = sum(np.sum((np.outer(y, y) - S0) ** 2) for y in Yc) / 200 / 100 ** 2
    yy = np.sum(Yc * Yc, 1)
    fast = float(np.sum(yy ** 2 - 2 * np.sum((Yc @ S0) * Yc, 1) + np.sum(S0 * S0))) / 200 / 100 ** 2
    check("Ledoit-Wolf b_bar^2: vectorised formula equals the outer-product loop", abs(fast - slow) / slow < 1e-10)
    LW, delta = ledoit_wolf(m["r"], 300, window=100)
    smp = sample_cov(m["r"], 300, window=100)
    check(f"Ledoit-Wolf: with N (200) > window (100) the sample covariance is singular, the shrunk one is positive definite; delta = {delta:.2f}",
          np.linalg.matrix_rank(smp) <= 99 and np.all(np.linalg.eigvalsh(LW) > 0) and 0 < delta < 1)
    wm = min_variance(S_true)
    check("minimum variance: weights sum to one; no budget-preserving perturbation lowers the variance",
          abs(wm.sum() - 1) < 1e-10 and all(float((wm + e) @ S_true @ (wm + e)) >= float(wm @ S_true @ wm) - 1e-18
                                         for e in (lambda E: E - E.mean(1, keepdims=True))(rng.standard_normal((20, 200)) * 1e-3)))
    return check.summary()


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    np.seterr(divide="ignore", over="ignore", invalid="ignore")
    print("Out-of-sample evaluation (synthetic market N = 500, T = 750; re-estimated every 21 days from day 500):")
    for k, v in evaluate(synthetic_market()).items():
        print(f"  {k}: {v}")
