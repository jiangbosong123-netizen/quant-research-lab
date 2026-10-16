"""Synthetic stock panel for the ML experiments: P months x N stocks x F characteristics with a known signal.

Characteristics persist like real firm characteristics (monthly AR(1), 0.9) and are standardised each month.
Next-month return = strength x true signal + noise. The true signal uses the first four features only (the rest are noise):
  nonlinear=True: x0 + (x1^2 - 1) + x2*x3   (linear + U-shape + interaction);  nonlinear=False: x0 + 0.5*x1.
The strength is set by the target IC (cross-sectional correlation between true signal and next-month return); good real signals have IC ~0.03-0.08.
`overlapping_labels` builds h-month forward returns sampled monthly, so adjacent labels share h-1 months (used to show leakage).
"""
import numpy as np

from ._util import Checker


def generate(P=180, N=300, F=20, nonlinear=True, target_ic=0.06, persistence=0.9, seed=0):
    """-> dict(X (P,N,F), y (P,N) next-month returns, signal (P,N))."""
    rng = np.random.default_rng(seed)
    X = np.zeros((P, N, F))
    x = rng.standard_normal((N, F))
    for p in range(P):
        if p:
            x = persistence * x + np.sqrt(1 - persistence ** 2) * rng.standard_normal((N, F))
        X[p] = (x - x.mean(0)) / x.std(0)
    if nonlinear:
        s = X[..., 0] + (X[..., 1] ** 2 - 1) + X[..., 2] * X[..., 3]
    else:
        s = X[..., 0] + 0.5 * X[..., 1]
    s = (s - s.mean(1, keepdims=True)) / s.std(1, keepdims=True)
    a = target_ic / np.sqrt(1 - target_ic ** 2)                      # corr(a*s + e, s) = a / sqrt(1 + a^2)
    y = a * s + rng.standard_normal((P, N))
    return dict(X=X, y=y, signal=s, nonlinear=nonlinear, target_ic=target_ic)


def overlapping_labels(y, h):
    """Label for month p = returns of months p .. p+h-1 summed; the last h-1 months are NaN."""
    out = np.full_like(y, np.nan)
    for p in range(y.shape[0] - h + 1):
        out[p] = y[p:p + h].sum(0)
    return out


def rank_corr(a, b):
    """Spearman correlation."""
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def selftest():
    check = Checker()
    d = generate(P=240, N=400, seed=1)
    X, y, s = d["X"], d["y"], d["signal"]
    check("features standardised each month", np.allclose(X.mean(1), 0, atol=1e-12) and np.allclose(X.std(1), 1, atol=1e-12))
    ac = np.mean([np.corrcoef(X[p, :, 5], X[p + 1, :, 5])[0, 1] for p in range(239)])
    check(f"features persist: month-to-month correlation ~0.9 ({ac:.3f})", 0.87 < ac < 0.93)
    ic = np.mean([np.corrcoef(s[p], y[p])[0, 1] for p in range(240)])
    check(f"true-signal IC ~ target 0.06 ({ic:.4f}, average of 240 months)", 0.045 < ic < 0.075)
    noise_ic = np.mean([abs(np.corrcoef(X[p, :, 10], y[p])[0, 1]) for p in range(240)])
    check(f"noise feature unrelated to returns: mean |IC| {noise_ic:.3f} ~ sampling error sqrt(1/400) = 0.05", noise_ic < 0.06)
    h = overlapping_labels(y, 3)
    check("overlapping labels: sum of 3 months; last 2 months NaN; adjacent labels correlate ~2/3",
          np.allclose(h[10], y[10] + y[11] + y[12]) and np.isnan(h[-1]).all() and np.isnan(h[-2]).all() and not np.isnan(h[-3]).any()
          and 0.6 < np.corrcoef(h[50], h[51])[0, 1] < 0.73)
    check("rank correlation: invariant to monotone transforms; reversed order = -1",
          abs(rank_corr(np.arange(10.0), np.exp(np.arange(10.0))) - 1) < 1e-12 and abs(rank_corr(np.arange(10.0), -np.arange(10.0)) + 1) < 1e-12)
    return check.summary()


if __name__ == "__main__":
    import sys
    sys.exit(0 if selftest() else 1)
