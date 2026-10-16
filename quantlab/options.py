"""Option pricing: Black-Scholes and Greeks, CRR binomial trees (European and American), implied volatility, Monte Carlo, delta hedging.

Checks: put-call parity, Hull's textbook example, every Greek against finite differences, tree convergence, American put > European put,
American call = European call without dividends, implied vol round trip, Monte Carlo within 3 standard errors, and delta-hedging P&L
-> C(sigma_implied) - C(sigma_realised) as hedging gets more frequent. Run `python -m quantlab.options` for the demos.
"""
import math

import numpy as np

from ._util import Checker


def Phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def phi(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def _d(S, K, T, r, sigma, q):
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    return d1, d1 - sigma * math.sqrt(T)


def bs_price(S, K, T, r, sigma, q=0.0, kind="call"):
    d1, d2 = _d(S, K, T, r, sigma, q)
    if kind == "call":
        return S * math.exp(-q * T) * Phi(d1) - K * math.exp(-r * T) * Phi(d2)
    return K * math.exp(-r * T) * Phi(-d2) - S * math.exp(-q * T) * Phi(-d1)


def bs_greeks(S, K, T, r, sigma, q=0.0, kind="call"):
    """-> dict(delta, gamma, vega (per unit sigma), theta (per year), rho (per unit r))."""
    d1, d2 = _d(S, K, T, r, sigma, q)
    eq, er, sT = math.exp(-q * T), math.exp(-r * T), math.sqrt(T)
    gamma = eq * phi(d1) / (S * sigma * sT)
    vega = S * eq * phi(d1) * sT
    if kind == "call":
        delta = eq * Phi(d1)
        theta = -S * eq * phi(d1) * sigma / (2 * sT) - r * K * er * Phi(d2) + q * S * eq * Phi(d1)
        rho = K * T * er * Phi(d2)
    else:
        delta = -eq * Phi(-d1)
        theta = -S * eq * phi(d1) * sigma / (2 * sT) + r * K * er * Phi(-d2) - q * S * eq * Phi(-d1)
        rho = -K * T * er * Phi(-d2)
    return dict(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)


def binomial(S, K, T, r, sigma, q=0.0, kind="call", american=False, steps=500):
    """CRR: u = exp(sigma sqrt(dt)), d = 1/u, risk-neutral p = (exp((r-q)dt) - d)/(u - d); backward induction (American: max with exercise)."""
    dt = T / steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1 / u
    p = (math.exp((r - q) * dt) - d) / (u - d)
    disc = math.exp(-r * dt)
    j = np.arange(steps + 1)
    ST = S * u ** j * d ** (steps - j)
    V = np.maximum(ST - K, 0) if kind == "call" else np.maximum(K - ST, 0)
    for n in range(steps - 1, -1, -1):
        V = disc * (p * V[1:] + (1 - p) * V[:-1])
        if american:
            Sn = S * u ** np.arange(n + 1) * d ** (n - np.arange(n + 1))
            V = np.maximum(V, (Sn - K) if kind == "call" else (K - Sn))
    return float(V[0])


def implied_vol(price, S, K, T, r, q=0.0, kind="call", lo=1e-4, hi=5.0, tol=1e-10):
    """Bisection (price is increasing in sigma). Raises ValueError if no volatility reproduces the price (e.g. below intrinsic value)."""
    f = lambda s: bs_price(S, K, T, r, s, q, kind) - price
    if f(lo) > 0 or f(hi) < 0:
        raise ValueError("no implied volatility for this price")
    for _ in range(200):
        m = 0.5 * (lo + hi)
        if f(m) > 0:
            hi = m
        else:
            lo = m
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def monte_carlo(S, K, T, r, sigma, q=0.0, kind="call", n=200000, seed=0):
    rng = np.random.default_rng(seed)
    ST = S * np.exp((r - q - 0.5 * sigma * sigma) * T + sigma * math.sqrt(T) * rng.standard_normal(n))
    pay = np.maximum(ST - K, 0) if kind == "call" else np.maximum(K - ST, 0)
    v = math.exp(-r * T) * pay
    return float(v.mean()), float(v.std(ddof=1) / math.sqrt(n))


def delta_hedge(S, K, T, r, sigma_implied, sigma_realised, steps, n_paths=2000, seed=0):
    """Sell one call at the implied-vol price, hold the realised-vol delta in stock, cash accrues at r; stock follows GBM at the realised vol.
    -> P&L per path at expiry (positive = profit)."""
    rng = np.random.default_rng(seed)
    dt = T / steps
    out = np.empty(n_paths)
    for i in range(n_paths):
        s = S
        delta = bs_greeks(s, K, T, r, sigma_realised)["delta"]
        cash = bs_price(S, K, T, r, sigma_implied) - delta * s
        for k in range(1, steps + 1):
            s = s * math.exp((r - 0.5 * sigma_realised ** 2) * dt + sigma_realised * math.sqrt(dt) * rng.standard_normal())
            cash *= math.exp(r * dt)
            if k < steps:
                nd = bs_greeks(s, K, T - k * dt, r, sigma_realised)["delta"]
                cash -= (nd - delta) * s
                delta = nd
        out[i] = cash + delta * s - max(s - K, 0)
    return out


def demos():
    np.seterr(divide="ignore", over="ignore", invalid="ignore")
    S, T, r = 100.0, 0.25, 0.03
    print("1. A synthetic equity skew (3 months): strike -> set implied vol / put price / implied vol recovered from the price")
    for K in (80, 90, 95, 100, 105, 110, 120):
        m = math.log(K / S)
        sig = 0.20 - 0.25 * m + 0.6 * m * m
        P = bs_price(S, K, T, r, sig, kind="put")
        print(f"   K = {K:>3}: {sig:.2%} / {P:7.3f} / {implied_vol(P, S, K, T, r, kind='put'):.2%}")
    print("2. Sell an at-the-money call (S = K = 100, 3 months) at 25 % implied vol and delta-hedge at the realised vol:")
    for sr in (0.15, 0.20, 0.25, 0.30):
        target = bs_price(100, 100, 0.25, 0.0, 0.25) - bs_price(100, 100, 0.25, 0.0, sr)
        d = delta_hedge(100, 100, 0.25, 0.0, 0.25, sr, 63, n_paths=1500, seed=3)
        w = delta_hedge(100, 100, 0.25, 0.0, 0.25, sr, 13, n_paths=1500, seed=3)
        print(f"   realised {sr:.0%}: theory {target:+.3f}; daily hedge mean {d.mean():+.3f} std {d.std():.3f}; weekly hedge mean {w.mean():+.3f} std {w.std():.3f}")


def selftest():
    np.seterr(divide="ignore", over="ignore", invalid="ignore")
    check = Checker()
    S, K, T, r, sigma, q = 100.0, 105.0, 0.5, 0.03, 0.25, 0.01
    c, p = bs_price(S, K, T, r, sigma, q), bs_price(S, K, T, r, sigma, q, "put")
    check("put-call parity: C - P = S e^{-qT} - K e^{-rT}", abs((c - p) - (S * math.exp(-q * T) - K * math.exp(-r * T))) < 1e-10)
    check("Hull's example (S = 42, K = 40, r = 10 %, sigma = 20 %, T = 0.5): call 4.76, put 0.81",
          abs(bs_price(42, 40, 0.5, 0.1, 0.2) - 4.76) < 0.005 and abs(bs_price(42, 40, 0.5, 0.1, 0.2, kind="put") - 0.81) < 0.005)
    ok = True
    for kind in ("call", "put"):
        g = bs_greeks(S, K, T, r, sigma, q, kind)
        h = 1e-4
        fd = dict(delta=(bs_price(S + h, K, T, r, sigma, q, kind) - bs_price(S - h, K, T, r, sigma, q, kind)) / (2 * h),
                  gamma=(bs_price(S + h, K, T, r, sigma, q, kind) - 2 * bs_price(S, K, T, r, sigma, q, kind) + bs_price(S - h, K, T, r, sigma, q, kind)) / h ** 2,
                  vega=(bs_price(S, K, T, r, sigma + h, q, kind) - bs_price(S, K, T, r, sigma - h, q, kind)) / (2 * h),
                  theta=-(bs_price(S, K, T + h, r, sigma, q, kind) - bs_price(S, K, T - h, r, sigma, q, kind)) / (2 * h),
                  rho=(bs_price(S, K, T, r + h, sigma, q, kind) - bs_price(S, K, T, r - h, sigma, q, kind)) / (2 * h))
        ok &= all(abs(g[k] - fd[k]) < 1e-4 * max(1, abs(g[k])) for k in g)
    check("Greeks (calls and puts): delta, gamma, vega, theta, rho all match finite differences", ok)
    errs = [abs(binomial(S, K, T, r, sigma, q, steps=n) - c) for n in (50, 200, 800)]
    check(f"binomial tree converges to Black-Scholes: errors {errs[0]:.4f} / {errs[1]:.4f} / {errs[2]:.4f} at 50 / 200 / 800 steps", errs[2] < errs[0] and errs[2] < 0.005)
    eu, am = binomial(S, 110, 1.0, 0.05, sigma, 0.0, "put", steps=800), binomial(S, 110, 1.0, 0.05, sigma, 0.0, "put", american=True, steps=800)
    amc, euc = binomial(S, K, 1.0, 0.05, sigma, 0.0, "call", american=True, steps=800), binomial(S, K, 1.0, 0.05, sigma, 0.0, "call", steps=800)
    check(f"American put worth more than European ({am:.4f} > {eu:.4f}); American call = European call without dividends", am > eu + 0.05 and abs(amc - euc) < 1e-9)
    check("implied volatility round trip (call and put)", abs(implied_vol(c, S, K, T, r, q) - sigma) < 1e-8 and abs(implied_vol(p, S, K, T, r, q, "put") - sigma) < 1e-8)
    try:
        implied_vol(1.0, 100, 50, 1, 0.0)
        raised = False
    except ValueError:
        raised = True
    check("implied volatility: a price below intrinsic value raises", raised)
    mc, se = monte_carlo(S, K, T, r, sigma, q)
    check(f"Monte Carlo within 3 standard errors ({mc:.4f} +/- {se:.4f} vs {c:.4f})", abs(mc - c) < 3 * se)
    atm = bs_price(100, 100, 1, 0.0, 0.2)
    check(f"at-the-money approximation: call ~ 0.4 sigma S sqrt(T) ({atm:.3f} vs 8.000)", abs(atm / 8.0 - 1) < 0.01)
    weekly = delta_hedge(100, 100, 0.25, 0.0, 0.25, 0.20, 13, n_paths=600, seed=1)
    daily = delta_hedge(100, 100, 0.25, 0.0, 0.25, 0.20, 252, n_paths=600, seed=1)
    target = bs_price(100, 100, 0.25, 0.0, 0.25) - bs_price(100, 100, 0.25, 0.0, 0.20)
    check(f"delta hedging (implied 25 %, realised 20 %): mean P&L {daily.mean():.3f} ~ C(25 %) - C(20 %) = {target:.3f}; "
          f"std falls from {weekly.std():.3f} (weekly) to {daily.std():.3f} (daily)", abs(daily.mean() - target) < 0.15 and daily.std() < weekly.std() / 2)
    return check.summary()


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    demos()
