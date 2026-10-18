"""The variance risk premium test (pre-registered test #19) and the hedge arithmetic used in Part 7, as pure functions.

Test #19 sells a one-month S&P 500 variance swap at the end of every month, struck at the VIX squared, and settles it on the
realised variance of daily index closes over the next month:

    strike      K   = VIX / 100 at the month's last trading day (only data available at that close)
    cost        K_c = K x (1 - c), c = 1 % of the vol strike (half the bid-ask spread dealers quote for short variance swaps)
    realised    RV  = (252 / n) x sum(ln(S_i / S_{i-1})^2) over the n trading days of the next month
    return      r   = s x (1 - RV / K_c^2), s = 10 % of capital in variance notional (measured at the strike)

The return is an excess return: a variance swap costs nothing to enter and the collateral earns the cash rate.
Futures hedge cost (Part 7, P7a): a long futures position of h (fraction of capital) pays h x (cash rate x days / 360 + spread x days / 365).
"""
import math

import numpy as np

from ._util import Checker

NOTIONAL = 0.10
COST = 0.01
DAYS_PER_YEAR = 252


def realised_variance(closes):
    """Annualised realised variance from a run of daily closes (first close = the previous month's last close)."""
    c = np.asarray(closes, dtype=float)
    if len(c) < 2 or np.any(c <= 0):
        raise ValueError("need at least two positive closes")
    r = np.diff(np.log(c))
    return DAYS_PER_YEAR / len(r) * float(np.sum(r * r))


def short_variance_return(vix, closes, cost=COST, notional=NOTIONAL, cap=None):
    """One month of test #19: sell variance at (VIX/100 x (1 - cost))^2, settle on realised variance; cap = optional vol cap as a multiple of the strike."""
    kc = vix / 100.0 * (1.0 - cost)
    rv = realised_variance(closes)
    if cap is not None:
        rv = min(rv, (cap * kc) ** 2)
    return notional * (1.0 - rv / (kc * kc))


def johnson_t(xs):
    """Johnson (1978) skewness-adjusted t for the mean: [m + mu3 / (6 s^2 n) + mu3 m^2 / (3 s^4)] / (s / sqrt(n)).
    With strongly negative skew (selling insurance) the ordinary t over-rejects; this correction pulls it down."""
    x = np.asarray(xs, dtype=float)
    n = len(x)
    m = float(x.mean())
    s2 = float(np.sum((x - m) ** 2) / (n - 1))
    mu3 = float(np.sum((x - m) ** 3) / n)
    return (m + mu3 / (6 * s2 * n) + mu3 * m * m / (3 * s2 * s2)) / math.sqrt(s2 / n)


def futures_funding_cost(h, cash_rate, days, spread):
    """Funding cost embedded in a futures hedge of size h (positive = long futures): h x (cash x days / 360 + spread x days / 365)."""
    return h * (cash_rate * days / 360.0 + spread * days / 365.0)


def contracts(notional_fraction, capital, index_level, multiplier):
    """Number of futures (or index option) contracts for a notional of notional_fraction x capital."""
    return abs(notional_fraction) * capital / (multiplier * index_level)


def selftest():
    check = Checker()
    rng = np.random.default_rng(19)
    closes = 100.0 * np.exp(np.cumsum(np.r_[0.0, rng.normal(0.0, 0.01, 21)]))
    r = np.diff(np.log(closes))
    rv = 252 / 21 * float(np.sum(r * r))
    check("realised variance = 252 / n x sum of squared log returns", abs(realised_variance(closes) - rv) < 1e-15)
    kc = 0.20 * 0.99
    check("one month: 0.10 x (1 - RV / (0.2 x 0.99)^2)", abs(short_variance_return(20.0, closes) - 0.10 * (1 - rv / kc ** 2)) < 1e-15)
    check("a month whose realised vol is twice the strike loses 30 % at 10 % notional", abs(NOTIONAL * (1 - 4.0) + 0.30) < 1e-15)
    check("a 2.5x vol cap only ever helps the seller",
          short_variance_return(20.0, closes, cap=2.5) >= short_variance_return(20.0, closes)
          and abs(short_variance_return(5.0, closes, cap=2.5) - NOTIONAL * (1 - 6.25)) < 1e-12)
    check("a higher cost lowers the return", short_variance_return(20.0, closes, cost=0.05) < short_variance_return(20.0, closes) < short_variance_return(20.0, closes, cost=0.0))
    # No premium: implied vol = true vol, no cost -> the mean monthly return is about zero (5,000 simulated months).
    sigma = 0.01 * math.sqrt(252)
    rets = [short_variance_return(sigma * 100, 100.0 * np.exp(np.cumsum(np.r_[0.0, rng.normal(0.0, 0.01, 21)])), cost=0.0) for _ in range(5000)]
    check("with no premium the seller earns about nothing (mean within 3 standard errors of 0)",
          abs(np.mean(rets)) < 3 * np.std(rets) / math.sqrt(len(rets)))
    sym = rng.normal(0.01, 0.02, 400)
    neg = np.where(rng.random(400) < 0.05, -0.30, 0.03)
    t = lambda x: float(np.mean(x) / (np.std(x, ddof=1) / math.sqrt(len(x))))
    check("Johnson t ~ ordinary t for symmetric returns, and smaller for insurance-like (negatively skewed) returns",
          abs(johnson_t(sym) - t(sym)) < 0.02 * abs(t(sym)) and johnson_t(neg) < t(neg))
    check("futures funding: long h = 0.25 for 30 days at 4 % cash + 0.5 % spread costs 0.25 x (0.04 x 30/360 + 0.005 x 30/365)",
          abs(futures_funding_cost(0.25, 0.04, 30, 0.005) - 0.25 * (0.04 * 30 / 360 + 0.005 * 30 / 365)) < 1e-15
          and futures_funding_cost(-0.25, 0.04, 30, 0.005) < 0)
    check("contracts: hedging 0.25 of $5m with micro E-mini ($5 x index) at 6,000 -> 41.7 contracts",
          abs(contracts(0.25, 5_000_000, 6000.0, 5) - 41.666666666666664) < 1e-9)
    return check.summary()
