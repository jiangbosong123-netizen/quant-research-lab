"""Order-book signals and intraday bar rules (Part 9), as pure functions.

Order book (top of book only; one row per book update; prices in cents):
  - order-flow imbalance (Cont, Kukanov & Stoikov 2014): per update
        e = [bid px did not fall]·bid qty − [bid px did not rise]·previous bid qty − [ask px did not rise]·ask qty + [ask px did not fall]·previous ask qty,
    summed over 10-second windows; the mid-price change is roughly linear in it, with a slope inversely related to depth;
  - queue imbalance I = (bid qty − ask qty) / (bid qty + ask qty): the larger, the likelier the next mid move is up;
  - the simplest micro-price m = (ask px · bid qty + bid px · ask qty) / (bid qty + ask qty), used to predict the mid 10 seconds later.
Intraday bars (1-minute closes aggregated to h-minute bars on the UTC clock):
  - returns only between adjacent bars; the position at the close of bar k is d · sign(return of bar k) (d = +1 momentum, −1 reversal),
    held to the close of bar k + 1; turnover counts a flip as 2 units, and after a gap the position restarts from flat.
"""
import math

import numpy as np

from ._util import Checker


# ---------- order book ----------

def mid(book):
    return (book["bid_px"] + book["ask_px"]) / 2.0


def ofi_events(book):
    """Per-update order-flow imbalance e (the first update gets 0)."""
    bp, bq, ap, aq = book["bid_px"], book["bid_qty"], book["ask_px"], book["ask_qty"]
    e = np.zeros(len(bp))
    e[1:] = ((bp[1:] >= bp[:-1]) * bq[1:] - (bp[1:] <= bp[:-1]) * bq[:-1]
             - (ap[1:] <= ap[:-1]) * aq[1:] + (ap[1:] >= ap[:-1]) * aq[:-1])
    return e


def ofi_windows(book, start, end, width=10.0):
    """Cut [start, end) into windows: -> (mid change over each window, OFI summed over the updates inside it).
    The window's opening mid is the book after the last update before it; windows with no update are skipped."""
    t, m, e = book["t"], mid(book), ofi_events(book)
    edges = np.arange(start, end + 1e-9, width)
    dp, of = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        i0 = np.searchsorted(t, lo, side="left")
        i1 = np.searchsorted(t, hi, side="left")
        if i1 <= i0 or i0 == 0:
            continue
        dp.append(m[i1 - 1] - m[i0 - 1])
        of.append(e[i0:i1].sum())
    return np.array(dp), np.array(of)


def ols(y, x):
    """y = a + b x: -> (a, b, R^2)."""
    xm, ym = x.mean(), y.mean()
    b = float(((x - xm) * (y - ym)).sum() / ((x - xm) ** 2).sum())
    a = float(ym - b * xm)
    return a, b, 1 - float(((y - a - b * x) ** 2).sum() / ((y - ym) ** 2).sum())


def out_of_sample_r2(y, x, a, b):
    """R^2 of a fit estimated elsewhere (1 − residual SS / this sample's own total SS; can be negative)."""
    return 1 - float(((y - a - b * x) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def next_mid_move(book):
    """Direction (+1 / −1) of the next change in the mid after each update, looking only at later updates; 0 if it never changes again."""
    m = mid(book)
    out = np.zeros(len(m))
    nxt = 0.0
    for i in range(len(m) - 1, -1, -1):
        out[i] = nxt
        if i > 0 and m[i] != m[i - 1]:
            nxt = np.sign(m[i] - m[i - 1])
    return out


def queue_imbalance(book):
    return (book["bid_qty"] - book["ask_qty"]) / (book["bid_qty"] + book["ask_qty"])


def microprice(book):
    return (book["ask_px"] * book["bid_qty"] + book["bid_px"] * book["ask_qty"]) / (book["bid_qty"] + book["ask_qty"])


def mid_after(book, seconds=10.0):
    """The mid `seconds` after each update (the book after the last update at or before t + seconds); nan past the end of the day."""
    t, m = book["t"], mid(book)
    j = np.searchsorted(t, t + seconds, side="right") - 1
    out = m[np.clip(j, 0, len(m) - 1)].copy()
    out[t + seconds > t[-1]] = np.nan
    return out


# ---------- intraday bars ----------

def to_bars(t_ms, close, minutes):
    """1-minute bars (open time in ms, close) -> h-minute bars on the UTC clock: -> (bar number, close of the bar's last minute)."""
    bucket = t_ms // (minutes * 60000)
    last = np.concatenate([bucket[1:] != bucket[:-1], [True]])
    return bucket[last], close[last]


def bar_returns(bucket, close):
    """Returns between adjacent bars only (a missing bar breaks the chain): -> (bar number of the later bar, return)."""
    adj = bucket[1:] == bucket[:-1] + 1
    r = close[1:] / close[:-1] - 1
    return bucket[1:][adj], r[adj]


def sign_rule(bucket, r, direction):
    """Position at bar k = direction · sign(r_k), earning r_{k+1} (bars must be adjacent): -> (bar, gross return, |change in position|).
    The first trade, and the first after a gap, counts the whole position as turnover."""
    adj = bucket[1:] == bucket[:-1] + 1
    pos = direction * np.sign(r[:-1])
    gross = pos * r[1:]
    prev = np.concatenate([[0.0], pos[:-1]])
    prev[1:][~adj[:-1]] = 0.0
    turn = np.abs(pos - prev)
    return bucket[1:][adj], gross[adj], turn[adj]


def annualised_sharpe(x, minutes):
    x = np.asarray(x, float)
    if len(x) < 2 or x.std(ddof=1) == 0:
        return float("nan")
    return float(x.mean() / x.std(ddof=1) * math.sqrt(365 * 24 * 60 / minutes))


def autocorr(x, lag=1):
    x = np.asarray(x, float) - np.mean(x)
    return float(np.sum(x[lag:] * x[:-lag]) / np.sum(x * x))


def selftest():
    check = Checker()
    book = dict(t=np.array([0.0, 1, 2, 3]), bid_px=np.array([10.0, 10, 11, 11]), bid_qty=np.array([100.0, 150, 50, 50]),
                ask_px=np.array([12.0, 12, 12, 11.5]), ask_qty=np.array([200.0, 200, 200, 80]))
    check("OFI by hand: bid size 100 -> 150 at the same price gives +50; bid price 10 -> 11 brings in a whole new level (+50); ask price 12 -> 11.5 gives -80",
          list(ofi_events(book)) == [0.0, 50.0, 50.0, -80.0])
    check("queue imbalance and micro-price by hand: bid 150, ask 200 -> I = -1/7; micro-price = (12·150 + 10·200) / 350",
          abs(queue_imbalance(book)[1] + 1 / 7) < 1e-12 and abs(microprice(book)[1] - (12 * 150 + 10 * 200) / 350) < 1e-12)
    check("next mid move looks only forward: mids 11, 11, 11.5, 11.25 -> up, up, down, then 0", list(next_mid_move(book)) == [1.0, 1.0, -1.0, 0.0])
    g = np.random.default_rng(1)
    x = g.normal(0, 1000, 5000)
    y = 0.002 * x + g.normal(0, 0.5, 5000)
    a, b, r2 = ols(y, x)
    check("OLS recovers a slope of 0.002 (within 3 %); out-of-sample R^2 with the same fit equals the in-sample one",
          abs(b - 0.002) < 0.00006 and abs(out_of_sample_r2(y, x, a, b) - r2) < 1e-12)
    t = np.arange(0, 100, 0.5)
    step = dict(t=t, bid_px=10 + np.floor(t / 7), bid_qty=np.full(len(t), 100.0), ask_px=11 + np.floor(t / 7), ask_qty=np.full(len(t), 100.0))
    dp, _ = ofi_windows(step, 10, 90)
    f = mid_after(step, 10)
    check("10-second windows (eight over 80 s) chain exactly; the mid 10 s later is taken from the last update at or before t + 10; nan past the end",
          len(dp) == 8 and abs(dp.sum() - (np.floor(89.5 / 7) - np.floor(9.5 / 7))) < 1e-12 and f[0] == 10.5 + np.floor(10 / 7) and np.isnan(f[-1]))
    tm = np.arange(0, 20) * 60000 + 7000 * 60000
    bk, cc = to_bars(tm, np.arange(100.0, 120.0), 5)
    check("bars on the UTC clock take each bucket's last minute", list(cc) == [104.0, 109.0, 114.0, 119.0] and list(np.diff(bk)) == [1, 1, 1])
    bb, rr = bar_returns(np.array([1, 2, 4, 5]), np.array([100.0, 110.0, 99.0, 99.0]))
    check("returns only between adjacent bars (2 -> 4 skips a bar and is dropped)", list(bb) == [2, 5] and abs(rr[0] - 0.1) < 1e-12 and rr[1] == 0.0)
    b3, gr, tu = sign_rule(np.array([1, 2, 3, 5, 6]), np.array([0.01, -0.02, 0.03, 0.01, -0.01]), 1)
    check("momentum rule by hand: position = sign of the last return; a flip is 2 units; after a gap it restarts from flat",
          list(b3) == [2, 3, 6] and np.allclose(gr, [-0.02, -0.03, -0.01]) and list(tu) == [1.0, 2.0, 1.0])
    n = 200000
    e = g.normal(0, 0.001, n)
    r = e.copy()
    r[1:] -= 0.3 * e[:-1]
    bucket = np.arange(n)
    _, g_rev, t_rev = sign_rule(bucket, r, -1)
    _, g_mom, _ = sign_rule(bucket, r, 1)
    check("on returns with a planted one-bar reversal the reversal rule earns before costs, momentum loses, and the break-even cost is far below 5 bp",
          g_rev.mean() > 0 > g_mom.mean() and autocorr(r) < -0.2 and g_rev.mean() / t_rev.mean() < 5e-4)
    return check.summary()


if __name__ == "__main__":
    import sys
    sys.exit(0 if selftest() else 1)
