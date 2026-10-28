"""Meta-labeling and bet sizing (López de Prado 2018, chapters 3 and 10), as used in Part 10.

A primary model only gives a side s = ±1. Each trade is labelled with the triple-barrier method and a secondary (meta) model
learns the probability p that following the primary model on that trade makes money after costs. Then:
  - filter:    trade only when p > 0.5 (the label already nets costs, so p > 0.5 means "more likely than not to clear costs");
  - bet size:  size = max(0, 2·Phi(z) − 1),  z = (p − 0.5) / sqrt(p (1 − p));
  - baseline D: trade only when the primary model's own |prediction| is large, at the same trade share as the filter —
                the comparison that tells whether the meta model learned anything beyond "big predictions are better".
Triple barrier: from the entry close, follow the next minutes' closes of s·(price / entry − 1); exit at the first close beyond +b (take profit)
or −b (stop loss); otherwise exit at the vertical barrier. Missing minutes are skipped.
Accounting per instrument: w = s·size; entry turnover = |w − position carried from the previous trade| (carried only when that trade ran to
the vertical barrier and this one starts right after it); a barrier exit costs |w| again, and so does a vertical exit with no trade right after.
"""
import math

import numpy as np

from ._util import Checker


def triple_barrier(entry, path, vertical_close, side, barrier):
    """entry [n]; path [n, m] the closes of the next m minutes (nan = missing); vertical_close [n]; side [n] = ±1; barrier [n] > 0 (a return).
    -> (r = side·(exit / entry − 1), touch [n] (1 take-profit, −1 stop-loss, 0 vertical), exit minute [n] (m if vertical))."""
    with np.errstate(invalid="ignore"):
        rp = side[:, None] * (path / entry[:, None] - 1)
        up, dn = rp >= barrier[:, None], rp <= -barrier[:, None]
    hit = up | dn
    any_hit = hit.any(axis=1)
    first = np.where(any_hit, hit.argmax(axis=1), path.shape[1])
    i, f0 = np.arange(len(entry)), np.minimum(first, path.shape[1] - 1)
    r = np.where(any_hit, rp[i, f0], side * (vertical_close / entry - 1))
    touch = np.where(any_hit, np.where(up[i, f0], 1, -1), 0)
    return r, touch, first


def meta_label(r, cost=5e-4):
    """1 if the trade made money after a round trip of costs (2 × one-way cost), else 0."""
    return (r - 2 * cost > 0).astype(float)


def _sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def _rowdot(a, w):
    return (a * w).sum(axis=1)


class LogisticRegression:
    """Minimises the summed log loss + ½·penalty·‖beta‖² (intercept not penalised) by Newton's method."""

    def __init__(self, penalty):
        self.penalty = float(penalty)

    def fit(self, x, z, iters=100):
        a = np.column_stack([np.ones(len(x)), x])
        pen = self.penalty * np.r_[0.0, np.ones(x.shape[1])]
        w = np.zeros(a.shape[1])
        for _ in range(iters):
            p = _sigmoid(_rowdot(a, w))
            grad = (a * (p - z)[:, None]).sum(axis=0) + pen * w
            hess = np.einsum("ni,nj->ij", a * (p * (1 - p))[:, None], a) + np.diag(pen)
            step = np.linalg.solve(hess, grad)
            w = w - step
            if np.max(np.abs(step)) < 1e-10:
                break
        self.w = w
        return self

    def predict(self, x):
        return _sigmoid(self.w[0] + _rowdot(x, self.w[1:]))


def log_loss(p, z):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(z * np.log(p) + (1 - z) * np.log(1 - p)))


def average_rank(x):
    """1-based ranks, ties averaged."""
    o = np.argsort(x, kind="mergesort")
    xs = x[o]
    head = np.r_[0, np.flatnonzero(np.diff(xs) != 0) + 1]
    tail = np.r_[head[1:], len(x)]
    r = np.empty(len(x))
    r[o] = np.repeat((head + tail - 1) / 2.0 + 1, tail - head)
    return r


def auc(p, z):
    """Mann–Whitney: the probability that a random positive gets a higher p than a random negative (ties count one half)."""
    n1, n0 = int(z.sum()), int(len(z) - z.sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((average_rank(p)[z == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def precision_recall(p, z, threshold=0.5):
    take = p > threshold
    prec = float(z[take].mean()) if take.any() else float("nan")
    rec = float((z * take).sum() / z.sum()) if z.sum() > 0 else float("nan")
    return prec, rec


def bet_size(p):
    """Chapter 10: z = (p − 0.5) / sqrt(p (1 − p)), size = 2·Phi(z) − 1 = erf(z / sqrt 2); no bet when p <= 0.5."""
    p = np.clip(np.asarray(p, float), 1e-9, 1 - 1e-9)
    z = (p - 0.5) / np.sqrt(p * (1 - p))
    return np.maximum(0.0, np.vectorize(math.erf)(z / math.sqrt(2)))


def threshold_for_share(abs_pred, share):
    """|prediction| threshold that selects the top round(share · n) trades (baseline D)."""
    k = int(round(share * len(abs_pred)))
    if k <= 0:
        return float("inf")
    return float(np.sort(abs_pred)[::-1][min(k, len(abs_pred)) - 1])


def account(bars, w, price_ret, touch, cost=5e-4, step=4):
    """One instrument, trades in time order (bars = entry bar numbers; consecutive trades are `step` bars apart).
    w = signed position, price_ret = exit / entry − 1 (unsigned). -> (gross, turnover, net) per trade."""
    follows = np.r_[False, np.diff(bars) == step]
    carried = np.where(follows, np.r_[0.0, np.where(touch[:-1] != 0, 0.0, w[:-1])], 0.0)
    next_follows = np.r_[follows[1:], False]
    turnover = np.abs(w - carried) + np.where((touch != 0) | ~next_follows, np.abs(w), 0.0)
    gross = w * price_ret
    return gross, turnover, gross - cost * turnover


def selftest():
    check = Checker()
    path = np.full((4, 60), 100.0)
    path[0, 9], path[0, 20] = 101.2, 98.0
    path[1, 5], path[1, 30] = 98.9, 102.0
    path[2, 7] = 98.9
    path[3, :] = 100.5
    path[3, 3] = np.nan
    r, touch, first = triple_barrier(np.full(4, 100.0), path, np.array([100.0, 100.0, 100.0, 100.4]), np.array([1.0, 1, -1, 1]), np.full(4, 0.01))
    check("triple barrier by hand: first touch wins, exit at that minute's close, short side mirrored, missing minutes skipped, otherwise the vertical close",
          np.allclose(r, [0.012, -0.011, 0.011, 0.004]) and list(touch) == [1, -1, 1, 0] and list(first) == [9, 5, 7, 60])
    check("meta label nets a 10 bp round trip", list(meta_label(np.array([0.0011, 0.0009, -0.01]))) == [1.0, 0.0, 0.0])
    bars = np.array([0, 4, 8, 16, 20, 24])
    w = np.array([1.0, 1.0, -1.0, 1.0, 0.0, 1.0])
    g_, tu, net = account(bars, w, np.array([0.01, 0.02, 0.005, -0.01, 0.03, 0.0]), np.array([0, 0, 1, 0, 0, 0]))
    check("accounting by hand: a same-side trade right after a vertical exit pays nothing, a flip pays 2, a barrier exit 1, a vertical exit before a gap 1, a skipped slot pays to close",
          list(tu) == [1.0, 0.0, 3.0, 1.0, 1.0, 2.0] and np.allclose(net, g_ - 5e-4 * tu))
    g = np.random.default_rng(0)
    n = 20000
    x = g.normal(size=(n, 3))
    z = (g.random(n) < _sigmoid(0.3 + _rowdot(x, np.array([0.8, -0.5, 0.0])))).astype(float)
    m = LogisticRegression(1e-6).fit(x, z)
    big = LogisticRegression(10.0 * n).fit(x, z)
    check("logistic regression recovers known coefficients (within 0.06); a heavy penalty shrinks the slopes but not the intercept",
          np.max(np.abs(m.w - np.r_[0.3, 0.8, -0.5, 0.0])) < 0.06 and np.sum(big.w[1:] ** 2) < 0.01 * np.sum(m.w[1:] ** 2) and abs(_sigmoid(big.w[0]) - z.mean()) < 0.01)
    pr, zr = g.random(300).round(1), (g.random(300) < 0.4).astype(float)
    brute = np.mean([1.0 if a > c else 0.5 if a == c else 0.0 for a in pr[zr == 1] for c in pr[zr == 0]])
    check("AUC with many ties equals the brute-force pairwise count", abs(auc(pr, zr) - brute) < 1e-12)
    s = bet_size(np.array([0.3, 0.5, 0.6, 0.9]))
    check("bet size: 0 at p <= 0.5, 2·Phi(0.204) − 1 at p = 0.6, increasing", s[0] == 0 and s[1] == 0 and abs(s[2] - math.erf(0.1 / math.sqrt(0.24) / math.sqrt(2))) < 1e-12 and s[3] > s[2])
    a = g.random(1000)
    check("baseline D's threshold selects exactly the requested share", int((a >= threshold_for_share(a, 0.3)).sum()) == 300 and threshold_for_share(a, 0.0) == float("inf"))
    # the pitfall seen on real data: with a fixed cost in the label, a side chosen by coin flip clears the cost more often when volatility is high
    sig = np.repeat([0.002, 0.004, 0.008], 40000)
    rr = g.normal(0, sig)
    lab = meta_label(rr)
    rates = [lab[sig == v].mean() for v in (0.002, 0.004, 0.008)]
    check(f"no skill, fixed 10 bp cost: the label rate rises with volatility ({rates[0]:.2f} -> {rates[1]:.2f} -> {rates[2]:.2f}) — a cost-netted label also learns volatility",
          rates[0] < rates[1] < rates[2] < 0.5 and rates[0] < 0.35)
    # end to end: the primary model is right 58 % of the time in regime 1 and 50 % in regime 0; the meta model should find the regime
    n = 30000
    regime = (g.random(n) < 0.5).astype(float)
    side = np.where(g.random(n) < 0.5, 1.0, -1.0)
    right = g.random(n) < np.where(regime == 1, 0.58, 0.50)
    move = np.abs(g.normal(0, 0.006, n))
    price_ret = side * np.where(right, move, -move)
    zz = meta_label(side * price_ret)
    feats = np.column_stack([regime, g.normal(size=n), g.normal(size=n)])
    tr, te = slice(0, 15000), slice(15000, None)
    p = LogisticRegression(1.0).fit(feats[tr], zz[tr]).predict(feats)
    take = (p > np.median(p[tr])).astype(float)
    bars = np.arange(n) * 4
    _, _, net_all = account(bars[te], side[te], price_ret[te], np.ones(n - 15000))
    _, _, net_meta = account(bars[te], side[te] * take[te], price_ret[te], np.ones(n - 15000))
    check("planted regime: the meta model ranks trades (AUC > 0.53) and trading only its upper half earns more per trade than trading everything",
          auc(p[te], zz[te]) > 0.53 and net_meta.sum() / take[te].sum() > net_all.sum() / (n - 15000))
    return check.summary()


if __name__ == "__main__":
    import sys
    sys.exit(0 if selftest() else 1)
