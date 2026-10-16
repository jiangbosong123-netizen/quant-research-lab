"""How to validate models on financial panels, and how to discount for the number of trials.

Splits (which months train, which test):
  shuffled K-fold   shuffle all stock-month rows into K folds: the textbook default, which leaks on financial data
                    (persistent characteristics and overlapping labels put a near-copy of each test row in the training set);
  purged K-fold     contiguous blocks of months; training months whose label window overlaps the test block's are removed (purging),
                    plus an embargo after the block (Lopez de Prado 2018, ch. 7);
  walk-forward      expanding window, training only on months whose labels are known before the test block starts.
Evaluation: out-of-sample monthly rank ICs, with a plain t and a Newey-West t (monthly ICs are autocorrelated).

Selection bias (Bailey & Lopez de Prado 2014):
  expected maximum Sharpe of N skill-less trials ~ sqrt(V) [(1-g) Phi^-1(1-1/N) + g Phi^-1(1-1/(N e))]  (g = Euler-Mascheroni);
  deflated Sharpe = probabilistic Sharpe with that as the threshold. It is conservative: with no skill it almost never exceeds 0.95.
"""
import math
from statistics import NormalDist

import numpy as np

from ._util import Checker

_N = NormalDist()
EULER_GAMMA = 0.5772156649015329


def shuffled_kfold(P, N, K=5, seed=0):
    """Row = month*N + stock; shuffled into K folds -> [(train rows, test rows)]."""
    rng = np.random.default_rng(seed)
    parts = np.array_split(rng.permutation(P * N), K)
    return [(np.sort(np.concatenate([parts[j] for j in range(K) if j != k])), np.sort(parts[k])) for k in range(K)]


def purged_kfold(P, K=5, h=1, embargo=0, usable=None):
    """Contiguous month blocks -> [(train months, test months)]. Test block [a, b] has label window [a, b+h-1]; training month p (label window
    [p, p+h-1]) is dropped if a-h+1 <= p <= b+h-1, and `embargo` more months after the block are dropped."""
    usable = np.arange(P - h + 1) if usable is None else np.asarray(usable)
    out = []
    for s in np.array_split(usable, K):
        a, b = s[0], s[-1]
        out.append((usable[(usable < a - h + 1) | (usable > b + h - 1 + embargo)], s))
    return out


def walk_forward(P, first_train=60, test_len=24, h=1, usable=None):
    """Expanding window -> [(train months, test months)]; test block [a, b] trains on p <= a-h (labels known before a)."""
    usable = np.arange(P - h + 1) if usable is None else np.asarray(usable)
    out = []
    a = first_train
    while a <= usable[-1]:
        b = min(a + test_len - 1, usable[-1])
        tr = usable[usable <= a - h]
        te = usable[(usable >= a) & (usable <= b)]
        if len(tr) and len(te):
            out.append((tr, te))
        a = b + 1
    return out


def months_to_rows(months, N):
    return (np.asarray(months)[:, None] * N + np.arange(N)).ravel()


def nw_t(x, lags=6):
    """Newey-West t of the mean of a time-ordered series: var(mean) = [g0 + 2 sum_{l<=L} (1 - l/(L+1)) g_l] / n."""
    x = np.asarray(x, float)
    n = len(x)
    e = x - x.mean()
    v = float(np.sum(e * e)) / n
    for l in range(1, min(lags, n - 1) + 1):
        v += 2 * (1 - l / (lags + 1)) * float(np.sum(e[l:] * e[:-l])) / n
    return float(x.mean() / np.sqrt(v / n))


def oos_ic(make_model, X, y, splits, by_row=False, lags=6):
    """X: (P, N, F); y: (P, N) labels (may contain NaN). by_row: splits are row indices (shuffled K-fold), otherwise months.
    -> dict(months, monthly_ic (month-ordered; averaged when a month is tested in several folds), mean, t (as if independent), t_nw)."""
    P, N, F = X.shape
    X2, y2 = X.reshape(P * N, F), y.reshape(P * N)
    month = np.repeat(np.arange(P), N)
    rec = {}
    for tr_, te_ in splits:
        tr = tr_ if by_row else months_to_rows(tr_, N)
        te = te_ if by_row else months_to_rows(te_, N)
        tr, te = tr[~np.isnan(y2[tr])], te[~np.isnan(y2[te])]
        pr = make_model().fit(X2[tr], y2[tr]).predict(X2[te])
        for p in np.unique(month[te]):
            sel = month[te] == p
            if sel.sum() >= 20:
                ra, rb = np.argsort(np.argsort(pr[sel])), np.argsort(np.argsort(y2[te][sel]))
                rec.setdefault(int(p), []).append(float(np.corrcoef(ra, rb)[0, 1]))
    months = sorted(rec)
    ics = np.array([np.mean(rec[p]) for p in months])
    return dict(months=np.array(months), monthly_ic=ics, mean=float(ics.mean()),
                t=float(ics.mean() / ics.std(ddof=1) * np.sqrt(len(ics))), t_nw=nw_t(ics, lags))


def expected_max_sharpe(n_trials, var):
    if n_trials <= 1:
        return 0.0
    return math.sqrt(var) * ((1 - EULER_GAMMA) * _N.inv_cdf(1 - 1 / n_trials) + EULER_GAMMA * _N.inv_cdf(1 - 1 / (n_trials * math.e)))


def probabilistic_sharpe(sr, T, threshold=0.0, skew=0.0, kurt=3.0):
    """P(true Sharpe > threshold) for a per-period sample Sharpe sr over T periods, with skewness / kurtosis corrections."""
    return _N.cdf((sr - threshold) * math.sqrt(T - 1) / math.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr ** 2))


def deflated_sharpe(sr, T, n_trials, var_of_trial_sharpes, skew=0.0, kurt=3.0):
    return probabilistic_sharpe(sr, T, expected_max_sharpe(n_trials, var_of_trial_sharpes), skew, kurt)


def selftest():
    np.seterr(divide="ignore", over="ignore", invalid="ignore")
    check = Checker()
    folds = shuffled_kfold(10, 7, K=5)
    check("shuffled K-fold: each row tested once; train and test disjoint", sorted(np.concatenate([t for _, t in folds]).tolist()) == list(range(70))
          and all(len(np.intersect1d(a, b)) == 0 for a, b in folds))
    h = 3
    folds = purged_kfold(60, K=4, h=h, embargo=2)
    win = lambda p: set(range(p, p + h))
    no_overlap = all(not (win(p) & set().union(*[win(q) for q in te])) for tr, te in folds for p in tr)
    check("purged K-fold (h = 3, embargo 2): no training label window touches a test label window; embargo respected; each usable month tested once",
          no_overlap and sorted(np.concatenate([t for _, t in folds]).tolist()) == list(range(58))
          and all((te[-1] + h - 1 + 2 not in tr) and (te[-1] + h + 2 in tr or te[-1] + h + 2 > 57) for tr, te in folds))
    wf = walk_forward(100, first_train=40, test_len=20, h=h)
    check("walk-forward (h = 3): every training label is known before the test block (p + h - 1 < a); blocks are contiguous",
          all(tr.max() + h - 1 < te[0] for tr, te in wf) and [te[0] for _, te in wf] == [40, 60, 80] and wf[-1][1][-1] == 97)
    rng = np.random.default_rng(0)
    iid = rng.standard_normal(2000)
    ar = np.zeros(2000)
    for i in range(1, 2000):
        ar[i] = 0.6 * ar[i - 1] + rng.standard_normal()
    t_plain, t_nw = ar.mean() / ar.std(ddof=1) * np.sqrt(2000), nw_t(ar, 20)
    check(f"Newey-West: ~ plain t for iid data ({nw_t(iid, 6):.2f} vs {iid.mean() / iid.std(ddof=1) * np.sqrt(2000):.2f}); "
          f"for AR(1) 0.6 the plain t is ~sqrt(1.6/0.4) = 2x too large ({t_plain:.2f} vs {t_nw:.2f})",
          abs(nw_t(iid, 6) - iid.mean() / iid.std(ddof=1) * np.sqrt(2000)) < 0.3 and 1.6 < abs(t_plain / t_nw) < 2.3)
    for n in (10, 100):
        sim = rng.standard_normal((20000, n)).max(1).mean()
        check(f"expected max Sharpe (N = {n}): formula {expected_max_sharpe(n, 1.0):.3f} vs simulation {sim:.3f} (within 3 %)",
              abs(expected_max_sharpe(n, 1.0) - sim) / sim < 0.03)
    check("expected max Sharpe: 0 for one trial; scales with sqrt(var)", expected_max_sharpe(1, 1.0) == 0.0
          and abs(expected_max_sharpe(50, 4.0) - 2 * expected_max_sharpe(50, 1.0)) < 1e-12)
    check("probabilistic Sharpe: matches the hand formula; 0.5 at the threshold; deflated with one trial = probabilistic",
          abs(probabilistic_sharpe(0.1, 120) - _N.cdf(0.1 * math.sqrt(119) / math.sqrt(1 + 0.01 / 2))) < 1e-12 and abs(probabilistic_sharpe(0.2, 60, 0.2) - 0.5) < 1e-12
          and abs(deflated_sharpe(0.1, 120, 1, 0.01) - probabilistic_sharpe(0.1, 120)) < 1e-12)
    T, M = 120, 100
    naive = deflated = 0
    for _ in range(400):
        r = rng.standard_normal((T, M))
        SR = r.mean(0) / r.std(0, ddof=1)
        b = SR.max()
        naive += probabilistic_sharpe(b, T) > 0.95
        deflated += deflated_sharpe(b, T, M, SR.var(ddof=1)) > 0.95
    check(f"simulation: best of 100 skill-less strategies looks significant {naive / 400:.0%} of the time; after deflation {deflated / 400:.0%} (conservative)",
          naive / 400 > 0.9 and deflated / 400 < 0.05)
    return check.summary()


if __name__ == "__main__":
    import sys
    sys.exit(0 if selftest() else 1)
