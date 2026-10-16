"""The factor test used by the research process (Part 1), with its validation on simulated data.

Monthly cross-sectional Spearman IC between a signal and next-month excess returns; quintile portfolios; a long-short spread net of costs
(each replaced name = one sale + one purchase at 0.2 % one-way, plus 1 %/yr borrow on the short leg).

Main statistic: the EWC test of the mean IC (equal-weighted cosine HAC; Lazarus, Lewis, Stock & Watson 2018):
    nu = floor(0.4 T^(2/3));  Lambda_j = sqrt(2/T) sum_t cos(pi j (t - 1/2) / T) (IC_t - mean);  Omega = mean_j Lambda_j^2
    t = sqrt(T) mean / sqrt(Omega) ~ t_nu under the null;  reported as a normal-equivalent t = Phi^-1(1 - p).
Newey-West (Bartlett) is reported alongside. Why EWC: monthly ICs are autocorrelated, and NW with normal critical values over-rejects in
samples of this size; EWC's false-positive rate stays at the nominal level (checked below), at the cost of some power.

Power: with true mean IC mu and long-run IC std s (s = sd sqrt((1+rho)/(1-rho)) for AR(1) ICs), delta = mu sqrt(T) / s;
EWC power ~ Phi((delta - c_nu (1 - 1/(4 nu))) / sqrt(1 + c_nu^2 / (2 nu))), c_nu the t_nu quantile with the normal tail of c.
Run `python -m quantlab.factor_test` for the validation tables.
"""
import math
from statistics import NormalDist

import numpy as np

from ._util import Checker

_N = NormalDist()


class LookAheadError(Exception):
    """A signal value became available after the allowed cutoff: the signal code used the future. Stop, never drop rows silently."""


# ---------- distributions ----------

def _betacf(a, b, x):
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 400):
        m2 = 2 * m
        for aa in (m * (b - m) * x / ((qam + m2) * (a + m2)), -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))):
            d = 1.0 + aa * d
            d = 1.0 / (d if abs(d) > tiny else tiny)
            c = 1.0 + aa / c
            c = c if abs(c) > tiny else tiny
            h *= d * c
        if abs(d * c - 1.0) < 3e-16:
            break
    return h


def _betai(a, b, x):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    bt = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x))
    if x < (a + 1) / (a + b + 2):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1 - x) / b


def t_sf(t, nu):
    """P(T_nu > t)."""
    p = 0.5 * _betai(nu / 2.0, 0.5, nu / (nu + t * t))
    return p if t > 0 else 1.0 - p


def t_ppf(q, nu):
    """Upper quantile: c with P(T_nu > c) = q (bisection)."""
    lo, hi = -60.0, 60.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if t_sf(mid, nu) > q:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# ---------- statistics ----------

def ranks(x):
    """Average ranks (ties share the mean rank), starting at 1."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    new = np.empty(n, dtype=bool)
    new[0] = True
    new[1:] = xs[1:] != xs[:-1]
    starts = np.flatnonzero(new)
    ends = np.r_[starts[1:], n] - 1
    r = np.empty(n)
    r[order] = ((starts + ends) / 2.0 + 1.0)[np.cumsum(new) - 1]
    return r


def spearman(x, y):
    rx, ry = ranks(x), ranks(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    den = math.sqrt(float((rx * rx).sum()) * float((ry * ry).sum()))
    return float((rx * ry).sum()) / den if den else float("nan")


def ewc_dof(T):
    return max(2, int(math.floor(0.4 * T ** (2.0 / 3.0))))


def ewc(x):
    """EWC test of the mean -> dict(mean, t, nu, p (one-sided), eq_t)."""
    x = np.asarray(x, dtype=float)
    T = len(x)
    nu = ewc_dof(T)
    e = x - x.mean()
    tt = np.arange(1, T + 1)
    lam = np.array([math.sqrt(2.0 / T) * float(np.sum(np.cos(math.pi * j * (tt - 0.5) / T) * e)) for j in range(1, nu + 1)])
    om = float((lam ** 2).mean())
    t = math.sqrt(T) * float(x.mean()) / math.sqrt(om)
    p = t_sf(t, nu)
    eq = -_N.inv_cdf(p) if 0 < p < 1 else (8.0 if p <= 0 else -8.0)
    return dict(mean=float(x.mean()), t=t, nu=nu, p=p, eq_t=eq)


def nw_lag(T, update_months=1):
    """12 for annually updated signals; otherwise floor(4 (T/100)^(2/9))."""
    return 12 if update_months >= 12 else max(1, int(math.floor(4 * (T / 100.0) ** (2.0 / 9.0))))


def nw_t(x, lag):
    x = np.asarray(x, float)
    T = len(x)
    e = x - x.mean()
    S = float(np.sum(e * e)) / T
    for k in range(1, lag + 1):
        S += 2 * (1 - k / (lag + 1)) * float(np.sum(e[k:] * e[:-k])) / T
    return float(x.mean() / math.sqrt(S / T))


def long_run_sd(sd, rho):
    return sd * math.sqrt((1 + rho) / (1 - rho))


def _Phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def power(mu, sd_lr, T, c, stat="EWC"):
    """P(normal-equivalent t > c) for true mean IC mu and long-run IC std sd_lr."""
    d = mu * math.sqrt(T) / sd_lr
    if stat == "NW":
        return _Phi(d - c)
    nu = ewc_dof(T)
    cn = t_ppf(1.0 - _Phi(c), nu)
    return _Phi((d - cn * (1 - 1 / (4 * nu))) / math.sqrt(1 + cn * cn / (2 * nu)))


def min_detectable_ic(sd_lr, T, c, target=0.8, stat="EWC"):
    lo, hi = 0.0, 1.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if power(mid, sd_lr, T, c, stat) < target:
            lo = mid
        else:
            hi = mid
    return hi


def spread_to_ic(spread, cs_vol=0.09):
    """Quintile long-short monthly spread -> IC ~ spread / (2.80 sigma_cs) (normal scores, linear returns)."""
    return spread / (2.80 * cs_vol)


def bonferroni_threshold(n_tests, one_sided_alpha=0.025):
    return -_N.inv_cdf(one_sided_alpha / n_tests)


# ---------- the test ----------

def factor_test(months, threshold=1.96, min_stocks=200, min_months=120, cost_one_way=0.002, borrow=0.01, cutoffs=None):
    """months: list of dicts with 'signal' (direction already applied: higher = better), 'ret' (next-month return), 'id' (stock ids),
    optional 'available' (availability dates, comparable with cutoffs[i]). Missing returns are NaN and dropped.
    -> dict(n_months, ic (array), ewc, nw_t, quintiles (mean excess per quintile), spread_gross, cost, spread_net, verdict, promoted)."""
    ics, quint, spreads, costs, prev = [], [], [], [], None
    for i, m in enumerate(months):
        if cutoffs is not None and "available" in m and any(a > cutoffs[i] for a in m["available"]):
            raise LookAheadError(f"month {i}: a signal value became available after {cutoffs[i]}")
        ok = ~np.isnan(m["ret"])
        s, r, ids = np.asarray(m["signal"])[ok], np.asarray(m["ret"])[ok], np.asarray(m["id"])[ok]
        if len(s) < min_stocks:
            prev = None
            continue
        exc = r - r.mean()                                            # excess over the equal-weight benchmark
        ics.append(spearman(s, exc))
        order = np.lexsort((ids, s))                                  # sort by signal, ties by id (deterministic)
        n = len(s)
        grp = np.empty(n, int)
        grp[order] = (5 * np.arange(n)) // n
        q = np.array([exc[grp == g].mean() for g in range(5)])
        quint.append(q)
        legs = (set(ids[grp == 4]), set(ids[grp == 0]))
        tau = [len(now - old) / len(now) for now, old in zip(legs, prev)] if prev else None
        costs.append(None if tau is None else 2 * cost_one_way * sum(tau) + borrow / 12)
        spreads.append(q[4] - q[0])
        prev = legs
    T = len(ics)
    if T < min_months:
        return dict(n_months=T, verdict="unusable")
    ic = np.array(ics)
    e = ewc(ic)
    c = np.array([x for x in costs if x is not None])
    net = float(np.mean(spreads)) - float(c.mean())
    passed = e["eq_t"] > threshold and e["mean"] > 0 and net > 0
    return dict(n_months=T, ic=ic, ewc=e, nw_t=nw_t(ic, nw_lag(T)), quintiles=np.mean(quint, 0), spread_gross=float(np.mean(spreads)),
                cost=float(c.mean()), spread_net=net, verdict="pass" if passed else "fail", promoted=passed and e["eq_t"] > 3)


def synthetic_panel(N=600, T=150, ic=0.0, ic_sd=0.055, rho=0.0, cs_vol=0.09, missing=0.02, seed=0):
    """Signals z ~ N(0,1) refreshed monthly; next-month returns = market + ic_t * cs_vol * z + idiosyncratic, with ic_t = ic + AR(1) noise."""
    rng = np.random.default_rng(seed)
    u = np.empty(T)
    e = rng.normal(size=T)
    u[0] = ic_sd * e[0]
    for t in range(1, T):
        u[t] = rho * u[t - 1] + math.sqrt(1 - rho * rho) * ic_sd * e[t]
    ic_t = np.clip(ic + u, -0.95, 0.95)
    out = []
    for t in range(T):
        z = rng.normal(size=N)
        r = rng.normal(0, 0.04) + ic_t[t] * cs_vol * z + math.sqrt(cs_vol ** 2 * (1 - ic_t[t] ** 2)) * rng.normal(size=N)
        r = np.where(rng.random(N) < missing, np.nan, r)
        out.append(dict(signal=z, ret=r, id=np.arange(N)))
    return out


# ---------- validation on simulated IC series ----------

def ic_series(T, mu, sd, rho, rng):
    e = rng.normal(size=T)
    u = np.empty(T)
    u[0] = sd * e[0]
    for t in range(1, T):
        u[t] = rho * u[t - 1] + math.sqrt(1 - rho * rho) * sd * e[t]
    return mu + u


def false_positive_rates(T=185, sd=0.06, rhos=(0.0, 0.3, 0.5), reps=20000, seed=0):
    """Share of zero-signal IC series with one-sided t > 1.96: EWC vs Newey-West (lag 4 and 12). Nominal: 2.5 %."""
    rng = np.random.default_rng(seed)
    out = {}
    for rho in rhos:
        n_ewc = n_nw4 = n_nw12 = 0
        for _ in range(reps):
            x = ic_series(T, 0.0, sd, rho, rng)
            n_ewc += ewc(x)["eq_t"] > 1.96
            n_nw4 += nw_t(x, 4) > 1.96
            n_nw12 += nw_t(x, 12) > 1.96
        out[rho] = dict(EWC=n_ewc / reps, NW4=n_nw4 / reps, NW12=n_nw12 / reps)
    return out


def power_check(T=185, sd=0.06, rho=0.3, mus=(0.01, 0.02, 0.03), c=3.0, reps=4000, seed=1):
    """Simulated P(EWC eq. t > c) vs the power formula with the long-run std."""
    rng = np.random.default_rng(seed)
    lr = long_run_sd(sd, rho)
    return {mu: dict(simulated=float(np.mean([ewc(ic_series(T, mu, sd, rho, rng))["eq_t"] > c for _ in range(reps)])),
                     formula=power(mu, lr, T, c)) for mu in mus}


def selftest():
    check = Checker()
    check("t distribution: textbook quantiles t(0.025, 10) = 2.228, t(0.05, 20) = 1.725, t(0.01, 5) = 3.365",
          abs(t_ppf(0.025, 10) - 2.228) < 1e-3 and abs(t_ppf(0.05, 20) - 1.725) < 1e-3 and abs(t_ppf(0.01, 5) - 3.365) < 1e-3)
    check("ranks: ties share the mean rank", np.array_equal(ranks([10, 20, 20, 30]), [1, 2.5, 2.5, 4]))
    check("Spearman: monotone transform = 1, reversed = -1", abs(spearman(np.arange(9.0), np.exp(np.arange(9.0))) - 1) < 1e-12
          and abs(spearman(np.arange(9.0), -np.arange(9.0)) + 1) < 1e-12)
    check("EWC degrees of freedom: T = 185 -> 12, T = 120 -> 9", ewc_dof(185) == 12 and ewc_dof(120) == 9)
    rng = np.random.default_rng(0)
    x = rng.normal(0.02, 0.06, 185)
    e = ewc(x)
    check("EWC: equivalent t equals the normal quantile of its t_nu p-value", abs(_N.cdf(e["eq_t"]) - (1 - e["p"])) < 1e-9 and e["nu"] == 12)
    fpr = false_positive_rates(reps=3000, seed=2)
    check(f"zero signal, 3000 simulated IC series each: EWC rejects {fpr[0.0]['EWC']:.1%} / {fpr[0.3]['EWC']:.1%} / {fpr[0.5]['EWC']:.1%} "
          f"(rho 0 / 0.3 / 0.5; nominal 2.5 %)", all(0.015 < v["EWC"] < 0.036 for v in fpr.values()))
    check(f"Newey-West with normal critical values over-rejects as autocorrelation grows (lag 4: {fpr[0.0]['NW4']:.1%} -> {fpr[0.5]['NW4']:.1%})",
          fpr[0.5]["NW4"] > fpr[0.5]["EWC"] and fpr[0.5]["NW4"] > 0.03)
    pc = power_check(reps=1500, seed=3)
    check("power formula (with the long-run std) matches simulation within 6 points at mu = 0.01 / 0.02 / 0.03: "
          + ", ".join(f"{v['simulated']:.0%} vs {v['formula']:.0%}" for v in pc.values()), all(abs(v["simulated"] - v["formula"]) < 0.06 for v in pc.values()))
    check(f"minimum detectable IC (T = 185, long-run sd 0.08, c = 1.96, 80 % power): {min_detectable_ic(0.08, 185, 1.96):.4f} (spec: about 0.018)",
          0.017 < min_detectable_ic(0.08, 185, 1.96) < 0.019)
    check(f"Bonferroni: 3 tests -> one-sided threshold {bonferroni_threshold(3):.3f} (2.394)", abs(bonferroni_threshold(3) - 2.394) < 1e-3)
    strong = factor_test(synthetic_panel(N=600, T=150, ic=0.05, seed=4))
    null = factor_test(synthetic_panel(N=600, T=150, ic=0.0, seed=5))
    short = factor_test(synthetic_panel(N=600, T=100, ic=0.05, seed=6))
    check(f"full test on synthetic panels: IC 0.05 -> pass (eq. t {strong['ewc']['eq_t']:.1f}, quintiles increasing); zero signal -> fail; 100 months -> unusable",
          strong["verdict"] == "pass" and np.all(np.diff(strong["quintiles"]) > 0) and null["verdict"] == "fail" and short["verdict"] == "unusable")
    check(f"recovered mean IC {strong['ewc']['mean']:.4f} for an injected 0.05 (expected ~0.048 since Spearman ~ 0.955 x Pearson; standard error ~0.005)",
          0.035 < strong["ewc"]["mean"] < 0.062)
    try:
        factor_test([dict(signal=np.zeros(3), ret=np.zeros(3), id=np.arange(3), available=["2020-02-01"] * 3)], cutoffs=["2020-01-30"])
        caught = False
    except LookAheadError:
        caught = True
    check("look-ahead guard: a signal available after the cutoff stops the test", caught)
    return check.summary()


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    print("False-positive rates under a zero signal (T = 185, 20,000 IC series each; nominal 2.5 %):")
    for rho, v in false_positive_rates().items():
        print(f"   IC autocorrelation {rho}: EWC {v['EWC']:.2%}, Newey-West lag 4 {v['NW4']:.2%}, lag 12 {v['NW12']:.2%}")
    print("Power for 'promotion' (EWC eq. t > 3; T = 185, IC sd 0.06, autocorrelation 0.3):")
    for mu, v in power_check().items():
        print(f"   true mean IC {mu}: simulated {v['simulated']:.1%}, formula {v['formula']:.1%}")
