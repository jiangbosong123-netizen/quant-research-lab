"""Same signal, three constructions: equal-weight long-short vs risk-model optimisation (synthetic market).

On top of the synthetic market (risk_model.synthetic_market) we add a predictable "true alpha" a_p per 21-day period (cross-sectional, AR(1) 0.8).
The signal = part of a_p + part of a "low-volatility" style tilt (deliberate: real signals often load on styles) + persistent noise.
Every 21 days, with a risk model estimated from past data only:
  A  equal weight: top 10 % long, bottom 10 % short, 100 % per side (like the lab's live portfolio);
  B  optimised, factor neutral: alpha = IC x specific vol x score; max alpha'w - (lam/2) w'Sigma w with zero exposure to every factor
     (market, styles, industries) and |w_i| <= 4 %; lam tuned so gross exposure is 200 %, like A;
  C  B plus a 0.2 % per-unit trading cost, starting from last period's holdings.
Reports information ratios before/after costs, turnover, correlation with the volatility factor and the factor share of predicted risk.
"""
import numpy as np

from . import optimizer as opt
from . import risk_model as rm
from ._util import Checker

COST = 0.002


def add_alpha(m, period=21, ann_cs=0.017, persistence=0.8, seed=0):
    """Add a predictable cross-sectional alpha a_p (annualised cross-sectional std ann_cs, AR(1) across periods), spread evenly over each period's days."""
    rng = np.random.default_rng(seed)
    T, N = m["r"].shape
    P = T // period
    s = ann_cs * np.sqrt(period / 252)
    a = np.zeros((P, N))
    a[0] = rng.standard_normal(N) * s
    for p in range(1, P):
        a[p] = persistence * a[p - 1] + np.sqrt(1 - persistence ** 2) * rng.standard_normal(N) * s
    r = m["r"].copy()
    for p in range(P):
        r[p * period:(p + 1) * period] += a[p] / period
    return r, a


def signal(a_p, X_t, noise_p, style_weight=0.6, noise=1.5):
    """Signal = standardised a_p - style_weight x (volatility exposure) + noise x noise_p (all with unit cross-sectional std)."""
    z = (a_p - a_p.mean()) / a_p.std()
    vol = X_t[:, rm.STYLE_NAMES.index("volatility")]
    return z - style_weight * vol + noise * noise_p


def next_noise(prev, rng, persistence=0.8):
    """Signal errors persist (AR(1), unit std)."""
    return persistence * prev + np.sqrt(1 - persistence ** 2) * rng.standard_normal(len(prev))


def equal_weight(s, frac=0.1):
    n = len(s)
    k = int(round(n * frac))
    o = np.argsort(s)
    w = np.zeros(n)
    w[o[-k:]] = 1.0 / k
    w[o[:k]] = -1.0 / k
    return w


def at_gross(alpha, S, A, cap, target=2.0, kappa=None, w0=None, steps=14):
    """Bisect log(lam) so that gross exposure sum|w| ~ target. Intermediate solves use tol 1e-6 (only the direction matters);
    the final one uses 1e-8 and must converge."""
    lo, hi = -6.0, 8.0
    for _ in range(steps):
        mid = (lo + hi) / 2
        r = opt.optimize(alpha, S, np.exp(mid), A, np.zeros(A.shape[0]), cap=cap, kappa=kappa, w0=w0, tol=1e-6)
        if np.abs(r["w"]).sum() > target:
            lo = mid
        else:
            hi = mid
    lam = float(np.exp((lo + hi) / 2))
    r = opt.optimize(alpha, S, lam, A, np.zeros(A.shape[0]), cap=cap, kappa=kappa, w0=w0, tol=1e-8)
    if not r["converged"]:
        raise RuntimeError(f"ADMM did not converge (lam = {lam:.3g}, primal residual {r['primal_res']:.1e})")
    return r["w"], lam


def _rank_corr(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def run(N=500, T=1500, start=250, period=21, n_industries=10, seed=11, ic=0.03, cap=0.04, show=True):
    np.seterr(divide="ignore", over="ignore", invalid="ignore")
    m = rm.synthetic_market(N=N, T=T, n_industries=n_industries, seed=seed)
    r, a = add_alpha(m, period=period, seed=seed + 1)
    m["r"] = r
    rng = np.random.default_rng(seed + 2)
    names = ("A equal weight", "B optimised, factor neutral", "C optimised + costs")
    rec = {k: dict(gross=[], turnover=[], factor_share=[], exposure=[]) for k in names}
    prev = {k: np.zeros(N) for k in rec}
    vol_factor, ics = [], []
    noise = rng.standard_normal(N)
    for t in range(start, T - period + 1, period):
        p = t // period
        fm = rm.factor_model(r, m["X"], m["cap"], n_industries, t, window=250)
        X_t = m["X"][t]
        S = rm.model_cov(X_t, fm["F"], fm["D"]) * period
        noise = next_noise(noise, rng)
        s = signal(a[p], X_t, noise)
        alpha = opt.expected_returns(s, np.sqrt(fm["D"] * period), ic)
        A = X_t.T
        ws = {names[0]: equal_weight(s)}
        ws[names[1]], _ = at_gross(alpha, S, A, cap)
        ws[names[2]], _ = at_gross(alpha, S, A, cap, kappa=COST, w0=prev[names[2]])
        ret = r[t:t + period].sum(0)
        ics.append(_rank_corr(s, ret))
        vol_factor.append(m["f"][t:t + period, rm.STYLE_NAMES.index("volatility")].sum())
        for k, w in ws.items():
            dec = rm.risk_decomposition(w, X_t, fm["F"] * period, fm["D"] * period)
            rec[k]["gross"].append(float(np.sum(w * ret)))
            rec[k]["turnover"].append(float(np.abs(w - prev[k]).sum()))
            rec[k]["factor_share"].append(dec["factor_var"] / dec["total_var"])
            rec[k]["exposure"].append(float(np.max(np.abs(dec["exposures"]))))
            prev[k] = w
    out = {}
    per_year = 252 / period
    for k, v in rec.items():
        g = np.array(v["gross"][1:])                                  # the first period builds from zero; its turnover is excluded
        tv = np.array(v["turnover"][1:])
        net = g - COST * tv
        out[k] = dict(periods=len(g), IR_gross=float(g.mean() / g.std() * np.sqrt(per_year)), IR_net=float(net.mean() / net.std() * np.sqrt(per_year)),
                     vol=float(g.std() * np.sqrt(per_year)), turnover=float(tv.mean()), corr_vol_factor=float(np.corrcoef(v["gross"], vol_factor)[0, 1]),
                     factor_share=float(np.mean(v["factor_share"])), max_exposure=float(np.max(v["exposure"])))
    out["realised IC"] = dict(mean=float(np.mean(ics)), std=float(np.std(ics)))
    if show:
        print(f"Synthetic market N = {N}, {len(rec[names[0]]['gross'])} periods of {period} days; realised rank IC {np.mean(ics):.3f} "
              f"(per-period std {np.std(ics):.3f}); signal tilted to low volatility; cost {COST:.1%} per unit traded")
        for k, v in out.items():
            if k == "realised IC":
                continue
            print(f"  {k}: IR gross {v['IR_gross']:.2f}, net {v['IR_net']:.2f}, vol {v['vol']:.1%}, turnover {v['turnover']:.0%}/period, "
                  f"corr with vol factor {v['corr_vol_factor']:+.2f}, factor share of risk {v['factor_share']:.0%}, max factor exposure {v['max_exposure']:.3f}")
    return out


def selftest():
    np.seterr(divide="ignore", over="ignore", invalid="ignore")
    check = Checker()
    m = rm.synthetic_market(N=120, T=300, n_industries=4, seed=2)
    r, a = add_alpha(m, seed=3)
    check("add_alpha: each period's alpha is spread over its days; consecutive periods correlate ~0.8",
          np.allclose(r[:21] - a[0] / 21, m["r"][:21]) and 0.6 < np.corrcoef(a[3], a[4])[0, 1] < 0.95)
    w = equal_weight(np.arange(120.0))
    check("equal weight: top 10 % long, bottom 10 % short; dollar neutral, gross 200 %",
          abs(w.sum()) < 1e-12 and abs(np.abs(w).sum() - 2) < 1e-12 and np.all(w[-12:] > 0) and np.all(w[:12] < 0) and np.all(w[12:-12] == 0))
    fm = rm.factor_model(r, m["X"], m["cap"], 4, 200, window=150)
    X_t = m["X"][200]
    S = rm.model_cov(X_t, fm["F"], fm["D"]) * 21
    s = signal(a[200 // 21], X_t, np.random.default_rng(0).standard_normal(120))
    alpha = opt.expected_returns(s, np.sqrt(fm["D"] * 21), 0.05)
    wB, _ = at_gross(alpha, S, X_t.T, 0.04)
    check(f"optimised factor neutral: all factor exposures < 1e-5 (max {np.max(np.abs(X_t.T @ wB)):.1e}), |w| <= 4 %, gross ~200 % ({np.abs(wB).sum():.3f})",
          np.max(np.abs(X_t.T @ wB)) < 1e-5 and np.all(np.abs(wB) <= 0.04 + 1e-12) and abs(np.abs(wB).sum() - 2) < 0.02)
    noise = [np.zeros(2000)]
    rg = np.random.default_rng(1)
    for _ in range(400):
        noise.append(next_noise(noise[-1], rg))
    noise = np.array(noise[50:])
    ac = np.corrcoef(noise[:-1].ravel(), noise[1:].ravel())[0, 1]
    check(f"signal noise persists: lag-1 correlation ~0.8 ({ac:.3f}), unit std ({noise.std():.3f})", 0.77 < ac < 0.83 and 0.95 < noise.std() < 1.05)
    res = run(N=120, T=500, start=150, n_industries=4, seed=5, show=False)
    grp = {k: v for k, v in res.items() if k != "realised IC"}
    check("small run: all three constructions report the same number of periods; optimised ones are factor neutral; equal weight is not",
          len({v["periods"] for v in grp.values()}) == 1 and grp["B optimised, factor neutral"]["max_exposure"] < 1e-4
          and grp["C optimised + costs"]["max_exposure"] < 1e-4 and grp["A equal weight"]["max_exposure"] > 0.1)
    return check.summary()


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    run()
