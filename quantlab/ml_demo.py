"""Three ML experiments on the synthetic panel (run: python -m quantlab.ml_demo [1|2|3]; all three take ~2 minutes).

1. When does ML beat linear? Nonlinear truth -> boosted trees beat ridge out of sample; linear truth -> ridge wins.
2. Shuffled CV cheats: on pure noise with overlapping 3-month labels, shuffled K-fold finds a "significant" IC; purged / walk-forward do not.
3. Best of many trials: try 30 hyper-parameter sets on pure noise and pick the best; it fails on fresh data and the deflated Sharpe flags it.
   The same search on data with a real signal survives deflation and holds up on fresh data.
"""
import sys

import numpy as np

from . import gbdt
from . import ml_data as md
from . import validation as va
from ._util import Checker


def shallow():
    return gbdt.GBDT(n_rounds=80, learning_rate=0.1, max_depth=3, min_leaf=50, subsample=0.8)


def deep():
    return gbdt.GBDT(n_rounds=80, learning_rate=0.1, max_depth=6, min_leaf=5, subsample=0.8)


def exp1(P=180, N=300, seed=1, show=True):
    out = {}
    for nonlinear in (True, False):
        d = md.generate(P=P, N=N, nonlinear=nonlinear, target_ic=0.06, seed=seed)
        splits = va.walk_forward(P, first_train=60, test_len=24, h=1)
        truth = float(np.mean([md.rank_corr(d["signal"][p], d["y"][p]) for p in range(60, P)]))
        r = {"GBDT": va.oos_ic(shallow, d["X"], d["y"], splits), "ridge": va.oos_ic(lambda: gbdt.Ridge(10.0), d["X"], d["y"], splits)}
        out["nonlinear" if nonlinear else "linear"] = dict(true_signal_ic=truth, **{k: dict(ic=v["mean"], t_nw=v["t_nw"]) for k, v in r.items()})
    if show:
        print("1. When does ML beat linear? (walk-forward, out of sample; 'true signal IC' is the best achievable)")
        for k, v in out.items():
            print(f"   {k} truth: true signal IC {v['true_signal_ic']:.4f}; GBDT IC {v['GBDT']['ic']:.4f} (t_NW {v['GBDT']['t_nw']:.1f}); "
                  f"ridge IC {v['ridge']['ic']:.4f} (t_NW {v['ridge']['t_nw']:.1f})")
    return out


def exp2(P=120, N=200, seeds=(4, 5, 6), show=True):
    out = {}
    for sd in seeds:
        d = md.generate(P=P, N=N, target_ic=0.0, seed=sd)
        y3 = md.overlapping_labels(d["y"], 3)
        r = {"shuffled K-fold": va.oos_ic(deep, d["X"], y3, va.shuffled_kfold(P, N, 5), by_row=True),
             "purged K-fold": va.oos_ic(deep, d["X"], y3, va.purged_kfold(P, 5, h=3, embargo=1)),
             "walk-forward": va.oos_ic(deep, d["X"], y3, va.walk_forward(P, first_train=48, test_len=24, h=3))}
        out[sd] = {k: dict(ic=v["mean"], t=v["t"], t_nw=v["t_nw"]) for k, v in r.items()}
    if show:
        print("2. Pure noise (no signal at all), overlapping 3-month labels, deeper trees (depth 6, min leaf 5):")
        for sd, r in out.items():
            print(f"   seed {sd}: " + "; ".join(f"{k} IC {v['ic']:+.4f} (t {v['t']:.1f}, t_NW {v['t_nw']:.1f})" for k, v in r.items()))
    return out


def configs(n=30, seed=0):
    rng = np.random.default_rng(seed)
    grid = [(d, lr, nr, ml, ss, cs) for d in (2, 3, 5) for lr in (0.05, 0.1) for nr in (30, 80) for ml in (20, 80) for ss in (0.6, 1.0) for cs in (0.5, 1.0)]
    return [grid[i] for i in rng.choice(len(grid), n, replace=False)]


def exp3(P=120, N=200, n_configs=30, show=True):
    out = {}
    for name, ic, seed, new_seed in (("pure noise", 0.0, 21, 22), ("real signal (nonlinear, IC 0.06)", 0.06, 31, 32)):
        d = md.generate(P=P, N=N, target_ic=ic, seed=seed)
        fresh = md.generate(P=P, N=N, target_ic=ic, seed=new_seed)
        splits = va.walk_forward(P, first_train=48, test_len=24, h=1)
        results = []
        for c in configs(n_configs):
            make = (lambda c=c: gbdt.GBDT(n_rounds=c[2], learning_rate=c[1], max_depth=c[0], min_leaf=c[3], subsample=c[4], colsample=c[5]))
            r = va.oos_ic(make, d["X"], d["y"], splits)
            results.append((r["mean"] / r["monthly_ic"].std(ddof=1), r, c, make))      # "Sharpe" = mean monthly IC / its std (per month)
        srs = np.array([x[0] for x in results])
        best = max(results, key=lambda x: x[0])
        T = len(best[1]["monthly_ic"])
        rf = va.oos_ic(best[3], fresh["X"], fresh["y"], va.walk_forward(P, first_train=48, test_len=24, h=1))
        out[name] = dict(best_config=best[2], best_monthly_sr=float(best[0]), best_ic=best[1]["mean"], annual_ir=float(best[0] * np.sqrt(12)),
                         psr=va.probabilistic_sharpe(best[0], T), dsr=va.deflated_sharpe(best[0], T, n_configs, float(srs.var(ddof=1))),
                         expected_max=va.expected_max_sharpe(n_configs, float(srs.var(ddof=1))), fresh_ic=rf["mean"], fresh_t_nw=rf["t_nw"])
    if show:
        print(f"3. Try {n_configs} hyper-parameter sets, keep the best walk-forward result:")
        for name, v in out.items():
            print(f"   {name}: best IC {v['best_ic']:+.4f}, annual IR {v['annual_ir']:.2f}; P(skill > 0) on its own {v['psr']:.0%}; "
                  f"deflated for {n_configs} trials (null-max threshold {v['expected_max'] * np.sqrt(12):.2f} annualised) {v['dsr']:.0%}; "
                  f"on fresh data IC {v['fresh_ic']:+.4f} (t_NW {v['fresh_t_nw']:.1f})")
    return out


def selftest():
    np.seterr(divide="ignore", over="ignore", invalid="ignore")
    check = Checker()
    c = configs(30)
    check("30 distinct configurations from the grid", len(set(c)) == 30 and all(x[0] in (2, 3, 5) and x[1] in (0.05, 0.1) for x in c))
    r1 = exp1(P=72, N=80, show=False)
    check("experiment 1 (small) runs: both truths, both models report IC and t_NW",
          set(r1) == {"nonlinear", "linear"} and all(np.isfinite(v[m]["ic"]) and np.isfinite(v[m]["t_nw"]) for v in r1.values() for m in ("GBDT", "ridge")))
    r2 = exp2(P=60, N=60, seeds=(4,), show=False)
    check("experiment 2 (small) runs: three splitting schemes", set(r2[4]) == {"shuffled K-fold", "purged K-fold", "walk-forward"})
    r3 = exp3(P=72, N=60, n_configs=4, show=False)
    check("experiment 3 (small) runs: probabilities in [0, 1]; deflated <= probabilistic Sharpe", all(0 <= v["dsr"] <= v["psr"] <= 1 for v in r3.values()))
    return check.summary()


if __name__ == "__main__":
    np.seterr(divide="ignore", over="ignore", invalid="ignore")
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    for k in (sys.argv[1:] or ["1", "2", "3"]):
        {"1": exp1, "2": exp2, "3": exp3}[k]()
