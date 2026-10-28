"""Regenerate every figure in figures/ from the quantlab code (about 1.5 minutes; NumPy only; deterministic, so a re-run gives identical files).

Usage: python make_figures.py            Figure 9 reads data/real_data_monthly.csv (aggregate portfolio returns from the private pipeline).
"""
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from quantlab import execution as ex  # noqa: E402
from quantlab import factor_test as ft  # noqa: E402
from quantlab import ml_demo  # noqa: E402
from quantlab import options as op  # noqa: E402
from quantlab import portfolio_demo as pd_  # noqa: E402
from quantlab import risk_model as rm  # noqa: E402
from quantlab.svg import Figure, bar_chart  # noqa: E402

OUT = os.path.join(HERE, "figures")
pct = lambda v: f"{v * 100:g}%"


def pct_log(v):
    """Readable percentages on a log axis: 9.6%, 11.0%, 2,746%."""
    return f"{v * 100:,.1f}%" if v * 100 < 100 else f"{v * 100:,.0f}%"


def holding_month_end(signal_month):
    """Signal formed at the end of month m, held through month m + 1: plot each return at the end of the holding month (in years)."""
    y, m = map(int, signal_month.split("-"))
    return y + (m + 1) / 12


def save(name, svg):
    with open(os.path.join(OUT, name), "w", encoding="utf-8") as fh:
        fh.write(svg)
    print("  wrote", name)


def fig1_tests():
    t = [1.50, 2.45, -0.36, 0.69, 1.78, 1.84, 1.16, -0.95, 2.49, -1.42, 2.29, 1.72, 1.75, 1.14, 1.27, -0.25, 0.25, 1.27, 4.43,
         1.35, 2.25, 2.90, 0.14, 1.24, 1.80, 1.17, 1.56, 2.84, 1.84, 1.91, 3.78, 1.23, -0.06]
    labels = [str(i) for i in range(1, 20)] + [str(i) for i in range(21, 35)]
    save("fig1_tests.svg", bar_chart("33 pre-registered tests run: EWC normal-equivalent t", labels, {"EWC eq. t": t},
                                     ylabel="t", hlines=[(1.96, "pass (single test) 1.96", "#c53030"), (2.84, "Bonferroni bar for 22-32: 2.84", "#dd6b20"),
                                                         (3.0, "", "#805ad5")],
                                     yfmt=lambda v: f"{v:.2f}", note="Purple: promotion bar 3.0. 2, 9: same signal. 5-7, 11-12: bar 2.39. 19, 21-32: derivatives, calendar. 33-34: Part 8. 20: not run."))


def fig2_minvar():
    np.seterr(divide="ignore", over="ignore", invalid="ignore")
    r = rm.evaluate(rm.synthetic_market())
    keys = ["true covariance (benchmark)", "factor model", "Ledoit-Wolf", "sample covariance"]
    save("fig2_minvar.svg", bar_chart("Minimum-variance portfolio: realised volatility out of sample", ["true cov.", "factor model", "Ledoit-Wolf", "sample cov."],
                                      {"realised vol": [r[k]["min_var_realised_vol"] for k in keys]}, ylabel="annualised volatility (log scale)", ylog=True,
                                      yfmt=pct_log, note="Synthetic market, 500 stocks, 250-day window: the sample covariance is singular and the optimiser levers into noise."))


def fig3_construction():
    res = [pd_.run(seed=s, show=False) for s in (11, 12, 13, 14, 15)]
    names = ["A equal weight", "B optimised, factor neutral", "C optimised + costs"]
    g = {"IR before costs": [float(np.mean([r[n]["IR_gross"] for r in res])) for n in names],
         "IR after costs": [float(np.mean([r[n]["IR_net"] for r in res])) for n in names]}
    save("fig3_construction.svg", bar_chart("Same signal, three constructions (synthetic, mean of 5 seeds)", ["A: equal weight", "B: factor neutral", "C: + cost-aware"], g,
                                            ylabel="information ratio", yfmt=lambda v: f"{v:.2f}", note="500 stocks, 59 monthly periods, 0.2 % per unit traded."))


def fig4_leakage():
    r = ml_demo.exp2(show=False)
    seeds = list(r)
    g = {k: [r[s][k]["ic"] for s in seeds] for k in ("shuffled K-fold", "purged K-fold", "walk-forward")}
    save("fig4_cv_leakage.svg", bar_chart("Pure noise, overlapping labels: out-of-sample IC by validation scheme", [f"seed {s}" for s in seeds], g,
                                          ylabel="mean monthly IC", yfmt=lambda v: f"{v:.3f}", note="There is no signal at all; shuffled K-fold leaks through persistent features and overlapping labels."))


def fig5_frontier():
    X, S0, sigma, eta, gamma, eps, T, N = 1e6, 50.0, 0.95, 2.5e-6, 2.5e-7, 0.0625, 5.0, 20
    lams = [0.0] + list(np.logspace(-8, -5, 13))
    pts = [ex.ac_cost(ex.ac_trajectory(X, N, T, l, sigma, eta, gamma), T / N, sigma, eta, gamma, eps) for l in lams]
    f = Figure("Almgren-Chriss efficient frontier (paper's example)", "standard deviation of cost ($m)", "expected cost (bp of $50m)", legend="top right")
    f.line([np.sqrt(v) / 1e6 for _, v in pts], [e / (X * S0) * 1e4 for e, _ in pts], label="optimal schedules (\u03bb from 0 to 1e-5)")
    f.points([np.sqrt(pts[0][1]) / 1e6], [pts[0][0] / (X * S0) * 1e4], label="TWAP (\u03bb = 0)", color="#c53030")
    f.xfmt = f.yfmt = lambda v: f"{v:g}"
    save("fig5_ac_frontier.svg", f.svg())


def fig6_capacity():
    cap = ex.capacity(aum_per_side=tuple(np.logspace(5, 9, 17)))
    f = Figure("Capacity: monthly net return vs capital per side (assumed liquidity)", "capital per side ($, log scale)", "monthly net return", xlog=True, yfmt=pct, legend="bottom left")
    f.line(list(cap), [v["monthly_net"] for v in cap.values()], label="net of square-root-law costs and 1 %/yr borrow")
    f.hline(ex.GROSS_MONTHLY - ex.BORROW / 12, "before trading costs", "#2f855a").hline(0.0, "zero", "#c53030")
    save("fig6_capacity.svg", f.svg())


def fig7_delta_hedge():
    rv = [0.15, 0.20, 0.25, 0.30]
    theory = [op.bs_price(100, 100, 0.25, 0.0, 0.25) - op.bs_price(100, 100, 0.25, 0.0, s) for s in rv]
    d = [op.delta_hedge(100, 100, 0.25, 0.0, 0.25, s, 63, n_paths=1500, seed=3) for s in rv]
    w = [op.delta_hedge(100, 100, 0.25, 0.0, 0.25, s, 13, n_paths=1500, seed=3) for s in rv]
    f = Figure("Short ATM call at 25 % implied vol, delta-hedged: P&L vs realised vol", "realised volatility", "P&L per option (mean, +/- 1 std)", xfmt=pct,
               yfmt=lambda v: f"{v:g}")
    f.line(rv, theory, label="theory: C(25 %) - C(realised)", color="#718096", dashed=True, markers=False)
    f.points([x - 0.004 for x in rv], [x.mean() for x in d], yerr=[x.std() for x in d], label="daily hedging")
    f.points([x + 0.004 for x in rv], [x.mean() for x in w], yerr=[x.std() for x in w], label="weekly hedging")
    save("fig7_delta_hedge.svg", f.svg())


def fig8_fpr():
    r = ft.false_positive_rates()
    rhos = list(r)
    g = {k: [r[p][k] for p in rhos] for k in ("EWC", "NW4", "NW12")}
    g = {"EWC": g["EWC"], "Newey-West lag 4": g["NW4"], "Newey-West lag 12": g["NW12"]}
    save("fig8_fpr.svg", bar_chart("False-positive rate under a zero signal (one-sided t > 1.96)", [f"IC autocorrelation {p}" for p in rhos], g,
                                   ylabel="rejection rate", hlines=[(0.025, "nominal 2.5 %", "#c53030")], yfmt=lambda v: f"{v * 100:.1f}%",
                                   note="20,000 simulated IC series of 185 months per cell."))


def fig9_real():
    rows = list(csv.DictReader(open(os.path.join(HERE, "data", "real_data_monthly.csv"), encoding="utf-8")))
    xs = [holding_month_end(r["signal_month"]) for r in rows]
    yr = lambda v: f"{v:.0f}"
    f = Figure("Real data, 2012-08 to 2017-12: cumulative net return of three constructions", "year", "growth of $1", yfmt=lambda v: f"{v:g}", xfmt=yr)
    for k, label in (("A", "A: equal-weight long-short (the lab's rule)"), ("B", "B: risk model, factor neutral"), ("C", "C: risk model + cost-aware")):
        f.line(xs, np.cumprod([1 + float(r[f"{k}_net"]) for r in rows]), label=label, markers=False)
    f.note("Free data (survivorship-biased universe); a period already used in the original test, so this is practice, not evidence.")
    save("fig9_real_cumulative.svg", f.svg())
    g = Figure("Real data: ex-ante market beta of the portfolio each month", "year", "ex-ante beta", yfmt=lambda v: f"{v:g}", xfmt=yr)
    g.line(xs, [float(r["A_exante_beta"]) for r in rows], label="A: equal weight", markers=False)
    g.line(xs, [float(r["C_exante_beta"]) for r in rows], label="C: risk model + cost-aware", markers=False)
    g.hline(0.0, None)
    save("fig9b_real_beta.svg", g.svg())


def _rows(name):
    return list(csv.DictReader(open(os.path.join(HERE, "data", name), encoding="utf-8")))


def _sharpe(x):
    x = np.asarray(x, float)
    return float(x.mean() / x.std(ddof=1) * np.sqrt(12))


def fig10_real_ml():
    rows = _rows("real_ml_monthly.csv")
    models = [("composite", "composite (no fitting)"), ("ridge13", "ridge, 13 feat."), ("gbdt13", "GBDT, 13 feat."), ("ridge8", "ridge, 8 feat.*"), ("gbdt8", "GBDT, 8 feat.*")]
    g = {"Sharpe before costs": [_sharpe([float(r[f"{m}_gross"]) for r in rows]) for m, _ in models],
         "Sharpe after costs": [_sharpe([float(r[f"{m}_net"]) for r in rows]) for m, _ in models]}
    save("fig10_real_ml.svg", bar_chart("Real stocks, 2014-08 to 2017-11: walk-forward ML vs the equal-weight composite", [lab for _, lab in models], g,
                                        ylabel="long-short Sharpe ratio (top 50 / bottom 50)", yfmt=lambda v: f"{v:.2f}",
                                        note="40 test months, ~650 stocks/month, retrained monthly on data known at the time. *Added after seeing the first three."))


def fig11_real_capacity():
    rows = _rows("real_capacity.csv")
    cap = ex.capacity(aum_per_side=tuple(float(r["capital_per_side"]) for r in rows))
    f = Figure("Trading cost vs capital per side: assumed liquidity vs the stocks P1 actually traded", "capital per side ($, log scale)",
               "monthly trading cost (share of capital per side)", xlog=True, yfmt=pct)
    f.line([float(r["capital_per_side"]) for r in rows], [v["monthly_cost"] for v in cap.values()], label="assumed liquidity (Part 4 demo, 24.5 % turnover)",
           color="#718096", dashed=True, markers=False)
    f.line([float(r["capital_per_side"]) for r in rows], [float(r["monthly_trading_cost"]) for r in rows], label="P1's actual trades 2012-2017 (~20 % turnover)")
    f.hline(float(rows[0]["gross_monthly_2012_2017"]) / 2, "half of P1's gross return, 2012-17", "#c53030", label_side="left")
    f.note("Impact from each traded stock's own 60-day dollar volume and 63-day volatility (square-root law, Y = 0.7); spreads assumed (4 / 8 bp).")
    save("fig11_real_capacity.svg", f.svg())


def fig12_real_short_vol():
    rows = _rows("real_short_vol_monthly.csv")
    xs = [int(r["month"][:4]) + int(r["month"][5:]) / 12 for r in rows]          # each position expires at the end of its month
    f = Figure("Selling one-month S&P 500 variance at the VIX, 2010-2017: cumulative P&L", "year", "cumulative P&L (vol points per $1 vega)",
               yfmt=lambda v: f"{v:g}", xfmt=lambda v: f"{v:.0f}", legend="top left")
    f.line(xs, np.cumsum([float(r["var_swap_pnl_vol_points"]) for r in rows]), label="short variance swap, monthly", markers=False)
    f.hline(0.0, None)
    f.note("86 % of months positive, Sharpe 2.0, skewness -3.3; worst: Aug 2011 (-28), Aug 2015 (-23). 2008, 2018 and 2020 are outside this window.")
    save("fig12_real_short_vol.svg", f.svg())


def fig13_real_ml_variants():
    rows = _rows("real_ml_variants.csv")
    save("fig13_real_ml_variants.svg", bar_chart("Ensemble plus eight standard fixes, real stocks 2014-10 to 2017-11: mean monthly IC",
                                                 [r["variant"] for r in rows], {"mean IC": [float(r["ic_mean"]) for r in rows]},
                                                 ylabel="mean rank IC (out of sample)", yfmt=lambda v: f"{v:.3f}",
                                                 note="V0 = RF + GBDT + NN averaged; V1-V7 change one thing each; V8 = all. 38 test months; see docs/03 for the list."))


def fig14_vrp():
    rows = _rows("derivatives_vrp_monthly.csv")
    f = Figure("Test #19: selling 1-month S&P 500 variance at the VIX, 1990-2026", "year",
               "cumulative return, % of capital (10 % variance notional)", yfmt=lambda v: f"{v:g}", xfmt=lambda v: f"{v:.0f}", legend="top left")
    cum, seg = 0.0, {"a": ([], []), "b": ([], []), "c": ([], [])}
    for r in rows:
        cum += float(r["net_excess_return"]) * 100
        k = "a" if r["signal_month"] <= "2009-11" else ("b" if r["signal_month"] <= "2017-11" else "c")
        seg[k][0].append(holding_month_end(r["signal_month"]))
        seg[k][1].append(cum)
    f.line(*seg["a"], label="test sample, 1990-2009 (t 3.73)", markers=False)
    f.line(*seg["b"], label="2010-2017: seen in earlier practice work, excluded from the test", dashed=True, markers=False)
    f.line(*seg["c"], label="test sample, 2018-2026 (t 1.94)", markers=False)
    f.note("Cumulative sum, net of a 1 % strike cost. Worst months: Sep 2008 -61 %, Mar 2020 -44 %, Apr 2025 -37 %.")
    save("fig14_vrp_cumulative.svg", f.svg())


def fig15_hedges():
    rows = {r["variant"]: r for r in _rows("derivatives_hedges.csv")}
    keys = [("P1", "P1 unhedged"), ("P2a", "P2a SPY hedge"), ("P7a_50bp", "futures hedge"), ("P7d", "linear, P7b size"), ("P7b", "buy calls"), ("P7c", "buy puts")]
    save("fig15_hedges.svg", bar_chart("Hedging the long-short book, 2012-08 to 2026-07: Sharpe ratio after costs", [lab for _, lab in keys],
                                       {"Sharpe": [float(rows[k]["sharpe"]) for k, _ in keys]}, ylabel="annualised Sharpe (net)", yfmt=lambda v: f"{v:.2f}",
                                       note="Futures hedge = P2a with funding at T-bill + 0.5 %. Calls/puts priced from the VIX (Black-Scholes) + 2 %. Same history; reported, not evidence."))


def fig16_queue_imbalance():
    rows = _rows("lob_signals.csv")
    f = Figure("Queue imbalance vs the next mid-price move, NASDAQ 2012-06-21 (morning)", "queue imbalance (bid - ask) / (bid + ask), bin centre",
               "share of next mid moves that are up", yfmt=pct, xfmt=lambda v: f"{v:g}", legend="top left")
    xs = [-0.9 + 0.2 * k for k in range(10)]
    for r in rows:
        tick = "1-tick spread" if float(r["share_one_tick_spread"]) > 0.5 else "multi-tick spread"
        f.line(xs, [float(r[f"p_next_up_qi_bin{k}"]) for k in range(1, 11)], label=f"{r['stock']} ({tick}, mean {float(r['mean_spread_cents']):.1f} c)",
               dashed=float(r["share_one_tick_spread"]) < 0.5)
    f.hline(0.5, None)
    f.note("Ten equal-width bins of the top-of-book imbalance, 9:35-12:45. LOBSTER sample data; one day, descriptive only.")
    save("fig16_queue_imbalance.svg", f.svg())


def fig17_meta_labeling():
    rows = _rows("meta_labeling_policies.csv")
    pol = [("A all trades (primary only)", "A all trades"), ("B meta filter p>0.5", "B meta filter"), ("C bet sizing 2Phi(z)-1", "C bet sizing"),
           ("D primary |prediction| at B's share", "D big predictions")]
    seg = [("meta_train_2022_2023", "meta-train 2022-23"), ("meta_valid_2024", "meta-valid 2024"), ("test_2025_01_2026_09", "test 2025-01 to 2026-09")]
    get = {(r["segment"], r["policy"]): float(r["sharpe_5bp"]) for r in rows}
    save("fig17_meta_labeling.svg", bar_chart("Meta-labeling on BTC and ETH hourly trades: Sharpe after 5 bp per side", [lab for _, lab in pol],
                                              {sl: [get[(s, p)] for p, _ in pol] for s, sl in seg}, ylabel="annualised Sharpe (net)", yfmt=lambda v: f"{v:.1f}",
                                              note="B, C and D trade 3.7-5.3 % of the hours; their gain over A is saved costs. Test, B vs D: NW t of the difference -0.54."))


if __name__ == "__main__":
    np.seterr(divide="ignore", over="ignore", invalid="ignore")
    os.makedirs(OUT, exist_ok=True)
    every = (fig1_tests, fig2_minvar, fig3_construction, fig4_leakage, fig5_frontier, fig6_capacity, fig7_delta_hedge, fig8_fpr, fig9_real,
             fig10_real_ml, fig11_real_capacity, fig12_real_short_vol, fig13_real_ml_variants, fig14_vrp, fig15_hedges, fig16_queue_imbalance,
             fig17_meta_labeling)
    wanted = sys.argv[1:]                                             # e.g. "fig2 fig9" to redraw only those
    for f in every:
        if not wanted or any(f.__name__.startswith(w) for w in wanted):
            f()
