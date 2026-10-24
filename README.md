# Quant Research Lab — a pre-registered factor research pipeline, and what it taught me

A personal research lab for US equities that tries to answer one question honestly:
**which return signals survive costs when you test them the way a careful buy-side researcher would?**

The short answer after 31 pre-registered tests: **one equity signal survived, weakly** (a composite of eight classic mispricing anomalies, EWC-equivalent t ≈ 2.5),
and the long–short portfolio built on it has a backtest Sharpe ratio of about **0.55** with a **−40 %** maximum drawdown — useful as evidence, mediocre as a product.
Outside equities, the **variance risk premium** passed strongly (t 4.4) — but it is insurance selling with crash-sized losses, and the instrument tested is not one an individual can trade.
A later round of eleven derivatives and calendar strategies on free data passed three, two of them only at the margin; the one with real statistical power (selling Treasury variance) failed.
The rest of this repository explains how that conclusion was reached, why most ideas failed, and the standard quant toolkit
(risk models, optimization, ML validation, execution, options) rebuilt from scratch in NumPy.

> Status: research and paper trading only. Nothing here is investment advice.

---

## What is in here

| Part | What it shows | Where |
|---|---|---|
| **At a glance** | One page: what the lab does, the signal in plain words, every portfolio's backtest next to SPY, how reliable and fragile it is, what real money would lose, and where it stands against the market and the literature | [`docs/00_at_a_glance.md`](docs/00_at_a_glance.md) |
| **1. Research process** | Pre-registration, multiple-testing control, power analysis, HAC inference (EWC / Newey–West), survivorship-bias correction, an append-only test registry, and a shadow (paper) portfolio with a rehearsed monthly runbook | [`docs/01_research_process.md`](docs/01_research_process.md), `quantlab/factor_test.py` (the test itself, validated by simulation) |
| **2. Risk model & portfolio optimization** | Barra-style factor risk model, Ledoit–Wolf shrinkage, an ADMM optimizer with neutrality constraints, position limits and transaction costs — on synthetic data **and applied to the surviving signal on real 2012–2017 data** | [`docs/02_risk_model_and_optimization.md`](docs/02_risk_model_and_optimization.md), `quantlab/risk_model.py`, `quantlab/optimizer.py` |
| **3. Machine learning & overfitting** | Histogram gradient-boosted trees, purged / embargoed cross-validation, Newey–West t-stats, the deflated Sharpe ratio — and a walk-forward test **on real stocks** against the simple composite | [`docs/03_ml_and_overfitting.md`](docs/03_ml_and_overfitting.md), `quantlab/gbdt.py`, `quantlab/validation.py` |
| **4. Execution & transaction costs** | Square-root impact, Almgren–Chriss optimal execution, implementation-shortfall TCA, strategy capacity — recomputed **with the stocks the portfolio actually traded** | [`docs/04_execution_and_costs.md`](docs/04_execution_and_costs.md), `quantlab/execution.py` |
| **5. Options, probability & market making** | Black–Scholes and Greeks, binomial trees, implied volatility, delta-hedging P&L, Kelly sizing, a Glosten–Milgrom market-making simulation, classic probability puzzles — plus **the variance risk premium on real 2010–2017 data** | [`docs/05_options_and_market_making.md`](docs/05_options_and_market_making.md), `quantlab/options.py`, `quantlab/probability.py` |
| **6. The strategy in detail** | The universe rules, the eight-anomaly mispricing signal (formulas, point-in-time rules), the base long–short portfolio and twelve variants (including a factor-neutral version) with their rules and backtests, what drives the returns (factor regression, risk decomposition), and the operating and risk controls | [`docs/06_strategy.md`](docs/06_strategy.md), `quantlab/strategy.py` (the rules as tested functions) |
| **7. Derivatives** | A pre-registered test of the variance risk premium on 1990–2026 data (with the years already seen excluded), the tradable test that was designed but not run and why, hedging the long–short book with futures vs. calls vs. puts, and a twelve-test round of derivatives and calendar strategies on free data | [`docs/07_derivatives.md`](docs/07_derivatives.md), `quantlab/variance.py` |

Every module in `quantlab/` has a self-test that checks it against an independent answer
(closed form vs. numerical solution, brute-force enumeration, finite differences, or Monte Carlo). Run them all, and regenerate every figure, with:

```bash
pip install numpy
python run_selftests.py
python make_figures.py
```

Parts 2–5 first check each method on **synthetic data with a known ground truth**, then apply it to **real data from 2010–2017**
(the years after 2017 are reserved for confirming pre-registered tests). Only aggregate results — monthly portfolio returns, ICs, cost curves, strategy P&L — are included, in `data/`;
no market data is redistributed.

---

## Headline results (Part 1)

32 hypotheses were registered before any data was examined; each had a fixed threshold, test statistic and budget. 31 were run; one (#20) was not, because its data cost more than it was worth to the project.

![EWC t-statistics of the 31 tests run](figures/fig1_tests.svg)

| # | Signal (US large/mid caps unless noted) | Result | Main statistic |
|---|---|---|---|
| 1 | Year-over-year change in 10-K text | fail | NW t 1.85 (threshold 1.96) |
| 2 | **Mispricing composite (8 anomalies)** | **pass** | IC 0.021, EWC t **2.45** |
| 3 | Value composite (B/M, E/P, C/P) | fail | EWC t −0.36 |
| 4 | Residual momentum | fail | EWC t 0.69 |
| 5–7 | Mined in 2010–17, confirmed on 2018–26: 10-K filing lag, leverage, change in leverage | fail | EWC t 1.78 / 1.84 / 1.16 (threshold 2.39) |
| 8 | Opportunistic insider trading | fail | EWC t −0.95 |
| 9 | **#2 re-run on survivorship-bias-free data, rules unchanged** | **pass** | IC 0.023, EWC t **2.49** |
| 10 | Calendar composite (earnings-announcement premium + seasonality) | fail | EWC t −1.42 |
| 11–12 | Second mining round: 8-K filing frequency, stock-price synchronicity | fail | EWC t 2.29 / 1.72 (threshold 2.39) |
| 13 | Literature composite (geographic dispersion, industry concentration, employee growth) | fail | EWC t 1.75 |
| 14 | #2 applied to small caps | fail | EWC t 1.14; negative after small-cap costs |
| 15 | Cross-asset ETF trend following | fail | EWC t 1.27 |
| 16 | Cross-asset carry + value + low-beta composite | fail | EWC t −0.25 |
| 17 | Macro composite (economic trend, carry timing, seasonality) | fail | EWC t 0.25 |
| 18 | Merger arbitrage (cash deals, identified from SEC filings) | fail | EWC t 1.27; market beta 0.60 explained the return |
| 19 | **Variance risk premium: sell one-month S&P 500 variance at the VIX** (1990–2026, years already seen excluded; a replication) | **pass, promoted** | EWC t **4.43**; 2018–26 alone 1.94; skew-adjusted t 2.47 |
| 20 | Sell index puts with daily delta hedging at real bid/ask prices | not run | needed a paid options dataset; decided not to buy |
| 21 | Sell 10-year Treasury variance at the Treasury-note volatility index (2003–2020; its own round, bar 1.96) | fail | EWC t 1.35; 2018–20 alone −0.57 |
| 22 | Sell crude-oil, gold, emerging-market and Brazil variance (composite) | fail | EWC t 2.25 (bar 2.84 for #22–32) |
| 23 | Sell Dow, Nasdaq-100 and Russell 2000 variance (composite) | **pass** (marginal) | EWC t **2.90**; correlation 0.97 with #19 — the same bet; skew-adjusted t 2.72 |
| 24 | Time the S&P 500 with the variance premium | fail | EWC t 0.14 |
| 25 | #19, skipping months with an inverted VIX curve (the extra return) | fail | EWC t 1.24 |
| 26 | #19 scaled down when volatility is high (alpha over #19) | fail | EWC t 1.80 |
| 27 | Hold SVXY (−0.5× short VIX futures), the version an individual can trade | fail | EWC t 1.17 |
| 28 | Commodity speculator positioning (CFTC reports), traded with commodity ETFs | fail | EWC t 1.56 |
| 29 | Treasury month-end: hold 10-year Treasuries over the last 3 trading days | **pass** (marginal) | EWC t **2.84** vs bar 2.838; fails at double costs (2.38) |
| 30 | Hold the S&P 500 on scheduled FOMC announcement days | fail | EWC t 1.84 |
| 31 | Turn of the month (last day to third trading day) | fail | EWC t 1.91 |
| 32 | **Equal-risk composite of the nine standalone strategies in #21–31** | **pass, promoted** | EWC t **3.78**; 2018–26 alone 1.37; correlation 0.69 with #19 |

Tests #21–32 are replications of published ideas on free public data; #21 had its own round (bar 1.96) and #22–32 shared a Bonferroni budget of eleven (bar 2.84). Details in [`docs/07_derivatives.md`](docs/07_derivatives.md).

**The portfolio built on the surviving signal** (50 long / 50 short, equal weight, monthly, 2012-08 → 2026-07, net of 0.2 % one-way costs and 1 %/yr borrow;
the exact rules, the twelve variants and the risk controls are in [`docs/06_strategy.md`](docs/06_strategy.md)):

| | Sharpe | CAGR | Volatility | Max drawdown |
|---|---|---|---|---|
| Long–short portfolio | 0.55 | 8.1 % | 16.5 % | −39.8 % |
| S&P 500 (SPY, same period) | 0.90 | 14.9 % | 14.9 % | −23.8 % |

- With 14 years of data the 95 % interval for that Sharpe ratio is roughly **0 to 1.1** — the backtest cannot distinguish "0.55" from "close to zero".
- The equal-weight construction leaves large unintended exposures: in a rehearsal of the live process the portfolio's ex-ante market beta was about **−1.4**, outside anything seen in the backtest.
- **Applying a risk model to the same signal on real 2012–2017 data** (Part 2): 57 % of the equal-weight portfolio's predicted risk was factor risk. Neutralising it cut volatility
  from 11.5 % to 3.7 % and the maximum drawdown from −17 % to −6 %, and raised the Sharpe ratio before costs from 0.60 to 0.84 — but after costs only from 0.34 to 0.39,
  and a cost-blind optimizer gave everything back (0.03). On real data the gain is mostly risk reduction, and cost-awareness is not optional.

![Real data: cumulative net return of three constructions](figures/fig9_real_cumulative.svg)

## What I took away

1. **Most ideas fail when tested honestly.** Of 31 tests run, 25 failed. One equity signal survived (test #2 and its survivorship-free re-run #9); outside equities the variance risk premium passed strongly (#19), and three more passed in the derivatives-and-calendar round (#23, #29, #32), two of them only at the margin — all replications of published ideas, not new alpha. The machinery that produced those numbers — pre-registration, a registry of every test,
   an exploration / confirmation split, HAC standard errors, power analysis — is the real deliverable.
2. **Construction matters as much as the signal.** On synthetic data, the same weak signal goes from an information ratio of 0.7–1.5 (equal-weight long–short)
   to 1.6–2.7 (factor-neutral, risk-model-optimized), and cost-aware optimization cuts turnover by about four-fifths. On real data the same machinery mainly removed risk
   (volatility down by two-thirds) — a more modest, more realistic result.
3. **Validation is where ML goes wrong.** Shuffled K-fold CV on pure noise with overlapping labels produced an IC of 0.21 (Newey–West t ≈ 27); purged, time-ordered validation gave ≈ 0.
   On real stocks, properly validated ridge, boosted trees, a random forest, a neural network and their average all failed to beat simply averaging the eight anomalies
   (IC between −0.014 and +0.006 vs +0.025), and eight standard fixes (industry-neutral labels, longer labels, smoothing, tuning, ...) did not change that.
4. **Costs set capacity.** With the liquidity of the stocks the strategy actually traded, costs are small at personal scale but consume about half its gross return at roughly $100 m per side.
5. **Selling volatility pays — until it doesn't.** On 2010–2017 data, selling one-month S&P 500 variance at the VIX earned money in 86 % of months (Sharpe 2.0),
   with skewness −3.3 and single months losing as much as nine average months. The pre-registered test on the other years (#19) passed with t 4.4 — and lost 61 % in September 2008 at a modest size.
6. **Derivatives are mostly tools.** For the long–short book, an index-futures beta hedge kept most of its benefit after funding costs (Sharpe 0.73–0.75 vs 0.55 unhedged);
   buying calls did worse than the linear hedge, and buying puts hedged a risk the book does not have (Part 7).

## License

MIT — see [LICENSE](LICENSE).
