# Quant Research Lab — a pre-registered factor research pipeline, and what it taught me

A personal research lab for US equities that tries to answer one question honestly:
**which return signals survive costs when you test them the way a careful buy-side researcher would?**

The short answer after 18 pre-registered tests: **one signal survived, weakly** (a composite of eight classic mispricing anomalies, EWC-equivalent t ≈ 2.5),
and the long–short portfolio built on it has a backtest Sharpe ratio of about **0.55** with a **−40 %** maximum drawdown — useful as evidence, mediocre as a product.
The rest of this repository explains how that conclusion was reached, why most ideas failed, and the standard quant toolkit
(risk models, optimization, ML validation, execution, options) rebuilt from scratch in NumPy.

> Status: research and paper trading only. Nothing here is investment advice.

---

## What is in here

| Part | What it shows | Where |
|---|---|---|
| **1. Research process** | Pre-registration, multiple-testing control, power analysis, HAC inference (EWC / Newey–West), survivorship-bias correction, an append-only test registry, and a shadow (paper) portfolio with a rehearsed monthly runbook | [`docs/01_research_process.md`](docs/01_research_process.md) |
| **2. Risk model & portfolio optimization** | Barra-style factor risk model, Ledoit–Wolf shrinkage, an ADMM optimizer with neutrality constraints, position limits and transaction costs | [`docs/02_risk_model_and_optimization.md`](docs/02_risk_model_and_optimization.md), `quantlab/risk_model.py`, `quantlab/optimizer.py` |
| **3. Machine learning & overfitting** | Histogram gradient-boosted trees, purged / embargoed cross-validation, Newey–West t-stats, the deflated Sharpe ratio | [`docs/03_ml_and_overfitting.md`](docs/03_ml_and_overfitting.md), `quantlab/gbdt.py`, `quantlab/validation.py` |
| **4. Execution & transaction costs** | Square-root impact, Almgren–Chriss optimal execution, implementation-shortfall TCA, strategy capacity | [`docs/04_execution_and_costs.md`](docs/04_execution_and_costs.md), `quantlab/execution.py` |
| **5. Options, probability & market making** | Black–Scholes and Greeks, binomial trees, implied volatility, delta-hedging P&L, Kelly sizing, a Glosten–Milgrom market-making simulation, classic probability puzzles | [`docs/05_options_and_market_making.md`](docs/05_options_and_market_making.md), `quantlab/options.py`, `quantlab/probability.py` |

Every module in `quantlab/` has a self-test that checks it against an independent answer
(closed form vs. numerical solution, brute-force enumeration, finite differences, or Monte Carlo). Run them all with:

```bash
pip install numpy
python run_selftests.py
```

Parts 2–5 use **synthetic data with a known ground truth**, so each method can be checked against the right answer. No market data is redistributed.

---

## Headline results (Part 1)

18 hypotheses were registered before any data was examined; each had a fixed threshold, test statistic and budget.

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

**The portfolio built on the surviving signal** (50 long / 50 short, equal weight, monthly, 2012-08 → 2026-07, net of 0.2 % one-way costs and 1 %/yr borrow):

| | Sharpe | CAGR | Volatility | Max drawdown |
|---|---|---|---|---|
| Long–short portfolio | 0.55 | 8.1 % | 16.5 % | −39.8 % |
| S&P 500 (SPY, same period) | 0.90 | 14.9 % | 14.9 % | −23.8 % |

- With 14 years of data the 95 % interval for that Sharpe ratio is roughly **0 to 1.1** — the backtest cannot distinguish "0.55" from "close to zero".
- The equal-weight construction leaves large unintended exposures: in a rehearsal of the live process the portfolio's ex-ante market beta was about **−1.4**, outside anything seen in the backtest.
  Part 2 shows, on synthetic data, how a risk model and a cost-aware optimizer address exactly this.

## What I took away

1. **Most ideas fail when tested honestly.** One signal survived (tests #2 and its survivorship-free re-run #9); everything else failed. The machinery that produced that number — pre-registration, a registry of every test,
   an exploration / confirmation split, HAC standard errors, power analysis — is the real deliverable.
2. **Construction matters as much as the signal.** On synthetic data, the same weak signal goes from an information ratio of 0.7–1.5 (equal-weight long–short)
   to 1.6–2.7 (factor-neutral, risk-model-optimized), and cost-aware optimization cuts turnover by about four-fifths.
3. **Validation is where ML goes wrong.** Shuffled K-fold CV on pure noise with overlapping labels produced an IC of 0.21 (Newey–West t ≈ 27); purged, time-ordered validation gave ≈ 0.
4. **Costs set capacity.** Under assumed liquidity, the surviving strategy's costs are small at personal scale but consume about half its gross return at roughly $100 m per side.

## License

MIT — see [LICENSE](LICENSE).
