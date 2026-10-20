# Part 6 — The strategy in detail: universe, signal, portfolios, risk controls

This part states exactly what the lab trades on paper. The rules below are the pre-registered ones from the private pipeline; `quantlab/strategy.py`
restates the signal and the portfolio rules as pure functions with self-tests, so they can be read and run without any data. All results are portfolio-level aggregates.

## 1. Universe (rebuilt at every month-end, point in time)

A stock is in the universe at month-end *d* if all of the following hold on *d*:

1. It filed a 10-K within the previous 400 days (later-delisted companies included; this also removes foreign filers, who file 20-F / 40-F).
2. Its common stock is listed on NYSE (or NYSE American / Arca) or Nasdaq.
3. Its actual traded price is at least $5, and it traded within the last 5 trading days.
4. Its median daily dollar volume over the last 60 trading days is at least $10 m.
5. Its market capitalisation (point-in-time shares outstanding × price) is at least $1 bn.
6. It is not a non-operating entity (SPACs, commodity / crypto trusts, asset-backed issuers, funds — by SIC code), and not on a small exclusion list kept for a separate project.

That gives about 1,000–1,500 stocks a month. For the signal below, financials and real estate (SIC 60–69) are also removed, because their balance sheets do not fit the accounting definitions.
From test #9 on, the universe and prices come from a survivorship-bias-free database (delisted firms and delisting returns included).

## 2. The signal: a mispricing composite

Stambaugh, Yu & Yuan (2012) average eleven anomalies into one "mispricing" score; eight of them can be built point-in-time from free SEC data, and those eight are the signal.

**Point-in-time rule.** The signal for month-end *d* uses only numbers that had been filed with the SEC (10-K, 10-Q and their amendments, as first filed or as revised *before* the cut-off)
by the trading day before *d*'s last trading day. The fiscal year *E0* is the period of the latest annual report filed by then, *E1* the year before (350–380 days earlier);
annual data more than 550 days old, or quarterly data more than 200 days old, counts as missing.

| # | Component | Definition | Higher is… |
|---|---|---|---|
| 1 | Net share issuance | ln(shares outstanding now ÷ (shares a year ago × split factor)); the two counts 270–460 days apart | worse |
| 2 | Accruals | (net income − operating cash flow) ÷ average total assets of *E0* and *E1* | worse |
| 3 | Net operating assets | (debt + common equity + minority interest + preferred − cash − short-term investments)<sub>E0</sub> ÷ total assets<sub>E1</sub> | worse |
| 4 | Momentum | price return from the month-end seven months ago to last month's end (the latest month is skipped) | better |
| 5 | Gross profitability | (revenue − cost of revenue)<sub>E0</sub> ÷ total assets<sub>E0</sub> (reported gross profit when available) | better |
| 6 | Asset growth | total assets<sub>E0</sub> ÷ total assets<sub>E1</sub> − 1 | worse |
| 7 | Return on assets | latest quarter's net income ÷ total assets one quarter earlier | better |
| 8 | Investment-to-assets | (Δ gross PP&E + Δ inventory) ÷ total assets<sub>E1</sub> | worse |

- XBRL tags are tried in a fixed order (e.g. revenue: `Revenues` → `RevenueFromContractWithCustomerExcludingAssessedTax` → `SalesRevenueNet`); the first tag with a value in every period needed is used.
  Missing debt, minority interest, preferred stock or short-term investments count as zero; missing core items make that component missing.
- **Composite.** Each component is ranked among the stocks that have it (percentile = (average rank − ½) ÷ n), flipped when "higher is worse",
  and the composite is the equal-weighted mean of the available percentiles. **Fewer than five components → no score.** No weights are estimated from data.
- **Data gates.** Momentum uses the last price on or before each month-end (at most five trading days earlier); any daily |return| above 100 % in the window makes it missing.
- **Known defect, kept on purpose.** Net operating assets counts preferred stock twice (common equity in XBRL already includes it). The tested signal is kept byte-for-byte unchanged,
  so the defect stays and is recorded in the lab's limitations list.

**Evidence.** Test #2 (free data, signal months 2012-07 → 2026-05): mean monthly rank IC 0.021, EWC normal-equivalent t 2.45 — a pass. Test #9, the same rules on
survivorship-bias-free data: IC 0.023, t 2.49, quintile long–short net of costs +0.37 % a month. Both halves of the sample have similar ICs (0.022 and 0.024). Not promoted (t < 3).

## 3. The base portfolio (P1)

- On the last trading day of each month (*L*) the composite is computed; positions are traded at the **close of the next trading day** (*T1*, market-on-close orders) and held to the next month's *T1* close.
- **Top 50 long, bottom 50 short, equal weight** (1/50 of each side per name): dollar-neutral, 200 % gross. Ties break on the company identifier, so the selection is deterministic.
  If fewer than 200 stocks have a score, last month's book is kept.
- **Costs**: each side's turnover τ = new names ÷ 50; monthly cost = 0.4 % × (τ<sub>long</sub> + τ<sub>short</sub>) + 1 %/yr borrow ÷ 12 (re-weighting the kept names back to 1/50 is not charged).
- **Borrow**: on paper all 50 shorts are kept; with real money a name that cannot be borrowed is not shorted and the other shorts are re-weighted equally.

Backtest on survivorship-bias-free data, 167 monthly holding periods (2012-08 → 2026-07):

| | P1 |
|---|---|
| Net return ÷ volatility (annualised) | 0.55 |
| CAGR / volatility | 8.1 % / 16.5 % |
| Maximum drawdown | −39.8 % (2019-09 → 2021-02) |
| Gross return / cost per month | 1.04 % / 0.28 % |
| Turnover per side per month | 24 % long, 25 % short |
| Long leg minus universe / universe minus short leg | +0.19 % / +0.86 % per month |
| Months with a profit | 53 % |

The ratio is annualised net return over volatility with cash and margin earning nothing — for a self-financing long–short book this is its Sharpe ratio.
Most of the return comes from the **short side** (overpriced stocks underperforming), as in Stambaugh, Yu & Yuan's original evidence.

## 4. What drives P1

- **Factor regression** (Fama–French five factors + momentum, monthly, Newey–West t): annual alpha 8.5 % (t 2.46), R² 0.40; significant loadings on
  profitability (RMW +0.73), momentum (+0.46) and value (HML +0.37) — the factors its components resemble. Against the market alone the alpha is 13.5 % with beta −0.32.
- **Ex-ante risk decomposition** (Barra-style model estimated each month on the universe's daily returns, see Part 2): median predicted volatility 11 %;
  **62 % of predicted risk comes from factors** (momentum 18 %, market 14 %, industries 10 %, volatility 8 %), the rest is stock-specific.
  Typical exposures: momentum +1.0 (standardised), volatility −0.6, size +0.4, net beta −0.20. In 2026 the net beta drifted to −1.34 and the factor share to 82 %, both outside the history.
- In plain words: besides "mispricing", P1 carries an unintended bet on recent winners, on calm stocks and — lately — against the market. The variants below try to remove some of it.

## 5. Variants (backtested on the same history; reported, never used to pick)

| Portfolio | Rule | Net return / vol | CAGR | Volatility | Max drawdown | Cost / month | Status |
|---|---|---|---|---|---|---|---|
| **P1** | Base: top 50 long / bottom 50 short, equal weight | 0.55 | 8.1 % | 16.5 % | −39.8 % | 0.28 % | shadow; real-money eligible |
| **P2a** | P1 + SPY hedge of the net beta | 0.78 | 11.2 % | 15.3 % | −35.9 % | 0.29 % | shadow; real-money eligible |
| **P2b** | Industry-neutral selection + SPY hedge | 0.45 | 4.8 % | 11.9 % | −33.0 % | 0.33 % | shadow; view only |
| **P3a** | P1 with a buffer band (keep names while within rank 100) | 0.50 | 7.1 % | 16.3 % | −36.6 % | 0.19 % | shadow; real-money eligible |
| **P3b** | P1 scaled by its own estimated volatility (max 1.5x) | 0.52 | 5.7 % | 12.1 % | −31.5 % | 0.26 % | shadow; real-money eligible |
| **P4a** | Longs only + short SPY | −0.37 | −4.0 % | 9.6 % | −49.2 % | 0.19 % | backtest only |
| **P4b** | P1 rebalanced quarterly | 0.50 | 7.4 % | 17.1 % | −34.8 % | 0.21 % | backtest only |
| **P4c** | P1 weighted by rank (2:1 extreme to least) | 0.57 | 8.7 % | 17.0 % | −40.6 % | 0.29 % | backtest only |
| **P4d** | 100 long / 100 short | 0.56 | 6.7 % | 13.1 % | −30.9 % | 0.26 % | backtest only |
| **P5a** | P1 without the 5 least liquid shorts | 0.59 | 9.0 % | 17.0 % | −40.7 % | 0.29 % | backtest only |
| **P5b** | Shorts only + long SPY | 0.92 | 13.9 % | 15.6 % | −33.6 % | 0.19 % | shadow; view only |
| **P6** | P1 scaled by S&P 500 volatility (max 1.5x) | 0.63 | 9.5 % | 16.3 % | −30.4 % | 0.30 % | backtest only |

Rule details (all monthly, same costs as P1 unless stated):
- **P2a**: each side's beta = OLS slope of its equal-weighted daily return on SPY over the last 252 trading days; hedge h = −(β<sub>long</sub> − β<sub>short</sub>) of one side's notional
  in SPY (bought when positive), SPY trades cost 0.05 %. **P2b**: within each two-digit SIC industry, the top and bottom *k* names (*k* ∝ the industry's share of the universe), then the same hedge.
- **P3a**: last month's names are kept while they stay within rank 100 on their side; the side is then filled to 50 from the top. **P3b**: s = min(1.5, σ\* ÷ σ̂), where σ̂ is the
  annualised volatility of the selected long–short's daily returns over 126 days and σ\* the median of σ̂ to date; both sides are scaled by s (up to 300 % gross).
- **P4a–d** change one design choice at a time; **P5a** drops the five shorts with the lowest dollar volume (a stand-in for hard-to-borrow names); **P5b** keeps only the shorts and buys SPY
  against their beta; **P6** scales P1 by the S&P 500's 252-day volatility relative to its long-run median.

How to read the table:
- **P2a and P5b look best because of their SPY leg, not better stock picking.** P2a buys SPY when the shorts are higher-beta than the longs (on average 0.23 of one side) and P5b holds
  about 1.3× SPY; in a rising market with no funding charge that leg adds return. P4a (longs only, hedged) loses money: the signal's return sits mostly in the short side.
- **P3a** cuts trading costs by a third but gives up more gross return; **P3b** and **P6** trade return for lower volatility; P6's smaller drawdown largely reflects its 252-day window
  happening to straddle March 2020.
- Twelve variants on one history will produce spread like this by chance alone (net-return differences against P1 have t between −2.8 and +1.7). By rule the lab does **not** choose
  the real-money portfolio by backtest returns.

**Break-even costs** (the cost at which the backtest's average monthly net return is zero):

| | P1 | P2a | P3a | P3b | P5a | P5b |
|---|---|---|---|---|---|---|
| Average borrow fee on the short side | 10.1 %/yr | 12.8 %/yr | 9.1 %/yr | 8.6 %/yr | 11.1 %/yr | 15.3 %/yr |
| Round-trip trading cost | 1.92 % | 2.37 % | 2.93 % | 1.49 % | 2.00 % | 5.07 % |

The assumed 0.4 % round trip and 1 %/yr borrow leave a wide margin; Part 4 estimates the real one-way cost at personal scale at about 5 bp.

## 6. Operations and risk controls

- **Shadow portfolios**: P1, P2a, P2b, P3a, P3b and P5b run on paper from the November 2026 rebalance, for at least six rebalances; they stop only for process failures
  (late or changed inputs), never because of their P&L. P2b and P5b are watched only, never traded with real money.
- **Monthly runbook**, one command: refresh prices → rebuild the universe → compute the signal → target holdings for each book → ex-ante risk decomposition → commit and merge.
  It was rehearsed end-to-end on two past months before going live.
- **Real money** is the owner's decision, allowed only after the first paper rebalance runs cleanly; orders are placed by hand as market-on-close orders on *T1*. No automated trading.
- **Stop line**: each book's backtest maximum drawdown (P1 −39.8 %, P2a −35.9 %, P3a −36.6 %, P3b −31.5 %), measured on its shadow ledger from the month real money starts;
  a breach closes the real-money position at the next close.
- **Momentum-crash rule**: if on the signal date SPY is down over the past 24 months and its 126-trading-day volatility is at least 20 % a year (the state after which
  Daniel and Moskowitz (2016) find momentum crashes), the real-money position is halved for that month. A linear stress test of P1 on March–May 2009 factor returns
  gives about −26 % over three months (90 % lower bound −40 %, the stop line), and a crash is probably worse than linear.
- **Monitoring** (a local dashboard): each month's ex-ante beta, volatility, hedge ratio and position size against the backtest's history; the factor risk decomposition of every book;
  and, once real money trades, the execution cost of each fill against the official close (in basis points).

## 7. Limitations

- One signal, t ≈ 2.5: the 95 % interval for P1's ratio is roughly 0 to 1.1.
- The backtests reuse history that was seen while building the variants; they are descriptions, not evidence.
- Costs are assumptions; borrow availability and fees for the short names are checked only at trading time.
- The live shadow runs on free daily prices and a universe that cannot include firms delisted later; the backtests use survivorship-bias-free data.
- The net-operating-assets component double-counts preferred stock (kept to leave the tested signal unchanged).
