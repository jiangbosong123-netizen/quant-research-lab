# At a glance — what the lab does and how it has done

*Numbers as of October 2026. The backtests reuse history that was already seen and the signal is a published one, so they are descriptions (replications), not evidence.
Nothing here shows an individual stock or an amount of money, and nothing here is investment advice. The details are in Parts 1, 6 and 7.*

## What it does

- **Universe**: US large and mid caps, about 1,000–1,500 stocks each month.
- **Goal**: find stock-selection signals that still make money after costs, and show with pre-registered tests that they are not luck.
- **Then**: run the survivors as long–short portfolios rebalanced monthly, first on paper (a *shadow* book). Real money is the owner's decision, allowed only after the first paper rebalance runs cleanly. Orders are placed by hand; nothing trades automatically.

The rules that keep it honest:

1. **Write it down first.** Every idea is pre-registered before any data is pulled: the formula, the test statistic, the threshold, how many tests. The git commit hash is the proof.
2. **One test for every signal.** The monthly rank correlation between the signal and next month's return (the IC) is tested with an autocorrelation-robust EWC t-statistic. A signal passes only if all three hold:
   - the t-statistic clears the pre-registered threshold (1.96, or higher when several tests share a budget);
   - the mean IC is positive;
   - the long–short spread is still positive after costs.
3. **Every result is recorded**, passed or failed, in an append-only registry. A failed idea is never re-tested under a different definition.
4. **Portfolio rules are separate.** How a passing signal becomes a portfolio (number of names, hedges, sizing) is pre-registered on its own, backtested for risk, then run as a shadow book. Portfolio results never feed back into the signal.
5. **Real-money rules are written in advance**:
   - close the position if a book falls further than its worst backtest drawdown;
   - halve the position in a month when the momentum-crash warning is on: SPY is down over the past 24 months and its 126-trading-day volatility is at least 20 % a year (after Daniel and Moskowitz, 2016).

## The signal: a mispricing composite

The signal uses the eight classic anomalies from Stambaugh, Yu and Yuan (2012) that can be computed point in time from free data.
Each month every stock is ranked on each of the eight and the percentiles are averaged. A high score means *underpriced* (buy); a low score means *overpriced* (short).

| Component | In plain words |
|---|---|
| Net share issuance | firms that issued a lot of stock in the past year — avoid |
| Accruals | earnings backed by little cash — avoid |
| Net operating assets | balance sheets bloated with operating assets — avoid |
| Asset growth | firms expanding too fast — avoid |
| Investment to assets | heavy investment relative to assets — avoid |
| Momentum | strong 12-month returns, skipping the last month — buy |
| Gross profitability | high gross profit relative to assets — buy |
| Return on assets | high ROA — buy |

**Test results**:
- Test #2, 2012–2026, on a universe of firms still listed today: IC 0.021, EWC t 2.45.
- Test #9, the same rules re-run with later-delisted firms included: IC 0.023, t 2.49.

Both tests pass the 1.96 bar but not the stricter 3 used to mark strong evidence.

## The portfolios (backtest 2012-08 → 2026-07, 167 monthly holding periods, net of assumed costs)

| Book | What it does | CAGR | Volatility | Return ÷ vol | Max drawdown | Beta to SPY | Status |
|---|---|---|---|---|---|---|---|
| **P1** | top 50 long, bottom 50 short, equal weight | 8.1 % | 16.5 % | 0.55 | −39.8 % | −0.30 | shadow; real-money eligible |
| **P2a** | P1 + SPY to hedge the net beta | 11.2 % | 15.3 % | 0.78 | −35.9 % | −0.05 | shadow; real-money eligible |
| **P3a** | P1 with a buffer band (keep names while within rank 100) | 7.1 % | 16.3 % | 0.50 | −36.6 % | −0.33 | shadow; real-money eligible |
| **P3b** | P1 scaled by its own estimated volatility (max 1.5x) | 5.7 % | 12.1 % | 0.52 | −31.5 % | −0.16 | shadow; real-money eligible |
| P2b | industry-neutral selection + SPY hedge | 4.8 % | 11.9 % | 0.45 | −33.0 % | −0.12 | shadow; view only |
| P5b | P1's 50 shorts + about 1.3x long SPY | 13.9 % | 15.6 % | 0.92 | −33.6 % | −0.10 | shadow; view only |
| SPY | reference: buy the S&P 500 | 14.9 % | 14.9 % | 1.01 | −23.8 % | — | — |
| Equal-weight universe | reference: every stock equally | 11.5 % | 18.7 % | 0.68 | −33.5 % | +1.15 | — |

- Annual return is compounded; drawdowns are measured at month ends. "Return ÷ vol" is annualised mean over volatility with cash earning nothing.
- Costs: 0.4 % round trip times turnover, plus 1 % a year to borrow the shorts.
- P2a and P5b look better mainly because of their SPY leg in a rising market, not better stock picking.
- The shadow books start with the November 2026 rebalance. Six more variants (P4a–d, P5a, P6) are backtest only (Part 6); hedging P1 with index futures or options is in Part 7.

## How it went, honestly

- **Backtest.** P1 makes about 0.76 % a month, a return-to-volatility ratio of 0.55 with a −40 % worst drawdown, and it leans against the market (beta −0.30).
- **How reliable.**
  - A stationary block bootstrap gives a 90 % interval of 0.09 to 1.00 for the ratio.
  - Taken alone, the probability that the true ratio is above zero is about 98 %.
  - Eighteen portfolio variants were tried. If all of them were luck, the best would be expected to show 0.51. The probability that P1's true ratio beats that bar (the deflated Sharpe ratio) is only about 56 %.
- **How fragile.**
  - The ratio is flat across 30–70 names per side (0.53–0.59), so it is not a tuned spike.
  - It is very sensitive to costs: doubling them gives 0.35.
  - It leans on momentum: dropping that one component gives 0.24.
- **What real money would lose.**
  - The backtest implicitly earns the T-bill rate on short-sale proceeds, as an institution would. A retail account earns little or nothing on them.
  - Earning nothing at today's T-bill rate puts P1's ratio at about 0.30. At a broker tier that pays 1.25 % a year below the benchmark it is about 0.48.
  - Rebalancing trades the backtest did not charge for (re-equalising drifted weights, resizing with equity) cost about another 0.03 % a month (ratio 0.53).
  - Hard-to-borrow shorts may simply be unavailable, and that cannot be measured historically.
- **The scenario to fear.**
  - A 2009-style "junk rally" after a bear market is not in the backtest period.
  - Applying P1's factor exposures to March–May 2009 gives about −26 % over three months, with a 90 % lower bound of −40 %, which is the stop line.
  - A linear factor model probably understates a crash.

## Where it stands

- **Against the market.**
  - Over the same period SPY had a return-to-volatility ratio of 1.01, or a Sharpe ratio of 0.90 after subtracting the T-bill rate. The 0.90 is the fair comparison, because P1's long-minus-short return is already an excess return. Either way SPY did clearly better than P1's 0.55.
  - But this was a long bull market, and P1 is a hedged long–short book with a −0.28 monthly correlation to SPY. Its value is that it moves differently, not that it beats the index.
- **A long-run yardstick.** In Kenneth French's public data for 1927–2009, before costs, the US equity premium has a ratio of about 0.39 and the academic momentum factor about 0.49. P1's backtest is of that order.
- **The literature's base rate.**
  - After publication and after costs, the average anomaly earns about 0.04 % a month (Chen and Velikov, 2023).
  - Across 162 anomalies the long–short profit comes from shorts that are expensive to borrow and is about zero after borrow fees (Muravyev, Pearson and Pollet, 2025).
  - The backtest should be discounted accordingly.
- **How a practitioner would see it** (a judgement, not data):
  - A long–short book on one signal, equally weighted and rebalanced monthly, is a building block in a larger book, not a product on its own.
  - On its own it is mediocre. Its use is diversification.

## What passed and what failed

Twenty tests are in the registry. Three passed:
- the mispricing signal, twice (#2 and its survivorship-bias-free re-run #9);
- the variance risk premium (#19, t 4.4), which is crash-prone insurance selling through an instrument individuals cannot trade (Part 7).

The rest:
- **Equity signals that failed (12)**:
  - 10-K text change, value and residual momentum;
  - three self-mined accounting signals, opportunistic insider trades and a calendar composite;
  - two more self-mined signals, a literature composite, and the mispricing signal on small caps.
- **Cross-asset (3)**: time-series trend, a carry / value / low-beta composite, a macro composite.
- **Event-driven (1)**: cash merger arbitrage.
- **Not run (1)**: put writing with daily hedging, which needed a paid options dataset.

**The latest literature survey** (October 2026) found nothing worth adding:
- no new free-data signal had a reasonable chance of passing;
- making the existing signal "safer", for example volatility-managed momentum, mostly helps over long samples before costs, and the gain largely disappears out of sample after costs.
