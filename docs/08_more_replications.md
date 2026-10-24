# Part 8 — Combining strategies, and two more replications that failed

*October 2026. Everything here is a **replication** (the ideas come from published papers and the data overlap the years already seen), so none of it is evidence
for trading. Rules were written and committed before any number was computed, and each test was run once. No individual stock or amount of money is shown.*

This part covers three pieces of work:

1. **P10 — combining the long–short book with two other passed strategies** (backtest only). Does diversification across unrelated return streams help?
2. **Test #33 — "betting against beta" in US Treasuries** (Frazzini and Pedersen, 2014). Picked from a new literature survey for low correlation with stocks.
3. **Test #34 — statistical arbitrage with PCA residuals and an Ornstein–Uhlenbeck model** (Avellaneda and Lee, 2010). A classic daily strategy, tested at the owner's request.

---

## 1. P10: three return streams at equal risk (backtest only)

**Rule.** Take three monthly return streams — the equity long–short book P1 (Part 6), the derivatives-and-calendar composite #32 (Part 7) and the Treasury
month-end strategy #29 (Part 7). Scale each to 10 % annual volatility using only the previous 36 months (at most 3× leverage; the same rule #32 already used inside itself),
then average them. Charge P1 for resizing its positions. The sample is every month where all three exist: **2014-08 → 2026-06, 143 months**.

| | Return ÷ volatility (90 % interval) | Max drawdown ÷ annual vol |
|---|---|---|
| **P10 (all three)** | **1.18** (0.61–1.91) | −2.06 |
| P1 alone, scaled the same way | 0.54 | −2.92 |
| P1 + #32 (without #29) | 0.83 | — |
| #29 alone, scaled | 1.05 | — |

- The three are almost uncorrelated (pairwise −0.06, −0.06, −0.03). In a paired block bootstrap P10 beats P1 in 99.6 % of resamples, so the rule fixed in advance reads
  "the diversification benefit is visible in this history".
- **How to read it.** Much of the gain comes from #29, which only just passed and fails at double costs. P10's two worst months (2025-04 −8.5 %, 2020-03 −5.8 %) are the
  variance-selling part of #32 losing in equity crashes. Part of #32 is variance swaps, which individuals cannot trade.
- **Discount it.** Bank-published alternative-risk-premia indices are reported to have lost roughly three quarters of their backtest Sharpe ratio once live (Suhonen, Lennkh and Perez, 2017),
  and all three streams were chosen on the same history. The lesson is the arithmetic of diversification, not a tradable 1.18.

## 2. Test #33: betting against beta in Treasuries

**Why this one.** A literature survey for strategies that diversify an equity book found four candidates (stock–bond month-end rebalancing flows, the Treasury auction
cycle, Treasury BAB, country ETF momentum). Power was first computed on the papers' **gross** returns, which made all four look testable. An independent review pointed out
that the test uses **net** returns: after costs the auction trade kept about 0.3 bp of a 4.3 bp edge per auction, and the other two had under 12 % power.
Only the Treasury BAB was kept, tested alone at a bar of 1.96.

**Rule** (Frazzini and Pedersen, 2014, Table 6). Six constant-maturity Treasuries (1, 2, 3, 5, 7, 10 years, par-bond approximation from daily yields).
Each month, estimate every maturity's beta to the equal-weighted Treasury market (12-month volatility, 60-month correlation); rank, put rank weights on the low-beta
and high-beta halves, lever each half to beta 1, and go long the low half and short the high half. Costs: 1 bp per unit of notional traded plus quarterly futures rolls.

A second review caught a trap: the paper's text says betas are shrunk towards 1, but its Treasury table plainly was not (an ex-ante beta of 0.14 is impossible after
shrinkage). Shrinking would turn the strategy into a net short of Treasuries — about −0.01 % a month on the paper's own numbers — so the table's construction was used.

**Result: fail.** 644 months (1965–2009 and 2018–2026; 2010–2017 excluded as already-explored years).

| | EWC t | Annual Sharpe |
|---|---|---|
| Main sample | **1.23** | 0.28 |
| 1965–2009 | 1.82 | 0.41 |
| 2018–2026 | −0.44 | −0.29 |
| No costs at all | 1.52 | — |

The realized beta to the Treasury market is 0.03 (the paper reports 0.01), so the construction matches. The effect is in the right direction before 2010 but much weaker than
in the paper (we only have 1–10 year maturities, where the paper spans under 1 year to over 10), and negative since 2018.

## 3. Test #34: statistical arbitrage with PCA residuals and an OU model

**Rule** (Avellaneda and Lee, 2010; parameters from the paper). Every day, for the 1,000 most-traded US stocks (one share class per company; delisted firms included):

1. PCA of the past 252 days of standardized returns; the top **15 eigenportfolios** are the common factors.
2. Regress each stock's last 60 days of returns on those factors; the cumulative residual is fitted as an **Ornstein–Uhlenbeck** (mean-reverting) process.
   Only stocks that revert within 30 days are traded.
3. The **s-score** measures how far the residual is from equilibrium (centred across stocks). Open long below −1.25, short above +1.25; close longs above −0.50 and shorts below +0.75.
4. Hedge every position with the eigenportfolios; trade at the close; 5 bp per side on **both** the stock and the hedge legs.

**Result: fail.** 237 months (1999–2009 and 2018–2026).

| | EWC t | Mean per month |
|---|---|---|
| Main result (costs on both legs) | **−0.06** | −0.03 % |
| No costs | **2.76** | +1.55 % |
| Costs on the stock leg only (the paper's formula) | 1.46 | +0.74 % |
| Trade one day later (signal at close t, execute at close t+1) | −1.64 | −0.61 % |
| 1999–2002 / 2003–2007 / 2008–09 / 2018–26 | 2.00 / −2.92 / 1.17 / −2.72 | |

On average the book held 184 longs and 160 shorts for about 9 days and traded about 1.5× its equity every day.

**What it says.**
- The residual mean reversion is real: before costs the t-statistic is 2.8.
- Costs at 5 bp per side on that turnover remove all of it.
- Part of the gross profit is an artefact of computing the signal and trading at the **same** close (bid–ask bounce): delaying execution by one day turns it negative.
- It worked in 1999–2002 and not since 2003, which matches the paper's own warning and twenty years of crowding.
- The hard part of statistical arbitrage is execution — near-zero costs, trading the moment the signal appears — not the PCA or the OU formula.

## 4. What I took away

- **Power must be computed on what the test actually measures.** For thin strategies, costs decide whether a test can say anything at all.
- **Read the paper's tables, not just its text.** One sentence about shrinkage would have tested the opposite of the strategy.
- **Diversification is the free lunch that survives scrutiny best**, but only if the components are real and tradable; a backtest that combines selected survivors
  overstates it.
- Both failed lines are closed under the project's rules: no re-test with a different definition.

### References

- Avellaneda, M. and Lee, J.-H. (2010). Statistical arbitrage in the US equities market. *Quantitative Finance* 10(7), 761–782.
- Frazzini, A. and Pedersen, L. H. (2014). Betting against beta. *Journal of Financial Economics* 111(1), 1–25.
- Durham, J. B. (2015). Betting against beta (and gamma) using government bonds. Federal Reserve Bank of New York Staff Report 708.
- Ilmanen, A., Israel, R., Lee, R., Moskowitz, T. and Thapar, A. (2021). How do factor premia vary over time? A century of evidence. *Journal of Investment Management* 19(4).
- Suhonen, A., Lennkh, M. and Perez, F. (2017). Quantifying backtest overfitting in alternative beta strategies. *Journal of Portfolio Management*, Winter 2017.
- DeMiguel, V., Garlappi, L. and Uppal, R. (2009). Optimal versus naive diversification: how inefficient is the 1/N portfolio strategy? *Review of Financial Studies* 22(5).
