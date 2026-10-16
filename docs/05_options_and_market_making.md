# Part 5 — Options, probability and market making

Code: `quantlab/options.py`, `quantlab/probability.py` (each has `demos()`; run `python -m quantlab.options` / `python -m quantlab.probability`).

## Options

- **Black–Scholes** with a dividend yield; all five Greeks match finite differences; Hull's textbook example (S = 42, K = 40, r = 10 %, σ = 20 %, T = 0.5) gives call 4.76, put 0.81.
- **CRR binomial tree** converges to Black–Scholes (errors 0.030 / 0.006 / 0.001 at 50 / 200 / 800 steps); an American put is worth more than the European (13.74 vs 12.66 at K = 110);
  an American call without dividends equals the European.
- **Implied volatility** by bisection; a synthetic equity skew (28.6 % at K = 80, 20.0 % at the money, 17.4 % at K = 120) is recovered exactly from prices.
- Rule of thumb: at-the-money call ≈ 0.4·σ·S·√T, straddle ≈ 0.8·σ·S·√T.

### Delta hedging: what a short option earns

Sell an at-the-money 3-month call at 25 % implied volatility and delta-hedge at the realised volatility:

| Realised vol | Theory: C(25 %) − C(realised) | Daily hedge: mean / std | Weekly hedge: mean / std |
|---|---|---|---|
| 15 % | +1.99 | +2.00 / 0.33 | +2.01 / 0.71 |
| 20 % | +1.00 | +1.00 / 0.44 | +1.02 / 0.94 |
| 25 % | 0.00 | +0.01 / 0.55 | +0.03 / 1.18 |
| 30 % | −1.00 | −0.99 / 0.66 | −0.96 / 1.42 |

A hedged option's P&L is driven by realised versus implied variance (≈ ½Γ S²(σ_impl² − σ_real²)dt per step for the seller); more frequent hedging
tightens the distribution. Selling options harvests the variance risk premium — with crash risk attached.

## Probability (each answer checked by simulation)

| Puzzle | Answer |
|---|---|
| Expected rolls until the first six | 6 |
| Expected flips until two heads in a row | 6 |
| Die paid at face value with one optional re-roll | 4.25 (re-roll on 1–3) |
| Rolls to see all six faces | 14.7 |
| Alternating coin flips, first head wins: first player's chance | 2/3 |
| Gambler's ruin, 3 → 10 with p = 0.45 | 0.128 |
| 23 people share a birthday | 0.507 |
| Monty Hall, switching | 2/3 |

## Kelly sizing

For p = 60 % at even odds, f\* = p − q/b = 20 %. Half Kelly keeps about three-quarters of the growth rate with far less variance; double Kelly has
negative long-run growth. In continuous form the Kelly leverage is μ/σ² — which is why estimation error in μ makes full Kelly dangerous in practice.

## Market making (simplified Glosten & Milgrom 1985)

The asset is worth 99 or 101; a fraction α of traders know which. Zero-profit quotes are ask = E[V | buy], bid = E[V | sell], and the first spread equals α × 2.
With α = 0.3, quoting a half-spread of 0.1 loses 0.20 per first trade to informed traders (theory −0.20), a half-spread of 0.6 earns 0.30 (absent competition),
and Glosten–Milgrom quotes break even. After 50 trades the dealer's price is on average 0.04 from the true value: trading moves information into prices.
