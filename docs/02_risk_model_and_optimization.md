# Part 2 — Risk model and portfolio optimisation

Code: `quantlab/risk_model.py`, `quantlab/optimizer.py`, `quantlab/portfolio_demo.py`. Everything runs on a synthetic market whose true factor
covariance and specific risks are known, so each estimate can be checked against the truth.

## Why a risk model

A 500-stock covariance matrix has N(N+1)/2 = 125,250 entries. With 250 days of data the sample covariance has rank ≤ 249: it believes some portfolios
are riskless, and an optimiser will lever into exactly those directions. Out of sample (re-estimated every 21 days, past data only):

| Covariance used for the minimum-variance portfolio | Realised annualised vol | Median leverage |
|---|---|---|
| True covariance (benchmark) | 9.6 % | 4.5 |
| Factor model | **10.9 %** | 2.8 |
| Ledoit–Wolf shrinkage | 15.3 % | 5.6 |
| Sample covariance | **2,745 %** | 205 |

A factor model (15 factors) needs 120 factor covariances plus 500 specific variances instead of 125,250 numbers.

## The model (Barra-style)

- **Exposures**: market (all ones), styles (size, value, momentum, volatility; standardised daily) and industry dummies.
- **Daily cross-sectional regression** with weights ∝ √(market cap) (a proxy for 1 / specific variance).
  The market column equals the sum of the industry dummies, so the system is collinear; the usual fix — cap-weighted industry returns sum to zero — is solved with a Lagrange multiplier.
- **Factor covariance**: exponentially weighted (half-life 90 days). **Specific variance**: exponentially weighted squared residuals (half-life 60), shrunk toward the cross-sectional median.
- Estimation error of a factor return ≈ specific vol / √N, so with N = 200 the estimated style returns correlate only 0.75–0.9 with the truth (the self-test checks > 0.7).
- **Testing a risk model**: the bias statistic (std of return / predicted vol) is ≈ 1 for random portfolios under every estimator — random portfolios are easy.
  The hard test is an optimised portfolio, because the optimiser seeks exactly the directions where the model underestimates risk ("error maximisation", Michaud 1989).
- **Risk decomposition**: variance = x'Fx (factor part, x = X'w) + Σ wᵢ²Dᵢ; asset risk contributions wᵢ(Σw)ᵢ/σ sum to σ (Euler).

## The optimiser

maximise α'w − (λ/2)w'Σw − Σκᵢ|wᵢ − w0ᵢ| subject to Aw = b (dollar, factor and industry neutrality) and |wᵢ| ≤ capᵢ.

- **ADMM** in plain NumPy: a KKT linear solve for the quadratic + equality part, a closed-form per-asset step (soft-threshold around w0, then clip) for costs and limits,
  residual-balanced ρ, and OSQP-style **polishing** of the final active set. Linearly dependent constraints are removed by SVD first.
- **Verified three independent ways**: against the closed-form solution (equalities only), against brute-force enumeration of all active sets
  (3⁶ with limits; 5⁵ with costs), and against 300 nearby feasible random points. Halving the soft-threshold (a deliberate bug) is caught.
- α from the signal follows Grinold: α = IC × specific vol × standardised score.

## Same signal, three constructions (`python -m quantlab.portfolio_demo`)

A weak signal (realised rank IC 0.017–0.032) that is partly a "low-volatility" style tilt — as real signals often are. 500 stocks, 59 monthly periods, 0.2 % cost per unit traded:

| Seed | A: equal-weight 10 % long/short — IR gross / net | B: optimised, factor-neutral | C: optimised + costs | Turnover A / B / C |
|---|---|---|---|---|
| 11 | 0.70 / 0.14 | 2.12 / 1.09 | 1.86 / 1.60 | 171 % / 129 % / 36 % |
| 12 | 1.13 / 0.69 | 1.64 / 0.82 | 0.88 / 0.66 | 168 % / 127 % / 36 % |
| 13 | 1.37 / 0.75 | 2.67 / 1.50 | 2.10 / 1.83 | 174 % / 127 % / 37 % |
| 14 | 1.14 / 0.49 | 2.53 / 1.52 | 2.41 / 2.14 | 176 % / 127 % / 36 % |
| 15 | 1.51 / 0.93 | 2.52 / 1.63 | 2.11 / 1.86 | 171 % / 128 % / 38 % |

- Factor neutrality (B) beats equal weighting (A) before costs in every seed: the style tilt added risk (45 % of A's predicted risk was factor risk) but no return.
- Cost-aware optimisation (C) trades about a fifth as much and has the best net IR in 4 of 5 seeds.
- The synthetic market is "clean" (the risk model has the right structure), so real-world gains are smaller; the direction is the point.

This is exactly what the live equal-weight portfolio in Part 1 lacks: its ex-ante beta of about −1.4 is an unintended market bet that a factor-neutral construction would remove.
