# Part 3 — Machine learning and overfitting

Code: `quantlab/ml_data.py` (synthetic panel), `quantlab/gbdt.py` (histogram gradient boosting + ridge), `quantlab/validation.py`
(purged CV, Newey–West, deflated Sharpe), `quantlab/ml_demo.py`. The panel has 20 persistent characteristics, of which 4 carry signal
(linear, U-shaped and an interaction) with a target IC of 0.06; the rest are noise.

## Gradient boosting from scratch

Features are binned at training quantiles (≤ 32 bins); each node tries every bin edge of every feature and keeps the largest
gain G_L²/(n_L+λ) + G_R²/(n_R+λ) − G²/(n+λ) (for squared loss with λ = 0 this equals the reduction in squared error — checked against brute force).
Boosting fits small trees to residuals with a learning rate and row / column subsampling.

## 1. When does ML beat a linear model?

Walk-forward, out of sample (180 months × 300 stocks):

| Truth | IC of the true signal | Gradient-boosted trees | Ridge |
|---|---|---|---|
| Nonlinear (U-shape + interaction) | 0.061 | **0.034** | 0.023 |
| Linear | 0.062 | 0.045 | **0.055** |

Trees win only when the truth is nonlinear; with a linear truth and 16 noise features, ridge wins. At low signal-to-noise, variance dominates.

## 2. Shuffled cross-validation cheats

Pure noise (no signal at all), overlapping 3-month labels, deeper trees (depth 6, min leaf 5):

| Seed | Shuffled K-fold | Purged K-fold | Walk-forward |
|---|---|---|---|
| 4 | IC **+0.207** (Newey–West t 26.8) | +0.006 (0.8) | +0.019 (1.7) |
| 5 | IC **+0.218** (26.3) | +0.004 (0.5) | −0.016 (−1.5) |
| 6 | IC **+0.209** (28.7) | +0.014 (1.6) | +0.019 (2.3) |

- Persistent characteristics plus overlapping labels put a near-copy of every test row in the training set; deep trees memorise it.
  An IC of 0.21 on pure noise is several times stronger than the best real equity signals.
- **Purging** removes training samples whose label windows overlap the test block; an **embargo** drops a few more after it (López de Prado 2018, ch. 7).
- Monthly ICs are autocorrelated, so t-statistics need a HAC correction (Newey–West here; the research framework uses EWC).
- Even then, walk-forward with only three re-fits produced t = 2.3 on noise once (seed 6): one number is never enough. (Over 24 further seeds,
  purged K-fold with shallower trees averaged −0.001 on noise, positive in 11 of 24 — no bias.)

## 3. The best of many trials

Thirty hyper-parameter sets, keep the best walk-forward result, then test it on fresh data from the same process:

| Data | Best IC / annual IR | P(skill > 0), on its own | Deflated for 30 trials | IC on fresh data |
|---|---|---|---|---|
| Pure noise | +0.007 / 0.34 | 79 % | **30 %** | −0.016 |
| Real signal (IC 0.06) | +0.036 / 1.75 | 100 % | **100 %** | +0.032 |

- Expected maximum Sharpe of N skill-less trials ≈ √V·[(1−γ)Φ⁻¹(1−1/N) + γΦ⁻¹(1−1/(Ne))]; using it as the threshold gives the **deflated Sharpe ratio**
  (Bailey & López de Prado 2014). It is conservative: in simulation the best of 100 skill-less strategies "looks significant" 99 % of the time and passes the deflated test 0 % of the time.
- The 30 configurations here are similar to each other, so the inflation is mild; real research tries far more varied things.
  That is why the research process in Part 1 records **every** test in an append-only registry.
