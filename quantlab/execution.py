"""Transaction costs, optimal execution and TCA (illustrative numbers or the original paper's example; no market data).

* Cost of one trade (bp) = half spread + commission + market impact; impact by the square-root law: Y * sigma_daily * sqrt(Q / ADV)
  (Y empirically ~0.5-1 with an exponent near 0.5; default Y = 0.7).
* Almgren & Chriss (2000): sell X shares over T in N intervals of length tau; temporary impact eta * (n_k / tau), permanent impact gamma * n_k.
  E = 0.5 gamma X^2 + eps sum|n_k| + (eta_tilde / tau) sum n_k^2,  V = sigma^2 tau sum x_k^2,  eta_tilde = eta - gamma tau / 2.
  The minimiser of E + lam V is x_j = X sinh(kappa (T - t_j)) / sinh(kappa T) with (2/tau^2)(cosh(kappa tau) - 1) = lam sigma^2 / eta_tilde;
  checked against a direct tridiagonal solve and against simulation.
* Implementation shortfall (Perold 1988) = delay + execution + opportunity + fees (identity checked on random cases).
* Capacity: monthly cost drag of a 50/50 long-short strategy under assumed liquidity, versus its gross backtest return.
Run `python -m quantlab.execution` for the three demos.
"""
import math

import numpy as np

from ._util import Checker


def impact_bp(Q, adv, sigma_d, Y=0.7):
    return Y * sigma_d * math.sqrt(Q / adv) * 1e4


def one_way_cost_bp(Q, adv, sigma_d, spread_bp, commission_bp=0.5, Y=0.7):
    return spread_bp / 2 + commission_bp + impact_bp(Q, adv, sigma_d, Y)


def ac_kappa(lam, sigma, eta, gamma, tau):
    eta_t = eta - gamma * tau / 2
    if eta_t <= 0:
        raise ValueError("eta - gamma*tau/2 must be positive")
    return math.acosh(1 + lam * sigma ** 2 / eta_t * tau ** 2 / 2) / tau


def ac_trajectory(X, N, T, lam, sigma, eta, gamma):
    """Optimal holdings x_0..x_N (x_0 = X, x_N = 0). lam = 0 gives the straight line (TWAP)."""
    tau = T / N
    t = np.arange(N + 1) * tau
    if lam == 0:
        return X * (1 - t / T)
    k = ac_kappa(lam, sigma, eta, gamma, tau)
    return X * np.sinh(k * (T - t)) / np.sinh(k * T)


def ac_cost(x, tau, sigma, eta, gamma, eps=0.0):
    """-> (expected cost E, variance V) of a selling trajectory x_0..x_N."""
    n = -np.diff(x)
    eta_t = eta - gamma * tau / 2
    E = 0.5 * gamma * x[0] ** 2 + eps * np.sum(np.abs(n)) + eta_t / tau * np.sum(n ** 2)
    return float(E), float(sigma ** 2 * tau * np.sum(x[1:] ** 2))


def ac_direct(X, N, T, lam, sigma, eta, gamma):
    """Solve d(E + lam V)/dx_j = 0 directly: (eta_t/tau)(2x_j - x_{j-1} - x_{j+1}) + lam sigma^2 tau x_j = 0, a tridiagonal system."""
    tau = T / N
    a = (eta - gamma * tau / 2) / tau
    m = N - 1
    A = np.zeros((m, m))
    for j in range(m):
        A[j, j] = 2 * a + lam * sigma ** 2 * tau
        if j > 0:
            A[j, j - 1] = -a
        if j < m - 1:
            A[j, j + 1] = -a
    b = np.zeros(m)
    b[0] = a * X
    return np.r_[X, np.linalg.solve(A, b), 0.0]


def simulate(x, S0, sigma, eta, gamma, tau, eps=0.0, n_paths=2000, seed=0):
    """Implementation shortfall X*S0 - proceeds, following the model's timing: interval k trades at the start-of-interval price minus temporary impact;
    the end-of-interval price = start + sigma sqrt(tau) Z - gamma n_k."""
    rng = np.random.default_rng(seed)
    n = -np.diff(x)
    out = np.empty(n_paths)
    for s in range(n_paths):
        S, cash = S0, 0.0
        for k in range(len(n)):
            cash += n[k] * (S - eps * np.sign(n[k]) - eta * n[k] / tau)
            S = S + sigma * math.sqrt(tau) * rng.standard_normal() - gamma * n[k]
        out[s] = x[0] * S0 - cash
    return out


def implementation_shortfall(side, Q, decision_px, arrival_px, fill_qty, fill_px, end_px, fees=0.0):
    """side: +1 buy, -1 sell -> dict(delay, execution, opportunity, fees, total, total_bp, paper_pnl, actual_pnl)."""
    q, p = np.asarray(fill_qty, float), np.asarray(fill_px, float)
    filled = q.sum()
    delay = side * filled * (arrival_px - decision_px)
    execution = side * float(np.sum(q * (p - arrival_px)))
    opportunity = side * (Q - filled) * (end_px - decision_px)
    total = delay + execution + opportunity + fees
    return dict(delay=delay, execution=execution, opportunity=opportunity, fees=fees, total=total, total_bp=total / (Q * decision_px) * 1e4,
                paper_pnl=side * Q * (end_px - decision_px), actual_pnl=side * float(np.sum(q * (end_px - p))) - fees)


GROSS_MONTHLY = 0.01042   # gross monthly long-short return of the lab's backtest (same history; illustrative only)
BORROW = 0.01
SIDE_TURNOVER = 0.245     # fraction of names replaced per side per month in that backtest


def capacity(aum_per_side=(1e5, 1e6, 1e7, 1e8, 1e9), n_draws=20000, seed=0):
    """Assumed liquidity: median ADV $60m (longs) / $40m (shorts), log-normal, floored at $10m; daily vol 2 % / 3.5 %; spreads 4 / 8 bp.
    Each name is 1/50 of a side; trading replaces SIDE_TURNOVER of each side per month, all at one close."""
    rng = np.random.default_rng(seed)
    sides = dict(long=dict(median=60e6, vol=0.02, spread=4.0), short=dict(median=40e6, vol=0.035, spread=8.0))
    advs = {}
    for k, v in sides.items():
        a = v["median"] * np.exp(rng.standard_normal(n_draws))
        advs[k] = a[a >= 10e6][: n_draws // 2]
    out = {}
    for A in aum_per_side:
        order = A / 50
        costs, big = [], []
        for k, v in sides.items():
            costs.append(np.mean([one_way_cost_bp(order, adv, v["vol"], v["spread"]) for adv in advs[k]]) / 1e4)
            big.append(float(np.mean(order / advs[k] > 0.10)))
        monthly_cost = sum(2 * SIDE_TURNOVER * c for c in costs)
        out[A] = dict(avg_bp=float(np.mean(costs) * 1e4), monthly_cost=monthly_cost, monthly_net=GROSS_MONTHLY - monthly_cost - BORROW / 12,
                      share_over_10pct_adv=max(big))
    return out


def demos():
    print("1. One-way cost by order size (ADV $50m, daily vol 2 %, spread 5 bp):")
    print("   " + "; ".join(f"${Q:,.0f}: {one_way_cost_bp(Q, 50e6, 0.02, 5):.1f} bp" for Q in (1e4, 1e5, 1e6, 1e7)))
    print("2. Capacity of a 50/50 long-short strategy (assumed liquidity; gross 1.042 %/month, borrow 1 %/yr):")
    for A, v in capacity().items():
        print(f"   ${A:>13,.0f} per side: avg {v['avg_bp']:6.1f} bp, monthly cost {v['monthly_cost']:.3%}, monthly net {v['monthly_net']:+.3%}, "
              f"orders > 10 % of ADV {v['share_over_10pct_adv']:.0%}")
    X, S0, sigma, eta, gamma, eps, T, N = 1e6, 50.0, 0.95, 2.5e-6, 2.5e-7, 0.0625, 5.0, 20
    tau = T / N
    print("3. Almgren-Chriss frontier (paper's example: 1m shares at $50 over 5 days, 20 intervals):")
    for name, lam in (("TWAP (lam = 0)", 0.0), ("lam = 1e-7", 1e-7), ("lam = 1e-6", 1e-6), ("lam = 1e-5", 1e-5)):
        x = ac_trajectory(X, N, T, lam, sigma, eta, gamma)
        E, V = ac_cost(x, tau, sigma, eta, gamma, eps)
        print(f"   {name:15}: expected cost {E / (X * S0) * 1e4:6.1f} bp, std ${math.sqrt(V):,.0f}")
    x1 = np.r_[X, np.zeros(N)]
    E1, _ = ac_cost(x1, tau, sigma, eta, gamma, eps)
    print(f"   {'sell immediately':15}: expected cost {E1 / (X * S0) * 1e4:6.1f} bp, no price risk")


def selftest():
    np.seterr(divide="ignore", over="ignore", invalid="ignore")
    check = Checker()
    c1, c4 = impact_bp(1e6, 50e6, 0.02), impact_bp(4e6, 50e6, 0.02)
    check(f"square-root law: 4x the order -> 2x the impact ({c1:.1f} -> {c4:.1f} bp); 4x the ADV -> half", abs(c4 / c1 - 2) < 1e-12
          and abs(impact_bp(1e6, 200e6, 0.02) / c1 - 0.5) < 1e-12)
    X, N, T, sigma, eta, gamma = 1e6, 20, 1.0, 0.95, 2.5e-6, 2.5e-7
    for lam in (1e-7, 1e-6, 2e-6):
        check(f"Almgren-Chriss (lam = {lam:g}): closed-form trajectory = direct tridiagonal solve",
              np.max(np.abs(ac_trajectory(X, N, T, lam, sigma, eta, gamma) - ac_direct(X, N, T, lam, sigma, eta, gamma))) < 1e-6 * X)
    x0, x1, x2 = (ac_trajectory(X, N, T, l, sigma, eta, gamma) for l in (0, 1e-7, 2e-6))
    check("lam = 0 is a straight line; larger lam trades faster", np.allclose(np.diff(x0), -X / N) and x2[N // 2] < x1[N // 2] < x0[N // 2])
    E0, V0 = ac_cost(x0, T / N, sigma, eta, gamma)
    E2, V2 = ac_cost(x2, T / N, sigma, eta, gamma)
    check("trading faster costs more and risks less", E2 > E0 and V2 < V0)
    J = lambda x, l: (lambda ev: ev[0] + l * ev[1])(ac_cost(x, T / N, sigma, eta, gamma))
    rng = np.random.default_rng(1)
    check("optimality: 200 perturbed trajectories all have higher E + lam V", all(J(x1, 1e-7) <= J(x1 + np.r_[0, rng.standard_normal(N - 1) * 2000, 0], 1e-7) + 1e-6
                                                                                 for _ in range(200)))
    sim = simulate(x0, 50.0, sigma, eta, gamma, T / N, n_paths=4000, seed=2)
    E, V = ac_cost(x0, T / N, sigma, eta, gamma)
    check(f"simulation matches the formulas: mean {sim.mean():,.0f} vs {E:,.0f}, std {sim.std():,.0f} vs {math.sqrt(V):,.0f}",
          abs(sim.mean() - E) < 0.03 * E + 3 * math.sqrt(V) / math.sqrt(4000) and abs(sim.std() / math.sqrt(V) - 1) < 0.03)
    ok = True
    for _ in range(200):
        s = rng.choice([-1, 1])
        Q = rng.integers(1000, 100000)
        q = rng.integers(0, Q // 4, 4)
        r = implementation_shortfall(s, Q, 50 + rng.normal(), 50 + rng.normal(), q, 50 + rng.normal(size=4), 50 + rng.normal(), rng.uniform(0, 50))
        ok &= abs(r["total"] - (r["paper_pnl"] - r["actual_pnl"])) < 1e-6
    check("implementation shortfall: delay + execution + opportunity + fees = paper - actual P&L (200 random cases)", bool(ok))
    r = implementation_shortfall(+1, 10000, 100.0, 100.2, [6000, 2000], [100.3, 100.5], 101.0, 10.0)
    check("worked example: delay 1600, execution 1200, opportunity 2000, total 4810 (48.1 bp)",
          abs(r["delay"] - 1600) < 1e-6 and abs(r["execution"] - 1200) < 1e-6 and abs(r["opportunity"] - 2000) < 1e-6 and abs(r["total_bp"] - 48.1) < 1e-6)
    cap = capacity(aum_per_side=(1e5, 1e8), n_draws=4000)
    check("capacity: more money -> higher costs, lower net, more orders above 10 % of ADV; at $100k the cost is below the lab's 0.2 % assumption",
          cap[1e8]["monthly_cost"] > cap[1e5]["monthly_cost"] and cap[1e8]["monthly_net"] < cap[1e5]["monthly_net"]
          and cap[1e8]["share_over_10pct_adv"] >= cap[1e5]["share_over_10pct_adv"] and cap[1e5]["monthly_cost"] < 2 * SIDE_TURNOVER * 2 * 0.002)
    return check.summary()


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    demos()
