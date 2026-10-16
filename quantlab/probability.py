"""Trading-interview probability, Kelly sizing and a Glosten-Milgrom market-making simulation.

1. Eight classic puzzles, each with a closed-form answer and a Monte Carlo check.
2. Kelly: win probability p, payoff b:1, bet fraction f: growth g(f) = p ln(1 + f b) + q ln(1 - f), optimum f* = p - q/b.
   Half Kelly keeps ~3/4 of the growth with far less variance; double Kelly has negative growth (for small edges).
3. Market making (simplified Glosten & Milgrom 1985): value is V_lo or V_hi (50/50, unknown to the dealer); each trader is informed with
   probability alpha. Zero-expected-profit quotes: ask = E[V | buy], bid = E[V | sell]; the first spread equals alpha x (V_hi - V_lo).
   Quoting tighter loses to informed traders; trades move the dealer's belief (Bayes) toward the true value.
Run `python -m quantlab.probability` for the demos.
"""
import math

import numpy as np

from ._util import Checker


def first_six(rng, n=200000):
    """Expected rolls until the first 6: geometric, 1/p = 6."""
    return 6.0, float(rng.geometric(1 / 6, n).mean())


def two_heads(rng, n=100000):
    """Expected flips until two heads in a row: E0 = 1 + E1/2 + E0/2, E1 = 1 + E0/2 -> E0 = 6."""
    tot = 0
    for _ in range(n):
        k = c = 0
        while c < 2:
            k += 1
            c = c + 1 if rng.random() < 0.5 else 0
        tot += k
    return 6.0, tot / n


def one_reroll(rng, n=400000):
    """Paid the face value of a die, with one optional re-roll: re-roll on 1-3 (re-roll is worth 3.5): (4+5+6)/6 + (3/6)(3.5) = 4.25."""
    a, b = rng.integers(1, 7, n), rng.integers(1, 7, n)
    return 4.25, float(np.where(a <= 3, b, a).mean())


def all_faces(rng, n=100000):
    """Expected rolls to see all six faces (coupon collector): 6 (1 + 1/2 + ... + 1/6) = 14.7."""
    tot = 0
    for _ in range(n):
        seen, k = set(), 0
        while len(seen) < 6:
            k += 1
            seen.add(int(rng.integers(1, 7)))
        tot += k
    return 6 * sum(1 / i for i in range(1, 7)), tot / n


def first_head_wins(rng, n=200000):
    """Players alternate flips; first head wins. P(first player) = 1/2 + P/4 -> 2/3."""
    return 2 / 3, float(np.mean(rng.geometric(0.5, n) % 2 == 1))


def gamblers_ruin(rng, n=40000, a=3, N=10, p=0.45):
    """Start with a, win/lose 1 with prob p/q, stop at N or 0: P(reach N) = (1 - (q/p)^a) / (1 - (q/p)^N)."""
    r = (1 - p) / p
    exact = (1 - r ** a) / (1 - r ** N)
    wins = 0
    for _ in range(n):
        x = a
        while 0 < x < N:
            x += 1 if rng.random() < p else -1
        wins += x == N
    return exact, wins / n


def birthday(rng, n=100000, k=23):
    exact = 1 - math.prod((365 - i) / 365 for i in range(k))
    d = np.sort(rng.integers(0, 365, (n, k)), 1)
    return exact, float(np.mean(np.any(np.diff(d, axis=1) == 0, axis=1)))


def monty_hall(rng, n=200000):
    """Switching wins iff the first pick was wrong: 2/3."""
    return 2 / 3, float(np.mean(rng.integers(0, 3, n) != rng.integers(0, 3, n)))


PUZZLES = [("rolls until the first six", first_six), ("flips until two heads in a row", two_heads), ("die with one optional re-roll", one_reroll),
           ("rolls to see all six faces", all_faces), ("first player wins (first head)", first_head_wins), ("gambler's ruin (3 -> 10, p = 0.45)", gamblers_ruin),
           ("birthday problem, 23 people", birthday), ("Monty Hall, switching", monty_hall)]


def kelly(p, b=1.0):
    return p - (1 - p) / b


def growth(f, p, b=1.0):
    if f >= 1:
        return -math.inf
    return p * math.log(1 + f * b) + (1 - p) * math.log(1 - f)


def simulate_wealth(f, p, b=1.0, n_bets=1000, n_paths=4000, seed=0):
    rng = np.random.default_rng(seed)
    win = rng.random((n_paths, n_bets)) < p
    return np.exp(np.where(win, math.log1p(f * b), math.log1p(-f)).sum(1))


def gm_quotes(pi, alpha, v_lo=99.0, v_hi=101.0):
    """pi = dealer's P(V = V_hi) -> (bid, ask) = (E[V | sell], E[V | buy])."""
    buy_hi, buy_lo = alpha + (1 - alpha) / 2, (1 - alpha) / 2
    pi_buy = pi * buy_hi / (pi * buy_hi + (1 - pi) * buy_lo)
    pi_sell = pi * buy_lo / (pi * buy_lo + (1 - pi) * buy_hi)
    return v_lo + (v_hi - v_lo) * pi_sell, v_lo + (v_hi - v_lo) * pi_buy


def update(pi, side, alpha):
    buy_hi, buy_lo = alpha + (1 - alpha) / 2, (1 - alpha) / 2
    if side > 0:
        return pi * buy_hi / (pi * buy_hi + (1 - pi) * buy_lo)
    return pi * buy_lo / (pi * buy_lo + (1 - pi) * buy_hi)


def market_making(alpha, half_spread=None, n_trades=50, n_games=4000, v_lo=99.0, v_hi=101.0, seed=0):
    """half_spread=None: Glosten-Milgrom quotes; otherwise quote E[V] +/- half_spread (belief still updated by Bayes).
    -> dict(pnl_per_trade, first_spread, final_error)."""
    rng = np.random.default_rng(seed)
    pnl = err = 0.0
    for _ in range(n_games):
        V = v_hi if rng.random() < 0.5 else v_lo
        pi = 0.5
        for _k in range(n_trades):
            if half_spread is None:
                bid, ask = gm_quotes(pi, alpha, v_lo, v_hi)
            else:
                m = v_lo + (v_hi - v_lo) * pi
                bid, ask = m - half_spread, m + half_spread
            if rng.random() < alpha:
                side = 1 if V > ask else (-1 if V < bid else 0)        # informed: buy if cheap, sell if rich, else no trade
            else:
                side = 1 if rng.random() < 0.5 else -1
            if side > 0:
                pnl += ask - V
            elif side < 0:
                pnl += V - bid
            if side:
                pi = update(pi, side, alpha)
        err += abs(v_lo + (v_hi - v_lo) * pi - V)
    b0, a0 = gm_quotes(0.5, alpha, v_lo, v_hi)
    return dict(pnl_per_trade=pnl / (n_games * n_trades), first_spread=a0 - b0, final_error=err / n_games)


def demos():
    rng = np.random.default_rng(42)
    print("1. Puzzles (closed form / simulation):")
    for name, f in PUZZLES:
        exact, sim = f(rng)
        print(f"   {name}: {exact:.4f} / {sim:.4f}")
    print("2. Kelly (p = 60 %, even odds): " + "; ".join(f"bet {x:.0%} -> log growth {growth(x, 0.6):+.4f}" for x in (0.05, 0.1, 0.2, 0.3, 0.4)))
    print("3. Market making (value 99 or 101; P&L per trade):")
    for alpha in (0.1, 0.3, 0.5):
        row = [f"GM quotes {market_making(alpha, None, n_games=2000)['pnl_per_trade']:+.3f} (first spread {2 * alpha:.1f})"]
        row += [f"half-spread {h} -> {market_making(alpha, h, n_games=2000)['pnl_per_trade']:+.3f}" for h in (0.05, 0.2, 0.5)]
        print(f"   alpha = {alpha}: " + "; ".join(row))


def selftest():
    check = Checker()
    rng = np.random.default_rng(0)
    for name, f in PUZZLES:
        exact, sim = f(rng)
        check(f"{name}: formula {exact:.4f}, simulation {sim:.4f}", abs(exact - sim) < 0.02 * max(1, abs(exact)))
    fs = np.linspace(0, 0.99, 991)
    best = fs[int(np.argmax([growth(x, 0.6) for x in fs]))]
    check(f"Kelly (p = 60 %, even odds): f* = 0.2 maximises growth (grid argmax {best:.2f}); max growth {growth(0.2, 0.6):.5f}",
          abs(kelly(0.6) - 0.2) < 1e-12 and abs(best - 0.2) < 0.006)
    check(f"half Kelly keeps ~3/4 of the growth ({growth(0.1, 0.6) / growth(0.2, 0.6):.2f}); double Kelly has negative growth ({growth(0.4, 0.6):.5f})",
          0.72 < growth(0.1, 0.6) / growth(0.2, 0.6) < 0.78 and -0.01 < growth(0.4, 0.6) < 0)
    w1, w2, wh = simulate_wealth(0.2, 0.6), simulate_wealth(0.4, 0.6), simulate_wealth(0.1, 0.6)
    check(f"1000 bets: median wealth Kelly {np.median(w1):,.0f}x, half Kelly {np.median(wh):,.0f}x, double Kelly {np.median(w2):.2f}x",
          np.median(w1) > np.median(wh) > 1 > np.median(w2))
    b, a = gm_quotes(0.5, 0.3)
    check(f"Glosten-Milgrom: first spread = alpha x (V_hi - V_lo) = 0.6 ({a - b:.6f}), centred on 100", abs((a - b) - 0.6) < 1e-12 and abs((a + b) / 2 - 100) < 1e-12)
    g1 = market_making(0.3, None, n_trades=1, n_games=40000, seed=1)
    n1 = market_making(0.3, 0.1, n_trades=1, n_games=40000, seed=1)
    w1_ = market_making(0.3, 0.6, n_trades=1, n_games=40000, seed=1)
    check(f"first trade (alpha = 0.3): GM breaks even ({g1['pnl_per_trade']:+.4f}); half-spread 0.1 loses ({n1['pnl_per_trade']:+.4f}, theory -0.20); "
          f"half-spread 0.6 earns ({w1_['pnl_per_trade']:+.4f}, theory +0.30)",
          abs(g1["pnl_per_trade"]) < 0.02 and abs(n1["pnl_per_trade"] + 0.20) < 0.02 and abs(w1_["pnl_per_trade"] - 0.30) < 0.02)
    gm = market_making(0.3, None, n_games=3000, seed=1)
    check(f"GM over 50 trades still breaks even ({gm['pnl_per_trade']:+.4f}); prices end {gm['final_error']:.3f} from the true value (started 1.0)",
          abs(gm["pnl_per_trade"]) < 0.01 and gm["final_error"] < 0.2)
    return check.summary()


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    demos()
