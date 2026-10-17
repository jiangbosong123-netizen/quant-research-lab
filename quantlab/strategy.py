"""The lab's strategy, as pure functions: the mispricing composite (signal) and the portfolio rules (P1 and its variants).

This module restates the rules of the private pipeline so that they can be read, run and tested without any data.
It is deliberately small: inputs are plain numbers or arrays that the caller has already aligned point-in-time
(see docs/06_strategy.md for the alignment rules); nothing here downloads or stores market data.

Signal (Stambaugh, Yu & Yuan 2012, eight of their eleven anomalies that can be built from free point-in-time data):
    net share issuance (-), accruals (-), net operating assets (-), momentum (+), gross profitability (+),
    asset growth (-), return on assets (+), investment-to-assets (-).
    Each is ranked within the month's universe (percentile = (average rank - 1/2) / n), flipped when the direction is "-",
    and the composite is the equal-weighted mean of the available percentiles; fewer than 5 available -> no score.
Portfolios (monthly, traded at the next trading day's close):
    P1  top 50 long / bottom 50 short, equal weight (dollar neutral, 200 % gross)
    P2a P1 + SPY hedge: h = -(beta_long - beta_short), betas from 252 days of daily returns
    P3a P1 with a buffer band: keep last month's names while they stay within the top (bottom) 100
    P3b P1 scaled by s = min(1.5, target_vol / estimated_vol), vol from 126 days of the long-short's daily returns
Costs: each side, new names / 50 = turnover tau; monthly cost = 0.4 % x (tau_long + tau_short) + 1 %/yr borrow / 12.
"""
import math

import numpy as np

from ._util import Checker

DIRECTIONS = {"net_share_issuance": -1, "accruals": -1, "net_operating_assets": -1, "momentum": +1,
              "gross_profitability": +1, "asset_growth": -1, "roa": +1, "investment_to_assets": -1}
MIN_COMPONENTS = 5
PER_SIDE = 50
ROUND_TRIP = 0.004
BORROW_PER_YEAR = 0.01


# ── the eight components (inputs already aligned to what was public by the signal date) ─────────────────────────────

def net_share_issuance(shares_now, shares_year_ago, split_factor=1.0, days_apart=365):
    """ln(shares_now / (shares_year_ago x split factor)); the two share counts must be 270-460 days apart, else missing."""
    if None in (shares_now, shares_year_ago, split_factor) or not 270 <= days_apart <= 460 or shares_now <= 0 or shares_year_ago <= 0:
        return None
    return math.log(shares_now / (shares_year_ago * split_factor))


def accruals(net_income, cash_from_operations, assets, assets_prev):
    """(net income - operating cash flow) / average total assets of the two fiscal year-ends."""
    if None in (net_income, cash_from_operations, assets, assets_prev) or assets + assets_prev <= 0:
        return None
    return (net_income - cash_from_operations) / ((assets + assets_prev) / 2)


def net_operating_assets(debt, common_equity, minority_interest, preferred, cash, short_term_investments, assets_prev):
    """(debt + common equity + minority interest + preferred - cash - short-term investments) / prior total assets.
    Missing debt, minority interest, preferred or short-term investments count as 0 (the private pipeline does the same)."""
    if None in (common_equity, cash, assets_prev) or assets_prev <= 0:
        return None
    z = lambda x: 0.0 if x is None else x
    return (z(debt) + common_equity + z(minority_interest) + z(preferred) - cash - z(short_term_investments)) / assets_prev


def momentum(price_month_end_t7, price_month_end_t1):
    """Return from the month-end seven months ago to last month's end (the most recent month is skipped)."""
    if None in (price_month_end_t7, price_month_end_t1) or price_month_end_t7 <= 0:
        return None
    return price_month_end_t1 / price_month_end_t7 - 1


def gross_profitability(assets, gross_profit=None, revenue=None, cost_of_revenue=None):
    """Gross profit / total assets; gross profit as reported, else revenue - cost of revenue."""
    if assets is None or assets <= 0:
        return None
    if gross_profit is None:
        if None in (revenue, cost_of_revenue):
            return None
        gross_profit = revenue - cost_of_revenue
    return gross_profit / assets


def asset_growth(assets, assets_prev):
    if None in (assets, assets_prev) or assets_prev <= 0:
        return None
    return assets / assets_prev - 1


def roa(quarterly_net_income, assets_prev_quarter):
    """Latest quarter's net income / total assets one quarter earlier."""
    if None in (quarterly_net_income, assets_prev_quarter) or assets_prev_quarter <= 0:
        return None
    return quarterly_net_income / assets_prev_quarter


def investment_to_assets(ppe, ppe_prev, inventory, inventory_prev, assets_prev):
    """(change in gross PP&E + change in inventory) / prior total assets; inventory missing in both years counts as 0."""
    if None in (ppe, ppe_prev, assets_prev) or assets_prev <= 0:
        return None
    if inventory is None and inventory_prev is None:
        inventory = inventory_prev = 0.0
    if None in (inventory, inventory_prev):
        return None
    return (ppe - ppe_prev + inventory - inventory_prev) / assets_prev


# ── composite ───────────────────────────────────────────────────────────────────────────────────────────────────

def percentiles(values):
    """{id: value or None} -> {id: (average rank - 1/2) / n} over the ids that have a value (ties share the mean rank)."""
    ids = [k for k, v in values.items() if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not ids:
        return {}
    x = np.array([values[k] for k in ids], float)
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    r[order] = np.arange(1, len(x) + 1)
    for u in np.unique(x):
        m = x == u
        if m.sum() > 1:
            r[m] = r[m].mean()
    return {k: (float(v) - 0.5) / len(x) for k, v in zip(ids, r)}


def composite(components, directions=DIRECTIONS, min_components=MIN_COMPONENTS):
    """components: {name: {id: value or None}} for one month -> {id: score in (0, 1)}; higher = more underpriced."""
    pct = {}
    for name, vals in components.items():
        p = percentiles(vals)
        pct[name] = {k: (v if directions[name] > 0 else 1.0 - v) for k, v in p.items()}
    ids = set().union(*[set(v) for v in pct.values()]) if pct else set()
    out = {}
    for k in ids:
        xs = [pct[n][k] for n in pct if k in pct[n]]
        if len(xs) >= min_components:
            out[k] = sum(xs) / len(xs)
    return out


# ── portfolios ──────────────────────────────────────────────────────────────────────────────────────────────────

def select(scores, n=PER_SIDE):
    """P1: sort by (score, id as text); the last n go long, the first n short."""
    order = sorted(scores, key=lambda k: (scores[k], str(k)))
    return order[-n:], order[:n]


def turnover(new, old):
    """Share of this side's names that are new this month (len(new - old) / len(new))."""
    new, old = set(map(str, new)), set(map(str, old))
    return len(new - old) / len(new) if new else 0.0


def monthly_cost(tau_long, tau_short, round_trip=ROUND_TRIP, borrow=BORROW_PER_YEAR, scale=1.0):
    """Cost as a fraction of one side's capital: round trip x scale x (tau_long + tau_short) + scale x borrow / 12."""
    return round_trip * scale * (tau_long + tau_short) + scale * borrow / 12


def ols_beta(y, x):
    y, x = np.asarray(y, float), np.asarray(x, float)
    ok = np.isfinite(y) & np.isfinite(x)
    y, x = y[ok], x[ok]
    xc = x - x.mean()
    return float(np.sum(xc * (y - y.mean())) / np.sum(xc * xc))


def hedge_ratio(long_daily, short_daily, index_daily, min_days=126):
    """P2a: h = -(beta_long - beta_short) on the index (buy the index when h > 0). Fewer than min_days valid days -> betas = 1."""
    ok = np.isfinite(long_daily) & np.isfinite(short_daily) & np.isfinite(index_daily)
    if ok.sum() < min_days:
        return 0.0
    return -(ols_beta(long_daily[ok], index_daily[ok]) - ols_beta(short_daily[ok], index_daily[ok]))


def buffer_select(scores, previous, n=PER_SIDE, keep_within=2 * PER_SIDE, side="long"):
    """P3a: keep last month's names that still rank within keep_within on this side, then fill to n from the best rank."""
    order = sorted(scores, key=lambda k: (scores[k], str(k)), reverse=(side == "long"))
    rank = {k: i + 1 for i, k in enumerate(order)}
    kept = [k for k in previous if k in rank and rank[k] <= keep_within]
    kept = sorted(kept, key=lambda k: rank[k])[:n]
    out = list(kept)
    for k in order:
        if len(out) >= n:
            break
        if k not in out:
            out.append(k)
    return out


def vol_scale(ls_daily, target_vol, cap=1.5, min_days=100):
    """P3b: s = min(cap, target_vol / annualised vol of the long-short's daily returns); too few days -> 1."""
    x = np.asarray(ls_daily, float)
    x = x[np.isfinite(x)]
    if len(x) < min_days:
        return 1.0
    sigma = float(x.std(ddof=1) * math.sqrt(252))
    return min(cap, target_vol / sigma) if sigma > 0 else 1.0


# ── self-test ───────────────────────────────────────────────────────────────────────────────────────────────────

def selftest():
    check = Checker()
    check("net share issuance: ln(110 / (50 x 2)) with a 2-for-1 split; dates too far apart -> missing",
          abs(net_share_issuance(110, 50, 2.0, 365) - math.log(1.1)) < 1e-12 and net_share_issuance(110, 50, 2.0, 500) is None)
    check("accruals: (90 - 120) / mean(1000, 800) = -1/30", abs(accruals(90, 120, 1000, 800) + 30 / 900) < 1e-12)
    check("net operating assets: missing debt / minority / preferred / short-term investments count as 0",
          abs(net_operating_assets(None, 500, None, None, 100, None, 1000) - 0.4) < 1e-12 and net_operating_assets(1, None, 0, 0, 0, 0, 1) is None)
    check("gross profitability: reported gross profit first, else revenue - cost", gross_profitability(1000, gross_profit=300) == 0.3
          and abs(gross_profitability(1000, revenue=900, cost_of_revenue=650) - 0.25) < 1e-12)
    check("investment-to-assets: inventory missing in both years -> 0; in one year only -> missing",
          abs(investment_to_assets(600, 500, None, None, 1000) - 0.1) < 1e-12 and investment_to_assets(600, 500, 10, None, 1000) is None)
    check("momentum skips the latest month: 120 / 100 - 1", abs(momentum(100, 120) - 0.2) < 1e-12)
    rng = np.random.default_rng(0)
    ids = [f"s{i:03d}" for i in range(300)]
    comps = {name: {k: float(rng.normal()) for k in ids} for name in DIRECTIONS}
    for k in ids[:20]:                                     # 20 names have only 4 components -> no score
        for name in list(DIRECTIONS)[:4]:
            comps[name][k] = None
    sc = composite(comps)
    check("composite: names with fewer than 5 components get no score; scores lie in (0, 1)",
          len(sc) == 280 and all(k not in sc for k in ids[:20]) and all(0 < v < 1 for v in sc.values()))
    one = {"momentum": {"a": 3.0, "b": 1.0, "c": 2.0}, "accruals": {"a": 3.0, "b": 1.0, "c": 2.0}}
    s1 = composite(one, min_components=2)
    check("composite: a '-' component is flipped, so equal ranks on a '+' and a '-' component cancel to 1/2", all(abs(v - 0.5) < 1e-12 for v in s1.values()))
    longs, shorts = select(sc)
    check("P1 selection: 50 per side, longs have the highest scores, deterministic", len(longs) == len(shorts) == 50
          and min(sc[k] for k in longs) > max(sc[k] for k in shorts) and select(sc) == (longs, shorts))
    check("costs: 0.4 % x (0.25 + 0.25) + 1 %/12 = 0.2833 % per month (the backtest's average was 0.283 %)",
          abs(monthly_cost(0.25, 0.25) - (0.002 + 0.01 / 12)) < 1e-15)
    n = 252
    idx = rng.normal(0, 0.01, n)
    L = 0.7 * idx + rng.normal(0, 0.005, n)
    S = 2.2 * idx + rng.normal(0, 0.005, n)
    h = hedge_ratio(L, S, idx)
    check(f"P2a hedge: long beta 0.7, short beta 2.2 -> buy about 1.5 of the index (h = {h:.2f}); the hedged book has beta ~0",
          abs(h - 1.5) < 0.1 and abs(ols_beta(L - S + h * idx, idx)) < 1e-9)
    prev = longs[:]
    sc2 = {k: v + float(rng.normal(0, 0.02)) for k, v in sc.items()}
    new_p1 = select(sc2)[0]
    new_p3a = buffer_select(sc2, prev)
    check(f"P3a buffer: 50 names, never worse than rank 100, and fewer new names than P1 ({turnover(new_p3a, prev):.0%} vs {turnover(new_p1, prev):.0%})",
          len(new_p3a) == 50 and turnover(new_p3a, prev) <= turnover(new_p1, prev))
    ls = rng.normal(0, 0.02, 126)
    s = vol_scale(ls, 0.1448)
    check(f"P3b scaling: s = target / estimated vol, capped at 1.5 (s = {s:.2f}); too few days -> 1",
          abs(s - min(1.5, 0.1448 / (ls.std(ddof=1) * math.sqrt(252)))) < 1e-12 and vol_scale(ls[:50], 0.1448) == 1.0
          and vol_scale(ls * 0.01, 0.1448) == 1.5)
    return check.summary()


if __name__ == "__main__":
    import sys
    sys.exit(0 if selftest() else 1)
