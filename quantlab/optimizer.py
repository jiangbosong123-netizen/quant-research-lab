"""Mean-variance portfolio optimisation with constraints and transaction costs, solved by ADMM in plain NumPy.

    maximise   alpha'w - (lam/2) w'Sigma w - sum_i kappa_i |w_i - w0_i|
    subject to A w = b             (equalities: dollar neutral 1'w = 0, factor neutral X'w = 0, ...)
               |w_i| <= cap_i      (position limits)

alpha: expected returns; Sigma: risk-model covariance; lam: risk aversion; kappa: linear cost per unit traded; w0: current holdings.

ADMM (Boyd et al. 2011) splits w = z:
  * the w-step handles the quadratic term and the equalities: one KKT linear system (the matrix is inverted once per rho);
  * the z-step handles costs and limits: separable per asset, closed form (soft-threshold around w0, then clip to the limits);
  * dual update; rho is adapted by residual balancing; at the end the solution is "polished" (OSQP-style): with the active set read off the
    approximate solution, the free variables are solved exactly.
Independent checks in the self-test: the closed-form solution (equalities only) and brute-force active-set enumeration (limits, and limits + costs).
"""
import itertools

import numpy as np

from ._util import Checker, mm


def objective(w, alpha, S, lam, kappa=None, w0=None):
    v = float(np.sum(alpha * w)) - 0.5 * lam * float(np.sum(w * mm(S, w)))
    if kappa is not None:
        v -= float(np.sum(kappa * np.abs(w - (w0 if w0 is not None else 0.0))))
    return v


def drop_redundant(A, b, tol=1e-10):
    """Remove linearly dependent equality rows (e.g. market column = sum of industry dummies), which make the KKT matrix singular.
    SVD A = U s V'; keep directions with s > tol * s_max: V_k' w = s_k^{-1} U_k' b. The dropped directions must have zero right-hand side."""
    if A.shape[0] == 0:
        return A, b
    U, sv, Vt = np.linalg.svd(A, full_matrices=False)
    k = int(np.sum(sv > tol * sv[0]))
    dropped = U[:, k:].T @ b
    if np.any(np.abs(dropped) > 1e-9 * max(1.0, float(np.abs(b).max(initial=0.0)))):
        raise ValueError("inconsistent equality constraints")
    return Vt[:k], (U[:, :k].T @ b) / sv[:k]


def closed_form(alpha, S, lam, A=None, b=None):
    """Equalities only: [lam*Sigma A'; A 0][w; nu] = [alpha; b]. Without constraints: w = Sigma^{-1} alpha / lam."""
    N = len(alpha)
    if A is None:
        return np.linalg.solve(lam * S, alpha)
    A, b = drop_redundant(np.atleast_2d(A), np.atleast_1d(b).astype(float))
    m = A.shape[0]
    K = np.zeros((N + m, N + m))
    K[:N, :N] = lam * S
    K[:N, N:] = A.T
    K[N:, :N] = A
    return np.linalg.solve(K, np.r_[alpha, b])[:N]


def optimize(alpha, S, lam, A=None, b=None, cap=None, kappa=None, w0=None, rho=None, max_iter=20000, tol=1e-9):
    """ADMM -> dict(w, iterations, primal_res, dual_res, eq_res, converged, rho_changes, polished)."""
    N = len(alpha)
    A = np.zeros((0, N)) if A is None else np.atleast_2d(A)
    b = np.zeros(A.shape[0]) if b is None else np.atleast_1d(b).astype(float)
    A0, b0 = A, b
    A, b = drop_redundant(A, b)
    m = A.shape[0]
    cap = np.full(N, np.inf) if cap is None else np.broadcast_to(np.asarray(cap, float), (N,))
    kappa = np.zeros(N) if kappa is None else np.broadcast_to(np.asarray(kappa, float), (N,))
    w0 = np.zeros(N) if w0 is None else np.asarray(w0, float)
    rho = rho or float(lam * np.mean(np.diag(S)))

    def kkt_inverse(rho):
        K = np.zeros((N + m, N + m))
        K[:N, :N] = lam * S + rho * np.eye(N)
        K[:N, N:] = A.T
        K[N:, :N] = A
        return np.linalg.inv(K)
    Kinv = kkt_inverse(rho)
    z = np.clip(w0, -cap, cap)
    y = np.zeros(N)                                                   # scaled dual variable
    changes = 0
    for it in range(1, max_iter + 1):
        w = mm(Kinv, np.r_[alpha + rho * (z - y), b])[:N]
        d = w + y - w0
        z_old = z
        z = np.clip(w0 + np.sign(d) * np.maximum(np.abs(d) - kappa / rho, 0.0), -cap, cap)
        y = y + w - z
        r_pri, r_dual = float(np.linalg.norm(w - z)), float(rho * np.linalg.norm(z - z_old))
        if r_pri < tol * np.sqrt(N) and r_dual < tol * np.sqrt(N):
            break
        if it % 25 == 0 and (r_pri > 10 * r_dual or r_dual > 10 * r_pri):       # residual balancing (Boyd et al. 2011, s. 3.4.1)
            new = rho * 2 if r_pri > 10 * r_dual else rho / 2
            y = y * rho / new
            rho = new
            Kinv = kkt_inverse(rho)
            changes += 1
    converged = it < max_iter
    pol = polish(z, alpha, S, lam, A0, b0, cap, kappa, w0)
    if pol is not None and objective(pol, alpha, S, lam, kappa, w0) >= objective(z, alpha, S, lam, kappa, w0) - 1e-12:
        z, converged = pol, True
    eq = float(np.max(np.abs(mm(A0, z) - b0))) if A0.shape[0] else 0.0
    return dict(w=z, iterations=it, primal_res=r_pri, dual_res=r_dual, eq_res=eq, converged=converged, rho_changes=changes, polished=pol is not None)


def polish(z, alpha, S, lam, A, b, cap, kappa, w0, tol=1e-7):
    """Read each asset's state off the ADMM solution (at a limit, parked at w0, above or below w0), fix the bound/parked ones and solve the rest
    exactly (linear term alpha -/+ kappa according to the side of w0). Returns w if it is feasible (limits, regions, equalities), else None.
    The caller keeps it only if its objective is not worse. (KKT signs are not checked: if the states are right this is the exact optimum,
    otherwise still a feasible point no worse than ADMM's.)"""
    at_hi = np.isfinite(cap) & (np.abs(z - cap) < tol)
    at_lo = np.isfinite(cap) & (np.abs(z + cap) < tol)
    parked = (kappa > 0) & (np.abs(z - w0) < tol) & ~at_hi & ~at_lo
    fixed = at_hi | at_lo | parked
    free = ~fixed
    wf = np.where(at_hi, cap, np.where(at_lo, -cap, np.where(parked, w0, 0.0)))
    above = free & (z > w0)
    lin = alpha - np.where(above, kappa, 0.0) + np.where(free & ~above, kappa, 0.0)
    if free.any():
        try:
            Af = A[:, free] if A.shape[0] else None
            bf = (b - A[:, fixed] @ wf[fixed]) if A.shape[0] else None
            if Af is not None and np.linalg.matrix_rank(Af) == 0:
                Af = bf = None
            x = closed_form(lin[free] - lam * S[np.ix_(free, fixed)] @ wf[fixed], S[np.ix_(free, free)], lam, Af, bf)
        except (np.linalg.LinAlgError, ValueError):
            return None
        wf = wf.copy()
        wf[free] = x
    if not (np.all(np.abs(wf) <= cap + 1e-12) and np.all(wf[above] >= w0[above] - 1e-12) and np.all(wf[free & ~above] <= w0[free & ~above] + 1e-12)):
        return None
    if A.shape[0] and np.max(np.abs(A @ wf - b)) > 1e-9:
        return None
    return wf


def brute_force(alpha, S, lam, A, b, cap, kappa=0.0, w0=None):
    """Exact solution for tiny problems (N <= 6) by enumerating states: at the lower / upper limit / interior (kappa = 0); with costs the interior
    splits into parked at w0 / above w0 (cost -kappa(w - w0)) / below w0. For each combination, fix the fixed ones, solve the rest under the equalities,
    keep the feasible ones and return the best. The optimum is always one of them."""
    N = len(alpha)
    w0 = np.zeros(N) if w0 is None else np.asarray(w0, float)
    states = ("lo", "hi", "mid") if kappa == 0 else ("lo", "hi", "park", "above", "below")
    best, bw = -np.inf, None
    for st in itertools.product(states, repeat=N):
        st = np.array(st)
        wf = np.where(st == "lo", -cap, np.where(st == "hi", cap, np.where(st == "park", w0, 0.0)))
        fixed = np.isin(st, ("lo", "hi", "park"))
        free = ~fixed
        if free.any():
            lin = alpha + np.where(st == "above", -kappa, np.where(st == "below", kappa, 0.0))
            af = lin[free] - lam * S[np.ix_(free, fixed)] @ wf[fixed]
            try:
                x = closed_form(af, S[np.ix_(free, free)], lam, A[:, free], b - A[:, fixed] @ wf[fixed])
            except np.linalg.LinAlgError:
                continue
            wf = wf.copy()
            wf[free] = x
        if not (np.all(np.abs(wf) <= cap + 1e-12) and np.allclose(A @ wf, b, atol=1e-10)
                and np.all(wf[st == "above"] >= w0[st == "above"] - 1e-12) and np.all(wf[st == "below"] <= w0[st == "below"] + 1e-12)):
            continue
        v = objective(wf, alpha, S, lam, kappa if kappa else None, w0)
        if v > best:
            best, bw = v, wf
    return bw


def expected_returns(score, specific_vol, ic):
    """Grinold: alpha = IC x volatility x standardised score."""
    z = (score - score.mean()) / score.std()
    return ic * specific_vol * z


def selftest():
    np.seterr(divide="ignore", over="ignore", invalid="ignore")
    check = Checker()
    rng = np.random.default_rng(7)
    N = 40
    B = rng.standard_normal((N, 3)) * 0.1
    S = B @ B.T + np.diag(rng.uniform(0.01, 0.04, N))
    alpha = rng.standard_normal(N) * 0.01
    lam = 5.0
    A = np.vstack([np.ones(N), B.T])
    b = np.zeros(4)
    w_cf = closed_form(alpha, S, lam, A, b)
    r = optimize(alpha, S, lam, A, b)
    check(f"equalities only (dollar + 3 factor neutral): ADMM = closed form (max diff {np.max(np.abs(r['w'] - w_cf)):.1e}, {r['iterations']} iterations)",
          r["converged"] and np.max(np.abs(r["w"] - w_cf)) < 1e-6 and r["eq_res"] < 1e-6)
    check("no constraints: ADMM = Sigma^{-1} alpha / lam", np.allclose(optimize(alpha, S, lam)["w"], np.linalg.solve(lam * S, alpha), atol=1e-6))
    A_dup = np.vstack([A, A[1] + 2 * A[2]])
    r_dup = optimize(alpha, S, lam, A_dup, np.zeros(5))
    try:
        drop_redundant(A_dup, np.r_[np.zeros(4), 1.0])
        inconsistent = False
    except ValueError:
        inconsistent = True
    check("linearly dependent constraint: same solution after dropping it; an inconsistent right-hand side raises",
          np.max(np.abs(r_dup["w"] - w_cf)) < 1e-6 and np.max(np.abs(closed_form(alpha, S, lam, A_dup, np.zeros(5)) - w_cf)) < 1e-9 and inconsistent)
    n = 6
    Bs = rng.standard_normal((n, 2)) * 0.1
    Ss = Bs @ Bs.T + np.diag(rng.uniform(0.01, 0.04, n))
    al = rng.standard_normal(n) * 0.02
    As, bs = np.ones((1, n)), np.zeros(1)
    capn = np.full(n, 0.15)
    wb = brute_force(al, Ss, 2.0, As, bs, capn)
    rb = optimize(al, Ss, 2.0, As, bs, cap=capn)
    n_at = int(np.sum(np.abs(np.abs(wb) - 0.15) < 1e-9))
    check(f"position limits (N = 6, all 3^6 active sets): ADMM = brute force (max diff {np.max(np.abs(rb['w'] - wb)):.1e}; {n_at} at a limit)",
          np.max(np.abs(rb["w"] - wb)) < 1e-6 and abs(objective(rb["w"], al, Ss, 2.0) - objective(wb, al, Ss, 2.0)) < 1e-10 and n_at >= 1)
    n5 = 5
    B5 = rng.standard_normal((n5, 2)) * 0.1
    S5 = B5 @ B5.T + np.diag(rng.uniform(0.01, 0.04, n5))
    a5 = rng.standard_normal(n5) * 0.02
    w05 = rng.uniform(-0.08, 0.08, n5)
    w05 -= w05.mean()
    cap5 = np.full(n5, 0.12)
    for k5 in (0.004, 0.02):
        wb5 = brute_force(a5, S5, 2.0, np.ones((1, n5)), np.zeros(1), cap5, kappa=k5, w0=w05)
        rb5 = optimize(a5, S5, 2.0, np.ones((1, n5)), np.zeros(1), cap=cap5, kappa=k5, w0=w05)
        parked = int(np.sum(np.abs(wb5 - w05) < 1e-9))
        check(f"transaction costs kappa = {k5} (N = 5, all 5^5 states): ADMM = brute force (max diff {np.max(np.abs(rb5['w'] - wb5)):.1e}; {parked} parked at w0)",
              np.max(np.abs(rb5["w"] - wb5)) < 1e-6)
    cap2 = np.full(N, 0.02)
    r2 = optimize(alpha, S, lam, A, b, cap=cap2)
    check("N = 40 with limits: within limits, equalities to 1e-6, objective below the equality-only optimum",
          np.all(np.abs(r2["w"]) <= 0.02 + 1e-12) and r2["eq_res"] < 1e-6 and objective(r2["w"], alpha, S, lam) <= objective(w_cf, alpha, S, lam) + 1e-12)
    r1 = optimize(alpha, S, lam, np.ones((1, N)), np.zeros(1), cap=cap2)
    interior = np.abs(r1["w"]) < 0.012
    e = np.where(interior, rng.uniform(-0.004, 0.004, N), 0.0)
    e[interior] -= e[interior].mean()
    w0 = r1["w"] + e
    turnover = []
    for k in (0.0, 0.0005, 0.002, 0.01, 1.0):
        rr = optimize(alpha, S, lam, np.ones((1, N)), np.zeros(1), cap=cap2, kappa=k, w0=w0)
        turnover.append(float(np.abs(rr["w"] - w0).sum()))
    check(f"costs: higher kappa keeps the portfolio closer to w0 (turnover {' -> '.join(f'{x:.4f}' for x in turnover)}); a large kappa means no trade",
          abs(w0.sum()) < 1e-6 and np.all(np.abs(w0) <= 0.02) and all(a >= c - 1e-9 for a, c in zip(turnover, turnover[1:])) and turnover[-1] < 1e-6
          and turnover[0] > 0.01)
    rr = optimize(alpha, S, lam, np.ones((1, N)), np.zeros(1), cap=cap2, kappa=0.0005, w0=w0)
    inner = np.abs(rr["w"]) < 0.02 - 2e-3
    E = np.where(inner, rng.uniform(-1e-3, 1e-3, (300, N)), 0.0)
    E[:, inner] -= E[:, inner].mean(1, keepdims=True)
    feasible = [e for e in E if abs(e.sum()) < 1e-12 and np.all(np.abs(rr["w"] + e) <= 0.02)]
    base = objective(rr["w"], alpha, S, lam, 0.0005, w0)
    check(f"with costs: none of {len(feasible)} nearby feasible random points ({int(inner.sum())} free assets) has a higher objective",
          all(base >= objective(rr["w"] + e, alpha, S, lam, 0.0005, w0) - 1e-12 for e in feasible) and len(feasible) == 300 and inner.sum() >= 5)
    a = expected_returns(rng.standard_normal(500), np.full(500, 0.3), 0.05)
    check("alpha = IC x vol x score: mean 0, std = IC x vol", abs(a.mean()) < 1e-15 and abs(a.std() - 0.05 * 0.3) < 1e-12)
    return check.summary()


if __name__ == "__main__":
    import sys
    sys.exit(0 if selftest() else 1)
