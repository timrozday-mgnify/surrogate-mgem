"""M12 / §13.4 — chemostat steady state, coexistence, stability and invasion.

Newton-solve ``dc/dt = 0, dX/dt = 0`` instead of integrating. At dilution ``D``

    D (c_feed - c) + sum_i X_i z_i(c) = 0        (M equations)
    X_i (mu_i(c) - D) = 0                        (G equations)

so every surviving member sits at ``mu_i(c*) = D`` exactly — the classical
chemostat statement — and the rest are washed out. The second family is a
complementarity condition, not an equation, and it is handled the way it always
is: solve the square system over an assumed survivor set, drop anyone whose ``X``
goes negative, re-admit anyone whose ``mu(c*)`` exceeds ``D``, repeat.

Why bother, when :func:`cfs.compose.dfba.simulate` already integrates a chemostat:
the equilibrium is where the numbers are worth quoting (a batch endpoint is P23),
its sensitivities are one linear solve rather than a badly-conditioned backprop
through 200 Euler steps, and stability, invasion and coexistence all fall out of
the same Jacobian.

**Two things §7 and §8.4 already measured, and both are respected here.** The
Hessian sum is rank ~10-25 of 365, so it is the supply term ``-D I`` that makes
the solve well-posed; and ``x_scale`` spans five decades across the index, so the
Newton system is **diagonally preconditioned** (row and column equilibration)
before it is solved, whatever the head is. The linear solve is `lstsq`, not
`solve`: rank deficiency is expected, not exceptional.

**And one §13.7 caveat that governs how to read the output.** A steady state *is*
a drawn-down medium, which is the regime where Head B is worst (§8.6g). The gate
numbers M5 quotes are at a 4-doubling batch horizon and must not be assumed to
transfer. `depth` and `reach` are reported at ``c*`` for exactly that reason, and
`--fallback-depth`'s per-state LP is the way to buy accuracy here if they are bad
— an equilibrium visits one state, so truth is cheap at it.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

LOGGER = logging.getLogger("cfs.science.steady")


def solve_steady(
    rhs,
    feed: np.ndarray,
    dilution: float,
    free: np.ndarray,
    scale: np.ndarray,
    c0: np.ndarray,
    x0: np.ndarray,
    *,
    rhs_jac=None,
    dmu_dc=None,
    tol: float = 1e-6,
    iters: int = 100,
    backtracks: int = 40,
) -> dict:
    """Newton + active set on the chemostat fixed point.

    ``rhs(c, X) -> (dc, mu)`` is any batch right-hand side over ``G`` members —
    :func:`cfs.compose.dfba.rhs_surrogate` in production, a Monod toy in the test.

    ``rhs_jac`` is the right-hand side the **Jacobian** is finite-differenced
    through, defaulting to ``rhs``. Passing the surrogate here and
    :func:`cfs.compose.dfba.rhs_truth` as ``rhs`` is an *inexact* Newton and it is
    the point of the whole split: the residual costs ``G`` LP solves per
    evaluation, the Jacobian costs ``n_free`` of them and must never pay that, and
    a converged root of the true residual is the **true** fixed point however
    approximate the Jacobian was. An inexact Jacobian costs convergence *rate*,
    not the answer.
    ``free`` selects the metabolites actually solved for: a metabolite no member
    exchanges has ``z = 0``, so ``c = c_feed`` solves its row exactly and carrying
    it would only add a rank-deficient direction. ``scale`` is the per-metabolite
    column scale (use ``Km``, the concentration scale the heads resolve in).
    """
    D = float(dilution)
    rhs_jac = rhs_jac or rhs
    G, idx = len(x0), np.flatnonzero(free)
    c, X = c0.astype(np.float64).copy(), x0.astype(np.float64).copy()
    # Competitive exclusion means the "everyone survives" system is usually
    # inconsistent -- one substrate cannot hold two `mu_i(c) = D` at once -- so the
    # active set is seeded from the warm start, which has already washed the losers
    # down, rather than from the full roster.
    alive = X > 1e-6 * X.max()
    X[~alive] = 0.0
    # Anti-cycling: a member dropped in this solve is never re-admitted. Without
    # it two members whose `mu` differ by 1e-4 at `c*` trade places forever --
    # dropped for a negative X, re-admitted for `mu > D` at the next iterate --
    # and the loop exits on its pass budget with whichever state it happened to be
    # in. It is the same rule Bland's does for the simplex, for the same reason.
    banned = np.zeros_like(alive)

    def _res(f):
        def residual(c, X):
            dc, mu = f(c, X)
            return np.concatenate([(D * (feed - c) + dc)[idx], (mu - D)[alive]]), mu

        return residual

    residual, residual_jac = _res(rhs), _res(rhs_jac)

    # The two families are in different units -- mmol/L/h and 1/h -- so the
    # convergence test is on the residual divided by its own row scale, and `tol`
    # is dimensionless. The residual itself stays unscaled, because `J` is also
    # the stability Jacobian and row scaling is not a similarity transform.
    #
    # `tol = 1e-6` is not conservatism: the heads run in float32 and the Jacobian
    # is finite-differenced through them, so the residual floors out around 5e-8
    # relative and a tighter tolerance only spends Newton iterations to report
    # `converged=False` about a state that is converged. §8.4's `rtol=1e-10` is a
    # statement about the root finder, not about what a float32 head can deliver.
    info = {"newton_iters": 0, "active_set_passes": 0, "converged": False}
    for _ in range(G + 2):
        info["active_set_passes"] += 1
        rscale = np.concatenate([D * np.maximum(scale[idx], 1e-30), np.full(int(alive.sum()), D)])
        c, X, J, ok, n_it = _newton(
            residual,
            residual_jac,
            rhs_jac,
            idx,
            scale,
            rscale,
            c,
            X,
            alive,
            tol,
            iters,
            backtracks,
            dmu_dc,
        )
        info["newton_iters"] += n_it
        if not ok and alive.sum() > 1:
            # An inconsistent survivor set does not have to produce a negative X;
            # it just fails to converge. Drop the least abundant member and retry.
            k = np.flatnonzero(alive)[int(np.argmin(X[alive]))]
            alive[k], banned[k], X[k] = False, True, 0.0
            continue
        if (drop := alive & (X <= 1e-9 * max(X[alive].max(), 1e-300))).any():
            alive, banned, X[drop] = alive & ~drop, banned | drop, 0.0
            continue
        _, mu = rhs(c, X)
        # A washed-out member is only consistent if it cannot grow at c*.
        if (back := (~alive) & ~banned & (mu > D * (1 + 1e-8))).any():
            alive = alive | back
            X[back] = 1e-9 * max(X[alive].max(), 1.0)
            continue
        info["converged"] = bool(ok)
        break
    LOGGER.info(
        "steady: %d survivors, %d Newton iterations, converged=%s",
        int(alive.sum()),
        info["newton_iters"],
        info["converged"],
    )

    _, mu = rhs(c, X)
    # Rebuild the Jacobian at the state actually returned: the loop's last one
    # belongs to whichever active set the last Newton *ran* on, and re-admitting a
    # member changes the shape without re-solving.
    J = _jacobian(residual_jac, rhs_jac, idx, scale, c, X, alive, dmu_dc)
    return {"c": c, "X": X, "alive": alive, "mu": mu, "J": J, **info}


def _newton(
    residual, residual_jac, rhs_jac, idx, scale, rscale, c, X, alive, tol, iters, bt, dmu_dc=None
):
    """Damped Newton on the square system over ``idx`` and the live members."""
    J = None
    for it in range(iters):
        r, _ = residual(c, X)
        rn = float((np.abs(r) / rscale).max())
        if rn < tol:
            return c, X, J, True, it
        J = _jacobian(residual_jac, rhs_jac, idx, scale, c, X, alive, dmu_dc)
        step = _lstsq_step(J, -r, scale[idx], np.maximum(np.abs(X[alive]), 1e-12))
        dc = np.zeros_like(c)
        dc[idx] = step[: len(idx)]
        dX = step[len(idx) :]
        # Fraction to the boundary, as an interior-point method does: never let a
        # step take an abundance through zero. Without it one Newton overshoot
        # early on reads as washout, the member is dropped, and -- with the
        # anti-cycling ban -- can never come back; the solve then converges to the
        # trivial `X = 0, c = c_feed` state, which is a fixed point and the wrong
        # one. A member that really is washing out reaches zero geometrically
        # instead, and is dropped on the relative test below.
        neg = dX < 0
        t = 1.0
        if neg.any():
            t = min(1.0, 0.99 * float((-X[alive][neg] / dX[neg]).min()))
        # Backtrack on the residual norm, with the pool clipped at zero the way
        # `integrate` clips it.
        for _ in range(bt):
            c_t, X_t = np.maximum(c + t * dc, 0.0), X.copy()
            X_t[alive] = X[alive] + t * dX
            if float((np.abs(residual(c_t, X_t)[0]) / rscale).max()) < rn:
                break
            t *= 0.5
        else:
            return c, X, J, False, it  # no descent direction — report it, P9
        c, X = c_t, X_t
    return c, X, J, float((np.abs(residual(c, X)[0]) / rscale).max()) < tol, iters


def _jacobian(residual, rhs, idx, scale, c, X, alive, dmu_dc=None):
    """Finite differences in ``c``; the ``X`` columns are ``z_i`` and are exact.

    One residual call per free metabolite. The ``X`` block needs no probing:
    ``d(dc/dt)/dX_i = z_i`` and ``d(mu_j - D)/dX_i = 0``, and ``z`` comes out of
    the unit-biomass right-hand side for free.

    **The step is relative to ``c``, and this is the subtle one.** Two failures
    bracket it, and both were measured on this system.

    Too small: the heads run in float32, so a perturbation near 1e-6 relative is at
    the noise floor and most columns come back as noise. ``sqrt(eps_float32)`` is
    ~3e-4, so 1e-3 is the smallest safe relative step.

    Too large -- and ``1e-3 * (c + Km)`` **is** too large, which cost a session:
    the limiting metabolite is by definition the scarce one, and at
    ``EX_k_e``'s ``c = 3.0e-8`` against ``Km = 1e-3`` that step is **33x c
    itself**. It secants straight across the Michaelis-Menten saturation and
    returns ``d(mu)/dc = 8.4e4`` where the LP's own dual, and a step of ``1e-4 c``,
    both give **5.12e6** -- 61x low, on precisely the row that sets the answer.
    So the step is relative to ``c``, with ``Km`` only in a floor for a metabolite
    at ``c = 0``, which has no scale of its own.
    """
    r0, _ = residual(c, X)
    n = len(r0)
    Jc = np.empty((n, len(idx)))
    for k, j in enumerate(idx):
        h = 1e-3 * max(c[j], 1e-3 * scale[j])
        cp = c.copy()
        cp[j] += h
        Jc[:, k] = (residual(cp, X)[0] - r0) / h
    live = np.flatnonzero(alive)
    if dmu_dc is not None:
        # The growth rows exactly, rather than finite-differenced. Free wherever
        # the residual is an LP -- the shadow prices come out of the same solve --
        # and it removes an inconsistency that is otherwise real: with an LP
        # residual and a surrogate Jacobian, `mu(c) = D` is enforced on one
        # function and differentiated on another.
        g = dmu_dc(c)[np.ix_(live, idx)]
        ok = np.isfinite(g).all(1)  # a non-finite row means "no exact value, keep the FD"
        Jc[len(idx) :][ok] = g[ok]
    Jx = np.zeros((n, len(live)))
    for k, i in enumerate(live):
        e = np.zeros_like(X)
        e[i] = 1.0
        Jx[: len(idx), k] = rhs(c, e)[0][idx]  # dc/dt is linear in X, so this is z_i
    return np.hstack([Jc, Jx])


def _lstsq_step(J, r, col_c, col_x):
    """§13.4: diagonally precondition, then least-squares. Rank deficiency is normal.

    Columns are scaled to the variables' own units (Km for a concentration, the
    member's own biomass for an abundance) and rows to their largest entry, which
    is what keeps `lstsq`'s `rcond` cut meaningful across five decades of `x_scale`.
    """
    dc = np.concatenate([col_c, col_x])
    A = J * dc
    dr = np.maximum(np.abs(A).max(1), 1e-30)
    w = np.linalg.lstsq(A / dr[:, None], r / dr, rcond=1e-10)[0]
    return w * dc


def stability(J_full: np.ndarray) -> dict:
    """Eigenvalues of the fixed point's Jacobian. Stable iff every real part < 0."""
    ev = np.linalg.eigvals(J_full)
    k = int(np.argmax(ev.real))
    return {
        "max_real_eigenvalue": float(ev[k].real),
        "dominant_eigenvalue_imag": float(ev[k].imag),
        "stable": bool(ev.real.max() < 0),
        "n_eigenvalues": int(len(ev)),
    }


def sensitivity(J: np.ndarray, idx: np.ndarray, scale: np.ndarray, dilution: float) -> np.ndarray:
    """Implicit-function derivative of the fixed point w.r.t. the feed, one solve.

    ``F(y, c_feed) = 0`` with ``dF/dc_feed = D I`` on the pool block, so
    ``dy*/dc_feed = -J^-1 D I``. Columns are feed metabolites (the free ones),
    rows are ``[c_free, X_live]`` — every medium component's effect on every
    concentration and every abundance, for the price of one factorisation. This is
    the object §13.4 wants and the one V4 finite-differences.
    """
    n = J.shape[0]
    rhs = np.zeros((n, len(idx)))
    rhs[: len(idx), :] = -dilution * np.eye(len(idx))
    dc = np.concatenate([scale[idx], np.ones(n - len(idx))])
    A = J * dc
    dr = np.maximum(np.abs(A).max(1), 1e-30)
    return np.linalg.lstsq(A / dr[:, None], rhs / dr[:, None], rcond=1e-10)[0] * dc[:, None]


def _lp_mu_rows(duals, models, exchanges, eps, km, c, n_g):
    """``d(mu_i)/dc`` for every member, exactly, from the LP's own shadow prices.

    The stored dual is ``d(mu_max)/d(uptake bound)``, and §3.3's bound is
    ``b_m = -Vmax_m * u_m`` with ``u = c/(Km+c)``, so the chain rule is

        d(mu)/dc = pi * (-Vmax) * Km/(Km+c)^2

    with **the same two corrections the label pipeline applies** and for the same
    reasons (`cfs.surrogate.data._organism_arrays`): the dual is that derivative
    only where the bound *binds*, and elsewhere it is the metabolite's value in
    the network -- positive for a waste product like CO2, which would claim that
    more nutrient lowers growth. Clamping at zero also drops the solver dust that
    makes up half the "non-zero" duals. Do not simplify this to a plain ``-pi``.

    The duals ride along on the residual's own solve, so this is free. It is
    re-solved only if the cache is not at ``c`` -- a stale Jacobian row here would
    be silent, and the caller's evaluation order is not something to rely on.
    """
    from cfs.surrogate.behaviour import VMAX
    from cfs.surrogate.data import _DUAL_TOL

    if duals.get("c") is None or not np.array_equal(duals["c"], c):
        from cfs.compose.dfba import rhs_truth

        rhs_truth(models, exchanges, c, np.zeros(n_g), eps, duals=duals)
    g = np.zeros((n_g, len(exchanges)))
    dudc = km / (km + c) ** 2
    for i in range(n_g):
        sh = duals.get(i)
        if sh is None:
            continue  # P2: a non-optimal member has no valid sensitivity
        pi = np.array([sh.get(ex, 0.0) for ex in exchanges])
        g[i] = np.where(pi < -_DUAL_TOL, -pi * VMAX, 0.0) * dudc
    return g


def _mixed_rhs(sur, models, eps, tol):
    """Solve both, and keep the **surrogate** wherever it agrees with the LP.

    The pure-LP residual paired with a surrogate Jacobian is an inexact Newton
    whose model error is Head B's off-distribution error, and measured, that is
    large enough to break the line search: 0-6 iterations before no descent
    direction can be found. This keeps the residual equal to the function the
    Jacobian differentiates *wherever the two agree*, and pays for truth only on
    the members where they do not -- which is where an inconsistent Jacobian was
    going to be wrong anyway.

    It costs the LP either way, since divergence cannot be detected without it.
    What it buys is consistency, not solves. ``tol`` is relative on ``mu``.
    """
    from cfs.groundtruth.solve import load_km_defaults, solve

    km_cfg = load_km_defaults()
    col = {ex: j for j, ex in enumerate(sur.exchanges)}
    n_g = len(sur.members)
    st = {"duals": {}, "c": None, "lp": np.zeros(n_g, bool), "n_lp": 0, "n": 0}

    def f(c, X):
        mu, z = sur.mu_and_z(c, np.ones(len(sur.genome_ids), dtype=np.float32))
        mu, z = mu[sur.members].copy(), z[sur.members].copy()
        conc = dict(zip(sur.exchanges, c.tolist(), strict=True))
        duals, use = {}, np.zeros(n_g, bool)
        for k in range(n_g):
            sol = solve(models[k], conc, 1.0, eps, km_cfg)
            st["n"] += 1
            if sol.status != "optimal":
                continue  # P2: keep the surrogate's row rather than zeroing it
            duals[k] = sol.shadow_prices
            if abs(mu[k] - sol.mu_max) > tol * max(abs(sol.mu_max), 1e-30):
                use[k] = True
                st["n_lp"] += 1
                mu[k] = sol.mu_max
                z[k] = 0.0
                for ex, v in sol.z.items():
                    z[k, col[ex]] = v
        st.update(duals=duals, c=c.copy(), lp=use)
        return (X[:, None] * z).sum(0), mu

    return f, st


def _mixed_mu_rows(st, exchanges, km, c, n_g):
    """Exact dual rows for the members the LP was used for; NaN for the rest.

    The Jacobian then differentiates, per member, whichever function the residual
    actually evaluated -- which is the whole point of the mix.
    """
    from cfs.surrogate.behaviour import VMAX
    from cfs.surrogate.data import _DUAL_TOL

    g = np.full((n_g, len(exchanges)), np.nan)
    if st["c"] is None or not np.array_equal(st["c"], c):
        return g  # stale: fall back to finite differences rather than lie
    dudc = km / (km + c) ** 2
    for k in range(n_g):
        if not st["lp"][k] or k not in st["duals"]:
            continue
        pi = np.array([st["duals"][k].get(ex, 0.0) for ex in exchanges])
        g[k] = np.where(pi < -_DUAL_TOL, -pi * VMAX, 0.0) * dudc
    return g


# --------------------------------------------------------------------------- #
# The command
# --------------------------------------------------------------------------- #


def run(
    value_dir: Path,
    behaviour_dir: Path,
    out: Path,
    *,
    organisms: list[str],
    labels_dir: Path | None = None,
    medium: Path | None = None,
    dilution: float | None = None,
    dilution_frac: float = 0.2,
    roster_path: Path | None = None,
    eps: float = 1e-3,
    mix_mu_rel: float | None = None,
    seed: int = 0,
    scales: Path | None = None,
    fd_check: int = 20,
) -> dict:
    """One community, one feed, one dilution: solve, then characterise the state."""
    from scipy.optimize import nnls

    from cfs.compose.dfba import Surrogate, _medium_vector, rhs_surrogate, rhs_truth

    sur = Surrogate(value_dir, behaviour_dir, organisms=organisms)
    feed = _medium_vector(sur, organisms, labels_dir, medium, seed, scales)
    rhs = lambda c, X: rhs_surrogate(sur, c, X)  # noqa: E731

    # `--roster`: solve the true LP for the *residual* and keep the surrogate for
    # the Jacobian. The economics are the opposite of §8.6g(4)'s trajectory
    # fallback: an equilibrium is **one state**, so exact truth costs `G` solves
    # per Newton iteration -- tens to hundreds for a whole steady state, against
    # the 24.6% of 15600 member-steps a trajectory pays. Never finite-difference
    # through the LP: that would be `n_free * G` solves per Jacobian, ~250x more.
    rhs_lp = dmu_dc = mix = None
    if roster_path is not None:
        import cobra

        from surrogate_mgem.data import read_roster

        by_id = {gm.genome_id: gm for gm in read_roster(Path(roster_path))}
        models = [cobra.io.read_sbml_model(str(by_id[g].model_path)) for g in organisms]
        duals: dict = {}
        rhs_lp = lambda c, X: rhs_truth(  # noqa: E731
            models, sur.exchanges, c, X, eps, duals=duals
        )
        dmu_dc = lambda c: _lp_mu_rows(  # noqa: E731
            duals, models, sur.exchanges, eps, sur.km, c, len(organisms)
        )
        if mix_mu_rel is not None:
            rhs_lp, mix = _mixed_rhs(sur, models, eps, mix_mu_rel)
            dmu_dc = lambda c: _mixed_mu_rows(  # noqa: E731
                mix, sur.exchanges, sur.km, c, len(organisms)
            )

    mu_feed = rhs(feed, np.zeros(len(organisms)))[1]
    if mu_feed.max() <= 0:
        raise ValueError("nobody grows on the feed")
    D = float(dilution) if dilution is not None else dilution_frac * float(mu_feed.max())

    # Warm start: bisect a *partially* scaled feed for `mu = D`, then match
    # abundances with one non-negative least squares on the pool balance. Both
    # halves are load-bearing and both were got wrong first.
    #
    # Only what the community consumes is scaled down. A secreted metabolite's
    # steady state sits at or above the feed, so dragging it down with everything
    # else asks the pool balance for the wrong sign -- measured, the NNLS then
    # returns `X = 0` on 8 of 9 cells, because water and protons dominate the
    # right-hand side and the community secretes both.
    #
    # And **do not warm-start by integrating**, which is the obvious thing and was
    # tried twice. The medium saturates `mu` at ~0.2% of the feed, so the dynamics
    # are stiff at the kink: explicit Euler ratchets `X` upward -- growing at
    # `mu - D` whenever `c > 0` and decaying only at `D` after it clips `c` to
    # zero -- and lands at `X ~ 1e8`, `|dc| ~ 1e10`. At a small enough step it
    # instead washes out to the *spurious extinction* fixed point (`X ~ 1e-9`,
    # residual 1e-13, and a genuine root). One of the two happens on most cells.
    consumed = rhs(feed, np.ones(len(organisms)))[0] < 0
    zero = np.zeros(len(organisms))

    def _draw(th):
        c = feed.copy()
        c[consumed] = th * feed[consumed]
        return c

    lo, hi = 0.0, 1.0
    for _ in range(60):
        th = 0.5 * (lo + hi)
        if rhs(_draw(th), zero)[1].max() > D:
            hi = th
        else:
            lo = th
    c0 = _draw(hi)

    free = np.zeros(len(sur.exchanges), dtype=bool)
    for i in sur.members:
        free |= sur.mask[i]  # a metabolite nobody exchanges solves as c = c_feed
    idx = np.flatnonzero(free)
    # `dc/dt` is linear in X, so the pool balance at `c0` is a non-negative least
    # squares, and its zeros are the first guess at the active set.
    Z = np.array([rhs(c0, np.eye(len(organisms))[i])[0][idx] for i in range(len(organisms))])
    x0 = nnls(Z.T, -D * (feed - c0)[idx])[0]
    if not x0.any():
        x0 = np.full(len(organisms), 1e-6)
    LOGGER.info("warm start: theta %.3e, X %s", hi, np.array2string(x0))

    sol = solve_steady(
        rhs_lp or rhs,
        feed,
        D,
        free,
        sur.km,
        c0,
        x0,
        rhs_jac=rhs if rhs_lp else None,
        dmu_dc=dmu_dc,
    )
    c, X, alive, J = sol["c"], sol["X"], sol["alive"], sol["J"]

    # The full (c, X) Jacobian: d(X_i (mu_i - D))/dc = X_i dmu_i/dc, and its X
    # block vanishes at the fixed point because mu_i = D there.
    n_c = len(idx)
    J_full = J.copy()
    J_full[n_c:, :] *= X[alive][:, None]
    S = sensitivity(J, idx, sur.km, D) if sol["converged"] else None

    _, mu = (rhs_lp or rhs)(c, X)
    _rs = np.concatenate([D * np.maximum(sur.km[idx], 1e-30), np.full(int(alive.sum()), D)])
    _r_used = np.concatenate([(D * (feed - c) + (rhs_lp or rhs)(c, X)[0])[idx], (mu - D)[alive]])
    _mu_s = rhs(c, X)[1]
    _r_sur = np.concatenate([(D * (feed - c) + rhs(c, X)[0])[idx], (_mu_s - D)[alive]])
    depth = mu / np.where(mu_feed > 0, mu_feed, np.nan)
    reach = sur.reach(c)
    report = {
        "genome_ids": organisms,
        "dilution": D,
        "residual": ("mixed" if mix else "lp") if rhs_lp else "surrogate",
        "residual_max": float(np.abs(_r_used).max()),
        # The dimensionless one the solver actually tests: pool rows over `D*Km`,
        # growth rows over `D`.
        "residual_max_scaled": float((np.abs(_r_used) / _rs).max()),
        # The *surrogate's* residual at the same state. In LP or mixed mode this
        # is not what was solved -- it is how far the surrogate alone is from
        # calling this a steady state, which is the honest accuracy number.
        "residual_max_scaled_surrogate": float((np.abs(_r_sur) / _rs).max()),
        "converged": sol["converged"],
        "newton_iterations": sol["newton_iters"],
        "active_set_passes": sol["active_set_passes"],
        # Coexistence: who is left at this dilution rate.
        "survivors": [g for g, a in zip(organisms, alive, strict=True) if a],
        "washed_out": [g for g, a in zip(organisms, alive, strict=True) if not a],
        "abundance": X.tolist(),
        "mu": mu.tolist(),
        # Invasion / colonisation resistance: mu - D at c* for everyone not in the
        # resident set. One head evaluation, no re-solve.
        "invasion_score": {g: float(mu[i] - D) for i, g in enumerate(organisms) if not alive[i]},
        **(stability(J_full) if sol["converged"] else {"stable": None}),
        # §13.7: an equilibrium is a drawn-down medium, the regime Head B is worst
        # in. Report the two §8.6g(1) predictors at c* so the answer is honest.
        "depth": depth.tolist(),
        "reach": None if reach is None else reach.tolist(),
    }
    if rhs_lp is not None:
        # What the split bought: the surrogate's own `mu` at the *true* fixed
        # point. This is the honest accuracy statement for §13.4, measured where
        # the use case actually evaluates rather than on held-out design media.
        mu_hat = _mu_s
        report["mu_surrogate"] = mu_hat.tolist()
        if mix is not None:
            report["lp_member_fraction"] = mix["n_lp"] / max(mix["n"], 1)
            report["lp_members_at_c_star"] = [
                g for g, u in zip(organisms, mix["lp"], strict=True) if u
            ]
        report["mu_rel_at_fixed_point"] = (
            np.abs(mu_hat - mu) / np.maximum(np.abs(mu), 1e-30)
        ).tolist()
    # V4 re-solves, so it is skipped in LP mode: it would cost `n * G * iters` LP
    # solves and would check the surrogate's sensitivity against the true root.
    if S is not None and fd_check and rhs_lp is None:
        report["v4_fd_check"] = _fd_check(rhs, feed, D, free, sur.km, c, X, alive, S, fd_check)
    Path(out).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        Path(out) / "steady_state.npz",
        c=c,
        X=X,
        alive=alive,
        J=J_full,
        sensitivity=np.zeros(0) if S is None else S,
        free=idx,
        exchanges=np.array(sur.exchanges),
        genome_ids=np.array(organisms),
    )
    (Path(out) / "steady_state.json").write_text(json.dumps(report, indent=2))
    return report


def _fd_check(rhs, feed, D, free, scale, c, X, alive, S, n):
    """V4: re-solve at a perturbed feed and compare against the implicit derivative."""
    rng = np.random.default_rng(0)
    idx = np.flatnonzero(free)
    cols = rng.choice(len(idx), size=min(n, len(idx)), replace=False)
    rel = []
    for k in cols:
        j = idx[k]
        h = 1e-3 * max(feed[j], scale[j])
        fp = feed.copy()
        fp[j] += h
        s = solve_steady(rhs, fp, D, free, scale, c, X)
        if not s["converged"] or not np.array_equal(s["alive"], alive):
            continue
        fd = np.concatenate([(s["c"][idx] - c[idx]), (s["X"] - X)[alive]]) / h
        den = max(float(np.abs(S[:, k]).max()), float(np.abs(fd).max()), 1e-30)
        rel.append(float(np.abs(fd - S[:, k]).max() / den))
    return {
        "n": len(rel),
        "max_rel_error": max(rel) if rel else None,
        "median_rel_error": float(np.median(rel)) if rel else None,
    }
