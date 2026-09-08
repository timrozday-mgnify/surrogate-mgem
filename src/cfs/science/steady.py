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

from cfs.compose.dfba import chain_to_c

LOGGER = logging.getLogger("cfs.science.steady")

# `log(X/X_ref)` is clipped here: 30 decades below the largest live member is
# washed out by any standard, and an unbounded log lets a solver spend its whole
# budget walking to minus infinity.
_LOG_X_FLOOR = 69.0


# Trust-region escalations per Newton iteration before declaring no descent.
_LM_ESCALATIONS = 8


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
    solver: str = "newton",
    ptc: float = 0.0,
    readmits: int = 1,
    invade_rel: float = 1e-2,
    rhs_batch=None,
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
    # Anti-cycling. Two members whose `mu` differ by 1e-4 at `c*` will otherwise
    # trade places forever -- dropped for a negative X, re-admitted for `mu > D` at
    # the next iterate -- and the loop exits on its pass budget with whichever
    # state it happened to be in. It is the problem Bland's rule solves for the
    # simplex.
    #
    # A *permanent* ban was the first version and it is too strong: measured over
    # the roster, 4 of 10 cells then returned a state an excluded member can
    # invade -- three within 0.2% of `D` of a genuine tie, one at `mu - D` = +41.
    # Termination only needs the re-admissions to be *finite*, not forbidden, so
    # each member gets `readmits` of them and the ban falls back to permanent once
    # they are spent. `readmits = 0` is the original rule.
    readmit = np.full(len(x0), int(readmits))

    def _res(f):
        def residual(c, X):
            dc, mu = f(c, X)
            return np.concatenate([(D * (feed - c) + dc)[idx], (mu - D)[alive]]), mu

        return residual

    def _res_batch(fb):
        """The same residual at `B` media at once, one row per medium."""

        def residual_b(C, X):
            dc, mu = fb(C, X)  # (B, M), (G, B)
            return np.concatenate(
                [(D * (feed[None] - C) + dc)[:, idx], (mu.T - D)[:, alive]], axis=1
            )

        return residual_b

    residual, residual_jac = _res(rhs), _res(rhs_jac)
    # The Jacobian's finite differences are `n_free` media differing in one
    # coordinate each -- exactly what a batched head evaluation is for.
    residual_jac_b = None if rhs_batch is None else _res_batch(rhs_batch)

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
    info = {
        "newton_iters": 0,
        "active_set_passes": 0,
        "converged": False,
        "invadable": False,
        "solver": solver,
    }
    J = None
    for _ in range(G * (1 + int(readmits)) + 2):
        info["active_set_passes"] += 1
        rscale = np.concatenate([D * np.maximum(scale[idx], 1e-30), np.full(int(alive.sum()), D)])
        if solver != "newton":
            c, X, ok, n_it = _root(residual, idx, scale, rscale, c, X, alive, tol, iters, solver)
        else:
            c, X, _J, ok, n_it = _newton(
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
                ptc,
                residual_jac_b,
            )
        info["newton_iters"] += n_it
        if not ok and alive.sum() > 1:
            # An inconsistent survivor set does not have to produce a negative X;
            # it just fails to converge. Drop the least abundant member and retry.
            k = np.flatnonzero(alive)[int(np.argmin(X[alive]))]
            alive[k], X[k] = False, 0.0
            continue
        if (drop := alive & (X <= 1e-9 * X[alive].max(initial=1e-300))).any():
            alive, X[drop] = alive & ~drop, 0.0
            continue
        _, mu = rhs(c, X)
        # A washed-out member is only consistent if it cannot grow at c*.
        if (back := (~alive) & (readmit > 0) & (mu > D * (1 + invade_rel))).any():
            alive = alive | back
            readmit[back] -= 1
            X[back] = 1e-9 * max(X[alive].max(initial=0.0), 1.0)
            continue
        # An excluded member that can grow at `c*` means the active set is wrong,
        # so the state is not a fixed point however small the residual is. Measured
        # over the roster at a 1e-8 threshold, 4 of 10 cells reported one.
        #
        # **But the threshold has to be the head's own accuracy, not machine
        # epsilon.** Those four split cleanly: three at 1.5e-5 to 1.6e-3 of `D` and
        # one at **3.7x `D`**. Head A's `mu_rel` at these fixed points is measured
        # at 1e-4 to 9e-3 (§13.4's mixed-residual arm), so the three small ones are
        # ties this surrogate cannot resolve -- reporting them as invasions claims
        # a precision the model does not have, and chasing them only spends the
        # re-admission budget re-deriving the same state. `invade_rel` is
        # deliberately used for *both* the re-admission test and this report, so
        # the loop never declines to chase a member it then calls an invader.
        info["invadable"] = bool((~alive & (mu > D * (1 + invade_rel))).any())
        info["converged"] = bool(ok) and not info["invadable"]
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
    J = _jacobian(residual_jac, rhs_jac, idx, scale, c, X, alive, dmu_dc, residual_jac_b)
    return {"c": c, "X": X, "alive": alive, "mu": mu, "J": J, **info}


def _root(residual, idx, scale, rscale, c, X, alive, tol, iters, method):
    """``scipy.optimize.root`` on the scaled system. Returns ``(c, X, ok, nfev)``.

    This replaces a hand-rolled damped Newton, and it replaces it because the
    hand-rolled one was **under-globalised**: a single bad step collapsed the whole
    pool to `c ~ 0` -- including metabolites the community *secretes*, whose steady
    state is at or above the feed -- and the clip at zero held it there, at a
    scaled residual of exactly `c_feed/Km` on every fed row at once.

    ``krylov`` is the default: Jacobian-free Newton-Krylov needs only ``J.v``
    products, which cost **one** right-hand-side evaluation each, against the
    ``n_free`` (230-365) a finite-differenced Jacobian costs. ``df-sane`` needs no
    Jacobian at all and ``hybr`` is Powell's dogleg trust region; both are kept
    selectable because they fail differently.

    **Abundances are solved as ``log X``, and that is not a scaling choice -- it
    is what makes a general-purpose root finder usable here at all.** Every method
    tried (``hybr``, ``df-sane``, ``broyden1``, ``krylov``) converges on the toy
    chemostat to `X = 0, c = c_feed`: the *trivial washout root*, which is always
    present, is usually nearest, and is the wrong answer. None of them knows
    ``X > 0``. In ``log X`` that root sits at minus infinity and cannot be reached,
    washout appears as ``w`` drifting down instead, and the active-set loop reads
    it off a threshold. The hand-rolled Newton avoided the same trap only via its
    fraction-to-the-boundary step.

    Scaling matters as much as the method (§7: ``x_scale`` spans five decades):
    concentrations are solved in units of their own ``Km``, abundances relative to
    the current largest, and the residual divided by its row scale, so the solver
    sees an O(1) problem. ``c`` keeps a clip at zero -- the same one
    :func:`cfs.compose.dfba.integrate` uses -- because ``c = 0`` is a legitimate
    part of a solution where an abundance of exactly zero is not.
    """
    from scipy.optimize import root

    live = np.flatnonzero(alive)
    xref = max(float(np.abs(X[live]).max(initial=0.0)), 1e-30)
    n_c = len(idx)

    def unpack(y):
        cc = c.copy()
        cc[idx] = np.maximum(y[:n_c] * scale[idx], 0.0)
        XX = X.copy()
        XX[live] = xref * np.exp(np.clip(y[n_c:], -_LOG_X_FLOOR, 30.0))
        return cc, XX

    def f(y):
        cc, XX = unpack(y)
        return residual(cc, XX)[0] / rscale

    y0 = np.concatenate(
        [c[idx] / scale[idx], np.log(np.maximum(X[live] / xref, np.exp(-_LOG_X_FLOOR)))]
    )
    opts = {
        "krylov": {"maxiter": iters, "fatol": tol},
        "df-sane": {"maxfev": 10 * iters, "fatol": tol},
        "broyden1": {"maxiter": iters, "fatol": tol},
        "hybr": {"maxfev": iters * (len(y0) + 1)},
    }.get(method, {})
    try:
        sol = root(f, y0, method=method, tol=tol, options=opts)
        y = sol.x
        nfev = int(getattr(sol, "nfev", 0) or 0)
    except Exception as exc:  # a diverged inner solve, not a bug (P9)
        LOGGER.warning("%s failed: %s", method, exc)
        y, nfev = y0, 0
    cc, XX = unpack(y)
    # A member driven to the log floor is washing out; hand the active-set loop a
    # literal zero so its relative drop test fires.
    XX[live] = np.where(y[n_c:] <= -_LOG_X_FLOOR + 1e-9, 0.0, XX[live])
    # Trust the residual, not the solver's own flag: the methods disagree about
    # what `tol` means and none of them measures it in this row scaling.
    return cc, XX, float(np.abs(f(y)).max()) < tol, nfev


def _newton(
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
    bt,
    dmu_dc=None,
    ptc=0.0,
    residual_b=None,
):
    """Damped Newton on the square system over ``idx`` and the live members.

    ``ptc`` turns on the **Levenberg-Marquardt trust region**: when backtracking
    fails, escalate the damping and take a *different* direction rather than give
    up. It is the globalisation §13.4's failing cells ask for and a line search
    cannot give -- backtracking only shortens a direction, and the measured failure
    is a *direction* that collapses the whole pool to ``c ~ 0``, including
    metabolites the community secretes, where the clip at zero then holds it.

    ``ptc`` is the initial damping and ``0`` disables the escalation entirely, so
    every number measured before this reproduces bit for bit: the escalation can
    only fire where plain Newton already returns "no descent direction".
    """
    J = None
    lam = 0.0
    for it in range(iters):
        r, _ = residual(c, X)
        rn = float((np.abs(r) / rscale).max())
        if rn < tol:
            return c, X, J, True, it
        J = _jacobian(residual_jac, rhs_jac, idx, scale, c, X, alive, dmu_dc, residual_b)
        accepted = False
        for _ in range(_LM_ESCALATIONS if ptc > 0.0 else 1):
            step = _lstsq_step(J, -r, scale[idx], np.maximum(np.abs(X[alive]), 1e-12), lam)
            dc = np.zeros_like(c)
            dc[idx] = step[: len(idx)]
            dX = step[len(idx) :]
            # Fraction to the boundary, as an interior-point method does: never
            # let a step take an abundance through zero. Without it one Newton
            # overshoot early on reads as washout, the member is dropped, and --
            # with the anti-cycling ban -- can never come back; the solve then
            # converges to the trivial `X = 0, c = c_feed` state, which is a fixed
            # point and the wrong one. A member that really is washing out reaches
            # zero geometrically instead, and is dropped on the relative test in
            # `solve_steady`.
            #
            # A matching rule on the *other* boundary is **refuted**, and it looked
            # obvious: the measured failure is a step collapsing the pool to
            # `c ~ 0`, so capping `c`'s move at 0.9 of the distance to zero should
            # stop it. Measured, it takes cell 1 from converged (1.8e-8, 11
            # iterations) to failed (20.0, 21) and cell 5 from 7.8e-6 to 14. A
            # concentration legitimately goes to zero -- a metabolite absent from
            # the steady state is a normal outcome, where an abundance of exactly
            # zero is a change of active set -- so the two boundaries are not
            # symmetric and only `X` gets the rule.
            neg = dX < 0
            t = 1.0
            if neg.any():
                t = min(1.0, 0.99 * float((-X[alive][neg] / dX[neg]).min()))
            # Backtrack on the residual norm, with the pool clipped at zero the
            # way `integrate` clips it.
            for _ in range(bt):
                c_t, X_t = np.maximum(c + t * dc, 0.0), X.copy()
                X_t[alive] = X[alive] + t * dX
                if float((np.abs(residual(c_t, X_t)[0]) / rscale).max()) < rn:
                    accepted = True
                    break
                t *= 0.5
            if accepted:
                break
            # Backtracking exhausted: the *direction* is wrong, not its length.
            # Tighten the trust region and ask for a different one.
            lam = max(ptc, lam * 10.0)
        if not accepted:
            return c, X, J, False, it  # no descent direction — report it, P9
        c, X = c_t, X_t
        lam = 0.0 if lam <= ptc else lam / 10.0
    return c, X, J, float((np.abs(residual(c, X)[0]) / rscale).max()) < tol, iters


def _jacobian(residual, rhs, idx, scale, c, X, alive, dmu_dc=None, residual_b=None):
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
    h = 1e-3 * np.maximum(c[idx], 1e-3 * scale[idx])
    if residual_b is not None:
        # One batched head evaluation instead of `n_free` sequential ones. The
        # media differ in a single coordinate each -- but evaluating them one at a
        # time is nearly all JAX dispatch: 11.6 ms per medium against 0.143 ms each
        # in a batch of 64.
        #
        # **The unperturbed medium goes in the same batch, and that is required,
        # not tidiness.** XLA does not compute a batch of `n` in float32 the way it
        # computes a batch of 1: `z` differs by ~7e-5 between the two. Differencing
        # a batched value against a singly-computed `r0` puts that discrepancy in
        # the numerator over a step of `1e-3 c`, which made the Jacobian wrong by a
        # relative 7e+07 -- larger than the derivative being measured. Taking both
        # sides from one call cancels it.
        CP = np.repeat(c[None], len(idx) + 1, axis=0)
        CP[1 + np.arange(len(idx)), idx] += h
        RB = residual_b(CP, X)
        Jc = ((RB[1:] - RB[0][None]) / h[:, None]).T
    else:
        Jc = np.empty((n, len(idx)))
        for k, j in enumerate(idx):
            cp = c.copy()
            cp[j] += h[k]
            Jc[:, k] = (residual(cp, X)[0] - r0) / h[k]
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


def _lstsq_step(J, r, col_c, col_x, damp=0.0):
    """§13.4: diagonally precondition, then least-squares. Rank deficiency is normal.

    Columns are scaled to the variables' own units (Km for a concentration, the
    member's own biomass for an abundance) and rows to their largest entry, which
    is what keeps `lstsq`'s `rcond` cut meaningful across five decades of `x_scale`.

    ``damp`` is the Levenberg-Marquardt trust-region parameter: the step solves
    ``(A^T A + damp I) w = A^T r`` in the scaled coordinates instead of ``A w = r``.
    ``damp = 0`` is the plain least-squares step, bit for bit. The normal-equation
    form is the load-bearing detail -- it is symmetric positive definite for any
    ``damp > 0``, so the step is *always* a descent direction for ``||r||^2`` and
    shrinks monotonically as ``damp`` grows. Two cheaper-looking dampings were
    tried on the Monod toy and both are wrong: ``A + damp I`` after row scaling
    perturbs a rank-deficient non-symmetric matrix arbitrarily and returns a step
    **4x larger** than the undamped one, and the pseudo-transient ``A + damp
    diag(rscale)`` is unbounded without a line search (``X`` reaches 1e80).
    """
    dc = np.concatenate([col_c, col_x])
    A = J * dc
    dr = np.maximum(np.abs(A).max(1), 1e-30)
    A, r = A / dr[:, None], r / dr
    if damp > 0.0:
        AtA = A.T @ A
        w = np.linalg.lstsq(AtA + damp * np.trace(AtA) / len(AtA) * np.eye(len(AtA)),
                            A.T @ r, rcond=1e-10)[0]
    else:
        w = np.linalg.lstsq(A, r, rcond=1e-10)[0]
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


def _head_mu_rows(sur, c, heads=None):
    """``d(mu_i)/dc`` for every member from **Head A's own analytic gradient**.

    The surrogate-path sibling of :func:`_lp_mu_rows`, and it matters for the same
    reason: a finite difference is worst exactly on the scarce metabolites that set
    the answer. Head A is analytically differentiable, so those rows never needed
    to be probed at all. Chain rule as in :func:`cfs.science.growth.mu_and_grad`,
    ``dmu/dc = dmu/dx . dx/du . du/dc`` with ``dx/du = (1-x)^2/s`` and
    ``du/dc = Km/(Km+c)^2``, plus the output calibration's derivative -- ``mu`` is
    *reported* calibrated, so the Jacobian must be too.

    **This is also where a warmer Jacobian temperature belongs, and why the two
    jobs collapsed into one.** Head B's ``z`` does not depend on Head A's
    temperature, so the growth rows are the *only* place the shipped
    ``gm_eval_temp = 1e-4`` -- effectively a hard min, hence piecewise-linear
    ``mu`` and a piecewise-constant derivative -- reaches the Jacobian. Passing a
    warmed copy of the heads here smooths it at **no cost to the answer**: the
    residual keeps the cold head, and in an inexact Newton the residual decides
    the fixed point while the Jacobian only decides the rate.

    Returns one row per **member** (``sur.members``), matching
    :func:`_lp_mu_rows` and the active set the Jacobian indexes with -- not one
    per genome in the stack. And NaN rows -- "keep the finite difference" -- for a
    multi-seed value stack, whose pointwise min this gradient does not describe.
    """
    from cfs.surrogate import calibrate

    if len(sur._ens) > 1:
        return np.full((len(sur.members), len(sur.exchanges)), np.nan)
    x = sur._x(c)  # (G, 1, M)
    mu, g = sur.mod.batched_value_and_grad(heads or sur._vheads, sur._jnp.asarray(x))
    raw = np.asarray(mu, dtype=np.float64)[:, 0]  # uncalibrated, unscaled
    gx = np.asarray(g, dtype=np.float64)[:, 0]  # d(raw)/dx
    xk = np.asarray(x, dtype=np.float64)[:, 0]
    dcal = calibrate.deriv(raw[:, None], sur.value_cal)[:, 0]
    # `chain_to_c` carries §13.10's rate scale and, under §13.11's inhibition,
    # the second (secretion) input block folded back onto the same `c`.
    rows = chain_to_c(sur, gx, xk, c) * (sur.mu_scale * dcal)[:, None]
    return rows[sur.members]


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

    ponytail: no §13.11 secretion term. `rhs_truth` does not take `ceq`, so an
    inhibited LP is unreachable from here; add `+ pi * Vmax / c^eq` (where the
    secretion bound binds and `c < c^eq`) at the same time as threading `ceq`
    into `rhs_truth`, or the hybrid Jacobian is silently missing a sign.

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


def _mixed_rhs(sur, models, eps, tol, z_tol=None):
    """Solve both, and keep the **surrogate** wherever it agrees with the LP.

    The pure-LP residual paired with a surrogate Jacobian is an inexact Newton
    whose model error is Head B's off-distribution error, and measured, that is
    large enough to break the line search: 0-6 iterations before no descent
    direction can be found. This keeps the residual equal to the function the
    Jacobian differentiates *wherever the two agree*, and pays for truth only on
    the members where they do not -- which is where an inconsistent Jacobian was
    going to be wrong anyway.

    It costs the LP either way, since divergence cannot be detected without it.
    What it buys is consistency, not solves.

    ``tol`` is relative on ``mu`` and ``z_tol``, if given, relative on ``z`` in the
    2-norm. **The second one is not optional in practice.** Head A is the accurate
    head and Head B is not, so a ``mu``-only trigger fires on nothing exactly where
    it is most needed: measured, one cell fired on **0%** of members at
    ``tol = 0.01`` and stayed at a residual of 7.8e-6, while the pure-LP residual
    converged to a *different* fixed point at which the surrogate's own residual is
    **4.4**. A member can have `mu` right to four decimals and `z` badly wrong --
    that is §8.6g's whole finding restated at an equilibrium.
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
            z_lp = np.zeros_like(z[k])
            for ex, v in sol.z.items():
                z_lp[col[ex]] = v
            diverged = abs(mu[k] - sol.mu_max) > tol * max(abs(sol.mu_max), 1e-30)
            if z_tol is not None and not diverged:
                den = max(float(np.linalg.norm(z_lp)), 1e-30)
                diverged = float(np.linalg.norm(z[k] - z_lp)) > z_tol * den
            if diverged:
                use[k] = True
                st["n_lp"] += 1
                mu[k], z[k] = sol.mu_max, z_lp
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
    mix_z_rel: float | None = None,
    jac_temp: float | None = None,
    solver: str = "newton",
    ptc: float = 0.0,
    d_steps: int = 0,
    readmits: int = 1,
    invade_rel: float = 1e-2,
    warm_start: Path | None = None,
    seed_mode: str = "monoculture",
    seed_probes: int = 4,
    seed: int = 0,
    scales: Path | None = None,
    fd_check: int = 20,
) -> dict:
    """One community, one feed, one dilution: solve, then characterise the state."""
    from scipy.optimize import nnls

    from cfs.compose.dfba import (
        Surrogate,
        _medium_vector,
        rhs_surrogate,
        rhs_surrogate_batch,
        rhs_truth,
    )

    sur = Surrogate(value_dir, behaviour_dir, organisms=organisms)
    feed = _medium_vector(sur, organisms, labels_dir, medium, seed, scales)
    rhs = lambda c, X: rhs_surrogate(sur, c, X)  # noqa: E731
    rhs_b = lambda C, X: rhs_surrogate_batch(sur, C, X)  # noqa: E731

    # `--roster`: solve the true LP for the *residual* and keep the surrogate for
    # the Jacobian. The economics are the opposite of §8.6g(4)'s trajectory
    # fallback: an equilibrium is **one state**, so exact truth costs `G` solves
    # per Newton iteration -- tens to hundreds for a whole steady state, against
    # the 24.6% of 15600 member-steps a trajectory pays. Never finite-difference
    # through the LP: that would be `n_free * G` solves per Jacobian, ~250x more.
    # The growth rows analytically, from Head A itself -- optionally at a warmer
    # temperature than the head ships at, which is free because only the Jacobian
    # sees it.
    jheads = sur._vheads
    if jac_temp is not None:
        from cfs.surrogate import groupmax

        jheads = groupmax.with_temp(sur._vheads, jac_temp)
    dmu_dc = lambda c: _head_mu_rows(sur, c, jheads)  # noqa: E731
    rhs_lp = mix = None
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
            rhs_lp, mix = _mixed_rhs(sur, models, eps, mix_mu_rel, mix_z_rel)

            # Per member, differentiate whichever function that member's residual
            # came from: the LP's duals where the LP was substituted, Head A's own
            # analytic gradient where it was not.
            def dmu_dc(c, _mix=mix, _h=jheads):
                g = _mixed_mu_rows(_mix, sur.exchanges, sur.km, c, len(organisms))
                return np.where(np.isnan(g), _head_mu_rows(sur, c, _h), g)

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
    zero = np.zeros(len(organisms))
    free = np.zeros(len(sur.exchanges), dtype=bool)
    for i in sur.members:
        free |= sur.mask[i]  # a metabolite nobody exchanges solves as c = c_feed
    idx = np.flatnonzero(free)

    def _bisect(who):
        """Warm start for the sub-community `who` (a boolean mask over members).

        Both halves are per-sub-community and both matter. Only what *those*
        members consume is scaled down, and the bisection targets *their* `mu`,
        so a monoculture probe gets the start it would have got from solving that
        member on its own -- which is the whole point of the probe. Reusing the
        full community's `c0` and merely zeroing the other abundances is **not**
        the same thing and does not work: measured, it leaves the two hardest
        roster cells at the collapsed-pool residual of 10, where a per-member
        bisection reaches 1e-05.
        """
        cons = rhs(feed, who.astype(float))[0] < 0

        def draw(th):
            c = feed.copy()
            c[cons] = th * feed[cons]
            return c

        lo, hi = 0.0, 1.0
        for _ in range(60):
            th = 0.5 * (lo + hi)
            if rhs(draw(th), zero)[1][who].max() > D:
                hi = th
            else:
                lo = th
        c = draw(hi)
        # `dc/dt` is linear in X, so the pool balance at `c` is a non-negative
        # least squares, and its zeros are the first guess at the active set.
        Z = np.array(
            [rhs(c, np.eye(len(organisms))[i])[0][idx] for i in range(len(organisms))]
        )
        x = nnls(Z.T, -D * (feed - c)[idx])[0]
        x[~who] = 0.0
        if not x.any():
            x = np.where(who, 1e-6, 0.0)
        LOGGER.info("warm start %s: theta %.3e, X %s", who.astype(int), hi, np.array2string(x))
        return c, x

    everyone = np.ones(len(organisms), dtype=bool)
    c0, x0 = _bisect(everyone)

    def _rstar():
        """Per member, the feed scaling at which it alone breaks even (`mu = D`).

        Tilman's R*, in the one coordinate this design varies. Lower wins: a
        chemostat is won by the member that persists at the scarcest medium, not
        by the fastest grower. 50 bisection steps and no steady-state solve.
        """
        out = np.ones(len(organisms))
        for i in range(len(organisms)):
            who = np.zeros(len(organisms), dtype=bool)
            who[i] = True
            cons = rhs(feed, who.astype(float))[0] < 0
            lo, hi = 0.0, 1.0
            for _ in range(50):
                th = 0.5 * (lo + hi)
                c = feed.copy()
                c[cons] = th * feed[cons]
                if rhs(c, zero)[1][i] > D:
                    hi = th
                else:
                    lo = th
            out[i] = hi
        LOGGER.info("R*: %s", np.array2string(out, precision=4, formatter={"float": "{:.4e}".format}))
        return out

    if warm_start is not None:
        # Continue from another solve's state instead. Members absent from that
        # run enter at `X = 0`, i.e. dead, so the active-set loop's own
        # re-admission test decides whether they can invade it -- which makes this
        # the direct check for multiple fixed points: re-insert the member a
        # leave-one-out dropped and ask whether the state it found is still an
        # equilibrium of the *full* community. `c` is aligned by the frozen index,
        # `X` by genome id.
        z = np.load(warm_start)
        prev = dict(zip([str(g) for g in z["genome_ids"]], z["X"].tolist(), strict=True))
        c0 = z["c"].astype(float).copy()
        x0 = np.array([prev.get(g, 0.0) for g in organisms])
        LOGGER.info("warm start from %s: X %s", warm_start, np.array2string(x0))

    # Natural-parameter continuation in `D`, off by default. The measured failure
    # is not a bad Newton direction -- a Levenberg-Marquardt trust region fires on
    # cells 2 and 3 and changes nothing -- it is that the warm start is nowhere
    # near the answer: the solver has to get from `mu ~ 0.6` to `mu = D = 11.5` in
    # one solve and instead collapses the pool.
    #
    # `D -> mu_max(feed)` is the transcritical end, where the answer is known
    # exactly: `c = feed`, `X = 0`. Walking `D` down from there keeps every step a
    # small perturbation of the last converged fixed point, which is the one thing
    # a warm start can be asked for.
    ladder = [D]
    if d_steps > 0:
        hi = 0.999 * float(mu_feed.max())
        if hi > D:
            ladder = list(D * (hi / D) ** (np.arange(d_steps, -1, -1) / d_steps))
            c0, x0 = feed.copy(), np.full(len(organisms), 1e-9)

    def _from(c_s, x_s):
        """One full solve from a given start, walking the `D` ladder."""
        cc, xx, out = c_s, x_s, None
        for Dk in ladder:
            out = solve_steady(
                rhs_lp or rhs,
                feed,
                Dk,
                free,
                sur.km,
                cc,
                xx,
                rhs_jac=rhs if rhs_lp else None,
                dmu_dc=dmu_dc,
                solver=solver,
                ptc=ptc,
                readmits=readmits,
                invade_rel=invade_rel,
                rhs_batch=rhs_b,
            )
            LOGGER.info("continuation: D %.4g converged=%s", Dk, out["converged"])
            # A failed rung is not fatal: its iterate is still closer to the next
            # rung's answer than the feed is, and the last rung is the one that
            # counts.
            cc, xx = out["c"], np.maximum(out["X"], 1e-12)
        al = out["alive"]
        rs = np.concatenate(
            [D * np.maximum(sur.km[idx], 1e-30), np.full(int(al.sum()), D)]
        )
        r = np.concatenate(
            [
                (D * (feed - out["c"]) + (rhs_lp or rhs)(out["c"], out["X"])[0])[idx],
                (out["mu"] - D)[al],
            ]
        )
        out["res_scaled"] = float((np.abs(r) / rs).max())
        return out

    def _margin(sol):
        """`max_j (mu_j(c*) - D) / D` over the excluded members; -inf if none."""
        dead = ~sol["alive"]
        if not dead.any():
            return -np.inf
        return float(((sol["mu"][dead] - D) / D).max())

    def _key(sol):
        """Rank candidate states: converged first, then margin, then residual.

        The **signed** margin rather than the `invadable` flag, because the flag at
        `invade_rel` calls a state valid whenever its excluded member is above
        break-even by less than that -- and 2 of the 4 converging roster cells
        returned exactly such a state.

        But the margin is only meaningful on a **converged** state, and getting
        that wrong cost a gate run: a collapsed pool has `c ~ 0`, so `mu ~ 0` for
        everybody, so *nobody* can invade it and it scores the most negative margin
        in the set. Ranking on margin alone therefore prefers the degenerate state
        it is supposed to reject -- the same trap as judging a root find by its
        residual, one level up. Among unconverged candidates, rank on residual.
        """
        return (
            sol["converged"],
            -_margin(sol) if sol["converged"] else -np.inf,
            -sol["res_scaled"],
        )

    sol = _from(c0, x0)
    if seed_mode == "monoculture" and warm_start is None and len(organisms) > 1:
        # Seed from each member's own monoculture equilibrium and keep the best
        # state by `_margin`. Measured: the bisection start commits to a survivor
        # and, on the two hardest roster cells, collapses the pool at a scaled
        # residual of 10 -- where the *other* member's monoculture basin reaches
        # 1.3e-05 and 3.8e-05. It also returns a strictly invadable state on 2 of
        # the 4 cells it does converge. An unconverged monoculture is still a far
        # better seed than the bisection, so the probe is not required to succeed.
        #
        # Each probe is two solves -- the monoculture (`readmits=0`, so nobody
        # re-enters and it stays a monoculture) and the full community from it --
        # so it is capped at `seed_probes` of them and stops at the first strictly
        # valid state. Both bounds are needed: the 21-member cell does not finish
        # in an hour uncapped, against ~15 min for the bisection alone.
        # Fastest grower at the feed first. The survivor of a chemostat is usually
        # the member with the lowest break-even concentration, which correlates
        # with `mu` at the feed, so this puts the likely answer early and the early
        # stop then ends the loop -- which is what keeps the cost near one probe
        # instead of `2G` on a 21-member community.
        # Order by **R\***, the break-even feed scaling, not by `mu` at the feed.
        # This is the chemostat's own theory (Hsu, Hubbell & Waltman 1977; Tilman):
        # the survivor is the member that persists at the *lowest* resource
        # concentration, which is a different statistic from growing fastest when
        # replete. It is not a refinement -- the two disagree, and where they do,
        # `mu`-at-feed is wrong: on roster cell 2 it ranks the member the solver
        # then returned in a strictly invadable state, while R* ranks the other.
        # Measured, R* names the valid survivor on 5 of 5 cells checked.
        #
        # It costs one bisection per member and no steady-state solve, and the
        # *gap* between the two smallest is the cell's difficulty: 5x on the one
        # cell with a decisive invasion margin, 0.02-0.15% on the near-tie cells.
        order = np.argsort(_rstar())[:seed_probes]
        # Pairs are **not** probed. Two species coexist only on two limiting
        # resources, and `k = 1` at every fixed point measured -- one metabolite
        # carries 100% of the growth gradient. The pairwise arm was built and run
        # anyway: 10 probes on the one cell with a two-member state, and it found
        # nothing the singles had not.
        for who_ix in [(int(i),) for i in order]:
            if _margin(sol) < 0.0 and sol["converged"]:
                break
            who = np.zeros(len(organisms), dtype=bool)
            who[list(who_ix)] = True
            ci, xi = _bisect(who)
            mono = solve_steady(
                rhs_lp or rhs, feed, D, free, sur.km, ci, xi,
                rhs_jac=rhs if rhs_lp else None, dmu_dc=dmu_dc, solver=solver,
                ptc=ptc, readmits=0, invade_rel=invade_rel,
            )
            cand = _from(mono["c"], np.maximum(mono["X"], 0.0))
            LOGGER.info(
                "seed %s: converged=%s margin=%.3g (best %.3g)",
                "+".join(organisms[int(k)] for k in who_ix),
                cand["converged"], _margin(cand), _margin(sol),
            )
            if _key(cand) > _key(sol):
                sol = cand
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
        "solver": solver,
        "ptc": ptc,
        "d_steps": d_steps,
        "readmits": readmits,
        "invade_rel": invade_rel,
        "jac_temp": jac_temp,
        "residual_max": float(np.abs(_r_used).max()),
        # The dimensionless one the solver actually tests: pool rows over `D*Km`,
        # growth rows over `D`.
        "residual_max_scaled": float((np.abs(_r_used) / _rs).max()),
        # The *surrogate's* residual at the same state. In LP or mixed mode this
        # is not what was solved -- it is how far the surrogate alone is from
        # calling this a steady state, which is the honest accuracy number.
        "residual_max_scaled_surrogate": float((np.abs(_r_sur) / _rs).max()),
        "converged": sol["converged"],
        "invadable": sol["invadable"],
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
        # The feed is drawn from the *union* of the members' active subspaces, so
        # dropping a member silently redraws it. Any follow-up that compares
        # communities -- keystone leave-one-out above all -- has to hold it fixed,
        # and cannot unless it is written down.
        feed=feed,
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
