"""M13 / §13.5 — explore metabolic interactions and the media that facilitate them.

The interaction rate is the mass actually *handed between* members:

    E(c, X) = sum_m min( sum_i X_i max(z_im, 0),  sum_i X_i max(-z_im, 0) )

per metabolite the smaller of total secretion and total uptake, so a metabolite
everyone excretes and nobody eats scores zero.

**It cannot be posed at the §13.4 steady state, and that is arithmetic rather
than a measurement.** For a single surviving organism every metabolite has
either ``z >= 0`` or ``z <= 0``, so one of the two sums is zero and the min with
it: ``E = 0`` exactly for a monoculture. §13.4 measured `k = 1` -- one metabolite
carries the whole growth gradient at every fixed point -- and competitive
exclusion then leaves **one survivor on every roster cell at every size** bar one
coexistence. So maximising `E` at the equilibrium is a maximisation of zero.

`E` is therefore evaluated at a **fixed reference abundance** (uniform by
default). That is not a workaround, it is the question worth asking: it makes `E`
a property of the *medium* -- the exchange rate a medium can support per unit
biomass -- which is what "the culture media that facilitate an interaction"
means. It is a capacity, not a prediction of what an assembled community settles
at.

**Exploratory, and P22 is why.** `E` is an objective on flux *magnitude*, Head
B's weakest axis: its direction is far better than its size (worst held-out flux
cosine 0.985 against worst R2 0.93), and §8.6e measured the magnitude to be
badly over-predicted once a member is starved. So every number here is
round-tripped through the true LP, the report leads with the LP's own `E`, and
the survey's *structure* (which metabolite, which donor, which recipient) should
be trusted well ahead of its *rate*.

Two halves, and the first needs no optimiser:

* :func:`survey` scores many media and aggregates the links that appear -- which
  metabolites get handed over, between whom, how often, and which medium was
  best for each. Cheap: one batched head call per chunk of media, no LP.
* :func:`maximise` designs a medium for a chosen objective by projected gradient
  ascent inside §13.2's multiplicative trust region. `E` is non-concave (a min of
  two functions that are neither), so this is multistart and reports the spread
  across starts rather than one point.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from cfs.science.growth import _c_of_x, project

LOGGER = logging.getLogger("cfs.science.interaction")

# Below this share of the largest link, a "link" is solver dust or head noise
# rather than an interaction. Same spirit as `data._DUAL_TOL`.
_LINK_TOL = 1e-6
# ...and this is the share below which a link is not worth *calling* one. The two
# differ on purpose: `_LINK_TOL` decides whether a number is nonzero, `_CALL_TOL`
# decides whether it is an interaction. Scoring structure at `_LINK_TOL` scores
# Head B's dust -- measured, precision 0.06 at recall 1.00, which is a statement
# about the threshold and not about the head.
_CALL_TOL = 1e-2

# **Buffered species**: held at fixed abundance by the vessel rather than by the
# community. A chemostat is pH-controlled and aqueous, so protons and water are
# supplied and absorbed by the buffer and the solvent, not by the members.
#
# That has two consequences and they are the same fact:
#
# 1. their concentration is **pinned** -- an experimenter cannot dial pH as a
#    design variable, so the designer must not either; and
# 2. they are **not interactions**. A proton one member secretes goes into the
#    buffer, not into another member, so scoring it as a handover is wrong.
#
# Without (2) the objective is literally proton exchange: measured on the first
# verified run, `EX_h_e` alone was 871.8 of one community's true interaction rate
# of 893.4 -- **97.6%** -- with `EX_h2o_e` leading two more cells, burying the
# acetaldehyde, glycerol and amino-acid handovers that are the actual biology.
#
# CO2, O2, ammonium and phosphate are deliberately **not** here. They are real
# cross-feeding currencies and nothing in a chemostat buffers them; excluding
# them would be excluding the answer.
_BUFFERED = ("EX_h_e", "EX_h2o_e")
# "Abundant" in the model's own coordinate: `u = c/(Km+c) = 0.999`, i.e. the
# transporters see a saturating supply, which is what a solvent and a pH
# controller provide.
_BUFFER_SAT = 1e3


def keep_mask(exchanges: list[str], buffered=_BUFFERED) -> np.ndarray:
    """Boolean over the metabolite index: which metabolites ``E`` counts."""
    drop = set(buffered or ())
    return np.array([ex not in drop for ex in exchanges])


def buffer_medium(sur, c: np.ndarray, keep: np.ndarray, sat: float = _BUFFER_SAT) -> np.ndarray:
    """Set the buffered species to a saturating concentration, in place of the draw.

    ``sat * Km`` puts ``u = c/(Km+c)`` at ``sat/(1+sat)``, so at the default the
    transporters see a supply they cannot exhaust -- which is what an aqueous,
    pH-controlled vessel provides. Applied to every medium the run considers, so
    the survey and the design are both chemostat media.
    """
    out = np.array(c, dtype=np.float64, copy=True)
    out[~keep] = sat * sur.km[~keep]
    return out


def exchange(z: np.ndarray, X: np.ndarray, keep: np.ndarray | None = None) -> np.ndarray:
    """Per-metabolite interaction rate ``min(total secretion, total uptake)``.

    ``z`` is ``(G, M)`` in mmol/gDW/h with the project's sign convention --
    positive is secretion, negative is uptake -- and ``X`` is ``(G,)`` gDW/L, so
    the result is ``(M,)`` in mmol/L/h.
    """
    w = np.asarray(X, dtype=np.float64)[:, None]
    e = np.minimum((w * np.maximum(z, 0.0)).sum(0), (w * np.maximum(-z, 0.0)).sum(0))
    return e if keep is None else e * keep


def exchange_batch(z: np.ndarray, X: np.ndarray, keep: np.ndarray | None = None) -> np.ndarray:
    """:func:`exchange` over a batch: ``z`` is ``(G, B, M)``, out ``(B, M)``."""
    w = np.asarray(X, dtype=np.float64)[:, None, None]
    e = np.minimum((w * np.maximum(z, 0.0)).sum(0), (w * np.maximum(-z, 0.0)).sum(0))
    return e if keep is None else e * keep[None]


def member_z(sur, c: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """``z`` for the community's members only, ``(n_members, M)``.

    A ``Surrogate`` carries the **whole** stack of trained organisms and indexes
    the community through ``sur.members``; ``alpha`` is likewise stack-wide.
    Getting that wrong is silent when the sizes happen to agree, so every entry
    point here goes through this and :func:`member_z_batch` rather than calling
    the heads directly.
    """
    _, z = sur.mu_and_z(c, np.full(len(sur.genome_ids), alpha, dtype=np.float32))
    return z[sur.members]


def member_z_batch(sur, C: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """:func:`member_z` at ``B`` media at once, ``(n_members, B, M)``."""
    _, z = sur.mu_and_z_batch(C, np.full(len(sur.genome_ids), alpha, dtype=np.float32))
    return z[sur.members]


def objective(sur, c, X, alpha: float = 1.0, keep=None) -> tuple[float, np.ndarray]:
    """``(E, per-metabolite exchange)`` at one medium."""
    e = exchange(member_z(sur, c, alpha), X, keep)
    return float(e.sum()), e


def objective_batch(sur, C, X, alpha: float = 1.0, keep=None):
    """``(E per medium, exchange matrix)`` at ``B`` media, one batched head call."""
    e = exchange_batch(member_z_batch(sur, C, alpha), X, keep)
    return e.sum(1), e


def grad_fd(sur, c: np.ndarray, X: np.ndarray, alpha: float = 1.0, rel: float = 1e-3, keep=None):
    """``(E, dE/dc)`` by forward differences, the whole Jacobian in one batch call.

    Head B is reached through numpy (§3.3's clamp, the active-set projection in
    ``_element_balance``), so there is no analytic gradient to take -- the same
    blocker that stopped ``jacfwd`` for the steady-state Jacobian. Batching makes
    the difference affordable anyway: one medium per free metabolite is exactly
    the shape ``mu_and_z_batch`` is fast at.

    **The step is relative to ``c``, not to ``Km``, and that is load-bearing**
    (§13.4). The metabolites that carry an interaction are the scarce ones, and a
    step sized by ``Km`` is then orders of magnitude larger than the
    concentration itself and secants clean across the Michaelis-Menten
    saturation -- measured at 61x wrong on a limiting metabolite. The floor keeps
    the step above the float32 heads' own noise.

    The unperturbed medium rides in the **same batch** as the perturbed ones.
    XLA does not compute a batch of ``n`` bitwise like a batch of 1, and
    differencing against a separately-evaluated baseline puts that ~1e-5
    discrepancy over a ~1e-3 step -- measured to make a Jacobian wrong by a
    relative 7e+07.
    """
    c = np.asarray(c, dtype=np.float64)
    h = rel * np.maximum(c, 1e-3 * sur.km)
    M = len(c)
    C = np.repeat(c[None], M + 1, axis=0)
    C[1 + np.arange(M), np.arange(M)] += h
    E, _ = objective_batch(sur, C, X, alpha, keep)
    return float(E[0]), (E[1:] - E[0]) / h


def maximise(
    sur,
    c0: np.ndarray,
    X: np.ndarray,
    alpha: float = 1.0,
    *,
    cost: np.ndarray,
    budget: float,
    c_lo: np.ndarray,
    c_hi: np.ndarray,
    iters: int = 120,
    tol: float = 1e-9,
    keep: np.ndarray | None = None,
) -> tuple[np.ndarray, list[float]]:
    """Projected gradient ascent on ``E``, backtracking step. Returns ``(c*, path)``.

    Unlike §13.2's this is **not** a convex program -- ``E`` is a min of two
    functions that are neither concave nor convex, and the relu kinks make the
    gradient piecewise. So a stationary point is all this returns, the caller is
    expected to multistart, and the spread across starts is part of the answer
    (§13.5: "how many *different* media achieve this" is the more useful output).

    **A rejected step costs a value, not a gradient**, which is what makes this
    affordable. ``grad_fd`` is one batched call of ``M+1`` media -- 1.7 s here --
    against ~0.05 s for a single evaluation, and the backtracking rejects most
    trials (measured: 1 accepted in 20). Taking the gradient only after a step is
    accepted turned 36 s into ~5 s for the same path.
    """
    c = project(np.asarray(c0, dtype=np.float64), cost, budget, c_lo, c_hi)
    E, g = grad_fd(sur, c, X, alpha, keep=keep)
    path = [E]
    gn = max(float(np.linalg.norm(g)), 1e-30)
    step = 0.1 * float(np.linalg.norm(c_hi)) / gn
    for _ in range(iters):
        cand = project(c + step * g, cost, budget, c_lo, c_hi)
        E_c = objective(sur, cand, X, alpha, keep)[0]
        if E_c > E:
            c, E = cand, E_c
            path.append(E)
            if len(path) > 2 and path[-1] - path[-2] < tol * max(abs(E), 1e-12):
                break
            E, g = grad_fd(sur, c, X, alpha, keep=keep)
            step *= 1.5
        else:
            step *= 0.5
            if step * float(np.linalg.norm(g)) < 1e-14:
                break
    return c, path


def trust_box(
    sur, c0: np.ndarray, decades: float, keep: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """§13.2's P21 region, widened to the community: the tightest box over members.

    :func:`cfs.science.growth.trust_box` is per organism because that use case
    designs for one. Here every member reads the same medium, so the region is
    the intersection -- a medium inside every member's own design is inside the
    design, which is what P21 asks for.
    """
    lo, hi = None, None
    for k in range(len(sur.genome_ids)):
        x0 = np.asarray(sur._x(c0)[k, 0], dtype=np.float64)
        f = 10.0 ** min(float(decades), 300.0)
        a = _c_of_x(sur, k, x0 / f)
        b = _c_of_x(sur, k, np.clip(x0 * f, 0.0, 1.0 - 1e-9))
        lo = a if lo is None else np.maximum(lo, a)
        hi = b if hi is None else np.minimum(hi, b)
    hi = np.maximum(hi, lo)
    if keep is not None:
        # Buffered species are pinned, not designed: the vessel holds them, so an
        # experimenter could not act on a recommendation to change them.
        lo[~keep] = hi[~keep] = c0[~keep]
    return lo, hi


# --------------------------------------------------------------------------- #
# The links themselves
# --------------------------------------------------------------------------- #


def link_rows(
    z: np.ndarray,
    X: np.ndarray,
    exchanges: list[str],
    gids: list[str],
    top: int = 40,
    z_true: np.ndarray | None = None,
    keep: np.ndarray | None = None,
):
    """The realised handovers at one medium, largest first.

    A metabolite's interaction is a *set* on each side, not a pair: several
    members may secrete it and several take it up. The rate is the metabolite's
    own ``min(secretion, uptake)``; splitting it between specific donor-recipient
    pairs would be a choice the flux data does not make, so donors and recipients
    are reported as sets with their individual rates.

    Pass ``z_true`` to put the LP's own rate and sides in the *same* row. That is
    deliberately not a second list: the question a reader has about a predicted
    interaction is whether it is real and how big it actually is, and two lists
    to join by hand is a worse answer than one table (P22 -- the structure is
    trustworthy well ahead of the rate, and only a side-by-side shows which).
    """
    e = exchange(z, X, keep)
    w = np.asarray(X, dtype=np.float64)[:, None]
    order = np.argsort(-e)[:top]
    cut = _LINK_TOL * float(e.max()) if e.size and e.max() > 0 else 0.0
    et = None if z_true is None else exchange(z_true, X, keep)
    wt = None if z_true is None else np.asarray(X, dtype=np.float64)[:, None]
    rows = []
    for m in order:
        if e[m] <= cut:
            break
        sec, upt = (w * np.maximum(z, 0.0))[:, m], (w * np.maximum(-z, 0.0))[:, m]
        row = {
            "metabolite": exchanges[m],
            "rate": float(e[m]),
            "donors": {gids[i]: float(sec[i]) for i in np.nonzero(sec > cut)[0]},
            "recipients": {gids[i]: float(upt[i]) for i in np.nonzero(upt > cut)[0]},
        }
        if et is not None:
            st, ut = (wt * np.maximum(z_true, 0.0))[:, m], (wt * np.maximum(-z_true, 0.0))[:, m]
            ct = _LINK_TOL * max(float(et.max()), 1e-30)
            row.update(
                {
                    "rate_true": float(et[m]),
                    "donors_true": {gids[i]: float(st[i]) for i in np.nonzero(st > ct)[0]},
                    "recipients_true": {gids[i]: float(ut[i]) for i in np.nonzero(ut > ct)[0]},
                }
            )
        rows.append(row)
    return rows


def structure_score(e_hat: np.ndarray, e_true: np.ndarray) -> dict:
    """How well the surrogate's *structure* matches the LP's, threshold-free where possible.

    Three numbers, because no single one is honest here. Precision and recall at
    a fixed cut depend entirely on the cut. ``precision_at_n_true`` takes exactly
    as many of the top predicted links as the LP actually has, which needs no
    threshold at all and is the number a reader ranking candidate interactions
    cares about. The Spearman is over the union of called links, so it says
    whether the *ordering* survives even where the magnitudes do not.
    """
    from scipy.stats import spearmanr

    ct = _CALL_TOL * max(float(e_true.max()), 1e-30)
    ch = _CALL_TOL * max(float(e_hat.max()), 1e-30)
    real, pred = e_true > ct, e_hat > ch
    n_true = int(real.sum())
    topk = np.argsort(-e_hat)[:n_true] if n_true else np.array([], dtype=int)
    union = real | pred
    rho = float("nan")
    if int(union.sum()) > 2:
        r = spearmanr(e_hat[union], e_true[union])
        rho = float(r.statistic)
    return {
        "n_links_true": n_true,
        "n_links_hat": int(pred.sum()),
        "precision_at_n_true": float(real[topk].sum() / n_true) if n_true else float("nan"),
        "recall": float((pred & real).sum() / n_true) if n_true else float("nan"),
        "precision": float((pred & real).sum() / max(int(pred.sum()), 1)),
        "rate_spearman": rho,
    }


def survey(sur, C: np.ndarray, X: np.ndarray, alpha: float = 1.0, chunk: int = 64, keep=None):
    """``E`` and the per-metabolite exchange matrix over many media. No LP.

    Returns ``(E, EX)`` with ``EX`` of shape ``(B, M)``, so the caller can ask
    both "which media are interactive" and "which interactions are reachable".
    """
    E, EX = [], []
    for i in range(0, len(C), chunk):
        e, ex = objective_batch(sur, C[i : i + chunk], X, alpha, keep)
        E.append(e)
        EX.append(ex)
    return np.concatenate(E), np.concatenate(EX)


def distinguishing(c: np.ndarray, c_ref: np.ndarray, exchanges: list[str], top: int = 10):
    """What a designed medium changed, largest log-fold first — the readable half.

    A 365-vector is not an answer to "which medium facilitates this"; the handful
    of components that moved is.
    """
    a = np.maximum(c, 1e-30)
    b = np.maximum(c_ref, 1e-30)
    lf = np.log10(a / b)
    idx = np.argsort(-np.abs(lf))[:top]
    return [
        {"metabolite": exchanges[i], "log10_fold": float(lf[i]),
         "from": float(c_ref[i]), "to": float(c[i])}
        for i in idx
        if abs(lf[i]) > 1e-6
    ]


# --------------------------------------------------------------------------- #
# V5: the true LP's own interaction rate
# --------------------------------------------------------------------------- #


def true_z(models: list, exchanges: list[str], c: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """``z`` per member from the true FBA + elastic-net solve, aligned to ``exchanges``.

    :func:`cfs.compose.dfba.rhs_truth` accumulates the pool sum and drops the
    per-member vectors, which is exactly what ``E`` needs to keep -- an
    interaction is defined by *who* secretes and *who* takes up.
    """
    from cfs.groundtruth.solve import load_km_defaults, solve

    km_cfg = load_km_defaults()
    conc = dict(zip(exchanges, c.tolist(), strict=True))
    col = {ex: j for j, ex in enumerate(exchanges)}
    z = np.zeros((len(models), len(exchanges)))
    for i, model in enumerate(models):
        sol = solve(model, conc, 1.0, eps, km_cfg)
        if sol.status != "optimal":
            LOGGER.debug("member %d non-optimal (%s) — no growth, no fluxes (P2)", i, sol.status)
            continue
        for ex, v in sol.z.items():
            z[i, col[ex]] = v
    return z


def verified_ascent(
    sur,
    models: list,
    c0: np.ndarray,
    X: np.ndarray,
    alpha: float = 1.0,
    *,
    cost: np.ndarray,
    budget: float,
    decades: float = 0.5,
    max_it: int = 8,
    iters: int = 120,
    eps: float = 1e-3,
    keep: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """Trust-region search on ``E`` with the **true LP as the acceptance test**.

    §13.2's bundle TRF corrects the model to the LP with a tangent, which works
    because ``mu_true`` is concave and the LP's dual *is* its gradient. Neither
    holds here: ``E`` is a min of two non-concave functions, and the elastic-net
    QP gives no supporting hyperplane for it. What does transfer is the other
    half of a trust-region method -- **propose with the model, accept with the
    truth**.

    That is not a refinement, it is what makes the use case reportable. Measured
    without it (5 communities, 64 draws, 3 starts each): the ascent raises
    ``E_hat`` 2-4x every time while the true rate is flat or worse on 3 of 5, and
    ``E_hat/E_true`` at the designed medium runs 1.6x to **infinite** -- one cell
    designed ``E_hat = 2124`` at a medium where the LP has no interaction at all.
    Head A is a certified upper bound off-distribution (§13.2c) so an optimistic
    ``mu`` is at least bounded; ``E`` inherits Head B's magnitude, which has no
    such guarantee, and P22 is exactly this.

    Every accepted point is LP-evaluated, so the returned medium **cannot** be
    worse than the start under the true LP: V5 passes by construction, and the
    number to read instead is how much true gain the search actually found.
    Cost is one FBA per member per iteration.
    """
    c = np.asarray(c0, dtype=np.float64)
    best_true = float(exchange(true_z(models, sur.exchanges, c, eps), X, keep).sum())
    start_true, r, n_lp, accepted = best_true, float(decades), 1, 0
    for _ in range(max_it):
        lo, hi = trust_box(sur, c, r, keep)
        cand, _ = maximise(
            sur, c, X, alpha, cost=cost, budget=budget, c_lo=lo, c_hi=hi, iters=iters, keep=keep
        )
        e_true = float(exchange(true_z(models, sur.exchanges, cand, eps), X, keep).sum())
        n_lp += 1
        if e_true > best_true:
            c, best_true, accepted = cand, e_true, accepted + 1
            r = min(r * 1.5, 2.0)
        else:
            r *= 0.5
            if r < 1e-3:
                break
    return c, {
        "E_true_start": start_true,
        "E_true": best_true,
        "trf_accepted": accepted,
        "trf_radius": r,
        "n_lp": n_lp,
    }


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def run(
    roster_path: Path,
    labels_dir: Path,
    value_dir: Path,
    behaviour_dir: Path,
    out: Path,
    *,
    communities: list[list[str]],
    draws: int = 64,
    starts: int = 4,
    trust_decades: float = 0.5,
    budget_mult: float = 1.0,
    iters: int = 120,
    alpha: float = 1.0,
    seed: int = 0,
    scales: Path | None = None,
    verify: bool = True,
    verify_steps: int = 0,
    buffered: tuple[str, ...] = _BUFFERED,
    screen: bool = True,
) -> dict:
    """Survey the interactions a community can reach, then design media for them.

    Per community: draw ``draws`` §4.3 media and score them (no LP); take the
    ``starts`` most interactive as starts for the ascent; report the designed
    optimum, what it changed, and the links it realises. With ``verify`` every
    reported medium -- the best draw and the best design -- is re-solved with the
    true LP, and the report leads with **that** ``E``, because P22 says the
    surrogate's weakest axis is exactly the magnitude this objective is built on.
    """
    from cfs.compose.dfba import Surrogate, community_medium

    Path(out).mkdir(parents=True, exist_ok=True)
    if verify:
        import cobra

        from surrogate_mgem.data import read_roster

        roster = {gm.genome_id: gm for gm in read_roster(Path(roster_path))}

    cells, saved = [], []
    for n, gids in enumerate(communities):
        sur = Surrogate(value_dir, behaviour_dir, organisms=gids)
        # Loaded per community and dropped, not cached across them: a CarveMe GEM
        # is large in memory and holding every genome the run has ever touched is
        # what killed a 5-community run with no traceback. Re-reading the SBML is
        # seconds; the cache was not worth an OOM.
        models = (
            [cobra.io.read_sbml_model(str(roster[g].model_path)) for g in gids]
            if verify
            else []
        )
        G = len(gids)
        X = np.ones(G)  # uniform reference abundance: E is then per unit biomass
        al = alpha
        keep = keep_mask(sur.exchanges, buffered)
        C = np.array(
            [buffer_medium(
                sur,
                community_medium(labels_dir, gids, sur.exchanges, seed + n * 1000 + d, scales),
                keep,
             )
             for d in range(draws)]
        )
        E, EX = survey(sur, C, X, al, keep=keep)

        # Which interactions are reachable *at all*, and how often. This is the
        # half to trust: it is structure, not magnitude.
        cut = _LINK_TOL * float(EX.max()) if EX.size and EX.max() > 0 else 0.0
        seen = (EX > cut).sum(0)
        order = np.argsort(-EX.max(0))
        reachable = [
            {
                "metabolite": sur.exchanges[m],
                "n_media": int(seen[m]),
                "frac_media": float(seen[m] / max(draws, 1)),
                "max_rate": float(EX[:, m].max()),
                "median_rate_when_present": float(np.median(EX[EX[:, m] > cut, m]))
                if seen[m]
                else 0.0,
                "best_draw": int(np.argmax(EX[:, m])),
            }
            for m in order[:40]
            if seen[m]
        ]

        # Design. E is non-concave, so multistart -- but **from the draws the LP
        # says are interactive, not the ones the head does**. Seeding by `E_hat`
        # seeds exactly where the head is most optimistic, which is the quantity
        # the search then exploits. Measured on AAXE02+ABCC02: Spearman(E_hat,
        # E_true) over 64 draws is **-0.053**, all three E_hat-seeded starts have
        # true E = 0 while the best true draw scores 529.9 at an E_hat ranked well
        # down -- and with every start at zero the LP acceptance test has nothing
        # to discriminate against either, since any positive step "improves" it.
        # One LP per member per draw is the highest-value LP in the whole run.
        rank_rho = float("nan")
        if verify and screen:
            E_true_draws = np.array(
                [float(exchange(true_z(models, sur.exchanges, c), X, keep).sum()) for c in C]
            )
            best = np.argsort(-E_true_draws)[:starts]
            if len(E_true_draws) > 2 and E_true_draws.std() > 0 and E.std() > 0:
                from scipy.stats import spearmanr

                rank_rho = float(spearmanr(E, E_true_draws).statistic)
        else:
            E_true_draws = None
            best = np.argsort(-E)[:starts]
        designs = []
        for s in best:
            c0 = C[s]
            lo, hi = trust_box(sur, c0, trust_decades, keep)
            cost = np.ones_like(c0)
            budget = budget_mult * float(cost @ c0)
            if verify_steps:
                c_star, extra = verified_ascent(
                    sur, models, c0, X, al, cost=cost, budget=budget,
                    decades=trust_decades, max_it=verify_steps, iters=iters, keep=keep,
                )
            else:
                c_star, path = maximise(
                    sur, c0, X, al, cost=cost, budget=budget, c_lo=lo, c_hi=hi,
                    iters=iters, keep=keep,
                )
                extra = {"iterations": len(path) - 1}
            E_star, _ = objective(sur, c_star, X, al, keep)
            designs.append(
                {
                    "start_draw": int(s),
                    "E_start": float(E[s]),
                    "E_hat": E_star,
                    "changed": distinguishing(c_star, c0, sur.exchanges),
                    **extra,
                }
            )
            saved.append((n, int(s), c0, c_star))

        # Rank on the truth where we have it. Ranking multistarts by `E_hat` is
        # ranking them by how optimistic the head is at each -- the exact quantity
        # the search is exploiting.
        key = "E_true" if verify_steps else "E_hat"
        top = int(np.argmax([d[key] for d in designs]))
        c_best = saved[-len(designs) + top][3]
        c_draw = C[int(best[0])]

        cell = {
            "community": gids,
            "n_draws": draws,
            "E_draws_median": float(np.median(E)),
            "E_draws_max": float(E.max()),
            # Whether the head can rank media for this objective at all. If this
            # is ~0 the survey's *ordering* is not usable and only its structure is.
            "draw_rank_spearman": rank_rho,
            "E_true_draws_max": (
                float(E_true_draws.max()) if E_true_draws is not None else None
            ),
            "n_draws_interactive_true": (
                int((E_true_draws > 1e-6).sum()) if E_true_draws is not None else None
            ),
            "E_designed": designs[top]["E_hat"],
            "n_reachable_metabolites": int((EX.max(0) > cut).sum()),
            "reachable": reachable,
            "designs": designs,
            "buffered": list(buffered),
            "links_at_best_design": link_rows(
                member_z(sur, c_best, al), X, sur.exchanges, gids, keep=keep
            ),
        }

        if verify:
            zt_draw = true_z(models, sur.exchanges, c_draw)
            zt_best = true_z(models, sur.exchanges, c_best)
            e_draw, e_best = exchange(zt_draw, X, keep), exchange(zt_best, X, keep)
            hat_draw = exchange(member_z(sur, c_draw, al), X, keep)
            hat_best = exchange(member_z(sur, c_best, al), X, keep)
            # The structural claim, scored on its own terms (P22 -- trust the
            # structure ahead of the rate).
            cell["links_at_best_design"] = link_rows(
                member_z(sur, c_best, al), X, sur.exchanges, gids, z_true=zt_best, keep=keep
            )
            cell.update(structure_score(hat_best, e_best))
            cell.update(
                {
                    "E_true_best_draw": float(e_draw.sum()),
                    "E_true_designed": float(e_best.sum()),
                    "E_hat_best_draw": float(hat_draw.sum()),
                    # V5: did the *true* LP's interaction rate improve?
                    "true_gain": float(e_best.sum() - e_draw.sum()),
                    # `None`, not a number, when the start had no interaction at
                    # all: "infinitely better than zero" is a divide by zero, and
                    # that case is common here -- the medium decides whether an
                    # interaction exists, so a start rate of 0 is a real outcome.
                    "true_gain_rel": (
                        float((e_best.sum() - e_draw.sum()) / e_draw.sum())
                        if e_draw.sum() > 0
                        else None
                    ),
                    "start_is_zero": bool(e_draw.sum() <= 0),
                    # `None` rather than a number when the LP finds no interaction
                    # at all: a ratio against a 1e-30 floor reads 2e33 and is not a
                    # magnitude error, it is a divide by zero. That case happened.
                    "magnitude_ratio": (
                        float(hat_best.sum() / e_best.sum()) if e_best.sum() > 0 else None
                    ),
                    "true_is_zero": bool(e_best.sum() <= 0),
                }
            )
        cells.append(cell)
        LOGGER.info(
            "community %d (%d members): E draws %.4g -> designed %.4g%s",
            n, G, float(E.max()), cell["E_designed"],
            "" if not verify else f" (true {cell['E_true_designed']:.4g})",
        )

    report = {"cells": cells, "n_communities": len(cells), "verified": verify, "exploratory": True}
    if verify:
        gain = np.array([c["true_gain"] for c in cells])
        # `n_created` is the outcome `true_gain_rel` cannot express: a medium that
        # turns an interaction on where there was none.
        n_created = int(sum(c["start_is_zero"] and not c["true_is_zero"] for c in cells))

        def med(key):
            """nanmedian: a cell with no true interaction scores NaN, not zero, and
            one such cell must not turn every summary statistic into NaN."""
            v = [c[key] for c in cells if c.get(key) is not None]
            return float(np.nanmedian(v)) if v and not np.isnan(v).all() else float("nan")

        report.update(
            {
                "n_true_improved": int((gain > 0).sum()),
                "n_true_zero": int(sum(c["true_is_zero"] for c in cells)),
                "n_created": n_created,
                "median_true_gain_rel": med("true_gain_rel"),
                "median_magnitude_ratio": med("magnitude_ratio"),
                "median_precision_at_n_true": med("precision_at_n_true"),
                "median_recall": med("recall"),
                "median_rate_spearman": med("rate_spearman"),
                "median_draw_rank_spearman": med("draw_rank_spearman"),
                # V5, in this use case's terms: the designed medium must not be
                # *worse* than the draw it started from under the true LP.
                "passed": bool((gain >= 0).all()),
            }
        )
    np.savez_compressed(
        Path(out) / "media.npz",
        community=np.array([s[0] for s in saved]),
        start_draw=np.array([s[1] for s in saved]),
        c_start=np.array([s[2] for s in saved]),
        c_design=np.array([s[3] for s in saved]),
        exchanges=np.array(sur.exchanges),
    )
    (Path(out) / "interactions.json").write_text(json.dumps(report, indent=2))
    return report
