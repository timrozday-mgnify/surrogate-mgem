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

import itertools
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np

from cfs.science.growth import _c_of_x, project

LOGGER = logging.getLogger("cfs.science.interaction")


def subseed(*keys: int) -> int:
    """An independent int seed per key tuple.

    Additive offsets (`seed + n*1000 + d`) made seed 1's draws seed 0's shifted
    by one: three "replicate" seeds shared 63 of 64 draws, and candidate media
    at k=0 reused the draw seeds outright. Hashing the tuple removes both.
    """
    return int(np.random.SeedSequence([int(k) for k in keys]).generate_state(1)[0])


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

# §13.5's interference step: the fraction of the time-to-first-depletion the
# medium is advanced by. Read the sign of the result, not its size.
_INTERFERE_FRAC = 0.1


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


def member_mu_batch(sur, C: np.ndarray) -> np.ndarray:
    """Head A only, community members only, ``(n_members, B)``."""
    return sur.mu_batch(C)[sur.members]


def interference_deltas(mu_alone: np.ndarray, mu_joint: np.ndarray, dt) -> np.ndarray:
    """``(mu_joint - mu_alone) / (mu_alone * dt)``, and 0 where that is undefined.

    Same quantity as :func:`interference`'s ``delta_rel_per_h`` -- a relative
    growth-rate change per hour of partner activity -- but vectorised and with
    the undefined cases (a member that does not grow alone, a medium nothing
    drains) folded to 0 rather than ``None``, because this one is inside an
    optimiser. Negative is suppression.
    """
    a = np.asarray(mu_alone, dtype=np.float64)
    j = np.asarray(mu_joint, dtype=np.float64)
    ok = (a > 0) & (np.asarray(dt, dtype=np.float64) > 0)
    return np.where(ok, (j - a) / np.where(ok, a * dt, 1.0), 0.0)


def interference_losses(mu_alone, mu_joint, dt, X) -> np.ndarray:
    """``X_i (mu_alone_i - mu_joint_i) / dt``, and 0 where ``dt`` is undefined.

    The **absolute, biomass-weighted** growth-rate loss the partners impose --
    the same finite difference as :func:`interference_deltas` with the
    normalisation by the member's own ``mu`` removed. Positive is suppression.

    That normalisation is exactly what made the relative form unusable as a
    design objective (§13.5): dividing a quantity proportional to ``c`` by
    another proportional to ``c`` gives a constant, so the relative rate
    saturates at ``Vmax/Km`` and the search reaches the cap by starving. This
    form has the numerator only, so in the same limit it goes to
    ``a (G-1) Vmax c / Km -> 0``: **starving everyone scores zero**, and the
    degenerate optimum is self-eliminating rather than optimal.
    """
    a = np.asarray(mu_alone, dtype=np.float64)
    j = np.asarray(mu_joint, dtype=np.float64)
    dt = np.asarray(dt, dtype=np.float64)
    ok = dt > 0
    return np.where(ok, np.asarray(X, dtype=np.float64) * (a - j) / np.where(ok, dt, 1.0), 0.0)


def objective_interference_batch(
    sur,
    C,
    X,
    alpha: float = 1.0,
    keep=None,
    frac: float = _INTERFERE_FRAC,
    relative: bool = False,
):
    """``(total suppression per medium, per-member detail)``, positive = suppression.

    ``relative=False`` (default) is :func:`interference_losses`, the absolute
    biomass-weighted loss. ``relative=True`` is the refuted
    :func:`interference_deltas` form, kept runnable because its negative result
    is worth being able to re-derive -- see §13.5.

    The surrogate reading of :func:`interference`, so the designer can propose
    against it: step each medium by the pool derivative with and without the
    partners (:func:`interference_media`) and score the growth-rate difference.
    ``E`` cannot express this at all -- it is a ``min`` of two non-negative sums,
    so suppression is invisible to it -- which is why maximising negative
    interaction needs its own objective rather than a sign on the old one.

    Cost is ``G+1`` Head-A evaluations per medium and **no Head B** beyond the
    one ``z`` the step is built from, hence :func:`member_mu_batch`: the
    per-state active-set projection in ``mu_and_z_batch`` is a Python loop over
    the batch and this objective would pay for it ``G+1`` times over.
    """
    C = np.asarray(C, dtype=np.float64)
    B, M = C.shape
    Z = member_z_batch(sur, C, alpha)  # (G, B, M)
    G = Z.shape[0]
    media = np.empty((B, G + 1, M))
    dt = np.empty(B)
    for b in range(B):
        dt[b], media[b, 1:], media[b, 0] = interference_media(C[b], Z[:, b, :], X, frac, keep)
    mu = member_mu_batch(sur, media.reshape(B * (G + 1), M)).reshape(G, B, G + 1)
    i = np.arange(G)
    alone = mu[i, :, 1 + i].T  # (B, G): member i in its own no-partner arm
    joint = mu[i, :, 0].T
    if relative:
        d = -interference_deltas(alone, joint, dt[:, None])
    else:
        d = interference_losses(alone, joint, dt[:, None], np.asarray(X)[None])
    return d.sum(1), d


def objective_interference(
    sur,
    c,
    X,
    alpha: float = 1.0,
    keep=None,
    frac: float = _INTERFERE_FRAC,
    relative: bool = False,
):
    """:func:`objective_interference_batch` at one medium."""
    v, d = objective_interference_batch(
        sur, np.asarray(c, dtype=np.float64)[None], X, alpha, keep, frac, relative
    )
    return float(v[0]), d[0]


def grad_fd(
    sur,
    c: np.ndarray,
    X: np.ndarray,
    alpha: float = 1.0,
    rel: float = 1e-3,
    keep=None,
    obj_batch=None,
):
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
    E, _ = (obj_batch or objective_batch)(sur, C, X, alpha, keep)
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
    obj: Objective | None = None,
) -> tuple[np.ndarray, list[float]]:
    """Projected gradient ascent on ``obj``, backtracking step. Returns ``(c*, path)``.

    ``obj`` defaults to the handover rate ``E`` (:func:`objective_spec`); the
    other choice is the interference rate, which is what makes *negative*
    interaction designable at all. Both are non-concave, so the multistart and
    the LP acceptance test below apply either way.

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
    obj = obj or objective_spec("handover")
    c = project(np.asarray(c0, dtype=np.float64), cost, budget, c_lo, c_hi)
    E, g = grad_fd(sur, c, X, alpha, keep=keep, obj_batch=obj.hat_batch)
    path = [E]
    gn = max(float(np.linalg.norm(g)), 1e-30)
    step = 0.1 * float(np.linalg.norm(c_hi)) / gn
    for _ in range(iters):
        cand = project(c + step * g, cost, budget, c_lo, c_hi)
        E_c = obj.hat(sur, cand, X, alpha, keep)[0]
        if E_c > E:
            c, E = cand, E_c
            path.append(E)
            if len(path) > 2 and path[-1] - path[-2] < tol * max(abs(E), 1e-12):
                break
            E, g = grad_fd(sur, c, X, alpha, keep=keep, obj_batch=obj.hat_batch)
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
        x0 = np.asarray(sur._x(c0)[k, 0], dtype=np.float64)[: sur.n_metabolites]
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


# --------------------------------------------------------------------------- #
# Candidate seeding: enumerate the handovers, then seed one start per one
# --------------------------------------------------------------------------- #

# "Scarce" for a metabolite we are deliberately taking off the table: `u =
# c/(Km+c) ~ 1e-3`, so §3.3's uptake bound is ~0 without the medium reaching
# zero. Not zero on purpose -- a dead member has `z = 0`, so `E_true = 0`, which
# is the all-zero-start failure `--screen` exists to avoid, and it also disarms
# the acceptance test (any positive step "improves" zero).
_SCARCE = 1e-3


def candidate_links(
    labels_dir, gids: list[str], exchanges: list[str], eps: str = "0.001", rows: int = 3000
) -> dict[str, dict]:
    """Which handovers this community could reach at all, from the labels. **No LP.**

    A metabolite is a candidate when some member is *observed* to secrete it
    (``z > 0`` at any labelled medium) and another is observed to take it up.
    That is a lower bound on capability -- the capability version is one
    exchange-FVA per organism -- but it costs zero solves and it is the regime
    the heads were trained in.

    Measured on the 21-genome roster: 2 to 16 directed links per ordered pair
    (median 8), and **11-13 distinct candidate metabolites for a 2-member
    community**, 62 for the whole roster out of 444 exchanges. So the start count
    is set by the *metabolites*, not by the links -- every donor/recipient pair
    sharing a metabolite shares one start, because the design variable is the
    medium.

    Returns ``(links, donor_media, donor_box)``. ``donor_media[m]`` is the labelled
    medium at which the best donor secreted ``m`` hardest and ``donor_box[m]`` is
    the ``(lo, hi)`` envelope of **every** medium where it secreted ``m`` at all,
    both mapped to the global index -- **the secretion half of the handover, read
    off the labels for free.**

    The box is what makes the region samplable rather than a single point. Most
    candidates are secreted in **under 1% of the design's media** (AAXE02:
    acetaldehyde 74.7%, arabinose 0.1%, H2S 0.03%), and a draw has to arrange the
    donor's half and the recipient's at once -- which is why random draws realise
    about half the reachable links. Per dimension the box is 0.08-0.30 of the full
    design range, i.e. a 1e-5 to 1e-19 volume fraction, and it is *predictive*:
    inside it the secretion rate rises 1.1-1704x over the base rate, with the
    largest lift exactly on the rare metabolites sampling misses. Opening
    the recipient's uptake bound is not enough on its own: ``E = min(secretion,
    uptake)``, so a start that only lifts the uptake side is pinned by a secretion
    the design never asked for. Measured that way first: 2-5 of 10-11 candidates
    realised, and 0 of 5 cells improved.
    """
    return _pair_links(gids, exchanges, *observed(labels_dir, gids, exchanges, eps, rows))


def observed(
    labels_dir, gids: list[str], exchanges: list[str], eps: str = "0.001", rows: int = 3000
):
    """``(S, U, best)`` per genome from the label shards. **No LP.**

    ``S[g]``/``U[g]`` are the metabolites ``g`` was observed to secrete / take up
    at any labelled medium, and ``best[g]`` is
    ``[top secretion, medium at top, box lo, box hi]``. Two pairings read this:
    :func:`candidate_links` wants ``S[a] & U[b]`` (a handover), and
    :func:`shared_secretion` wants ``S[a] & S[b]`` (contention for the same
    disposal route).
    """
    import pyarrow.parquet as pq

    col = {e: i for i, e in enumerate(exchanges)}
    S, U, best = {}, {}, {}
    for g in gids:
        idx = _local_index(labels_dir, g, col)
        acc = _new_stats(len(exchanges))
        for f in sorted((Path(labels_dir) / g / f"eps_{eps}").glob("part*.parquet")):
            tab = pq.read_table(f, columns=["z", "medium"]).slice(0, rows)
            _link_stats(
                acc,
                idx,
                np.stack(tab.column("z").to_numpy(zero_copy_only=False)),
                np.stack(tab.column("medium").to_numpy(zero_copy_only=False)),
            )
        S[g], U[g], best[g] = acc[0], acc[1], acc[2:]
    return S, U, best


def _local_index(labels_dir, g: str, col: dict[str, int]) -> np.ndarray:
    """The organism's own exchange order, mapped to the global index."""
    loc = json.loads((Path(labels_dir) / f"{g}.exchanges.json").read_text())
    loc = loc["exchanges"] if isinstance(loc, dict) else loc
    return np.array([col[e] for e in loc if e in col])


def _new_stats(n: int) -> list:
    """``[secreted, taken_up, top, medium_at_top, box_lo, box_hi]``, all global."""
    return [
        np.zeros(n, bool),
        np.zeros(n, bool),
        np.zeros(n),
        np.zeros((n, n)),
        np.full((n, n), np.inf),
        np.full((n, n), -np.inf),
    ]


def _link_stats(acc: list, idx: np.ndarray, Z: np.ndarray, M: np.ndarray) -> None:
    """Fold one ``(Z, M)`` block of solved media into ``acc``, in place."""
    s, u, top, med, lo, hi = acc
    # Same threshold discipline as `data._DUAL_TOL`: half the "non-zero" entries
    # in these shards are solver dust.
    t = _LINK_TOL * float(np.abs(Z).max()) if Z.size else 0.0
    s[idx] |= (Z > t).any(0)
    u[idx] |= (Z < -t).any(0)
    r = Z.argmax(0)
    v = Z[r, np.arange(Z.shape[1])]
    better = v > top[idx]
    top[idx[better]] = v[better]
    med[np.ix_(idx[better], idx)] = M[r[better]]
    for a, j in enumerate(idx):
        sel = Z[:, a] > t
        if sel.any():
            lo[np.ix_([j], idx)] = np.minimum(lo[j, idx], M[sel].min(0))
            hi[np.ix_([j], idx)] = np.maximum(hi[j, idx], M[sel].max(0))


def _pair_links(gids, exchanges, S, U, best):
    links: dict[str, dict] = {}
    donor_media: dict[str, np.ndarray] = {}
    donor_box: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for a, b in itertools.permutations(gids, 2):
        for m in np.flatnonzero(S[a] & U[b]):
            ex = exchanges[m]
            d = links.setdefault(
                ex, {"donors": [], "recipients": [], "best_donor": None, "best_secretion": 0.0}
            )
            if a not in d["donors"]:
                d["donors"].append(a)
            if b not in d["recipients"]:
                d["recipients"].append(b)
            if best[a][0][m] > d["best_secretion"]:
                d["best_donor"], d["best_secretion"] = a, float(best[a][0][m])
                donor_media[ex] = best[a][1][m]
                donor_box[ex] = (best[a][2][m], best[a][3][m])
    return links, donor_media, donor_box


def inhibited_links(
    models: list,
    labels_dir,
    gids: list[str],
    exchanges: list[str],
    ceq: dict[str, float],
    *,
    n_media: int = 300,
    eps_str: str = "0.001",
    eps: float = 1e-3,
    seed: int = 0,
) -> tuple[dict, dict, dict]:
    """:func:`candidate_links`, re-enumerated under the **inhibited** LP.

    §13.11's option (i). The label shards are plain FBA, so an enumeration over
    them is over-inclusive under inhibition (secretions the bound forbids) and
    **under-inclusive**: stage 3' found five designs whose ``E_true`` is 0.000
    under FBA and up to 1326 under it, and no FBA enumeration can propose those.
    This re-solves a subsample of the *same* labelled media with ``ceq`` on and
    folds the result through the identical aggregation, so the donor's medium and
    its box come from the inhibited model too -- which is what
    ``--extra-candidates`` (option ii) cannot supply.

    Costs ``n_media`` solves per member -- FBA plus the elastic-net QP, since it
    is ``z`` that is wanted. No relabel: the heads and ``x_scale`` do not move.
    """
    import pyarrow.parquet as pq

    from cfs.groundtruth.solve import load_km_defaults, solve

    km_cfg = load_km_defaults()
    col = {e: i for i, e in enumerate(exchanges)}
    S, U, best = {}, {}, {}
    for model, g in zip(models, gids, strict=True):
        loc = json.loads((Path(labels_dir) / f"{g}.exchanges.json").read_text())
        loc = loc["exchanges"] if isinstance(loc, dict) else loc
        idx = _local_index(labels_dir, g, col)
        parts = sorted((Path(labels_dir) / g / f"eps_{eps_str}").glob("part*.parquet"))
        M = np.concatenate(
            [
                np.stack(
                    pq.read_table(f, columns=["medium"])
                    .column("medium")
                    .to_numpy(zero_copy_only=False)
                )
                for f in parts
            ]
        )
        take = np.random.default_rng(seed).permutation(len(M))[:n_media]
        M = M[take]
        Z = np.zeros((len(M), len(idx)))
        for r, m in enumerate(M):
            sol = solve(model, dict(zip(loc, m.tolist(), strict=False)), 1.0, eps, km_cfg, ceq)
            if sol.status != "optimal":
                continue  # P2: no growth, no fluxes -- and no link either
            for ex, v in sol.z.items():
                if ex in loc:
                    Z[r, loc.index(ex)] = v
        acc = _new_stats(len(exchanges))
        _link_stats(acc, idx, Z, M)
        S[g], U[g], best[g] = acc[0], acc[1], acc[2:]
    return _pair_links(gids, exchanges, S, U, best)


# A candidate is dropped when neither half of the handover can reach this
# fraction of its own capacity. Not a window test -- there is no hard window (see
# `target_level`) -- just a floor below which a start is not worth an LP screen.
_MIN_FEASIBLE = 1e-3


def target_level(km: float, ceq_m: float | None) -> tuple[float, float]:
    """``(concentration, achievable fraction)`` for a candidate metabolite.

    **Under §13.11 the two halves of a handover want opposite concentrations of
    the same metabolite.** The recipient's uptake bound is ``-Vmax * c/(Km+c)``,
    rising in ``c``; the donor's secretion bound is ``Vmax * max(0, 1 - c/c^eq)``,
    falling in ``c``. ``E`` is a **min** of the two, so the level that maximises
    it is where they are equal -- and that is a quadratic with a closed form:

        c/(Km+c) = 1 - c/c^eq   =>   c = (-Km + sqrt(Km^2 + 4 Km c^eq)) / 2

    which is ``sqrt(Km c^eq)`` -- the geometric centre of the window -- whenever
    ``Km << c^eq``, and stays correct where it is not. The second return value is
    the fraction of capacity **both** halves reach there.

    **This retracts "the window can be empty".** The earlier reading required
    ``c >= Km`` for the uptake half, which is not a requirement but a preference:
    uptake at ``c < Km`` is weak, not forbidden. So a handover is possible at
    *any* ``c^eq > 0``, with the achievable fraction shrinking smoothly (at
    ``c^eq = Km`` both halves reach 0.382; at ``c^eq = Km/100``, 0.0098). The
    number to report is that fraction, not a feasibility flag.

    Uninhibited (``ceq_m is None``) this is the original ``_BUFFER_SAT * Km`` at
    full uptake capacity.
    """
    sat = _BUFFER_SAT * km
    if ceq_m is None:
        return sat, sat / (km + sat)
    c = 0.5 * (-km + np.sqrt(km * km + 4.0 * km * ceq_m))
    if c >= sat:  # c^eq so high it never binds: the original level is better
        return sat, min(sat / (km + sat), max(0.0, 1.0 - sat / ceq_m))
    return float(c), float(c / (km + c))


def candidate_media(
    sur,
    labels_dir,
    gids: list[str],
    links: dict,
    donor_media: dict,
    donor_box: dict,
    keep: np.ndarray,
    seed: int,
    box: int = 3,
    scales=None,
    ceq: dict[str, float] | None = None,
    extra: int = 0,
):
    """Starts per candidate metabolite: ``(C, targets)``.

    ``E = min(total secretion, total uptake)``, so a start has to arrange **both
    halves** of the handover. Every variant sets the candidate metabolite
    saturating -- the *uptake* half, since §3.3's bound is ``-Vmax * u`` and a
    scarce metabolite caps the recipient however much the donor leaks. They differ
    in how they arrange the *secretion* half, over the best donor's own active
    subspace, which is what fixes its limitation and therefore what it overflows:

    * ``uptake`` -- not at all. Measured first and not enough: 2-5 of 10-11
      candidates realised under the true LP, and 0 of 5 cells improved.
    * ``uptake+secretion`` -- pinned to the labelled medium where that donor
      secreted the metabolite hardest. One point, so no diversity.
    * ``uptake+secretion+exclusive`` -- the same, plus every *other* candidate made
      scarce, so the targeted one is the only handover on offer. Restricted to
      metabolites in no member's active subspace, which by construction cannot
      change any member's growth rate, so it cannot starve the recipient into the
      dead state that would make the LP screen meaningless.
    * ``box`` (``box`` of them) -- drawn log-uniformly inside the envelope of
      *every* medium where the donor secreted it. The point of the box over the
      pin: the region is 1e-5 to 1e-19 of the design volume, so a §4.3 draw never
      finds it, but inside it the secretion rate is 1.1-1704x the base rate -- so
      this samples the interaction-competent region instead of one corner of it.

    The lower end of each box dimension is floored at ``_SCARCE * Km``: a labelled
    medium can have a metabolite at 0, and a log-uniform draw from there spends
    most of its range on concentrations no experiment distinguishes from absent.
    """
    from cfs.compose.dfba import community_medium
    from cfs.sampling.active_subspace import load_subspaces

    col = {e: i for i, e in enumerate(sur.exchanges)}
    cand = sorted(m for m in links if m in col and keep[col[m]])
    subs = {}
    for g in gids:
        subs |= load_subspaces(Path(labels_dir) / f"{g}.subspace.json")
    active = {e for g in gids for e in subs[g].active}
    variants = ["uptake", "uptake+secretion", "uptake+secretion+exclusive"] + ["box"] * box

    # **Capability, not observed behaviour.** `links` comes from the label
    # shards, which are plain FBA, so it cannot contain a handover that exists
    # *because of* inhibition -- and stage 3' found five designs whose `E_true`
    # is 0.000 under FBA and up to 1326 under it. Any metabolite two members
    # both exchange is a candidate on the model's own structure, with no solve
    # and no label. Appended, never substituted: §13.5 measured neither the
    # candidate set nor the draws to contain the other.
    extras: list[str] = []
    if extra > 0:
        shared = [
            e
            for j, e in enumerate(sur.exchanges)
            if keep[j] and sum(bool(sur.mask[i][j]) for i in range(len(gids))) >= 2
        ]
        pool = sorted(set(shared) - set(cand))
        extras = (
            sorted(
                np.random.default_rng(seed)
                .permutation(np.array(pool, dtype=object))[:extra]
                .tolist()
            )
            if pool
            else []
        )

    C, targets = [], []
    for k, m in enumerate(cand + extras):
        donor = links.get(m, {}).get("best_donor")
        # `None` means no concentration satisfies both halves at once under
        # §13.11's secretion bound -- a start for it cannot work, so do not
        # emit one. Without this every candidate start sits at `1000 * Km`,
        # which is >= `c^eq` for 100% of this index's exchanges at
        # `c^eq <= 1 mM`: the donor's secretion of the very metabolite the
        # start exists to hand over is pinned at exactly zero.
        level, feasible = target_level(float(sur.km[col[m]]), None if ceq is None else ceq.get(m))
        if feasible < _MIN_FEASIBLE:
            continue
        for d, var in enumerate(["analytic"] if m in extras else variants):
            rng = np.random.default_rng(subseed(seed, k, d, 0))
            spec: dict[int, float] = {}
            c = buffer_medium(
                sur,
                community_medium(labels_dir, gids, sur.exchanges, subseed(seed, k, d, 1), scales),
                keep,
            )
            dims = (
                [col[e] for e in subs[donor].active if e in col and keep[col[e]]]
                if donor is not None
                else []
            )
            if var.startswith("uptake+secretion") and m in donor_media:
                dm = donor_media[m]
                spec |= {j: dm[j] for j in dims}
            elif var == "box" and m in donor_box:
                blo, bhi = donor_box[m]
                for j in dims:
                    lo = max(float(blo[j]), _SCARCE * float(sur.km[j]))
                    hi = max(float(bhi[j]), lo)
                    spec[j] = float(10.0 ** rng.uniform(np.log10(lo), np.log10(hi)))
            spec[col[m]] = level
            if var.endswith("exclusive"):
                spec |= {
                    col[o]: _SCARCE * sur.km[col[o]] for o in cand if o != m and o not in active
                }
            for j, v in spec.items():
                c[j] = v
            C.append(c)
            targets.append(
                {
                    "metabolite": m,
                    "variant": var,
                    "donors": links.get(m, {}).get("donors"),
                    "recipients": links.get(m, {}).get("recipients"),
                    "feasible_frac": feasible,
                    "best_donor": donor,
                }
            )
    return np.array(C), targets


# Straddle the threshold. Below it the donor's own increment has to carry `c_p`
# across; at 0.99 the bound is already all but shut. Which one bites depends on
# `dt * X_j * z_jp`, which is not knowable before the medium exists -- and
# measured, the level barely matters (`EX_glyc_e` scored an identical 4093 at all
# three on one cell), so the work is done by putting the product on the `c^eq`
# scale at all, not by the fraction.
_COND_LEVELS = (0.5, 0.9, 0.99)


def shared_secretion(gids: list[str], exchanges: list[str], S: dict, best: dict) -> dict:
    """Metabolites two or more members secrete, with the hardest secretor. **No LP.**

    The handover pairing asks for ``S[a] & U[b]``; this asks for ``S[a] & S[b]``,
    which is the structure §13.11's secretion bound turns into an interaction:
    raising a product both members must excrete tightens
    ``Vmax max(0, 1 - c/ceq)`` for *both*, so one member's overflow is a cost to
    the other. Competition for **disposal** capacity rather than for a substrate,
    and it is the only mechanism by which the conditioning term can be non-zero.
    """
    out: dict[str, dict] = {}
    for m in range(len(exchanges)):
        secretors = [g for g in gids if S[g][m]]
        if len(secretors) < 2:
            continue
        donor = max(secretors, key=lambda g: best[g][0][m])
        out[exchanges[m]] = {
            "secretors": secretors,
            "best_donor": donor,
            "best_secretion": float(best[donor][0][m]),
            "medium": np.asarray(best[donor][1][m], dtype=np.float64),
        }
    return out


def conditioning_media(
    sur,
    gids: list[str],
    shared: dict,
    ceq: dict[str, float],
    keep: np.ndarray,
    levels: tuple[float, ...] = _COND_LEVELS,
):
    """Starts constructed to *have* product inhibition: ``(C, targets)``. **No LP.**

    Conditioning is a needle at random media -- 1 to 2 of 40 draws across three
    cells and two thresholds, median 0 -- and it cannot be searched for, since the
    surrogate's version of the term is identically <= 0
    (:func:`objective_spec`). But its precondition is closed form: a product two
    members both secrete, standing at ``c^eq`` scale so the bound is tight. So
    build the medium instead of looking for it -- the donor's own hardest-
    secreting labelled medium, with that product placed at a fraction of its
    equilibrium concentration.

    Measured over the five 2-member cells at ``c^eq`` = 0.1 mM: **37 of 132
    constructed media show conditioning (28%) against ~4% at random**, and per
    cell 53% / 40% / 25% / 0% / 0%. The two zeros are real -- a community can
    simply have no contended disposal route, and there the pairs are often not
    even live because a member does not grow on the donor's background. Report
    the per-cell rate; do not quote the pooled one as if it were uniform.
    """
    col = {e: i for i, e in enumerate(sur.exchanges)}
    C, targets = [], []
    for m in sorted(shared):
        j = col.get(m)
        if j is None or not keep[j] or ceq.get(m, 0.0) <= 0:
            continue
        for lv in levels:
            c = buffer_medium(sur, shared[m]["medium"].copy(), keep)
            c[j] = lv * ceq[m]
            C.append(c)
            targets.append({"metabolite": m, "variant": f"conditioning@{lv}"})
    return (np.array(C) if C else np.empty((0, len(sur.exchanges)))), targets


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
        {
            "metabolite": exchanges[i],
            "log10_fold": float(lf[i]),
            "from": float(c_ref[i]),
            "to": float(c[i]),
        }
        for i in idx
        if abs(lf[i]) > 1e-6
    ]


# --------------------------------------------------------------------------- #
# V5: the true LP's own interaction rate
# --------------------------------------------------------------------------- #


def ceq_map(spec: dict, exchanges: list[str], keep: np.ndarray) -> dict[str, float]:
    """Expand an ``--inhibition`` spec to one ``c^eq`` per exchange (§13.11 stage 2').

    ``spec`` is exchange id -> equilibrium concentration, plus an optional
    ``"default"`` applied to every other exchange. The default is not a
    convenience: P30 says an *unparameterised* reaction is modelled as infinitely
    tolerant of its own product, and a growth-maximising LP routes flux through
    exactly those, so a partial layer biases towards whatever was not measured.
    A complete-but-approximate layer is the one an optimisation model can carry;
    a 75%-coverage table is the shape P30 warns about.

    **Buffered species are excluded, and that is load-bearing.** §13.5 pins them
    at ``1e3 * Km`` to represent a solvent and a pH controller, so any finite
    ``c^eq`` puts their secretion bound at exactly zero -- the community could not
    excrete a proton or a water molecule, which is not inhibition, it is an
    infeasible model. Naming one explicitly still works; it is only the default
    that skips them.
    """
    default = spec.get("default")
    out = {}
    for ex, wanted in zip(exchanges, keep, strict=True):
        if ex in spec:
            out[ex] = float(spec[ex])
        elif default is not None and wanted:
            out[ex] = float(default)
    return out


def true_z(
    models: list,
    exchanges: list[str],
    c: np.ndarray,
    eps: float = 1e-3,
    ceq: dict[str, float] | None = None,
) -> np.ndarray:
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
        sol = solve(model, conc, 1.0, eps, km_cfg, ceq)
        if sol.status != "optimal":
            LOGGER.debug("member %d non-optimal (%s) — no growth, no fluxes (P2)", i, sol.status)
            continue
        for ex, v in sol.z.items():
            z[i, col[ex]] = v
    return z


def interference_media(
    c: np.ndarray,
    z: np.ndarray,
    X: np.ndarray,
    frac: float = _INTERFERE_FRAC,
    keep: np.ndarray | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Step the medium by the pool derivative, with and without the partners.

    Returns ``(dt, alone, joint)`` -- ``alone[i]`` is ``c`` advanced by member
    ``i``'s own exchange only, ``joint`` by the whole pool. Comparing ``mu_i``
    across the two attributes the difference to the *partners*: self-depletion
    is present in both arms and cancels, which comparing against ``mu_i(c)``
    would not do.

    ``dt`` is ``frac`` of the time to the first depletion under the pool
    derivative, so there is no arbitrary time unit -- but it is still a scale.
    Read the sign; the magnitude is a one-step linearisation.

    ``keep`` is the buffered mask (:func:`keep_mask`), and it belongs here for
    the same reason it belongs in ``E``: the vessel holds protons and water, so
    the members can neither deplete nor accumulate them. Without it they are
    ordinarily the fastest-draining entries in ``dc`` and therefore **set
    ``dt``** -- the whole step is then scaled by a species the buffer is holding
    constant.
    """
    c = np.asarray(c, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    zw = np.asarray(X, dtype=np.float64)[:, None] * z
    if keep is not None:
        zw = zw * np.asarray(keep, dtype=np.float64)[None]
    dc = zw.sum(0)
    drain = (dc < 0) & (c > 0)
    # **min, so no metabolite is ever exhausted inside the step.** The fastest
    # draining metabolite loses exactly ``frac`` of itself and every other one
    # less, which is what makes the difference a derivative rather than an
    # outcome. Measured with ``median`` instead: the step exhausts a trace
    # metabolite and a member reads ``delta_rel = -1`` -- saturated, and
    # unstable in ``frac`` (one cell's member died in its *own* arm at 0.01 and
    # lived at 0.001). Report the rate, not the step's own number.
    dt = frac * float(np.min(c[drain] / -dc[drain])) if drain.any() else 0.0
    return dt, np.clip(c + dt * zw, 0.0, None), np.clip(c + dt * dc, 0.0, None)


def _mu_at(exchanges: list[str], ceq: dict[str, float] | None = None):
    """``(model, medium) -> mu_max``: FBA only, §3.3 bounds, no elastic-net stage."""
    from cfs.groundtruth.solve import apply_mm_bounds, load_km_defaults, mu_optimize

    km_cfg = load_km_defaults()

    def mu(model, medium):
        with model:
            apply_mm_bounds(model, dict(zip(exchanges, medium.tolist(), strict=True)), km_cfg, ceq)
            return float(mu_optimize(model, "spent-medium"))

    return mu


def interference(
    models: list,
    exchanges: list[str],
    c: np.ndarray,
    z: np.ndarray,
    X: np.ndarray,
    genome_ids: list[str],
    *,
    frac: float = _INTERFERE_FRAC,
    ceq: dict[str, float] | None = None,
    keep: np.ndarray | None = None,
) -> dict:
    """Each member's ``mu`` with and without its partners, at the same medium.

    ``E = min(secretion, uptake) >= 0`` by construction, so a member's waste
    suppressing its neighbour shows up only as a *smaller* handover -- never as
    a negative link. This is the second metric §13.11's change table promised:
    a negative ``delta_rel`` is interference, a positive one facilitation, and
    both are read off the same solve.

    ``2G`` FBAs and no QP: only ``mu`` is wanted, and the elastic-net stage is
    the expensive half of :func:`cfs.groundtruth.solve.solve`.
    """
    dt, alone, joint = interference_media(c, z, X, frac, keep)
    mu = _mu_at(exchanges, ceq)

    Xa = np.asarray(X, dtype=np.float64)
    rows = []
    for i, model in enumerate(models):
        a, j = mu(model, alone[i]), mu(model, joint)
        rows.append(
            {
                "genome_id": genome_ids[i],
                "mu_alone": a,
                "mu_with_partners": j,
                # The absolute, biomass-weighted form -- what `--objective
                # interference` maximises. Positive is suppression. Unlike
                # `delta_rel_per_h` it is defined at a member that does not grow
                # alone, and it goes to 0 rather than to a constant as the
                # medium is starved (§13.5).
                "abs_loss_per_h": (float(Xa[i] * (a - j) / dt) if dt > 0 else None),
                # None, not a number: a member that does not grow alone has no
                # baseline to be suppressed relative to.
                "delta_rel": float((j - a) / a) if a > 0 else None,
                # The step-free number: a relative growth-rate change per hour of
                # partner activity. `delta_rel` alone is a property of `frac`.
                "delta_rel_per_h": float((j - a) / a / dt) if a > 0 and dt > 0 else None,
            }
        )
    deltas = [r["delta_rel_per_h"] for r in rows if r["delta_rel_per_h"] is not None]
    return {
        "dt": dt,
        "frac": frac,
        "members": rows,
        "n_interfered": int(sum(d < -1e-6 for d in deltas)),
        "n_facilitated": int(sum(d > 1e-6 for d in deltas)),
        "min_delta_rel_per_h": min(deltas) if deltas else None,
        # The scalar `--objective interference-rel` maximises. **Refuted as a
        # design objective** -- it saturates at `(4/19) Vmax/Km` and the search
        # reaches that constant by starving a trace metal (§13.5). Kept because
        # it is the bench-comparable relative number and because the negative
        # result is worth being able to re-derive.
        "total_suppression_per_h": -float(sum(deltas)) if deltas else 0.0,
        # The scalar `--objective interference` maximises: positive is
        # suppression. Summed rather than `min`, so a medium that suppresses two
        # members is worth more than one that suppresses one -- and because the
        # ascent is on a finite difference, where a `min` over members changes
        # which member it is halfway through a step.
        "total_abs_loss_per_h": float(
            sum(r["abs_loss_per_h"] for r in rows if r["abs_loss_per_h"] is not None)
        ),
    }


# `2G(G-1) + G` FBAs, so it is quadratic in the community size. Capped rather
# than made a knob: the §13.5 communities are 2-5 members and the roster cell is
# 21, where the pair set is 840 solves for a question every pair answers the same
# way. Raise it if a mid-size community ever needs it.
_MAX_PAIRS_G = 8


def spent_medium_assay(
    models: list,
    exchanges: list[str],
    c: np.ndarray,
    z: np.ndarray,
    X: np.ndarray,
    genome_ids: list[str],
    *,
    frac: float = _INTERFERE_FRAC,
    ceq: dict[str, float] | None = None,
    keep: np.ndarray | None = None,
) -> dict | None:
    """The **directional** version of :func:`interference`: one donor at a time.

    :func:`interference` is simultaneous, so a suppressed community reads as
    "this community suppresses itself" -- it cannot say *who* suppresses *whom*.
    This is the assay a bench would run instead: condition the medium with one
    donor, filter, and grow each other member in the filtrate. The conditioned
    medium is exactly :func:`interference_media`'s ``alone[j]``, and the
    recipient is absent while it is made, so the baseline is the *fresh* medium
    ``c`` -- no self-depletion arm is needed here, unlike the simultaneous case.

    **A spent medium is depleted as well as conditioned**, so a drop mixes "your
    waste inhibits me" with "you ate my substrate". The control separates them:
    re-supplement every component the donor consumed back to ``c``
    (``max(spent, c)``, which keeps what the donor secreted and restores what it
    removed) and re-solve. What is left is the conditioning-only term. Both are
    reported, as rates -- ``delta_rel`` alone is a property of ``frac``.

    Returns ``None`` above :data:`_MAX_PAIRS_G` members.
    """
    G = len(models)
    if G > _MAX_PAIRS_G:
        return None
    dt, spent, _ = interference_media(c, z, X, frac, keep)
    mu = _mu_at(exchanges, ceq)
    fresh = [mu(m, np.asarray(c, dtype=np.float64)) for m in models]

    def rate(v, base):
        return float((v - base) / base / dt) if base > 0 and dt > 0 else None

    Xa = np.asarray(X, dtype=np.float64)

    pairs = []
    for j in range(G):
        resup = np.maximum(spent[j], np.asarray(c, dtype=np.float64))
        for i in range(G):
            if i == j:
                continue
            # One solve each, reused by both readings of it: the absolute form
            # below needs the same `mu` the relative rate is built from.
            mu_resup = mu(models[i], resup)
            tot = rate(mu(models[i], spent[j]), fresh[i])
            inh = rate(mu_resup, fresh[i])
            pairs.append(
                {
                    "donor": genome_ids[j],
                    "recipient": genome_ids[i],
                    "mu_fresh": fresh[i],
                    # The whole spent-medium effect, and the half of it that
                    # survives restoring what the donor ate.
                    "total_per_h": tot,
                    "conditioning_per_h": inh,
                    "depletion_per_h": (None if tot is None or inh is None else tot - inh),
                    # The conditioning half in absolute, biomass-weighted form:
                    # positive is suppression, and unlike the relative rate it
                    # does not saturate at a model constant as the medium is
                    # starved (§13.5). This is what `--objective conditioning`
                    # sums. Defined wherever `dt` is, including at a recipient
                    # that does not grow on the fresh medium.
                    "abs_conditioning_per_h": (
                        float(Xa[i] * (fresh[i] - mu_resup) / dt) if dt > 0 else None
                    ),
                }
            )
    live = [p for p in pairs if p["conditioning_per_h"] is not None]
    return {
        "dt": dt,
        "frac": frac,
        "pairs": pairs,
        # What `--objective conditioning` maximises: the chemical half of the
        # interference, with what the donor *ate* put back so only what it
        # secreted is left. Under plain FBA this is 0 by construction -- raising
        # a concentration cannot hurt an FBA -- so it is only a target when
        # `--inhibition` gives secretion a bound to tighten.
        "total_abs_conditioning_per_h": float(
            sum(
                p["abs_conditioning_per_h"]
                for p in pairs
                if p["abs_conditioning_per_h"] is not None
            )
        ),
        "n_suppressed_by_conditioning": int(sum(p["conditioning_per_h"] < -1e-6 for p in live)),
        "n_facilitated_by_conditioning": int(sum(p["conditioning_per_h"] > 1e-6 for p in live)),
        "worst_conditioning_per_h": (min(p["conditioning_per_h"] for p in live) if live else None),
    }


@dataclass(frozen=True)
class Objective:
    """What the design maximises: the surrogate value, its batch form, the truth.

    Two are defined (:func:`objective_spec`), and they answer different
    questions rather than being two signs of one:

    * ``handover`` -- ``E = min(secretion, uptake) >= 0``, the mass actually
      passed between members. Positive interaction, and structurally incapable
      of going negative.
    * ``interference`` -- the **absolute, biomass-weighted** growth-rate loss the
      partners impose at the same medium, ``sum_i X_i (mu_alone_i - mu_joint_i)
      / dt``, positive being suppression. This is competition for a limited
      component and product inhibition together; :func:`spent_medium_assay` is
      what separates them afterwards, and under `--inhibition` the conditioning
      half has a mechanism at all.
    * ``conditioning`` -- the **chemical** half alone, from
      :func:`spent_medium_assay`: each recipient's growth loss on the donor's
      spent medium *with what the donor consumed put back*, so substrate
      competition is controlled out. This is product inhibition and nothing else,
      and it is identically 0 under plain FBA. It has **no surrogate half** --
      see :func:`objective_spec` -- so it is a target and a ranking rather than
      something to ascend, and `--seed-mode conditioning` is what constructs the
      media for it.
    * ``interference-rel`` -- the same difference normalised by each member's own
      ``mu``. **Refuted as a design objective** (§13.5): a relative rate divides
      a quantity proportional to ``c`` by another proportional to ``c``, so in
      the scarce regime it saturates at ``(4/19) Vmax/Km`` -- a model constant,
      2.105e6 on 10 of 10 runs -- and the search reaches it by starving a trace
      metal, destroying the handover on the way. Kept runnable, off by default.

    ``truth`` takes the true ``z`` the caller already solved for, so the
    handover objective costs no extra LP and the interference one costs ``2G``
    FBAs rather than another ``G`` elastic-net solves.
    """

    name: str
    #: ``None`` when the surrogate cannot represent the objective at all, in
    #: which case there is nothing to propose with and the mode is
    #: screen-and-verify over constructed starts rather than an ascent.
    hat: Callable[..., tuple[float, np.ndarray]] | None
    hat_batch: Callable[..., tuple[np.ndarray, np.ndarray]] | None
    truth: Callable[..., float]


def objective_spec(name: str, frac: float = _INTERFERE_FRAC) -> Objective:
    """The named :class:`Objective`. ``name`` is ``handover`` or ``interference``."""
    if name == "handover":
        return Objective(
            name,
            objective,
            objective_batch,
            lambda models, exchanges, c, z, X, gids, keep=None, ceq=None: float(
                exchange(z, X, keep).sum()
            ),
        )
    if name in ("interference", "interference-rel"):
        rel = name.endswith("-rel")
        key = "total_suppression_per_h" if rel else "total_abs_loss_per_h"
        return Objective(
            name,
            partial(objective_interference, frac=frac, relative=rel),
            partial(objective_interference_batch, frac=frac, relative=rel),
            lambda models, exchanges, c, z, X, gids, keep=None, ceq=None: interference(
                models, exchanges, c, z, X, gids, frac=frac, ceq=ceq, keep=keep
            )[key],
        )
    if name == "conditioning":
        # **No surrogate half, and that is measured rather than assumed.** The
        # conditioning term is `mu_i(max(spent_j, c)) - mu_i(c)`: resupplementing
        # what the donor ate leaves a medium >= `c` in every coordinate, and Head
        # A is monotone non-decreasing in every input channel by construction, so
        # its version of this is <= 0 everywhere. Measured on the designed media
        # of the five 2-member cells: 0 of 30 ordered pairs positive, 27 exactly
        # 0. Product inhibition reaches only the true LP (§13.11 leaves the heads
        # untouched), so there is no channel that could carry it -- an ascent
        # here would climb away from any conditioning at all.
        return Objective(
            name,
            None,
            None,
            lambda models, exchanges, c, z, X, gids, keep=None, ceq=None: (
                spent_medium_assay(models, exchanges, c, z, X, gids, frac=frac, ceq=ceq, keep=keep)
                or {}
            ).get("total_abs_conditioning_per_h", 0.0),
        )
    raise ValueError(f"unknown objective {name!r}")


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
    ceq: dict[str, float] | None = None,
    obj: Objective | None = None,
    genome_ids: list[str] | None = None,
) -> tuple[np.ndarray, dict]:
    """Trust-region search on ``obj`` with the **true LP as the acceptance test**.

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
    obj = obj or objective_spec("handover")
    gids = genome_ids or [f"m{i}" for i in range(len(models))]

    def truth(medium):
        z = true_z(models, sur.exchanges, medium, eps, ceq)
        return float(obj.truth(models, sur.exchanges, medium, z, X, gids, keep, ceq))

    c = np.asarray(c0, dtype=np.float64)
    best_true = truth(c)
    start_true, r, n_lp, accepted = best_true, float(decades), 1, 0
    for _ in range(max_it):
        lo, hi = trust_box(sur, c, r, keep)
        cand, _ = maximise(
            sur,
            c,
            X,
            alpha,
            cost=cost,
            budget=budget,
            c_lo=lo,
            c_hi=hi,
            iters=iters,
            keep=keep,
            obj=obj,
        )
        e_true = truth(cand)
        n_lp += 1
        if e_true > best_true:
            c, best_true, accepted = cand, e_true, accepted + 1
            r = min(r * 1.5, 2.0)
        else:
            r *= 0.5
            if r < 1e-3:
                break
    return c, {
        "obj_true_start": start_true,
        "obj_true": best_true,
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
    verify_steps: int = 8,
    buffered: tuple[str, ...] = _BUFFERED,
    screen: bool = True,
    seed_mode: str = "candidate",
    box: int = 3,
    inhibition: Path | None = None,
    extra_candidates: int = 0,
    inhibited_media: int = 0,
    objective_name: str = "handover",
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

    # What the design maximises. The survey, the candidate enumeration and the
    # structural report are unchanged either way -- only the ascent's objective,
    # the LP screen's ranking and the acceptance test move. `handover` is `E`;
    # `interference` is the growth-rate cost partners impose, which is the
    # competition-and-inhibition half `E` cannot express by construction.
    spec = objective_spec(objective_name)

    # §13.11: exchange id -> equilibrium concentration, in the medium's own units.
    # It reaches only the *true* LP -- the heads are unchanged and the labels are
    # not relabelled, so "FBA" and "FBA + product inhibition" differ in exactly
    # one place: what the acceptance test and the screen consider possible.
    ceq_spec = json.loads(Path(inhibition).read_text()) if inhibition else None
    ceq = None  # expanded per community, once `sur.exchanges` and `keep` exist

    Path(out).mkdir(parents=True, exist_ok=True)
    if verify:
        import cobra

        from surrogate_mgem.data import read_roster

        roster = {gm.genome_id: gm for gm in read_roster(Path(roster_path))}

    cells, saved, pools = [], [], []
    for n, gids in enumerate(communities):
        sur = Surrogate(value_dir, behaviour_dir, organisms=gids)
        # Loaded per community and dropped, not cached across them: a CarveMe GEM
        # is large in memory and holding every genome the run has ever touched is
        # what killed a 5-community run with no traceback. Re-reading the SBML is
        # seconds; the cache was not worth an OOM.
        # Freed *before* the next community's are read, not by the rebinding: a
        # list comprehension builds its result before it is assigned, so the
        # previous community's GEMs are still referenced while the new ones load
        # and the peak is two communities' worth. That is what killed three
        # 5-community runs here, each mid-SBML-load with no traceback -- the same
        # signature as the models cache this comment used to be about.
        models = []
        if verify:
            models = [cobra.io.read_sbml_model(str(roster[g].model_path)) for g in gids]
        G = len(gids)
        X = np.ones(G)  # uniform reference abundance: E is then per unit biomass
        al = alpha
        keep = keep_mask(sur.exchanges, buffered)
        ceq = None if ceq_spec is None else ceq_map(ceq_spec, sur.exchanges, keep)
        # `draws`: random §4.3 media, and whether one contains a handover is luck.
        # `candidate`: one start per metabolite the labels say *could* be handed
        # over, with that metabolite's uptake bound opened -- so the multistart
        # covers every reachable link by construction instead of by sampling.
        # Enumerated in both modes: it costs no LP, and it is what makes the two
        # comparable -- "which reachable handovers did this seeding actually
        # realise" is the question, and a random draw has to be scored on it too.
        links, donor_media, donor_box = candidate_links(labels_dir, gids, sur.exchanges)
        if ceq and inhibited_media and verify:
            # §13.11(i): re-enumerate under the inhibited LP, so the candidate
            # set and the donor's own recipe come from the model the acceptance
            # test uses. **Unioned, not substituted** -- §13.5's measured rule,
            # and measured again here: substituting raises the realised handover
            # count 5 -> 8 and *drops* the designed rate 641 -> 605 (200 media)
            # and 383 (800), because the inhibited pass re-picks `best_donor`
            # from a secretion ranking the bound has compressed. The union keeps
            # the FBA recipe where there is one and adds the links only the
            # inhibited model has.
            ilinks, imedia, ibox = inhibited_links(
                models,
                labels_dir,
                gids,
                sur.exchanges,
                ceq,
                n_media=inhibited_media,
                seed=subseed(seed, n, 3),
            )
            fresh = [m for m in ilinks if m not in links]
            links = {**links, **{m: ilinks[m] for m in fresh}}
            donor_media |= {m: imedia[m] for m in fresh if m in imedia}
            donor_box |= {m: ibox[m] for m in fresh if m in ibox}
            LOGGER.info(
                "community %d: %d FBA candidates + %d inhibition-only = %d",
                n,
                len(links) - len(fresh),
                len(fresh),
                len(links),
            )
        C = np.array(
            [
                buffer_medium(
                    sur,
                    community_medium(
                        labels_dir, gids, sur.exchanges, subseed(seed, n, 0, d), scales
                    ),
                    keep,
                )
                for d in range(draws)
            ]
        )
        targets = None
        if seed_mode == "conditioning":
            # Constructed, not searched: the conditioning term has no surrogate
            # half to ascend and a ~4% base rate at random draws, but its
            # precondition -- a product two members both secrete, standing at
            # `c^eq` scale -- is closed form. Appended to the draws for the same
            # reason candidate seeding is: neither set contains the other.
            if not ceq:
                raise ValueError(
                    "--seed-mode conditioning needs --inhibition: without a "
                    "secretion bound to tighten, raising a product cannot hurt "
                    "anyone and the conditioning term is identically 0."
                )
            _S, _U, _best = observed(Path(labels_dir), gids, sur.exchanges)
            shared = shared_secretion(gids, sur.exchanges, _S, _best)
            Cc, targets = conditioning_media(sur, gids, shared, ceq, keep)
            LOGGER.info(
                "community %d: %d contended disposal routes -> %d constructed starts",
                n,
                len(shared),
                len(Cc),
            )
            targets = [{"metabolite": None, "variant": "draw"}] * len(C) + targets
            C = np.concatenate([C, Cc]) if len(Cc) else C
        if seed_mode == "candidate":
            # **Appended, not substituted.** Measured over five 2-member cells:
            # the candidate media realise 24 of 52 reachable handovers and the
            # draws 22, with only 20 in common -- 2 links are draws-only and 4
            # candidate-only, so neither set contains the other. A candidate start
            # fixes the *donor's* limitation to the medium that made it secrete in
            # isolation; a draw can land on a joint condition neither member
            # reaches from its own recipe. The draws also still win the raw rate
            # on 4 of 5 cells.
            Cc, targets = candidate_media(
                sur,
                labels_dir,
                gids,
                links,
                donor_media,
                donor_box,
                keep,
                subseed(seed, n, 1),
                box=box,
                scales=scales,
                ceq=ceq,
                extra=extra_candidates,
            )
            targets = [{"metabolite": None, "variant": "draw"}] * len(C) + targets
            C = np.concatenate([C, Cc])
        # NOT `draws`: that is the per-community parameter, and reassigning it
        # leaked the appended count into the *next* community's draw loop -- cell
        # 2 drew 130 media where cell 1 drew 64, which reads as the seeding
        # getting better with position.
        n_media = len(C)
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
        rank_rho = obj_rho = float("nan")
        if verify and screen:
            # Keep the per-metabolite vector, not just its sum: the sum ranks
            # the starts, the vector says which handovers were realised at all.
            Zt = [true_z(models, sur.exchanges, c, ceq=ceq) for c in C]
            EX_true = np.array([exchange(z, X, keep) for z in Zt])
            E_true_draws = EX_true.sum(1)
            # Rank the starts on the objective being designed. In `handover`
            # mode this *is* `E_true_draws`; in `interference` mode it is 2G
            # FBAs per draw on top of the solves already done, and ranking by
            # the handover rate instead would seed the suppression search at the
            # media with the most cross-feeding -- the wrong end.
            obj_true_draws = (
                E_true_draws
                if spec.name == "handover"
                else np.array(
                    [
                        spec.truth(models, sur.exchanges, c, z, X, gids, keep, ceq)
                        for c, z in zip(C, Zt, strict=True)
                    ]
                )
            )
            Z_pool = np.array(Zt)  # (media, G, M): kept for the report's survey
            del Zt
            best = np.argsort(-obj_true_draws)[:starts]
            obj_hat_draws = (
                None
                if spec.hat_batch is None
                else E
                if spec.name == "handover"
                else spec.hat_batch(sur, C, X, al, keep)[0]
            )
            if len(E_true_draws) > 2 and E_true_draws.std() > 0 and E.std() > 0:
                from scipy.stats import spearmanr

                rank_rho = float(spearmanr(E, E_true_draws).statistic)
            if (
                obj_hat_draws is not None
                and len(obj_true_draws) > 2
                and obj_true_draws.std() > 0
                and np.std(obj_hat_draws) > 0
            ):
                from scipy.stats import spearmanr

                obj_rho = float(spearmanr(obj_hat_draws, obj_true_draws).statistic)
        else:
            EX_true = E_true_draws = obj_true_draws = Z_pool = None
            if spec.hat_batch is None:
                raise ValueError(
                    f"--objective {spec.name} has no surrogate half, so it cannot "
                    "rank media without the LP screen. Drop --no-screen."
                )
            best = np.argsort(
                -(E if spec.name == "handover" else spec.hat_batch(sur, C, X, al, keep)[0])
            )[:starts]
        # Every screened medium, so a report can show what the starts were made of
        # and which handovers the LP found at each -- not only the few refined.
        tg = targets or [{"metabolite": None, "variant": "draw"}] * len(C)
        pools.append((n, C, tg, EX_true, Z_pool))

        # Coverage: of the metabolites the labels say could be handed over, how
        # many have a start the *LP* calls interactive. The point of candidate
        # seeding, and it needs the screen to be measurable.
        cand_cov = None
        if EX_true is not None:
            col = {e: i for i, e in enumerate(sur.exchanges)}
            cand = sorted(m for m in links if m in col and keep[col[m]])
            hit = [m for m in cand if EX_true[:, col[m]].max() > 1e-6]
            cand_cov = {
                "n_candidates": len(cand),
                "n_true_interactive": len(hit),
                "interactive": hit,
                "missed": [m for m in cand if m not in hit],
                # Did the "make every other candidate scarce" variant earn its place?
                # Which variant the LP's own best start came from -- the only
                # way to tell whether the secretion half earned its screens.
                "best_variant": (
                    None if targets is None else targets[int(np.argmax(E_true_draws))]["variant"]
                ),
                # Per variant: the best rate and the most handovers *at one
                # medium*. A combination start is not scored by whether its last
                # merged metabolite worked -- the point of it is simultaneity.
                "by_variant": (
                    None
                    if targets is None
                    else {
                        v: {
                            "n_media": int(sel.sum()),
                            "E_true_max": float(E_true_draws[sel].max()),
                            "max_links_at_one_medium": int((EX_true[sel] > 1e-6).sum(1).max()),
                            # Which handovers this variant actually realised --
                            # the question the `analytic` arm exists for, since
                            # a link outside `cand` is one the FBA labels could
                            # not have proposed.
                            "interactive": [
                                sur.exchanges[j]
                                for j in np.flatnonzero((EX_true[sel] > 1e-6).any(0))
                            ],
                        }
                        for v in dict.fromkeys(t["variant"] for t in targets)
                        for sel in [np.array([t["variant"] == v for t in targets])]
                    }
                ),
            }
        designs = []
        for s in best:
            c0 = C[s]
            lo, hi = trust_box(sur, c0, trust_decades, keep)
            cost = np.ones_like(c0)
            budget = budget_mult * float(cost @ c0)
            if spec.hat is None:
                # Nothing to propose with -- the surrogate cannot represent this
                # objective at all (`objective_spec`), so the "design" is the
                # screened start and the work was done by the seeding. Reported
                # through the same path so every downstream key still exists.
                zt0 = true_z(models, sur.exchanges, c0, ceq=ceq)
                t0 = float(spec.truth(models, sur.exchanges, c0, zt0, X, gids, keep, ceq))
                c_star = c0
                extra = {
                    "iterations": 0,
                    "obj_true_start": t0,
                    "obj_true": t0,
                    "no_surrogate_half": True,
                }
            elif verify_steps:
                c_star, extra = verified_ascent(
                    sur,
                    models,
                    c0,
                    X,
                    al,
                    cost=cost,
                    budget=budget,
                    decades=trust_decades,
                    max_it=verify_steps,
                    iters=iters,
                    keep=keep,
                    ceq=ceq,
                    obj=spec,
                    genome_ids=gids,
                )
                # In `handover` mode the design objective *is* E, so the older
                # key names still mean what every report on disk says they mean.
                if spec.name == "handover":
                    extra["E_true"] = extra["obj_true"]
            else:
                c_star, path = maximise(
                    sur,
                    c0,
                    X,
                    al,
                    cost=cost,
                    budget=budget,
                    c_lo=lo,
                    c_hi=hi,
                    iters=iters,
                    keep=keep,
                    obj=spec,
                )
                extra = {"iterations": len(path) - 1}
            E_star, _ = objective(sur, c_star, X, al, keep)
            designs.append(
                {
                    "start_draw": int(s),
                    "target": None if targets is None else targets[int(s)],
                    "E_start": float(E[s]),
                    "E_hat": E_star,
                    "obj_hat": (
                        E_star
                        if spec.name == "handover"
                        else None
                        if spec.hat is None
                        else float(spec.hat(sur, c_star, X, al, keep)[0])
                    ),
                    "changed": distinguishing(c_star, c0, sur.exchanges),
                    **extra,
                }
            )
            saved.append((n, int(s), c0, c_star))

        # Rank on the truth where we have it. Ranking multistarts by `E_hat` is
        # ranking them by how optimistic the head is at each -- the exact quantity
        # the search is exploiting.
        key = "obj_true" if (verify_steps or spec.hat is None) else "obj_hat"
        top = int(np.argmax([d[key] for d in designs]))
        c_best = saved[-len(designs) + top][3]
        c_draw = C[int(best[0])]

        cell = {
            "community": gids,
            "seed_mode": seed_mode,
            "n_candidate_metabolites": len(links),
            "candidates": (
                None
                if targets is None
                else sorted({t["metabolite"] for t in targets if t["metabolite"]})
            ),
            # Per candidate metabolite: did any of its starts have a true
            # interaction? This is the number the seeding is for -- coverage of
            # the reachable links, which a random draw gets by luck.
            "candidates_true_interactive": cand_cov,
            "n_draws": n_media,
            "E_draws_median": float(np.median(E)),
            "E_draws_max": float(E.max()),
            # Whether the head can rank media for this objective at all. If this
            # is ~0 the survey's *ordering* is not usable and only its structure is.
            "draw_rank_spearman": rank_rho,
            # The same question for the objective actually being designed:
            # can the head order media for it? Equal to `draw_rank_spearman`
            # in `handover` mode.
            "objective": spec.name,
            "obj_rank_spearman": obj_rho,
            "E_true_draws_max": (float(E_true_draws.max()) if E_true_draws is not None else None),
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
            zt_draw = true_z(models, sur.exchanges, c_draw, ceq=ceq)
            zt_best = true_z(models, sur.exchanges, c_best, ceq=ceq)
            # The interference observable: `E` cannot go negative, so suppression
            # is invisible to it. 2G FBAs at the designed medium.
            cell["interference"] = interference(
                models, sur.exchanges, c_best, zt_best, X, gids, ceq=ceq, keep=keep
            )
            # ...and the directional form: which member's waste suppresses which,
            # with the donor's depletion controlled for. `None` on a large cell.
            cell["spent_medium"] = spent_medium_assay(
                models, sur.exchanges, c_best, zt_best, X, gids, ceq=ceq, keep=keep
            )
            e_draw, e_best = exchange(zt_draw, X, keep), exchange(zt_best, X, keep)
            hat_draw = exchange(member_z(sur, c_draw, al), X, keep)
            hat_best = exchange(member_z(sur, c_best, al), X, keep)
            # The structural claim, scored on its own terms (P22 -- trust the
            # structure ahead of the rate).
            cell["links_at_best_design"] = link_rows(
                member_z(sur, c_best, al), X, sur.exchanges, gids, z_true=zt_best, keep=keep
            )
            cell.update(structure_score(hat_best, e_best))
            if ceq:
                # §13.11's actual deliverable: which handovers exist *only*
                # because of inhibition, and which it destroyed. Same medium,
                # same abundances, only the bound differing -- the arms' own
                # optima are not comparable, since each designed its own medium.
                # G solves.
                e_fba = exchange(true_z(models, sur.exchanges, c_best, ceq=None), X, keep)
                t = _LINK_TOL * max(float(e_best.max()), float(e_fba.max()), 1e-30)
                cell.update(
                    {
                        "E_true_designed_under_fba": float(e_fba.sum()),
                        "inhibition_only_links": [
                            sur.exchanges[j] for j in np.flatnonzero((e_best > t) & (e_fba <= t))
                        ],
                        "inhibition_suppressed_links": [
                            sur.exchanges[j] for j in np.flatnonzero((e_fba > t) & (e_best <= t))
                        ],
                    }
                )
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
                    # V5 in the designed objective's own terms. Equal to the
                    # E_* pair above in `handover` mode; in `interference` mode
                    # the E_* keys are still the handover rate at the same
                    # medium, which is the structural half of the report.
                    "obj_true_best_draw": float(
                        spec.truth(models, sur.exchanges, c_draw, zt_draw, X, gids, keep, ceq)
                    ),
                    # Not reused from `cell["interference"]` above, even though
                    # that is the same 2G FBAs at the same medium: naming the key
                    # here duplicated `Objective.truth`'s choice of it, and the
                    # duplicate went stale the moment a second interference
                    # objective existed -- V5 was then stated on the *relative*
                    # rate while the search maximised the absolute loss. The
                    # saving was 2G mu-only FBAs; the cost was the one number the
                    # gate is read from.
                    "obj_true_designed": float(
                        spec.truth(models, sur.exchanges, c_best, zt_best, X, gids, keep, ceq)
                    ),
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
            n,
            G,
            float(E.max()),
            cell["E_designed"],
            "" if not verify else f" (true {cell['E_true_designed']:.4g})",
        )

    report = {
        "cells": cells,
        "n_communities": len(cells),
        "verified": verify,
        "exploratory": True,
        # Which model the truth was taken under, so an FBA-only run and an
        # inhibited one are never confused in a directory of reports.
        "inhibition": None if ceq_spec is None else str(inhibition),
        "n_inhibited_exchanges": 0 if ceq is None else len(ceq),
        "objective": spec.name,
    }
    if verify:
        for c in cells:
            c["obj_gain"] = c["obj_true_designed"] - c["obj_true_best_draw"]
        gain = np.array([c["true_gain"] for c in cells])
        obj_gain = np.array([c["obj_gain"] for c in cells])
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
                "median_obj_rank_spearman": med("obj_rank_spearman"),
                "n_obj_improved": int((obj_gain > 0).sum()),
                "n_cells_with_inhibition_only_links": int(
                    sum(bool(c.get("inhibition_only_links")) for c in cells)
                ),
                # V5, in this use case's terms: the designed medium must not be
                # *worse* than the draw it started from under the true LP --
                # measured on the objective that was actually designed.
                #
                # **Relative, not `>= 0`.** An objective with no surrogate half
                # (`--objective conditioning`) has no ascent, so the design *is*
                # the top-ranked start and the two sides of this comparison are
                # the same medium solved twice. Exact arithmetic gives 0 and the
                # LP gives whatever it gives: measured at -1.2e-05 on 2.03e+05,
                # a relative 6e-11, which failed the gate and exited 1. A
                # "not worse" test between two solves of one medium needs a
                # tolerance, and 1e-6 relative is far below any real regression.
                "passed": bool(
                    (
                        obj_gain
                        >= -1e-6
                        * np.maximum(
                            np.abs([c["obj_true_designed"] for c in cells]),
                            np.abs([c["obj_true_best_draw"] for c in cells]),
                        )
                    ).all()
                ),
            }
        )
    np.savez_compressed(
        Path(out) / "media.npz",
        community=np.array([s[0] for s in saved]),
        start_draw=np.array([s[1] for s in saved]),
        c_start=np.array([s[2] for s in saved]),
        c_design=np.array([s[3] for s in saved]),
        exchanges=np.array(sur.exchanges),
        # the screened pool: draws + constructed starts, per community, with the
        # true-LP per-metabolite handover rate (NaN when the screen was off)
        pool_community=np.concatenate([np.full(len(p[1]), p[0]) for p in pools]),
        pool_c=np.concatenate([p[1] for p in pools]),
        pool_variant=np.array([t["variant"] for p in pools for t in p[2]]),
        pool_metabolite=np.array([str(t.get("metabolite") or "") for p in pools for t in p[2]]),
        pool_ex_true=np.concatenate(
            [np.full(p[1].shape, np.nan) if p[3] is None else p[3] for p in pools]
        ),
        # per-member true-LP z at every screened medium, one key per community
        # because communities can differ in size
        **{f"pool_z_true_{p[0]}": p[4] for p in pools if p[4] is not None},
    )
    (Path(out) / "interactions.json").write_text(json.dumps(report, indent=2))
    return report
