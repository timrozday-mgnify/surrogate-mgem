"""M11 / §13.3 — the smallest defined medium the whole community grows on.

    minimise  cost . c   s.t.   mu_i(c) >= mu_target_i for every member,  c_lo <= c <= c_hi

``mu_i`` is concave in ``c`` (§13.2), so each growth floor is a *concave >=
constant* constraint and the feasible set is convex; the objective is linear. This
is §9's program with the structure made explicit, and it is the headline claim.

Solved as an exact penalty — ``cost . c + rho * sum_i max(0, 1 - mu_i/target_i)``,
convex, projected onto the box, with ``rho`` raised until the answer is feasible
and a final restoration step that walks back up the violated members' gradients.
No new solver: the head's own analytic gradient and a backtracking step, the same
machinery §13.2 uses.

**P21 bites here too, and the growth designer's trust region cannot be reused.**
:func:`cfs.science.growth.trust_box` is multiplicative precisely so nothing reaches
zero, and zeroing things is this program's entire job. Measured, unrestricted: the
program plus a greedy cardinality prune takes a 273-component medium to **41** and
the true LP then grows **none** of the three members (``mu_true`` 55/70/38 -> 0/0/0)
while every surrogate floor is satisfied. That is §13.3's stated risk arriving
exactly where it was expected — the binding set — with a mechanism: §4.3 depletes
only each organism's *active subspace* and holds the background rich, so a head
asked about a background metabolite at zero is being asked a question its labels
never contained.

So the design variable is the **union of the members' active subspaces** and the
background is frozen at the rich medium (``--all-metabolites`` to lift it and
reproduce the failure). The count that comes out is therefore a minimal medium
*within the design*, not the smallest medium that exists, and it is not comparable
to the MILP's cardinality — the MILP is free over every exchange. Report both.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from cfs.science.growth import mu_true

LOGGER = logging.getLogger("cfs.science.minimal")

# Below this fraction of the rich medium a component is "not in the medium": the
# MM bound is Vmax*c/(Km+c), so 1e-6 of a band-scale concentration is no uptake.
_PRESENT = 1e-6


def mu_and_grads(
    sur, c: np.ndarray, members: list[int], cuts: list[list] | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """``(mu, dmu/dc)`` for every member at once, **calibrated**.

    §13.2 evaluates the head raw because an increasing map cannot move an argmax.
    Here the constraint is on the *level* of ``mu``, so the calibration is what
    decides feasibility and has to be in both the value and the chain rule:
    ``g(m) = a m - d0 e^{-m/beta}``, ``g' = a + (d0/beta) e^{-m/beta} > 0``.

    ``cuts[i]`` is a list of ``(mu_j, g_j, c_j)`` from true LP solves for member
    ``i``, entering as ``min_j [mu_j + g_j . (c - c_j)]`` alongside the head.
    ``mu_true`` is concave, so each tangent is an **upper** bound on it everywhere
    and requiring the tangent to clear the floor is *necessary* for the true
    constraint: the min can only remove points the truth does not admit, never
    points it does. See :func:`cut_loop`.
    """
    from cfs.surrogate import calibrate

    jnp = sur._jnp
    x = sur._x(c)
    mu, g = sur.mod.batched_value_and_grad(sur._vheads, jnp.asarray(x))
    xa = np.asarray(x[:, 0], dtype=np.float64)
    ga = np.asarray(g[:, 0], dtype=np.float64)
    dxdu = (1.0 - xa) ** 2 / np.asarray(sur.x_scale, dtype=np.float64)
    dudc = sur.km / (sur.km + c) ** 2
    scale = np.asarray(sur.mu_scale, dtype=np.float64)[:, None]
    raw = np.asarray(mu, dtype=np.float64)[:, :1]
    cal = np.asarray(sur.value_cal, dtype=np.float64)
    dcal = cal[:, 2:3] + (cal[:, 0:1] / cal[:, 1:2]) * np.exp(-raw / cal[:, 1:2])
    k = np.asarray(members)
    v = np.maximum(calibrate.apply(raw, cal)[:, 0] * scale[:, 0], 0.0)
    mu_k, g_k = v[k], (ga * dxdu * dudc * scale * dcal)[k]
    for i, per_member in enumerate(cuts or ()):
        for mu_j, g_j, c_j in per_member:
            vj = mu_j + float(g_j @ (c - c_j))
            if vj <= mu_k[i]:  # ties to the cut: it is the one that is exact
                mu_k[i], g_k[i] = vj, g_j
    return mu_k, g_k


def _penalty(mu: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, 1.0 - mu / target)


def minimise(
    sur,
    c_hi: np.ndarray,
    members: list[int],
    target: np.ndarray,
    *,
    cost: np.ndarray,
    c_lo: np.ndarray | None = None,
    iters: int = 400,
    rounds: int = 8,
    cuts: list[list] | None = None,
) -> tuple[np.ndarray, dict]:
    """Return ``(c*, info)``: the smallest medium the surrogate says all members grow on."""
    c = c_hi.astype(np.float64).copy()
    c_lo = np.zeros_like(c) if c_lo is None else c_lo.astype(np.float64)
    rho = float(cost @ c_hi)  # penalty in the objective's own units
    for _ in range(rounds):
        c = _descend(sur, c, c_lo, c_hi, members, target, cost, rho, iters, cuts)
        mu, _ = mu_and_grads(sur, c, members, cuts)
        if _penalty(mu, target).max() <= 0.0:
            break
        rho *= 4.0
    c, restored = _restore(sur, c, c_lo, c_hi, members, target, cuts)
    c, dropped = _prune(sur, c, c_lo, members, target, cost, cuts)
    mu, _ = mu_and_grads(sur, c, members, cuts)
    return c, {
        "rho": rho,
        "restored": restored,
        "dropped_by_pruning": dropped,
        "feasible_surrogate": bool(_penalty(mu, target).max() <= 1e-9),
        "worst_slack": float((mu / target - 1.0).min()),
    }


def _descend(sur, c, c_lo, c_hi, members, target, cost, rho, iters, cuts=None):
    """Projected subgradient descent on the exact penalty, backtracking step."""

    def obj(c):
        mu, g = mu_and_grads(sur, c, members, cuts)
        p = _penalty(mu, target)
        f = float(cost @ c) + rho * float((p**2).sum())
        # A *quadratic* penalty, so the objective is smooth: the exact (linear)
        # one is non-differentiable exactly at the boundary the answer sits on,
        # and a backtracking line search halts there with the cost term still
        # pushing. Quadratic under-shoots feasibility instead, which `_restore`
        # repairs and `rho` continuation tightens.
        sub = cost - 2.0 * rho * ((p > 0)[:, None] * g * p[:, None] / target[:, None]).sum(axis=0)
        return f, sub

    f, sub = obj(c)
    step = 0.1 * float(np.linalg.norm(c_hi)) / max(float(np.linalg.norm(sub)), 1e-30)
    for _ in range(iters):
        cand = np.clip(c - step * sub, c_lo, c_hi)
        f_c, sub_c = obj(cand)
        if f_c < f:
            c, f, sub = cand, f_c, sub_c
            step *= 1.5
        else:
            step *= 0.5
            if step * float(np.linalg.norm(sub)) < 1e-14:
                break
    return c


def _prune(sur, c, c_lo, members, target, cost, cuts=None):
    """Greedily zero one component at a time, keeping the drop if all floors hold.

    The convex program above minimises ``cost . c``; V6 and §9.1 are about
    *cardinality*, and the two differ — a metabolite the penalty solve has already
    pushed to 1% of its rich level costs almost nothing to keep and is still a
    component of the medium. One pass in decreasing spend, ``M`` head evaluations,
    and it is the step that makes the count comparable to the MILP's.
    """
    order = np.argsort(-cost * c)
    dropped = 0
    for m in order:
        if c[m] <= c_lo[m]:
            continue
        trial = c.copy()
        trial[m] = c_lo[m]
        mu, _ = mu_and_grads(sur, trial, members, cuts)
        if _penalty(mu, target).max() <= 0.0:
            c, dropped = trial, dropped + 1
    return c, dropped


def _restore(sur, c, c_lo, c_hi, members, target, cuts=None):
    """Walk back up the violated members' gradients until every floor is met.

    The penalty solution sits *on* the boundary and a subgradient method lands
    just inside or just outside it; an answer that misses a growth floor is not an
    answer, so feasibility is repaired rather than reported.
    """
    for n in range(200):
        mu, g = mu_and_grads(sur, c, members, cuts)
        p = _penalty(mu, target)
        if p.max() <= 0.0:
            return c, n
        d = ((p > 0)[:, None] * g / target[:, None]).sum(axis=0)
        step = 0.05 * float(np.linalg.norm(c_hi)) / max(float(np.linalg.norm(d)), 1e-30)
        c = np.clip(c + step * d, c_lo, c_hi)
    return c, 200


# --------------------------------------------------------------------------- #
# V6 — the exact MILP on the true models
# --------------------------------------------------------------------------- #


def _lp_restore(sur, models, c, c_hi, floors, max_restore: int = 8):
    """Raise components back to rich until the **true** LP meets every floor.

    `--keep-essential` pins the metabolites the LP calls essential, but it audits
    *single* knockouts, and that is blind by construction to an alternative-route
    set. Measured on a 10-member community: the design zeroed **both**
    `EX_trp__L_e` and `EX_indole_e` -- tryptophan and the precursor it is made
    from -- so neither is essential alone and the pair is lethal. `mu_true` for
    that member was 0 on all three draws.

    The repair is the §13.4 economics again: a design is **one** state, so LP
    solves are affordable where they never are along a trajectory. V6 already
    spends them to *score* the answer; this spends a few more to *fix* it.

    **Ordered by effect, not by how much each component was cut** -- that is the
    whole difference. Restoring the largest cuts first needed 29 of 46 components;
    restoring whichever single component buys the most growth found the
    synthetic-lethal partner immediately and needed **2**, at 94 LP solves.
    """
    c = c.astype(np.float64).copy()
    cand = list(np.flatnonzero(c < c_hi - 1e-12))
    restored: list[str] = []
    for _ in range(max_restore):
        mu = [mu_true(m, sur.exchanges, c) for m in models]
        short = [i for i, (v, f) in enumerate(zip(mu, floors, strict=True)) if v < f - 1e-9]
        if not short:
            break
        # The member furthest below its floor, in relative terms, sets the target.
        k = min(short, key=lambda i: mu[i] / max(floors[i], 1e-30))
        best, best_v = None, mu[k]
        for j in cand:
            if c[j] >= c_hi[j]:
                continue
            t = c.copy()
            t[j] = c_hi[j]
            v = mu_true(models[k], sur.exchanges, t)
            if v > best_v:
                best, best_v = j, v
        if best is None:
            break  # nothing single helps; report the shortfall rather than loop
        c[best] = c_hi[best]
        restored.append(sur.exchanges[best])
    return c, restored


def cut_loop(
    sur,
    models,
    c_hi: np.ndarray,
    members: list[int],
    target: np.ndarray,
    floors: list[float],
    *,
    cost: np.ndarray,
    c_lo: np.ndarray,
    rounds: int = 6,
    km_cfg=None,
) -> tuple[np.ndarray, dict, dict]:
    """Kelley cutting planes on the growth constraints: design, check, cut, repeat.

    §13.2's trust-region machinery does not transfer, because here the surrogate is
    in the **constraints** and the objective ``cost . c`` is exact. What does
    transfer is the bundle: ``mu_true_i`` is concave, so an LP tangent at ``c_j``
    satisfies ``mu_true_i(c) <= mu_ij + g_ij . (c - c_j)`` *everywhere*, and
    demanding that affine function clear the floor is therefore **necessary** for
    the true constraint. Adding cuts can only remove points the truth does not
    admit — an outer approximation of the true feasible set, tightening monotonically
    — so the cost rises toward the true minimum rather than wandering.

    Each round costs ``G`` FBAs (one per member) and excludes the design it just
    checked: the fresh cut reads ``mu_true_i(c*) < target_i`` at ``c*`` itself.

    This is the *optimality* half of the same economics :func:`_lp_restore` gives
    the *feasibility* half of. The repair raises components until the LP is happy
    and says nothing about how much it overpaid; the cut loop puts that knowledge
    into the design and keeps the program convex while doing it.

    **Measured 2026-09-06, 4 communities x 3 draws, against the same V6 the
    `--lp-repair` numbers were taken on. It is a second route to V6, not a
    replacement, and it is cell-dependent:**

    | cell | n | base | ``--lp-repair`` | ``--cuts 6`` | cuts + repair |
    | --- | --- | --- | --- | --- | --- |
    | 6 | 3 | **fail** 0.4999 | pass, 247/250/252 | **pass, 247/247/247** | pass, 247/247/247 |
    | 7 | 3 | pass, 228 | pass, 228 | pass, 228 | pass, 228 |
    | 8 | 5 | pass, 298 | pass, 298 | pass, 298 | pass, 298 |
    | 9 | 10 | **fail** -0.000 | **pass, 370/372/375** | fail, 402 | pass, 370/406/404 |

    * **Cell 6 is the win and it is the optimality claim landing**: V6 passes on
      cuts alone, with **no LP repair**, at 247 components on all three draws where
      the repair needs 250 and 252. Putting the LP's tangent *in* the program beats
      bolting a correction on afterwards.
    * **Cells 7 and 8 are the correct null** -- one round, no change, ~3 FBAs per
      member. A design that is already feasible pays only for the check.
    * **Cell 9 is a loss**: 406/404 components against the repair's 372/375. Cuts
      tighten the constraint set, and where the head was not the binding problem
      that tightening is paid for in components with nothing bought.

    Off by default (``--cuts 0``) for that reason. Use it where the design is
    feasible-but-marginal; use ``--lp-repair`` where a member can die.
    """
    from cfs.science.growth import lp_value_and_grad

    cuts: list[list] = [[] for _ in members]
    c_star, info = minimise(sur, c_hi, members, target, cost=cost, c_lo=c_lo)
    trace = []
    for r in range(rounds):
        vals, grads = [], []
        for m in models:
            v, g = lp_value_and_grad(m, sur.exchanges, c_star, km_cfg)
            vals.append(v)
            grads.append(g)
        short = [i for i, (v, f) in enumerate(zip(vals, floors, strict=True)) if v < f - 1e-9]
        trace.append(
            {
                "round": r,
                "n_short": len(short),
                "worst_true_frac": float(
                    min(v / max(f, 1e-30) for v, f in zip(vals, floors, strict=True))
                ),
                "spend": float(cost @ c_star),
                "n_components": int((c_star > _PRESENT * np.maximum(c_hi, 0.0)).sum()),
            }
        )
        if not short:
            break
        # A **dead** member carries no cut. At `mu_true = 0` the LP's duals are all
        # zero, so the tangent is the constraint `0 >= target` -- flat and
        # satisfiable nowhere. Adding it makes the model infeasible and the penalty
        # then walks the design back toward rich with no direction: measured on a
        # 10-member community, 369 -> 402 components and V6 still failing, where
        # `_lp_restore` alone passes at 0.538. That is the synthetic-lethal case,
        # and it is the repair's job, not the cut loop's.
        alive = [i for i in short if vals[i] > 1e-12 and np.linalg.norm(grads[i]) > 0.0]
        if not alive:
            trace[-1]["stopped"] = "a violated member is dead; no informative cut"
            break
        # Cut every *live* member, not only the violated ones: a tangent is valid
        # regardless, and the ones that pass now are what stop the next design
        # paying for them twice.
        for i in range(len(members)):
            if vals[i] > 1e-12 and np.linalg.norm(grads[i]) > 0.0:
                cuts[i].append((vals[i], grads[i], c_star.copy()))
        c_star, info = minimise(sur, c_hi, members, target, cost=cost, c_lo=c_lo, cuts=cuts)
    return c_star, info, {"cut_rounds": len(trace), "cut_trace": trace,
                          "n_cuts": sum(len(x) for x in cuts)}


def milp_components(model, exchanges: list[str], c_hi: np.ndarray, min_growth: float) -> int | None:
    """``cobra.medium.minimal_medium``: the exact minimum-cardinality medium.

    Per organism, so the union over members is an *upper* bound on the joint
    optimum and ``max_i`` is a lower bound — the joint MILP is a different program
    and is not worth writing to bracket a number. Returns ``None`` if the MILP
    fails (it is a MILP on a genome-scale model; it does).
    """
    import cobra

    from cfs.groundtruth.solve import apply_mm_bounds, load_km_defaults

    with model:
        apply_mm_bounds(model, dict(zip(exchanges, c_hi.tolist(), strict=True)), load_km_defaults())
        try:
            med = cobra.medium.minimal_medium(model, min_growth, minimize_components=True)
        except Exception as exc:  # noqa: BLE001 - a failed MILP is a missing number
            LOGGER.warning("minimal_medium failed: %s", exc)
            return None
    return None if med is None else int((med > 0).sum())


def knockout_audit(sur, models, c_hi, members, free) -> list[dict]:
    """Head A vs the LP on the single knockouts — the binding set, one at a time.

    This is the number M11 is blocked on and the cheapest possible statement of it:
    |free| head evaluations and |free| LPs per member. §4.3 *does* emit an
    all-but-one-depleted corner per active metabolite, so the labels contain these
    points; there are ~23 of them in 32 000 rows and an absolute MSE does not care.
    """
    mu0, _ = mu_and_grads(sur, c_hi, members)
    rows = []
    for j in np.flatnonzero(free):
        c = c_hi.copy()
        c[j] = 0.0
        if c_hi[j] <= 0.0:
            continue
        hat, _ = mu_and_grads(sur, c, members)
        true = [mu_true(m, sur.exchanges, c) for m in models]
        rows.append(
            {
                "exchange": sur.exchanges[j],
                "mu_hat_ko": hat.tolist(),
                "mu_true_ko": true,
                # The failure: the LP says essential, the head says it barely matters.
                "missed_essential": [
                    bool(t <= 1e-9 < h / max(m0, 1e-12) - 0.5)
                    for t, h, m0 in zip(true, hat.tolist(), mu0.tolist(), strict=True)
                ],
            }
        )
    return rows


def run(
    roster_path: Path,
    labels_dir: Path,
    value_dir: Path,
    out: Path,
    *,
    organisms: list[str],
    cases: int = 5,
    target_frac: float = 0.5,
    lp_repair: bool = False,
    cut_rounds: int = 0,
    seed: int = 0,
    scales: Path | None = None,
    milp: bool = True,
    all_metabolites: bool = False,
    keep_essential: bool = True,
) -> dict:
    """One case per medium draw: design a minimal medium, then V6 it."""
    import cobra

    from cfs.compose.dfba import Surrogate, community_medium
    from cfs.sampling.active_subspace import load_subspaces
    from surrogate_mgem.data import read_roster

    roster = {gm.genome_id: gm for gm in read_roster(Path(roster_path))}
    sur = Surrogate(value_dir, organisms=organisms)
    members = list(sur.members)
    models = [cobra.io.read_sbml_model(str(roster[g].model_path)) for g in organisms]
    Path(out).mkdir(parents=True, exist_ok=True)

    # The design variable: the union of the members' active subspaces, the only
    # metabolites §4.3 ever depleted. Everything else stays at the rich level.
    subs = {}
    for gid in organisms:
        subs |= load_subspaces(Path(labels_dir) / f"{gid}.subspace.json")
    active = {e for gid in organisms for e in subs[gid].active}
    free = np.array([True if all_metabolites else (e in active) for e in sur.exchanges], dtype=bool)

    rows, media = [], []
    for n in range(cases):
        c_hi = community_medium(labels_dir, organisms, sur.exchanges, seed + n, scales)
        mu_hi, _ = mu_and_grads(sur, c_hi, members)
        target = target_frac * np.maximum(mu_hi, 1e-9)
        cost = 1.0 / np.maximum(c_hi, np.max(c_hi) * 1e-12)

        audit = knockout_audit(sur, models, c_hi, members, free)
        lethal = {r["exchange"] for r in audit if min(r["mu_true_ko"]) <= 1e-9}
        # Head A cannot represent essentiality (see `knockout_audit`), so the
        # support is pinned from the models themselves: one FBA per free
        # metabolite per member, once, and it is a static property of the GEM
        # rather than anything the design has to search. Off, V6 fails outright.
        keep = np.array([e in lethal for e in sur.exchanges]) & keep_essential
        c_lo = np.where(free & ~keep, 0.0, c_hi)
        t_hi = [mu_true(m, sur.exchanges, c_hi) for m in models]
        if cut_rounds:
            c_star, info, cut_info = cut_loop(
                sur, models, c_hi, members, target,
                [target_frac * h for h in t_hi],
                cost=cost, c_lo=c_lo, rounds=cut_rounds,
            )
            info |= cut_info
        else:
            c_star, info = minimise(sur, c_hi, members, target, cost=cost, c_lo=c_lo)
        present = c_star > _PRESENT * np.maximum(c_hi, 0.0)
        # V6: does the *true* LP hit the floor on the designed medium, member by
        # member? A dropped essential shows up here as mu_true = 0.
        t_star = [mu_true(m, sur.exchanges, c_star) for m in models]
        repaired: list[str] = []
        if lp_repair:
            c_star, repaired = _lp_restore(
                sur, models, c_star, c_hi, [target_frac * h for h in t_hi]
            )
            present = c_star > _PRESENT * np.maximum(c_hi, 0.0)
            t_star = [mu_true(m, sur.exchanges, c_star) for m in models]
        met = [t >= target_frac * h - 1e-9 for t, h in zip(t_star, t_hi, strict=True)]
        milp_n = (
            [
                milp_components(m, sur.exchanges, c_hi, target_frac * h)
                for m, h in zip(models, t_hi, strict=True)
            ]
            if milp
            else []
        )
        rows.append(
            {
                "seed": seed + n,
                "n_components_rich": int((c_hi > 0).sum()),
                "n_components": int(present.sum()),
                "n_free": int(free.sum()),
                "n_free_dropped": int((free & ~present).sum()),
                "n_essential_pinned": int(keep.sum()),
                "lp_repaired": repaired,
                "milp_components": milp_n,
                "target_frac": target_frac,
                "mu_true_rich": t_hi,
                "mu_true_min": t_star,
                "n_members_met": int(sum(met)),
                "worst_true_frac": float(
                    min(t / max(h, 1e-12) for t, h in zip(t_star, t_hi, strict=True))
                ),
                **info,
            }
        )
        if n == 0:
            report_audit = audit
        media.append((c_hi, c_star))
        LOGGER.info(
            "case %d: %d -> %d components, %d/%d members met the floor under the LP",
            n,
            rows[-1]["n_components_rich"],
            rows[-1]["n_components"],
            rows[-1]["n_members_met"],
            len(members),
        )

    report = {
        "cases": rows,
        "n_cases": len(rows),
        "organisms": organisms,
        "median_components": float(np.median([r["n_components"] for r in rows])),
        "median_components_rich": float(np.median([r["n_components_rich"] for r in rows])),
        "median_free_dropped": float(np.median([r["n_free_dropped"] for r in rows])),
        "all_metabolites": all_metabolites,
        "keep_essential": keep_essential,
        # The blocker, measured: single knockouts the LP calls lethal and the head
        # does not (case 0's rich medium).
        "knockout_audit": report_audit,
        "n_missed_essential": int(sum(any(r["missed_essential"]) for r in report_audit)),
        "worst_true_frac": float(min(r["worst_true_frac"] for r in rows)),
        # V6: every member has to clear its floor under the true LP, not the head.
        "passed": bool(all(r["n_members_met"] == len(members) for r in rows)),
    }
    np.savez_compressed(
        Path(out) / "minimal_media.npz",
        c_rich=np.array([m[0] for m in media]),
        c_min=np.array([m[1] for m in media]),
        exchanges=np.array(sur.exchanges),
    )
    (Path(out) / "minimal_medium.json").write_text(json.dumps(report, indent=2))
    return report
