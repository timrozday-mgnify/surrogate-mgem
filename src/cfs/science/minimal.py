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


def mu_and_grads(sur, c: np.ndarray, members: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """``(mu, dmu/dc)`` for every member at once, **calibrated**.

    §13.2 evaluates the head raw because an increasing map cannot move an argmax.
    Here the constraint is on the *level* of ``mu``, so the calibration is what
    decides feasibility and has to be in both the value and the chain rule:
    ``g(m) = a m - d0 e^{-m/beta}``, ``g' = a + (d0/beta) e^{-m/beta} > 0``.
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
    return v[k], (ga * dxdu * dudc * scale * dcal)[k]


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
) -> tuple[np.ndarray, dict]:
    """Return ``(c*, info)``: the smallest medium the surrogate says all members grow on."""
    c = c_hi.astype(np.float64).copy()
    c_lo = np.zeros_like(c) if c_lo is None else c_lo.astype(np.float64)
    rho = float(cost @ c_hi)  # penalty in the objective's own units
    for _ in range(rounds):
        c = _descend(sur, c, c_lo, c_hi, members, target, cost, rho, iters)
        mu, _ = mu_and_grads(sur, c, members)
        if _penalty(mu, target).max() <= 0.0:
            break
        rho *= 4.0
    c, restored = _restore(sur, c, c_lo, c_hi, members, target)
    c, dropped = _prune(sur, c, c_lo, members, target, cost)
    mu, _ = mu_and_grads(sur, c, members)
    return c, {
        "rho": rho,
        "restored": restored,
        "dropped_by_pruning": dropped,
        "feasible_surrogate": bool(_penalty(mu, target).max() <= 1e-9),
        "worst_slack": float((mu / target - 1.0).min()),
    }


def _descend(sur, c, c_lo, c_hi, members, target, cost, rho, iters):
    """Projected subgradient descent on the exact penalty, backtracking step."""

    def obj(c):
        mu, g = mu_and_grads(sur, c, members)
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


def _prune(sur, c, c_lo, members, target, cost):
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
        mu, _ = mu_and_grads(sur, trial, members)
        if _penalty(mu, target).max() <= 0.0:
            c, dropped = trial, dropped + 1
    return c, dropped


def _restore(sur, c, c_lo, c_hi, members, target):
    """Walk back up the violated members' gradients until every floor is met.

    The penalty solution sits *on* the boundary and a subgradient method lands
    just inside or just outside it; an answer that misses a growth floor is not an
    answer, so feasibility is repaired rather than reported.
    """
    for n in range(200):
        mu, g = mu_and_grads(sur, c, members)
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
        c_star, info = minimise(sur, c_hi, members, target, cost=cost, c_lo=c_lo)
        present = c_star > _PRESENT * np.maximum(c_hi, 0.0)
        # V6: does the *true* LP hit the floor on the designed medium, member by
        # member? A dropped essential shows up here as mu_true = 0.
        t_hi = [mu_true(m, sur.exchanges, c_hi) for m in models]
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
