"""§4.2 active-subspace reduction — each organism's sensitive metabolites.

20k samples in a 60-D box is accurate nowhere in particular (P12). Restricting
the design to the ~15-30 metabolites whose depletion actually moves ``mu_max``
(the active subspace A_i) is what makes the budget viable. Procedure per
organism (plan §4.2):

1. Solve on a rich medium (all uptakes saturated); record ``mu_max``.
2. One-at-a-time: deplete each uptake metabolite, re-solve ``mu_max``.
3. ``A_i`` = metabolites whose depletion drops ``mu_max`` by more than ``tol``
   (relative). Expect 15-30. The rest are held rich in the design.

``mu_max``-only (no elastic-net QP), so this is cheap: one LP per uptake
exchange. COBRApy only; imports of the solver stack are function-local.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

LOGGER = logging.getLogger("cfs.active_subspace")

# Rich concentration: >> every Km default (all <= 0.1 mmol/L), so MM saturates
# (uptake bound ~ -Vmax). Depleted = 0 -> no uptake (mm_lower_bound returns 0).
_C_RICH = 1e3


@dataclass(frozen=True)
class ActiveSubspace:
    """The sampled (``active``) vs. held-rich (``background``) uptake exchanges."""

    genome_id: str
    active: list[str]  # A_i: uptake exchanges whose depletion moves mu_max
    background: list[str]  # the other uptake exchanges (held rich in the design)
    sensitivity: dict[str, float]  # exchange -> relative mu_max drop on depletion
    mu_rich: float


def _uptake_exchanges(model) -> list[str]:
    return [ex.id for ex in model.exchanges if ex.lower_bound < 0]


def active_subspace(
    model, genome_id: str = "", km_cfg: dict | None = None, *, tol: float = 1e-3
) -> ActiveSubspace:
    """Coarse one-at-a-time sensitivity sweep for one model (plan §4.2)."""
    from cfs.groundtruth.solve import apply_mm_bounds, load_km_defaults, mu_optimize

    km_cfg = km_cfg if km_cfg is not None else load_km_defaults()
    uptakes = _uptake_exchanges(model)
    rich = dict.fromkeys(uptakes, _C_RICH)

    with model:
        apply_mm_bounds(model, rich, km_cfg)
        mu_rich = mu_optimize(model, f"{genome_id} rich")
    if mu_rich <= 0.0:
        LOGGER.warning("%s: no growth on rich medium; active subspace empty", genome_id)
        return ActiveSubspace(genome_id, [], sorted(uptakes), dict.fromkeys(uptakes, 0.0), 0.0)

    sensitivity = {}
    for ex in uptakes:
        medium = {**rich, ex: 0.0}  # deplete this one, keep the rest rich
        with model:
            apply_mm_bounds(model, medium, km_cfg)
            mu = mu_optimize(model, f"{genome_id} -{ex}")
        sensitivity[ex] = (mu_rich - mu) / mu_rich

    active = sorted(ex for ex, s in sensitivity.items() if s > tol)
    active_set = set(active)
    background = sorted(ex for ex in uptakes if ex not in active_set)
    LOGGER.info("%s: |A_i|=%d active of %d uptakes", genome_id, len(active), len(uptakes))
    return ActiveSubspace(genome_id, active, background, sensitivity, float(mu_rich))


def demand_probe(
    model,
    exchanges: list[str],
    km_cfg: dict | None = None,
    *,
    lo: float = -4.0,
    hi: float = 1.0,
    steps: int = 12,
    target_frac: float = 0.1,
    tol: float = 1e-3,
) -> dict[str, float]:
    """Where each metabolite starts limiting, in ``c/Km`` — before any labels (§4.7).

    This is what makes the design self-anchoring. ``limiting_scales`` reads the
    same quantity off a *previous run's* labels, so a genome the roster has never
    seen falls back to scale 1.0 and its first pass carries the ~1137:9:1 coverage
    skew that holds the M3 gate down. A bisection is a few LPs per metabolite
    against the ~32 000 solves its labels cost.

    Every other uptake is held at ``Km * 10**hi`` — the *design's* rich level, not
    ``active_subspace``'s absolute ``_C_RICH``, so the probe measures the geometry
    the focus strata will actually sample. Bisects ``log10(c/Km)`` for the point
    where ``mu_max`` has recovered ``target_frac`` of that metabolite's *own*
    range, so a metabolite that only costs 5% of growth is anchored as well as an
    essential one. ``mu_max`` is non-decreasing in an uptake bound (relaxing it
    only enlarges the LP's feasible set), so bisection is sound.

    ``target_frac`` is calibrated, not chosen: measured against the ``u*`` anchors
    the second run produced (4 organisms, 64 shared metabolites), the median
    ``log10`` difference is -0.15 at 0.05, **+0.09 at 0.1**, +0.46 at 0.25 and
    +0.77 at the range midpoint. 0.1 is the fraction at which a probe with no
    labels lands where a full labelled run measured; the onset of limitation sits
    well below the midpoint because the MM ramp is concave.

    Metabolites that never limit inside ``[lo, hi]`` are **absent** from the
    result rather than defaulted — that is the caller's fallback chain to decide
    (:func:`cfs.sampling.design.band_scales`).
    """
    from cfs.groundtruth.solve import (
        apply_mm_bounds,
        km_for_exchange,
        load_km_defaults,
        mu_optimize,
    )

    km_cfg = km_cfg if km_cfg is not None else load_km_defaults()
    uptakes = _uptake_exchanges(model)
    km = {ex: km_for_exchange(ex, km_cfg) for ex in uptakes}
    rich = {ex: km[ex] * 10.0**hi for ex in uptakes}

    def mu_at(ex: str, a: float) -> float:
        with model:
            apply_mm_bounds(model, {**rich, ex: km[ex] * 10.0**a}, km_cfg)
            return mu_optimize(model, f"probe {ex}@{a:.2f}")

    out = {}
    for ex in exchanges:
        if ex not in km:  # not an uptake exchange on this model
            continue
        mu_hi, mu_lo = mu_at(ex, hi), mu_at(ex, lo)
        if mu_hi <= 0.0 or (mu_hi - mu_lo) <= tol * mu_hi:
            continue  # never limits inside the band: no probe result
        target = mu_lo + target_frac * (mu_hi - mu_lo)
        a_lo, a_hi = lo, hi  # mu(a_lo) < target <= mu(a_hi)
        for _ in range(steps):
            mid = 0.5 * (a_lo + a_hi)
            if mu_at(ex, mid) < target:
                a_lo = mid
            else:
                a_hi = mid
        out[ex] = float(10.0 ** (0.5 * (a_lo + a_hi)))
    LOGGER.info("demand probe: %d/%d metabolites anchored", len(out), len(exchanges))
    return out


def write_subspaces(
    subspaces: list[ActiveSubspace], path: Path, bands: dict[str, dict] | None = None
) -> None:
    """Write per-organism active subspaces to one JSON (fed to ``generate``).

    ``bands`` (``{genome_id: {exchange: {"scale": s, "source": src}}}``) records
    where each sampling band came from — §4.7: a band anchored at the default is
    a known blind spot, not a silent one.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bands = bands or {}
    blob = {
        s.genome_id: {
            "active": s.active,
            "background": s.background,
            "sensitivity": s.sensitivity,
            "mu_rich": s.mu_rich,
            **({"bands": bands[s.genome_id]} if s.genome_id in bands else {}),
        }
        for s in subspaces
    }
    path.write_text(json.dumps(blob, indent=2, sort_keys=True))


def load_subspaces(path: Path) -> dict[str, ActiveSubspace]:
    """Load ``write_subspaces`` output back into ``{genome_id: ActiveSubspace}``."""
    blob = json.loads(Path(path).read_text())
    return {
        gid: ActiveSubspace(gid, d["active"], d["background"], d["sensitivity"], d["mu_rich"])
        for gid, d in blob.items()
    }
