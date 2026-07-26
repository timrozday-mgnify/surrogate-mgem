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


def active_subspace(model, genome_id: str = "", km_cfg: dict | None = None,
                    *, tol: float = 1e-3) -> ActiveSubspace:
    """Coarse one-at-a-time sensitivity sweep for one model (plan §4.2)."""
    from cfs.groundtruth.solve import apply_mm_bounds, load_km_defaults

    km_cfg = km_cfg if km_cfg is not None else load_km_defaults()
    uptakes = _uptake_exchanges(model)
    rich = dict.fromkeys(uptakes, _C_RICH)

    with model:
        apply_mm_bounds(model, rich, km_cfg)
        mu_rich = model.optimize().objective_value or 0.0
    if mu_rich <= 0.0:
        LOGGER.warning("%s: no growth on rich medium; active subspace empty", genome_id)
        return ActiveSubspace(genome_id, [], sorted(uptakes),
                              dict.fromkeys(uptakes, 0.0), 0.0)

    sensitivity = {}
    for ex in uptakes:
        medium = {**rich, ex: 0.0}  # deplete this one, keep the rest rich
        with model:
            apply_mm_bounds(model, medium, km_cfg)
            mu = model.optimize().objective_value or 0.0
        sensitivity[ex] = (mu_rich - mu) / mu_rich

    active = sorted(ex for ex, s in sensitivity.items() if s > tol)
    active_set = set(active)
    background = sorted(ex for ex in uptakes if ex not in active_set)
    LOGGER.info("%s: |A_i|=%d active of %d uptakes", genome_id, len(active), len(uptakes))
    return ActiveSubspace(genome_id, active, background, sensitivity, float(mu_rich))


def write_subspaces(subspaces: list[ActiveSubspace], path: Path) -> None:
    """Write per-organism active subspaces to one JSON (fed to ``generate``)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = {
        s.genome_id: {
            "active": s.active,
            "background": s.background,
            "sensitivity": s.sensitivity,
            "mu_rich": s.mu_rich,
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
