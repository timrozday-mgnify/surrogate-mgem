"""Thin-shell reaction-mask sampling for §14's mask-conditioned Head A (P18).

A uniform draw over an organism's reactions is nearly information-free: it either
hits a redundant reaction, where ``mu_max`` does not move, or an essential one, where
``mu_max`` goes to 0. Both are cheap to predict from a single exact solve and teach
the head nothing about the regime it is for.

The useful signal lives in a **thin shell** around a carved model: masks that differ
from it by ``k`` reactions drawn from the *candidate set* actually being screened —
the low-evidence annotations, the gap-fill candidates, the pathway unit under test.
That is also the deployment distribution, so training and use agree. The cost is that
the resulting head is only valid **near** the models it was trained around, which is
weaker than "network-editable" in general and belongs beside any number it produces.

The mask index is **frozen and hashed** like the metabolite index (P13): a run that
silently reorders reactions produces labels that cannot be compared with any other.
Only *internal* reactions are maskable (P22) — an exchange edit goes through the
medium input instead, or the two inputs contradict each other.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Sequence

import numpy as np

LOGGER = logging.getLogger(__name__)


def maskable_reactions(model) -> list[str]:
    """Internal reaction ids, sorted — the mask index for one model (P13, P22).

    Exchanges, sinks and demands are excluded: they are the medium's job. The biomass
    reaction is excluded because masking it makes ``mu_max`` trivially 0 and the row
    carries no information.
    """
    boundary = {r.id for r in model.boundary}
    biomass = {r.id for r in model.reactions if "biomass" in r.id.lower()}
    return sorted(r.id for r in model.reactions if r.id not in boundary and r.id not in biomass)


def index_hash(reaction_ids: Sequence[str]) -> str:
    """Stable hash of a mask index, to be recorded alongside every label shard."""
    payload = json.dumps(list(reaction_ids), sort_keys=False).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def sample_knockout_masks(
    reaction_ids: Sequence[str],
    candidates: Sequence[str],
    n_masks: int,
    seed: int = 0,
    k_max: int = 5,
    p_unmasked: float = 0.05,
) -> np.ndarray:
    """``(n_masks, R)`` float32 0/1 masks over ``reaction_ids`` (1 = present).

    Each row removes ``k`` reactions drawn uniformly from ``candidates``, with ``k``
    uniform on ``1..k_max`` — ``k > 1`` is what carries the pathway-AND structure
    that single knockouts cannot show (a route needs two steps; neither alone moves
    ``mu``). A ``p_unmasked`` share of rows is the all-present mask, so the head sees
    the unedited network in the same batch as its edits and the medium axis stays
    anchored.

    Raises if a candidate is not in the index, because a silently-dropped candidate
    is the failure that looks like "the edit had no effect".
    """
    pos = {r: i for i, r in enumerate(reaction_ids)}
    missing = [c for c in candidates if c not in pos]
    if missing:
        raise ValueError(f"{len(missing)} candidates not in the mask index, e.g. {missing[:5]}")
    if not candidates:
        raise ValueError("no candidate reactions to mask")

    rng = np.random.default_rng(seed)
    cand_idx = np.array([pos[c] for c in candidates])
    masks = np.ones((n_masks, len(reaction_ids)), dtype=np.float32)
    k_max = max(1, min(k_max, len(cand_idx)))
    for row in range(n_masks):
        if rng.random() < p_unmasked:
            continue
        k = rng.integers(1, k_max + 1)
        masks[row, rng.choice(cand_idx, size=k, replace=False)] = 0.0
    return masks


def mask_to_knockouts(reaction_ids: Sequence[str], mask: np.ndarray) -> list[str]:
    """The reaction ids a mask row turns off — what :func:`cfs.groundtruth.solve.solve`
    takes as ``knockouts``."""
    return [str(reaction_ids[int(i)]) for i in np.flatnonzero(np.asarray(mask) == 0)]
