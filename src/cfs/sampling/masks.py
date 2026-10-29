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
from pathlib import Path

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


def universe_additions(universe_path: str | Path, model, n: int, seed: int = 0) -> list[str]:
    """``n`` internal universe reactions this model does **not** have — the addition
    pool (§15.4).

    A mask over the organism's own reactions can only *remove*, and gap-filling is
    *addition*, so an addition has to be a bit that goes 0 -> 1. That needs an index
    containing reactions the carved model lacks, and the CarveMe universe is the right
    source because every carved model is a subset of it — verified here on
    ``CP009913.1``: all 852 of its internal reactions are in ``bigg_universe.xml``,
    and the only ids missing are the exchanges and ``ATPM``, which CarveMe adds per
    model and which the mask must not touch anyway (P22).

    The pool is **frozen up front and becomes part of the mask index**, rather than
    the whole 25 348-reaction universe: the index sets the head's input width, and a
    gap-filling candidate set is a few hundred reactions, not all of BiGG. Parsed by
    regex, not cobra — the universe SBML is 61 MB and only its reaction ids are needed
    here.
    """
    import re

    text = Path(universe_path).read_text(encoding="utf-8", errors="replace")
    u_ids = re.findall(r'<reaction[^>]*\bid="(R_[^"]+)"', text)
    have = {r.id for r in model.reactions}
    # cobra strips the SBML `R_` prefix, so compare on the stripped form.
    absent = sorted({r[2:] for r in u_ids if not r.startswith("R_EX_")} - have)
    rng = np.random.default_rng(seed)
    if n >= len(absent):
        return absent
    return sorted(absent[i] for i in rng.choice(len(absent), size=n, replace=False))


def extend_with_additions(model, universe, addition_ids: Sequence[str]):
    """Copy ``addition_ids`` from ``universe`` into ``model`` **with their universe
    bounds**, and return the ids actually added.

    Keeping the real bounds is the subtle part. Grafting them at ``(0, 0)`` instead
    looks safer — the extended model is then identical to the original until a mask
    turns one on — but nothing in the solve path ever restores a real bound, so a
    *selected* addition would silently stay off and every addition row would read as a
    no-op. With real bounds the convention is uniform: **everything the mask does not
    select is passed to ``solve(..., knockouts=)``**, additions included, and the
    unedited control must likewise be taken with every addition off
    (:func:`mu_unmasked_with`).
    """
    present = {r.id for r in model.reactions}
    to_add = [
        universe.reactions.get_by_id(r_id).copy() for r_id in addition_ids if r_id not in present
    ]
    if to_add:
        model.add_reactions(to_add)
    return [r.id for r in to_add]


def sample_edit_masks(
    reaction_ids: Sequence[str],
    base_present: Sequence[str],
    candidates: Sequence[str],
    n_masks: int,
    seed: int = 0,
    k_max: int = 5,
    p_unmasked: float = 0.05,
) -> np.ndarray:
    """``(n_masks, R)`` float32 0/1 masks over ``reaction_ids`` (1 = present).

    Each row flips ``k`` candidate bits **away from their base value**, with ``k``
    uniform on ``1..k_max``. So a candidate the carved model has is *removed* and one
    it lacks is *added* — the symmetric version of what gap-filling does, and the
    reason the index has to contain absent reactions (§15.4). ``k > 1`` is what
    carries the pathway-AND structure single edits cannot show: a route needs two
    steps, and neither alone moves ``mu``.

    A ``p_unmasked`` share of rows is the base mask itself, so the head sees the
    unedited network in the same batch as its edits and the medium axis stays
    anchored.

    Raises if a candidate is not in the index, because a silently-dropped candidate is
    the failure that looks exactly like "the edit had no effect".
    """
    pos = {r: i for i, r in enumerate(reaction_ids)}
    missing = [c for c in candidates if c not in pos]
    if missing:
        raise ValueError(f"{len(missing)} candidates not in the mask index, e.g. {missing[:5]}")
    if not candidates:
        raise ValueError("no candidate reactions to mask")

    rng = np.random.default_rng(seed)
    base = np.zeros(len(reaction_ids), dtype=np.float32)
    present = set(base_present)
    for r, i in pos.items():
        if r in present:
            base[i] = 1.0
    cand_idx = np.array([pos[c] for c in candidates])
    masks = np.tile(base, (n_masks, 1))
    k_max = max(1, min(k_max, len(cand_idx)))
    for row in range(n_masks):
        if rng.random() < p_unmasked:
            continue
        k = rng.integers(1, k_max + 1)
        flip = rng.choice(cand_idx, size=k, replace=False)
        masks[row, flip] = 1.0 - masks[row, flip]
    return masks


def sample_knockout_masks(
    reaction_ids: Sequence[str],
    candidates: Sequence[str],
    n_masks: int,
    seed: int = 0,
    k_max: int = 5,
    p_unmasked: float = 0.05,
) -> np.ndarray:
    """Removal-only masks: :func:`sample_edit_masks` with every reaction present.

    Kept because it is what the first M19 run used, so that run stays reproducible.
    New work wants :func:`sample_edit_masks`, which can also add.
    """
    return sample_edit_masks(
        reaction_ids, reaction_ids, candidates, n_masks, seed, k_max, p_unmasked
    )


def mask_to_knockouts(reaction_ids: Sequence[str], mask: np.ndarray) -> list[str]:
    """The reaction ids a mask row turns off — what :func:`cfs.groundtruth.solve.solve`
    takes as ``knockouts``.

    On an extended model (:func:`extend_with_additions`) this includes every *unselected
    addition*, which is correct: an addition that is not selected must stay at zero
    bounds, exactly like a knocked-out reaction.
    """
    return [str(reaction_ids[int(i)]) for i in np.flatnonzero(np.asarray(mask) == 0)]


def screen_single_edits(
    model,
    concentrations: dict,
    reaction_ids: Sequence[str],
    base_present: Sequence[str],
    km_cfg: dict | None = None,
    rel_tol: float = 1e-6,
) -> dict[str, str]:
    """Classify every candidate by its **own** single edit: the M19 selection screen.

    Returns ``{reaction_id: "lethal" | "partial" | "noop" | "helpful"}``, measured by
    one LP each against the unedited control on the same medium.

    This exists because the first M19 run came out **62.7% lethal, 26.5% no-op, 10.8%
    partial** — the thin shell landed almost entirely on rows that exact FBA answers
    in 3.5 ms and that published essentiality classifiers already do (§15.2). The
    cause was the candidate proxy: a reaction *carrying flux* on a rich medium is
    usually essential. Screening first is cheap (one warm LP per candidate, ~3.5 ms)
    and lets the sampler draw from the classes that actually carry signal.

    ``helpful`` is an *addition* that raises ``mu`` — the gap-filling case, and the
    only class whose sign is positive.
    """
    from cfs.groundtruth.solve import load_km_defaults

    km_cfg = km_cfg if km_cfg is not None else load_km_defaults()
    present = set(base_present)
    absent_off = [r for r in reaction_ids if r not in present]
    base = mu_unmasked_with(model, concentrations, absent_off, km_cfg)
    if base <= 1e-9:
        raise ValueError("the control does not grow on this medium; nothing to screen")

    out: dict[str, str] = {}
    for r_id in reaction_ids:
        if r_id in present:  # a removal
            mu = mu_unmasked_with(model, concentrations, absent_off + [r_id], km_cfg)
        else:  # an addition: everything else absent stays off, this one comes on
            mu = mu_unmasked_with(
                model, concentrations, [a for a in absent_off if a != r_id], km_cfg
            )
        rel = (mu - base) / max(abs(base), 1e-12)
        if mu <= 1e-9:
            out[r_id] = "lethal"
        elif rel > rel_tol:
            out[r_id] = "helpful"
        elif abs(rel) <= rel_tol:
            out[r_id] = "noop"
        else:
            out[r_id] = "partial"
    return out


def mu_unmasked_with(model, concentrations: dict, off: Sequence[str], km_cfg: dict) -> float:
    """``mu_max`` with ``off`` forced to zero flux. Plain FBA, bounds restored after."""
    from cfs.groundtruth.solve import apply_mm_bounds, mu_optimize

    with model:
        apply_mm_bounds(model, concentrations, km_cfg, None)
        for r_id in off:
            model.reactions.get_by_id(r_id).bounds = (0.0, 0.0)
        return float(mu_optimize(model, "single-edit screen"))
