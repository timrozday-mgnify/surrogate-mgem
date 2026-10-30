"""§4.5 label-generation driver -> parquet, sharded by (organism, eps).

Per organism: build the active subspace (§4.2), the sampling design (§4.3), then
solve (§3.2 elastic net) over media × alpha × eps. The primary eps level (§5.5)
gets the full media set; the other two run on a stratified subset (§4.5) — the
smoothing family must span the region, not resolve it at equal density.

Stores ``mu_max``, ``z``, shadow prices, solver status and the medium vector; NOT
internal fluxes (large, unused). Every row records ``index_hash`` so a silent
metabolite-index change invalidates loudly (P13). The exchange order for the
``z``/``shadow``/``medium`` list columns is written once as a sidecar.

Cross-organism / cross-shard parallelism is the Nextflow layer's job (one process
per organism); this driver is serial per organism. Needs the ``data`` extra
(cobra + highspy) and ``pyarrow`` for parquet.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from cfs.sampling.active_subspace import active_subspace, demand_probe, write_subspaces
from cfs.sampling.design import SamplingConfig, band_scales, sample_media

# Round r's medium ids start here, so a top-up shard never collides with the base
# run's (the train/val split is by `medium_id`, so a collision would silently fuse
# two different media into one unit).
_ROUND_STRIDE = 1_000_000

# §13.11 stage 4': the label root records the `c^eq` it was solved with.
_INHIBITION_FILE = "inhibition.json"

# §14/M19: the mask label root records its own frozen reaction index, hashed for the
# same reason the metabolite index is (P13) — a silent reorder would fuse two
# incompatible designs, and a mask is only meaningful against the index it was drawn
# over.
_MASKS_FILE = "masks.json"

LOGGER = logging.getLogger("cfs.generate")


@dataclass
class OrganismShards:
    genome_id: str
    n_media: int
    paths: list[Path]


def _row(
    genome_id,
    index_hash,
    medium_id,
    medium,
    sol,
    ex_order,
    mask_id=None,
    knockouts=None,
    mu_base=None,
):
    """One solve -> one flat row; vectors aligned to ``ex_order`` (this organism's M_i).

    ``mask_id``/``knockouts`` are present only in a §14 mask run, which is therefore
    a *different label root* with a wider schema — the same rule as ``--inhibition``.
    The knocked-out ids are stored rather than the dense ``z`` bit vector: ``k <= 5``
    strings beat ~2000 floats a row, and the dense mask is rebuilt at load time from
    the hashed index in ``masks.json``. ``mu_base`` is the unmasked control on the
    *same* medium (:func:`cfs.groundtruth.solve.mu_unmasked`) — without it a mask row
    cannot distinguish "this edit did nothing" from "this medium is poor".
    """
    row = {
        "genome_id": genome_id,
        "index_hash": index_hash,
        "medium_id": medium_id,
        "alpha": sol.alpha,
        "eps": sol.eps,
        "mu_max": sol.mu_max,
        "status": sol.status,
        "medium": [float(medium.get(ex, 0.0)) for ex in ex_order],
        "z": [float(sol.z.get(ex, 0.0)) for ex in ex_order],
        "shadow": [float(sol.shadow_prices.get(ex, 0.0)) for ex in ex_order],
    }
    if mask_id is not None:
        row["mask_id"] = int(mask_id)
        row["knockouts"] = list(knockouts or [])
        row["mu_unmasked"] = float("nan") if mu_base is None else float(mu_base)
        row["rel_dmu"], row["edit_class"] = _edit_effect(sol.mu_max, mu_base)
        if sol.reduced_costs:
            # §16.3, aligned to the sidecar's `candidates` order: which edits would
            # have mattered on this medium, free from the solve already done.
            row["reduced_cost"] = [float(v) for v in sol.reduced_costs.values()]
    return row


# A row's own target, derived once at label time rather than in every consumer.
_MU_FLOOR = 1e-9  # below this `mu` is zero to the LP, not a small number
_REL_TOL = 1e-6  # relative change below this is solver noise, not an effect


def _edit_effect(mu, mu_base) -> tuple[float, str]:
    """``(rel_dmu, edit_class)`` for one mask row against its paired control.

    The **relative** change is the trainable target, not absolute ``mu``: M19 measured
    `baseline_r2_mu_unchanged` swinging from -2.0 to +0.51 between two label sets that
    differed only in their candidates, because absolute ``mu`` is dominated by whichever
    class happens to prevail. ``rel_dmu`` is invariant to that, and ``edit_class``
    makes the mix explicit so training and scoring can stratify on it instead of
    discovering it.

    ``dead`` is a medium the control cannot grow on: the ratio is meaningless there and
    the row says nothing about the edit (the same solver-dust trap as the duals).
    Classes otherwise: ``lethal`` (mu to zero), ``helpful`` (mu up — the gap-filling
    direction, and the only positive sign), ``noop``, ``partial``.
    """
    if mu_base is None:
        return float("nan"), "unknown"
    base = float(mu_base)
    if base <= _MU_FLOOR:
        return float("nan"), "dead"
    rel = (float(mu) - base) / base
    if float(mu) <= _MU_FLOOR:
        return rel, "lethal"
    if rel > _REL_TOL:
        return rel, "helpful"
    if abs(rel) <= _REL_TOL:
        return rel, "noop"
    return rel, "partial"


def generate_organism(
    model,
    genome_id: str,
    index_hash: str,
    outdir: Path,
    cfg: SamplingConfig | None = None,
    km_cfg: dict | None = None,
    subspace=None,
    scales: dict[str, float] | None = None,
    focus_weights: dict[str, float] | None = None,
    roster_median: dict[str, float] | None = None,
    round_idx: int = 0,
    media: list[dict] | None = None,
    ceq: dict[str, float] | None = None,
    mask_candidates: Sequence[str] | None = None,
    mask_k_max: int = 5,
    mask_additions: Sequence[str] | None = None,
    universe=None,
    mask_k_min: int = 1,
    mask_deficient: float = 0.0,
    mask_drops: Sequence[str] | None = None,
    mask_anchor: bool = False,
) -> OrganismShards:
    """Generate all label shards for one organism (plan §4.5).

    Each metabolite's sampling band is centred on its own limiting regime; where
    that centre comes from is the §4.7 chain, resolved here: an LP
    :func:`~cfs.sampling.active_subspace.demand_probe` on *this* model
    (``cfg.probe``), else ``scales`` (a previous run's ``u*`` via
    :func:`~cfs.sampling.design.limiting_scales`), else ``roster_median``, else
    1.0 — the un-anchored design, whose coverage of "which metabolite limits" is
    skewed ~1000:1 (see :func:`~cfs.sampling.design.sample_media`). The choice is
    recorded per metabolite in the ``.subspace.json`` sidecar.

    ``focus_weights`` skews the focus budget toward the metabolites a previous
    run got measurably wrong (:func:`~cfs.sampling.design.topup_weights`).
    ``round_idx`` > 0 writes a top-up shard alongside the base run's rather than
    over it (§4.6).

    ``ceq`` turns on §13.11's thermodynamic product inhibition in the ground truth
    (stage 4'): secretion of a metabolite stops as its external concentration
    reaches ``c^eq``. It changes no column of the label schema -- ``theta`` is a
    function of the ``medium`` already stored and its dual is the same ``shadow``
    column read on the other side -- but it makes a *different label root*, so it
    is recorded in an ``inhibition.json`` sidecar beside the shards and
    :func:`cfs.surrogate.data.load_ceq` reads it back. ``None`` is plain FBA, bit
    for bit.

    ``mask_candidates`` turns on §14's reaction-presence labels (M19): each medium is
    paired with one **thin-shell** mask drawn over those candidate reactions, and the
    row records which reactions were knocked out. The pairing is 1:1 and cycling, not
    the ``media x masks`` product — the product is the combinatorial object the
    surrogate exists to avoid, and spending the budget on it here would buy one
    organism's corner of it. Like ``ceq``, this makes a **different label root** (two
    extra columns), recorded in ``masks.json`` beside the shards. P18: a uniform draw
    over every reaction is nearly information-free, which is why candidates are
    required rather than defaulted to the whole index.

    ``media`` bypasses the whole design: label exactly these media instead of
    sampling them. That is how the media §8.1 actually visits get labelled — the
    dFBA states are a point set no design produces, and Head B's residual is
    coverage of them. The subspace, the probe and the sidecars are then skipped
    (nothing here chose the media, so there is no band to record, and the base
    run's sidecars must not be overwritten by a round).
    """
    from cfs.groundtruth.solve import load_km_defaults, mu_unmasked, solve

    cfg = cfg if cfg is not None else SamplingConfig()
    km_cfg = km_cfg if km_cfg is not None else load_km_defaults()
    designed = media is None

    if designed:
        subspace = subspace if subspace is not None else active_subspace(model, genome_id, km_cfg)
        sampled = subspace.active or subspace.background
        probe = (
            demand_probe(
                model, sampled, km_cfg, lo=cfg.probe_lo, hi=cfg.log10_hi, steps=cfg.probe_steps
            )
            if cfg.probe
            else {}
        )
        scales, sources = band_scales(probe, scales, roster_median, sampled)
        # B2 (§8.5): a second bisection at a higher recovery target bounds the mid-`mu`
        # stratum's band from above. Same ~350 LPs and ~2 s as the first probe, and it
        # is the only way to place that band without labels; there is no fallback
        # chain because a metabolite with no probe result never limits in the band at
        # all, and `sample_media` stands in with half a decade above its own anchor.
        mid_scales = (
            demand_probe(
                model,
                sampled,
                km_cfg,
                lo=cfg.probe_lo,
                hi=cfg.log10_hi,
                steps=cfg.probe_steps,
                target_frac=cfg.mid_target_frac,
            )
            if cfg.probe and cfg.frac_mid_mu > 0
            else {}
        )
        media = sample_media(subspace, km_cfg, cfg, scales, focus_weights, mid_scales)
    ex_order = [ex.id for ex in model.exchanges]
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if designed:
        (outdir / f"{genome_id}.exchanges.json").write_text(
            json.dumps({"index_hash": index_hash, "exchanges": ex_order}, indent=2)
        )
        # The design's own record: which metabolites were sampled vs. held rich, and
        # where each band was anchored (§4.7). Needed to audit |A_i| after a bulk run
        # and to target the §4.6 active-learning reserve.
        bands = {ex: {"scale": scales[ex], "source": sources[ex]} for ex in sampled}
        write_subspaces([subspace], outdir / f"{genome_id}.subspace.json", {genome_id: bands})

    # §14/M19. Built per medium so the stratified eps subset carries the *same*
    # (medium, mask) pair: scoring one eps against another otherwise compares two
    # different networks and reads as a smoothing effect.
    knock: list[list[str]] | None = None
    added: list[str] = []
    if mask_candidates is not None:
        from cfs.sampling.masks import (
            extend_with_additions,
            mask_to_knockouts,
            maskable_reactions,
            sample_edit_masks,
        )
        from cfs.sampling.masks import (
            index_hash as mask_index_hash,
        )

        # §15.4: the index must contain reactions the carved model LACKS, or a mask can
        # only ever remove and the gap-filling case is unrepresentable. The addition
        # pool is grafted in at zero bounds, so the extended model is numerically
        # identical to the original until a mask turns one on.
        base_present = maskable_reactions(model)
        if mask_additions:
            if universe is None:
                raise ValueError("mask_additions needs a universe model to copy from")
            added = extend_with_additions(model, universe, mask_additions)
        rxn_ids = maskable_reactions(model)
        if mask_deficient > 0:
            # An addition can only help where something is missing, so a share of the
            # media are stripped of a nutrient. The rest stay rich on purpose: all
            # deficient would teach the head that edits always matter.
            from cfs.sampling.masks import deficient_media

            # The drop set must be the one the addition pool was screened against,
            # not any exchange: a targeted pool rescues a *specific* deficiency. With
            # random drops, 5 of 240 additions helped; with the matched pair, 23 of 64.
            media = deficient_media(
                media,
                mask_drops or [ex.id for ex in model.exchanges],
                seed=cfg.seed,
                frac=mask_deficient,
            )
        if mask_anchor:
            # §16.5 step 2: pair each medium with an edit that can actually bite there,
            # instead of drawing from the whole candidate set and hoping. Two LPs per
            # removal candidate per medium, which is why it is opt-in.
            from cfs.sampling.masks import live_candidates, sample_anchored_masks

            live = [
                live_candidates(model, medium, mask_candidates, base_present, km_cfg)
                for medium in media
            ]
            n_live = sum(len(x) for x in live)
            LOGGER.info(
                "%s: anchored masks, %d live candidate slots over %d media (%.2f/medium)",
                genome_id,
                n_live,
                len(media),
                n_live / max(len(media), 1),
            )
            masks = sample_anchored_masks(
                rxn_ids,
                base_present,
                live,
                seed=cfg.seed,
                k_max=mask_k_max,
                k_min=mask_k_min,
            )
        else:
            masks = sample_edit_masks(
                rxn_ids,
                base_present,
                mask_candidates,
                len(media),
                seed=cfg.seed,
                k_max=mask_k_max,
                k_min=mask_k_min,
            )
        knock = [mask_to_knockouts(rxn_ids, row) for row in masks]
        (outdir / f"{genome_id}.{_MASKS_FILE}").write_text(
            json.dumps(
                {
                    "mask_index_hash": mask_index_hash(rxn_ids),
                    "n_reactions": len(rxn_ids),
                    "n_base_present": len(base_present),
                    "n_additions": len(added),
                    # Both lists, not just their counts: without them an analysis
                    # cannot tell a row's *edits* from its unselected additions,
                    # which are off for the same reason but are not edits.
                    "base_present": base_present,
                    "additions": added,
                    "candidates": list(mask_candidates),
                    "k_max": mask_k_max,
                    "reactions": rxn_ids,
                },
                indent=2,
            )
        )

    primary = cfg.eps_levels[cfg.eps_primary_idx]
    step = max(1, round(1.0 / cfg.subset_frac))
    subset = media[::step]  # deterministic stratified subset for the non-primary eps
    part = "part.parquet" if round_idx == 0 else f"part.round{round_idx}.parquet"
    mid0 = round_idx * _ROUND_STRIDE

    paths = []
    for eps in cfg.eps_levels:
        media_e = media if eps == primary else subset
        knock_e = None if knock is None else (knock if eps == primary else knock[::step])
        rows = [
            _row(
                genome_id,
                index_hash,
                mid0 + mid,
                medium,
                solve(
                    model,
                    medium,
                    alpha,
                    eps,
                    km_cfg,
                    ceq,
                    None if knock_e is None else knock_e[mid],
                    reduced_cost_for=mask_candidates,
                ),
                ex_order,
                mask_id=None if knock_e is None else mid,
                knockouts=None if knock_e is None else knock_e[mid],
                # The control is the *base* network: every addition off. On an
                # extended model `mu_unmasked` alone would leave the grafted
                # reactions at their universe bounds, i.e. compare each edit
                # against a network that already has every candidate addition.
                mu_base=(
                    None if knock_e is None else mu_unmasked(model, medium, km_cfg, ceq, off=added)
                ),
            )
            for mid, medium in enumerate(media_e)
            for alpha in cfg.alphas
        ]
        path = outdir / genome_id / f"eps_{eps:g}" / part
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(path, index=False)
        LOGGER.info("%s eps=%g: %d rows -> %s", genome_id, eps, len(rows), path)
        paths.append(path)
    return OrganismShards(genome_id, len(media), paths)


def generate_roster(
    roster,
    index_path: Path,
    outdir: Path,
    cfg: SamplingConfig | None = None,
    scales: dict[str, dict[str, float]] | None = None,
    focus_weights: dict[str, dict[str, float]] | None = None,
    round_idx: int = 0,
    media: list[dict] | None = None,
    ceq: dict[str, float] | None = None,
    mask_candidates: dict[str, Sequence[str]] | None = None,
    mask_k_max: int = 5,
    mask_additions: dict[str, Sequence[str]] | None = None,
    universe=None,
    mask_k_min: int = 1,
    mask_deficient: float = 0.0,
    mask_drops: dict[str, Sequence[str]] | None = None,
    mask_anchor: bool = False,
) -> list[OrganismShards]:
    """Run :func:`generate_organism` for every roster model (serial; see module doc).

    ``scales`` is keyed by ``genome_id`` — the limiting regime is per organism, not
    a roster constant (measured spread across organisms: 2x for the ions, 2585x
    for ``EX_arg__L_e``). Its per-exchange median over the roster is the third
    link of the §4.7 chain, derived here rather than asked for: it is only ever a
    prior for a metabolite this organism's probe could not measure.

    ``media`` is the same explicit medium list for every organism — the point of
    it is a shared point set (the §8.1 trajectory states), and each organism's
    row keeps only its own exchanges anyway.
    """
    import numpy as np
    from cobra.io import read_sbml_model

    from cfs.groundtruth.index import index_hash
    from cfs.groundtruth.solve import load_km_defaults

    cfg = cfg if cfg is not None else SamplingConfig()
    km_cfg = load_km_defaults()
    ihash = index_hash(index_path)
    scales = scales or {}
    focus_weights = focus_weights or {}
    by_exchange: dict[str, list[float]] = {}
    for per_organism in scales.values():
        for ex, s in per_organism.items():
            by_exchange.setdefault(ex, []).append(float(s))
    roster_median = {ex: float(np.median(v)) for ex, v in by_exchange.items()}
    shards = []
    for gm in roster:
        model = read_sbml_model(str(gm.model_path))
        shards.append(
            generate_organism(
                model,
                gm.genome_id,
                ihash,
                outdir,
                cfg,
                km_cfg,
                scales=scales.get(gm.genome_id),
                focus_weights=focus_weights.get(gm.genome_id),
                roster_median=roster_median,
                round_idx=round_idx,
                media=media,
                ceq=ceq,
                mask_candidates=(
                    None if mask_candidates is None else mask_candidates.get(gm.genome_id)
                ),
                mask_k_max=mask_k_max,
                mask_additions=(
                    None if mask_additions is None else mask_additions.get(gm.genome_id)
                ),
                universe=universe,
                mask_k_min=mask_k_min,
                mask_deficient=mask_deficient,
                mask_drops=(None if mask_drops is None else mask_drops.get(gm.genome_id)),
                mask_anchor=mask_anchor,
            )
        )
    if ceq is not None:
        # The root's own record of which model made it. Read by
        # `cfs.surrogate.data.load_ceq`, which is what gives Head A its second
        # input channel -- so a root and a checkpoint cannot silently disagree
        # about whether inhibition was on.
        (Path(outdir) / _INHIBITION_FILE).write_text(json.dumps({"ceq": ceq}, indent=2))
    return shards
