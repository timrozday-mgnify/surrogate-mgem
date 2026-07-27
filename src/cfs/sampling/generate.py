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
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from cfs.sampling.active_subspace import active_subspace, demand_probe, write_subspaces
from cfs.sampling.design import SamplingConfig, band_scales, sample_media

# Round r's medium ids start here, so a top-up shard never collides with the base
# run's (the train/val split is by `medium_id`, so a collision would silently fuse
# two different media into one unit).
_ROUND_STRIDE = 1_000_000

LOGGER = logging.getLogger("cfs.generate")


@dataclass
class OrganismShards:
    genome_id: str
    n_media: int
    paths: list[Path]


def _row(genome_id, index_hash, medium_id, medium, sol, ex_order):
    """One solve -> one flat row; vectors aligned to ``ex_order`` (this organism's M_i)."""
    return {
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


def generate_organism(model, genome_id: str, index_hash: str, outdir: Path,
                      cfg: SamplingConfig | None = None, km_cfg: dict | None = None,
                      subspace=None, scales: dict[str, float] | None = None,
                      focus_weights: dict[str, float] | None = None,
                      roster_median: dict[str, float] | None = None,
                      round_idx: int = 0) -> OrganismShards:
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
    """
    from cfs.groundtruth.solve import load_km_defaults, solve

    cfg = cfg if cfg is not None else SamplingConfig()
    km_cfg = km_cfg if km_cfg is not None else load_km_defaults()
    subspace = subspace if subspace is not None else active_subspace(model, genome_id, km_cfg)

    sampled = subspace.active or subspace.background
    probe = (demand_probe(model, sampled, km_cfg, lo=cfg.log10_lo, hi=cfg.log10_hi,
                          steps=cfg.probe_steps) if cfg.probe else {})
    scales, sources = band_scales(probe, scales, roster_median, sampled)
    media = sample_media(subspace, km_cfg, cfg, scales, focus_weights)
    ex_order = [ex.id for ex in model.exchanges]
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"{genome_id}.exchanges.json").write_text(
        json.dumps({"index_hash": index_hash, "exchanges": ex_order}, indent=2)
    )
    # The design's own record: which metabolites were sampled vs. held rich, and
    # where each band was anchored (§4.7). Needed to audit |A_i| after a bulk run
    # and to target the §4.6 active-learning reserve.
    bands = {ex: {"scale": scales[ex], "source": sources[ex]} for ex in sampled}
    write_subspaces([subspace], outdir / f"{genome_id}.subspace.json", {genome_id: bands})

    primary = cfg.eps_levels[cfg.eps_primary_idx]
    step = max(1, round(1.0 / cfg.subset_frac))
    subset = media[::step]  # deterministic stratified subset for the non-primary eps
    part = "part.parquet" if round_idx == 0 else f"part.round{round_idx}.parquet"
    mid0 = round_idx * _ROUND_STRIDE

    paths = []
    for eps in cfg.eps_levels:
        media_e = media if eps == primary else subset
        rows = [
            _row(genome_id, index_hash, mid0 + mid, medium,
                 solve(model, medium, alpha, eps, km_cfg), ex_order)
            for mid, medium in enumerate(media_e)
            for alpha in cfg.alphas
        ]
        path = outdir / f"genome_id={genome_id}" / f"eps={eps:g}" / part
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(path, index=False)
        LOGGER.info("%s eps=%g: %d rows -> %s", genome_id, eps, len(rows), path)
        paths.append(path)
    return OrganismShards(genome_id, len(media), paths)


def generate_roster(roster, index_path: Path, outdir: Path,
                    cfg: SamplingConfig | None = None,
                    scales: dict[str, dict[str, float]] | None = None,
                    focus_weights: dict[str, dict[str, float]] | None = None,
                    round_idx: int = 0) -> list[OrganismShards]:
    """Run :func:`generate_organism` for every roster model (serial; see module doc).

    ``scales`` is keyed by ``genome_id`` — the limiting regime is per organism, not
    a roster constant (measured spread across organisms: 2x for the ions, 2585x
    for ``EX_arg__L_e``). Its per-exchange median over the roster is the third
    link of the §4.7 chain, derived here rather than asked for: it is only ever a
    prior for a metabolite this organism's probe could not measure.
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
        shards.append(generate_organism(model, gm.genome_id, ihash, outdir, cfg, km_cfg,
                                        scales=scales.get(gm.genome_id),
                                        focus_weights=focus_weights.get(gm.genome_id),
                                        roster_median=roster_median, round_idx=round_idx))
    return shards
