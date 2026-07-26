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

from cfs.sampling.active_subspace import active_subspace
from cfs.sampling.design import SamplingConfig, sample_media

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
                      subspace=None) -> OrganismShards:
    """Generate all label shards for one organism (plan §4.5)."""
    from cfs.groundtruth.solve import load_km_defaults, solve

    cfg = cfg if cfg is not None else SamplingConfig()
    km_cfg = km_cfg if km_cfg is not None else load_km_defaults()
    subspace = subspace if subspace is not None else active_subspace(model, genome_id, km_cfg)

    media = sample_media(subspace, km_cfg, cfg)
    ex_order = [ex.id for ex in model.exchanges]
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"{genome_id}.exchanges.json").write_text(
        json.dumps({"index_hash": index_hash, "exchanges": ex_order}, indent=2)
    )

    primary = cfg.eps_levels[cfg.eps_primary_idx]
    step = max(1, round(1.0 / cfg.subset_frac))
    subset = media[::step]  # deterministic stratified subset for the non-primary eps

    paths = []
    for eps in cfg.eps_levels:
        media_e = media if eps == primary else subset
        rows = [
            _row(genome_id, index_hash, mid, medium, solve(model, medium, alpha, eps, km_cfg),
                 ex_order)
            for mid, medium in enumerate(media_e)
            for alpha in cfg.alphas
        ]
        path = outdir / f"genome_id={genome_id}" / f"eps={eps:g}" / "part.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(path, index=False)
        LOGGER.info("%s eps=%g: %d rows -> %s", genome_id, eps, len(rows), path)
        paths.append(path)
    return OrganismShards(genome_id, len(media), paths)


def generate_roster(roster, index_path: Path, outdir: Path,
                    cfg: SamplingConfig | None = None) -> list[OrganismShards]:
    """Run :func:`generate_organism` for every roster model (serial; see module doc)."""
    from cobra.io import read_sbml_model

    from cfs.groundtruth.index import index_hash
    from cfs.groundtruth.solve import load_km_defaults

    cfg = cfg if cfg is not None else SamplingConfig()
    km_cfg = load_km_defaults()
    ihash = index_hash(index_path)
    shards = []
    for gm in roster:
        model = read_sbml_model(str(gm.model_path))
        shards.append(generate_organism(model, gm.genome_id, ihash, outdir, cfg, km_cfg))
    return shards
