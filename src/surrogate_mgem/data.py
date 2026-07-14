"""Phase 0: generate surrogate training data from real MICOM community solves.

Samples community subsets (from a genome roster) and media (Latin-hypercube /
Dirichlet over each subset's exchange universe), solves the MICOM cooperative
tradeoff with ``fluxes=True``, and records the labels the surrogate learns:
per-member growth and per-member signed exchange fluxes, alongside the medium.

Only this module imports micom/cobra (the optional ``data`` extra); the imports
are function-local so the module itself loads solver-free. Output is tidy long
tables so downstream training can align columns against a shared exchange
universe regardless of which members a given sample contained.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from surrogate_mgem import sampling

LOGGER = logging.getLogger("surrogate-mgem.data")

# COBRA exchange sign convention: positive flux = secretion, negative = uptake.
_FLUX_EPS = 1e-9


@dataclass(frozen=True)
class GenomeModel:
    """One genome's CarveMe/GEM SBML model."""

    genome_id: str
    model_path: Path

    @property
    def taxon_id(self) -> str:
        """MICOM-safe taxon id (micom rejects '.'/'-' in ids)."""
        return self.genome_id.replace(".", "_").replace("-", "_")


@dataclass
class GenerateConfig:
    """Parameters for a training-data generation run."""

    out_dir: Path
    n_communities: int = 50
    size_range: tuple[int, int] = (2, 6)
    media_per_community: int = 20
    max_uptake: float = 1000.0
    tradeoff: float = 0.35
    sampler: str = "lhs"  # "lhs" | "dirichlet"
    solver: str = "hybrid"
    seed: int = 0
    workers: int = 1
    knockouts: bool = False  # also record single-member-drop growth changes
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Roster / community construction
# ---------------------------------------------------------------------------


def read_roster(path: Path) -> list[GenomeModel]:
    """Read a roster CSV with ``genome_id`` and ``model_path`` columns."""
    table = pd.read_csv(path)
    missing = {"genome_id", "model_path"}.difference(table.columns)
    if missing:
        raise ValueError(f"Roster is missing required columns: {sorted(missing)}")
    base = path.parent
    return [
        GenomeModel(
            genome_id=str(row["genome_id"]),
            model_path=(base / str(row["model_path"])).resolve()
            if not Path(str(row["model_path"])).is_absolute()
            else Path(str(row["model_path"])),
        )
        for _, row in table.iterrows()
    ]


def _taxonomy_frame(members: list[GenomeModel]) -> pd.DataFrame:
    """Return a MICOM taxonomy frame (equal abundances) for a subset of genomes."""
    abundance = 1.0 / len(members)
    return pd.DataFrame(
        [
            {
                "id": m.taxon_id,
                "genus": "roster",
                "species": m.genome_id,
                "file": str(m.model_path),
                "abundance": abundance,
            }
            for m in members
        ]
    )


def _build_community(members: list[GenomeModel], solver: str):
    """Build a MICOM community for a subset of genomes (micom imported here)."""
    from micom import Community

    return Community(_taxonomy_frame(members), solver=solver, progress=False)


# ---------------------------------------------------------------------------
# Solving one (community, medium) sample
# ---------------------------------------------------------------------------


def _medium_exchanges(community) -> list[str]:
    """Every community-level medium exchange id (the uptake vector's coordinates)."""
    return sorted(rxn.id for rxn in community.exchanges)


def _member_exchange_rows(sample_id: int, taxon_to_genome: dict[str, str], fluxes) -> list[dict]:
    """Extract per-member signed exchange fluxes (``EX_*_e``) from a solution."""
    rows = []
    for taxon_id, genome_id in taxon_to_genome.items():
        if taxon_id not in fluxes.index:
            continue
        member = fluxes.loc[taxon_id]
        for col in member.index:
            name = str(col)
            if not (name.startswith("EX_") and name.endswith("_e")):
                continue
            flux = float(member[col])
            if abs(flux) < _FLUX_EPS:
                continue
            rows.append(
                {
                    "sample_id": sample_id,
                    "genome_id": genome_id,
                    "exchange_id": name,
                    "flux": flux,
                }
            )
    return rows


def _solve_sample(community, uptake: dict[str, float], tradeoff: float):
    """Apply a medium and solve the cooperative tradeoff; return solution or None.

    Returns ``None`` on any solver failure (infeasible medium), so a bad draw
    is skipped rather than aborting the run.
    """
    community.medium = {ex: b for ex, b in uptake.items() if b > 0}
    try:
        return community.cooperative_tradeoff(fraction=tradeoff, fluxes=True, pfba=True)
    except Exception as error:  # noqa: BLE001 - micom/solver raise many types on infeasible media
        LOGGER.debug("Infeasible sample (%s: %s).", type(error).__name__, error)
        return None


def _run_subset(
    members: list[GenomeModel],
    community_index: int,
    config: GenerateConfig,
) -> dict[str, list[dict]]:
    """Solve all media draws for one community subset; return long-format rows.

    Runs in a worker process, so it (re)builds its own community and returns
    plain dict rows (picklable) rather than a live community object.
    """
    community = _build_community(members, config.solver)
    med_ex = _medium_exchanges(community)
    dim = len(med_ex)
    taxon_to_genome = {m.taxon_id: m.genome_id for m in members}
    community_id = "+".join(sorted(m.genome_id for m in members))
    seed = config.seed + community_index * 1000

    if config.sampler == "dirichlet":
        design = sampling.dirichlet_sample(config.media_per_community, dim, config.max_uptake, seed)
    else:
        design = sampling.latin_hypercube(config.media_per_community, dim, config.max_uptake, seed)

    out = {"samples": [], "membership": [], "media": [], "member_growth": [], "member_exchange": []}
    for draw, vector in enumerate(design):
        sample_id = community_index * config.media_per_community + draw
        uptake = {ex: float(b) for ex, b in zip(med_ex, vector, strict=True)}
        solution = _solve_sample(community, uptake, config.tradeoff)
        feasible = solution is not None
        out["samples"].append(
            {
                "sample_id": sample_id,
                "community_id": community_id,
                "n_members": len(members),
                "feasible": feasible,
                "community_growth": float(solution.growth_rate) if feasible else np.nan,
            }
        )
        for m in members:
            out["membership"].append({"sample_id": sample_id, "genome_id": m.genome_id})
        for ex, b in uptake.items():
            if b > 0:
                out["media"].append({"sample_id": sample_id, "exchange_id": ex, "uptake": b})
        if not feasible:
            continue
        member_growth = solution.members.loc[list(taxon_to_genome), "growth_rate"]
        for taxon_id, growth in member_growth.items():
            out["member_growth"].append(
                {
                    "sample_id": sample_id,
                    "genome_id": taxon_to_genome[taxon_id],
                    "growth": float(growth),
                }
            )
        out["member_exchange"].extend(
            _member_exchange_rows(sample_id, taxon_to_genome, solution.fluxes)
        )
    LOGGER.info(
        "Community %d (%d members): %d media solved.", community_index, len(members), len(design)
    )
    return out


# ---------------------------------------------------------------------------
# Exchange universe (shared coordinate system across all samples)
# ---------------------------------------------------------------------------


def build_exchange_universe(roster: list[GenomeModel], solver: str) -> dict[str, list[str]]:
    """Return the union of medium (``EX_*_m``) and member (``EX_*_e``) exchange ids.

    Built once from the full-roster community so training can align every
    sample's long rows to a fixed coordinate system regardless of membership.
    """
    community = _build_community(roster, solver)
    medium = sorted(rxn.id for rxn in community.exchanges)
    member = sorted(
        str(rxn.id)
        for rxn in community.reactions
        if str(rxn.id).startswith("EX_") and str(rxn.id).endswith("_e")
    )
    return {"medium_exchanges": medium, "member_exchanges": sorted(set(member))}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def generate(roster: list[GenomeModel], config: GenerateConfig) -> dict[str, Path]:
    """Generate training tables for a roster; return the written file paths."""
    config.out_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Building exchange universe from %d genomes...", len(roster))
    universe = build_exchange_universe(roster, config.solver)
    (config.out_dir / "exchange_universe.json").write_text(json.dumps(universe, indent=2))

    subsets = sampling.sample_membership(
        len(roster), config.n_communities, config.size_range, config.seed
    )
    member_subsets = [[roster[i] for i in idx] for idx in subsets]
    LOGGER.info(
        "Solving %d communities x %d media (%s sampler)...",
        len(member_subsets),
        config.media_per_community,
        config.sampler,
    )

    collected = {
        k: [] for k in ("samples", "membership", "media", "member_growth", "member_exchange")
    }

    def absorb(result: dict[str, list[dict]]) -> None:
        for key, rows in result.items():
            collected[key].extend(rows)

    if config.workers <= 1:
        for i, members in enumerate(member_subsets):
            absorb(_run_subset(members, i, config))
    else:
        with ProcessPoolExecutor(max_workers=config.workers) as executor:
            futures = {
                executor.submit(_run_subset, members, i, config): i
                for i, members in enumerate(member_subsets)
            }
            for future in as_completed(futures):
                absorb(future.result())

    written: dict[str, Path] = {}
    for name, rows in collected.items():
        path = config.out_dir / f"{name}.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        written[name] = path
    written["exchange_universe"] = config.out_dir / "exchange_universe.json"
    n_feasible = sum(1 for r in collected["samples"] if r["feasible"])
    LOGGER.info(
        "Wrote %d samples (%d feasible) to %s",
        len(collected["samples"]),
        n_feasible,
        config.out_dir,
    )
    return written
