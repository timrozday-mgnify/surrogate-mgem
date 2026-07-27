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
    sampler: str = "titrate"  # "titrate" | "perturb" | "sparse" | "dirichlet" | "lhs"
    n_active: int = 20  # sparse sampler: active components per medium
    keep_min: float = 0.5  # titrate sampler: lowest per-medium keep probability
    n_limiting: int = 3  # titrate sampler: nutrients made scarce per medium (rest replete)
    target_frac: float = 0.5  # titrate sampler: calibrate to this fraction of saturated growth
    solver: str = "hybrid"
    seed: int = 0
    workers: int = 1
    knockouts: bool = False  # also record single-member-drop growth changes
    # HPC fan-out: work units are (community, media-range) pairs, assigned round
    # robin over shards, so a single-community run still parallelises. Membership
    # sampling and the media design are deterministic in `seed`, so every shard
    # derives the same work list and takes its own slice. The shared exchange
    # universe is written only by shard 0 (the merge step reuses it).
    shard_index: int = 0
    num_shards: int = 1
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


def members_for_community(roster: list[GenomeModel], community_id: str) -> list[GenomeModel]:
    """Return the roster members named in a ``"g1+g2+..."`` community id.

    ``community_id`` is the ``+``-joined sorted genome ids written by
    :func:`generate`; used to rebuild a fixed community's members for the active
    loop's solver oracle. Raises if any named genome is missing from the roster.
    """
    wanted = set(community_id.split("+"))
    members = [m for m in roster if m.genome_id in wanted]
    missing = wanted - {m.genome_id for m in members}
    if missing:
        raise ValueError(f"Community members not in roster: {sorted(missing)}")
    return members


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
            # NaN: this taxon has no such reaction (pivoted frame gap). Skip it
            # and near-zero fluxes (numerical noise, not a real exchange).
            if not np.isfinite(flux) or abs(flux) < _FLUX_EPS:
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


@dataclass(frozen=True)
class MediumSpec:
    """What the ``titrate`` sampler needs to know about one community's medium.

    ``bound`` is the uptake bound at which growth is nutrient-limited overall
    (roughly ``target_frac`` of the saturated rate); ``essential`` flags the
    exchanges the community cannot grow without; ``demand`` is the *per-nutrient*
    uptake scale -- how much of each the community actually draws -- in the order
    of the exchange list it was built from.

    ``demand`` is what makes a medium informative. It is an LP output, not
    something to be learned: a nutrient is only ever growth-limiting near the flux
    the organism draws of it, and those fluxes span orders of magnitude (measured
    on the example genome: 1e-3 to 20, five decades). Sampling every exchange in
    one shared band leaves the small-demand ones permanently saturated, so growth
    does not respond to them and no amount of data or rescaling can recover a
    sensitivity that was never sampled. ``demand == 0`` marks an exchange the
    community never consumes: it cannot affect growth, so the sampler leaves it at
    zero and it drops out of the feature space entirely.
    """

    bound: float
    essential: np.ndarray  # bool mask over the community's medium exchanges
    demand: np.ndarray  # per-exchange uptake scale (0 = never consumed)

    def scale(self) -> np.ndarray:
        """Per-exchange sampling scale, falling back to the global bound."""
        return np.where(self.demand > 0, self.demand, 0.0)

    def to_json(self, exchanges: list[str]) -> dict:
        return {
            "bound": float(self.bound),
            "essential": [e for e, keep in zip(exchanges, self.essential, strict=True) if keep],
            "demand": {
                e: float(d) for e, d in zip(exchanges, self.demand, strict=True) if d > 0
            },
        }

    @classmethod
    def from_json(cls, blob: dict, exchanges: list[str]) -> MediumSpec:
        essential = set(blob.get("essential", ()))
        demand = blob.get("demand", {})
        return cls(
            bound=float(blob["bound"]),
            essential=np.array([e in essential for e in exchanges], dtype=bool),
            demand=np.array([float(demand.get(e, 0.0)) for e in exchanges], dtype=float),
        )


def _uptake_flux(solution, exchanges: list[str]) -> np.ndarray:
    """Community uptake flux per medium exchange (0 where nothing is consumed).

    micom renames each member's ``EX_<met>_e``; summing the member rows gives the
    community's draw on the shared pool exchange ``EX_<met>_m``.
    """
    fluxes = solution.fluxes
    drawn = np.zeros(len(exchanges))
    for i, ex in enumerate(exchanges):
        member_id = medium_to_member_exchange(ex)
        if member_id not in fluxes.columns:
            continue
        column = fluxes[member_id]
        taken = -column[np.isfinite(column) & (column < 0)].sum()
        drawn[i] = float(max(taken, 0.0))
    return drawn


def estimate_demand(
    community,
    exchanges: list[str],
    tradeoff: float,
    bound: float,
    essential: np.ndarray,
    *,
    n_probes: int = 24,
    seed: int = 0,
) -> np.ndarray:
    """Per-exchange uptake scale: how much of each nutrient the community draws.

    Read straight off the LP -- the uptake fluxes of a solved medium -- rather
    than learned, because a nutrient can only limit growth near the flux the
    organism actually consumes. One solve at a fully-open medium covers whatever
    the community prefers; the probe media (with the sampler's own dropout) then
    expose the substitutes that only get used once a preferred source is missing.
    On the example genome that is the difference between 50 and 114 exchanges of
    259 showing any demand at all -- the remaining ~145 cannot affect growth and
    are left out of the medium design.
    """
    drawn = np.zeros(len(exchanges))
    designs = [np.full(len(exchanges), bound)]
    if n_probes > 0:
        designs.extend(
            sampling.titrate_media(
                n_probes,
                len(exchanges),
                seed,
                scale=np.full(len(exchanges), bound),
                keep_range=(0.3, 1.0),
                essential=essential,
            )
        )
    for vector in designs:
        solution = _solve_sample(community, dict(zip(exchanges, vector, strict=True)), tradeoff)
        if solution is not None:
            drawn = np.maximum(drawn, _uptake_flux(solution, exchanges))
    return drawn


def medium_spec(
    community,
    exchanges: list[str],
    tradeoff: float,
    *,
    target_frac: float = 0.5,
    max_bound: float = 1000.0,
    steps: int = 20,
    n_probes: int = 24,
) -> MediumSpec:
    """Calibrate the limiting uptake bound and scan for essential exchanges.

    Growth saturates well below a fully-open medium (measured on the example
    genome: identical growth at bounds 1000 and 100, half of it at 10), so
    sampling around ``max_bound`` gives a constant target. Bisecting in log space
    for the bound where growth is ``target_frac`` of saturated puts the sampler in
    the regime where growth actually responds to the medium. The single-drop scan
    that follows costs one solve per exchange and identifies the nutrients every
    medium must keep to be viable at all.
    """

    def growth(vector: np.ndarray) -> float:
        solution = _solve_sample(community, dict(zip(exchanges, vector, strict=True)), tradeoff)
        return 0.0 if solution is None else float(solution.growth_rate)

    dim = len(exchanges)
    saturated = growth(np.full(dim, max_bound))
    if saturated <= 0:
        raise ValueError(
            "Community cannot grow on a fully-open medium: no medium will produce growth. "
            "Check the GEM(s) and the MICOM tradeoff."
        )
    lo, hi = max_bound * 1e-5, max_bound
    target = target_frac * saturated
    for _ in range(steps):
        mid = float(np.sqrt(lo * hi))
        if growth(np.full(dim, mid)) < target:
            lo = mid
        else:
            hi = mid
    bound = float(np.sqrt(lo * hi))

    base = np.full(dim, bound)
    base_growth = growth(base)
    essential = np.zeros(dim, dtype=bool)
    for i in range(dim):
        dropped = base.copy()
        dropped[i] = 0.0
        essential[i] = growth(dropped) < 0.01 * base_growth
    demand = estimate_demand(
        community, exchanges, tradeoff, bound, essential, n_probes=n_probes
    )
    # An exchange the community never draws on cannot change growth; keeping it in
    # the design only adds a dimension the surrogate can memorise noise in.
    demand[essential & (demand <= 0)] = bound
    LOGGER.info(
        "Calibrated uptake bound %.4g (saturated growth %.4g at %.4g); "
        "%d/%d exchanges essential, %d ever consumed (demand %.3g - %.3g).",
        bound,
        saturated,
        max_bound,
        int(essential.sum()),
        dim,
        int((demand > 0).sum()),
        float(demand[demand > 0].min()) if (demand > 0).any() else 0.0,
        float(demand.max()),
    )
    return MediumSpec(bound=bound, essential=essential, demand=demand)


def make_fixed_community_evaluator(
    members: list[GenomeModel],
    feature_names: list[str],
    target_names: list[str],
    solver: str,
    tradeoff: float,
    spec_json: dict | None = None,
):
    """Return ``(evaluate, active_mask, spec)`` for a fixed community (built once).

    ``evaluate(vector)`` takes a full-length medium vector (aligned to
    ``feature_names``), solves the cooperative tradeoff, and returns per-member
    growth in ``target_names`` order, or ``None`` if infeasible. ``active_mask``
    flags which ``feature_names`` this community can actually exchange, so the
    active loop only perturbs real coordinates. The returned ``spec`` is the
    community's :class:`MediumSpec` over those active coordinates -- rebuilt from
    ``spec_json`` (this community's ``medium_spec.json`` entry) when the caller
    has it, recomputed here otherwise, so the active loop proposes candidates from
    the same distribution ``generate`` used. Used by the active-learning loop as
    its expensive oracle.
    """
    community = _build_community(members, solver)
    med_ex = set(_medium_exchanges(community))
    active_mask = np.array([f in med_ex for f in feature_names], dtype=bool)
    active_names = [f for f in feature_names if f in med_ex]
    spec = (
        MediumSpec.from_json(spec_json, active_names)
        if spec_json
        else medium_spec(community, active_names, tradeoff)
    )
    genome_to_taxon = {m.genome_id: m.taxon_id for m in members}
    taxon_order = [genome_to_taxon[g] for g in target_names]

    def evaluate(vector: np.ndarray) -> np.ndarray | None:
        uptake = {f: float(v) for f, v in zip(feature_names, vector, strict=True) if v > 0}
        solution = _solve_sample(community, uptake, tradeoff)
        if solution is None:
            return None
        return solution.members.loc[taxon_order, "growth_rate"].to_numpy(dtype=float)

    return evaluate, active_mask, spec


def _make_design(
    config: GenerateConfig, dim: int, seed: int, spec: MediumSpec | None = None
) -> np.ndarray:
    """Generate the full ``(media_per_community, dim)`` medium design for a community.

    Deterministic in ``seed``, so a media shard can be regenerated identically in
    any worker and sliced -- no need to ship the array between processes.
    """
    if config.sampler == "titrate":
        if spec is None:
            raise ValueError("The 'titrate' sampler needs a MediumSpec.")
        return sampling.titrate_media(
            config.media_per_community,
            dim,
            seed,
            scale=spec.scale(),
            keep_range=(config.keep_min, 1.0),
            essential=spec.essential,
            n_limiting=config.n_limiting,
        )
    if config.sampler == "dirichlet":
        return sampling.dirichlet_sample(config.media_per_community, dim, config.max_uptake, seed)
    if config.sampler == "lhs":
        return sampling.latin_hypercube(config.media_per_community, dim, config.max_uptake, seed)
    if config.sampler == "sparse":
        return sampling.sparse_media(
            config.media_per_community, dim, config.n_active, config.max_uptake, seed
        )
    base = np.full(dim, config.max_uptake, dtype=float)  # perturb the full environment
    return sampling.perturb_media(config.media_per_community, base, seed)


def _shard_ranges(total: int, workers: int) -> list[tuple[int, int]]:
    """Split ``total`` media draws into ``workers`` contiguous ``(start, count)`` ranges."""
    n = max(1, min(workers, total))
    size, rem = divmod(total, n)
    ranges, start = [], 0
    for i in range(n):
        count = size + (1 if i < rem else 0)
        if count:
            ranges.append((start, count))
            start += count
    return ranges


def _work_units(
    n_communities: int,
    media_per_community: int,
    workers: int,
    num_shards: int,
    shard_index: int,
) -> list[tuple[int, int, int]]:
    """Return the ``(community_index, start, count)`` units this HPC shard owns.

    Each community's media are split into ``num_shards * workers`` ranges and the
    flat list of (community, range) pairs is dealt round robin over the shards, so
    a single-community run still fans out. Sharding used to split *communities*
    only, which left every shard but one idle whenever ``n_communities <
    num_shards``. Community indices are never renumbered, so seeds and sample ids
    stay globally consistent.
    """
    ranges = _shard_ranges(media_per_community, max(1, num_shards * workers))
    units = [(ci, start, count) for ci in range(n_communities) for start, count in ranges]
    return [u for i, u in enumerate(units) if i % num_shards == shard_index]


def _run_media_shard(
    members: list[GenomeModel],
    community_index: int,
    config: GenerateConfig,
    start: int,
    count: int,
    spec: MediumSpec | None = None,
) -> dict[str, list[dict]]:
    """Solve ``count`` media (from ``start``) of one community; return long-format rows.

    Runs in a worker process: rebuilds its own community, regenerates the
    deterministic design and takes its ``[start:start+count]`` slice, so media
    *within* a single community parallelise across workers (not just whole
    communities). ``spec`` (the ``titrate`` sampler's calibration + essential set)
    is computed once per community by the parent and passed in, not recomputed per
    work unit -- the scan costs one solve per exchange.
    """
    community = _build_community(members, config.solver)
    med_ex = _medium_exchanges(community)
    taxon_to_genome = {m.taxon_id: m.genome_id for m in members}
    community_id = "+".join(sorted(m.genome_id for m in members))
    seed = config.seed + community_index * 1000
    design = _make_design(config, len(med_ex), seed, spec)[start : start + count]

    out = {"samples": [], "membership": [], "media": [], "member_growth": [], "member_exchange": []}
    for local, vector in enumerate(design):
        draw = start + local
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
        "Community %d media [%d:%d]: %d solved.", community_index, start, start + count, len(design)
    )
    return out


# ---------------------------------------------------------------------------
# Exchange universe (shared coordinate system across all samples)
# ---------------------------------------------------------------------------


def medium_to_member_exchange(medium_id: str) -> str:
    """Map a community pool exchange id (``EX_<met>_m``) to its member id (``EX_<met>_e``).

    micom couples each member's ``EX_<met>_e`` to a shared medium-pool exchange
    ``EX_<met>_m``; the member reactions are renamed with a taxon suffix in the
    merged community, so the pool ids are the stable way to enumerate the
    member-exchange coordinate system. Non-``_m`` ids are returned unchanged.
    """
    return f"{medium_id[:-2]}_e" if medium_id.endswith("_m") else medium_id


def build_exchange_universe(roster: list[GenomeModel], solver: str) -> dict[str, list[str]]:
    """Return the union of medium (``EX_*_m``) and member (``EX_*_e``) exchange ids.

    Built once from the full-roster community so training can align every
    sample's long rows to a fixed coordinate system regardless of membership.
    Member ids are derived from the medium pool (see
    ``medium_to_member_exchange``), since the merged community renames the raw
    ``EX_*_e`` reactions with a taxon suffix.
    """
    community = _build_community(roster, solver)
    medium = sorted(rxn.id for rxn in community.exchanges)
    member = sorted({medium_to_member_exchange(ex) for ex in medium if ex.endswith("_m")})
    return {"medium_exchanges": medium, "member_exchanges": member}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def generate(roster: list[GenomeModel], config: GenerateConfig) -> dict[str, Path]:
    """Generate training tables for a roster; return the written file paths."""
    config.out_dir.mkdir(parents=True, exist_ok=True)
    # The exchange universe is the shared coordinate system for every shard; only
    # shard 0 builds+writes it (a full-roster community build), the merge reuses it.
    if config.shard_index == 0:
        LOGGER.info("Building exchange universe from %d genomes...", len(roster))
        universe = build_exchange_universe(roster, config.solver)
        (config.out_dir / "exchange_universe.json").write_text(json.dumps(universe, indent=2))

    subsets = sampling.sample_membership(
        len(roster), config.n_communities, config.size_range, config.seed
    )
    member_subsets = [[roster[i] for i in idx] for idx in subsets]
    units = _work_units(
        len(member_subsets),
        config.media_per_community,
        config.workers,
        config.num_shards,
        config.shard_index,
    )
    LOGGER.info(
        "Solving %d communities x %d media (%s sampler): %d work units over %d workers...",
        len(member_subsets),
        config.media_per_community,
        config.sampler,
        len(units),
        config.workers,
    )

    # The titrate sampler's calibration + essentiality scan costs ~one solve per
    # exchange, so do it once per community here and hand the result to the
    # workers (they each solve several media ranges of the same community).
    specs: dict[int, MediumSpec] = {}
    spec_blob: dict[str, dict] = {}
    if config.sampler == "titrate":
        for ci in sorted({ci for ci, _, _ in units}):
            members = member_subsets[ci]
            community = _build_community(members, config.solver)
            exchanges = _medium_exchanges(community)
            specs[ci] = medium_spec(
                community,
                exchanges,
                config.tradeoff,
                target_frac=config.target_frac,
                max_bound=config.max_uptake,
            )
            community_id = "+".join(sorted(m.genome_id for m in members))
            spec_blob[community_id] = specs[ci].to_json(exchanges)

    collected = {
        k: [] for k in ("samples", "membership", "media", "member_growth", "member_exchange")
    }

    def absorb(result: dict[str, list[dict]]) -> None:
        for key, rows in result.items():
            collected[key].extend(rows)

    tasks = [(member_subsets[ci], ci, start, count, specs.get(ci)) for ci, start, count in units]
    if config.workers <= 1:
        for members, ci, start, count, spec in tasks:
            absorb(_run_media_shard(members, ci, config, start, count, spec))
    else:
        with ProcessPoolExecutor(max_workers=config.workers) as executor:
            futures = [
                executor.submit(_run_media_shard, members, ci, config, start, count, spec)
                for members, ci, start, count, spec in tasks
            ]
            for future in as_completed(futures):
                absorb(future.result())

    if spec_blob:
        # One entry per community: what the active loop needs to propose media from
        # the same distribution (the merge step unions these across shards).
        (config.out_dir / "medium_spec.json").write_text(json.dumps(spec_blob, indent=2))

    written: dict[str, Path] = {}
    for name, rows in collected.items():
        path = config.out_dir / f"{name}.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        written[name] = path
    if config.shard_index == 0:
        written["exchange_universe"] = config.out_dir / "exchange_universe.json"
    n_feasible = sum(1 for r in collected["samples"] if r["feasible"])
    LOGGER.info(
        "Wrote %d samples (%d feasible) to %s",
        len(collected["samples"]),
        n_feasible,
        config.out_dir,
    )
    return written
