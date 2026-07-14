"""Per-genome encoder: turn a genome's exchange capabilities into features.

For the composable (Phase 2) surrogate, each genome is represented by a fixed
feature vector derived from its *own* model — the set of extracellular
metabolites it can import and/or export. Because the vector comes from the model
alone (no community, no training data), a brand-new genome gets a representation
with no retraining, which is what lets the community surrogate add/drop/swap
members.

Capability extraction (reading SBML via cobra) is separated from vectorisation
(pure numpy over a fixed exchange universe) so the pure part is unit-tested
without the solver stack.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Capability codes per member exchange, over the shared universe.
CAN_IMPORT = 1
CAN_EXPORT = 2


@dataclass(frozen=True)
class GenomeCapabilities:
    """Which metabolites a genome can import / export, as member-exchange id sets."""

    genome_id: str
    importable: frozenset[str]  # member exchange ids (EX_*_e) with a possible uptake
    exportable: frozenset[str]  # member exchange ids (EX_*_e) with a possible secretion


def read_capabilities(genome_id: str, model_path: Path) -> GenomeCapabilities:
    """Read a genome's exchange capabilities from its SBML (cobra imported here).

    An exchange reaction can import when its lower bound < 0 and export when its
    upper bound > 0 — the direction(s) the network is allowed to run it, i.e. the
    capability, independent of any particular medium.
    """
    import cobra

    model = cobra.io.read_sbml_model(str(model_path))
    importable, exportable = set(), set()
    for rxn in model.exchanges:
        ex_id = rxn.id if rxn.id.endswith("_e") else f"{rxn.id}_e"
        if rxn.lower_bound < 0:
            importable.add(ex_id)
        if rxn.upper_bound > 0:
            exportable.add(ex_id)
    return GenomeCapabilities(genome_id, frozenset(importable), frozenset(exportable))


def capability_vector(caps: GenomeCapabilities, member_exchanges: list[str]) -> np.ndarray:
    """Encode capabilities as a fixed-length vector over ``member_exchanges``.

    Each coordinate is a bitmask: ``CAN_IMPORT`` and/or ``CAN_EXPORT`` (0 if the
    genome cannot exchange that metabolite at all). Pure -- the coordinate system
    is the shared exchange universe, so vectors from different genomes align.
    """
    vec = np.zeros(len(member_exchanges), dtype=np.float32)
    for i, ex in enumerate(member_exchanges):
        code = 0
        if ex in caps.importable:
            code |= CAN_IMPORT
        if ex in caps.exportable:
            code |= CAN_EXPORT
        vec[i] = float(code)
    return vec


def capability_matrix(
    capabilities: list[GenomeCapabilities], member_exchanges: list[str]
) -> np.ndarray:
    """Stack per-genome capability vectors into a ``(n_genomes, n_exchanges)`` matrix."""
    return np.stack([capability_vector(c, member_exchanges) for c in capabilities])
