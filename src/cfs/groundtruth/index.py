"""M0 metabolite universe: deterministic derivation, partition, freeze (plan §2).

The shared metabolite universe (D2) is derived, not curated. Only ``shared``
metabolites (exchangeable by ≥2 organisms) enter the composition coupling, so the
Newton Jacobian is ``|shared| × |shared|``, not the full universe. The index is
frozen to ``config/metabolite_index.json`` and hashed: silently changing it later
invalidates every trained checkpoint with no error (plan §2.2, P13), so every
downstream artefact records this hash.

COBRApy only. ``cobra`` (in the ``data`` extra) is imported by callers that pass
loaded models; this module only touches ``model.exchanges`` ids.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class IndexResult:
    """The frozen metabolite universe and per-organism exchange mask."""

    genome_ids: list[str]
    index: list[str]  # all exchange ids, sorted — the M coordinates
    shared: list[str]  # exchangeable by ≥2 organisms (the coupling set)
    private: list[str]  # exchangeable by exactly 1 organism
    mask: np.ndarray  # (N, M) bool: organism n can exchange metabolite index[m]


def derive_index(genome_ids: list[str], exchanges_per_model: list[set[str]]) -> IndexResult:
    """Derive the universe and partition it (plan §2.1).

    ``exchanges_per_model[i]`` is the set of exchange-reaction ids for
    ``genome_ids[i]``. Kept model-free (takes id sets, not cobra objects) so it is
    trivially unit-testable and callable from either the CLI or a loaded roster.
    """
    if len(genome_ids) != len(exchanges_per_model):
        raise ValueError("genome_ids and exchanges_per_model length mismatch")
    index = sorted(set().union(*exchanges_per_model)) if exchanges_per_model else []
    counts = {ex: sum(ex in s for s in exchanges_per_model) for ex in index}
    shared = [ex for ex in index if counts[ex] >= 2]
    private = [ex for ex in index if counts[ex] == 1]
    mask = np.array([[ex in s for ex in index] for s in exchanges_per_model], dtype=bool).reshape(
        len(genome_ids), len(index)
    )
    return IndexResult(list(genome_ids), index, shared, private, mask)


def derive_index_from_roster(roster) -> IndexResult:
    """Load each roster model and derive the index (plan §2.1)."""
    from cobra.io import read_sbml_model

    genome_ids, exchanges = [], []
    for gm in roster:
        model = read_sbml_model(str(gm.model_path))
        genome_ids.append(gm.genome_id)
        exchanges.append({rxn.id for rxn in model.exchanges})
    return derive_index(genome_ids, exchanges)


def _index_dict(result: IndexResult) -> dict:
    return {
        "genome_ids": result.genome_ids,
        "index": result.index,
        "shared": result.shared,
        "private": result.private,
        "mask": result.mask.astype(int).tolist(),
    }


def write_index(result: IndexResult, path: Path) -> str:
    """Write ``metabolite_index.json`` and return its content hash (plan §2.2)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = _index_dict(result)
    text = json.dumps(blob, indent=2, sort_keys=True)
    path.write_text(text)
    return _hash_text(text)


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def index_hash(path: Path) -> str:
    """Hash of a written index file — canonicalised so formatting can't change it."""
    blob = json.loads(Path(path).read_text())
    return _hash_text(json.dumps(blob, indent=2, sort_keys=True))


def load_index(path: Path) -> IndexResult:
    """Load a frozen index back into an :class:`IndexResult`."""
    blob = json.loads(Path(path).read_text())
    return IndexResult(
        genome_ids=blob["genome_ids"],
        index=blob["index"],
        shared=blob["shared"],
        private=blob["private"],
        mask=np.array(blob["mask"], dtype=bool).reshape(
            len(blob["genome_ids"]), len(blob["index"])
        ),
    )
