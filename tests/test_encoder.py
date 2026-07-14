"""Tests for the pure capability-vectorisation (SBML reading is a slow test)."""

import numpy as np

from surrogate_mgem.encoder import (
    CAN_EXPORT,
    CAN_IMPORT,
    GenomeCapabilities,
    capability_matrix,
    capability_vector,
)


def _caps(genome_id, imp, exp):
    return GenomeCapabilities(genome_id, frozenset(imp), frozenset(exp))


def test_capability_vector_bitmask():
    universe = ["EX_a_e", "EX_b_e", "EX_c_e", "EX_d_e"]
    caps = _caps("g1", imp={"EX_a_e", "EX_b_e"}, exp={"EX_b_e", "EX_c_e"})
    vec = capability_vector(caps, universe)
    assert vec.tolist() == [
        float(CAN_IMPORT),  # a: import only
        float(CAN_IMPORT | CAN_EXPORT),  # b: both
        float(CAN_EXPORT),  # c: export only
        0.0,  # d: neither
    ]


def test_capability_matrix_aligns_genomes():
    universe = ["EX_a_e", "EX_b_e"]
    mat = capability_matrix(
        [_caps("g1", {"EX_a_e"}, set()), _caps("g2", set(), {"EX_b_e"})], universe
    )
    assert mat.shape == (2, 2)
    assert np.array_equal(mat, [[CAN_IMPORT, 0], [0, CAN_EXPORT]])
