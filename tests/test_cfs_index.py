"""M0 metabolite-index derivation, partition, and hash stability (plan §2)."""

from __future__ import annotations

import numpy as np

from cfs.groundtruth.index import (
    derive_index,
    index_hash,
    load_index,
    write_index,
)


def test_partition_and_mask():
    # A,B share EX_a; EX_b private to A; EX_c private to B.
    res = derive_index(["A", "B"], [{"EX_a", "EX_b"}, {"EX_a", "EX_c"}])
    assert res.index == ["EX_a", "EX_b", "EX_c"]
    assert res.shared == ["EX_a"]
    assert sorted(res.private) == ["EX_b", "EX_c"]
    assert res.mask.shape == (2, 3)
    # Row A can exchange EX_a, EX_b but not EX_c.
    np.testing.assert_array_equal(res.mask[0], [True, True, False])
    np.testing.assert_array_equal(res.mask[1], [True, False, True])


def test_hash_is_stable_and_sensitive(tmp_path):
    res = derive_index(["A", "B"], [{"EX_a", "EX_b"}, {"EX_a", "EX_c"}])
    p = tmp_path / "idx.json"
    digest = write_index(res, p)
    # Same content -> same hash (re-hash of the written file matches).
    assert index_hash(p) == digest
    # Round-trips.
    back = load_index(p)
    assert back.shared == res.shared
    assert back.mask.shape == res.mask.shape
    # A different index yields a different hash (P13: silent drift must be visible).
    other = derive_index(["A", "B"], [{"EX_a"}, {"EX_a", "EX_c"}])
    assert write_index(other, tmp_path / "other.json") != digest
