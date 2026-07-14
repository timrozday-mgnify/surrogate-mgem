"""Solver-free tests for data.py helpers (the micom path is covered by slow tests)."""

import numpy as np
import pandas as pd
import pytest

from surrogate_mgem.data import (
    GenomeModel,
    _member_exchange_rows,
    _shard_ranges,
    medium_to_member_exchange,
    read_roster,
)


def test_shard_ranges_partition_media_evenly():
    # 100 media over 8 workers: contiguous, non-overlapping, covers everything.
    ranges = _shard_ranges(100, 8)
    assert len(ranges) == 8
    assert ranges[0][0] == 0
    total = sum(count for _, count in ranges)
    assert total == 100
    # contiguity: each start == previous start + previous count
    for (s0, c0), (s1, _c1) in zip(ranges, ranges[1:], strict=False):
        assert s1 == s0 + c0
    # remainder distributed to the first shards (sizes differ by at most 1)
    counts = [c for _, c in ranges]
    assert max(counts) - min(counts) <= 1


def test_shard_ranges_more_workers_than_media():
    ranges = _shard_ranges(3, 8)
    assert sum(c for _, c in ranges) == 3
    assert all(c >= 1 for _, c in ranges)  # no empty shards


def test_medium_to_member_exchange():
    assert medium_to_member_exchange("EX_glc__D_m") == "EX_glc__D_e"
    assert medium_to_member_exchange("EX_weird") == "EX_weird"  # non-_m left as-is


def test_taxon_id_sanitises_ids():
    assert GenomeModel("GCF_000.1-x", "m.xml").taxon_id == "GCF_000_1_x"


def test_read_roster_resolves_relative_paths(tmp_path):
    (tmp_path / "roster.csv").write_text("genome_id,model_path\ng1,models/g1.xml\ng2,/abs/g2.xml\n")
    roster = read_roster(tmp_path / "roster.csv")
    assert [g.genome_id for g in roster] == ["g1", "g2"]
    assert roster[0].model_path == (tmp_path / "models/g1.xml").resolve()
    assert str(roster[1].model_path) == "/abs/g2.xml"  # absolute left untouched


def test_read_roster_rejects_missing_columns(tmp_path):
    (tmp_path / "bad.csv").write_text("genome_id\ng1\n")
    with pytest.raises(ValueError, match="model_path"):
        read_roster(tmp_path / "bad.csv")


def test_member_exchange_rows_keeps_only_signed_exchanges():
    # Rows: two taxa; columns include an EX_*_e exchange, an internal rxn, and a tiny flux.
    fluxes = pd.DataFrame(
        {
            "EX_glc__D_e": [-5.0, 2.0],
            "EX_ac_e": [1e-12, 3.0],  # below eps for taxon 0 -> dropped
            "EX_o2_e": [np.nan, 4.0],  # NaN for taxon 0 (no such reaction) -> dropped
            "PGI": [10.0, -4.0],  # internal, never an exchange
        },
        index=["t1", "t2"],
    )
    rows = _member_exchange_rows(0, {"t1": "g1", "t2": "g2"}, fluxes)
    got = {(r["genome_id"], r["exchange_id"]): r["flux"] for r in rows}
    assert got == {
        ("g1", "EX_glc__D_e"): -5.0,
        ("g2", "EX_glc__D_e"): 2.0,
        ("g2", "EX_ac_e"): 3.0,
        ("g2", "EX_o2_e"): 4.0,
    }
