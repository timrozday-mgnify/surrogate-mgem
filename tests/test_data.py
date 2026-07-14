"""Solver-free tests for data.py helpers (the micom path is covered by slow tests)."""

import pandas as pd
import pytest

from surrogate_mgem.data import GenomeModel, _member_exchange_rows, read_roster


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
    }
