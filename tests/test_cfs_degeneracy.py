"""M1 exchange-degeneracy diagnostic and D4 recommendation (plan §5.2)."""

from __future__ import annotations

import pandas as pd

from cfs.validate.degeneracy import recommend_d4


def _frame(ranges: dict[str, list[float]]) -> pd.DataFrame:
    """Build a survey frame: {exchange: [range per (medium,alpha)]}."""
    rows = []
    for exch, vals in ranges.items():
        for i, r in enumerate(vals):
            rows.append({"medium": i, "alpha": 1.0, "exchange": exch, "range": r})
    return pd.DataFrame(rows)


def test_recommend_clean():
    rec = recommend_d4(_frame({"EX_a": [0.0, 1e-9], "EX_b": [0.0, 0.0]}))
    assert rec["verdict"] == "clean"
    assert "plain FBA" in rec["action"]


def test_recommend_localised():
    # One exchange degenerate in every medium; the rest clean.
    rec = recommend_d4(_frame({"EX_a": [5.0, 5.0, 5.0], "EX_b": [0.0, 0.0, 0.0]}))
    assert rec["verdict"] == "localised"
    assert "EX_a" in rec["degenerate_exchanges"]


def test_recommend_genuine():
    rec = recommend_d4(_frame({f"EX_{i}": [5.0, 5.0, 5.0] for i in range(6)}))
    assert rec["verdict"] == "genuine"


def test_empty_survey_is_unknown():
    assert recommend_d4(pd.DataFrame())["verdict"] == "unknown"


def test_exchange_degeneracy_on_toy():
    """A redundant transporter pair creates a degenerate exchange range."""
    import pytest

    pytest.importorskip("cobra")
    from cobra import Metabolite, Model, Reaction

    from cfs.validate.degeneracy import exchange_degeneracy

    m = Model("toy")
    a_e = Metabolite("a_e", compartment="e")
    a_c = Metabolite("a_c", compartment="c")
    bio = Metabolite("bio_c", compartment="c")
    ex = Reaction("EX_a", lower_bound=-10, upper_bound=1000)
    ex.add_metabolites({a_e: -1})
    # Two identical transporters e->c: FVA on EX_a is pinned, but the pair is a
    # classic alternate-optimum; growth consumes a_c.
    t1 = Reaction("T1", lower_bound=0, upper_bound=1000)
    t1.add_metabolites({a_e: -1, a_c: 1})
    t2 = Reaction("T2", lower_bound=0, upper_bound=1000)
    t2.add_metabolites({a_e: -1, a_c: 1})
    grow = Reaction("Growth", lower_bound=0, upper_bound=1000)
    grow.add_metabolites({a_c: -1, bio: 1})
    sink = Reaction("EX_bio", lower_bound=0, upper_bound=1000)
    sink.add_metabolites({bio: -1})
    m.add_reactions([ex, t1, t2, grow, sink])
    m.objective = "Growth"
    ranges = exchange_degeneracy(m, alpha=1.0)
    assert (ranges >= -1e-9).all()  # ranges are non-negative
    assert "EX_a" in ranges.index
