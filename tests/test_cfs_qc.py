"""M0 EGC pre-flight (plan §3.0). Uses in-memory cobra models (needs cobra)."""

from __future__ import annotations

import pytest

cobra = pytest.importorskip("cobra")
from cobra import Metabolite, Model, Reaction  # noqa: E402

from cfs.groundtruth.qc import has_egc  # noqa: E402


def _egc_model() -> Model:
    """A model that makes ATP from nothing: an EGC the gate must catch."""
    m = Model("egc")
    atp = Metabolite("atp_c", compartment="c")
    gen = Reaction("GEN", lower_bound=0, upper_bound=1000)  # atp <- nothing
    gen.add_metabolites({atp: 1})
    atpm = Reaction("ATPM", lower_bound=0, upper_bound=1000)
    atpm.add_metabolites({atp: -1})
    m.add_reactions([gen, atpm])
    return m


def _clean_model() -> Model:
    """No free ATP: ATP only from an uptaken substrate. Not an EGC."""
    m = Model("clean")
    a = Metabolite("a_e", compartment="e")
    atp = Metabolite("atp_c", compartment="c")
    ex = Reaction("EX_a", lower_bound=-10, upper_bound=1000)
    ex.add_metabolites({a: -1})
    conv = Reaction("CONV", lower_bound=0, upper_bound=1000)
    conv.add_metabolites({a: -1, atp: 1})
    atpm = Reaction("ATPM", lower_bound=0, upper_bound=1000)
    atpm.add_metabolites({atp: -1})
    m.add_reactions([ex, conv, atpm])
    return m


def test_egc_detected():
    assert has_egc(_egc_model()) is True


def test_clean_model_passes():
    assert has_egc(_clean_model()) is False


def test_missing_atpm_returns_false():
    m = Model("no_atpm")
    a = Metabolite("a_e", compartment="e")
    ex = Reaction("EX_a", lower_bound=-10, upper_bound=1000)
    ex.add_metabolites({a: -1})
    m.add_reactions([ex])
    assert has_egc(m) is False  # cannot pose the test -> conservatively False
