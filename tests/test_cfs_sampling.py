"""§4 sampling design: active subspace, stratified-Sobol media, generate shards."""

from __future__ import annotations

import pytest

from cfs.groundtruth.solve import load_km_defaults
from cfs.sampling.active_subspace import ActiveSubspace
from cfs.sampling.design import SamplingConfig, sample_media


def _toy_two_uptakes():
    """Growth limited by 'a'; 'b' is takeable but feeds nothing (inert)."""
    from cobra import Metabolite, Model, Reaction

    m = Model("toy2")
    a_e, a_c = Metabolite("a_e", compartment="e"), Metabolite("a_c", compartment="c")
    b_e, b_c = Metabolite("b_e", compartment="e"), Metabolite("b_c", compartment="c")
    bio = Metabolite("bio_c", compartment="c")
    rxns = []
    for mid, met in (("EX_a_e", a_e), ("EX_b_e", b_e)):
        ex = Reaction(mid, lower_bound=-10.0, upper_bound=1000.0)
        ex.add_metabolites({met: -1})
        rxns.append(ex)
    ta = Reaction("Ta", lower_bound=0, upper_bound=1000)
    ta.add_metabolites({a_e: -1, a_c: 1})
    tb = Reaction("Tb", lower_bound=0, upper_bound=1000)
    tb.add_metabolites({b_e: -1, b_c: 1})
    grow = Reaction("Growth", lower_bound=0, upper_bound=1000)
    grow.add_metabolites({a_c: -1, bio: 1})
    exb = Reaction("EX_bio_e", lower_bound=0, upper_bound=1000)
    exb.add_metabolites({bio: -1})
    m.add_reactions([*rxns, ta, tb, grow, exb])
    m.objective = "Growth"
    return m


def test_sample_media_shape_and_below_km_fraction():
    # One active metabolite (Km = default 0.01); the below-Km stratum is exact.
    sub = ActiveSubspace("g", ["EX_a_e"], [], {"EX_a_e": 1.0}, 10.0)
    cfg = SamplingConfig(n_media=100, seed=1)
    media = sample_media(sub, load_km_defaults(), cfg)

    assert len(media) == cfg.n_media
    conc = [m["EX_a_e"] for m in media]
    assert sum(c == 0.0 for c in conc) == 1  # exactly one all-but-one-depleted corner
    km = 0.01
    n_bulk = cfg.n_media - 1
    assert sum(0.0 < c < km for c in conc) == round(cfg.frac_below_km * n_bulk)
    assert max(conc) <= km * 10.0 ** cfg.log10_hi + 1e-9  # capped at the rich level


def test_sample_media_falls_back_to_background_when_no_active():
    sub = ActiveSubspace("g", [], ["EX_x_e"], {"EX_x_e": 0.0}, 5.0)
    media = sample_media(sub, load_km_defaults(), SamplingConfig(n_media=10))
    assert len(media) == 10 and all("EX_x_e" in m for m in media)


def test_active_subspace_separates_essential_from_inert():
    pytest.importorskip("cobra")
    from cfs.sampling.active_subspace import active_subspace

    sub = active_subspace(_toy_two_uptakes(), "toy2", tol=1e-3)
    assert sub.active == ["EX_a_e"]
    assert "EX_b_e" in sub.background
    assert sub.mu_rich == pytest.approx(10.0, rel=1e-3)


def test_generate_organism_writes_readable_shards(tmp_path):
    pytest.importorskip("highspy")
    pytest.importorskip("pyarrow")
    import pandas as pd

    from cfs.sampling.generate import generate_organism

    model = _toy_two_uptakes()
    cfg = SamplingConfig(n_media=4, alphas=(0.0, 1.0), seed=0)
    shards = generate_organism(model, "toy2", "deadbeef", tmp_path, cfg)

    assert shards.n_media == 4
    assert len(shards.paths) == len(cfg.eps_levels)  # one shard per eps (§4.5)
    df = pd.read_parquet(shards.paths[cfg.eps_primary_idx])
    assert len(df) == 4 * len(cfg.alphas)  # full media set at the primary eps
    assert set(df.columns) >= {"genome_id", "index_hash", "alpha", "eps", "mu_max", "z"}
    assert len(df["z"].iloc[0]) == len(model.exchanges)  # z aligned to this organism's M_i
    assert (df["index_hash"] == "deadbeef").all()  # P13 provenance on every row
