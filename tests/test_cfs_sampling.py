"""§4 sampling design: active subspace, stratified-Sobol media, generate shards."""

from __future__ import annotations

import json
from collections import Counter

import pytest

from cfs.groundtruth.solve import load_km_defaults
from cfs.sampling.active_subspace import ActiveSubspace
from cfs.sampling.design import SamplingConfig, limiting_scales, sample_media


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
    # One active metabolite (Km = default 0.01). With d == 1 the focus stratum is
    # itself the below-Km stratum, so every focus medium lands below Km too.
    sub = ActiveSubspace("g", ["EX_a_e"], [], {"EX_a_e": 1.0}, 10.0)
    cfg = SamplingConfig(n_media=100, seed=1)
    media = sample_media(sub, load_km_defaults(), cfg)

    assert len(media) == cfg.n_media
    conc = [m["EX_a_e"] for m in media]
    assert sum(c == 0.0 for c in conc) == 1  # exactly one all-but-one-depleted corner
    km = 0.01
    n_bulk = cfg.n_media - 1
    # Without `scales` the focus stratum has no anchor to aim at, so it spans
    # [-1.5, 0.5] decades around Km rather than the full [-4, 0) below-Km band.
    assert sum(0.0 < c < km for c in conc) > 0
    assert max(conc) <= km * 10.0 ** cfg.log10_hi + 1e-9  # capped at the rich level


def test_per_metabolite_bands_flatten_which_metabolite_limits():
    """The M3 fix. A metabolite limits when it is the scarcest *relative to where
    it starts limiting* — ``c / (Km * s)`` minimal, ``s`` its own regime. On the
    real roster ``s`` spans 5107x, so a shared band hands nearly every limiting
    medium to whichever metabolite has the smallest ``s``.

    Here ``s`` spans four decades across 8 metabolites. Without ``scales`` the
    design cannot see that; with ``scales`` every band is centred on its own
    regime and coverage flattens. Solver-free: ``u = c/(Km+c)`` is monotone in
    ``c/Km``, so the argmin is the metabolite the LP is most likely to bind on.
    """
    ex = [f"EX_m{i}_e" for i in range(8)]
    sub = ActiveSubspace("g", ex, [], dict.fromkeys(ex, 1.0), 10.0)
    km_cfg = load_km_defaults()
    # True limiting regimes, 1e-4 .. 1e0 in c/Km — the roster's actual spread.
    s = {e: 10.0 ** (-4.0 * i / 7) for i, e in enumerate(ex)}
    u_star = {e: v / (1.0 + v) for e, v in s.items()}
    assert limiting_scales(u_star) == pytest.approx(s, rel=1e-9)

    n_media = 2048

    def coverage(scales):
        media = sample_media(sub, km_cfg, SamplingConfig(n_media=n_media, seed=1), scales)
        counts = Counter(min(ex, key=lambda e: media_i[e] / s[e]) for media_i in media)
        n = [counts.get(e, 0) for e in ex]
        return min(n) / n_media, n

    flat, n_flat = coverage(None)
    aimed, n_aimed = coverage(limiting_scales(u_star))
    # The floor per metabolite is the quantity that matters, not min/max: the
    # unfocused half of the budget still samples multi-limitation media on
    # purpose and lands on the steepest metabolite, so the *ratio* stays skewed
    # by design. What must not happen is a metabolite getting ~no media at all.
    assert flat < 0.002, (flat, n_flat)  # shared band: the rarest gets 1 medium in 2048
    assert aimed > 0.04, (aimed, n_aimed)  # own bands: every metabolite gets a real share


def test_focus_strata_need_scales_to_help():
    """``frac_focus`` reserves budget per metabolite, but the reserved media only
    land in the right regime once ``scales`` says where that regime is. Budget
    without aim is the honest contract here — asserted so nobody assumes
    ``frac_focus`` alone fixes coverage."""
    ex = [f"EX_m{i}_e" for i in range(8)]
    sub = ActiveSubspace("g", ex, [], dict.fromkeys(ex, 1.0), 10.0)
    km_cfg = load_km_defaults()
    s = {e: 10.0 ** (-4.0 * i / 7) for i, e in enumerate(ex)}
    u_star = {e: v / (1.0 + v) for e, v in s.items()}

    def coverage(scales, frac_focus=0.5):
        cfg = SamplingConfig(n_media=2048, frac_focus=frac_focus, seed=1)
        media = sample_media(sub, km_cfg, cfg, scales)
        counts = Counter(min(ex, key=lambda e: media_i[e] / s[e]) for media_i in media)
        return [counts.get(e, 0) for e in ex]

    assert min(coverage(None)) <= 2  # budget alone: the small-s metabolites starve
    assert min(coverage(limiting_scales(u_star))) > 20  # aimed budget reaches them


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


def test_demand_probe_anchors_the_limiting_metabolite_without_labels():
    """§4.7: the band comes from an LP on *this* model, not a previous run."""
    pytest.importorskip("cobra")
    from cfs.sampling.active_subspace import demand_probe

    model = _toy_two_uptakes()
    probe = demand_probe(model, ["EX_a_e", "EX_b_e"], load_km_defaults(), steps=14)

    # 'a' limits growth through MM: mu = vmax * c/(Km+c), so the half-way point of
    # its own mu range sits within a decade of Km, i.e. scale ~ 1.
    assert 0.1 < probe["EX_a_e"] < 10.0
    # 'b' feeds nothing: no limiting regime, so no anchor — the chain decides.
    assert "EX_b_e" not in probe


def test_band_scales_follows_the_stated_fallback_chain():
    from cfs.sampling.design import band_scales

    scales, source = band_scales(
        probe={"EX_a_e": 0.5},
        previous={"EX_a_e": 9.0, "EX_b_e": 2.0},
        roster_median={"EX_b_e": 3.0, "EX_c_e": 4.0},
        exchanges=["EX_a_e", "EX_b_e", "EX_c_e", "EX_d_e"],
    )
    assert scales == {"EX_a_e": 0.5, "EX_b_e": 2.0, "EX_c_e": 4.0, "EX_d_e": 1.0}
    assert source == {"EX_a_e": "probe", "EX_b_e": "previous",
                      "EX_c_e": "roster_median", "EX_d_e": "default"}


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

    # §4.7: every sampled metabolite records where its band came from.
    blob = json.loads((tmp_path / "toy2.subspace.json").read_text())
    assert blob["toy2"]["bands"]["EX_a_e"]["source"] == "probe"


def test_topup_round_appends_a_shard_and_both_rounds_load(tmp_path):
    """§4.6: a top-up round must add to the label set, not overwrite it."""
    pytest.importorskip("highspy")
    pytest.importorskip("pyarrow")
    import pandas as pd

    from cfs.sampling.generate import generate_organism

    model = _toy_two_uptakes()
    cfg = SamplingConfig(n_media=4, alphas=(1.0,), eps_levels=(1e-3,), eps_primary_idx=0,
                         probe=False, seed=0)
    base = generate_organism(model, "toy2", "deadbeef", tmp_path, cfg)
    topup = generate_organism(model, "toy2", "deadbeef", tmp_path, cfg, round_idx=1)

    assert base.paths[0] != topup.paths[0] and base.paths[0].exists()
    both = pd.concat([pd.read_parquet(p) for p in base.paths + topup.paths])
    # Media ids stay disjoint: the train/val split is by medium_id.
    assert both["medium_id"].nunique() == len(both)
