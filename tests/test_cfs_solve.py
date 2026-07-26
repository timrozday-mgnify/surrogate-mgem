"""M2 ground-truth solve: MM bounds, Km classification, elastic-net QP (§3.2-3.4)."""

from __future__ import annotations

import numpy as np
import pytest

from cfs.groundtruth.solve import km_for_exchange, load_km_defaults, mm_lower_bound


def test_mm_lower_bound():
    assert mm_lower_bound(10.0, 0.0, 1.0) == 0.0        # no substrate -> no uptake
    assert mm_lower_bound(10.0, 1.0, 1.0) == -5.0       # c == Km -> half Vmax
    assert mm_lower_bound(10.0, 1e9, 1.0) == pytest.approx(-10.0, rel=1e-6)  # saturated


def test_km_classification():
    km = load_km_defaults()
    assert km_for_exchange("EX_glc__D_e", km) == km["classes"]["sugar"]
    assert km_for_exchange("EX_ala__L_e", km) == km["classes"]["amino_acid"]
    assert km_for_exchange("EX_o2_e", km) == km["classes"]["gas"]
    assert km_for_exchange("EX_na1_e", km) == km["classes"]["ion"]
    assert km_for_exchange("EX_notarealmet_e", km) == km["default"]  # fallback


def _toy_model():
    """A minimal model whose growth is limited by the uptake of 'a'."""
    from cobra import Metabolite, Model, Reaction

    m = Model("toy")
    a_e = Metabolite("a_e", compartment="e")
    a_c = Metabolite("a_c", compartment="c")
    bio = Metabolite("bio_c", compartment="c")
    ex_a = Reaction("EX_a_e", lower_bound=-10.0, upper_bound=1000.0)
    ex_a.add_metabolites({a_e: -1})
    t = Reaction("T", lower_bound=0.0, upper_bound=1000.0)
    t.add_metabolites({a_e: -1, a_c: 1})
    grow = Reaction("Growth", lower_bound=0.0, upper_bound=1000.0)
    grow.add_metabolites({a_c: -1, bio: 1})
    ex_bio = Reaction("EX_bio_e", lower_bound=0.0, upper_bound=1000.0)
    ex_bio.add_metabolites({bio: -1})
    m.add_reactions([ex_a, t, grow, ex_bio])
    m.objective = "Growth"
    return m


def test_solve_toy_optimal_and_at_target():
    pytest.importorskip("highspy")
    from cfs.groundtruth.solve import solve

    m = _toy_model()
    # Rich 'a': MM bound saturates near Vmax=10, so mu_max ~ 10.
    s = solve(m, {"EX_a_e": 1e6}, alpha=0.7, eps=1e-3)
    assert s.status == "optimal"
    assert s.mu_max == pytest.approx(10.0, rel=1e-4)
    # Growth is pinned to alpha*mu_max, and 'a' is taken up to feed it.
    assert s.fluxes[[r.id for r in m.reactions].index("Growth")] == pytest.approx(7.0, rel=1e-4)
    assert s.z["EX_a_e"] == pytest.approx(-7.0, rel=1e-3)  # uptake (negative)


def test_solve_repeatable_and_lipschitz():
    pytest.importorskip("highspy")
    from cfs.groundtruth.solve import solve

    m = _toy_model()
    conc = {"EX_a_e": 5.0}
    a = solve(m, conc, 0.7, 1e-3)
    b = solve(m, conc, 0.7, 1e-3)
    assert np.allclose(a.fluxes, b.fluxes, atol=1e-6)  # repeatable to solver tol
    # Lipschitz: a small change in c gives a small change in z (elastic net is smooth).
    c = solve(m, {"EX_a_e": 5.0 + 1e-4}, 0.7, 1e-3)
    dz = max(abs(a.z.get(k, 0) - c.z.get(k, 0)) for k in set(a.z) | set(c.z))
    assert dz < 1e-2  # O(dc), not O(1)


def test_shadow_price_matches_finite_difference():
    pytest.importorskip("highspy")
    from cfs.groundtruth.solve import solve

    m = _toy_model()
    # Saturate 'a' so mu_max is at the raw uptake bound; then FD the raw bound to
    # match the (MM-saturated) shadow price without the MM slope confounding it.
    s = solve(m, {"EX_a_e": 1e12}, alpha=1.0, eps=1e-3)  # 'a' uptake is binding
    shadow = s.shadow_prices["EX_a_e"]  # dual of a_e mass balance = dmu_max/d supply
    with m:
        r = m.reactions.get_by_id("EX_a_e")
        h = 1e-3
        r.lower_bound = r.lower_bound - h  # allow h more uptake
        mu2 = m.optimize().objective_value
    fd = (mu2 - s.mu_max) / (-h)  # dmu_max/d(lower bound)
    assert shadow == pytest.approx(fd, abs=1e-2)
