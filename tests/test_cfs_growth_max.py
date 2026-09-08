"""§13.2 / M10 — the projection and the ascent, without a checkpoint."""

from types import SimpleNamespace

import numpy as np
import pytest

from cfs.science.growth import maximise, project


def test_projection_respects_box_and_budget():
    rng = np.random.default_rng(0)
    cost = rng.uniform(0.5, 2.0, 40)
    c_max = rng.uniform(1.0, 5.0, 40)
    y = rng.uniform(-1.0, 8.0, 40)
    c_lo = np.zeros(40)
    c = project(y, cost, 10.0, c_lo, c_max)
    assert (c >= -1e-12).all() and (c <= c_max + 1e-12).all()
    assert cost @ c <= 10.0 + 1e-6
    # A feasible point is its own projection.
    inside = 0.5 * c_max * min(1.0, 10.0 / (cost @ c_max))
    assert np.allclose(project(inside, cost, 10.0, c_lo, c_max), inside)
    # A lower bound is honoured, and eats the budget first.
    floor = 0.2 * c_max
    c = project(y, cost, float(cost @ floor) + 1.0, floor, c_max)
    assert (c >= floor - 1e-12).all()


class _Head:
    """A stand-in Surrogate: mu = sum_m weight_m * u_m, concave in c, with the
    exact interface `mu_and_grad` uses."""

    def __init__(self, w, km):
        self.w, self.km = w, km
        self.mu_scale = np.array([1.0])
        self.x_scale = np.ones((1, len(w)))
        self.mod = SimpleNamespace(batched_value_and_grad=self._vg)
        self._jnp = np
        self._vheads = None

    def _x(self, c):
        u = c / (self.km + c)
        return (u / (u + self.x_scale[0]))[None, None, :]

    def _vg(self, _heads, x):
        # mu = w . u, and u = s*x/(1-x) with s = 1, so d mu/dx = w/(1-x)^2.
        u = x[0, 0] / (1.0 - x[0, 0])
        return np.array([[float(self.w @ u)]]), (self.w / (1.0 - x[0, 0]) ** 2)[None, None, :]


def test_ascent_spends_the_budget_on_the_valuable_metabolite():
    km = np.array([1.0, 1.0, 1.0])
    sur = _Head(np.array([10.0, 1.0, 0.1]), km)
    c0 = np.array([1.0, 1.0, 1.0])
    cost = np.ones(3)
    c_star, path = maximise(
        sur, c0, 0, cost=cost, budget=3.0, c_lo=np.zeros(3), c_hi=np.full(3, 10.0), iters=400
    )
    assert path[-1] > path[0]
    assert cost @ c_star <= 3.0 + 1e-6
    assert c_star[0] > c_star[1] > c_star[2]


def _toy():
    """One nutrient, one exchange, biomass = uptake. `mu_true` is the uptake bound."""
    cobra = pytest.importorskip("cobra")
    m = cobra.Model("toy")
    a = cobra.Metabolite("a_e")
    ex = cobra.Reaction("EX_a_e", lower_bound=-10.0, upper_bound=1000.0)
    ex.add_metabolites({a: -1.0})  # +ve flux = secretion, -ve = uptake
    bio = cobra.Reaction("BIOMASS", lower_bound=0.0, upper_bound=1000.0)
    bio.add_metabolites({a: -1.0})
    m.add_reactions([ex, bio])
    m.objective = bio
    return m


def test_mu_lower_is_a_bound_and_uses_the_predicted_uptake():
    from cfs.science.growth import mu_lower, mu_true

    m, c = _toy(), np.array([1e6])  # replete, so §3.3's bound is the model's own
    assert mu_true(m, ["EX_a_e"], c) == pytest.approx(10.0, rel=1e-6)
    # Head B predicting an uptake of 3 restricts the LP to a feasible subset.
    assert mu_lower(m, ["EX_a_e"], c, np.array([-3.0])) == pytest.approx(3.0, rel=1e-6)
    # Tighten only: a prediction *beyond* the MM bound cannot loosen it.
    assert mu_lower(m, ["EX_a_e"], c, np.array([-99.0])) == pytest.approx(10.0, rel=1e-6)
