"""The solver-backed medium calibration (slow: needs the micom/cobra stack).

Guards the fix for the run that produced a constant target: growth saturates far
below a fully-open medium, and random nutrient subsets never contain the
essential set, so the sampler must calibrate the uptake bound and keep the
essentials. Run with ``pytest -m slow``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from surrogate_mgem import sampling
from surrogate_mgem.data import (
    MediumSpec,
    _build_community,
    _medium_exchanges,
    _solve_sample,
    medium_spec,
    read_roster,
)

ROSTER = Path(__file__).resolve().parents[1] / "examples/single_mgem/roster.csv"

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def community():
    return _build_community(read_roster(ROSTER), "hybrid")


def test_medium_spec_calibrates_below_saturation_and_finds_essentials(community):
    exchanges = _medium_exchanges(community)
    spec = medium_spec(community, exchanges, tradeoff=0.35, max_bound=1000.0)

    assert 0 < spec.bound < 100.0  # growth saturates by ~100; 1000 is flat
    assert 0 < int(spec.essential.sum()) < len(exchanges)
    lo, hi = spec.bound_range()
    assert 0 < lo < spec.bound < hi

    # Round-tripping through medium_spec.json preserves the mask.
    restored = MediumSpec.from_json(spec.to_json(exchanges), exchanges)
    assert np.array_equal(restored.essential, spec.essential)
    assert restored.bound == pytest.approx(spec.bound)


def test_demand_spans_orders_of_magnitude_and_marks_unused_exchanges(community):
    """The per-nutrient scale is an LP output, and it is not one number.

    Sampling every exchange in one shared band is what made 250 of 259 inputs
    uninformative: the low-demand ones were never limiting, so growth could not
    respond to them.
    """
    exchanges = _medium_exchanges(community)
    spec = medium_spec(community, exchanges, tradeoff=0.35, max_bound=1000.0)
    consumed = spec.demand[spec.demand > 0]

    assert 0 < len(consumed) < len(exchanges)  # some exchanges are never consumed
    assert consumed.max() / consumed.min() > 100  # demands differ by orders of magnitude
    assert (spec.demand[spec.essential] > 0).all()  # essentials always have a scale
    # Never-consumed exchanges are simply not offered.
    design = sampling.titrate_media(8, len(exchanges), seed=0, scale=spec.scale())
    assert (design[:, spec.demand == 0] == 0).all()


def test_titrate_media_actually_grow(community):
    """The whole point: most sampled media grow, and growth varies."""
    exchanges = _medium_exchanges(community)
    spec = medium_spec(community, exchanges, tradeoff=0.35, max_bound=1000.0)
    design = sampling.titrate_media(
        12, len(exchanges), seed=0, scale=spec.scale(), essential=spec.essential
    )
    growth = []
    for vector in design:
        solution = _solve_sample(community, dict(zip(exchanges, vector, strict=True)), 0.35)
        growth.append(0.0 if solution is None else float(solution.growth_rate))
    growth = np.array(growth)
    assert (growth > 0).mean() > 0.5  # sparse/perturb gave 0/8 and 2/8 here
    assert growth[growth > 0].std() > 0.01  # graded, not a single saturated level
