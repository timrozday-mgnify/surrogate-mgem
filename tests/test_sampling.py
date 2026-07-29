"""Unit tests for the pure media/membership samplers (no solver)."""

import numpy as np
import pytest

from surrogate_mgem import sampling


def test_latin_hypercube_shape_and_bounds():
    design = sampling.latin_hypercube(n=32, dim=5, max_uptake=1000.0, seed=1)
    assert design.shape == (32, 5)
    assert design.min() >= 0.0
    assert design.max() <= 1000.0
    # Each column is stratified: exactly one sample per 1/n band.
    bands = np.floor(design[:, 0] / 1000.0 * 32).astype(int)
    assert sorted(bands) == list(range(32))


def test_latin_hypercube_degenerate():
    assert sampling.latin_hypercube(0, 3, 1.0, 0).shape == (0, 3)
    assert sampling.latin_hypercube(4, 0, 1.0, 0).shape == (4, 0)


def test_dirichlet_rows_sum_to_budget():
    design = sampling.dirichlet_sample(n=20, dim=4, total=500.0, seed=2)
    assert design.shape == (20, 4)
    assert np.allclose(design.sum(axis=1), 500.0)
    assert design.min() >= 0.0


def test_sparse_media_has_exactly_n_active_components():
    design = sampling.sparse_media(n=30, dim=100, n_active=8, max_uptake=500.0, seed=4)
    assert design.shape == (30, 100)
    nonzero_per_row = (design > 0).sum(axis=1)
    assert (nonzero_per_row == 8).all()  # each medium activates exactly n_active
    assert design.max() <= 500.0


def test_sparse_media_clamps_n_active_to_dim():
    design = sampling.sparse_media(n=5, dim=3, n_active=10, max_uptake=1.0, seed=0)
    assert ((design > 0).sum(axis=1) == 3).all()  # can't exceed dim


def test_perturb_media_drops_and_scales_a_base():
    base = np.full(50, 100.0)
    design = sampling.perturb_media(n=40, base_vector=base, seed=7)
    assert design.shape == (40, 50)
    # Every value is either 0 (dropped) or a scaled-down base value (<= base).
    assert design.min() >= 0.0
    assert design.max() <= 100.0
    # Rows vary in how many components survive (keep_p is random per row).
    kept = (design > 0).sum(axis=1)
    assert kept.min() < kept.max()
    # A never-present base component (0) stays 0.
    base2 = np.array([0.0, 100.0])
    d2 = sampling.perturb_media(n=10, base_vector=base2, seed=0)
    assert np.all(d2[:, 0] == 0.0)


def test_titrate_media_limits_a_few_nutrients_against_a_replete_background():
    essential = np.zeros(6, dtype=bool)
    essential[0] = True
    demand = np.array([2.0, 0.01, 100.0, 5.0, 0.5, 0.0])  # orders of magnitude apart
    design = sampling.titrate_media(
        300,
        6,
        seed=1,
        scale=demand,
        keep_range=(0.8, 1.0),
        essential=essential,
        span=(0.05, 1.0),
        replete=(2.0, 5.0),
        n_limiting=2,
    )
    assert design.shape == (300, 6)
    assert (design[:, 0] > 0).all()  # essentials are never dropped
    assert (design[:, 5] == 0).all()  # zero demand -> never offered
    for j, d in enumerate(demand[:5]):
        offered = design[design[:, j] > 0, j]
        # Every nutrient is expressed relative to its own demand: sometimes scarce
        # (limiting), mostly replete.
        assert offered.min() >= 0.05 * d
        assert offered.max() <= 5.0 * d
        assert (offered < d).any() and (offered > d).any()
    # ...but only a few nutrients are scarce in any one medium, so growth stays
    # attributable instead of being a minimum over everything at once.
    scarce_per_row = (design < demand[None, :]) & (design > 0)
    assert scarce_per_row.sum(axis=1).max() <= 2


def test_titrate_media_rejects_bad_arguments_and_degenerate_shapes():
    assert sampling.titrate_media(0, 5, seed=0, scale=1.0).shape == (0, 5)
    assert sampling.titrate_media(4, 0, seed=0, scale=1.0).shape == (4, 0)
    design = sampling.titrate_media(20, 5, seed=0, scale=1.0)  # scalar scale broadcasts
    assert (design == 0).any()  # dropout still applies
    with pytest.raises(ValueError, match="span"):
        sampling.titrate_media(4, 3, seed=0, scale=1.0, span=(0.0, 1.0))
    with pytest.raises(ValueError, match="non-negative"):
        sampling.titrate_media(4, 3, seed=0, scale=np.array([-1.0, 1.0, 1.0]))


def test_sample_membership_sizes_and_distinctness():
    subsets = sampling.sample_membership(n_genomes=10, n_communities=25, size_range=(2, 5), seed=3)
    assert len(subsets) == 25
    for subset in subsets:
        assert 2 <= len(subset) <= 5
        assert len(set(subset.tolist())) == len(subset)  # distinct indices
        assert subset.max() < 10


def test_sample_membership_clamps_and_rejects_empty_range():
    # hi clamped to n_genomes; still valid.
    subsets = sampling.sample_membership(n_genomes=3, n_communities=5, size_range=(2, 99), seed=0)
    assert all(len(s) <= 3 for s in subsets)
    with pytest.raises(ValueError):
        sampling.sample_membership(n_genomes=1, n_communities=1, size_range=(2, 2), seed=0)


def test_seed_is_deterministic():
    a = sampling.latin_hypercube(8, 3, 10.0, seed=7)
    b = sampling.latin_hypercube(8, 3, 10.0, seed=7)
    assert np.array_equal(a, b)
