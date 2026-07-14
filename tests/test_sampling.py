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
