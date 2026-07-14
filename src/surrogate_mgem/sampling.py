"""Pure media- and membership-sampling helpers (no solver, no micom).

These generate the *inputs* the data-generation module feeds to MICOM: media
uptake vectors over an exchange universe, and community subsets drawn from a
genome roster. Kept solver-free so they are unit-tested directly and importable
in CI without the heavy stack.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "latin_hypercube",
    "dirichlet_sample",
    "sparse_media",
    "sample_membership",
]


def latin_hypercube(n: int, dim: int, max_uptake: float, seed: int) -> np.ndarray:
    """Return an ``(n, dim)`` Latin-hypercube design in ``[0, max_uptake]``.

    One stratified sample per dimension (each column is a random permutation of
    ``n`` equal strata, jittered within its stratum), so the design is
    space-filling. Components at 0 are simply absent from the medium.
    """
    if dim == 0 or n == 0:
        return np.zeros((n, dim))
    rng = np.random.default_rng(seed)
    # Column j: strata (0..n-1)+U(0,1), scaled to [0,1), then permuted.
    strata = (np.arange(n)[:, None] + rng.random((n, dim))) / n
    for j in range(dim):
        rng.shuffle(strata[:, j])
    return strata * max_uptake


def dirichlet_sample(n: int, dim: int, total: float, seed: int) -> np.ndarray:
    """Return an ``(n, dim)`` design of nutrient mixtures on the simplex.

    Each row is drawn from ``Dirichlet(alpha=ones(dim))`` (uniform over the
    simplex) and scaled to a fixed uptake budget ``total`` -- the
    literature-standard way to sample the environment space. With ``alpha=1``
    most components come out small, so the media are naturally sparse.
    """
    if dim == 0 or n == 0:
        return np.zeros((n, dim))
    rng = np.random.default_rng(seed)
    return rng.dirichlet(np.ones(dim), size=n) * total


def sparse_media(n: int, dim: int, n_active: int, max_uptake: float, seed: int) -> np.ndarray:
    """Return an ``(n, dim)`` design where each row has only ``n_active`` non-zero components.

    Each medium activates a random subset of ``n_active`` components at
    ``U(0, max_uptake)`` and leaves the rest at 0. Unlike LHS/Dirichlet (which
    touch every component), this produces genuinely *limiting* media with a small,
    identifiable set of nutrients -- the regime minimal-media search targets, and
    the one where per-nutrient growth signal is learnable from few samples.
    """
    if dim == 0 or n == 0:
        return np.zeros((n, dim))
    rng = np.random.default_rng(seed)
    k = min(n_active, dim)
    design = np.zeros((n, dim))
    for i in range(n):
        idx = rng.choice(dim, size=k, replace=False)
        design[i, idx] = rng.random(k) * max_uptake
    return design


def sample_membership(
    n_genomes: int,
    n_communities: int,
    size_range: tuple[int, int],
    seed: int,
) -> list[np.ndarray]:
    """Return ``n_communities`` index subsets of ``range(n_genomes)``.

    Each subset's size is drawn uniformly from ``size_range`` (inclusive), then
    that many distinct genome indices are chosen without replacement. Varying
    the size is what lets the surrogate learn to add/remove members.
    """
    lo, hi = size_range
    lo = max(1, lo)
    hi = min(hi, n_genomes)
    if lo > hi:
        raise ValueError(f"Empty size range after clamping to n_genomes={n_genomes}: {size_range}")
    rng = np.random.default_rng(seed)
    sizes = rng.integers(lo, hi + 1, size=n_communities)
    return [np.sort(rng.choice(n_genomes, size=int(k), replace=False)) for k in sizes]
