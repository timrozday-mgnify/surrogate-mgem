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
    "perturb_media",
    "titrate_media",
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


def perturb_media(
    n: int,
    base_vector: np.ndarray,
    seed: int,
    keep_range: tuple[float, float] = (0.05, 1.0),
    scale_range: tuple[float, float] = (0.25, 1.0),
) -> np.ndarray:
    """Return ``n`` media by randomly dropping/scaling components of ``base_vector``.

    Each row keeps every base component independently with probability
    ``keep_p ~ U(keep_range)`` and scales the survivors by ``U(scale_range)``.
    Starting from a growth-supporting base (e.g. the full environment), this
    spans the feasible->limiting gradient with guaranteed coverage of the region
    where growth actually varies -- unlike random subsets, which usually miss the
    essential nutrient set and give uniformly zero growth. This is also the
    nutrient-removal regime of interest for minimal-media design.
    """
    base = np.asarray(base_vector, dtype=float)
    dim = len(base)
    if dim == 0 or n == 0:
        return np.zeros((n, dim))
    rng = np.random.default_rng(seed)
    keep_p = rng.uniform(keep_range[0], keep_range[1], size=(n, 1))
    mask = rng.random((n, dim)) < keep_p
    scale = rng.uniform(scale_range[0], scale_range[1], size=(n, dim))
    return base[None, :] * mask * scale


def titrate_media(
    n: int,
    dim: int,
    seed: int,
    *,
    scale: np.ndarray | float,
    keep_range: tuple[float, float] = (0.5, 1.0),
    essential: np.ndarray | None = None,
    span: tuple[float, float] = (0.05, 1.0),
    replete: tuple[float, float] = (2.0, 5.0),
    n_limiting: int = 3,
) -> np.ndarray:
    """Return ``(n, dim)`` media that titrate a *few* nutrients against a replete background.

    Per medium: drop non-essential components with probability ``1 - keep_p``
    (``keep_p ~ U(keep_range)``), then pick up to ``n_limiting`` of the survivors
    and give them a log-uniform bound in ``span * scale[i]`` -- scarce, so they
    set the growth rate -- while every other survivor gets ``U(replete) *
    scale[i]``, comfortably above its demand. Components with ``scale[i] == 0``
    are never offered.

    Two failure modes this steers between, both measured on the example genome
    (3000 media, held-out R^2):

    * One shared band for every nutrient (``scale`` a constant) leaves the
      low-demand ones permanently saturated: growth never responds to them, and
      the surrogate gets ~250 pure-noise coordinates. Nutrient demands there
      spanned 1e-3 to 42.
    * Titrating *all* of them at once around their own demand makes growth a
      minimum over ~110 simultaneously-scarce nutrients: physically fine, but the
      target collapses (std 1.78 -> 0.18) and even a random forest drops from
      0.89 to 0.75.

    Limiting a handful at a time keeps every nutrient's own scale while leaving
    growth attributable to the few that are actually scarce -- the same reason wet
    experiments vary one factor against a replete background.
    """
    if dim == 0 or n == 0:
        return np.zeros((n, dim))
    scale = np.broadcast_to(np.asarray(scale, dtype=float), (dim,))
    if np.any(scale < 0):
        raise ValueError("scale must be non-negative")
    for name, rng_pair in (("span", span), ("replete", replete)):
        if rng_pair[0] <= 0 or rng_pair[1] < rng_pair[0]:
            raise ValueError(f"{name} must be 0 < lo <= hi, got {rng_pair}")
    rng = np.random.default_rng(seed)

    keep_p = rng.uniform(keep_range[0], keep_range[1], size=(n, 1))
    offered = rng.random((n, dim)) < keep_p
    if essential is not None:
        offered |= np.asarray(essential, dtype=bool)[None, :]
    offered &= scale[None, :] > 0  # never offer what the community cannot consume

    # Replete background, then knock a few nutrients down into the limiting range.
    factor = rng.uniform(replete[0], replete[1], size=(n, dim))
    limiting = np.exp(rng.uniform(np.log(span[0]), np.log(span[1]), size=(n, dim)))
    scarce = np.zeros((n, dim), dtype=bool)
    for i in range(n):
        candidates = np.flatnonzero(offered[i])
        if not len(candidates):
            continue
        k = min(rng.integers(1, n_limiting + 1), len(candidates))
        scarce[i, rng.choice(candidates, size=int(k), replace=False)] = True
    factor = np.where(scarce, limiting, factor)
    return factor * scale[None, :] * offered


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
