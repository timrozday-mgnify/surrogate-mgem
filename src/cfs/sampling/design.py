"""§4.3-4.4 stratified-Sobol sampling design over the active subspace.

Media are concentrations relative to Km (the MM half-saturation). ``log10(c/Km)``
is sampled over ``[-4, 1]`` with **40% of samples below Km** (plan §4.3, biased
for D9's low-concentration regime, where MM is steepest and feasibility flips).
Background metabolites are held rich, perturbed in 10% of media so the surrogate
learns they are inert rather than never seeing them vary. All-but-one-depleted
corners pin the single-limitation facets of the value function. Sobol within each
stratum, not uniform random.

Pure numpy + ``scipy.stats.qmc`` — no solver. The growth-rate grid (§4.4) and the
eps family (§4.5/§5.5) live on :class:`SamplingConfig` and are consumed by
``generate``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SamplingConfig:
    """Design knobs for one organism (plan §4.3-4.5). Defaults are the D10 scale."""

    n_media: int = 20000
    log10_lo: float = -4.0  # log10(c/Km) lower end
    log10_hi: float = 1.0  # log10(c/Km) upper end (also the "rich" level)
    frac_below_km: float = 0.40  # fraction of bulk media with c < Km (§4.3)
    frac_bg_perturb: float = 0.10  # fraction of media that perturb the background
    # Growth-rate grid, K=8, densified near 1 where dFBA lives and z moves fastest.
    alphas: tuple[float, ...] = (0.0, 0.25, 0.5, 0.7, 0.85, 0.93, 0.97, 1.0)
    # eps == tau (D7/§5.5): the smoothing/regularisation family, middle is primary.
    eps_levels: tuple[float, ...] = (1e-2, 1e-3, 1e-4)
    eps_primary_idx: int = 1
    subset_frac: float = 0.20  # non-primary eps run on a stratified subset (§4.5)
    seed: int = 0


def sample_media(subspace, km_cfg: dict, cfg: SamplingConfig | None = None) -> list[dict]:
    """Return ``[{exchange_id: concentration}]`` media for one organism (§4.3).

    Varies the active subspace ``A_i`` (or the background if ``A_i`` is empty),
    holding the rest rich. Length is ``cfg.n_media`` (corners + Sobol bulk) unless
    the organism has no uptake exchanges, in which case it is empty.
    """
    from scipy.stats import qmc

    from cfs.groundtruth.solve import km_for_exchange

    cfg = cfg if cfg is not None else SamplingConfig()
    rng = np.random.default_rng(cfg.seed)

    # Sample A_i; if the sweep found nothing sensitive, vary the background instead
    # so labels are not a single point.
    sampled = list(subspace.active) if subspace.active else list(subspace.background)
    held = list(subspace.background) if subspace.active else []
    if not sampled:
        return []

    km = {ex: km_for_exchange(ex, km_cfg) for ex in sampled + held}
    held_rich = {ex: km[ex] * 10.0 ** cfg.log10_hi for ex in held}
    sampled_rich = {ex: km[ex] * 10.0 ** cfg.log10_hi for ex in sampled}

    media: list[dict] = []
    # All-but-one-depleted corners: single-limitation facets (§4.3).
    for ex in sampled:
        media.append({**held_rich, **sampled_rich, ex: 0.0})

    n_bulk = max(0, cfg.n_media - len(media))
    d = len(sampled)
    n_below = int(round(cfg.frac_below_km * n_bulk))
    kmv = np.array([km[ex] for ex in sampled])
    # Two strata in log10(c/Km): below Km [lo, 0), at/above Km [0, hi].
    for n_str, lo, hi in ((n_below, cfg.log10_lo, 0.0), (n_bulk - n_below, 0.0, cfg.log10_hi)):
        if n_str <= 0:
            continue
        # ponytail: n_str is rarely a power of 2 (Sobol's balance warning); the
        # stratification already fixes coverage where it matters, so we silence it.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            u = qmc.Sobol(d=d, seed=int(rng.integers(1 << 31))).random(n_str)
        conc = kmv * 10.0 ** (lo + u * (hi - lo))
        for row in conc:
            m = {**held_rich, **{ex: float(c) for ex, c in zip(sampled, row, strict=True)}}
            if held and rng.random() < cfg.frac_bg_perturb:
                for ex in held:
                    m[ex] = km[ex] * 10.0 ** rng.uniform(cfg.log10_lo, cfg.log10_hi)
            media.append(m)
    return media
