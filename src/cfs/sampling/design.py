"""§4.3-4.4 stratified-Sobol sampling design over the active subspace.

Media are concentrations relative to Km (the MM half-saturation). ``log10(c/Km)``
is sampled over ``[-4, 1]`` with **40% of samples below Km** (plan §4.3, biased
for D9's low-concentration regime, where MM is steepest and feasibility flips).
Background metabolites are held rich, perturbed in 10% of media so the surrogate
learns they are inert rather than never seeing them vary. All-but-one-depleted
corners pin the single-limitation facets of the value function. Sobol within each
stratum, not uniform random.

Half the bulk budget goes to **per-metabolite focus strata** (``frac_focus``):
each metabolite in ``A_i`` gets an equal share of media in which it is the scarce
one. Plain Sobol over ``A_i`` leaves coverage of "which metabolite actually
limits" skewed ~1000:1, and that skew — not the surrogate's architecture — is
what holds the M3 gradient gate down. See ``sample_media`` for the measurements.

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
    # Share of the bulk budget given to the per-metabolite focus strata below.
    # This is the single biggest lever on M3 — see `sample_media`.
    frac_focus: float = 0.50
    # Growth-rate grid, K=8, densified near 1 where dFBA lives and z moves fastest.
    alphas: tuple[float, ...] = (0.0, 0.25, 0.5, 0.7, 0.85, 0.93, 0.97, 1.0)
    # eps == tau (D7/§5.5): the smoothing/regularisation family, middle is primary.
    eps_levels: tuple[float, ...] = (1e-2, 1e-3, 1e-4)
    eps_primary_idx: int = 1
    subset_frac: float = 0.20  # non-primary eps run on a stratified subset (§4.5)
    # §4.7: measure each metabolite's limiting regime by LP before sampling, so a
    # genome with no previous labels still anchors its own bands.
    probe: bool = True
    probe_steps: int = 12  # bisection steps over the 5-decade band
    seed: int = 0


def limiting_scales(u_star: dict[str, float], clip: float = 1e4) -> dict[str, float]:
    """``u*`` (saturation where a metabolite limits) -> ``scales`` for `sample_media`.

    ``u = c/(Km+c)``, so a metabolite observed to limit at saturation ``u*`` does
    so at ``c/Km = u*/(1-u*)`` — and that is where its band should be centred.

    This is the resampling mechanism: run once with the default design, read
    ``u*`` off the labels (it is exactly ``cfs.surrogate.data._kink_scale``, the
    median saturation over the rows where that metabolite's dual is non-zero),
    then regenerate with the band aimed at each metabolite's own regime. No model
    in the loop and no extra solves — the first run's labels are the acquisition
    signal. Metabolites that never limited have no ``u*`` and keep scale 1.0,
    which is the only honest default: nothing was learned about them.
    """
    out = {}
    for ex, u in u_star.items():
        if not (0.0 < u < 1.0):
            continue
        out[ex] = float(min(max(u / (1.0 - u), 1.0 / clip), clip))
    return out


def band_scales(
    probe: dict[str, float] | None,
    previous: dict[str, float] | None,
    roster_median: dict[str, float] | None,
    exchanges: list[str],
) -> tuple[dict[str, float], dict[str, str]]:
    """§4.7 fallback chain: probe -> previous ``u*`` -> roster median -> 1.0.

    Returns the scales for :func:`sample_media` and, per metabolite, which link
    of the chain supplied it. The order is not arbitrary: the probe is measured
    on *this* organism with no labels needed; a previous run's ``u*`` is measured
    but stale; the roster median is a prior that is fine for the ions (spread 2-5x
    across organisms) and useless for ``EX_arg__L_e`` (2585x); 1.0 is the
    un-anchored default whose coverage skew is the thing being fixed.
    """
    sources = ("probe", "previous", "roster_median")
    tables = (probe or {}, previous or {}, roster_median or {})
    scales, chosen = {}, {}
    for ex in exchanges:
        for src, table in zip(sources, tables, strict=True):
            if ex in table:
                scales[ex], chosen[ex] = float(table[ex]), src
                break
        else:
            scales[ex], chosen[ex] = 1.0, "default"
    return scales, chosen


def topup_weights(
    per_limiting_metabolite: dict[str, dict], floor: float = 0.25
) -> dict[str, float]:
    """Held-out per-metabolite diagnostics -> a focus-budget split for the next run.

    Takes ``diagnostics["per_organism"][gid]["per_limiting_metabolite"]`` from
    :func:`cfs.surrogate.train.evaluate` and returns weights summing to 1, to be
    passed as ``focus_weights`` to :func:`sample_media`. Weight is ``1 - cosine``:
    the budget goes where the surrogate is *measured* to be wrong, not where a
    model guesses it might be.

    ``floor`` keeps a share for metabolites that are already right — a cell at
    cosine 0.95 still needs media to stay there once the design stops handing it
    the budget by accident, and a pure ``1 - cosine`` split would starve it.

    Metabolites absent from the diagnostics never limited in the previous run, so
    they have neither a measured error nor a ``u*``. They are invisible here *and*
    to :func:`limiting_scales`, and they are what an ensemble's gradient
    disagreement is for (:func:`cfs.surrogate.ensemble.gradient_disagreement`).
    """
    if not per_limiting_metabolite:
        return {}
    err = {ex: max(0.0, 1.0 - float(d["grad_cosine"])) for ex, d in per_limiting_metabolite.items()}
    n = len(err)
    total = sum(err.values())
    if total <= 0:  # everything already perfect: split evenly
        return dict.fromkeys(err, 1.0 / n)
    return {ex: floor / n + (1.0 - floor) * e / total for ex, e in err.items()}


def sample_media(
    subspace,
    km_cfg: dict,
    cfg: SamplingConfig | None = None,
    scales: dict[str, float] | None = None,
    focus_weights: dict[str, float] | None = None,
) -> list[dict]:
    """Return ``[{exchange_id: concentration}]`` media for one organism (§4.3).

    Varies the active subspace ``A_i`` (or the background if ``A_i`` is empty),
    holding the rest rich. Length is ``cfg.n_media`` (corners + Sobol bulk) unless
    the organism has no uptake exchanges, in which case it is empty.

    ``focus_weights`` splits the focus budget unevenly across ``A_i`` — see
    :func:`topup_weights`, which derives it from a previous run's held-out error.
    Omitted, every metabolite in ``A_i`` gets an equal share.

    ``scales`` is a per-exchange multiplier on ``Km`` setting where that
    metabolite's band is centred; missing entries default to 1.0, which is the
    plain Km-relative design. **Supplying it is what makes the design usable**
    (see :func:`limiting_scales`): sampling every metabolite over the same
    ``log10(c/Km) in [-4, 1]`` assumes they all start limiting at comparable
    ``c/Km``, and on the 21-genome roster they do not — measured medians span
    5107x. ``EX_mg2_e``/``EX_ca2_e``/``EX_cl_e`` begin limiting at
    ``log10(c/Km) ~ -3.8``, i.e. 3-4% into the band, so almost no draw lands in
    the regime where they carry a dual; ``EX_o2_e``/``EX_malt_e``/``EX_h_e`` sit
    at 82-88%. That is the same defect the legacy pipeline hit and fixed with
    ``surrogate_mgem.data.estimate_demand`` ("a shared sampling band leaves the
    small ones saturated ... no rescaling or extra data can undo"), which was
    never carried into this package.
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

    scales = scales or {}
    km = {ex: km_for_exchange(ex, km_cfg) for ex in sampled + held}
    # Where each metabolite starts limiting, in log10(c/Km). Used *only* to move
    # the bottom of its own focus stratum — never the rich level.
    #
    # Scaling `km` itself instead (i.e. shifting the whole band) is wrong and was
    # measured to be wrong: a metabolite whose band centre moves down to its
    # limiting point then has its "replete" level only 10x above limiting, so
    # nothing in the medium is actually replete and everything starves together.
    # On AAXE02 that collapsed the number of distinct limiting metabolites from
    # 27 to 7 — `EX_k_e` and `EX_acnam_e` took 85% of media between them and
    # `EX_o2_e` never limited at all.
    anchor = {ex: np.log10(float(scales.get(ex, 1.0))) for ex in sampled}
    held_rich = {ex: km[ex] * 10.0**cfg.log10_hi for ex in held}
    sampled_rich = {ex: km[ex] * 10.0**cfg.log10_hi for ex in sampled}

    media: list[dict] = []
    # All-but-one-depleted corners: single-limitation facets (§4.3).
    for ex in sampled:
        media.append({**held_rich, **sampled_rich, ex: 0.0})

    n_bulk = max(0, cfg.n_media - len(media))
    d = len(sampled)
    kmv = np.array([km[ex] for ex in sampled])

    def _sobol(n):
        # ponytail: n is rarely a power of 2 (Sobol's balance warning); the
        # stratification already fixes coverage where it matters, so we silence it.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            return qmc.Sobol(d=d, seed=int(rng.integers(1 << 31))).random(n)

    def _emit(u, lo, hi):
        """Unit cube -> media. ``lo``/``hi`` are scalars or per-column log10 bounds."""
        for row in kmv * 10.0 ** (lo + u * (hi - lo)):
            m = {**held_rich, **{ex: float(c) for ex, c in zip(sampled, row, strict=True)}}
            if held and rng.random() < cfg.frac_bg_perturb:
                for ex in held:
                    m[ex] = km[ex] * 10.0 ** rng.uniform(cfg.log10_lo, cfg.log10_hi)
            media.append(m)

    # Per-metabolite focus strata: an equal share of the budget for each sampled
    # metabolite, in media where *that* metabolite is the scarce one (drawn below
    # Km) and the rest sit at or above Km.
    #
    # Without this the budget is spent re-learning whichever one or two
    # metabolites happen to dominate the min. Measured on the 4000-media run, per
    # organism: the top metabolite limits 473-2060 media, the *median* metabolite
    # limits 6-12, the rarest limits 1 — against 41-133 for an even split. Held-out
    # gradient cosine tracks that count directly (Spearman 0.72 over 156
    # (organism, metabolite) cells; <50 rows -> 0.49, >400 rows -> 0.95), so the
    # skew, not the architecture, is what holds the M3 gate at 0.66.
    #
    # The all-but-one-depleted corners above do not cover this: they set the
    # metabolite to *exactly* zero, which for an essential one means no growth,
    # and a zero-growth row is dropped from the Sobolev term (`gvalid`). Scarce
    # but non-zero is what yields a usable dual.
    n_focus = int(round(cfg.frac_focus * n_bulk))
    # Equal shares by default; `focus_weights` (from `topup_weights`) skews them
    # toward the metabolites a previous run got measurably wrong.
    w = np.array([(focus_weights or {}).get(ex, 1.0 / d) for ex in sampled], dtype=float)
    w = w / w.sum() if w.sum() > 0 else np.full(d, 1.0 / d)
    quota = np.floor(w * n_focus).astype(int)
    n_focused = 0
    for j in range(d):
        if quota[j] <= 0:
            continue
        # Everything else stays genuinely replete, at or above its literature Km;
        # only the focused metabolite is driven down, and only as far as its *own*
        # limiting regime (`anchor`), which for the ions is ~3.8 decades below Km
        # and for EX_o2_e is ~0.1 decades above it. A single shared [-4, 0) band
        # is what left `EX_mg2_e` limiting in 2% of media and `EX_k_e` in 14%.
        lo = np.full(d, 0.0)
        hi = np.full(d, cfg.log10_hi)
        a = anchor[sampled[j]]
        lo[j], hi[j] = max(cfg.log10_lo, a - 1.5), min(cfg.log10_hi, a + 0.5)
        _emit(_sobol(int(quota[j])), lo, hi)
        n_focused += int(quota[j])

    # The remainder keeps the original two strata in log10(c/Km): below Km
    # [lo, 0), at/above Km [0, hi]. Both regimes still need unfocused coverage —
    # real media limit on several things at once and the head has to see that.
    n_rest = n_bulk - n_focused
    n_below = int(round(cfg.frac_below_km * n_rest))
    for n_str, lo, hi in ((n_below, cfg.log10_lo, 0.0), (n_rest - n_below, 0.0, cfg.log10_hi)):
        if n_str <= 0:
            continue
        _emit(_sobol(n_str), lo, hi)
    return media
