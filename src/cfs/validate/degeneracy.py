"""M1 exchange-flux degeneracy survey — decides D4 (plan §5.2, gate V1).

`mu_max` (the LP optimum) is always unique, but the exchange fluxes `z` that
Head B learns need not be: primal degeneracy means alternate optima that can
differ in exchange profile. The plan leaves D4 (label-uniqueness scheme) open on
purpose — it is "decided by diagnostic, not by preference". This module runs that
diagnostic (exchange-FVA at fixed growth) and reports the plan's three-case
recommendation. The **human records the D4 choice**; :func:`recommend_d4` is
advisory.

COBRApy only.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger("cfs.degeneracy")

# Plan §5.2 thresholds: a range below this is "unique"; a metabolite degenerate
# in more than `_MANY_FRAC` of media at any alpha tips into "genuine" degeneracy.
_UNIQUE_TOL = 1e-6
_MANY_FRAC = 0.10


def _biomass_reaction(model):
    """The biomass/objective reaction (CarveMe: id starts with ``Growth``/``BIOMASS``)."""
    obj = [r for r in model.reactions if r.objective_coefficient != 0]
    if obj:
        return obj[0]
    for rid in ("Growth", "BIOMASS"):
        cand = [r for r in model.reactions if r.id.upper().startswith(rid.upper())]
        if cand:
            return cand[0]
    raise ValueError(f"{model.id}: cannot identify biomass reaction")


def exchange_degeneracy(model, alpha: float = 1.0) -> pd.Series:
    """Exchange-FVA range per exchange at growth fixed to ``alpha * mu_max`` (§5.2).

    Returns ``max - min`` for every exchange reaction; large ranges mark
    metabolites whose optimal flux is not pinned down (alternate optima).
    """
    from cobra.flux_analysis import flux_variability_analysis

    biomass = _biomass_reaction(model)
    sol = model.optimize()
    mu_max = sol.objective_value or 0.0
    with model:
        biomass.bounds = (alpha * mu_max, alpha * mu_max)
        fva = flux_variability_analysis(
            model, reaction_list=list(model.exchanges), fraction_of_optimum=1.0
        )
    return (fva["maximum"] - fva["minimum"]).rename("range")


def degeneracy_survey(
    model, media: list[dict], alphas: tuple[float, ...] = (1.0, 0.7)
) -> pd.DataFrame:
    """Exchange-FVA ranges across ``media`` × ``alphas`` for one model (§5.2).

    ``media`` is a list of ``{exchange_id: uptake_bound}`` dicts (a stratified
    sample; ~50 for a small roster, ~200 at scale). Each medium is applied as
    lower bounds within a context so the model is left unchanged. Returns a tidy
    frame ``(medium, alpha, exchange, range)``; infeasible media are skipped.
    """
    rows = []
    for m_i, medium in enumerate(media):
        for alpha in alphas:
            try:
                with model:
                    _apply_medium(model, medium)
                    ranges = exchange_degeneracy(model, alpha=alpha)
            except Exception as exc:  # infeasible / no-growth medium — skip it
                LOGGER.debug("skip medium %d alpha %.2f: %s", m_i, alpha, exc)
                continue
            for exch, rng in ranges.items():
                rows.append(
                    {"medium": m_i, "alpha": alpha, "exchange": exch, "range": float(rng)}
                )
    return pd.DataFrame(rows, columns=["medium", "alpha", "exchange", "range"])


def _apply_medium(model, medium: dict) -> None:
    """Set exchange uptake lower bounds from a ``{exchange_id: bound}`` medium."""
    for ex_id, bound in medium.items():
        if ex_id in model.reactions:
            model.reactions.get_by_id(ex_id).lower_bound = -abs(float(bound))


def sample_media(model, n: int, seed: int = 0, lo: float = -3.0, hi: float = 1.0) -> list[dict]:
    """Draw ``n`` randomised per-model media for the survey (plan §5.2 "~stratified").

    Each medium gives every exchange the model can take up a log-uniform uptake
    bound in ``10**[lo, hi]`` (multiplied by the model's own default magnitude).
    This is a diagnostic sampler only — the plan's proper active-subspace /
    stratified-Sobol design (§4) is a later milestone; here we just need enough
    variety to expose degeneracy.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    uptakes = [ex.id for ex in model.exchanges if ex.lower_bound < 0]
    defaults = {ex.id: abs(ex.lower_bound) or 1.0 for ex in model.exchanges}
    media = []
    for _ in range(n):
        scale = 10.0 ** rng.uniform(lo, hi, size=len(uptakes))
        media.append({ex: defaults[ex] * s for ex, s in zip(uptakes, scale, strict=True)})
    return media


def recommend_d4(survey: pd.DataFrame) -> dict:
    """Collapse a survey into the plan's §5.2 three-case D4 recommendation.

    - **clean** (ranges < 1e-6 almost everywhere) → plain FBA for Head B.
    - **localised** (a few metabolites degenerate) → weak regularisation.
    - **genuine** (many metabolites degenerate) → full elastic net (§5.4).

    Returns a dict with the verdict, the offending exchanges, and summary stats.
    The human makes the final D4 call and documents it (plan is explicit).
    """
    if survey.empty:
        return {
            "verdict": "unknown",
            "reason": "no feasible media surveyed",
            "degenerate_exchanges": [],
        }

    degenerate = survey["range"] > _UNIQUE_TOL
    frac_degenerate = float(degenerate.mean())
    # Per-exchange worst-case fraction of media in which it is degenerate.
    per_exch = (
        survey.assign(deg=degenerate)
        .groupby("exchange")["deg"]
        .mean()
        .sort_values(ascending=False)
    )
    many = per_exch[per_exch > _MANY_FRAC]

    if frac_degenerate < 1e-3:
        verdict, action = "clean", "plain FBA for Head B; no regularisation needed"
    elif len(many) <= 3:
        verdict = "localised"
        action = "weak regularisation; inspect the redundant transporter pairs"
    else:
        verdict, action = "genuine", "full elastic-net scheme (plan §5.4)"

    return {
        "verdict": verdict,
        "action": action,
        "frac_observations_degenerate": frac_degenerate,
        "n_media": int(survey["medium"].nunique()),
        "degenerate_exchanges": {k: float(v) for k, v in many.items()},
        "note": "Advisory only — the human decides D4 and documents it (plan §5).",
    }


def survey_roster(roster, media_per_model: dict, out_dir: Path, alphas=(1.0, 0.7)) -> dict:
    """Run the survey for every roster model, write tables + a D4 recommendation.

    ``media_per_model`` maps ``genome_id -> list[medium dict]``. Writes one
    ``{genome_id}.degeneracy.csv`` per model and a combined
    ``d4_recommendation.json``. Returns the recommendation dict.
    """
    from cobra.io import read_sbml_model

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for gm in roster:
        model = read_sbml_model(str(gm.model_path))
        media = media_per_model.get(gm.genome_id, [])
        survey = degeneracy_survey(model, media, alphas=alphas)
        survey.insert(0, "genome_id", gm.genome_id)
        survey.to_csv(out_dir / f"{gm.genome_id}.degeneracy.csv", index=False)
        frames.append(survey)
        LOGGER.info("degeneracy survey %s: %d rows", gm.genome_id, len(survey))

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    rec = recommend_d4(combined)
    (out_dir / "d4_recommendation.json").write_text(json.dumps(rec, indent=2))
    return rec
