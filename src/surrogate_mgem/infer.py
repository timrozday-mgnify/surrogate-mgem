"""Inference helpers, including the verify-in-loop safety net.

The surrogate is only ever trusted to *rank* candidate media; the final answer
is always re-checked with the real evaluator. This downgrades the requirement
from "the surrogate must be numerically perfect" to "the surrogate must be a
good ranker", and guarantees the reported optimum is a genuine solver result.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class VerifiedCandidate:
    """One candidate re-scored by the real evaluator."""

    item: Any
    surrogate_score: float
    true_score: float


def verify_shortlist(
    candidates: Sequence[Any],
    surrogate_scores: Sequence[float],
    evaluate_fn: Callable[[Any], float],
    k: int,
) -> list[VerifiedCandidate]:
    """Re-evaluate the top-``k`` surrogate-ranked candidates with the real solver.

    Takes the ``k`` candidates with the highest surrogate score, calls the
    (expensive) ``evaluate_fn`` on each, and returns them sorted by the true
    score, best first. Only ``k`` real solves are spent regardless of how many
    candidates the surrogate screened. ``k`` is clamped to the number available.
    """
    if len(candidates) != len(surrogate_scores):
        raise ValueError("candidates and surrogate_scores must have the same length.")
    if k <= 0 or len(candidates) == 0:
        return []
    scores = np.asarray(surrogate_scores, dtype=float)
    top = np.argsort(scores)[::-1][: min(k, len(candidates))]
    verified = [
        VerifiedCandidate(candidates[i], float(scores[i]), float(evaluate_fn(candidates[i])))
        for i in top
    ]
    verified.sort(key=lambda c: c.true_score, reverse=True)
    return verified
