"""Active-learning loop: spend real solves where the ensemble is most uncertain.

Each round trains an ensemble on the labelled data, proposes many candidate media
cheaply, scores them by ensemble disagreement (epistemic uncertainty), selects a
diverse high-uncertainty batch, solves *only that batch* with the real evaluator,
and folds the results back in. This concentrates expensive LP solves on the gaps
where the surrogate is weak, rather than sampling uniformly.

The real evaluator is injected as a callable, so the loop is unit-tested with a
synthetic function and wired to micom only at the CLI.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from surrogate_mgem.ensemble import GrowthEnsemble
from surrogate_mgem.sampling import dirichlet_sample, latin_hypercube, sparse_media

LOGGER = logging.getLogger("surrogate-mgem.active")

# An evaluator maps a full-length medium vector to per-member growth (in the
# target column order), or None if the medium is infeasible.
Evaluator = Callable[[np.ndarray], "np.ndarray | None"]


@dataclass
class ActiveConfig:
    """Parameters for the active-learning loop."""

    rounds: int = 5
    batch_size: int = 16
    n_candidates: int = 2000
    max_uptake: float = 1000.0
    sampler: str = "sparse"  # candidate proposal distribution
    n_active: int = 20  # sparse proposer: active components per candidate
    n_models: int = 5
    epochs: int = 300
    pool_factor: int = 4  # diversity pool = pool_factor * batch_size
    seed: int = 0


def propose_candidates(
    active_mask: np.ndarray,
    n: int,
    max_uptake: float,
    sampler: str,
    seed: int,
    n_active: int = 20,
) -> np.ndarray:
    """Sample ``n`` candidate media, non-zero only on the community's own exchanges.

    ``active_mask`` is a boolean over the full feature space; components outside
    it stay 0 (the community cannot exchange them), so proposals live in the
    community's real medium subspace while keeping the surrogate's full-width
    coordinate system. ``n_active`` applies to the ``sparse`` sampler.
    """
    full_dim = len(active_mask)
    active_dim = int(active_mask.sum())
    if sampler == "lhs":
        sub = latin_hypercube(n, active_dim, max_uptake, seed)
    elif sampler == "dirichlet":
        sub = dirichlet_sample(n, active_dim, max_uptake, seed)
    else:
        sub = sparse_media(n, active_dim, n_active, max_uptake, seed)
    candidates = np.zeros((n, full_dim), dtype=np.float32)
    candidates[:, active_mask] = sub
    return candidates


def diverse_topk(
    candidates: np.ndarray, scores: np.ndarray, k: int, pool_factor: int = 4
) -> list[int]:
    """Select ``k`` high-score, mutually-distant candidates (uncertainty + diversity).

    Takes the top ``pool_factor * k`` by score, then greedily picks farthest-point
    samples from that pool (seeded by the single highest score), so a batch covers
    several weak regions instead of clustering in one.
    """
    n = len(candidates)
    if k >= n:
        return list(range(n))
    pool = np.argsort(scores)[::-1][: min(n, max(k, pool_factor * k))]
    selected = [int(pool[0])]
    remaining = [int(i) for i in pool[1:]]
    while len(selected) < k and remaining:
        sel_pts = candidates[selected]
        # distance of each remaining point to its nearest already-selected point
        dists = [min(np.linalg.norm(candidates[i] - sel_pts, axis=1)) for i in remaining]
        pick = int(np.argmax(dists))
        selected.append(remaining.pop(pick))
    return selected


def _metrics(ensemble: GrowthEnsemble, X_test, Y_test) -> dict[str, float]:
    """R^2 / MAE of the ensemble mean on a held-out set (empty if none given)."""
    if X_test is None or len(X_test) == 0:
        return {}
    from sklearn.metrics import mean_absolute_error, r2_score

    pred = ensemble.predict(X_test)
    return {"r2": float(r2_score(Y_test, pred)), "mae": float(mean_absolute_error(Y_test, pred))}


def active_learning_loop(
    X0: np.ndarray,
    Y0: np.ndarray,
    evaluate: Evaluator,
    active_mask: np.ndarray,
    config: ActiveConfig,
    X_test: np.ndarray | None = None,
    Y_test: np.ndarray | None = None,
) -> tuple[GrowthEnsemble, pd.DataFrame, tuple[np.ndarray, np.ndarray]]:
    """Run the loop; return (final ensemble, per-round history, grown (X, Y)).

    History rows carry ``n_train`` and the mean uncertainty of the selected batch
    (so the report can show error and uncertainty falling as solves accrue), plus
    held-out ``r2``/``mae`` when a test set is supplied.
    """
    X = np.asarray(X0, dtype=np.float32).copy()
    Y = np.asarray(Y0, dtype=np.float32).copy()
    n_out = Y.shape[1]
    history: list[dict[str, float]] = []

    def train() -> GrowthEnsemble:
        ens = GrowthEnsemble(X.shape[1], n_out, n_models=config.n_models)
        ens.fit(X, Y, base_seed=config.seed, epochs=config.epochs)
        return ens

    ensemble = train()
    for r in range(config.rounds):
        row: dict[str, float] = {
            "round": r,
            "n_train": len(X),
            **_metrics(ensemble, X_test, Y_test),
        }
        candidates = propose_candidates(
            active_mask,
            config.n_candidates,
            config.max_uptake,
            config.sampler,
            config.seed + r + 1,
            config.n_active,
        )
        _, std = ensemble.predict_with_uncertainty(candidates)
        acquisition = std.mean(axis=1)
        picks = diverse_topk(candidates, acquisition, config.batch_size, config.pool_factor)
        row["mean_selected_uncertainty"] = float(acquisition[picks].mean())

        new_X, new_Y = [], []
        for i in picks:
            y = evaluate(candidates[i])
            if y is not None:
                new_X.append(candidates[i])
                new_Y.append(np.asarray(y, dtype=np.float32))
        row["n_selected"] = len(picks)
        row["n_new_feasible"] = len(new_X)
        history.append(row)
        LOGGER.info(
            "Round %d: n_train=%d, mean uncertainty=%.4g, +%d feasible solves.",
            r,
            int(row["n_train"]),
            row["mean_selected_uncertainty"],
            len(new_X),
        )
        if new_X:
            X = np.vstack([X, np.stack(new_X)])
            Y = np.vstack([Y, np.stack(new_Y)])
        ensemble = train()  # refit on the enlarged set for the next round / final model

    final = {"round": config.rounds, "n_train": len(X), **_metrics(ensemble, X_test, Y_test)}
    history.append(final)
    return ensemble, pd.DataFrame(history), (X, Y)
