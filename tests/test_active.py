"""Tests for the active-learning loop with a synthetic (solver-free) evaluator."""

import numpy as np

from surrogate_mgem.active import (
    ActiveConfig,
    active_learning_loop,
    diverse_topk,
    propose_candidates,
)


def test_propose_candidates_respects_active_mask():
    mask = np.array([True, True, False, True, False])
    cand = propose_candidates(mask, n=50, max_uptake=100.0, sampler="dirichlet", seed=0)
    assert cand.shape == (50, 5)
    assert np.all(cand[:, ~mask] == 0.0)  # inactive coords stay zero
    assert cand[:, mask].sum() > 0  # active coords are populated


def test_diverse_topk_prefers_high_scores_and_is_distinct():
    rng = np.random.default_rng(0)
    cand = rng.random((40, 3))
    scores = np.arange(40, dtype=float)  # last indices best
    picks = diverse_topk(cand, scores, k=5, pool_factor=4)
    assert len(picks) == len(set(picks)) == 5
    # The single best-scoring point always seeds the batch.
    assert 39 in picks


def test_active_loop_grows_training_set_and_tracks_history():
    rng = np.random.default_rng(1)
    mask = np.array([True, True, True, False])
    W = rng.standard_normal((3, 2))

    def evaluate(vector):
        # Deterministic, always feasible: growth is linear in the active coords.
        return vector[mask] @ W

    X0 = propose_candidates(mask, 30, 100.0, "dirichlet", seed=5)
    Y0 = np.stack([evaluate(x) for x in X0])
    X_te = propose_candidates(mask, 20, 100.0, "dirichlet", seed=6)
    Y_te = np.stack([evaluate(x) for x in X_te])

    config = ActiveConfig(
        rounds=2, batch_size=5, n_candidates=100, max_uptake=100.0, n_models=2, epochs=40, seed=0
    )
    ensemble, history, (X_all, Y_all) = active_learning_loop(
        X0, Y0, evaluate, mask, config, X_test=X_te, Y_test=Y_te
    )
    assert len(history) == config.rounds + 1  # one row per round + final
    assert history["n_train"].iloc[-1] > history["n_train"].iloc[0]  # set grew
    assert len(X_all) == len(Y_all) == int(history["n_train"].iloc[-1])
    assert "r2" in history.columns  # test metrics tracked across rounds
    assert ensemble.predict(X_te).shape == (20, 2)
