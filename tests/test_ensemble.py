"""Tests for the ensemble: it predicts, quantifies uncertainty, and round-trips."""

import numpy as np

from surrogate_mgem.ensemble import GrowthEnsemble


def _linear(seed=0, n=200):
    rng = np.random.default_rng(seed)
    X = rng.random((n, 4)).astype(np.float32) * 5
    W = rng.standard_normal((4, 2)).astype(np.float32)
    return X, (X @ W).astype(np.float32)


def test_ensemble_predicts_and_reports_uncertainty():
    X, Y = _linear()
    ens = GrowthEnsemble(4, 2, n_models=3, hidden=(16, 16))
    ens.fit(X[:160], Y[:160], epochs=120)
    mean, std = ens.predict_with_uncertainty(X[160:])
    assert mean.shape == std.shape == (40, 2)
    assert (std >= 0).all()
    # Uncertainty off the training range should exceed uncertainty in-range.
    far = np.full((5, 4), 100.0, dtype=np.float32)
    _, std_far = ens.predict_with_uncertainty(far)
    assert std_far.mean() > std.mean()


def test_ensemble_save_load(tmp_path):
    X, Y = _linear(n=80)
    ens = GrowthEnsemble(4, 2, n_models=3, hidden=(8,))
    ens.fit(X, Y, epochs=20)
    before = ens.predict(X[:5])
    ens.save(tmp_path / "ens")
    after = GrowthEnsemble.load(tmp_path / "ens", hidden=(8,)).predict(X[:5])
    assert np.allclose(before, after, atol=1e-5)
