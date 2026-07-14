"""Tests for the growth surrogate: it learns a simple map and round-trips to disk."""

import numpy as np

from surrogate_mgem.model import GrowthSurrogate


def _linear_dataset(seed=0, n=400, n_in=5, n_out=2):
    rng = np.random.default_rng(seed)
    X = rng.random((n, n_in)).astype(np.float32) * 10.0
    W = rng.standard_normal((n_in, n_out)).astype(np.float32)
    Y = (X @ W + 0.01 * rng.standard_normal((n, n_out))).astype(np.float32)
    return X, Y


def test_surrogate_learns_linear_map():
    X, Y = _linear_dataset()
    X_tr, Y_tr, X_te, Y_te = X[:320], Y[:320], X[320:], Y[320:]
    model = GrowthSurrogate(n_in=5, n_out=2, hidden=(32, 32))
    model.fit(X_tr, Y_tr, epochs=200, seed=0)
    pred = model.predict(X_te)
    # R^2 on held-out data should be high for an easy linear target.
    ss_res = ((Y_te - pred) ** 2).sum()
    ss_tot = ((Y_te - Y_te.mean(0)) ** 2).sum()
    assert 1 - ss_res / ss_tot > 0.95


def test_save_load_roundtrip(tmp_path):
    X, Y = _linear_dataset(n=120)
    model = GrowthSurrogate(n_in=5, n_out=2, hidden=(16,))
    model.fit(X, Y, epochs=30, seed=1)
    before = model.predict(X[:5])
    path = tmp_path / "m.pt"
    model.save(path)
    reloaded = GrowthSurrogate.load(path, hidden=(16,))
    after = reloaded.predict(X[:5])
    assert np.allclose(before, after, atol=1e-5)  # buffers + weights preserved


def test_fit_handles_tiny_dataset():
    # Fewer rows than a val split; fit must not crash (train==val fallback).
    X, Y = _linear_dataset(n=3)
    GrowthSurrogate(n_in=5, n_out=2, hidden=(8,)).fit(X, Y, epochs=5, val_split=0.2)
