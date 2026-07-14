"""Tests for the growth surrogate: it learns a simple map and round-trips to disk."""

import numpy as np

from surrogate_mgem.model import GrowthSurrogate, inverse_density_weights


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


def test_inverse_density_downweights_dense_region():
    # A tight dense cluster + a few far-flung sparse points. Dense points should
    # get lower weight than sparse ones.
    rng = np.random.default_rng(0)
    dense = rng.normal(0, 0.05, size=(100, 3))
    sparse = rng.normal(0, 5.0, size=(8, 3)) + 20.0  # far away, spread out
    X = np.vstack([dense, sparse]).astype(np.float32)
    w = inverse_density_weights(X, k=5)
    assert w[:100].mean() < w[100:].mean()
    assert np.all(w > 0)


def test_fit_early_stops_and_restores_best():
    # With a high cap, an easy target converges fast: fit must early-stop well
    # short of the cap and end on its best (not last) val loss.
    X, Y = _linear_dataset(n=400)
    model = GrowthSurrogate(n_in=5, n_out=2, hidden=(32, 32))
    hist = model.fit(X[:320], Y[:320], epochs=5000, patience=20, seed=0)
    assert hist["stopped_early"] is True
    assert hist["epochs_run"] < 5000
    assert hist["best_val"] <= min(hist["val"]) + 1e-9


def test_uniform_weights_match_plain_mse_convergence():
    # sample_weight=None (plain MSE) must still learn the easy linear map.
    X, Y = _linear_dataset()
    model = GrowthSurrogate(n_in=5, n_out=2, hidden=(32, 32))
    model.fit(X[:320], Y[:320], epochs=1000, sample_weight=None, seed=0)
    pred = model.predict(X[320:])
    ss_res = ((Y[320:] - pred) ** 2).sum()
    ss_tot = ((Y[320:] - Y[320:].mean(0)) ** 2).sum()
    assert 1 - ss_res / ss_tot > 0.95
