"""The composition map itself, with a hand-written right-hand side.

`integrate` is the one piece of §8.1 that is neither a network nor an LP, and it
is shared by the surrogate and the ground-truth path — an error there would move
both trajectories the same way and hide itself in the comparison. So it is checked
against a case with a closed form: constant growth rate, constant consumption.

No JAX, no solver, no labels.
"""

from __future__ import annotations

import numpy as np

from cfs.compose.dfba import _cos, _rel, integrate


def test_exponential_growth_is_exact():
    # mu constant => X(t) = X0 exp(mu t) exactly, whatever the step size, because
    # the biomass update is the exponential map rather than an Euler step.
    mu = np.array([2.0, 0.5])
    traj = integrate(lambda c, X: (np.zeros(3), mu), np.ones(3), np.array([1.0, 4.0]), 0.1, 10)
    assert np.allclose(traj.x[-1], np.array([1.0, 4.0]) * np.exp(mu * 1.0))
    assert np.allclose(traj.c, 1.0)  # nothing consumed


def test_pool_is_clipped_at_zero():
    # A constant drain of 1/h from a pool of 0.5 must stop at 0, not go negative:
    # the LP's regime changes there (no uptake), so a negative concentration would
    # be fed straight into the MM bound.
    traj = integrate(
        lambda c, X: (np.array([-1.0]), np.zeros(1)), np.array([0.5]), np.ones(1), 0.1, 20
    )
    assert traj.c.min() == 0.0
    assert traj.c[-1, 0] == 0.0


def test_consumption_scales_with_biomass():
    # dc/dt = sum_i X_i z_i is the whole composition. Two organisms with the same
    # z drain twice as fast as one, and doubling biomass doubles the rate.
    z = np.array([-1.0])

    def rhs(c, X):
        return (X[:, None] * np.tile(z, (len(X), 1))).sum(0), np.zeros(len(X))

    one = integrate(rhs, np.array([10.0]), np.array([1.0]), 0.01, 100)
    two = integrate(rhs, np.array([10.0]), np.array([1.0, 1.0]), 0.01, 100)
    assert np.isclose(10.0 - one.c[-1, 0], 1.0)
    assert np.isclose(10.0 - two.c[-1, 0], 2.0)


def test_metrics():
    a = np.array([1.0, 0.0])
    assert _cos(a, a) == 1.0
    assert _rel(a, a) == 0.0
    assert np.isclose(_rel(np.array([2.0, 0.0]), a), 1.0)


# --------------------------------------------------------------------------- #
# Head B's loader: the alpha axis has to line up with the medium axis
# --------------------------------------------------------------------------- #


def test_behaviour_dataset_alpha_and_medium_alignment(tmp_path):
    """``z`` must land in the right (alpha, medium) cell, and share Head A's split.

    The shards store one row per (medium, alpha) in whatever order the generator
    wrote them, and the value arrays are sorted by ``medium_id``. Getting these two
    orderings out of step is silent: the shapes are right, the loss falls, and the
    head learns the wrong alpha response.
    """
    import json

    import pandas as pd
    import pytest

    pytest.importorskip("pyarrow")
    pytest.importorskip("yaml")

    from cfs.groundtruth.index import derive_index, write_index
    from cfs.surrogate.data import load_behaviour_dataset, load_value_dataset

    ex = ["EX_glc__D_e", "EX_o2_e"]
    alphas, mids = [0.0, 1.0], np.arange(10)
    index_path = tmp_path / "metabolite_index.json"
    digest = write_index(derive_index(["g0"], [set(ex)]), index_path)

    root = tmp_path / "labels"
    shard = root / "g0" / "eps_0.001"
    shard.mkdir(parents=True)
    (root / "g0.exchanges.json").write_text(json.dumps({"exchanges": ex}))
    # z is a function of both axes and of nothing else, so a misalignment shows up
    # as an exact value mismatch rather than as a worse fit.
    pd.DataFrame(
        [
            {
                "genome_id": "g0",
                "index_hash": digest,
                "medium_id": int(m),
                "alpha": a,
                "eps": 1e-3,
                "mu_max": 1.0,
                "status": "optimal",
                "medium": [0.02, 0.01],
                "z": [float(m), -100.0 * a],
                "shadow": [-0.5, 0.0],
            }
            # Deliberately not in (alpha, medium) order.
            for m in mids[::-1]
            for a in alphas
        ]
    ).to_parquet(shard / "part.parquet", index=False)

    ds = load_behaviour_dataset(root, index_path, eps=1e-3, seed=0)
    va = load_value_dataset(root, index_path, eps=1e-3, seed=0)
    assert ds.alphas.tolist() == alphas
    # z[:, 0] is the medium id and z[:, 1] is -100 * alpha, everywhere.
    assert np.allclose(ds.z_train[0][:, 1], -100.0 * ds.a_train[0])
    assert np.allclose(ds.z_val[0][:, 1], -100.0 * ds.a_val[0])
    # Same held-out media as Head A, each appearing once per alpha.
    assert ds.x_val.shape[1] == va.x_val.shape[1] * len(alphas)
    val_mids, train_mids = set(ds.z_val[0][:, 0]), set(ds.z_train[0][:, 0])
    assert not (val_mids & train_mids)
    assert val_mids | train_mids == {float(m) for m in mids}
    # Each held-out medium appears once per alpha and no more.
    assert len(ds.z_val[0]) == len(val_mids) * len(alphas)
    assert np.allclose(ds.x_scale, va.x_scale)
