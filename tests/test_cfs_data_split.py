"""The held-out set must not move when a §4.6 top-up round is added.

This is the regression that produced a wrong conclusion once already: top-up media
are drawn where the model is worst, and when they leaked into the validation split
the *test* got harder every round. The probe-band runs went 491 -> 595 -> 781
usable held-out rows and the gate fell 0.695 -> 0.648, which was read as "more
labels where the model is worst make it worse" when it was mostly the ruler
changing. No solver, no JAX — parquet and the frozen index only.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("yaml")

from cfs.groundtruth.index import derive_index, write_index  # noqa: E402
from cfs.surrogate.data import load_value_dataset  # noqa: E402

EX = ["EX_glc__D_e", "EX_o2_e", "EX_mg2_e"]
GIDS = ["g0", "g1"]
N_BASE, N_ROUND1 = 20, 8


def _rows(gid: str, mids: np.ndarray, rng) -> pd.DataFrame:
    n = len(mids)
    return pd.DataFrame(
        {
            "genome_id": gid,
            "index_hash": "",
            "medium_id": mids,
            "alpha": 1.0,
            "eps": 1e-3,
            "mu_max": rng.uniform(0.5, 2.0, n),
            "status": "optimal",
            # The medium doubles as the medium's identity, so an assertion on x_val is
            # an assertion on *which media* were held out.
            "medium": [[float(m), 0.02, 0.01] for m in mids],
            "z": [[] for _ in mids],
            "shadow": [[-0.5, -0.1, 0.0] for _ in mids],
        }
    )


def _u_space(ds):
    """The held-out gradient target in ``u`` space — ``train._du``, in numpy."""
    return ds.g_val * (1.0 - ds.x_val) ** 2 / ds.x_scale[:, None, :]


@pytest.fixture
def labels(tmp_path):
    """Base shards for two organisms, plus a round-1 shard written separately."""
    rng = np.random.default_rng(0)
    result = derive_index(GIDS, [set(EX), set(EX)])
    index_path = tmp_path / "metabolite_index.json"
    digest = write_index(result, index_path)

    root = tmp_path / "labels"
    round1 = {}
    for gid in GIDS:
        (root / f"{gid}.exchanges.json").parent.mkdir(parents=True, exist_ok=True)
        (root / f"{gid}.exchanges.json").write_text(json.dumps({"exchanges": EX}))
        shard = root / gid / "eps_0.001"
        shard.mkdir(parents=True)
        base = _rows(gid, np.arange(N_BASE), rng)
        base["index_hash"] = digest
        base.to_parquet(shard / "part.parquet", index=False)
        # Round N offsets medium_id by N * _ROUND_STRIDE (cfs.sampling.generate).
        extra = _rows(gid, 1_000_000 + np.arange(N_ROUND1), rng)
        extra["index_hash"] = digest
        round1[gid] = (shard / "part.round1.parquet", extra)
    return root, index_path, round1


def test_topup_media_train_only_and_the_val_set_does_not_move(labels):
    root, index_path, round1 = labels

    before = load_value_dataset(root, index_path, eps=1e-3, seed=0)
    assert before.rounds_present == [0]
    assert before.x_val.shape[1] == 4 and before.x_train.shape[1] == 16

    for path, df in round1.values():
        df.to_parquet(path, index=False)
    after = load_value_dataset(root, index_path, eps=1e-3, seed=0)

    # Every top-up medium went to training, and not one of them displaced a
    # held-out medium: same size, same media, same order. `mu_max` is the raw
    # label, so it identifies the medium; `x_val` deliberately does *not* have to
    # match, because the extra rows refit the per-metabolite input scale.
    assert after.rounds_present == [0, 1]
    assert after.x_val.shape[1] == before.x_val.shape[1]
    assert after.x_train.shape[1] == before.x_train.shape[1] + N_ROUND1
    assert np.array_equal(after.mu_val, before.mu_val)
    assert np.array_equal(after.gvalid_val, before.gvalid_val)

    # `before` had only round-0 media, so equality with it *is* the statement that
    # the held-out set is drawn from the base design alone.

    # And the gate itself is unmoved by the refit transform: the loader chain-rules
    # the target by `x_scale` and `train._du` divides it back out, so the u-space
    # shadow price is exactly what the solver reported either way. This is why a
    # shifting `x_scale` is a model change and not a metric change.
    assert np.allclose(_u_space(after), _u_space(before), rtol=1e-5)


def test_val_split_is_stable_under_seed_and_organism_stacking(labels):
    root, index_path, round1 = labels
    for path, df in round1.values():
        df.to_parquet(path, index=False)
    a = load_value_dataset(root, index_path, eps=1e-3, seed=0)
    b = load_value_dataset(root, index_path, eps=1e-3, seed=0)
    c = load_value_dataset(root, index_path, eps=1e-3, seed=1)
    assert np.array_equal(a.x_val, b.x_val)
    assert not np.array_equal(a.x_val, c.x_val)  # the seed still chooses the split
