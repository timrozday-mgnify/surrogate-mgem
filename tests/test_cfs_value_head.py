"""M3 Head A: the structural checks, on synthetic data (no parquet, no solver).

Fits the stacked value head to a known concave function whose gradient is known
analytically, then asserts the milestone gate on it. This is what fails if the
ICNN weight constraint, the negation, the mask, or the autodiff gradient wiring
breaks — none of which shows up as a bad training loss.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("equinox")

from cfs.surrogate.data import ValueDataset, _organism_arrays  # noqa: E402
from cfs.surrogate.picnn import batched_value, stack_heads  # noqa: E402
from cfs.surrogate.train import evaluate, train_value_heads  # noqa: E402

G, M, N = 3, 6, 400


def _synthetic() -> ValueDataset:
    """mu = -sum_m w_m (x_m - a_m)^2 — concave, gradient known exactly."""
    rng = np.random.default_rng(0)
    mask = np.ones((G, M), dtype=bool)
    mask[1, -2:] = False  # one organism missing two metabolites
    x = rng.uniform(0, 1, (G, N, M)).astype(np.float32) * mask[:, None, :]
    a = rng.uniform(0.2, 0.8, (G, 1, M)).astype(np.float32)
    w = rng.uniform(0.5, 2.0, (G, 1, M)).astype(np.float32) * mask[:, None, :]
    mu = -np.sum(w * (x - a) ** 2, axis=-1)
    g = (-2 * w * (x - a)) * mask[:, None, :]
    gvalid = np.ones((G, N), dtype=bool)
    n_val = N // 5
    return ValueDataset(
        genome_ids=[f"g{i}" for i in range(G)], exchanges=[f"EX_{m}" for m in range(M)],
        mask=mask,
        x_train=x[:, n_val:], mu_train=mu[:, n_val:], g_train=g[:, n_val:],
        gvalid_train=gvalid[:, n_val:],
        x_val=x[:, :n_val], mu_val=mu[:, :n_val], g_val=g[:, :n_val], gvalid_val=gvalid[:, :n_val],
        mu_scale=mu.std(axis=1).astype(np.float32), x_scale=np.ones((G, M), np.float32),
        index_hash="test",
    )


def test_gate_on_a_known_concave_target():
    ds = _synthetic()
    heads = train_value_heads(ds, width=64, depth=2, epochs=600, batch=64, lr=1e-2)
    diag = evaluate(heads, ds)
    assert diag["passed"], diag["per_organism"]
    for gid, d in diag["per_organism"].items():
        assert d["concavity_violation_rate"] == 0.0, gid


def test_untrained_head_is_concave_and_masked():
    """Concavity is structural, so it must hold at initialisation too."""
    mask = np.ones((2, M), dtype=bool)
    mask[0, 0] = False
    heads = stack_heads(jax.random.PRNGKey(1), 2, M, mask, width=16, depth=3)
    rng = np.random.default_rng(1)
    xa, xb = (rng.uniform(0, 1, (2, 64, M)).astype(np.float32) for _ in range(2))
    lam = 0.37
    mid = batched_value(heads, lam * xa + (1 - lam) * xb)
    chord = lam * batched_value(heads, xa) + (1 - lam) * batched_value(heads, xb)
    assert np.all(np.asarray(mid) >= np.asarray(chord) - 1e-5)

    # A masked metabolite cannot move the output.
    x0 = np.zeros((2, 1, M), dtype=np.float32)
    x1 = x0.copy()
    x1[0, 0, 0] = 1.0
    assert np.allclose(batched_value(heads, x0)[0], batched_value(heads, x1)[0])


def test_organism_arrays_signs_and_scatter(tmp_path):
    """The loader's saturation transform, dual sign, and zero-growth masking."""
    import json

    ex = ["EX_glc__D_e", "EX_o2_e"]
    (tmp_path / "g0.exchanges.json").write_text(json.dumps({"exchanges": ex}))
    shard = tmp_path / "genome_id=g0" / "eps=0.001"
    shard.mkdir(parents=True)
    pd.DataFrame([
        # a growing medium and a dead one; alpha != 1 rows must be dropped
        {"genome_id": "g0", "index_hash": "h", "medium_id": 0, "alpha": 1.0, "eps": 1e-3,
         "mu_max": 2.0, "status": "optimal", "medium": [0.02, 0.0], "z": [], "shadow": [-0.5, 0.0]},
        {"genome_id": "g0", "index_hash": "h", "medium_id": 0, "alpha": 0.5, "eps": 1e-3,
         "mu_max": 2.0, "status": "optimal", "medium": [0.02, 0.0], "z": [], "shadow": [-0.5, 0.0]},
        {"genome_id": "g0", "index_hash": "h", "medium_id": 1, "alpha": 1.0, "eps": 1e-3,
         "mu_max": 0.0, "status": "optimal", "medium": [0.0, 0.0], "z": [],
         "shadow": [-10000.0, 0.0]},
    ]).to_parquet(shard / "part.parquet", index=False)

    km_cfg = {"classes": {"sugars": 0.01}, "default": 0.01, "keywords": {"sugars": ["glc"]}}
    col = {"EX_o2_e": 0, "EX_glc__D_e": 1, "EX_other_e": 2}
    x, mu, g, gvalid, mask, ihash = _organism_arrays(tmp_path, "g0", 1e-3, col, km_cfg, 3)

    assert x.shape == (2, 3) and ihash == "h"
    assert mask.tolist() == [True, True, False]
    # saturation c/(Km+c) at c=0.02, Km=0.01 -> 2/3, scattered to the index column
    assert x[0, 1] == pytest.approx(2 / 3)
    # dual sign: stored shadow is the negated derivative, so dmu/dx > 0 here
    assert g[0, 1] == pytest.approx(0.5 * 1000.0)
    # the zero-growth row keeps its value label but its garbage duals are dropped
    assert gvalid.tolist() == [True, False] and np.all(g[1] == 0.0) and mu[1] == 0.0
