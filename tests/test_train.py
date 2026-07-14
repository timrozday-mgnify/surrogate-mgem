"""Tests for dataset assembly + training driver on synthetic tidy tables."""

import json

import numpy as np
import pandas as pd

from surrogate_mgem.train import load_fixed_community_dataset, train_fixed_community


def _write_synthetic_tables(data_dir, n=60, seed=0):
    """Write generate()-shaped tables for one fixed 2-member community."""
    rng = np.random.default_rng(seed)
    features = ["EX_a_m", "EX_b_m", "EX_c_m"]
    uptake = rng.random((n, len(features))) * 10.0

    samples, media, growth = [], [], []
    for sid in range(n):
        samples.append(
            {
                "sample_id": sid,
                "community_id": "g1+g2",
                "n_members": 2,
                "feasible": True,
                "community_growth": float(uptake[sid].sum()),
            }
        )
        for j, ex in enumerate(features):
            if uptake[sid, j] > 0:
                media.append({"sample_id": sid, "exchange_id": ex, "uptake": float(uptake[sid, j])})
        # growth is a simple linear function of the medium (learnable).
        growth.append({"sample_id": sid, "genome_id": "g1", "growth": float(uptake[sid, 0] * 2)})
        growth.append(
            {"sample_id": sid, "genome_id": "g2", "growth": float(uptake[sid, 1] + uptake[sid, 2])}
        )

    pd.DataFrame(samples).to_csv(data_dir / "samples.csv", index=False)
    pd.DataFrame(media).to_csv(data_dir / "media.csv", index=False)
    pd.DataFrame(growth).to_csv(data_dir / "member_growth.csv", index=False)
    (data_dir / "exchange_universe.json").write_text(
        json.dumps(
            {"medium_exchanges": features, "member_exchanges": ["EX_a_e", "EX_b_e", "EX_c_e"]}
        )
    )


def test_load_fixed_community_dataset(tmp_path):
    _write_synthetic_tables(tmp_path)
    ds = load_fixed_community_dataset(tmp_path)
    assert ds.community_id == "g1+g2"
    assert ds.feature_names == ["EX_a_m", "EX_b_m", "EX_c_m"]
    assert ds.target_names == ["g1", "g2"]
    assert ds.X.shape == (60, 3)
    assert ds.Y.shape == (60, 2)


def test_train_writes_model_and_metrics(tmp_path):
    _write_synthetic_tables(tmp_path)
    ds = load_fixed_community_dataset(tmp_path)
    out = tmp_path / "model"
    metrics = train_fixed_community(ds, out, n_models=3, epochs=150, seed=0)

    assert (out / "ensemble").is_dir()
    assert list((out / "ensemble").glob("member_*.pt"))  # ensemble members saved
    assert (out / "predictions.csv").exists()
    assert (out / "surrogate_meta.json").exists()
    assert set(metrics["r2_per_member"]) == {"g1", "g2"}
    # The synthetic target is exactly linear, so the surrogate should fit it well.
    assert metrics["r2_overall"] > 0.9
