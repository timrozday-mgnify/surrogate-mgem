"""One active round appends feasible solves that reload as extra training rows.

Exercises the fiddly bit -- ``active.active_round`` (torch, no solver) plus
``train._write_augmented_dataset`` writing tidy tables that
``load_fixed_community_dataset`` reads back -- with a synthetic oracle, so it
stays off the micom/cobra ``slow`` path.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from surrogate_mgem.active import ActiveConfig, active_round
from surrogate_mgem.train import _write_augmented_dataset, load_fixed_community_dataset

COMMUNITY = "gA+gB"
FEATURES = ["EX_a_m", "EX_b_m", "EX_c_m"]
MEMBERS = ["gA", "gB"]


def _write_seed_dataset(data_dir, n_samples=8):
    """Write a tiny single-community tidy dataset the loader accepts."""
    rng = np.random.default_rng(0)
    (data_dir / "exchange_universe.json").write_text(
        json.dumps({"medium_exchanges": FEATURES, "member_exchanges": ["EX_a_e"]})
    )
    samples, media, growth = [], [], []
    for s in range(n_samples):
        samples.append(
            {
                "sample_id": s,
                "community_id": COMMUNITY,
                "n_members": len(MEMBERS),
                "feasible": True,
                "community_growth": 1.0,
            }
        )
        uptake = rng.uniform(1.0, 10.0, size=len(FEATURES))
        for feat, u in zip(FEATURES, uptake, strict=True):
            media.append({"sample_id": s, "exchange_id": feat, "uptake": float(u)})
        for m in MEMBERS:
            growth.append({"sample_id": s, "genome_id": m, "growth": float(uptake.mean())})
    pd.DataFrame(samples).to_csv(data_dir / "samples.csv", index=False)
    pd.DataFrame(media).to_csv(data_dir / "media.csv", index=False)
    pd.DataFrame(growth).to_csv(data_dir / "member_growth.csv", index=False)


def test_active_round_roundtrip(tmp_path):
    data_dir = tmp_path / "seed"
    data_dir.mkdir()
    _write_seed_dataset(data_dir)
    dataset = load_fixed_community_dataset(data_dir, COMMUNITY)
    prior_n = len(dataset.X)

    # Always-feasible synthetic oracle: growth = mean uptake per member.
    def evaluate(vector: np.ndarray) -> np.ndarray:
        return np.full(len(MEMBERS), float(vector.mean()), dtype=np.float32)

    active_mask = np.ones(len(dataset.feature_names), dtype=bool)
    config = ActiveConfig(batch_size=3, n_candidates=20, epochs=5, n_models=2, hidden=(8, 8))
    X_new, Y_new = active_round(dataset.X, dataset.Y, evaluate, active_mask, config, round_index=0)

    assert len(X_new) == 3  # every pick feasible
    out_dir = tmp_path / "round0"
    _write_augmented_dataset(data_dir, out_dir, dataset, X_new, Y_new, round_index=0)

    reloaded = load_fixed_community_dataset(out_dir, COMMUNITY)
    assert len(reloaded.X) == prior_n + 3
    assert reloaded.Y.shape[1] == len(MEMBERS)
    # New rows are present and feasible under fresh ids.
    samples = pd.read_csv(out_dir / "samples.csv")
    new_ids = [sid for sid in samples["sample_id"].astype(str) if sid.startswith("act_r0_")]
    assert len(new_ids) == 3
