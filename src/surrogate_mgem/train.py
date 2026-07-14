"""Phase 1 training: fit a fixed-community growth surrogate from the tidy tables.

Assembles a feature matrix (medium uptake vector, aligned to the shared exchange
universe) and target matrix (per-member growth) for one community, trains a
:class:`~surrogate_mgem.model.GrowthSurrogate`, and reports held-out accuracy.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from surrogate_mgem.model import GrowthSurrogate

LOGGER = logging.getLogger("surrogate-mgem.train")


@dataclass
class FixedCommunityDataset:
    """Feature/target matrices for one fixed community."""

    community_id: str
    X: np.ndarray  # (n_samples, n_medium_exchanges)
    Y: np.ndarray  # (n_samples, n_members)
    feature_names: list[str]
    target_names: list[str]  # genome ids, column order of Y


def load_fixed_community_dataset(
    data_dir: Path, community_id: str | None = None
) -> FixedCommunityDataset:
    """Load the tidy tables and build (X, Y) for one community.

    ``community_id`` defaults to the community with the most feasible samples.
    Features are aligned to ``exchange_universe.json``'s medium exchanges (0 for
    components absent from a given medium); targets are per-member growth.
    """
    samples = pd.read_csv(data_dir / "samples.csv")
    media = pd.read_csv(data_dir / "media.csv")
    growth = pd.read_csv(data_dir / "member_growth.csv")
    universe = json.loads((data_dir / "exchange_universe.json").read_text())
    feature_names = list(universe["medium_exchanges"])

    feasible = samples[samples["feasible"]]
    if community_id is None:
        community_id = feasible["community_id"].value_counts().idxmax()
    sample_ids = feasible.loc[feasible["community_id"] == community_id, "sample_id"].tolist()
    if not sample_ids:
        raise ValueError(f"No feasible samples for community {community_id!r}.")

    # X: medium uptake wide, reindexed to the universe (missing -> 0).
    media_c = media[media["sample_id"].isin(sample_ids)]
    x_wide = media_c.pivot_table(
        index="sample_id", columns="exchange_id", values="uptake", fill_value=0.0
    ).reindex(index=sample_ids, columns=feature_names, fill_value=0.0)
    # Y: per-member growth wide (genomes as columns, stable sorted order).
    growth_c = growth[growth["sample_id"].isin(sample_ids)]
    target_names = sorted(growth_c["genome_id"].unique())
    y_wide = growth_c.pivot_table(index="sample_id", columns="genome_id", values="growth").reindex(
        index=sample_ids, columns=target_names
    )
    if y_wide.isna().any().any():
        raise ValueError(f"Community {community_id!r} has samples missing a member's growth.")

    return FixedCommunityDataset(
        community_id=str(community_id),
        X=x_wide.to_numpy(dtype=np.float32),
        Y=y_wide.to_numpy(dtype=np.float32),
        feature_names=feature_names,
        target_names=target_names,
    )


def train_fixed_community(
    dataset: FixedCommunityDataset,
    out_dir: Path,
    *,
    epochs: int = 300,
    lr: float = 1e-3,
    test_size: float = 0.2,
    seed: int = 0,
) -> dict:
    """Train a surrogate for one community; write the model + metrics; return metrics."""
    out_dir.mkdir(parents=True, exist_ok=True)
    X_tr, X_te, Y_tr, Y_te = train_test_split(
        dataset.X, dataset.Y, test_size=test_size, random_state=seed
    )
    model = GrowthSurrogate(n_in=dataset.X.shape[1], n_out=dataset.Y.shape[1])
    model.fit(X_tr, Y_tr, epochs=epochs, lr=lr, seed=seed)

    pred = model.predict(X_te)
    metrics = {
        "community_id": dataset.community_id,
        "n_samples": int(len(dataset.X)),
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
        "n_members": int(dataset.Y.shape[1]),
        # multioutput='uniform_average' would hide a bad member; report both.
        "r2_overall": float(r2_score(Y_te, pred)),
        "r2_per_member": {
            g: float(r2_score(Y_te[:, i], pred[:, i])) for i, g in enumerate(dataset.target_names)
        },
        "mae_overall": float(mean_absolute_error(Y_te, pred)),
    }
    model.save(out_dir / "growth_surrogate.pt")
    (out_dir / "surrogate_meta.json").write_text(
        json.dumps(
            {
                "community_id": dataset.community_id,
                "feature_names": dataset.feature_names,
                "target_names": dataset.target_names,
            },
            indent=2,
        )
    )
    (out_dir / "train_metrics.json").write_text(json.dumps(metrics, indent=2))
    LOGGER.info(
        "Trained %s: %d samples, overall test R2=%.3f MAE=%.4g",
        dataset.community_id,
        metrics["n_samples"],
        metrics["r2_overall"],
        metrics["mae_overall"],
    )
    return metrics
