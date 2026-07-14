"""Phase 1 training: fit a fixed-community growth surrogate from the tidy tables.

Assembles a feature matrix (medium uptake vector, aligned to the shared exchange
universe) and target matrix (per-member growth) for one community, trains an
ensemble :class:`~surrogate_mgem.ensemble.GrowthEnsemble` (optionally growing the
training set with the active-learning loop), and writes report inputs:
held-out predictions, metrics, and -- for the active loop -- the per-round
history. A Quarto report renders these for inspection.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from surrogate_mgem.active import ActiveConfig, active_learning_loop
from surrogate_mgem.ensemble import GrowthEnsemble

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

    # Drop features this community never exchanges (all-zero across its media):
    # they carry no signal and only inflate the input dimension. The reduced
    # feature list stays a subset of the shared universe, so it still aligns.
    active_cols = x_wide.columns[(x_wide != 0).any()].tolist()
    x_wide = x_wide[active_cols]

    return FixedCommunityDataset(
        community_id=str(community_id),
        X=x_wide.to_numpy(dtype=np.float32),
        Y=y_wide.to_numpy(dtype=np.float32),
        feature_names=active_cols,
        target_names=target_names,
    )


def _test_metrics(pred: np.ndarray, Y_te: np.ndarray, target_names: list[str]) -> dict:
    """Overall + per-member R^2/MAE on the held-out set."""
    return {
        "r2_overall": float(r2_score(Y_te, pred)),
        "mae_overall": float(mean_absolute_error(Y_te, pred)),
        "r2_per_member": {
            g: float(r2_score(Y_te[:, i], pred[:, i])) for i, g in enumerate(target_names)
        },
    }


def _write_report_inputs(
    out_dir: Path,
    dataset: FixedCommunityDataset,
    ensemble: GrowthEnsemble,
    X_te: np.ndarray,
    Y_te: np.ndarray,
    metrics: dict,
    history: pd.DataFrame | None,
) -> None:
    """Write predictions, metrics, meta (and history) the Quarto report reads."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ensemble.save(out_dir / "ensemble")

    mean, std = ensemble.predict_with_uncertainty(X_te)
    rows = []
    for i, g in enumerate(dataset.target_names):
        for r in range(len(X_te)):
            rows.append(
                {
                    "genome_id": g,
                    "y_true": float(Y_te[r, i]),
                    "y_pred": float(mean[r, i]),
                    "y_std": float(std[r, i]),
                }
            )
    pd.DataFrame(rows).to_csv(out_dir / "predictions.csv", index=False)
    (out_dir / "train_metrics.json").write_text(json.dumps(metrics, indent=2))
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
    if history is not None:
        history.to_csv(out_dir / "active_history.csv", index=False)


def train_fixed_community(
    dataset: FixedCommunityDataset,
    out_dir: Path,
    *,
    n_models: int = 5,
    hidden: tuple[int, ...] = (256, 256),
    epochs: int = 300,
    test_size: float = 0.2,
    n_train: int | None = None,
    seed: int = 0,
) -> dict:
    """Train an ensemble for one community (no active loop); write report inputs.

    ``hidden`` sets the MLP architecture (layers x width); ``n_train`` optionally
    caps the training split to that many rows (seeded subsample) so a fixed
    dataset can be swept along a learning curve. The held-out test split is
    unaffected by the cap.
    """
    X_tr, X_te, Y_tr, Y_te = train_test_split(
        dataset.X, dataset.Y, test_size=test_size, random_state=seed
    )
    if n_train is not None and n_train < len(X_tr):
        idx = np.random.default_rng(seed).choice(len(X_tr), size=n_train, replace=False)
        X_tr, Y_tr = X_tr[idx], Y_tr[idx]
    ensemble = GrowthEnsemble(
        dataset.X.shape[1], dataset.Y.shape[1], n_models=n_models, hidden=hidden
    )
    ensemble.fit(X_tr, Y_tr, base_seed=seed, epochs=epochs)
    metrics = {
        "community_id": dataset.community_id,
        "mode": "static",
        "hidden": list(hidden),
        "n_models": int(n_models),
        "n_samples": int(len(dataset.X)),
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
        "n_members": int(dataset.Y.shape[1]),
        **_test_metrics(ensemble.predict(X_te), Y_te, dataset.target_names),
    }
    _write_report_inputs(out_dir, dataset, ensemble, X_te, Y_te, metrics, history=None)
    LOGGER.info(
        "Trained %s (static): %d samples, test R2=%.3f",
        dataset.community_id,
        metrics["n_samples"],
        metrics["r2_overall"],
    )
    return metrics


def train_fixed_community_active(
    dataset: FixedCommunityDataset,
    members: list,
    out_dir: Path,
    *,
    active_config: ActiveConfig,
    solver: str = "hybrid",
    tradeoff: float = 0.35,
    test_size: float = 0.2,
    seed: int = 0,
) -> dict:
    """Train with the active-learning loop; write report inputs incl. per-round history.

    ``members`` are the :class:`~surrogate_mgem.data.GenomeModel`\\ s of this
    community (used to build the real solver oracle). The held-out test split is
    fixed up front and never fed to the loop, so the history's R^2 curve measures
    honest generalisation as solves accrue.
    """
    from surrogate_mgem.data import make_fixed_community_evaluator

    X_tr, X_te, Y_tr, Y_te = train_test_split(
        dataset.X, dataset.Y, test_size=test_size, random_state=seed
    )
    evaluate, active_mask = make_fixed_community_evaluator(
        members, dataset.feature_names, dataset.target_names, solver, tradeoff
    )
    ensemble, history, (X_all, _) = active_learning_loop(
        X_tr, Y_tr, evaluate, active_mask, active_config, X_test=X_te, Y_test=Y_te
    )
    metrics = {
        "community_id": dataset.community_id,
        "mode": "active",
        "n_seed_samples": int(len(X_tr)),
        "n_final_train": int(len(X_all)),
        "n_test": int(len(X_te)),
        "n_members": int(dataset.Y.shape[1]),
        "rounds": active_config.rounds,
        **_test_metrics(ensemble.predict(X_te), Y_te, dataset.target_names),
    }
    _write_report_inputs(out_dir, dataset, ensemble, X_te, Y_te, metrics, history=history)
    LOGGER.info(
        "Trained %s (active): seed %d -> %d train over %d rounds, test R2=%.3f",
        dataset.community_id,
        metrics["n_seed_samples"],
        metrics["n_final_train"],
        active_config.rounds,
        metrics["r2_overall"],
    )
    return metrics


# ---------------------------------------------------------------------------
# Single active-learning round (pipeline step): augment the tidy tables in place
# ---------------------------------------------------------------------------

_TABLES = ("samples", "media", "member_growth", "membership")


def _write_augmented_dataset(
    data_dir: Path,
    out_dir: Path,
    dataset: FixedCommunityDataset,
    X_new: np.ndarray,
    Y_new: np.ndarray,
    round_index: int,
) -> None:
    """Write a single-community dataset dir: this community's rows + the new solves.

    Emits the tidy tables (and ``exchange_universe.json``) in the same schema
    :func:`load_fixed_community_dataset` reads, so the next round / the model
    sweep can consume the output as an ordinary dataset directory. New samples get
    fresh string ids ``act_r{round}_{k}``; per-member exchange fluxes are not
    recorded (the oracle returns growth only, which is all fixed-community training
    uses).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(data_dir / "exchange_universe.json", out_dir / "exchange_universe.json")

    samples = pd.read_csv(data_dir / "samples.csv")
    kept_ids = samples.loc[samples["community_id"] == dataset.community_id, "sample_id"].tolist()

    new_samples, new_media, new_growth, new_membership = [], [], [], []
    for k in range(len(X_new)):
        sid = f"act_r{round_index}_{k}"
        new_samples.append(
            {
                "sample_id": sid,
                "community_id": dataset.community_id,
                "n_members": len(dataset.target_names),
                "feasible": True,
                "community_growth": np.nan,
            }
        )
        for j, feat in enumerate(dataset.feature_names):
            if X_new[k, j] > 0:
                new_media.append(
                    {"sample_id": sid, "exchange_id": feat, "uptake": float(X_new[k, j])}
                )
        for i, genome in enumerate(dataset.target_names):
            new_growth.append({"sample_id": sid, "genome_id": genome, "growth": float(Y_new[k, i])})
            new_membership.append({"sample_id": sid, "genome_id": genome})

    new_rows = {
        "samples": new_samples,
        "media": new_media,
        "member_growth": new_growth,
        "membership": new_membership,
    }
    for name in _TABLES:
        path = data_dir / f"{name}.csv"
        table = pd.read_csv(path) if path.exists() else pd.DataFrame()
        if not table.empty:
            table = table[table["sample_id"].isin(kept_ids)]
        combined = pd.concat([table, pd.DataFrame(new_rows[name])], ignore_index=True)
        combined.to_csv(out_dir / f"{name}.csv", index=False)


def run_active_round(
    data_dir: Path,
    community_id: str | None,
    roster_path: Path,
    out_dir: Path,
    *,
    active_config: ActiveConfig,
    solver: str = "hybrid",
    tradeoff: float = 0.35,
    round_index: int = 0,
) -> dict:
    """One active round for one community; write the augmented single-community dir.

    Loads the community's current tables, builds the real solver oracle, solves a
    diverse high-uncertainty batch (:func:`~surrogate_mgem.active.active_round`),
    and appends the new samples to a fresh output dataset dir.
    """
    from surrogate_mgem.active import active_round
    from surrogate_mgem.data import (
        make_fixed_community_evaluator,
        members_for_community,
        read_roster,
    )

    dataset = load_fixed_community_dataset(data_dir, community_id)
    members = members_for_community(read_roster(roster_path), dataset.community_id)
    evaluate, active_mask = make_fixed_community_evaluator(
        members, dataset.feature_names, dataset.target_names, solver, tradeoff
    )
    X_new, Y_new = active_round(
        dataset.X, dataset.Y, evaluate, active_mask, active_config, round_index
    )
    _write_augmented_dataset(data_dir, out_dir, dataset, X_new, Y_new, round_index)
    metrics = {
        "community_id": dataset.community_id,
        "round": round_index,
        "n_prior": int(len(dataset.X)),
        "n_new_feasible": int(len(X_new)),
        "n_total": int(len(dataset.X) + len(X_new)),
    }
    (out_dir / "active_round_metrics.json").write_text(json.dumps(metrics, indent=2))
    LOGGER.info(
        "Active round %d on %s: +%d feasible (%d -> %d).",
        round_index,
        dataset.community_id,
        metrics["n_new_feasible"],
        metrics["n_prior"],
        metrics["n_total"],
    )
    return metrics
