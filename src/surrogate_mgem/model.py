"""Surrogate models that map a medium uptake vector to per-member growth.

Phase 1 is the fixed-community surrogate: a small standardising ReLU MLP,
``g(uptake) -> per-member growth``. ReLU MLPs are a natural fit because the
LP/QP optimum is piecewise-linear in the RHS bounds. The model is differentiable
so Phase 3 can optimise media by gradient ascent; standardisation stats are kept
as buffers so a saved checkpoint is self-contained.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

_STD_FLOOR = 1e-8  # guard against divide-by-zero on constant features/targets


def inverse_density_weights(
    X: np.ndarray, k: int = 15, cap: float = 10.0
) -> np.ndarray:
    """Per-row weights that down-weight densely-sampled regions of feature space.

    The active-learning loop deliberately clusters real solves in "tricky"
    high-uncertainty regions; left unweighted, those points dominate the MSE and
    the surrogate over-fits the hard corners at the expense of the rest of the
    space. Here density is estimated from ``X`` itself: standardise the features
    (so every exchange axis is comparable), take each row's mean Euclidean
    distance to its ``k`` nearest neighbours -- small distance = dense region --
    and weight ``w_i`` proportional to that distance (inverse density). Weights
    are normalised to mean 1 and clipped to ``[1/cap, cap]`` so a few outliers
    can't dominate and no dense point is fully ignored. On roughly-uniform data
    the weights come out ~1 (near no-op).

    ponytail: naive kNN density (KDTree, O(n log n)); swap for a KDE / adaptive
    bandwidth only if this clip-and-normalise heuristic proves too blunt.
    """
    from sklearn.neighbors import NearestNeighbors

    X = np.asarray(X, dtype=np.float64)
    n = len(X)
    if n < 3:
        return np.ones(n, dtype=np.float32)
    Xs = (X - X.mean(0)) / np.clip(X.std(0), _STD_FLOOR, None)
    k = min(k, n - 1)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(Xs)  # +1: first neighbour is self
    dist, _ = nn.kneighbors(Xs)
    mean_dist = dist[:, 1:].mean(axis=1)  # drop the self (zero) distance
    w = mean_dist / max(mean_dist.mean(), _STD_FLOOR)
    return np.clip(w, 1.0 / cap, cap).astype(np.float32)


class GrowthSurrogate(nn.Module):
    """Standardising ReLU-MLP regressor from medium vector to per-member growth."""

    def __init__(self, n_in: int, n_out: int, hidden: tuple[int, ...] = (256, 256)):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.hidden = tuple(hidden)
        layers: list[nn.Module] = []
        width = n_in
        for h in hidden:
            layers += [nn.Linear(width, h), nn.ReLU()]
            width = h
        layers.append(nn.Linear(width, n_out))
        self.net = nn.Sequential(*layers)
        # Standardisation buffers (identity until fit() sets them).
        self.register_buffer("x_mean", torch.zeros(n_in))
        self.register_buffer("x_std", torch.ones(n_in))
        self.register_buffer("y_mean", torch.zeros(n_out))
        self.register_buffer("y_std", torch.ones(n_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict de-standardised growth for standardised-then-raw input ``x``."""
        z = (x - self.x_mean) / self.x_std
        return self.net(z) * self.y_std + self.y_mean

    def _set_standardisation(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        self.x_mean.copy_(X.mean(0))
        self.x_std.copy_(X.std(0).clamp_min(_STD_FLOOR))
        self.y_mean.copy_(Y.mean(0))
        self.y_std.copy_(Y.std(0).clamp_min(_STD_FLOOR))

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        *,
        epochs: int = 1000,
        lr: float = 1e-3,
        batch_size: int | None = None,
        max_batch_size: int | None = None,
        sample_weight: np.ndarray | str | None = "auto",
        patience: int = 30,
        plateau_patience: int = 8,
        min_lr: float = 1e-5,
        weight_decay: float = 0.0,
        val_split: float = 0.2,
        seed: int = 0,
        verbose: bool = False,
    ) -> dict:
        """Train on ``(X, Y)`` self-tuning the schedule; return the training history.

        ``epochs`` is only a **cap** -- training early-stops on validation loss
        (``patience`` epochs without improvement) and restores the best weights,
        so the caller sets architecture, not a schedule. The batch size grows and
        the learning rate decays automatically, staged so they don't both react to
        the same plateau: on ``plateau_patience`` epochs without improvement the
        batch first doubles (up to a cap), and only once the batch is maxed does
        the LR halve (down to ``min_lr``).

        ``sample_weight`` down-weights dense regions of feature space by default
        (``"auto"`` -> :func:`inverse_density_weights`); pass ``None`` for uniform
        or an explicit per-row array. Weights enter a weighted MSE and are used
        for the validation loss too, so early stopping tracks the same objective.

        Standardisation is fit on the training split only (no leakage from the
        validation rows). With too few rows to split, all rows are used for both.
        Returns per-epoch ``train``/``val`` curves plus summary scalars
        (``epochs_run``, ``best_epoch``, ``best_val``, ``stopped_early``,
        ``final_lr``, ``final_batch_size``).
        """
        torch.manual_seed(seed)
        X_t = torch.as_tensor(np.asarray(X, dtype=np.float32))
        Y_t = torch.as_tensor(np.asarray(Y, dtype=np.float32))
        if Y_t.ndim == 1:
            Y_t = Y_t[:, None]
        n = len(X_t)

        if sample_weight is None:
            w_t = torch.ones(n)
        elif isinstance(sample_weight, str) and sample_weight == "auto":
            w_t = torch.as_tensor(inverse_density_weights(X_t.numpy()))
        else:
            w_t = torch.as_tensor(np.asarray(sample_weight, dtype=np.float32))

        n_val = int(round(n * val_split))
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
        val_idx, train_idx = perm[:n_val], perm[n_val:]
        has_val = len(train_idx) > 0 and len(val_idx) > 0
        if not has_val:  # tiny dataset: train and validate on everything
            train_idx = val_idx = perm
        self._set_standardisation(X_t[train_idx], Y_t[train_idx])

        n_train = len(train_idx)
        if batch_size is None:
            batch_size = int(np.clip(n_train / 10, 32, 512))
        if max_batch_size is None:
            max_batch_size = min(n_train, max(batch_size * 8, 1024))

        def weighted_mse(idx: torch.Tensor) -> torch.Tensor:
            w = w_t[idx][:, None]
            sq = (self(X_t[idx]) - Y_t[idx]) ** 2
            return (w * sq).sum() / (w.sum() * sq.shape[1]).clamp_min(_STD_FLOOR)

        opt = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        history: dict = {"train": [], "val": []}
        best_val, best_epoch, best_state = float("inf"), 0, None
        since_improve = 0  # epochs since val improved (drives batch growth / LR decay)
        cur_lr = lr
        for epoch in range(epochs):
            self.train()
            batch_perm = train_idx[torch.randperm(len(train_idx))]
            for start in range(0, len(batch_perm), batch_size):
                batch = batch_perm[start : start + batch_size]
                opt.zero_grad()
                weighted_mse(batch).backward()
                opt.step()
            self.eval()
            with torch.no_grad():
                tr = float(weighted_mse(train_idx))
                va = float(weighted_mse(val_idx))
            history["train"].append(tr)
            history["val"].append(va)
            if verbose and epoch % 50 == 0:
                print(f"epoch {epoch}: train={tr:.4g} val={va:.4g} bs={batch_size} lr={cur_lr:.2g}")

            if va < best_val - 1e-9:
                best_val, best_epoch, best_state = va, epoch, {
                    k: v.detach().clone() for k, v in self.state_dict().items()
                }
                since_improve = 0
                continue
            since_improve += 1
            if since_improve >= patience:  # early stop
                break
            # Staged plateau response: grow the batch first, then decay the LR.
            if since_improve % plateau_patience == 0:
                if batch_size < max_batch_size:
                    batch_size = min(batch_size * 2, max_batch_size)
                elif cur_lr > min_lr:
                    cur_lr = max(cur_lr * 0.5, min_lr)
                    for g in opt.param_groups:
                        g["lr"] = cur_lr

        stopped_early = since_improve >= patience
        if best_state is not None:
            self.load_state_dict(best_state)
        history.update(
            epochs_run=len(history["train"]),
            best_epoch=best_epoch,
            best_val=best_val,
            stopped_early=bool(stopped_early),
            final_lr=cur_lr,
            final_batch_size=batch_size,
        )
        return history

    @torch.no_grad()
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return de-standardised growth predictions for ``X`` as a numpy array."""
        self.eval()
        X_t = torch.as_tensor(np.asarray(X, dtype=np.float32))
        return self(X_t).cpu().numpy()

    def save(self, path: Path) -> None:
        """Save weights, buffers, and shape/architecture to a checkpoint."""
        torch.save(
            {
                "state_dict": self.state_dict(),
                "n_in": self.n_in,
                "n_out": self.n_out,
                "hidden": list(self.hidden),
            },
            path,
        )

    @classmethod
    def load(cls, path: Path, hidden: tuple[int, ...] | None = None) -> GrowthSurrogate:
        """Load a checkpoint saved by :meth:`save`.

        Architecture is read from the checkpoint (older checkpoints without it
        fall back to ``(256, 256)``); pass ``hidden`` only to override.
        """
        blob = torch.load(path, weights_only=True)
        if hidden is None:
            hidden = tuple(blob.get("hidden", (256, 256)))
        model = cls(blob["n_in"], blob["n_out"], hidden=hidden)
        model.load_state_dict(blob["state_dict"])
        model.eval()
        return model
