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
        epochs: int = 300,
        lr: float = 1e-3,
        batch_size: int = 128,
        val_split: float = 0.2,
        seed: int = 0,
        verbose: bool = False,
    ) -> dict[str, list[float]]:
        """Train on ``(X, Y)``; return per-epoch train/val MSE history.

        Standardisation is fit on the training split only (no leakage from the
        validation rows). With too few rows to split, all rows are used for both.
        """
        torch.manual_seed(seed)
        X_t = torch.as_tensor(np.asarray(X, dtype=np.float32))
        Y_t = torch.as_tensor(np.asarray(Y, dtype=np.float32))
        if Y_t.ndim == 1:
            Y_t = Y_t[:, None]
        n = len(X_t)
        n_val = int(round(n * val_split))
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
        val_idx, train_idx = perm[:n_val], perm[n_val:]
        if len(train_idx) == 0:  # tiny dataset: train and validate on everything
            train_idx = val_idx = perm
        self._set_standardisation(X_t[train_idx], Y_t[train_idx])

        opt = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        history: dict[str, list[float]] = {"train": [], "val": []}
        for _ in range(epochs):
            self.train()
            batch_perm = train_idx[torch.randperm(len(train_idx))]
            for start in range(0, len(batch_perm), batch_size):
                batch = batch_perm[start : start + batch_size]
                opt.zero_grad()
                loss = loss_fn(self(X_t[batch]), Y_t[batch])
                loss.backward()
                opt.step()
            self.eval()
            with torch.no_grad():
                history["train"].append(float(loss_fn(self(X_t[train_idx]), Y_t[train_idx])))
                history["val"].append(float(loss_fn(self(X_t[val_idx]), Y_t[val_idx])))
            if verbose and _ % 50 == 0:
                print(f"epoch {_}: train={history['train'][-1]:.4g} val={history['val'][-1]:.4g}")
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
