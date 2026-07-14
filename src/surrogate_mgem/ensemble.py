"""Deep ensemble of growth surrogates for predictions *with* uncertainty.

A single MLP gives a point estimate but no usable uncertainty. Training K
surrogates from different seeds and reading their disagreement (predictive std)
is a cheap, robust epistemic-uncertainty signal -- the quantity the active loop
uses to decide which media are worth a real solve.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from surrogate_mgem.model import GrowthSurrogate, inverse_density_weights


class GrowthEnsemble:
    """K independently-seeded :class:`GrowthSurrogate` models."""

    def __init__(
        self, n_in: int, n_out: int, n_models: int = 5, hidden: tuple[int, ...] = (256, 256)
    ):
        self.n_in = n_in
        self.n_out = n_out
        self.hidden = hidden
        self.models = [GrowthSurrogate(n_in, n_out, hidden) for _ in range(n_models)]

    def fit(self, X: np.ndarray, Y: np.ndarray, *, base_seed: int = 0, **fit_kwargs) -> list[dict]:
        """Fit every member; each gets a distinct seed so they disagree off-data.

        Returns the per-member training histories. Density weights (the default
        ``sample_weight="auto"``) are computed once from ``X`` here and shared
        across members -- they depend only on the data, not the seed, so this
        avoids ``n_models`` redundant kNN builds while keeping every member's
        weighting identical.
        """
        if fit_kwargs.get("sample_weight", "auto") == "auto":
            fit_kwargs["sample_weight"] = inverse_density_weights(np.asarray(X))
        self.last_history = [
            model.fit(X, Y, seed=base_seed + i, **fit_kwargs)
            for i, model in enumerate(self.models)
        ]
        return self.last_history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Ensemble mean prediction (n_samples, n_out)."""
        return self.predict_with_uncertainty(X)[0]

    def predict_with_uncertainty(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (mean, std) across ensemble members, both (n_samples, n_out).

        The per-output std is the epistemic uncertainty; aggregate it (e.g. mean
        over outputs) to get one acquisition score per candidate medium.
        """
        stacked = np.stack([m.predict(X) for m in self.models])  # (K, n, out)
        return stacked.mean(0), stacked.std(0)

    def save(self, directory: Path) -> None:
        """Save each member to ``member_{i}.pt`` under ``directory``."""
        directory.mkdir(parents=True, exist_ok=True)
        for i, model in enumerate(self.models):
            model.save(directory / f"member_{i}.pt")

    @classmethod
    def load(cls, directory: Path, hidden: tuple[int, ...] | None = None) -> GrowthEnsemble:
        """Load an ensemble saved by :meth:`save`.

        Each member reads its architecture from its own checkpoint; pass
        ``hidden`` only to override.
        """
        paths = sorted(directory.glob("member_*.pt"))
        if not paths:
            raise FileNotFoundError(f"No ensemble members found in {directory}.")
        models = [GrowthSurrogate.load(p, hidden=hidden) for p in paths]
        ensemble = cls.__new__(cls)
        ensemble.n_in = models[0].n_in
        ensemble.n_out = models[0].n_out
        ensemble.hidden = models[0].hidden
        ensemble.models = models
        return ensemble
