"""Random-forest baseline — the architecture to beat, scored on the same gate.

A forest is the obvious thing to reach for on tabular data and it has none of the
structure the plan asks for: not concave, not monotone, and piecewise constant, so
its analytic gradient is zero everywhere. It exists here to localise where the
concave head's deficit actually is. If the forest wins on ``value_r2`` (0.538 is
not a strong number) but loses on ``grad_cosine``, the Sobolev structure is
earning its keep and the value head is what needs capacity; if it wins on both,
the problem is not the concavity constraint.

Shadow prices come from **central finite differences in ``u`` space**, the same
coordinate the gate lives in (:func:`cfs.surrogate.train._du`), so the number is
directly comparable to a trained head's. The step is ``delta`` times each
metabolite's own kink scale ``s_m``, because a step that is uniform in ``u`` is
five decades wrong for something and every metabolite's ramp sits somewhere
different (§4.7). The stored inputs are already ``x = u / (u + s)``, so the probe
inverts that, steps in ``u``, and maps back.

``delta`` defaults to **0.05**, and the choice matters more than it looks. Two
opposing errors bracket it: a step far below the forest's leaf width lands both
probes in the same leaf and returns 0 (scoring the forest's *resolution*, not its
slope), while a step comparable to the ramp averages *across* the kink and returns
a slope from the wrong regime. Both are real on the roster, so the worst-organism
cosine is U-shaped in delta::

    delta   0.01   0.02   0.05   0.25    0.5    1.0
    worst  0.398  0.451  0.374  0.209 -0.102 -0.023
    mean   0.618  0.643  0.653  0.643  0.623  0.498

and the two ends hit *different* metabolites. Carbon sources want a small step and
invert outright with a large one (``EX_cytd_e`` 0.883 / 0.786 / 0.613 / -0.107 /
-0.729 over the same sweep) -- those negatives are an artifact of the probe, not
non-monotonicity in the forest. Ions want a larger one and fall off below 0.05
(``EX_mg2_e`` 0.824 / 0.882 / 0.998 / 0.996 / 0.993 / 0.982).

So **no single delta is right for every metabolite**, and an aggregate gradient
figure for the forest is not a well-defined number. 0.05 sits at the best mean,
near the worst-case optimum, and in the middle of ``EX_mg2_e``'s stable plateau --
and it is that plateau, not any single run, that makes the ion result trustworthy.
Quote a per-cell number only where it is flat in delta; sweep first.

A *linear* synthetic cannot detect any of this, having no kink to step over, which
is how 0.25 was originally justified on evidence that could not falsify it.
``tests/test_cfs_baseline_rf.py`` therefore also carries a kinked case.

Diagnostics land in :func:`cfs.surrogate.train.score`'s schema, minus the two
fields a forest has no answer for (concavity violations, Hessian conditioning).
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np

from cfs.surrogate.data import load_value_dataset
from cfs.surrogate.train import score

LOGGER = logging.getLogger("cfs.surrogate.baseline")

# x = u/(u+s) reaches 1.0 in float32 for a replete metabolite; the inverse
# u = x s/(1-x) would then be inf.
_X_MAX = 1.0 - 1e-6


def _u_space_gradients(rf, xv: np.ndarray, s: np.ndarray, delta: float) -> np.ndarray:
    """``d(mu)/du`` by central difference, one column at a time. ``(B, |M_i|)``."""
    u = np.minimum(xv, _X_MAX) * s / (1.0 - np.minimum(xv, _X_MAX))
    out = np.zeros_like(xv)
    for j in range(xv.shape[1]):
        # Clipped at 0 (a concentration cannot go negative), so the denominator is
        # the step actually taken, not the nominal 2 * delta * s.
        up = u[:, j] + delta * s[j]
        um = np.maximum(u[:, j] - delta * s[j], 0.0)
        xp, xm = xv.copy(), xv.copy()
        xp[:, j] = up / (up + s[j])
        xm[:, j] = um / (um + s[j])
        out[:, j] = (rf.predict(xp) - rf.predict(xm)) / (up - um)
    return out


def run(labels_dir: Path, index_path: Path, outdir: Path, *, eps: float = 1e-3,
        n_estimators: int = 100, delta: float = 0.05, seed: int = 0) -> dict:
    """Fit one forest per organism on the same split, score it on the same gate."""
    from sklearn.ensemble import RandomForestRegressor

    ds = load_value_dataset(labels_dir, index_path, eps=eps, seed=seed)
    n_org, n_val, n_met = ds.x_val.shape
    mu_hat = np.zeros((n_org, n_val), dtype=np.float64)
    g_hat = np.zeros((n_org, n_val, n_met), dtype=np.float64)

    t0 = time.time()
    for i, gid in enumerate(ds.genome_ids):
        dims = np.flatnonzero(ds.mask[i])
        rf = RandomForestRegressor(n_estimators=n_estimators, random_state=seed, n_jobs=-1)
        rf.fit(ds.x_train[i][:, dims], ds.mu_train[i] / ds.mu_scale[i])
        xv = ds.x_val[i][:, dims].astype(np.float64)
        mu_hat[i] = rf.predict(xv)
        g_hat[i][:, dims] = _u_space_gradients(rf, xv, ds.x_scale[i, dims].astype(np.float64),
                                               delta)
        LOGGER.info("%s (%d/%d): %d dims, %.0fs", gid, i + 1, n_org, len(dims), time.time() - t0)

    diagnostics = score(ds, mu_hat, g_hat, arch="random-forest")
    diagnostics["baseline"] = {"model": "RandomForestRegressor", "n_estimators": n_estimators,
                               "gradient": f"central finite difference, step {delta} * s_m in u",
                               "eps": eps, "seed": seed}
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2))
    return diagnostics
