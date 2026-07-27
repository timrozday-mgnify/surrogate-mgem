"""§4.5 label shards -> stacked arrays for Head A (plan §3.1, §6.1, §7.1).

Head A learns ``mu_max_i`` as a function of the **saturation**
``x_m = c_m / (Km_m + c_m) in [0, 1]``, which is the Michaelis-Menten uptake
bound ``u_m = Vmax_m * x_m`` up to a constant. That is the space §1 says the
value function is concave in -- ``log c`` is not (the MM map is a sigmoid in
``log c``, convex then concave), so an ICNN over ``log c`` would impose a prior
the target does not satisfy. Saturation is also bounded and finite at ``c = 0``,
so the §4.3 depletion corners and the padded absent metabolites are both just 0.

Gradient targets come straight from the stored duals. ``check_v2.py`` pinned the
sign convention: the stored ``shadow`` is the *negated* derivative w.r.t. supply,
so ``dmu/du_m = -pi_m`` and ``dmu/dx_m = -pi_m * Vmax_m``.

Only the primary ``eps`` shard at ``alpha == 1.0`` is used: ``mu_max`` does not
depend on ``alpha`` (the other 7 rows per medium are Head B's), so 8x the rows
would be 8 copies of the same label.

Needs ``pyarrow`` (parquet). No JAX, no cobra -- ``km_for_exchange`` is pure.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("cfs.surrogate.data")

# CarveMe sets every exchange lower bound to -1000; verified across all 21 roster
# models. Vmax only scales the gradient targets, so a per-exchange override lives
# in the `<id>.exchanges.json` sidecar if a future roster breaks the assumption.
_VMAX_DEFAULT = 1000.0
_KINK_FLOOR = 0.05  # cap the rescale: unbounded, the replete dims reach x ~ 1e4
_DUAL_TOL = 1e-9  # below this a dual is solver dust, not a sensitivity


@dataclass
class ValueDataset:
    """Stacked Head A training data. Leading axis is the organism (§6.1)."""

    genome_ids: list[str]
    exchanges: list[str]  # the frozen shared index (§2.2); x/g/mask columns
    mask: np.ndarray  # (G, M) bool -- organism i can exchange metabolite m
    x_train: np.ndarray  # (G, N, M) saturation c/(Km+c)
    mu_train: np.ndarray  # (G, N) mu_max, unscaled
    g_train: np.ndarray  # (G, N, M) dmu/dx target, 0 where invalid
    gvalid_train: np.ndarray  # (G, N) bool -- row's duals are usable
    x_val: np.ndarray
    mu_val: np.ndarray
    g_val: np.ndarray
    gvalid_val: np.ndarray
    mu_scale: np.ndarray  # (G,) per-organism label std; loss is dimensionless
    x_scale: np.ndarray  # (G, M) per-metabolite kink scale; x is already divided by it
    index_hash: str


def _saturation(c: np.ndarray, km: np.ndarray) -> np.ndarray:
    return c / (km + c)


def _organism_arrays(labels_dir: Path, gid: str, eps: float, col: dict[str, int],
                     km_cfg: dict, n_shared: int):
    """Read one organism's primary-eps shard into shared-index arrays."""
    from cfs.groundtruth.solve import km_for_exchange

    side = json.loads((labels_dir / f"{gid}.exchanges.json").read_text())
    ex_order = side["exchanges"]
    vmax_side = side.get("vmax", {})
    # Column positions in the shared index; -1 for an exchange the frozen index
    # does not know about (should not happen -- the index is their union).
    pos = np.array([col.get(ex, -1) for ex in ex_order])
    if (pos < 0).any():
        missing = [ex for ex, p in zip(ex_order, pos, strict=True) if p < 0]
        raise ValueError(f"{gid}: exchanges absent from the frozen index: {missing[:5]}")

    df = pd.read_parquet(labels_dir / f"genome_id={gid}" / f"eps={eps:g}" / "part.parquet")
    df = df[df["alpha"] == 1.0].sort_values("medium_id").reset_index(drop=True)

    km = np.array([km_for_exchange(ex, km_cfg) for ex in ex_order])
    vmax = np.array([float(vmax_side.get(ex, _VMAX_DEFAULT)) for ex in ex_order])

    c = np.stack(df["medium"].to_numpy())  # (N, M_i)
    pi = np.stack(df["shadow"].to_numpy())
    mu = df["mu_max"].to_numpy(dtype=np.float64)
    # Zero-growth / non-optimal rows carry garbage duals (pi = -1e4 on the
    # depletion corners). Keep them for the value term, drop them from the
    # Sobolev term (P2).
    gvalid = (mu > 0.0) & (df["status"].to_numpy() == "optimal")

    n = len(df)
    x = np.zeros((n, n_shared), dtype=np.float32)
    g = np.zeros((n, n_shared), dtype=np.float32)
    x[:, pos] = _saturation(c, km)
    # `shadow` is the dual of the metabolite's mass balance, which equals
    # d(mu_max)/d(uptake bound) only where that bound is *binding*. Where it is
    # not, the dual is the metabolite's value in the network -- for a waste
    # product like CO2 that is positive, implying growth falls when you allow
    # more uptake, which is impossible: relaxing a bound can only enlarge the
    # feasible set. Finite differences confirm it (12/12 positive-dual cases on
    # CP009913.1 gave d(mu_max)/d(supply) = 0.000000 exactly). So clamp at 0:
    # non-binding means no sensitivity. Left unclamped this is ~12% of the
    # non-zero duals, on metabolites that appear in half the media.
    #
    # The same clamp also drops solver dust: half the "non-zero" duals are
    # O(1e-14), and a row whose whole target is dust would otherwise dominate a
    # norm-relative Sobolev loss by ~1e14.
    g[:, pos] = np.where(pi < -_DUAL_TOL, -pi * vmax, 0.0)
    g[~gvalid] = 0.0

    mask = np.zeros(n_shared, dtype=bool)
    mask[pos] = True
    return x, mu.astype(np.float32), g, gvalid, mask, df["index_hash"].iloc[0]


def _kink_scale(x: np.ndarray, g: np.ndarray, floor: float = _KINK_FLOOR) -> np.ndarray:
    """Per-metabolite input scale: where the value function's ramp actually lives.

    Measured on a real GEM, ``mu_max`` rises linearly in ``x`` up to ``x* ~ 5e-3``
    and is flat over the remaining 99% of ``[0, 1]`` — the uptake bound saturates
    almost immediately because ``Vmax = 1000`` dwarfs the demand. A first layer
    initialised at ``1/sqrt(M)`` cannot put a kink there; it would need weights of
    order 200, and Adam does not travel that far in a training run. Dividing each
    coordinate by its own ramp scale moves the kink to O(1).

    The scale is the median ``x`` over the rows where that metabolite is actually
    limiting (its dual is non-zero), so it is read off the labels rather than
    guessed. Metabolites that never limit keep scale 1 — nothing to resolve.
    """
    scale = np.ones(x.shape[-1], dtype=np.float32)
    lim = g > 0
    for m in np.flatnonzero(lim.any(axis=0)):
        vals = x[lim[:, m], m]
        vals = vals[vals > 0]
        if len(vals):
            scale[m] = max(float(np.median(vals)), floor)
    return scale


def load_value_dataset(labels_dir: Path | str, index_path: Path | str,
                       eps: float = 1e-3, val_frac: float = 0.2,
                       seed: int = 0) -> ValueDataset:
    """Load every organism's primary-eps labels into stacked train/val arrays.

    The split is by ``medium_id``, never by row: media are the independent unit.
    Every shard must carry the same ``index_hash`` (P13) or this raises.
    """
    from cfs.groundtruth.index import index_hash, load_index
    from cfs.groundtruth.solve import load_km_defaults

    labels_dir, index_path = Path(labels_dir), Path(index_path)
    frozen = load_index(index_path)
    exchanges = frozen.index
    col = {ex: i for i, ex in enumerate(exchanges)}
    km_cfg = load_km_defaults()

    shard_ids = {p.name.split("=", 1)[1] for p in labels_dir.glob("genome_id=*")}
    if not shard_ids:
        raise ValueError(f"no genome_id=* shards under {labels_dir}")
    # Stack in the frozen index's organism order, so row i of `mask` is organism i.
    gids = [g for g in frozen.genome_ids if g in shard_ids]
    if set(gids) != shard_ids:
        raise ValueError(f"shards not in the frozen index: {sorted(shard_ids - set(gids))}")

    parts = [_organism_arrays(labels_dir, g, eps, col, km_cfg, len(exchanges)) for g in gids]
    hashes = {p[5] for p in parts} | {index_hash(index_path)}
    if len(hashes) != 1:
        raise ValueError(f"labels and index disagree on index_hash: {hashes} (P13)")
    counts = {len(p[1]) for p in parts}
    if len(counts) != 1:
        raise ValueError(f"organisms have different media counts {counts}; cannot stack (§6.1)")

    x = np.stack([p[0] for p in parts])
    mu = np.stack([p[1] for p in parts])
    g = np.stack([p[2] for p in parts])
    gvalid = np.stack([p[3] for p in parts])
    mask = np.stack([p[4] for p in parts])
    frozen_mask = frozen.mask[[frozen.genome_ids.index(g) for g in gids]]
    if not (mask == frozen_mask).all():
        raise ValueError("sidecar exchange lists disagree with the frozen index mask (P13)")

    n = x.shape[1]
    perm = np.random.default_rng(seed).permutation(n)
    n_val = max(1, int(round(val_frac * n)))
    vi, ti = perm[:n_val], perm[n_val:]

    # Rescale the inputs to their kink scale, and the gradient targets by the
    # same factor (chain rule for a diagonal linear map). Concavity is preserved:
    # a positive diagonal rescale is affine.
    x_scale = np.stack([_kink_scale(x[i], g[i]) for i in range(x.shape[0])])
    x = x / x_scale[:, None, :]
    g = g * x_scale[:, None, :]

    mu_scale = mu.std(axis=1)
    mu_scale[mu_scale <= 0] = 1.0
    LOGGER.info("%d organisms x %d media, %d shared exchanges, |M_i| %d-%d",
                len(gids), n, len(exchanges), mask.sum(1).min(), mask.sum(1).max())
    return ValueDataset(
        genome_ids=gids, exchanges=exchanges, mask=mask,
        x_train=x[:, ti], mu_train=mu[:, ti], g_train=g[:, ti], gvalid_train=gvalid[:, ti],
        x_val=x[:, vi], mu_val=mu[:, vi], g_val=g[:, vi], gvalid_val=gvalid[:, vi],
        mu_scale=mu_scale.astype(np.float32), x_scale=x_scale, index_hash=hashes.pop(),
    )
