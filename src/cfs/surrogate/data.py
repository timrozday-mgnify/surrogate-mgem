"""§4.5 label shards -> stacked arrays for Head A (plan §3.1, §6.1, §7.1).

Head A learns ``mu_max_i`` as a function of the **saturation**
``x_m = c_m / (Km_m + c_m) in [0, 1]``, which is the Michaelis-Menten uptake
bound ``u_m = Vmax_m * x_m`` up to a constant. That is the space §1 says the
value function is concave in -- ``log c`` is not (the MM map is a sigmoid in
``log c``, convex then concave), so an ICNN over ``log c`` would impose a prior
the target does not satisfy. Saturation is also bounded and finite at ``c = 0``,
so the §4.3 depletion corners and the padded absent metabolites are both just 0.

The saturation is then passed through a *second* MM map with a per-metabolite
constant read off the labels (:func:`_kink_scale`), which is what puts each
metabolite's ramp at O(1). That map is concave rather than affine, so the head is
constrained monotone non-decreasing to keep the composition concave — see
:mod:`cfs.surrogate.picnn`.

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
    x_scale: np.ndarray  # (G, M) per-metabolite saturation constant s; x is already x/(x+s)
    index_hash: str
    rounds_present: list[int]  # §4.6 top-up rounds in the training set; val is always round 0


@dataclass
class BehaviourDataset:
    """Stacked Head B training data (§6.3). One row is one (medium, alpha) pair."""

    genome_ids: list[str]
    exchanges: list[str]
    mask: np.ndarray  # (G, M) bool
    alphas: np.ndarray  # (A,) the §4.4 grid the labels were solved on
    x_train: np.ndarray  # (G, N*A, M) saturation, same transform as Head A
    a_train: np.ndarray  # (G, N*A) normalised growth rate
    z_train: np.ndarray  # (G, N*A, M) net exchange flux per unit biomass
    x_val: np.ndarray
    a_val: np.ndarray
    z_val: np.ndarray
    mu_train: np.ndarray  # (G, N*A) the *floored* mu_max of each row's medium
    mu_val: np.ndarray
    mu_floor: np.ndarray  # (G,) the floor itself, so the head can be composed
    z_scale: np.ndarray  # (G, M) per-metabolite flux scale; loss is dimensionless
    x_scale: np.ndarray  # (G, M) the Head A kink scale, so the heads compose
    index_hash: str
    rounds_present: list[int]


# Head B divides its target by the medium's `mu_max`; below this fraction of the
# organism's mean the division amplifies noise instead of removing a scale.
_MU_FLOOR_FRAC = 0.05


def _saturation(c: np.ndarray, km: np.ndarray) -> np.ndarray:
    return c / (km + c)


def _shard_dir(labels_dir: Path, gid: str, eps: float) -> Path:
    """``<gid>/eps_<e>``, falling back to the pre-c3c1862 hive layout.

    Label sets written before the ``key=`` prefixes were dropped -- which is every
    set on disk, including `20hm_bands` and `20hm_probe` -- use
    ``genome_id=<gid>/eps=<e>``. Regenerating them is ~1 h/organism, so read both.
    """
    d = labels_dir / gid / f"eps_{eps:g}"
    return d if d.is_dir() else labels_dir / f"genome_id={gid}" / f"eps={eps:g}"


def _organism_arrays(
    labels_dir: Path,
    gid: str,
    eps: float,
    col: dict[str, int],
    km_cfg: dict,
    n_shared: int,
    with_z: bool = False,
):
    """Read one organism's primary-eps shard into shared-index arrays.

    ``with_z`` additionally returns Head B's targets: the elastic-net exchange
    fluxes at every ``alpha`` on the §4.4 grid, shaped ``(n_alpha, n_media, M)``
    and in the same ``medium_id`` order as the value arrays, plus the alpha grid.
    """
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

    # Every parquet in the (organism, eps) directory: the base run writes
    # `part.parquet`, each §4.6 top-up round adds `part.round<n>.parquet`.
    shard_dir = _shard_dir(labels_dir, gid, eps)
    parts = sorted(shard_dir.glob("*.parquet"))
    if not parts:
        raise FileNotFoundError(f"{gid}: no label shards under {shard_dir}")
    full = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df = full[full["alpha"] == 1.0].sort_values("medium_id").reset_index(drop=True)

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
    mid = df["medium_id"].to_numpy(dtype=np.int64)

    z = alphas = None
    if with_z:
        # Keep the stored float64 values for the `==` selection; cast only on return.
        alphas = np.sort(full["alpha"].unique())
        z = np.zeros((len(alphas), n, n_shared), dtype=np.float32)
        for k, a in enumerate(alphas):
            sub = full[full["alpha"] == a].sort_values("medium_id")
            if not np.array_equal(sub["medium_id"].to_numpy(dtype=np.int64), mid):
                raise ValueError(f"{gid}: alpha={a} covers different media than alpha=1")
            z[k][:, pos] = np.stack(sub["z"].to_numpy())

    return (
        x,
        mu.astype(np.float32),
        g,
        gvalid,
        mask,
        df["index_hash"].iloc[0],
        mid,
        z,
        None if alphas is None else alphas.astype(np.float32),
    )


def _kink_scale(x: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Per-metabolite saturation constant: where the value function's ramp lives.

    Measured on a real GEM, ``mu_max`` rises linearly in ``x`` up to ``x* ~ 5e-3``
    and is flat over the remaining 99% of ``[0, 1]`` — the uptake bound saturates
    almost immediately because ``Vmax = 1000`` dwarfs the demand. A first layer
    initialised at ``1/sqrt(M)`` cannot put a kink there; it would need weights of
    order 200, and Adam does not travel that far in a training run.

    The scale is the median ``x`` over the rows where that metabolite is actually
    limiting (its dual is non-zero), so it is read off the labels rather than
    guessed. Metabolites that never limit keep scale 1 — nothing to resolve.

    It is used as the constant of a *second* MM map (see
    :func:`load_value_dataset`), not as a divisor, so there is no floor: the ion
    metabolites have their ramp at ``x ~ 1.4e-4`` and a linear rescale big enough
    to reach it sends the replete dimensions to ``x ~ 1e4``. That tension is what
    a saturating rescale removes.
    """
    scale = np.ones(x.shape[-1], dtype=np.float32)
    lim = g > 0
    for m in np.flatnonzero(lim.any(axis=0)):
        vals = x[lim[:, m], m]
        vals = vals[vals > 0]
        if len(vals):
            scale[m] = float(np.median(vals))
    return scale


def _stack(
    labels_dir: Path | str,
    index_path: Path | str,
    eps: float,
    val_frac: float,
    seed: int,
    organisms: list[str] | None,
    with_z: bool = False,
):
    """Everything both heads share: read, check (P13), split by medium, rescale.

    Returns the stacked arrays plus the train/val medium indices. Head A and Head
    B must see the *same* input transform and the same held-out media, so this
    lives in one place rather than in two loaders that can drift apart.
    """
    from cfs.groundtruth.index import index_hash, load_index
    from cfs.groundtruth.solve import load_km_defaults
    from cfs.sampling.generate import _ROUND_STRIDE

    labels_dir, index_path = Path(labels_dir), Path(index_path)
    frozen = load_index(index_path)
    exchanges = frozen.index
    col = {ex: i for i, ex in enumerate(exchanges)}
    km_cfg = load_km_defaults()

    shard_ids = {p.name.removeprefix("genome_id=") for p in labels_dir.iterdir() if p.is_dir()}
    if not shard_ids:
        raise ValueError(f"no per-genome shard dirs under {labels_dir}")
    # Stack in the frozen index's organism order, so row i of `mask` is organism i.
    gids = [g for g in frozen.genome_ids if g in shard_ids]
    if set(gids) != shard_ids:
        raise ValueError(f"shards not in the frozen index: {sorted(shard_ids - set(gids))}")
    if organisms is not None:
        missing = sorted(set(organisms) - set(gids))
        if missing:
            raise ValueError(f"requested organisms have no shards: {missing}")
        gids = [g for g in gids if g in set(organisms)]

    parts = [
        _organism_arrays(labels_dir, g, eps, col, km_cfg, len(exchanges), with_z=with_z)
        for g in gids
    ]
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

    # Round-0 media are the base design; `_organism_arrays` sorts by `medium_id`
    # and round N is offset by N * _ROUND_STRIDE, so they are the leading block and
    # their indices do not move when a top-up shard is added.
    mid = np.stack([p[6] for p in parts])
    if not (mid == mid[0]).all():
        raise ValueError("organisms have different medium_ids; cannot stack (§6.1)")
    rounds = mid[0] // _ROUND_STRIDE
    base = np.flatnonzero(rounds == 0)
    if not len(base):
        raise ValueError("no round-0 media: the held-out set is the base design only")
    perm = np.random.default_rng(seed).permutation(base)
    n_val = max(1, int(round(val_frac * len(base))))
    # Top-up media are appended to train rather than mixed into the permutation, so
    # a round-free label set reproduces the pre-fix split exactly.
    vi = perm[:n_val]
    ti = np.concatenate([perm[n_val:], np.flatnonzero(rounds > 0)])

    # Put each metabolite's ramp at O(1) with a second MM map, x' = x / (x + s_m).
    # A *linear* rescale cannot do this: the ion metabolites limit at x ~ 1.4e-4,
    # and dividing by that sends every replete dimension to x ~ 1e4. The
    # saturating map takes the median limiting entry to exactly x' = 0.5 and the
    # replete ones to ~1, and it shrinks the interquartile spread of ||dmu/dx||^2
    # from 3.5 decades to 1.6. Composed with `_saturation` it is just a smaller
    # effective Km (Km' ~ s * Km) — the nutrient's own demand, not the literature
    # constant, sets where it saturates.
    #
    # Unlike the linear rescale this is *concave*, not affine, so the head must be
    # non-decreasing for the composition to stay concave. `picnn.ValueHead`
    # enforces that, and it is true of the target anyway: relaxing an uptake bound
    # can only enlarge the LP's feasible set.
    x_scale = np.stack([_kink_scale(x[i], g[i]) for i in range(x.shape[0])])
    s = x_scale[:, None, :]
    # Chain rule: dx'/dx = s / (x + s)^2. A few depletion-corner rows (x = 0 with
    # a huge dual on a metabolite that never limits elsewhere, so s = 1) stay
    # extreme; the per-row normalisation in the Sobolev loss is what handles them.
    g = g * (x + s) ** 2 / s
    x = x / (x + s)

    present = sorted(int(r) for r in np.unique(rounds))
    LOGGER.info(
        "%d organisms x %d media (%d train / %d held out, base design only), "
        "rounds %s, %d shared exchanges, |M_i| %d-%d",
        len(gids),
        len(rounds),
        len(ti),
        len(vi),
        present,
        len(exchanges),
        mask.sum(1).min(),
        mask.sum(1).max(),
    )
    return {
        "gids": gids,
        "exchanges": exchanges,
        "mask": mask,
        "x": x,
        "mu": mu,
        "g": g,
        "gvalid": gvalid,
        "z": np.stack([p[7] for p in parts]) if with_z else None,  # (G, A, N, M)
        "alphas": parts[0][8],
        "x_scale": x_scale,
        "ti": ti,
        "vi": vi,
        "index_hash": hashes.pop(),
        "rounds_present": present,
    }


def load_value_dataset(
    labels_dir: Path | str,
    index_path: Path | str,
    eps: float = 1e-3,
    val_frac: float = 0.2,
    seed: int = 0,
    organisms: list[str] | None = None,
) -> ValueDataset:
    """Load the labels of ``organisms`` (default: all) into stacked train/val arrays.

    Any number of organisms stacks, one included: the stack is a vmap axis, not a
    modelling choice, and only the shared-trunk ``deepset`` pools anything across
    it. Stacking one organism per job is what the sweep does — the split is by
    ``medium_id`` and the media are identical across organisms, so the held-out
    set of a one-organism stack is the same media as the full stack's.

    The split is by ``medium_id``, never by row: media are the independent unit.
    Every shard must carry the same ``index_hash`` (P13) or this raises.

    **The held-out set is drawn from the base design only** — round-0 media, never
    a §4.6 top-up round. Top-up rounds deliberately sample where the model is
    worst, so letting them into the validation set makes the *test* harder every
    round and the gate stops being comparable to its own previous value: the
    probe-band runs went 491 -> 595 -> 781 usable held-out rows over two rounds,
    and the cosine they reported fell accordingly. Permuting the round-0 media
    alone keeps the held-out set byte-identical whether the run has 0 rounds or 3,
    which is what makes a round-over-round number mean anything. Top-up media all
    go to training, which is what they were generated for.
    """
    d = _stack(labels_dir, index_path, eps, val_frac, seed, organisms)
    ti, vi = d["ti"], d["vi"]
    mu_scale = d["mu"].std(axis=1)
    mu_scale[mu_scale <= 0] = 1.0
    return ValueDataset(
        genome_ids=d["gids"],
        exchanges=d["exchanges"],
        mask=d["mask"],
        x_train=d["x"][:, ti],
        mu_train=d["mu"][:, ti],
        g_train=d["g"][:, ti],
        gvalid_train=d["gvalid"][:, ti],
        x_val=d["x"][:, vi],
        mu_val=d["mu"][:, vi],
        g_val=d["g"][:, vi],
        gvalid_val=d["gvalid"][:, vi],
        mu_scale=mu_scale.astype(np.float32),
        x_scale=d["x_scale"],
        index_hash=d["index_hash"],
        rounds_present=d["rounds_present"],
    )


def load_behaviour_dataset(
    labels_dir: Path | str,
    index_path: Path | str,
    eps: float = 1e-3,
    val_frac: float = 0.2,
    seed: int = 0,
    organisms: list[str] | None = None,
) -> BehaviourDataset:
    """Head B labels: ``z_i(c, alpha)`` over the §4.4 alpha grid.

    Same media split and same input transform as :func:`load_value_dataset` — the
    two heads are evaluated at one medium in §8, so a Head B trained on a different
    held-out set or a different ``x_scale`` cannot be composed with a Head A.

    The (medium, alpha) pairs are flattened into one row axis. Alpha is a *model
    input*, not a batch axis: §8.2 evaluates ``z`` at an ``alpha`` off the grid.
    """
    d = _stack(labels_dir, index_path, eps, val_frac, seed, organisms, with_z=True)
    ti, vi, a = d["ti"], d["vi"], d["alphas"]
    x, z = d["x"], d["z"]  # (G, N, M), (G, A, N, M)
    g, n_a, _, m = z.shape

    # Exchange flux is very nearly proportional to how fast the organism is
    # growing, and Head A already predicts that accurately. Measured on these
    # labels: one constant per (metabolite, alpha) times `mu_max` explains a
    # median 0.807 of the held-out z variance (0.63-0.86 over the 21 organisms),
    # against the trained head's 0.921 -- i.e. most of what Head B currently
    # learns is a magnitude Head A knows. Leaving it in the net is what produced
    # M5's failing cells: at a scarce community medium the head predicted a
    # replete organism's fluxes (|z| 2780 against a true 1040, cosine 0.40, on an
    # organism whose held-out p05 is 0.93). So the head predicts *specific* flux
    # z / mu_max and the magnitude is put back at composition time.
    #
    # Floored, because 1% of media have mu_max below 1% of the organism's median
    # and dividing by those turns the target into noise -- the same trade
    # `calibrate._W_FLOOR` makes.
    mu_floor = _MU_FLOOR_FRAC * d["mu"].mean(axis=1)
    mus = np.maximum(d["mu"], mu_floor[:, None])  # (G, N)

    def flat(idx):
        # (G, A, |idx|, M) -> (G, A*|idx|, M); alpha and mu broadcast to match.
        xa = np.broadcast_to(x[:, None, idx, :], (g, n_a, len(idx), m))
        aa = np.broadcast_to(a[None, :, None], (g, n_a, len(idx)))
        mm = np.broadcast_to(mus[:, None, idx], (g, n_a, len(idx)))
        return (
            xa.reshape(g, -1, m),
            aa.reshape(g, -1).astype(np.float32),
            z[:, :, idx, :].reshape(g, -1, m),
            mm.reshape(g, -1).astype(np.float32),
        )

    xt, at, zt, mt = flat(ti)
    xv, av, zv, mv = flat(vi)
    # Per-(organism, metabolite) scale so the loss is dimensionless and one
    # high-flux exchange (usually O2 or CO2, O(10) against an ion's O(1e-3)) does
    # not own the objective. Robust to the ~63% exact zeros: std over the rows
    # where the organism can exchange it at all, floored. Taken on the *specific*
    # flux, which is what the head emits.
    z_scale = (zt / mt[:, :, None]).std(axis=1)
    z_scale[z_scale <= 0] = 1.0
    LOGGER.info(
        "Head B: %d alphas %s, %d train / %d held-out rows per organism",
        n_a,
        np.round(a, 3).tolist(),
        xt.shape[1],
        xv.shape[1],
    )
    return BehaviourDataset(
        genome_ids=d["gids"],
        exchanges=d["exchanges"],
        mask=d["mask"],
        alphas=a,
        x_train=xt,
        a_train=at,
        z_train=zt,
        x_val=xv,
        a_val=av,
        z_val=zv,
        mu_train=mt,
        mu_val=mv,
        mu_floor=mu_floor.astype(np.float32),
        z_scale=z_scale.astype(np.float32),
        x_scale=d["x_scale"],
        index_hash=d["index_hash"],
        rounds_present=d["rounds_present"],
    )
