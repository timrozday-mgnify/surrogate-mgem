"""M4 / §6.3 Head B — the behaviour map ``z_i(c, alpha)``.

Head A says how fast an organism *can* grow. Head B says what it does to the
medium while doing it, and it is the half §8 composes: the dFBA right-hand side
is ``dc/dt = sum_i X_i z_i(c, alpha_i) + inflow(c)``, so every cross-feeding
interaction in a community is a Head B output of one organism landing in another
organism's Head A input.

Unlike Head A this head carries **no structural constraint** (§6.3): ``z`` is a
signed vector (negative = uptake, positive = secretion) and nothing about the LP
makes it convex, monotone or even continuous in general — D4's elastic net is what
makes it continuous, which is the whole reason §5.4 exists. So this is a plain
softplus MLP over ``(x, alpha)``, masked to ``M_i`` on both ends.

**The head predicts specific flux ``z / mu_max``, not flux.** Exchange flux is
nearly proportional to growth rate, Head A predicts growth rate accurately, and
leaving that factor inside Head B is what broke M5's small communities: at a
scarce community medium the head produced a replete organism's fluxes (|z| 2780
against a true 1040) while Head A had ``mu`` right to 1%. Measured on the labels,
one constant per (metabolite, alpha) times ``mu_max`` already explains a median
0.807 of the held-out ``z`` variance. See :func:`cfs.surrogate.data.load_behaviour_dataset`.

Two deliberate simplifications:

* §6.3 suggests predicting non-negative uptake and secretion heads and returning
  the difference. A difference of two non-negative outputs is any real number, so
  it constrains nothing — it is presentational. One signed output instead.
* The loss is scaled per ``(organism, metabolite)``, not per row. Exchange fluxes
  span O(1e-3) (ions) to O(10) (O2/CO2) *within* one organism, and an unscaled
  MSE simply fits the gas exchanges. Head A's per-row norm-relative trick is not
  needed here because there is no all-zero-target row problem: ``z`` is nonzero
  on ~37% of entries but essentially never all-zero.

Trained by ``cfs train-behaviour``; consumed by :mod:`cfs.compose.dfba`.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import Array

from cfs.surrogate.data import BehaviourDataset, load_behaviour_dataset

# Every exchange of every CarveMe GEM on the roster has |lower_bound| = 1000, which
# is what §3.3 scales by saturation to make the uptake bound. `compose.dfba` reads
# it from here so the training bound and the inference projection cannot drift.
VMAX = 1000.0

LOGGER = logging.getLogger("cfs.surrogate.behaviour")


class BehaviourHead(eqx.Module):
    """``(x, alpha) -> z`` in the shared index, zero outside ``M_i``."""

    w: list[Array]
    b: list[Array]
    mask: Array
    basis: Array | None

    def __init__(self, key, n_in: int, mask, width: int = 256, depth: int = 3, basis=None):
        keys = jax.random.split(key, depth + 1)
        # +1 input for alpha; the output is the full shared index, masked -- or,
        # with a basis, its `r` coordinates in the label flux subspace (§8.6f).
        n_out = n_in if basis is None else basis.shape[0]
        sizes = [n_in + 1] + [width] * depth + [n_out]
        self.w = [
            jax.random.normal(k, (o, i)) * jnp.sqrt(2.0 / i)
            for k, i, o in zip(keys, sizes[:-1], sizes[1:], strict=True)
        ]
        self.b = [jnp.zeros(o) for o in sizes[1:]]
        self.mask = jnp.asarray(mask, dtype=bool)
        self.basis = None if basis is None else jnp.asarray(basis, dtype=jnp.float32)

    def __call__(self, x: Array, alpha: Array) -> Array:
        """Returns ``z / (mu_max * z_scale)`` -- **specific** flux, not mmol/gDW/h.

        The head works in units of each metabolite's own flux std. Raw ``z`` runs
        to O(400) on the gas exchanges and O(1e-3) on the ions, and a net
        initialised at ``sqrt(2/n_in)`` starts ~400x short on the ones that carry
        the variance; it then spends the run walking biases, which is the same
        failure `picnn_u`'s scale-aware init exists to avoid. And it is *per unit
        growth*: exchange flux is nearly proportional to ``mu_max``, which Head A
        already predicts, so leaving that factor in the net makes it guess the
        medium's richness -- see :func:`cfs.surrogate.data.load_behaviour_dataset`.
        Multiply by ``mu_max * z_scale`` to get fluxes -- :func:`flux`.
        """
        h = jnp.concatenate([x * self.mask, jnp.atleast_1d(alpha)])
        for w, b in zip(self.w[:-1], self.b[:-1], strict=True):
            h = jax.nn.softplus(w @ h + b)
        out = self.w[-1] @ h + self.b[-1]
        return (out if self.basis is None else out @ self.basis) * self.mask


def stack_heads(
    key, n_organisms: int, n_in: int, mask, width: int = 256, depth: int = 3, basis=None
):
    keys = jax.random.split(key, n_organisms)
    if basis is None:
        make = eqx.filter_vmap(lambda k, m: BehaviourHead(k, n_in, m, width, depth), in_axes=(0, 0))
        return make(keys, jnp.asarray(mask, dtype=bool))
    make = eqx.filter_vmap(
        lambda k, m, v: BehaviourHead(k, n_in, m, width, depth, v), in_axes=(0, 0, 0)
    )
    return make(keys, jnp.asarray(mask, dtype=bool), jnp.asarray(basis, dtype=jnp.float32))


@eqx.filter_vmap(in_axes=(0, 0, 0))
def batched_z(heads: BehaviourHead, x: Array, alpha: Array) -> Array:
    return jax.vmap(heads)(x, alpha)


def flux(heads, x: Array, alpha: Array, z_scale: Array, mu: Array | None = None) -> Array:
    """``z`` in mmol/gDW/h: specific flux times ``mu_max`` times the label scale.

    ``mu`` is ``(G, B)``, already floored at ``mu_floor``. ``None`` reads a
    pre-2026-08-30 checkpoint, whose head emits flux directly.
    """
    z = batched_z(heads, x, alpha) * z_scale[:, None, :]
    return z if mu is None else z * mu[:, :, None]


def organism(heads: BehaviourHead, i: int) -> BehaviourHead:
    """Slice organism ``i`` out of the stack (all leaves carry the organism axis)."""
    return jax.tree.map(lambda a: a[i] if eqx.is_array(a) else a, heads)


def flux_basis(ds: BehaviourDataset, var: float = 0.9999) -> np.ndarray:
    """Per-organism basis of the label flux set, in the head's own output units.

    §8.6f: the LP's optimum is a vertex, so ``c -> z`` is piecewise affine over
    critical regions and every feasible flux is a conical combination of the flux
    cone's generators. Measured on these labels, the training specific-flux matrix
    has effective rank **3-12 at 99%** and **12-39 at 99.99%** of variance out of
    138-259 exchanges, and that basis reconstructs *held-out* truth to a median
    relative error of 0.001-0.005 -- 30-100x below the trained head's own 0.09-0.26.
    So a head emitting ~200 free fluxes is spending its output layer on directions
    the LP cannot produce, and its off-manifold component grows 2.5x at the dFBA
    states §8.1 actually visits.

    Every *conservation* relation (elemental balance, a conserved moiety) is by
    construction a direction of zero variance here, so this collects all of them
    without reading a single metabolite formula.

    **Measured and refuted as an M5 lever; default off** (``--basis-var 0``). Rank
    12-39 at the default cutoff, held-out worst R2 0.9354 -> 0.9199, and the
    composition is identical to three decimals at every community size over 3
    medium draws x 10 communities (overall 0.002, max 0.318 in both arms). It
    trades as the restriction predicts and the trade nets to nothing: on the 15
    easy cells ``dc_rel`` 0.079 -> 0.147 (worse on 13), on the 10 hard ones
    0.920 -> 0.840 (better on 5). Kept because the negative result is worth being
    able to re-derive, and because the basis itself is a useful object -- it
    reconstructs held-out truth to 0.001-0.005. See design spec §8.6f.

    **The SVD is taken on the raw specific flux, not on the head's ``z_scale``d
    coordinate.** ``z_scale`` divides each metabolite by its own std, which
    equalises the ions with the gases and inflates the rank straight back: 59-101
    against 12-39 on the same labels. The compression is a property of the flux
    space the *composition* consumes -- ``dc = sum_i X_i z_i`` is in mmol/gDW/h --
    so the basis is built there and mapped into the head's units afterwards.

    Rows are then scaled by the training coefficients' own std, so the head emits
    O(1) numbers -- the same reason ``z_scale`` exists, one coordinate change
    later. Organisms are padded to the roster's largest rank so the stack vmaps;
    the padding rows are real singular directions, merely unnecessary ones.
    """
    ranks, vts = [], []
    for i in range(len(ds.genome_ids)):
        zs = ds.z_train[i] / ds.mu_train[i][:, None] * ds.mask[i]
        _, sv, vt = np.linalg.svd(zs, full_matrices=False)
        e = sv**2
        ranks.append(int(np.searchsorted(np.cumsum(e) / max(e.sum(), 1e-30), var) + 1))
        vts.append(vt)
    r = max(ranks)
    out = np.zeros((len(ds.genome_ids), r, ds.mask.shape[1]), dtype=np.float32)
    for i, vt in enumerate(vts):
        # Into the head's output coordinate: it emits z / (mu * z_scale). The rows
        # stop being orthonormal, which does not matter -- only the span does.
        v = vt[:r] / ds.z_scale[i]
        zn = ds.z_train[i] / ds.mu_train[i][:, None] / ds.z_scale[i] * ds.mask[i]
        coeff = zn @ np.linalg.pinv(v)
        out[i] = v * np.maximum(coeff.std(axis=0), 1e-12)[:, None]
    LOGGER.info("Head B basis: rank %d (per organism %s at var=%.4g)", r, ranks, var)
    return out


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #


def mm_floor(x, mu, x_scale, z_scale):
    """§3.3's uptake bound in the head's own *normalised* output units.

    The LP that made the labels cannot take up faster than ``-Vmax_m * u_m``, and
    every exchange of every roster GEM has ``|lower_bound| = 1000``, so the bound
    is a constant times the head's own input saturation -- no fit, nothing stored.
    The head emits specific flux ``z / (mu * z_scale)``, hence the divisor.

    ``u`` is recovered from the input coordinate exactly: ``x = u/(u+s)``.
    """
    u = x / jnp.maximum(1.0 - x, 1e-12) * x_scale[:, None, :]
    return -VMAX * u / (mu[:, :, None] * z_scale[:, None, :])


def _loss(heads, x, a, zn, lo, w_mm: float):
    """MSE against the *normalised* target, over the organism's own exchanges.

    ``w_mm`` adds a one-sided hinge on :func:`mm_floor`. Every label satisfies the
    bound, so a prediction below it is a **provable violation**, not an accuracy
    trade -- the same shape as Head A's ``--w-under``, and for the same reason:
    pay at the violation and stay exactly silent where there is nothing to fix.
    Measured: the inference-time projection's bite ``|dz|/|z|`` ranks the 10 §8.1
    communities by their trajectory error (0.000 at the best cell, 0.41 and 0.28
    at the two worst), with individual predictions 13x outside the bound.
    """
    w = heads.mask[:, None, :]  # (G, 1, M)
    z = batched_z(heads, x, a)
    n = jnp.maximum(jnp.sum(w) * x.shape[1], 1.0)
    loss = jnp.sum(w * (z - zn) ** 2) / n
    if w_mm:
        loss = loss + w_mm * jnp.sum(w * jax.nn.relu(lo - z) ** 2) / n
    return loss


@eqx.filter_jit
def _step(heads, opt_state, x, a, zn, lo, optimiser, w_mm):
    loss, grads = eqx.filter_value_and_grad(_loss)(heads, x, a, zn, lo, w_mm)
    updates, opt_state = optimiser.update(grads, opt_state, eqx.filter(heads, eqx.is_inexact_array))
    return eqx.apply_updates(heads, updates), opt_state, loss


def train_behaviour_heads(
    ds: BehaviourDataset,
    *,
    width: int = 256,
    depth: int = 3,
    epochs: int = 300,
    batch: int = 512,
    lr: float = 3e-3,
    w_mm: float = 0.0,
    basis_var: float = 0.0,
    seed: int = 0,
) -> eqx.Module:
    x = jnp.asarray(ds.x_train)
    a = jnp.asarray(ds.a_train)
    zn = jnp.asarray(ds.z_train / ds.mu_train[:, :, None] / ds.z_scale[:, None, :])
    # Per batch, not precomputed: the full (G, N, M) bound is ~1.2 GB in float32.
    mu_fl = jnp.asarray(ds.mu_train)  # already floored by `data`
    x_scale, z_scale = jnp.asarray(ds.x_scale), jnp.asarray(ds.z_scale)
    basis = flux_basis(ds, basis_var) if basis_var > 0 else None
    heads = stack_heads(
        jax.random.PRNGKey(seed), len(ds.genome_ids), x.shape[-1], ds.mask, width, depth, basis
    )
    n = x.shape[1]
    steps = max(1, n // batch)
    optimiser = optax.adam(optax.cosine_decay_schedule(lr, epochs * steps))
    opt_state = optimiser.init(eqx.filter(heads, eqx.is_inexact_array))
    rng = np.random.default_rng(seed)
    t0 = time.time()
    for epoch in range(epochs):
        perm = rng.permutation(n)
        for s in range(steps):
            idx = jnp.asarray(perm[s * batch : (s + 1) * batch])
            xb = x[:, idx]
            lo = mm_floor(xb, mu_fl[:, idx], x_scale, z_scale) if w_mm else 0.0
            heads, opt_state, loss = _step(
                heads, opt_state, xb, a[:, idx], zn[:, idx], lo, optimiser, w_mm
            )
        if epoch % 20 == 0 or epoch == epochs - 1:
            LOGGER.info("epoch %4d  loss=%.5f  (%.0fs)", epoch, float(loss), time.time() - t0)
    return heads


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #


def _predict(heads, x, a, z_scale, mu, size: int = 512) -> np.ndarray:
    """Held-out fluxes in label units, in slices (the val set is A x media rows).

    ``mu`` is the label's own ``mu_max``: this scores Head B alone, not the pair.
    """
    sc = jnp.asarray(z_scale)
    out = [
        np.asarray(
            flux(
                heads,
                jnp.asarray(x[:, i : i + size]),
                jnp.asarray(a[:, i : i + size]),
                sc,
                jnp.asarray(mu[:, i : i + size]),
            )
        )
        for i in range(0, x.shape[1], size)
    ]
    return np.concatenate(out, axis=1)


def evaluate(heads, ds: BehaviourDataset) -> dict:
    """Held-out flux accuracy, per organism and per (organism, metabolite).

    Three numbers, because they fail independently:

    * ``r2`` — how much of the flux variance is explained. The composition budget.
    * ``cosine`` — the *pattern* of exchange in one medium. What decides which
      metabolite crosses from one organism to another; a model can score well on
      R2 while getting a small cross-fed flux's sign wrong.
    * ``sign_agreement`` on the entries where the true flux is non-trivial. A sign
      error is a cross-feeding link pointing backwards, which §8's Newton will
      happily converge to.
    """
    z_hat = _predict(heads, ds.x_val, ds.a_val, ds.z_scale, ds.mu_val)
    z = ds.z_val
    out = {"genome_ids": ds.genome_ids, "per_organism": {}}
    for i, gid in enumerate(ds.genome_ids):
        m = ds.mask[i]
        t, p = z[i][:, m], z_hat[i][:, m]
        ss_res = float(((t - p) ** 2).sum())
        ss_tot = float(((t - t.mean(0)) ** 2).sum())
        num = (t * p).sum(1)
        den = np.linalg.norm(t, axis=1) * np.linalg.norm(p, axis=1)
        ok = den > 0
        big = np.abs(t) > 1e-6
        out["per_organism"][gid] = {
            "r2": 1.0 - ss_res / max(ss_tot, 1e-30),
            "cosine_median": float(np.median(num[ok] / den[ok])),
            "cosine_p05": float(np.percentile(num[ok] / den[ok], 5)),
            "sign_agreement": float((np.sign(t[big]) == np.sign(p[big])).mean()),
            "n_exchanges": int(m.sum()),
        }
    r2 = [v["r2"] for v in out["per_organism"].values()]
    cos = [v["cosine_median"] for v in out["per_organism"].values()]
    sgn = [v["sign_agreement"] for v in out["per_organism"].values()]
    out["summary"] = {
        "worst_r2": min(r2),
        "median_r2": float(np.median(r2)),
        "worst_cosine_median": min(cos),
        "median_cosine": float(np.median(cos)),
        "worst_sign_agreement": min(sgn),
        "n_val_rows": int(ds.x_val.shape[1]),
        "index_hash": ds.index_hash,
    }
    return out


# --------------------------------------------------------------------------- #
# Checkpoint (P13 / P14)
# --------------------------------------------------------------------------- #


def save(heads, ds: BehaviourDataset, outdir: Path, arch: dict, diagnostics: dict) -> None:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(outdir / "behaviour_heads.eqx", heads)
    # §8.6g(1): the reach proxy (NN distance in `x` to this organism's own training
    # media) is the one measured per-cell predictor of `dc_rel` (Spearman +0.673),
    # and `cfs simulate` has nothing but the checkpoint to compute it from. Strided
    # rather than deduplicated: a repeated row cannot change a nearest neighbour.
    stride = max(1, ds.x_train.shape[1] // 512)
    np.savez_compressed(
        outdir / "reference_x.npz", x=ds.x_train[:, ::stride][:, :512].astype(np.float16)
    )
    (outdir / "behaviour_heads.json").write_text(
        json.dumps(
            {
                "index_hash": ds.index_hash,
                "genome_ids": ds.genome_ids,
                "exchanges": ds.exchanges,
                "mask": ds.mask.astype(int).tolist(),
                "x_scale": ds.x_scale.tolist(),
                "z_scale": ds.z_scale.tolist(),
                "mu_floor": ds.mu_floor.tolist(),
                "alphas": ds.alphas.tolist(),
                "input_transform": (
                    "u = c / (Km + c), x = u / (u + x_scale); Km from km_defaults.yaml"
                ),
                "output_units": (
                    "specific flux: multiply by z_scale and by max(mu_max, mu_floor) "
                    "for mmol / gDW / h per unit biomass; negative = uptake"
                ),
                "arch": arch,
            },
            indent=2,
        )
    )
    (outdir / "behaviour_diagnostics.json").write_text(json.dumps(diagnostics, indent=2))


def load(outdir: Path) -> tuple[eqx.Module, dict]:
    outdir = Path(outdir)
    meta = json.loads((outdir / "behaviour_heads.json").read_text())
    arch = meta.get("arch", {})
    rank = arch.get("basis_rank")
    like = stack_heads(
        jax.random.PRNGKey(0),
        len(meta["genome_ids"]),
        len(meta["exchanges"]),
        np.array(meta["mask"], dtype=bool),
        arch.get("width", 256),
        arch.get("depth", 3),
        # Shape only -- `tree_deserialise_leaves` overwrites it, as it does `mask`.
        None
        if not rank
        else np.zeros((len(meta["genome_ids"]), rank, len(meta["exchanges"])), dtype=np.float32),
    )
    return eqx.tree_deserialise_leaves(outdir / "behaviour_heads.eqx", like), meta


def run(
    labels_dir: Path,
    index_path: Path,
    outdir: Path,
    *,
    eps: float = 1e-3,
    width: int = 256,
    depth: int = 3,
    epochs: int = 300,
    batch: int = 512,
    lr: float = 3e-3,
    w_mm: float = 0.0,
    basis_var: float = 0.0,
    seed: int = 0,
    organisms: list[str] | None = None,
) -> dict:
    ds = load_behaviour_dataset(labels_dir, index_path, eps=eps, seed=seed, organisms=organisms)
    heads = train_behaviour_heads(
        ds,
        width=width,
        depth=depth,
        epochs=epochs,
        batch=batch,
        lr=lr,
        w_mm=w_mm,
        basis_var=basis_var,
        seed=seed,
    )
    diagnostics = evaluate(heads, ds)
    save(
        heads,
        ds,
        outdir,
        {
            "width": width,
            "depth": depth,
            "epochs": epochs,
            "lr": lr,
            "w_mm": w_mm,
            "basis_var": basis_var,
            "basis_rank": None if heads.basis is None else int(heads.basis.shape[1]),
            "eps": eps,
            "seed": seed,
        },
        diagnostics,
    )
    LOGGER.info(
        "Head B: worst R2 %.3f, median %.3f; worst median cosine %.3f; worst sign agreement %.3f",
        diagnostics["summary"]["worst_r2"],
        diagnostics["summary"]["median_r2"],
        diagnostics["summary"]["worst_cosine_median"],
        diagnostics["summary"]["worst_sign_agreement"],
    )
    return diagnostics
