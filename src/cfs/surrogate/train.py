"""M3 — train Head A on the §4.5 label shards (plan §7).

One stacked :class:`~cfs.surrogate.picnn.ValueHead` PyTree with a leading organism
axis, vmapped (§6.1), trained on the value + Sobolev-gradient loss (§7.1):

```
L = w_v * mean((mu_hat - mu)^2)  +  w_g * mean(||grad_x mu_hat - g||^2)
```

both terms on ``mu`` scaled by the per-organism label std, so they are
dimensionless and one weight balances them. **The gradient term is the one that
matters**: the master problem (§8) and HMC follow slopes and never look at
values. It is masked to rows whose duals are usable and to the organism's own
``M_i``.

The milestone gate is held-out per-sample gradient cosine > 0.99. §7.3's other
diagnostics ride along: concavity violation rate (structural — non-zero means the
ICNN constraint is broken), Hessian condition number (the early warning for
Newton failure in Phase 5), and per-metabolite gradient error.
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

from cfs.surrogate.data import ValueDataset, load_value_dataset
from cfs.surrogate.picnn import (
    ValueHead,
    batched_value,
    batched_value_and_grad,
    stack_heads,
)

LOGGER = logging.getLogger("cfs.surrogate.train")

GRAD_COSINE_GATE = 0.99


_GRAD_FLOOR = 1e-6  # keeps replete media (every dual 0) from dividing by ~nothing


def _loss(heads, x, mu, g, gvalid, w_grad):
    mu_hat, g_hat = batched_value_and_grad(heads, x)
    value = jnp.mean((mu_hat - mu) ** 2)
    # (G, B, 1) row mask * the head's own (G, 1, M) exchange mask.
    w = gvalid[..., None] * heads.mask[:, None, :]
    # Per-row *relative* Sobolev error, not the raw ||grad - pi||^2 of §7.1. The
    # duals span four orders of magnitude across media (median |pi| ~ 1.6e-2, 99th
    # percentile ~ 1.2e2), so an absolute MSE optimises a handful of ion-limited
    # rows and ignores everything else — measured grad cosine ~ 0.05. Dividing by
    # the target norm makes every medium contribute equally and, when the
    # magnitudes match, this term IS 2(1 - cos), which is the milestone gate.
    sq = jnp.sum(w * (g_hat - g) ** 2, axis=-1)
    raw = jnp.sum(w * g**2, axis=-1)
    # Media where nothing is limiting have an all-zero target: no direction to
    # get right, and normalising them means dividing by ~nothing. Drop them from
    # this term (~5% of rows); the value term still pins them.
    rows = gvalid * (raw > 0)
    grad = jnp.sum(rows * sq / (raw + _GRAD_FLOOR)) / jnp.maximum(jnp.sum(rows), 1.0)
    return value + w_grad * grad, (value, grad)


@eqx.filter_jit
def _step(heads, opt_state, x, mu, g, gvalid, w_grad, optimiser):
    (total, parts), grads = eqx.filter_value_and_grad(_loss, has_aux=True)(
        heads, x, mu, g, gvalid, w_grad
    )
    updates, opt_state = optimiser.update(grads, opt_state, eqx.filter(heads, eqx.is_inexact_array))
    return eqx.apply_updates(heads, updates), opt_state, total, parts


def train_value_heads(ds: ValueDataset, *, width: int = 128, depth: int = 3,
                      epochs: int = 400, batch: int = 512, lr: float = 3e-3,
                      w_grad: float = 1.0, seed: int = 0) -> ValueHead:
    """Fit the stacked Head A. Labels are scaled by ``ds.mu_scale`` (§7.1)."""
    key = jax.random.PRNGKey(seed)
    scale = jnp.asarray(ds.mu_scale)[:, None]
    x = jnp.asarray(ds.x_train)
    mu = jnp.asarray(ds.mu_train) / scale
    g = jnp.asarray(ds.g_train) / scale[..., None]
    gvalid = jnp.asarray(ds.gvalid_train, dtype=jnp.float32)

    heads = stack_heads(key, len(ds.genome_ids), x.shape[-1], ds.mask, width, depth)
    n = x.shape[1]
    steps_per_epoch = max(1, n // batch)
    optimiser = optax.adam(optax.cosine_decay_schedule(lr, epochs * steps_per_epoch))
    opt_state = optimiser.init(eqx.filter(heads, eqx.is_inexact_array))

    rng = np.random.default_rng(seed)
    t0 = time.time()
    for epoch in range(epochs):
        perm = rng.permutation(n)
        for s in range(steps_per_epoch):
            # The same media indices for every organism: the batch axis is media,
            # and all organisms share the design size (checked by the loader).
            idx = jnp.asarray(perm[s * batch:(s + 1) * batch])
            heads, opt_state, total, (v, gl) = _step(
                heads, opt_state, x[:, idx], mu[:, idx], g[:, idx], gvalid[:, idx],
                w_grad, optimiser,
            )
        if epoch % 20 == 0 or epoch == epochs - 1:
            LOGGER.info("epoch %4d  loss=%.5f  value=%.5f  grad=%.5f  (%.0fs)",
                        epoch, float(total), float(v), float(gl), time.time() - t0)
    return heads


# --------------------------------------------------------------------------- #
# Diagnostics (§7.3)
# --------------------------------------------------------------------------- #

def _cosine(a, b, mask):
    """Per-row cosine over the masked dims; NaN where the target vector is 0."""
    a, b = a * mask, b * mask
    denom = jnp.linalg.norm(a, axis=-1) * jnp.linalg.norm(b, axis=-1)
    return jnp.where(denom > 0, jnp.sum(a * b, axis=-1) / jnp.where(denom > 0, denom, 1.0), jnp.nan)


def _concavity_violations(heads, x, key, n_pairs: int = 2000, tol: float = 1e-5):
    """f(la + (1-l)b) >= l f(a) + (1-l) f(b) on random convex combinations."""
    g, n = x.shape[0], x.shape[1]
    ka, kb, kl = jax.random.split(key, 3)
    ia = jax.random.randint(ka, (g, n_pairs), 0, n)
    ib = jax.random.randint(kb, (g, n_pairs), 0, n)
    lam = jax.random.uniform(kl, (g, n_pairs, 1))
    xa = jnp.take_along_axis(x, ia[..., None], axis=1)
    xb = jnp.take_along_axis(x, ib[..., None], axis=1)
    mid = batched_value(heads, lam * xa + (1 - lam) * xb)
    chord = lam[..., 0] * batched_value(heads, xa) + (1 - lam[..., 0]) * batched_value(heads, xb)
    return jnp.mean(mid < chord - tol, axis=1)


def _hessian_cond(heads, x, n_points: int = 8):
    """cond(Hessian) on the organism's own dims — predicts Phase-5 Newton failure."""
    conds = []
    for i in range(x.shape[0]):
        head = jax.tree.map(lambda p, i=i: p[i] if eqx.is_array(p) else p, heads)
        dims = np.flatnonzero(np.asarray(head.mask))
        h = jax.vmap(jax.hessian(head))(x[i, :n_points])[:, dims][:, :, dims]
        ev = jnp.abs(jnp.linalg.eigvalsh(h))
        conds.append(float(jnp.median(ev.max(axis=1) / jnp.maximum(ev.min(axis=1), 1e-30))))
    return conds


def evaluate(heads: ValueHead, ds: ValueDataset, seed: int = 0) -> dict:
    """Held-out diagnostics per organism (§7.3). The gate is ``grad_cosine``."""
    scale = jnp.asarray(ds.mu_scale)[:, None]
    x = jnp.asarray(ds.x_val)
    mu = jnp.asarray(ds.mu_val) / scale
    g = jnp.asarray(ds.g_val) / scale[..., None]
    gvalid = jnp.asarray(ds.gvalid_val)

    mu_hat, g_hat = batched_value_and_grad(heads, x)
    cos = _cosine(g_hat, g, heads.mask[:, None, :])
    cos = jnp.where(gvalid, cos, jnp.nan)
    r2 = 1.0 - jnp.sum((mu_hat - mu) ** 2, axis=1) / jnp.sum((mu - mu.mean(1, keepdims=True)) ** 2,
                                                             axis=1)
    err = jnp.abs(g_hat - g) * heads.mask[:, None, :]
    viol = _concavity_violations(heads, x, jax.random.PRNGKey(seed))
    conds = _hessian_cond(heads, x)

    per = {}
    for i, gid in enumerate(ds.genome_ids):
        worst = np.argsort(-np.asarray(err[i].mean(axis=0)))[:5]
        per[gid] = {
            "grad_cosine": float(jnp.nanmean(cos[i])),
            "grad_cosine_p05": float(jnp.nanpercentile(cos[i], 5)),
            "value_r2": float(r2[i]),
            "concavity_violation_rate": float(viol[i]),
            "hessian_cond_median": conds[i],
            "worst_grad_metabolites": [ds.exchanges[j] for j in worst],
        }
    gate = min(v["grad_cosine"] for v in per.values())
    return {"gate": "grad_cosine > 0.99", "worst_grad_cosine": gate,
            "passed": bool(gate > GRAD_COSINE_GATE), "per_organism": per}


# --------------------------------------------------------------------------- #
# Checkpoint (P13 / P14)
# --------------------------------------------------------------------------- #

def save(heads: ValueHead, ds: ValueDataset, outdir: Path, arch: dict,
         diagnostics: dict) -> None:
    """Serialise the stacked heads plus everything needed to use them again.

    The metadata is not optional: the input transform, ``Vmax``, the label scale
    and the ``index_hash`` are what stop a checkpoint being silently reinterpreted
    against a different metabolite index or a different unit convention.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(outdir / "value_heads.eqx", heads)
    (outdir / "value_heads.json").write_text(json.dumps({
        "index_hash": ds.index_hash,
        "genome_ids": ds.genome_ids,
        "exchanges": ds.exchanges,
        "mask": ds.mask.astype(int).tolist(),
        "mu_scale": ds.mu_scale.tolist(),
        "x_scale": ds.x_scale.tolist(),
        "input_transform": "x = c / (Km + c) / x_scale, Km from km_defaults.yaml",
        "gradient_units": "d(mu_max)/dx = max(-shadow, 0) * Vmax * x_scale, Vmax = 1000",
        "arch": arch,
    }, indent=2))
    (outdir / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2))


def load(outdir: Path, width: int = 128, depth: int = 3) -> tuple[ValueHead, dict]:
    """Reload a saved stacked head and its metadata."""
    outdir = Path(outdir)
    meta = json.loads((outdir / "value_heads.json").read_text())
    arch = meta.get("arch", {})
    like = stack_heads(jax.random.PRNGKey(0), len(meta["genome_ids"]), len(meta["exchanges"]),
                       np.array(meta["mask"], dtype=bool),
                       arch.get("width", width), arch.get("depth", depth))
    return eqx.tree_deserialise_leaves(outdir / "value_heads.eqx", like), meta


def run(labels_dir: Path, index_path: Path, outdir: Path, *, eps: float = 1e-3,
        width: int = 128, depth: int = 3, epochs: int = 400, batch: int = 512,
        lr: float = 3e-3, w_grad: float = 1.0, seed: int = 0) -> dict:
    """Load labels, train, evaluate, checkpoint. Returns the diagnostics."""
    ds = load_value_dataset(labels_dir, index_path, eps=eps, seed=seed)
    heads = train_value_heads(ds, width=width, depth=depth, epochs=epochs, batch=batch,
                              lr=lr, w_grad=w_grad, seed=seed)
    diagnostics = evaluate(heads, ds, seed=seed)
    arch = {"width": width, "depth": depth, "epochs": epochs, "lr": lr, "w_grad": w_grad,
            "eps": eps, "seed": seed}
    save(heads, ds, outdir, arch, diagnostics)
    return diagnostics
