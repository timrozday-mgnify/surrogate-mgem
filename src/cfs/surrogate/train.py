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

Pick ``w_grad`` by held-out ``grad_cosine`` **subject to ``value_r2 >= 0.9``**:
at ``w_grad = 10`` the value head collapses (R2 -0.61) while cosine still climbs,
and the composition in §8 needs both.

``arch`` selects the head from :data:`_ARCH`: ``icnn`` (the concave default),
``deepset``/``deepset-private`` (:mod:`cfs.surrogate.deepset`) and ``mlp``
(:mod:`cfs.surrogate.mlp`, an unconstrained ceiling measurement, not a usable
head). Everything below is shared across them — the loss, the split and the gate
are what make the numbers comparable.

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
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from cfs.surrogate import deepset, mlp, picnn
from cfs.surrogate.data import ValueDataset, load_value_dataset

LOGGER = logging.getLogger("cfs.surrogate.train")

GRAD_COSINE_GATE = 0.99

# Architecture registry. The coupling surface is four names — `stack_heads`,
# `batched_value`, `batched_value_and_grad`, `organism` — so a variant is a module,
# not a plugin system. `deepset-private` is the same module with `shared=False`.
_ARCH = {"icnn": picnn, "deepset": deepset, "deepset-private": deepset, "mlp": mlp}


def _build(arch: str, key, n_organisms: int, n_in: int, mask, width: int, depth: int,
           emb_dim: int, phi_hidden: int | None = None, k_code: int | None = None):
    mod = _ARCH[arch]
    if mod is deepset:
        return mod.stack_heads(key, n_organisms, n_in, mask, width, depth, emb_dim,
                               shared=arch == "deepset", phi_hidden=phi_hidden, k_code=k_code)
    return mod.stack_heads(key, n_organisms, n_in, mask, width, depth)


def _du(x, x_scale):
    """``du/dx`` for the ``x = u/(u + s)`` input map, from ``x`` alone: ``(1-x)^2/s``.

    Gradients are compared in ``u`` space — ``u = c/(Km+c)``, the coordinate the
    stored duals live in — never in the network's own input space. A cosine taken
    in the input space is a different number for every input transform, so it
    cannot be compared between runs and improves for free when the transform is
    changed: the previous linear-rescale checkpoint reported 0.72 in its own
    coordinate and scores 0.09 here.
    """
    return (1.0 - x) ** 2 / x_scale[:, None, :]


def _loss(heads, x, mu, g, gvalid, x_scale, gfloor, w_grad, bvg):
    mu_hat, g_hat = bvg(heads, x)
    value = jnp.mean((mu_hat - mu) ** 2)
    # (G, B, 1) row mask * the head's own (G, 1, M) exchange mask.
    w = gvalid[..., None] * heads.mask[:, None, :]
    # Per-row *relative* Sobolev error, not the raw ||grad - pi||^2 of §7.1. The
    # duals span four orders of magnitude across media (median |pi| ~ 1.6e-2, 99th
    # percentile ~ 1.2e2), so an absolute MSE optimises a handful of ion-limited
    # rows and ignores everything else — measured grad cosine ~ 0.05. Dividing by
    # the target norm makes every medium contribute equally and, when the
    # magnitudes match, this term IS 2(1 - cos), which is the milestone gate.
    du = _du(x, x_scale)
    sq = jnp.sum(w * ((g_hat - g) * du) ** 2, axis=-1)
    raw = jnp.sum(w * (g * du) ** 2, axis=-1)
    # Media where nothing is limiting have an all-zero target — 23% of rows. They
    # are not excused from this term: their true gradient is the zero vector, and
    # dropping them is exactly what let the model spread 56% of its predicted
    # gradient magnitude onto non-limiting metabolites for free. They divide by
    # the per-organism denominator floor (the median row norm, `gfloor`) instead,
    # which makes their ||g_hat||^2 penalty comparable to everyone else's error.
    #
    # The floor does downweight genuinely-small-but-non-zero rows (bottom raw-norm
    # quintile: loss weight 0.006, held-out cosine 0.90, against 1.00 for the top
    # quintile). Restricting it to the all-zero rows, or clamping it at 1e-3..0.3
    # of gfloor, was measured across that whole range and moved the synthetic gate
    # by <=0.01 either way — it is not the binding constraint.
    grad = jnp.sum(gvalid * sq / (raw + gfloor[:, None])) / jnp.maximum(jnp.sum(gvalid), 1.0)
    return value + w_grad * grad, (value, grad)


@eqx.filter_jit
def _step(heads, opt_state, x, mu, g, gvalid, x_scale, gfloor, w_grad, optimiser, bvg):
    # `bvg` and `optimiser` are non-arrays, so `filter_jit` holds them static.
    (total, parts), grads = eqx.filter_value_and_grad(_loss, has_aux=True)(
        heads, x, mu, g, gvalid, x_scale, gfloor, w_grad, bvg
    )
    updates, opt_state = optimiser.update(grads, opt_state, eqx.filter(heads, eqx.is_inexact_array))
    return eqx.apply_updates(heads, updates), opt_state, total, parts


def _shard_organisms(n_organisms: int, tree):
    """Spread the organism axis over the available CPU devices, if there are any.

    The 21 heads are independent in the loss, so the organism axis is
    embarrassingly parallel — but on one device XLA has to rediscover that inside
    the deepset's 444 small per-metabolite matmuls, and only reaches ~5 of 12
    cores. Stating the split explicitly gives 1.8x (10.1 -> 5.6 s/epoch measured,
    deepset at batch 512) while changing no hyperparameter.

    No-op unless the caller asked for extra devices, so every existing run is
    untouched::

        XLA_FLAGS=--xla_force_host_platform_device_count=3 cfs train-value ...

    3 divides 21 and was the fastest tried; more devices is worse (21 shards of
    one organism just adds coordination to ops that are already small).
    """
    n_dev = jax.device_count()
    if n_dev < 2 or n_organisms % n_dev:
        if n_dev >= 2:
            LOGGER.info("%d organisms do not divide %d devices — not sharding",
                        n_organisms, n_dev)
        return tree
    # Auto, not the `jax.make_mesh` default of Explicit: under Explicit axis types
    # the shard spec is part of the array's *type*, so it propagates into traces
    # that never asked for it and `vmap` then rejects a sharded head against a
    # replicated `x_val` -- after training has finished. Auto keeps sharding a
    # layout hint, which is all this needs.
    mesh = jax.make_mesh((n_dev,), ("org",), axis_types=(jax.sharding.AxisType.Auto,))
    org = NamedSharding(mesh, P("org"))
    repl = NamedSharding(mesh, P())
    LOGGER.info("sharding the organism axis over %d CPU devices", n_dev)
    # Leaves without a leading organism axis (the deepset's shared trunk, scalars)
    # replicate; everything else splits.
    return jax.tree.map(
        lambda a: jax.device_put(a, org if getattr(a, "shape", ()) and a.shape[0] == n_organisms
                                 else repl) if eqx.is_array(a) else a, tree)


def train_value_heads(ds: ValueDataset, *, arch: str = "icnn", width: int = 128, depth: int = 3,
                      epochs: int = 400, batch: int = 512, lr: float = 3e-3,
                      w_grad: float = 1.0, emb_dim: int = 8, phi_hidden: int | None = None,
                      k_code: int | None = None, seed: int = 0) -> eqx.Module:
    """Fit the stacked Head A. Labels are scaled by ``ds.mu_scale`` (§7.1)."""
    bvg = _ARCH[arch].batched_value_and_grad
    key = jax.random.PRNGKey(seed)
    scale = jnp.asarray(ds.mu_scale)[:, None]
    x = jnp.asarray(ds.x_train)
    mu = jnp.asarray(ds.mu_train) / scale
    g = jnp.asarray(ds.g_train) / scale[..., None]
    gvalid = jnp.asarray(ds.gvalid_train, dtype=jnp.float32)
    x_scale = jnp.asarray(ds.x_scale)
    # Per-organism denominator floor for the Sobolev term: the median u-space row
    # norm over the rows that do have a limiting metabolite.
    raw = np.sum(np.asarray(g * _du(x, x_scale)) ** 2 * ds.mask[:, None, :], axis=-1)
    ok = (raw > 0) & np.asarray(ds.gvalid_train)
    gfloor = jnp.asarray([float(np.median(r[o])) if o.any() else 1.0
                          for r, o in zip(raw, ok, strict=True)])

    heads = _build(arch, key, len(ds.genome_ids), x.shape[-1], ds.mask, width, depth, emb_dim,
                   phi_hidden=phi_hidden, k_code=k_code)
    n = x.shape[1]
    steps_per_epoch = max(1, n // batch)
    optimiser = optax.adam(optax.cosine_decay_schedule(lr, epochs * steps_per_epoch))
    opt_state = optimiser.init(eqx.filter(heads, eqx.is_inexact_array))
    heads, opt_state, x, mu, g, gvalid, x_scale, gfloor = _shard_organisms(
        len(ds.genome_ids), (heads, opt_state, x, mu, g, gvalid, x_scale, gfloor))

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
                x_scale, gfloor, w_grad, optimiser, bvg,
            )
        if epoch % 20 == 0 or epoch == epochs - 1:
            LOGGER.info("epoch %4d  loss=%.5f  value=%.5f  grad=%.5f  (%.0fs)",
                        epoch, float(total), float(v), float(gl), time.time() - t0)
    # Gather back onto one device before returning. Sharding is an optimisation
    # internal to this function, and everything downstream (`evaluate`, `save`)
    # builds its own unsharded arrays from `ds` -- vmap rejects a sharded head
    # against a replicated input with "inconsistent axis specs: org vs None",
    # which is a crash *after* training, i.e. it costs the whole run.
    return jax.device_put(heads, jax.devices()[0])


# --------------------------------------------------------------------------- #
# Diagnostics (§7.3)
# --------------------------------------------------------------------------- #

def _cosine(a, b, mask):
    """Per-row cosine over the masked dims; NaN where the target vector is 0."""
    a, b = a * mask, b * mask
    denom = jnp.linalg.norm(a, axis=-1) * jnp.linalg.norm(b, axis=-1)
    return jnp.where(denom > 0, jnp.sum(a * b, axis=-1) / jnp.where(denom > 0, denom, 1.0), jnp.nan)


def _concavity_violations(heads, x, key, bv, n_pairs: int = 2000, tol: float = 1e-5):
    """f(la + (1-l)b) >= l f(a) + (1-l) f(b) on random convex combinations."""
    g, n = x.shape[0], x.shape[1]
    ka, kb, kl = jax.random.split(key, 3)
    ia = jax.random.randint(ka, (g, n_pairs), 0, n)
    ib = jax.random.randint(kb, (g, n_pairs), 0, n)
    lam = jax.random.uniform(kl, (g, n_pairs, 1))
    xa = jnp.take_along_axis(x, ia[..., None], axis=1)
    xb = jnp.take_along_axis(x, ib[..., None], axis=1)
    mid = bv(heads, lam * xa + (1 - lam) * xb)
    chord = lam[..., 0] * bv(heads, xa) + (1 - lam[..., 0]) * bv(heads, xb)
    return jnp.mean(mid < chord - tol, axis=1)


def _hessian_cond(heads, x, organism, n_points: int = 8):
    """cond(Hessian) on the organism's own dims — predicts Phase-5 Newton failure."""
    conds = []
    for i in range(x.shape[0]):
        head = organism(heads, i)
        dims = np.flatnonzero(np.asarray(head.mask))
        h = jax.vmap(jax.hessian(head))(x[i, :n_points])[:, dims][:, :, dims]
        ev = jnp.abs(jnp.linalg.eigvalsh(h))
        conds.append(float(jnp.median(ev.max(axis=1) / jnp.maximum(ev.min(axis=1), 1e-30))))
    return conds


def held_out_targets(ds: ValueDataset):
    """``(x, mu, g)`` for the held-out media, with ``g`` already in ``u`` space.

    Shared by every scorer so the gate means the same thing whatever produced the
    prediction — including the non-differentiable baselines, which reach it by
    finite differences (:mod:`cfs.surrogate.baseline`).
    """
    scale = jnp.asarray(ds.mu_scale)[:, None]
    x = jnp.asarray(ds.x_val)
    du = _du(x, jnp.asarray(ds.x_scale))
    return x, jnp.asarray(ds.mu_val) / scale, jnp.asarray(ds.g_val) / scale[..., None] * du


def score(ds: ValueDataset, mu_hat, g_hat, extra: dict[str, dict] | None = None,
          arch: str = "") -> dict:
    """Assemble §7.3's diagnostics from held-out predictions.

    ``mu_hat`` is ``(G, B)`` on the ``mu_scale``d target and ``g_hat`` is
    ``(G, B, M)`` **in u space** — the caller converts, because how it gets a
    gradient is what differs between a network and a forest. ``extra`` carries the
    per-organism fields only some models have (concavity, Hessian conditioning).
    """
    _, mu, g = held_out_targets(ds)
    mask = jnp.asarray(ds.mask)
    gvalid = jnp.asarray(ds.gvalid_val)
    extra = extra or {}

    cos = _cosine(g_hat, g, mask[:, None, :])
    cos = jnp.where(gvalid, cos, jnp.nan)
    r2 = 1.0 - jnp.sum((mu_hat - mu) ** 2, axis=1) / jnp.sum((mu - mu.mean(1, keepdims=True)) ** 2,
                                                             axis=1)
    err = jnp.abs(g_hat - g) * mask[:, None, :]

    # Share of predicted gradient norm landing on the *true* argmax metabolite.
    # 54.7% of rows have a single non-zero dual, so this is the number that says
    # why a cosine is what it is: a dense predictor scores near 1/|M_i| here even
    # when its cosine looks respectable.
    m = mask[:, None, :]
    top = jnp.take_along_axis(jnp.abs(g_hat * m), jnp.argmax(jnp.abs(g * m), axis=-1)[..., None],
                              axis=-1)[..., 0]
    share = jnp.where(jnp.linalg.norm(g_hat * m, axis=-1) > 0,
                      top / jnp.linalg.norm(g_hat * m + 1e-30, axis=-1), jnp.nan)
    share = jnp.where(gvalid & (jnp.linalg.norm(g * m, axis=-1) > 0), share, jnp.nan)

    # Held-out cosine and row count *per limiting metabolite* — the unit the next
    # label budget is allocated in (`cfs.sampling.design.topup_weights`). Cosine
    # tracks the cell's training row count at Spearman 0.72 (<50 rows -> 0.49,
    # >400 -> 0.95), so this is a direct read of where labels are missing. It is
    # the measured-error signal an ensemble's predictive variance would only be a
    # proxy for; we have a val split, so we do not need the proxy.
    kt = np.asarray(jnp.argmax(jnp.abs(g * m), axis=-1))
    okv = np.asarray(gvalid & (jnp.linalg.norm(g * m, axis=-1) > 0))
    cos_np = np.asarray(cos)

    per = {}
    for i, gid in enumerate(ds.genome_ids):
        worst = np.argsort(-np.asarray(err[i].mean(axis=0)))[:5]
        by_met = {}
        for j in np.unique(kt[i][okv[i]]):
            s = okv[i] & (kt[i] == j)
            by_met[ds.exchanges[j]] = {"rows": int(s.sum()),
                                       "grad_cosine": float(np.nanmean(cos_np[i][s]))}
        per[gid] = {
            "grad_cosine": float(jnp.nanmean(cos[i])),
            "grad_cosine_p05": float(jnp.nanpercentile(cos[i], 5)),
            "grad_top1_share": float(jnp.nanmean(share[i])),
            "value_r2": float(r2[i]),
            **extra.get(gid, {}),
            "worst_grad_metabolites": [ds.exchanges[j] for j in worst],
            "per_limiting_metabolite": by_met,
        }
    gate = min(v["grad_cosine"] for v in per.values())
    # The split provenance is part of the result: two runs are only comparable if
    # they were scored on the same held-out media, and the §4.6 rounds are exactly
    # what can silently change that (see `load_value_dataset`).
    return {"gate": "grad_cosine > 0.99, in u = c/(Km+c) space", "worst_grad_cosine": gate,
            "passed": bool(gate > GRAD_COSINE_GATE), "arch": arch,
            "n_val_media": int(ds.x_val.shape[1]), "n_train_media": int(ds.x_train.shape[1]),
            "rounds_present": ds.rounds_present, "per_organism": per}


def evaluate(heads: eqx.Module, ds: ValueDataset, seed: int = 0, arch: str = "icnn") -> dict:
    """Held-out diagnostics per organism (§7.3). The gate is ``grad_cosine``."""
    mod = _ARCH[arch]
    x = jnp.asarray(ds.x_val)
    mu_hat, g_hat = mod.batched_value_and_grad(heads, x)
    # In u space (see `_du`), so the gate means the same thing across runs.
    g_hat = g_hat * _du(x, jnp.asarray(ds.x_scale))
    viol = _concavity_violations(heads, x, jax.random.PRNGKey(seed), mod.batched_value)
    conds = _hessian_cond(heads, x, mod.organism)
    extra = {gid: {"concavity_violation_rate": float(viol[i]), "hessian_cond_median": conds[i]}
             for i, gid in enumerate(ds.genome_ids)}
    return score(ds, mu_hat, g_hat, extra, arch=arch)


# --------------------------------------------------------------------------- #
# Checkpoint (P13 / P14)
# --------------------------------------------------------------------------- #

def save(heads: eqx.Module, ds: ValueDataset, outdir: Path, arch: dict,
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
        "input_transform": "u = c / (Km + c), x = u / (u + x_scale); Km from km_defaults.yaml",
        "gradient_units": ("d(mu_max)/dx = max(-shadow, 0) * Vmax * (u + x_scale)^2 / x_scale, "
                           "Vmax = 1000"),
        "arch": arch,
    }, indent=2))
    (outdir / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2))


def load(outdir: Path, width: int = 128, depth: int = 3) -> tuple[eqx.Module, dict]:
    """Reload a saved stacked head and its metadata."""
    outdir = Path(outdir)
    meta = json.loads((outdir / "value_heads.json").read_text())
    arch = meta.get("arch", {})
    like = _build(arch.get("arch", "icnn"), jax.random.PRNGKey(0), len(meta["genome_ids"]),
                  len(meta["exchanges"]), np.array(meta["mask"], dtype=bool),
                  arch.get("width", width), arch.get("depth", depth), arch.get("emb_dim", 8),
                  phi_hidden=arch.get("phi_hidden"), k_code=arch.get("k_code"))
    return eqx.tree_deserialise_leaves(outdir / "value_heads.eqx", like), meta


def run(labels_dir: Path, index_path: Path, outdir: Path, *, eps: float = 1e-3,
        arch: str = "icnn", width: int = 128, depth: int = 3, epochs: int = 400,
        batch: int = 512, lr: float = 3e-3, w_grad: float = 1.0, emb_dim: int = 8,
        phi_hidden: int | None = None, k_code: int | None = None, seed: int = 0) -> dict:
    """Load labels, train, evaluate, checkpoint. Returns the diagnostics."""
    ds = load_value_dataset(labels_dir, index_path, eps=eps, seed=seed)
    heads = train_value_heads(ds, arch=arch, width=width, depth=depth, epochs=epochs,
                              batch=batch, lr=lr, w_grad=w_grad, emb_dim=emb_dim,
                              phi_hidden=phi_hidden, k_code=k_code, seed=seed)
    diagnostics = evaluate(heads, ds, seed=seed, arch=arch)
    meta = {"arch": arch, "width": width, "depth": depth, "epochs": epochs, "lr": lr,
            "w_grad": w_grad, "eps": eps, "seed": seed}
    if arch.startswith("deepset"):
        # Only when set: an absent key is what makes `load` fall back to the
        # width-derived default, so a checkpoint written before these flags existed
        # still reconstructs.
        meta["emb_dim"] = emb_dim
        if phi_hidden is not None:
            meta["phi_hidden"] = phi_hidden
        if k_code is not None:
            meta["k_code"] = k_code
    save(heads, ds, outdir, meta, diagnostics)
    return diagnostics
