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

from cfs.surrogate import calibrate, deepset, deepset_u, groupmax, mlp, picnn, picnn_u
from cfs.surrogate.data import ValueDataset, load_value_dataset

LOGGER = logging.getLogger("cfs.surrogate.train")

GRAD_COSINE_GATE = 0.99

# Temperature-anneal stages. Each one retraces the step, so this is not a sweep axis.
_ANNEAL_STAGES = 3

# Architecture registry. The coupling surface is four names — `stack_heads`,
# `batched_value`, `batched_value_and_grad`, `organism` — so a variant is a module,
# not a plugin system. `deepset-private` is the same module with `shared=False`.
_ARCH = {
    "icnn": picnn,
    "icnn-u": picnn_u,
    "deepset": deepset,
    "deepset-private": deepset,
    "deepset-u": deepset_u,
    "deepset-u-private": deepset_u,
    "groupmax-u": groupmax,
    "mlp": mlp,
}


def _build(
    arch: str,
    key,
    n_organisms: int,
    n_in: int,
    mask,
    width: int,
    depth: int,
    emb_dim: int,
    phi_hidden: int | None = None,
    k_code: int | None = None,
    gm_group: int | None = None,
    gm_temp: float | None = None,
):
    mod = _ARCH[arch]
    # Both deepset modules take the same extra knobs; `shared` is the arch name, not
    # the module, since each module serves a shared and a private variant.
    if arch.startswith("deepset"):
        return mod.stack_heads(
            key,
            n_organisms,
            n_in,
            mask,
            width,
            depth,
            emb_dim,
            shared=arch in ("deepset", "deepset-u"),
            phi_hidden=phi_hidden,
            k_code=k_code,
        )
    if mod is groupmax:
        return mod.stack_heads(
            key,
            n_organisms,
            n_in,
            mask,
            width,
            depth,
            group=gm_group or groupmax.DEFAULT_GROUP,
            temp=groupmax.DEFAULT_TEMP if gm_temp is None else gm_temp,
        )
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


def _trial_points(ds, media_npz) -> np.ndarray | None:
    """``(G, P, M)`` trial points in ``w`` for SDDP-style Level 1 cut selection.

    ``media_npz`` is a ``cfs community-holdout make`` archive — media drawn over
    the *union* of a community's members' active subspaces. Feeding those as the
    trial-point set is the whole point of Level 1: cuts are kept because they are
    the active minimum where the head will actually be evaluated, rather than where
    the design happened to sample. None (the default) leaves the training rows as
    the point set, which is the direct analogue of SDDP's "points the forward pass
    visited".
    """
    if media_npz is None:
        return None
    z = np.load(Path(media_npz), allow_pickle=True)
    ex = [str(e) for e in z["exchanges"]]
    n_met = ds.n_metabolites or len(ds.exchanges)
    if ex != list(ds.exchanges)[:n_met]:
        raise ValueError("trial media and labels disagree on the metabolite index (P13)")
    from cfs.groundtruth.solve import km_for_exchange, load_km_defaults

    km_cfg = load_km_defaults()
    km = np.array([km_for_exchange(e, km_cfg) for e in ex])
    u = z["media"] / (km + z["media"])  # (P, M)
    # §13.11 stage 4': under inhibition the head's input is [u | theta], so a
    # trial point needs both halves. `theta` comes from the same medium
    # (`data.theta`); the archive stores concentrations, not the coordinate.
    if n_met < len(ds.exchanges):
        from cfs.surrogate.data import theta as _theta

        ceq = np.array([(ds.ceq or {}).get(e, np.nan) for e in ex])
        u = np.concatenate([u, _theta(z["media"], ceq)], axis=-1)  # (P, 2M)
    # Same map the head reads: w = min(u / x_scale, W_CAP), per organism.
    return np.minimum(u[None] / ds.x_scale[:, None, :], groupmax.W_CAP)


def _loss(heads, x, mu, g, gvalid, x_scale, gfloor, w_grad, w_rel, w_under, w_tau, a0, w_prox, bvg):
    mu_hat, g_hat = bvg(heads, x)
    # `w_tau` makes the value term an **expectile** (asymmetric least squares, Newey
    # & Powell 1987) rather than a mean: residuals on the under-predicting side get
    # weight `tau`, the rest `1 - tau`. `w_under`'s hinge only sees rows already in
    # violation, so it can push a plane back over a label without changing where the
    # plane points; the expectile reweights *every* row, which is what moves the
    # slopes -- and slopes are what the validity projection (`--gm-repair`) provably
    # cannot fix. The 2x is chosen so `tau = 0.5` reproduces the plain MSE exactly,
    # bit for bit, keeping `lr` and `w_grad` on the scale every number on file used.
    resid = mu_hat - mu
    aw = 1.0 if w_tau == 0.5 else 2.0 * jnp.where(resid < 0, w_tau, 1.0 - w_tau)
    value = jnp.mean(aw * resid**2)
    # `w_rel` adds the same error measured *relatively*. The plain MSE is absolute
    # and 74% of rows sit on the plateau, so the 15% below a quarter of max mu carry
    # no weight and the head over-predicts every one of them -- median +98% at
    # mu < 5% of max, and 100% of rows below 75% of max over-predicted (seeded
    # `groupmax-u`, `20hm_bands`, 21 organisms). Composition integrates the relative
    # error (`d(log X)/dt = mu`), so that is what a slow member costs in §8.
    # The floor in the denominator caps the reweighting at 10x the organism's mean
    # mu; without it the plateau goes unweighted and R2 collapses to -1.66.
    if w_rel:
        den = mu + 0.1 * jnp.abs(mu).mean(axis=1, keepdims=True)
        value = value + w_rel * jnp.mean(aw * (resid / den) ** 2)
    # `w_under` penalises *under*-prediction only, relative, and it is a different
    # object from `w_rel`: it forbids a **provable violation** rather than trading
    # accuracy. `mu_max` is concave in `u` and the head is a min of affine pieces,
    # so a correct head is an upper bound on every labelled point — a row where
    # `mu_hat < mu` proves some plane has drifted below the target, and the min
    # then locks that in everywhere near it.
    #
    # It binds at the bottom, which is where the family loses the property and
    # where composition cannot afford it. Measured on the training rows of the
    # `probe_lo` design (`labels_p4`), bottom 5% of `mu`: **53-68% of rows are
    # under-predicted**, against 0.2-0.5% on the pre-relabel design — the design
    # went bottom-heavy (62% of media below 0.2 of max `mu` against r1's 10%), so
    # the absolute MSE now has thousands of low-`mu` rows and straddles them.
    # `d(log X)/dt = mu` then reads the slowest member as dead: at the failing
    # 21-member medium the head predicts 0.055 against a true 0.363, where the
    # cutting-plane model over the *same* labels' tangents is exact (0.363).
    if w_under:
        den = mu + 0.1 * jnp.abs(mu).mean(axis=1, keepdims=True)
        value = value + w_under * jnp.mean((jnp.maximum(mu - mu_hat, 0.0) / den) ** 2)
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
    total = value + w_grad * grad
    # `w_prox` is a stability centre at the seeded label tangents (proximal /
    # level bundle methods -- Lemarechal, Nemirovskii & Nesterov 1995; Kiwiel).
    # Measured motivation, not a guess: frozen cuts win the n=21 tail (log-X 0.169
    # against every trained head's 0.27-0.36) while gradient training wins the bulk
    # (overall 0.007 vs 0.013), and cut *selection* is exhausted -- 10x the trial
    # points and 2x the budget both do nothing. So the remaining lever is how far
    # the slopes are allowed to leave the duals that are exactly right at the tail.
    #
    # Normalised **per plane**, and that is not cosmetic: a single global
    # `mean(d^2)/mean(a0^2)` is what the first version used, and the label slopes
    # span five decades, so the denominator is set by a handful of enormous planes
    # and the ratio is ~0 for any drift the rest of them have. Measured: the term
    # was then bit-for-bit inert over `w_prox` 0 -> 10 (identical composition to
    # three decimals) while 2.5 million of 9.3 million slope entries had in fact
    # moved, by up to 3.2 in absolute terms. A "scale-free" normalisation over a
    # heavy-tailed quantity is not scale-free.
    if w_prox and a0 is not None:
        d = jax.nn.softplus(heads.wx[0]) - a0
        num = jnp.sum(d**2, axis=-1)
        den = jnp.sum(a0**2, axis=-1)
        # A tangent whose dual is all-zero (a medium where nothing limits) seeds a
        # flat plane, so its `den` is 0 and a bare epsilon floor would hand it a
        # weight of 1e12. Floor on the organism's own median instead, which leaves
        # every real plane untouched.
        den = jnp.maximum(den, 0.01 * jnp.median(den, axis=-1, keepdims=True) + 1e-30)
        total = total + w_prox * jnp.mean(num / den)
    return total, (value, grad)


@eqx.filter_jit
def _step(
    heads,
    opt_state,
    x,
    mu,
    g,
    gvalid,
    x_scale,
    gfloor,
    w_grad,
    w_rel,
    w_under,
    w_tau,
    a0,
    w_prox,
    optimiser,
    bvg,
):
    # `bvg` and `optimiser` are non-arrays, so `filter_jit` holds them static.
    (total, parts), grads = eqx.filter_value_and_grad(_loss, has_aux=True)(
        heads, x, mu, g, gvalid, x_scale, gfloor, w_grad, w_rel, w_under, w_tau, a0, w_prox, bvg
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
            LOGGER.info("%d organisms do not divide %d devices — not sharding", n_organisms, n_dev)
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
        lambda a: (
            jax.device_put(
                a, org if getattr(a, "shape", ()) and a.shape[0] == n_organisms else repl
            )
            if eqx.is_array(a)
            else a
        ),
        tree,
    )


def train_value_heads(
    ds: ValueDataset,
    *,
    arch: str = "icnn",
    width: int = 128,
    depth: int = 3,
    epochs: int = 400,
    batch: int = 512,
    lr: float = 3e-3,
    w_grad: float = 1.0,
    w_rel: float = 0.0,
    w_under: float = 0.0,
    w_tau: float = 0.5,
    w_prox: float = 0.0,
    emb_dim: int = 8,
    phi_hidden: int | None = None,
    k_code: int | None = None,
    gm_group: int | None = None,
    gm_temp: float | None = None,
    gm_init: str | None = None,
    gm_reanchor: int = 0,
    gm_select: str = "active-set",
    gm_trial_media=None,
    gm_temp_final: float | None = None,
    seed: int = 0,
) -> eqx.Module:
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
    gfloor = jnp.asarray(
        [float(np.median(r[o])) if o.any() else 1.0 for r, o in zip(raw, ok, strict=True)]
    )

    heads = _build(
        arch,
        key,
        len(ds.genome_ids),
        x.shape[-1],
        ds.mask,
        width,
        depth,
        emb_dim,
        phi_hidden=phi_hidden,
        k_code=k_code,
        gm_group=gm_group,
        gm_temp=gm_temp,
    )
    if gm_init == "labels":
        # Seeded from the labels' own supporting hyperplanes, not from noise. Done
        # here rather than in `_build` so the architecture registry stays data-free.
        heads = groupmax.init_from_tangents(
            heads,
            ds,
            seed=seed,
            select=gm_select,
            trial_points=_trial_points(ds, gm_trial_media),
        )
    # The stability centre for `w_prox`: the seeded slopes, in slope space rather
    # than in the raw parameter, so the penalty means the same thing regardless of
    # where softplus is being evaluated.
    a0 = (
        jax.nn.softplus(heads.wx[0])
        if w_prox and gm_init == "labels" and arch.startswith("groupmax")
        else None
    )
    if w_prox and a0 is None:
        LOGGER.warning("--w-prox needs a seeded groupmax head; ignored")
    n = x.shape[1]
    steps_per_epoch = max(1, n // batch)
    # `--epochs 0` is a supported mode, not a degenerate one: with `--gm-init labels`
    # it is the pure selected-tangent model -- SDDP's outer approximation with the
    # cuts left exactly as the label duals wrote them. The schedule still has to be
    # constructible, and optax rejects zero decay steps.
    optimiser = optax.adam(optax.cosine_decay_schedule(lr, max(1, epochs * steps_per_epoch)))
    opt_state = optimiser.init(eqx.filter(heads, eqx.is_inexact_array))
    heads, opt_state, x, mu, g, gvalid, x_scale, gfloor, a0 = _shard_organisms(
        len(ds.genome_ids), (heads, opt_state, x, mu, g, gvalid, x_scale, gfloor, a0)
    )

    # Evenly spaced over the run, none at the very end: a re-anchored plane needs
    # epochs left to settle.
    reanchor_at = {round(epochs * (k + 1) / (gm_reanchor + 1)) for k in range(gm_reanchor)}
    # Temperature homotopy (§5.4's pattern, applied to `T` instead of `eps`). The
    # smoothing sits ~T*ln(K_active) *below* the hard min, an absolute offset that
    # is 4% of a plateau mu and >100% of a starving one, so a low `T` is what stops
    # the head over-predicting slow media, and `T` = 0.003 *fixed* is worse than
    # 0.01 on every axis -- an optimisation failure, not a representation one.
    #
    # **Measured and it does not pay off, 2026-08-29.** On 3 organisms annealing
    # looked decisive (0.03 -> 0.003 took the low-mu bias +1.845 -> -0.009 with the
    # plateau intact). On all 21 it does not: bias +0.315, worst cosine 0.899, and
    # every §8.1 community worse than fixed T=0.01 (log-X 0.102 vs 0.047 at size
    # 21). Kept, off by default, so the next person measures something else --
    # the same 3-vs-21 trap the `icnn-u` `w_grad` frontier fell into.
    # `temp` is a static field, so each step retraces -- keep the stage count small.
    anneal_at = {}
    if gm_temp_final:
        t_hi = groupmax.DEFAULT_TEMP if gm_temp is None else gm_temp
        anneal_at = {
            round(epochs * (k + 1) / (_ANNEAL_STAGES + 1)): float(
                t_hi * (gm_temp_final / t_hi) ** ((k + 1) / _ANNEAL_STAGES)
            )
            for k in range(_ANNEAL_STAGES)
        }
    rng = np.random.default_rng(seed)
    t0 = time.time()
    for epoch in range(epochs):
        perm = rng.permutation(n)
        for s in range(steps_per_epoch):
            # The same media indices for every organism: the batch axis is media,
            # and all organisms share the design size (checked by the loader).
            idx = jnp.asarray(perm[s * batch : (s + 1) * batch])
            heads, opt_state, total, (v, gl) = _step(
                heads,
                opt_state,
                x[:, idx],
                mu[:, idx],
                g[:, idx],
                gvalid[:, idx],
                x_scale,
                gfloor,
                w_grad,
                w_rel,
                w_under,
                w_tau,
                a0,
                w_prox,
                optimiser,
                bvg,
            )
        if epoch + 1 in anneal_at:
            t_new = anneal_at[epoch + 1]
            heads = groupmax.with_temp(heads, t_new)
            # Adam's moments are themselves head-shaped pytrees, so they carry the
            # old temperature in their metadata and stop matching `heads`.
            opt_state = jax.tree.map(
                lambda z, t=t_new: (
                    groupmax.with_temp(z, t) if isinstance(z, groupmax.GroupMaxHead) else z
                ),
                opt_state,
                is_leaf=lambda z: isinstance(z, groupmax.GroupMaxHead),
            )
            LOGGER.info("epoch %4d  temperature -> %.4g", epoch, heads.temp)
        if epoch + 1 in reanchor_at:
            # Adam's moments for an overwritten plane describe the plane that used
            # to be there and would walk it straight back, so clear them.
            heads, slots = groupmax.reanchor(heads, ds, seed=seed + epoch)
            opt_state = jax.tree.map(
                lambda a, sl=slots, shape=heads.wx[0].shape[:2]: (
                    a.at[np.arange(len(sl))[:, None], sl].set(0.0)
                    if eqx.is_array(a) and a.shape[: sl.ndim] == shape
                    else a
                ),
                opt_state,
            )
            if a0 is not None:
                # A re-anchored plane IS a fresh label tangent, so it becomes its
                # own centre. Leaving the old anchor would pull the new cut back to
                # the dead one it replaced -- the two passes would fight.
                a0 = a0.at[np.arange(len(slots))[:, None], slots].set(
                    jax.nn.softplus(heads.wx[0])[np.arange(len(slots))[:, None], slots]
                )
            LOGGER.info("epoch %4d  re-anchored %d planes/organism", epoch, slots.shape[1])
        if epoch % 20 == 0 or epoch == epochs - 1:
            LOGGER.info(
                "epoch %4d  loss=%.5f  value=%.5f  grad=%.5f  (%.0fs)",
                epoch,
                float(total),
                float(v),
                float(gl),
                time.time() - t0,
            )
    # Gather back onto one device before returning. Sharding is an optimisation
    # internal to this function, and everything downstream (`evaluate`, `save`)
    # builds its own unsharded arrays from `ds` -- vmap rejects a sharded head
    # against a replicated input with "inconsistent axis specs: org vs None",
    # which is a crash *after* training, i.e. it costs the whole run.
    return jax.device_put(heads, jax.devices()[0])


# --------------------------------------------------------------------------- #
# Diagnostics (§7.3)
# --------------------------------------------------------------------------- #


def _over_media(fn, x, size: int = 256):
    """Map ``fn`` over the media axis in slices and rejoin — an OOM guard, not an
    optimisation.

    Training batches; evaluation did not. The held-out set is ``val_frac`` of the
    label budget, so at D10's 20000 media it is (21, 4000, 444) — 8x the default
    training batch on the axis the deepset prices *per metabolite*.

    Honest status: this is a precaution, not a measured fix. At the 4000-media
    laptop scale (800 held-out media) evaluation adds **0.00 GB** over the training
    peak, so what OOM-killed the sweep's deepset cells was the training step, not
    this. The slice keeps evaluation below the training batch as the held-out set
    grows 5x, which is the one extrapolation the laptop cannot check.
    """
    outs = [fn(x[:, i : i + size]) for i in range(0, x.shape[1], size)]
    return jax.tree.map(lambda *parts: jnp.concatenate(parts, axis=1), *outs)


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

    def bvc(xx):
        return _over_media(lambda c: bv(heads, c), xx)

    mid = bvc(lam * xa + (1 - lam) * xb)
    chord = lam[..., 0] * bvc(xa) + (1 - lam[..., 0]) * bvc(xb)
    return jnp.mean(mid < chord - tol, axis=1)


def _hessian_cond(heads, x, organism, n_points: int = 8, compose=None):
    """cond(Hessian) on the organism's own dims — predicts Phase-5 Newton failure.

    ``compose`` re-expresses the head as a function of the diagnostic coordinate,
    for a head whose concavity lives somewhere other than ``x``
    (:mod:`cfs.surrogate.picnn_u`).
    """
    conds = []
    for i in range(x.shape[0]):
        head = organism(heads, i)
        fn = head if compose is None else compose(head)
        dims = np.flatnonzero(np.asarray(head.mask))
        h = jax.vmap(jax.hessian(fn))(x[i, :n_points])[:, dims][:, :, dims]
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


def score(
    ds: ValueDataset, mu_hat, g_hat, extra: dict[str, dict] | None = None, arch: str = ""
) -> dict:
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
    r2 = 1.0 - jnp.sum((mu_hat - mu) ** 2, axis=1) / jnp.sum(
        (mu - mu.mean(1, keepdims=True)) ** 2, axis=1
    )
    err = jnp.abs(g_hat - g) * mask[:, None, :]

    # Share of predicted gradient norm landing on the *true* argmax metabolite.
    # 54.7% of rows have a single non-zero dual, so this is the number that says
    # why a cosine is what it is: a dense predictor scores near 1/|M_i| here even
    # when its cosine looks respectable.
    m = mask[:, None, :]
    top = jnp.take_along_axis(
        jnp.abs(g_hat * m), jnp.argmax(jnp.abs(g * m), axis=-1)[..., None], axis=-1
    )[..., 0]
    share = jnp.where(
        jnp.linalg.norm(g_hat * m, axis=-1) > 0,
        top / jnp.linalg.norm(g_hat * m + 1e-30, axis=-1),
        jnp.nan,
    )
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

    # Value error at *low* mu, which nothing else here can see: `value_r2` and the
    # MSE are absolute and 74% of held-out rows sit on the plateau, so a head can
    # score R2 0.99 while over-predicting a starving medium by +98% (measured,
    # `value_ra3`). §8 integrates `d(log X)/dt = mu`, so it is the relative error on
    # the slow members that sets the composition's log-X error -- the M5 runs whose
    # log-X error was 0.12-0.32 are exactly the ones containing a slow organism.
    low = np.asarray(mu) < 0.25 * np.asarray(mu).max(axis=1, keepdims=True)
    relerr = np.asarray((mu_hat - mu) / jnp.maximum(mu, 1e-9))

    per = {}
    for i, gid in enumerate(ds.genome_ids):
        worst = np.argsort(-np.asarray(err[i].mean(axis=0)))[:5]
        by_met = {}
        for j in np.unique(kt[i][okv[i]]):
            s = okv[i] & (kt[i] == j)
            by_met[ds.exchanges[j]] = {
                "rows": int(s.sum()),
                "grad_cosine": float(np.nanmean(cos_np[i][s])),
            }
        per[gid] = {
            "grad_cosine": float(jnp.nanmean(cos[i])),
            "grad_cosine_p05": float(jnp.nanpercentile(cos[i], 5)),
            "grad_top1_share": float(jnp.nanmean(share[i])),
            "value_r2": float(r2[i]),
            "value_rel_err_low_mu": float(np.median(np.abs(relerr[i][low[i]]))),
            "value_bias_low_mu": float(np.median(relerr[i][low[i]])),
            # The one-sided invariant, as a rate. `mu_max` is concave and the head
            # is a min of affine pieces, so `mu_hat < mu` at a labelled row is a
            # *provable* violation -- this is the coverage `--w-tau` buys, and it is
            # measurable per checkpoint instead of after a 5 h composition run.
            "value_under_rate": float((relerr[i] < 0).mean()),
            "value_under_rate_low_mu": float((relerr[i][low[i]] < 0).mean()),
            **extra.get(gid, {}),
            "worst_grad_metabolites": [ds.exchanges[j] for j in worst],
            "per_limiting_metabolite": by_met,
        }
    gate = min(v["grad_cosine"] for v in per.values())
    # The split provenance is part of the result: two runs are only comparable if
    # they were scored on the same held-out media, and the §4.6 rounds are exactly
    # what can silently change that (see `load_value_dataset`).
    return {
        "gate": "grad_cosine > 0.99, in u = c/(Km+c) space",
        "worst_grad_cosine": gate,
        "passed": bool(gate > GRAD_COSINE_GATE),
        "arch": arch,
        "n_val_media": int(ds.x_val.shape[1]),
        "n_train_media": int(ds.x_train.shape[1]),
        "rounds_present": ds.rounds_present,
        "per_organism": per,
    }


def evaluate(
    heads: eqx.Module, ds: ValueDataset, seed: int = 0, arch: str = "icnn", cal=None
) -> dict:
    """Held-out diagnostics per organism (§7.3). The gate is ``grad_cosine``."""
    mod = _ARCH[arch]
    x = jnp.asarray(ds.x_val)
    mu_hat, g_hat = _over_media(lambda xx: mod.batched_value_and_grad(heads, xx), x)
    if cal is not None:
        # A positive per-row scalar on the gradient, so `grad_cosine` is untouched
        # and only the value diagnostics move (`cfs.surrogate.calibrate`).
        mu_hat = jnp.asarray(calibrate.apply(mu_hat, cal))
    # In u space (see `_du`), so the gate means the same thing across runs.
    g_hat = g_hat * _du(x, jnp.asarray(ds.x_scale))
    # Concavity and conditioning are only meaningful in the coordinate the head is
    # actually concave in. `icnn-u` is concave in w = u/s, and reads 98% violating
    # with cond ~1e32 if these are taken in x. Heads without the hook are concave
    # in x and see the identity, so every existing arch is byte-identical.
    to_diag = getattr(mod, "to_diag", None)
    if to_diag is None:
        x_diag, bv, compose = x, mod.batched_value, None
    else:
        x_diag, bv, compose = to_diag(x), mod.batched_value_diag, mod.head_in_diag
    viol = _concavity_violations(heads, x_diag, jax.random.PRNGKey(seed), bv)
    conds = _hessian_cond(heads, x_diag, mod.organism, compose=compose)
    extra = {
        gid: {"concavity_violation_rate": float(viol[i]), "hessian_cond_median": conds[i]}
        for i, gid in enumerate(ds.genome_ids)
    }
    return score(ds, mu_hat, g_hat, extra, arch=arch)


# --------------------------------------------------------------------------- #
# Checkpoint (P13 / P14)
# --------------------------------------------------------------------------- #


def _identity_cal(n: int) -> np.ndarray:
    """``(d0, beta, a) = (0, 1, 1)``. ``beta`` is 1 rather than 0 because it is a
    divisor: at 0 the identity is only saved from a division by zero by a 1e-12
    floor, and the resulting overflow makes `calibrate.apply` return NaN for a
    negative raw prediction (`0 * inf`). `d0 = 0` makes the term vanish either
    way, so any positive `beta` is the identity."""
    c = np.zeros((n, 3))
    c[:, 1:] = 1.0
    return c


def save(
    heads: eqx.Module,
    ds: ValueDataset,
    outdir: Path,
    arch: dict,
    diagnostics: dict,
    cal=None,
) -> None:
    """Serialise the stacked heads plus everything needed to use them again.

    The metadata is not optional: the input transform, ``Vmax``, the label scale
    and the ``index_hash`` are what stop a checkpoint being silently reinterpreted
    against a different metabolite index or a different unit convention.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(outdir / "value_heads.eqx", heads)
    (outdir / "value_heads.json").write_text(
        json.dumps(
            {
                "index_hash": ds.index_hash,
                "genome_ids": ds.genome_ids,
                "exchanges": ds.exchanges,
                "mask": ds.mask.astype(int).tolist(),
                "mu_scale": ds.mu_scale.tolist(),
                "value_cal": (_identity_cal(len(ds.genome_ids)) if cal is None else cal).tolist(),
                "x_scale": ds.x_scale.tolist(),
                # §13.11 stage 4'. `exchanges` is 2M long under inhibition
                # ([u | theta]); these two are what let a consumer split it and
                # rebuild the theta channel from a medium.
                "n_metabolites": ds.n_metabolites or len(ds.exchanges),
                "ceq": ds.ceq,
                "input_transform": getattr(
                    _ARCH.get(arch.get("arch", "icnn")),
                    "INPUT_TRANSFORM",
                    "u = c / (Km + c), x = u / (u + x_scale); Km from km_defaults.yaml",
                ),
                "gradient_units": (
                    "d(mu_max)/dx = max(-shadow, 0) * Vmax * (u + x_scale)^2 / x_scale, Vmax = 1000"
                ),
                "arch": arch,
            },
            indent=2,
        )
    )
    (outdir / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2))


def load(outdir: Path, width: int = 128, depth: int = 3) -> tuple[eqx.Module, dict]:
    """Reload a saved stacked head and its metadata."""
    outdir = Path(outdir)
    meta = json.loads((outdir / "value_heads.json").read_text())
    arch = meta.get("arch", {})
    like = _build(
        arch.get("arch", "icnn"),
        jax.random.PRNGKey(0),
        len(meta["genome_ids"]),
        len(meta["exchanges"]),
        np.array(meta["mask"], dtype=bool),
        arch.get("width", width),
        arch.get("depth", depth),
        arch.get("emb_dim", 8),
        phi_hidden=arch.get("phi_hidden"),
        k_code=arch.get("k_code"),
        gm_group=arch.get("gm_group"),
        gm_temp=arch.get("gm_temp"),
    )
    return eqx.tree_deserialise_leaves(outdir / "value_heads.eqx", like), meta


def run(
    labels_dir: Path,
    index_path: Path,
    outdir: Path,
    *,
    eps: float = 1e-3,
    arch: str = "icnn",
    width: int = 128,
    depth: int = 3,
    epochs: int = 400,
    batch: int = 512,
    lr: float = 3e-3,
    w_grad: float = 1.0,
    w_rel: float = 0.0,
    w_under: float = 0.0,
    w_tau: float = 0.5,
    w_prox: float = 0.0,
    emb_dim: int = 8,
    phi_hidden: int | None = None,
    k_code: int | None = None,
    gm_group: int | None = None,
    gm_temp: float | None = None,
    gm_init: str | None = None,
    gm_reanchor: int = 0,
    gm_select: str = "active-set",
    gm_trial_media=None,
    gm_temp_final: float | None = None,
    gm_repair: bool = False,
    gm_eval_temp: float | None = None,
    seed: int = 0,
    organisms: list[str] | None = None,
    x_scale_from: Path | None = None,
) -> dict:
    """Load labels, train, evaluate, checkpoint. Returns the diagnostics.

    ``organisms`` restricts the stack (default: every shard under ``labels_dir``).
    One organism per stack is the sweep's default -- see `load_value_dataset`.
    """
    # §8.6f trap 1: pin the input coordinate to an existing checkpoint's, so
    # rows added by a fallback round do not move `x` under the head that is
    # being extended (P14). Without this no incremental loop is comparable.
    pin = None
    if x_scale_from is not None:
        import json as _json

        for name in ("value_heads.json", "behaviour_heads.json"):
            cand = Path(x_scale_from) / name
            if cand.exists():
                pin = np.asarray(_json.loads(cand.read_text())["x_scale"], dtype=float)
                break
        if pin is None:
            raise ValueError(f"no heads json with an x_scale in {x_scale_from}")
    ds = load_value_dataset(
        labels_dir, index_path, eps=eps, seed=seed, organisms=organisms, x_scale=pin
    )
    heads = train_value_heads(
        ds,
        arch=arch,
        width=width,
        depth=depth,
        epochs=epochs,
        batch=batch,
        lr=lr,
        w_grad=w_grad,
        w_rel=w_rel,
        w_under=w_under,
        w_tau=w_tau,
        w_prox=w_prox,
        emb_dim=emb_dim,
        phi_hidden=phi_hidden,
        k_code=k_code,
        gm_group=gm_group,
        gm_temp=gm_temp,
        gm_init=gm_init,
        gm_reanchor=gm_reanchor,
        gm_select=gm_select,
        gm_trial_media=gm_trial_media,
        gm_temp_final=gm_temp_final,
        seed=seed,
    )
    # Training needs a soft argmax for gradient to reach every plane; *inference*
    # does not, and the smoothing is what the low-`mu` floor is made of. The gap
    # is `c*T*ln(n_active)`, and `repair_intercepts` compensates it with one
    # uniform lift = the **max** over training rows, so anywhere fewer planes are
    # active than at that maximum -- which is exactly a starved medium, where one
    # plane is active -- the lift is uncancelled and the head reads high by a
    # constant. Measured on the n=1 titration: the residual is proportional to `T`,
    # 0.0107 -> 0.0011 -> 0.0001 `mu_scale` units over T 1e-2/1e-3/1e-4, and every
    # held-out axis improves with it (worst grad cosine 0.909 -> 0.952, low-`mu`
    # bias +0.046 -> +0.0004, under-rate still 0). Dropping the temperature for
    # inference costs curvature (P3), which `cfs master-jacobian` already showed
    # does not reach §8.4 -- the supply term sets the conditioning, not the head.
    if gm_eval_temp is not None:
        heads = groupmax.with_temp(heads, gm_eval_temp)
    # Before the calibration is fit, so the fit sees the head it will ship with.
    if gm_repair:
        heads = groupmax.repair_intercepts(heads, ds)
    mu_tr = _over_media(lambda xx: _ARCH[arch].batched_value(heads, xx), jnp.asarray(ds.x_train))
    # The calibration is a 1-D post-hoc fit and the identity is a valid value for it,
    # so it must never be able to discard a finished training run -- which it did
    # once, losing 813 s to an import error inside `calibrate.fit`. On the cluster
    # that would be hours.
    if gm_repair:
        # The two corrections fight. `repair_intercepts` guarantees `mu_hat >= mu`
        # on the training rows; `calibrate` is a least-squares fit on those same
        # rows and its map is downward, so it pulls the head straight back under
        # them -- measured end to end: `value_under_rate_low_mu` 0.000 -> 0.977 on
        # an otherwise-exact repaired head. A repaired head is already unbiased
        # (n=1 titration residual 1e-4 `mu_scale` units), so there is nothing left
        # for a 1-D output map to buy, and it can only break the invariant.
        cal = _identity_cal(len(ds.genome_ids))
    else:
        try:
            cal = calibrate.fit(np.asarray(mu_tr), ds.mu_train / ds.mu_scale[:, None])
        except Exception:
            LOGGER.exception("calibration fit failed — shipping the identity instead")
            cal = _identity_cal(len(ds.genome_ids))
    diagnostics = evaluate(heads, ds, seed=seed, arch=arch, cal=cal)
    meta = {
        "arch": arch,
        "width": width,
        "depth": depth,
        "epochs": epochs,
        "lr": lr,
        "w_grad": w_grad,
        "w_rel": w_rel,
        "w_under": w_under,
        "w_tau": w_tau,
        "w_prox": w_prox,
        "eps": eps,
        "seed": seed,
    }
    if arch == "groupmax-u":
        meta["gm_group"] = gm_group or groupmax.DEFAULT_GROUP
        # The shipped temperature, so `load` reconstructs the head that was scored.
        meta["gm_temp"] = (
            gm_eval_temp
            if gm_eval_temp is not None
            else (groupmax.DEFAULT_TEMP if gm_temp is None else gm_temp)
        )
        meta["gm_train_temp"] = groupmax.DEFAULT_TEMP if gm_temp is None else gm_temp
        meta["gm_init"] = gm_init or "random"
        meta["gm_reanchor"] = gm_reanchor
        meta["gm_select"] = gm_select
        meta["gm_repair"] = gm_repair
        meta["gm_trial_media"] = str(gm_trial_media) if gm_trial_media else None
        if gm_temp_final:
            meta["gm_temp_final"] = gm_temp_final
    if arch.startswith("deepset"):
        # Only when set: an absent key is what makes `load` fall back to the
        # width-derived default, so a checkpoint written before these flags existed
        # still reconstructs.
        meta["emb_dim"] = emb_dim
        if phi_hidden is not None:
            meta["phi_hidden"] = phi_hidden
        if k_code is not None:
            meta["k_code"] = k_code
    save(heads, ds, outdir, meta, diagnostics, cal=cal)
    return diagnostics
