"""§8.6g(3) — trajectory-level training: fit the endpoint, not the state.

Head B is fitted per state against label fluxes, and six times on file a strictly
better right-hand side has failed to move the batch endpoint the M5 gate measures.
The premise check says why, and says this is worth building
(`20hm_bands/traj_sens{,2}.py`, no LP solves, 20 cells):

* the endpoint is **smooth and monotone** in `z` on 20/20 cells -- the pool clip
  in :func:`cfs.compose.dfba.integrate` is a subgradient, not a wall, so nothing
  blocks a gradient;
* **17 of 20** cells could close their entire endpoint error with a <= 10%
  relative move in `z`, and the linearisation holds at two probe sizes;
* the endpoint is **~3 orders more sensitive to the direction of the `z` error
  than to its magnitude** (`||g||` 0.067 uniform against 272 full-space). A
  per-state loss divided by `z_scale` spends itself on magnitude, which is
  exactly the half the endpoint ignores. That is the mechanism this module
  exists to route around.

So: re-integrate the surrogate in JAX, compare `log X` against the *stored true*
trajectory at every step, and backprop into Head B. Head A is frozen -- its
`mu_rel_median` is <= 5e-4 on every §8.1 cell, and the one community this cannot
fix is a Head A mid-`mu` over-prediction, measured.

Two things the premise check hands over and this module obeys:

* the loss is over the **whole trajectory**, not the endpoint alone. `||g||`
  reaches ~300 through 40 Euler steps, and every intermediate step is a supervised
  point that costs nothing extra;
* gradients are **clipped** by global norm, for the same reason.

**The training communities must not be the benchmark's.** `--runs` takes
`cfs community` output directories; use the 16 n=15 communities, never the 10 the
M5 numbers are quoted on. The same discipline as §8.6d's round 2.

The §8.6g(2) elemental projection is *not* applied here: it is an inference-time
correction whose active set is discrete, and it can only reduce the error of
whatever comes out, so training under it would be optimising through a projection
the composition re-applies anyway.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

LOGGER = logging.getLogger("cfs.surrogate.traj")


def _cells(runs: list[Path]) -> list[dict]:
    """Every (community, medium draw) in ``runs`` as a training trajectory."""
    out = []
    for run in runs:
        run = Path(run)
        cells = json.loads((run / "community.json").read_text())["communities"]
        for n, cell in enumerate(cells):
            hits = sorted(run.glob(f"trajectory_{n}_*.npz"))
            if not hits:
                LOGGER.warning("%s cell %d has no trajectory file — skipped", run, n)
                continue
            z = np.load(hits[0], allow_pickle=True)
            out.append(
                {
                    "genome_ids": list(cell["genome_ids"]),
                    "dt": float(cell["dt"]),
                    "steps": int(cell["steps"]),
                    "c0": z["c_true"][0].astype(np.float64),
                    "x_true": z["x_true"].astype(np.float64),
                    "run": run.name,
                    "cell": n,
                }
            )
    return out


def _forward(sur, members: np.ndarray):
    """A JAX replica of ``dfba.rhs_surrogate`` + ``integrate`` for one community.

    Mirrors the numpy path exactly -- same MM clamp, same pool clip at zero, same
    ``X exp(dt mu)`` biomass update -- so a trajectory scored here and one scored
    by ``cfs community`` are the same map. The elemental projection is left out
    (see the module docstring).
    """
    from cfs.surrogate import behaviour as B

    km = jnp.asarray(sur.km)
    x_scale = jnp.asarray(sur.x_scale)
    mu_scale = jnp.asarray(sur.mu_scale)
    z_scale = jnp.asarray(sur.z_scale)
    mask = jnp.asarray(sur.mask)
    floor = jnp.asarray(sur.mu_floor)
    alpha = jnp.ones((len(sur.genome_ids), 1), dtype=jnp.float32)
    mem = jnp.asarray(members)
    vheads, mod = sur._vheads, sur.mod

    def step(bheads, c, X):
        u = c / (km + c)
        x = (u / (u + x_scale))[:, None, :].astype(jnp.float32)
        mu = jnp.maximum(mod.batched_value(vheads, x)[:, 0] * mu_scale, 0.0)  # P2
        z = B.flux(bheads, x, alpha, z_scale, jnp.maximum(mu, floor)[:, None])[:, 0]
        z = jnp.maximum(z, -B.VMAX * u) * mask  # §3.3
        return (X[:, None] * z[mem]).sum(0), mu[mem]

    def run(bheads, c0, x0, dt, steps):
        def one(carry, _):
            c, X = carry
            dc, mu = step(bheads, c, X)
            c = jnp.maximum(c + dt * dc, 0.0)
            X = X * jnp.exp(dt * mu)
            return (c, X), X

        _, xs = jax.lax.scan(one, (c0, x0), None, length=steps)
        return xs  # (steps, members), matching x_true[1:]

    return run


def _anchor(params, initial, weight: float):
    """``weight`` x mean squared parameter drift, normalised **per leaf**.

    32 trajectories cannot support unconstrained fine-tuning of a 256x3 stack:
    measured, 60 epochs at lr 1e-4 buy 9% of the trajectory loss and take held-out
    label R2 from 0.577 to **-31.98**, with both community gates 2.5-7x worse. So
    the fine-tune is a *move* from the label-fitted head, not a fit from scratch.

    Per leaf, not one global ratio: Head B's weight arrays differ in scale by
    orders of magnitude, and `--w-prox` already demonstrated that a single global
    normalisation over a heavy-tailed quantity is set by its largest members and
    is ~0 for everything else.
    """
    if not weight:
        return 0.0
    total = 0.0
    for a, b in zip(
        jax.tree.leaves(eqx.filter(params, eqx.is_inexact_array)),
        jax.tree.leaves(eqx.filter(initial, eqx.is_inexact_array)),
        strict=True,
    ):
        total = total + jnp.mean((a - b) ** 2) / jnp.maximum(jnp.mean(b**2), 1e-12)
    return weight * total


def _loss_fn(sur, cell: dict, initial=None, w_anchor: float = 0.0):
    """Mean squared ``log X`` error over the whole trajectory, for one cell."""
    members = np.array([sur.genome_ids.index(g) for g in cell["genome_ids"]])
    run = _forward(sur, members)
    c0 = jnp.asarray(cell["c0"])
    x0 = jnp.asarray(cell["x_true"][0])
    # A member the truth never grows carries no information and divides by zero.
    live = cell["x_true"][-1] > cell["x_true"][0] * (1 + 1e-9)
    target = jnp.asarray(np.log(np.maximum(cell["x_true"][1:], 1e-300)))
    keep = jnp.asarray(live.astype(np.float64))
    dt, steps = cell["dt"], cell["steps"]

    def loss(bheads):
        xs = run(bheads, c0, x0, dt, steps)
        err = (jnp.log(jnp.maximum(xs, 1e-300)) - target) ** 2
        traj = (err * keep).sum() / jnp.maximum(keep.sum() * steps, 1.0)
        return traj + _anchor(bheads, initial, w_anchor)

    return loss


def run(
    value_dir: Path,
    behaviour_dir: Path,
    runs: list[Path],
    out: Path,
    *,
    epochs: int = 20,
    lr: float = 1e-5,
    clip: float = 1.0,
    w_anchor: float = 0.0,
    seed: int = 0,
) -> dict:
    """Fine-tune Head B on stored community trajectories. Writes a checkpoint.

    The output is a copy of ``behaviour_dir`` with ``behaviour_heads.eqx``
    replaced: only the weights move, so ``z_scale`` / ``x_scale`` / ``mu_floor``
    stay exactly what the labels set and P13/P14 still hold against Head A.
    """
    from cfs.compose.dfba import Surrogate

    sur = Surrogate(value_dir, behaviour_dir)
    cal = np.asarray(sur.value_cal)
    if not (np.allclose(cal[:, 0], 0.0) and np.allclose(cal[:, 2], 1.0)):
        # The JAX replica leaves the output calibration out; a repaired head forces
        # the identity, so this only bites on a checkpoint that predates §8.6c.
        raise ValueError("Head A carries a non-identity calibration; not replicated here")

    cells = _cells([Path(r) for r in runs])
    if not cells:
        raise ValueError("no trajectories found in --runs")
    LOGGER.info(
        "%d trajectories over %d communities, sizes %s",
        len(cells),
        len({tuple(c["genome_ids"]) for c in cells}),
        sorted({len(c["genome_ids"]) for c in cells}),
    )
    initial = sur._bheads
    losses = [
        (c, eqx.filter_jit(eqx.filter_value_and_grad(_loss_fn(sur, c, initial, w_anchor))))
        for c in cells
    ]

    params = sur._bheads
    opt = optax.chain(optax.clip_by_global_norm(clip), optax.adam(lr))
    state = opt.init(eqx.filter(params, eqx.is_inexact_array))
    order = np.random.default_rng(seed)
    history = []
    for epoch in range(epochs):
        total, seen, skipped = 0.0, 0, 0
        for i in order.permutation(len(losses)):
            cell, fn = losses[i]
            value, grad = fn(params)
            # A single non-finite gradient is permanent damage, not a bad step:
            # `clip_by_global_norm` puts the NaN into the global norm and Adam's
            # moments carry it forever, so every later cell reads non-finite too.
            # It happens on starved states, where Head A's softmin at the shipped
            # `gm_eval_temp` (1e-4) amplifies by ~1/T in float32. Skip the update.
            leaves = jax.tree.leaves(eqx.filter(grad, eqx.is_inexact_array))
            if not np.isfinite(float(value)) or not all(
                bool(jnp.isfinite(a).all()) for a in leaves
            ):
                skipped += 1
                LOGGER.debug(
                    "non-finite loss/grad on %s cell %d — skipped", cell["run"], cell["cell"]
                )
                continue
            updates, state = opt.update(grad, state, eqx.filter(params, eqx.is_inexact_array))
            params = eqx.apply_updates(params, updates)
            total += float(value)
            seen += 1
        # Averaging over *all* cells would report a diverged epoch, where every
        # trajectory is non-finite and skipped, as a loss of exactly 0 -- which is
        # what a too-large `lr` produces, and it reads as a perfect fit.
        if seen == 0:
            raise RuntimeError(
                f"every trajectory was non-finite at epoch {epoch}: the run has "
                f"diverged (lr={lr}). The weights are already damaged; lower --lr."
            )
        history.append(total / seen)
        LOGGER.info(
            "epoch %3d  mean trajectory loss %.6g  (%d/%d cells, %d skipped)",
            epoch,
            history[-1],
            seen,
            len(losses),
            skipped,
        )

    out = Path(out)
    if out.resolve() != Path(behaviour_dir).resolve():
        out.mkdir(parents=True, exist_ok=True)
        for name in ("behaviour_heads.json", "behaviour_diagnostics.json", "reference_x.npz"):
            src = Path(behaviour_dir) / name
            if src.exists():
                shutil.copy2(src, out / name)
    eqx.tree_serialise_leaves(out / "behaviour_heads.eqx", params)
    report = {
        "source": str(behaviour_dir),
        "runs": [str(r) for r in runs],
        "n_trajectories": len(cells),
        "epochs": epochs,
        "lr": lr,
        "clip": clip,
        "w_anchor": w_anchor,
        "loss_first": history[0] if history else None,
        "loss_last": history[-1] if history else None,
        "history": history,
    }
    (out / "traj_finetune.json").write_text(json.dumps(report, indent=2))
    return report
