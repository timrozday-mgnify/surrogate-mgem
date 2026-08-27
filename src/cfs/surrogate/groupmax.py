"""Head A as a *smoothed GroupMax* network over ``w`` — kinks as the primitive.

Every concave head measured so far builds its kinks out of smooth ridges: the
activation is ``softplus``, so a corner is a non-negative sum of many soft bends.
The target does not look like that. ``mu_max(u) = min_k (a_k . u + c_k)`` is
piecewise **linear** with axis-aligned corners — a metabolite starts limiting
abruptly — and per organism the labels contain ~1700 distinct active sets. Depth
compounds smoothness rather than manufacturing sharpness, which is why width
128->1024 and depth 3->6 moved paired cosine and R2 by +-0.001.

This head changes the *activation*, which is the thing that sets what the class
can represent (a constrained MLP whose activation is monotone convex approximates
only that family). The activation here is a **group max**: reshape the
pre-activation to ``(width, group)`` and reduce the last axis. A max of affine
functions is exactly the target's form, so one corner costs one unit instead of
many.

```
z_1     = gmax(-softplus(W_x^0) w + b_0)                   (width*g,) -> (width,)
z_{k+1} = gmax(softplus(W_z^k) z_k - softplus(W_x^k) w + b_k)
out     = softplus(o_z) . z_L - softplus(o_x) . w + o_b
mu_hat  = -out
```

Convexity of ``out`` in ``w`` survives for the usual two reasons (Amos et al.
2017), with the group max standing in for softplus: it is convex and
non-decreasing in each argument, and the pass-through weights are non-negative.
The sign on the ``w`` skips makes ``mu_hat`` non-decreasing. Both are structural,
so ``concavity_violation_rate`` must read exactly 0.

**The max is smoothed, and the temperature is the point.** A hard max has zero
Hessian inside every piece — P3 exactly ("Newton stalls or NaNs, gradients look
fine"), and the failure mode the parameter-free cutting-plane model cannot escape.
``T * logsumexp(a / T)`` is convex, non-decreasing, C-infinity and tends to the max
as ``T -> 0``, with curvature scaling as ``1/T``. So ``T`` is an **explicit
conditioning knob** — the accuracy/curvature trade §8's Newton pays for becomes a
swept axis instead of an emergent number. It is a fixed hyperparameter, not a
learned one: a learned temperature collapses toward the hard max (measured: median
Hessian condition 1e32), which is the one place the composition cannot follow.

``DEFAULT_TEMP`` is **measured, on the pruned label-tangent model with no optimiser
in the way** (100 active-set-ranked planes, held-out media, 2 organisms):

| ``T`` | CR626927.1 cos / R2 | ABCC02 cos / R2 | curvature |
| --- | --- | --- | --- |
| 0 (hard min) | 0.9567 / 0.9964 | 0.9611 / 0.9969 | **exactly 0** |
| 0.01 | 0.9508 / 0.9964 | 0.9635 / 0.9970 | non-zero |
| 0.03 | 0.9498 / 0.9963 | 0.9562 / 0.9973 | non-zero |
| 0.1 | 0.9229 / 0.9875 | 0.9291 / 0.9934 | non-zero |
| 0.3 | 0.8327 / 0.7398 | 0.7941 / 0.7917 | non-zero |

So there is a window, ``T`` ~ 0.01-0.03, that keeps essentially all of the hard
min's accuracy *and* buys the curvature §8 needs; 0.1 is already past the knee and
0.3 collapses. A trained head's pre-activations need not sit on the label scale, so
treat this as a prior on the axis rather than a transferred optimum.

The design **nests max-affine exactly**: ``width=1, depth=1, group=K`` is
``min_k(a_k . w + c_k)`` and nothing else. Wider and deeper generalises it.

Input coordinate is ``w = min(u/s, W_CAP)`` from :mod:`cfs.surrogate.picnn_u`,
for the reason documented there — the target is concave in ``u`` and *not* in
``x``, so a head constrained in ``x`` cannot represent it.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from cfs.surrogate.picnn import _softplus_inv
from cfs.surrogate.picnn_u import INPUT_TRANSFORM, W_CAP, to_diag  # noqa: F401

DEFAULT_GROUP = 8
DEFAULT_TEMP = 0.03


class GroupMaxHead(eqx.Module):
    """Concave, non-decreasing ``mu_max(w)`` with a smoothed group-max activation."""

    wx: list[Array]  # (width*g, M) input skips, non-positive via -softplus
    wz: list[Array]  # (width*g, width) pass-through, non-negative via softplus
    b: list[Array]  # (width*g,)
    out_z: Array  # (width,) non-negative via softplus
    out_x: Array  # (M,) non-positive via -softplus
    out_b: Array
    mask: Array
    group: int = eqx.field(static=True)
    temp: float = eqx.field(static=True)

    def __init__(
        self,
        key,
        n_in: int,
        mask,
        width: int = 128,
        depth: int = 3,
        group: int = DEFAULT_GROUP,
        temp: float = DEFAULT_TEMP,
    ):
        keys = jax.random.split(key, 3 * depth + 3)
        h = width * group
        # Scale-aware, as in `picnn_u`: `w` runs to W_CAP with ~45% of cells at the
        # far end, and the plain sqrt(2/n_in) init starts the head saturated in the
        # activation's linear regime with no curvature to train against.
        scale = jnp.sqrt(2.0 / n_in) / (W_CAP / 2.0)
        self.wx = [
            _softplus_inv(jnp.abs(jax.random.normal(keys[i], (h, n_in)) * scale) + 1e-9)
            for i in range(depth)
        ]
        # The pass-through is a non-negative sum over `width` units, so centre it at
        # softplus^-1(1/width): without it the output compounds to O(100) at init.
        w0 = _softplus_inv(1.0 / width)
        self.wz = [
            jax.random.normal(keys[depth + i], (h, width)) * 0.1 + w0 for i in range(1, depth)
        ]
        # Diverse biases: with every unit at the same offset the group max is
        # decided by one unit everywhere and the layer collapses to one affine piece.
        self.b = [
            jax.random.uniform(keys[2 * depth + i], (h,), minval=-1.0, maxval=1.0)
            for i in range(depth)
        ]
        self.out_z = jax.random.normal(keys[-1], (width,)) * 0.1 + w0
        self.out_x = _softplus_inv(jnp.abs(jax.random.normal(keys[-2], (n_in,)) * scale) + 1e-9)
        self.out_b = jnp.zeros(())
        self.mask = jnp.asarray(mask, dtype=bool)
        self.group = int(group)
        self.temp = float(temp)

    def _gmax(self, a: Array) -> Array:
        """``(width*g,) -> (width,)``: smooth max within each group.

        Convex and non-decreasing in every argument for any ``T > 0``, which is
        what keeps the composition convex; ``-> max`` as ``T -> 0``.
        """
        return self.temp * jax.nn.logsumexp(a.reshape(-1, self.group) / self.temp, axis=-1)

    def on_w(self, w: Array) -> Array:
        """The head in its own coordinate — no input map applied."""
        y = w * self.mask
        z = self._gmax(-jax.nn.softplus(self.wx[0]) @ y + self.b[0])
        for wx, wz, b in zip(self.wx[1:], self.wz, self.b[1:], strict=True):
            z = self._gmax(jax.nn.softplus(wz) @ z - jax.nn.softplus(wx) @ y + b)
        out = jax.nn.softplus(self.out_z) @ z - jax.nn.softplus(self.out_x) @ y + self.out_b
        return -out

    def __call__(self, x: Array) -> Array:
        return self.on_w(to_diag(x))


def stack_heads(
    key,
    n_organisms: int,
    n_in: int,
    mask,
    width: int = 128,
    depth: int = 3,
    group: int = DEFAULT_GROUP,
    temp: float = DEFAULT_TEMP,
) -> GroupMaxHead:
    """One :class:`GroupMaxHead` PyTree with a leading organism axis (§6.1)."""
    keys = jax.random.split(key, n_organisms)
    make = eqx.filter_vmap(
        lambda k, m: GroupMaxHead(k, n_in, m, width, depth, group, temp), in_axes=(0, 0)
    )
    return make(keys, jnp.asarray(mask, dtype=bool))


def organism(heads: GroupMaxHead, i: int) -> GroupMaxHead:
    """Slice organism ``i`` out of a stacked head."""
    return jax.tree.map(lambda p: p[i] if eqx.is_array(p) else p, heads)


def head_in_diag(head: GroupMaxHead):
    """The head as a function of ``w``. See :func:`cfs.surrogate.picnn_u.head_in_diag`
    for why the diagnostics must not reach ``w`` by inverting the input map."""
    return head.on_w


@eqx.filter_vmap(in_axes=(0, 0))
def batched_value(heads: GroupMaxHead, x: Array) -> Array:
    """``(G, B, M) -> (G, B)``."""
    return jax.vmap(heads)(x)


@eqx.filter_vmap(in_axes=(0, 0))
def batched_value_diag(heads: GroupMaxHead, w: Array) -> Array:
    """``batched_value`` in the concavity coordinate — takes ``w``, not ``x``."""
    return jax.vmap(head_in_diag(heads))(w)


@eqx.filter_vmap(in_axes=(0, 0))
def batched_value_and_grad(heads: GroupMaxHead, x: Array):
    """``(G, B, M) -> ((G, B), (G, B, M))`` — the gradient IS the shadow price."""
    return jax.vmap(jax.value_and_grad(heads))(x)


# --------------------------------------------------------------------------- #
# Label-tangent initialisation
# --------------------------------------------------------------------------- #


def rank_by_active_set(g_w: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Row indices ordered by how common their dual's support pattern is.

    A row's non-zero duals are its LP basis — the *active set* — and a handful of
    patterns carry most of the media (2-4 cover 50% of rows, 15-33 cover 80%).
    Taking one representative of the commonest pattern, then one of the next, and
    so on, spends a small budget of planes on the regimes that actually occur.

    Measured against drawing the same number of tangents uniformly at random, on
    held-out media (cutting-plane model, 2 organisms):

    | K | CR626927.1 random -> ranked | ABCC02 random -> ranked |
    | --- | --- | --- |
    | 50 | 0.885 -> **0.938** | 0.841 -> **0.950** |
    | 100 | 0.918 -> **0.957** | 0.905 -> **0.961** |
    | 250 | 0.951 -> 0.959 | 0.945 -> **0.969** |

    ~20x fewer planes for the same accuracy: 100 ranked tangents match ~2000
    random ones.
    """
    ok = np.flatnonzero(valid)
    buckets: dict[tuple, list[int]] = {}
    for j in ok:
        buckets.setdefault(tuple(np.flatnonzero(g_w[j] > 0)), []).append(int(j))
    order = sorted(buckets, key=lambda p: -len(buckets[p]))
    out, r = [], 0
    while len(out) < len(ok):
        add = [buckets[p][r] for p in order if r < len(buckets[p])]
        if not add:
            break
        out += add
        r += 1
    return np.asarray(out, dtype=int)


def init_from_tangents(heads: GroupMaxHead, ds, seed: int = 0) -> GroupMaxHead:
    """Seed the first layer's units with real supporting hyperplanes of ``mu_max``.

    Every labelled row is an *exact* tangent of the target — ``mu_max`` is concave
    in ``u``, so ``(w_j, mu_j, pi_j)`` gives a supporting hyperplane and any set of
    them is a valid concave upper bound. The first layer holds ``width * group``
    affine units, so it can simply be *told* what they are instead of discovering
    them from random noise.

    This matters because random init demonstrably does not find them. Measured on
    CR626927.1 at ``w_grad`` 10, same architecture and the same ``T = 0.1``: the
    pruned tangent model scores held-out cosine **0.923**, and the identical head
    trained from random init scores **0.712**. A 0.21 gap at identical class and
    temperature is an optimisation gap — the failure that
    LSPA/CAP-style max-affine fitting exists to fix, showing up here despite the
    softmax weights all being strictly positive.

    **Exact only at ``width=1, depth=1``**, where the head *is* ``min_k(a_k.w + c_k)``
    and this reproduces the tangent model outright (``out_z`` -> 1, ``out_x`` -> 0).
    Wider or deeper, the first layer is still seeded with real duals — correct
    subspace, correct scale, kinks on real kinks — but the head starts as a
    non-negative sum of group-wise minima rather than one global minimum, so it is
    a warm start and not a reproduction. The docstring says so because the
    difference is measurable and someone will otherwise assume the exact case.
    """
    n_slots = heads.wx[0].shape[1]
    G, _, M = ds.x_train.shape
    wx0 = np.asarray(heads.wx[0]).copy()
    b0 = np.asarray(heads.b[0]).copy()
    rng = np.random.default_rng(seed)

    for i in range(G):
        x = ds.x_train[i]
        # The head reads `w`, so the tangent's slope must be d(mu)/dw, not d(mu)/dx:
        # w = x/(1-x) => dx/dw = (1-x)^2. Capped coordinates carry a zero dual (they
        # are replete), so the clip never truncates a slope that matters.
        g_w = ds.g_train[i] * (1.0 - x) ** 2 / ds.mu_scale[i]
        w = np.asarray(to_diag(jnp.asarray(x)))
        mu = ds.mu_train[i] / ds.mu_scale[i]
        idx = rank_by_active_set(g_w, ds.gvalid_train[i])
        if idx.size == 0:
            continue
        if idx.size < n_slots:  # too few usable rows: cycle, then jitter the rest
            idx = np.concatenate([idx, rng.choice(idx, n_slots - idx.size)])
        idx = idx[:n_slots]
        a = g_w[idx] * ds.mask[i]  # (n_slots, M), >= 0
        c = mu[idx] - np.einsum("km,km->k", a, w[idx])
        # pre-act_j = -softplus(wx_j).y + b_j, and gmax = max_j, so
        # mu = softplus(out_z) * min_j(a_j.y - b_j) + ... => b_j = -c_j.
        # softplus^-1, stable: `expm1` overflows at a ~ 88 in float32 and the head
        # is then seeded with inf weights and NaN curvature. Real label tangents
        # reach it -- 3/21 organisms on `20hm_bands` -- and softplus is the
        # identity to float precision well before that.
        wx0[i] = np.where(a > 30.0, a, np.log(np.expm1(np.clip(a, 1e-9, 30.0))))
        b0[i] = -c

    heads = eqx.tree_at(lambda h: (h.wx[0], h.b[0]), heads, (jnp.asarray(wx0), jnp.asarray(b0)))
    if len(heads.wx) == 1 and heads.out_z.shape[1] == 1:
        # The exact case: one unit, one group of `n_slots` planes. Make the output
        # layer the identity on it -- unit gain, no linear skip, no offset.
        one = np.log(np.expm1(1.0))
        heads = eqx.tree_at(
            lambda h: (h.out_z, h.out_x, h.out_b),
            heads,
            (
                jnp.full_like(heads.out_z, one),
                jnp.full_like(heads.out_x, -25.0),  # softplus(-25) ~ 1e-11, i.e. 0
                jnp.zeros_like(heads.out_b),
            ),
        )
    return heads
