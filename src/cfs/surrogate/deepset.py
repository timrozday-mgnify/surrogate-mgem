"""Head A as a DeepSet over metabolites: ``mu(x) = rho( mean_m phi(e_m, x_m) )``.

The ICNN head (:mod:`cfs.surrogate.picnn`) gives every organism its own dense
``3 x 128 x 444`` stack — ~36M parameters over the roster, and every organism
learns ``EX_mg2_e``'s ramp from scratch off its own ~30 magnesium-limited media.
The ions are what the gate is stuck on, on 21/21 organisms.

This head is the other end of that trade:

- ``phi`` is a **PICNN** — the saturation ``x_m`` on the convex path, the
  metabolite's learned embedding ``e_m`` on the unconstrained conditioning path
  (§6.2's "unconstrained pathway for any conditioning inputs"). It is *shared
  across organisms*, so every organism's magnesium rows train one magnesium shape;
- ``rho`` is a small negated ICNN over the pooled code, and is the only
  per-organism part.

That also matches the target's structure better than a dense layer: with pooling,
``d(mu)/d(x_m) = rho'(S) . d(phi)/d(x_m)``, a per-metabolite term times one shared
scalar. 54.7% of label rows have exactly one non-zero dual.

**Concavity is structural, as in the ICNN.** ``phi`` is concave and
non-decreasing in ``x_m`` for *any* conditioning vector (the sign constraints are
additive, pre-softplus, so no value of ``h`` can flip them), a mean of concave
functions is concave, and ``rho`` is concave and non-decreasing — so the
composition is concave and non-decreasing. ``_concavity_violations`` must read
exactly 0; anything else means a sign reparameterisation is wrong, not that the
model is merely inaccurate.

``shared=False`` builds the identical architecture with nothing crossing
organisms. It is the ablation that says whether a win here is the inductive bias
or the cross-organism pooling; it is also the variant that stays inside D1 ("no
generalisation to unseen organisms"), which the shared trunk supersedes.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from cfs.surrogate.picnn import ValueHead, _softplus_inv


class Trunk(eqx.Module):
    """Metabolite embeddings + the shared PICNN ``phi: (e_m, x_m) -> R^k``.

    Pooling is the **mean** over the organism's own exchanges, not the sum: the
    roster's ``|M_i|`` spans 138-259, and a sum would hand ``rho`` an input whose
    scale is a per-organism constant. A positive constant scaling preserves
    concavity either way.
    """

    emb: Array  # (M, d)
    ce_w: Array  # (dh, d)
    ce_b: Array
    wx: list[Array]  # (H,) x-slopes, non-positive via -softplus (monotone in x_m)
    cx: list[Array]  # (H, dh) embedding's contribution to those slopes
    wz: list[Array]  # (H, H) pass-through, non-negative via softplus
    b: list[Array]
    cb: list[Array]  # (H, dh) embedding's contribution to the biases
    oz: Array  # (k, H)
    ox: Array  # (k,)
    cox: Array  # (k, dh)
    ob: Array  # (k,)
    cob: Array  # (k, dh)

    def __init__(self, key, n_in: int, emb_dim: int, k: int, hidden: int, depth: int):
        keys = jax.random.split(key, 5 * depth + 8)
        dh = 2 * emb_dim
        self.emb = jax.random.normal(keys[0], (n_in, emb_dim))
        self.ce_w = jax.random.normal(keys[1], (dh, emb_dim)) * 0.5
        self.ce_b = jnp.zeros(dh)
        # `phi` takes one scalar, so the picnn init's sqrt(2/n_in) is sqrt(2) here:
        # slopes of order 1, which is where the rescaled input's ramp already is.
        s = jnp.sqrt(2.0)
        self.wx = [
            _softplus_inv(jnp.abs(jax.random.normal(keys[2 + i], (hidden,)) * s) + 1e-6)
            for i in range(depth)
        ]
        w0 = _softplus_inv(1.0 / hidden)
        self.wz = [
            jax.random.normal(keys[2 + depth + i], (hidden, hidden)) * 0.1 + w0
            for i in range(1, depth)
        ]
        # Diverse biases: the head is a non-negative sum of softplus ridges, and at
        # a shared offset they all sit on the same part of the curve (picnn.py:92).
        self.b = [
            jax.random.uniform(keys[2 + 2 * depth + i], (hidden,), minval=-1.0, maxval=1.0)
            for i in range(depth)
        ]
        # Conditioning starts near zero, so phi begins as one shape for every
        # metabolite and differentiates as the embeddings train.
        self.cx = [
            jax.random.normal(keys[2 + 3 * depth + i], (hidden, dh)) * 0.1 for i in range(depth)
        ]
        self.cb = [
            jax.random.normal(keys[2 + 4 * depth + i], (hidden, dh)) * 0.1 for i in range(depth)
        ]
        self.oz = jax.random.normal(keys[-4], (k, hidden)) * 0.1 + w0
        self.ox = _softplus_inv(jnp.abs(jax.random.normal(keys[-3], (k,)) * s) + 1e-6)
        self.cox = jax.random.normal(keys[-2], (k, dh)) * 0.1
        self.ob = jnp.zeros(k)
        self.cob = jax.random.normal(keys[-1], (k, dh)) * 0.1

    def phi(self, e: Array, y: Array) -> Array:
        """``(d,), () -> (k,)``, concave and non-decreasing in ``y``."""
        h = jnp.tanh(self.ce_w @ e + self.ce_b)
        z = jax.nn.softplus(
            -jax.nn.softplus(self.wx[0] + self.cx[0] @ h) * y + self.b[0] + self.cb[0] @ h
        )
        for wx, cx, wz, b, cb in zip(
            self.wx[1:], self.cx[1:], self.wz, self.b[1:], self.cb[1:], strict=True
        ):
            z = jax.nn.softplus(
                jax.nn.softplus(wz) @ z - jax.nn.softplus(wx + cx @ h) * y + b + cb @ h
            )
        out = (
            jax.nn.softplus(self.oz) @ z
            - jax.nn.softplus(self.ox + self.cox @ h) * y
            + self.ob
            + self.cob @ h
        )
        return -out

    def __call__(self, x: Array, mask: Array) -> Array:
        """``(M,) -> (k,)``: pooled code over the organism's own exchanges."""
        codes = jax.vmap(self.phi)(self.emb, x)  # (M, k)
        return jnp.sum(codes * mask[:, None], axis=0) / jnp.maximum(mask.sum(), 1.0)


class DeepSetHead(eqx.Module):
    """``rho(trunk(x))`` — concave, non-decreasing, one head per organism.

    Callable on a single organism's single medium; :func:`batched_value_and_grad`
    is what maps it over the roster and the batch.
    """

    trunk: Trunk
    rho: ValueHead  # over the pooled code, so its own mask is all-ones
    mask: Array  # (M,) bool — the organism's exchanges
    shared: bool = eqx.field(static=True)

    def __call__(self, x: Array) -> Array:
        return self.rho(self.trunk(x, self.mask))


def stack_heads(
    key,
    n_organisms: int,
    n_in: int,
    mask,
    width: int = 128,
    depth: int = 3,
    emb_dim: int = 8,
    shared: bool = True,
    phi_hidden: int | None = None,
    k_code: int | None = None,
) -> DeepSetHead:
    """Stack ``n_organisms`` heads. Only ``rho``/``mask`` get the organism axis.

    ``phi``'s internals default to being derived from ``width`` — hidden
    ``width // 8``, code width 16, context ``2 * emb_dim`` — and ``phi_hidden`` /
    ``k_code`` override those. The derivation was a laptop-runtime choice, not a
    measured optimum (see below); the M3b sweep varies them directly, which is why
    they are flags and not just ``width``.

    ``phi`` is deliberately much narrower than ``rho``. It runs **once per
    metabolite**, so the trunk costs ``|M| * hidden^2`` per row against the ICNN's
    single ``|M| * width`` matmul: at ``hidden = width // 2`` that is 123x the
    ICNN's FLOPs and 47 s/epoch against 0.35 (measured), which is 19 h for a 1500
    epoch run. At ``width // 8`` it is 7.8x. The capacity is not missed — ``phi``
    is a monotone concave ramp in one scalar, conditioned on an ``emb_dim`` vector,
    not a 444-input function. Widen ``rho`` instead; it is priced per row, not per
    metabolite.
    """
    kt, kr = jax.random.split(key)
    k_code = k_code or 16
    hidden = phi_hidden or max(8, width // 8)
    rho = eqx.filter_vmap(
        lambda k_: ValueHead(k_, k_code, jnp.ones(k_code, dtype=bool), width, depth)
    )(jax.random.split(kr, n_organisms))
    make_trunk = lambda k_: Trunk(k_, n_in, emb_dim, k_code, hidden, depth)  # noqa: E731
    trunk = (
        make_trunk(kt) if shared else eqx.filter_vmap(make_trunk)(jax.random.split(kt, n_organisms))
    )
    return DeepSetHead(trunk, rho, jnp.asarray(mask, dtype=bool), shared)


def stack_heads_private(
    key,
    n_organisms: int,
    n_in: int,
    mask,
    width: int = 128,
    depth: int = 3,
    emb_dim: int = 8,
    phi_hidden: int | None = None,
    k_code: int | None = None,
) -> DeepSetHead:
    """The D1-preserving ablation: same architecture, nothing shared."""
    return stack_heads(
        key,
        n_organisms,
        n_in,
        mask,
        width,
        depth,
        emb_dim,
        shared=False,
        phi_hidden=phi_hidden,
        k_code=k_code,
    )


def _slice(tree, i: int):
    return jax.tree.map(lambda p: p[i] if eqx.is_array(p) else p, tree)


def organism(heads: DeepSetHead, i: int) -> DeepSetHead:
    """Slice organism ``i`` out — the trunk is whole when it is shared."""
    trunk = heads.trunk if heads.shared else _slice(heads.trunk, i)
    return DeepSetHead(trunk, _slice(heads.rho, i), heads.mask[i], heads.shared)


def _map(heads: DeepSetHead, x: Array, fn):
    # in_axes is `None` on a shared trunk so one copy serves the whole roster, and
    # 0 on the per-organism leaves. It is a Python-level choice off a static field.
    axes = (None if heads.shared else 0, 0, 0, 0)
    mapped = eqx.filter_vmap(
        lambda t, r, m, xx: jax.vmap(fn(DeepSetHead(t, r, m, heads.shared)))(xx),
        in_axes=axes,
    )
    return mapped(heads.trunk, heads.rho, heads.mask, x)


def batched_value(heads: DeepSetHead, x: Array) -> Array:
    """``(G, B, M) -> (G, B)``."""
    return _map(heads, x, lambda h: h)


def batched_value_and_grad(heads: DeepSetHead, x: Array):
    """``(G, B, M) -> ((G, B), (G, B, M))`` — the gradient IS the shadow price."""
    return _map(heads, x, jax.value_and_grad)
