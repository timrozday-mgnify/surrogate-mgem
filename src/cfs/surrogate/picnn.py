"""Head A — ``mu_max`` as a concave function of uptake saturation (plan §6.2).

An input-convex neural network, negated. Convexity in the input needs two things
(Amos et al. 2017): the weights on the hidden pass-through path are non-negative,
and the activation is convex and non-decreasing. Negating the output gives a
concave function, which is what an LP value function is in its bounds (§1).

```
z_1     = softplus(W_x^0 x + b_0)
z_{k+1} = softplus(W_z^k z_k + W_x^k x + b_k)      W_z^k >= 0
out     = w_z . z_L + w_x . x + b                  w_z >= 0
mu_hat  = -out
```

A concave piecewise-linear function is a *min of affine functions*, which is
exactly what a negated softplus ICNN approximates — the structure matches the
target. What the target also is, measured on a real GEM: a ramp that reaches its
plateau by ``x ~ 5e-3`` and is flat over the remaining 99% of ``[0, 1]``. So the
inputs arrive already divided by a per-metabolite kink scale
(:func:`cfs.surrogate.data.load_value_dataset`), which puts those kinks at O(1)
where a normally-initialised first layer can reach them. A diagonal positive
rescale is affine, so it costs nothing in concavity.

**Softplus, never ReLU** (P3): Newton in Phase 5 differentiates the *network*
twice, and a ReLU net has zero Hessian almost everywhere — the gradients look
fine right up until the composition stalls.

Head A takes no conditioning inputs, so this is a plain ICNN rather than the
partial one; the ``x`` skip connections are unconstrained in sign because ``x``
enters affinely.

The organism mask is applied on entry: a metabolite the organism cannot exchange
contributes a hard zero, not a learned weight. That keeps one architecture across
the roster so the whole thing stacks and vmaps (§6.1).
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array


def _softplus_inv(y: float) -> float:
    return float(jnp.log(jnp.expm1(y)))


class ValueHead(eqx.Module):
    """Concave ``mu_max(x)`` over rescaled saturation ``x = c/(Km+c) / x_scale``."""

    wx: list[Array]  # (H, M) input skips, unconstrained
    wz: list[Array]  # (H, H_prev) pass-through, non-negative via softplus
    b: list[Array]
    out_z: Array  # (H,) non-negative via softplus
    out_x: Array  # (M,)
    out_b: Array
    # Bool, so `eqx.is_inexact_array` filtering leaves it out of the trained
    # parameters and out of the optimiser state.
    mask: Array

    def __init__(self, key, n_in: int, mask, width: int = 128, depth: int = 3):
        keys = jax.random.split(key, 3 * depth + 3)
        scale = jnp.sqrt(2.0 / n_in)
        widths = [width] * depth
        self.wx = [
            jax.random.normal(keys[i], (w, n_in)) * scale for i, w in enumerate(widths)
        ]
        # The pass-through weights are a non-negative sum over `width` units, so
        # initialise them at softplus^-1(1/width): without the 1/width the output
        # compounds to O(100) at init and Adam spends thousands of steps just
        # walking the bias back down.
        w0 = _softplus_inv(1.0 / width)
        self.wz = [
            jax.random.normal(keys[depth + i], (widths[i], widths[i - 1])) * 0.1 + w0
            for i in range(1, depth)
        ]
        # Biases must be *diverse*, not zero: the model is a non-negative sum of
        # softplus ridge functions, so with every unit at the same offset they all
        # sit in the same part of the curve and the net collapses to roughly one
        # effective unit. Zero-init here costs ~0.5 R^2 on a plain quadratic.
        self.b = [
            jax.random.uniform(keys[2 * depth + i], (w,), minval=-1.0, maxval=1.0)
            for i, w in enumerate(widths)
        ]
        self.out_z = jax.random.normal(keys[-1], (width,)) * 0.1 + w0
        self.out_x = jax.random.normal(keys[-2], (n_in,)) * scale
        self.out_b = jnp.zeros(())
        self.mask = jnp.asarray(mask, dtype=bool)

    def __call__(self, x: Array) -> Array:
        y = x * self.mask
        z = jax.nn.softplus(self.wx[0] @ y + self.b[0])
        for wx, wz, b in zip(self.wx[1:], self.wz, self.b[1:], strict=True):
            z = jax.nn.softplus(jax.nn.softplus(wz) @ z + wx @ y + b)
        out = jax.nn.softplus(self.out_z) @ z + self.out_x @ y + self.out_b
        return -out


def stack_heads(key, n_organisms: int, n_in: int, mask, width: int = 128,
                depth: int = 3) -> ValueHead:
    """One :class:`ValueHead` PyTree with a leading organism axis (§6.1).

    20 organisms then cost about what 1 costs — this is the single biggest
    engineering win at this scale, and it has to be in from the start.
    """
    keys = jax.random.split(key, n_organisms)
    make = eqx.filter_vmap(
        lambda k, m: ValueHead(k, n_in, m, width, depth), in_axes=(0, 0)
    )
    return make(keys, jnp.asarray(mask, dtype=bool))


@eqx.filter_vmap(in_axes=(0, 0))
def batched_value(heads: ValueHead, x: Array) -> Array:
    """``(G, B, M) -> (G, B)``: every organism on its own medium batch."""
    return jax.vmap(heads)(x)


@eqx.filter_vmap(in_axes=(0, 0))
def batched_value_and_grad(heads: ValueHead, x: Array):
    """``(G, B, M) -> ((G, B), (G, B, M))`` — the gradient IS the shadow price.

    Taken by autodiff of Head A, never as a separate output head (§7.1): a second
    head is not constrained to be the derivative and would break concavity.
    """
    return jax.vmap(jax.value_and_grad(heads))(x)
