"""An unconstrained softplus MLP — a measurement, not a candidate head.

It drops both structural constraints: no concavity, no monotonicity. That makes it
useless downstream (§8's Newton needs a PSD Jacobian, and P3/P11 are about exactly
this), and it will report a non-zero ``concavity_violation_rate`` — the expected
result, not a bug.

What it answers is the one question the concave family cannot ask of itself: if an
unconstrained net of the same width also caps near the ICNN's 0.80 mean u-space
cosine, the ceiling is in the labels or the input transform and no further search
*inside* the concave family will reach the 0.99 gate. If it sails past, the
constraints are the binding cost and the difference-of-convex escape hatch (P11)
is worth its complexity.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from cfs.surrogate.picnn import organism  # noqa: F401  — same all-leaves-stacked slice


class MLPHead(eqx.Module):
    """Masked-input softplus MLP. Softplus not ReLU, so the Hessian exists (P3).

    Hand-rolled rather than :class:`equinox.nn.MLP` because that stores its
    activation as a pytree leaf, which `filter_vmap` then tries to map over the
    organism axis. Same weight layout as :class:`cfs.surrogate.picnn.ValueHead`,
    minus every sign constraint — that difference is the whole point of the file.
    """

    w: list[Array]
    b: list[Array]
    mask: Array

    def __init__(self, key, n_in: int, mask, width: int = 128, depth: int = 3):
        keys = jax.random.split(key, depth + 1)
        sizes = [n_in] + [width] * depth + [1]
        self.w = [
            jax.random.normal(k, (o, i)) * jnp.sqrt(2.0 / i)
            for k, i, o in zip(keys, sizes[:-1], sizes[1:], strict=True)
        ]
        self.b = [jnp.zeros(o) for o in sizes[1:]]
        self.mask = jnp.asarray(mask, dtype=bool)

    def __call__(self, x: Array) -> Array:
        h = x * self.mask
        for w, b in zip(self.w[:-1], self.b[:-1], strict=True):
            h = jax.nn.softplus(w @ h + b)
        return (self.w[-1] @ h + self.b[-1])[0]


def stack_heads(
    key, n_organisms: int, n_in: int, mask, width: int = 128, depth: int = 3
) -> MLPHead:
    keys = jax.random.split(key, n_organisms)
    make = eqx.filter_vmap(lambda k, m: MLPHead(k, n_in, m, width, depth), in_axes=(0, 0))
    return make(keys, jnp.asarray(mask, dtype=bool))


@eqx.filter_vmap(in_axes=(0, 0))
def batched_value(heads: MLPHead, x: Array) -> Array:
    return jax.vmap(heads)(x)


@eqx.filter_vmap(in_axes=(0, 0))
def batched_value_and_grad(heads: MLPHead, x: Array):
    return jax.vmap(jax.value_and_grad(heads))(x)
