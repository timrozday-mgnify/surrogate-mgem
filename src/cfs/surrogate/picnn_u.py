"""Head A, concave in ``u`` instead of in the network's input ``x`` (plan §6.2).

:mod:`cfs.surrogate.picnn` is concave and non-decreasing in ``x = u/(u + s)``.
That is **strictly stronger than the physics** and it excludes the target:

``mu_max`` is an LP value function in its right-hand side, and the §3.3 uptake
bound is ``lb = -Vmax * u``, so

```
mu_max(u) = min_k (a_k . u + c_k),      a_k >= 0
```

— exactly concave and piecewise *linear* in ``u``. But ``u = s * x/(1-x)`` is
**convex** in ``x``, so each ramp is convex in ``x`` and a concave-in-``x`` head
can only fit it with a chord. ``{concave in x}`` is a proper subset of
``{concave in u}`` and the target sits in the gap. Measured on the labels
themselves (tangent test, 1500 random row pairs, ``mu(p2) <= mu(p1) +
grad(p1).(p2-p1)``):

| organism | violated in ``x`` | violated in ``u`` |
| --- | --- | --- |
| CR626927.1 | 32.3% (median 6.4, mu std 18.3) | **0.0%** |
| CP001820.1 | 39.7% | **0.0%** |
| ABCC02 | 56.9% | **0.0%** |

That is what the M3b sweep measured as an inert capacity axis: ICNN width
128->1024 and depth 3->6 move held-out cosine and R2 by +-0.001, and 5x labels
move R2 by +0.01, because every variant converges to the same object — the
projection of the target onto a class that does not contain it. Capacity cannot
move a projection.

**The head is the existing ICNN; only the coordinate changes.** It is fed
``w = u/s = x/(1-x)``, which is *affine* in ``u`` per metabolite, so
``{f(w) : f concave non-decreasing}`` is the full concave-in-``u`` class. The
§4.7 demand probe already anchors ``s`` at the metabolite's own limiting regime,
so ``w = 1`` **is** the kink — measured median ``w`` over limiting cells is
1.00 on 3/3 organisms, which is where a normally-initialised first layer looks.

Any *concave* squash of ``w`` (including the ``x`` map) reintroduces the same
defect, so the only admissible rescale is affine — hence ``w``, not ``x``, and
hence the hard clip below rather than a soft one.

``W_CAP`` bounds the far field: ``min(w, cap)`` is a min of two affine functions,
so it preserves concavity in ``u`` exactly, and it is exact below the cap. It is
a constant rather than a flag because the value was measured, not guessed —
against the parameter-free cutting-plane model (which is this same hypothesis
class with no optimiser in the way) on the 20000-media labels:

| cap | CR626927.1 | ABCC02 | CP001820.1 |
| --- | --- | --- | --- |
| 30 | 0.960 | **0.432** | — |
| 100 | 0.971 | 0.793 | — |
| 300 | 0.9693 / R2 0.9979 | 0.9816 / 0.9995 | 0.9729 / 0.9967 |
| 1000 | 0.9690 | 0.9817 | 0.9722 |
| none | 0.9690 | 0.9817 | 0.9722 |

300 is where clipping stops costing anything (<=0.001 against uncapped, p05
marginally better) while still shrinking the input range 25x. Capping *low* is
actively harmful: a limiting cell above the cap gets zero predicted gradient on
the one metabolite that matters, which is why ABCC02 collapses at 30 — 4.1% of
its limiting cells sit above it.

The diagnostics have to move with the constraint. :func:`to_diag`,
:func:`batched_value_diag` and :func:`head_in_diag` tell
:func:`cfs.surrogate.train.evaluate` to take the concavity and Hessian checks in
``w``; run in ``x`` this head reads 98% violating and
``cond ~ 1e32``, both meaningless there.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from cfs.surrogate.picnn import ValueHead, _softplus_inv, organism  # noqa: F401

# Measured, see the module docstring. Not a flag: no sweep axis has asked for it.
W_CAP = 300.0

INPUT_TRANSFORM = (
    "u = c / (Km + c), w = min(u / x_scale, 300) = min(x / (1 - x), 300); "
    "Km from km_defaults.yaml. Concavity is imposed in w, which is affine in u."
)


def to_diag(x: Array) -> Array:
    """``x = u/(u+s)`` -> ``w = u/s``, clipped. The head's own input map."""
    return jnp.minimum(x / jnp.maximum(1.0 - x, 1e-12), W_CAP)


def head_in_diag(head: ValueHeadU):
    """The head as a function of ``w`` directly — the parent ICNN, unwrapped.

    Diagnostics must not reach ``w`` by round-tripping through ``x``: ``1 - x``
    cancels catastrophically in float32 once ``x`` is near 1, which is exactly the
    replete far field (~45% of cells), and a correctly concave head then reads as
    violating. This bypasses the map instead.
    """
    return lambda w: ValueHead.__call__(head, w)


class ValueHeadU(ValueHead):
    """The ICNN of :mod:`cfs.surrogate.picnn`, reading ``w`` instead of ``x``.

    Concave and non-decreasing in ``w``, hence concave and non-decreasing in
    ``u`` — and, unlike the parent, *not* constrained to be concave in ``x``,
    which is the point.
    """

    def __init__(self, key, n_in: int, mask, width: int = 128, depth: int = 3):
        super().__init__(key, n_in, mask, width, depth)
        # The parent initialises the input weights at `sqrt(2/n_in)`, sized for
        # `x` in [0, 1]. `w` runs to `W_CAP` and ~45% of cells sit at the far end,
        # so the same weights put the first layer at O(1e3): softplus is affine
        # there, the head starts *linear* (measured: initial loss 1.7e6, median
        # Hessian condition exactly 0, i.e. no curvature anywhere) and Adam spends
        # the run walking the bias back instead of fitting. Divide by the typical
        # `w` a row carries -- half the cells clipped at the cap -- which is a
        # pure reparameterisation of the initial point, not of the class.
        scale = W_CAP / 2.0
        rescale = lambda p: _softplus_inv(jax.nn.softplus(p) / scale)  # noqa: E731
        self.wx = [rescale(p) for p in self.wx]
        self.out_x = rescale(self.out_x)

    def __call__(self, x: Array) -> Array:
        return super().__call__(to_diag(x))


def stack_heads(
    key, n_organisms: int, n_in: int, mask, width: int = 128, depth: int = 3
) -> ValueHeadU:
    """One :class:`ValueHeadU` PyTree with a leading organism axis (§6.1)."""
    keys = jax.random.split(key, n_organisms)
    make = eqx.filter_vmap(lambda k, m: ValueHeadU(k, n_in, m, width, depth), in_axes=(0, 0))
    return make(keys, jnp.asarray(mask, dtype=bool))


@eqx.filter_vmap(in_axes=(0, 0))
def batched_value(heads: ValueHeadU, x: Array) -> Array:
    """``(G, B, M) -> (G, B)``."""
    return jax.vmap(heads)(x)


@eqx.filter_vmap(in_axes=(0, 0))
def batched_value_diag(heads: ValueHeadU, w: Array) -> Array:
    """``batched_value`` in the concavity coordinate — takes ``w``, not ``x``."""
    return jax.vmap(head_in_diag(heads))(w)


@eqx.filter_vmap(in_axes=(0, 0))
def batched_value_and_grad(heads: ValueHeadU, x: Array):
    """``(G, B, M) -> ((G, B), (G, B, M))`` — the gradient IS the shadow price.

    Still ``d(mu)/dx``: the chain rule through :func:`to_diag` is autodiff's job,
    and :func:`cfs.surrogate.train._du` converts to ``u`` space as for every
    other arch, so the gate is the same number.
    """
    return jax.vmap(jax.value_and_grad(heads))(x)
