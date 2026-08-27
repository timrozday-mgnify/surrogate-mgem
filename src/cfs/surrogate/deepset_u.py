"""The DeepSet head with ``phi`` concave in ``u`` instead of in ``x``.

:mod:`cfs.surrogate.deepset`'s ``phi`` is concave and non-decreasing in the
*scalar* ``x_m = u_m/(u_m + s_m)``. That is the same coordinate error
:mod:`cfs.surrogate.picnn_u` documents, one dimension at a time: ``mu_max`` is
concave and piecewise linear in ``u``, ``u = s*x/(1-x)`` is convex in ``x``, so a
per-metabolite ramp that is concave in ``x_m`` cannot follow it either. Measured
on the labels, the ``x`` tangent test is violated on 32-57% of row pairs and the
``u`` one on 0.0%.

So this module is the deepset arm of that correction: identical architecture,
identical pooling, ``phi`` fed ``w_m = min(u_m/s_m, W_CAP)`` — affine in ``u``,
so ``{f(w_m) : f concave non-decreasing}`` is the full concave-in-``u`` class in
that coordinate. ``rho`` is untouched: it reads the pooled *code*, not a
saturation, and its concavity there was never in the wrong space.

**This does not fix the other limit of the architecture**, which is separate and
was not measured away. Pooling is a mean, so

```
d(mu)/dx_m = <rho'(S), d(phi_m)/dx_m>
```

— the rest of the medium reaches the gradient *pattern* only through the
``k_code``-wide vector ``rho'(S)``. Liebig limitation is an argmin across
metabolites, and a ``k_code`` = 16 or 64 channel is a narrow place to run that
comparison. The coordinate fix removes a *structural exclusion*; it does not
widen that channel.

Cost is unchanged and it is the reason this arm needs a cluster: ``phi`` is
priced **per metabolite**, so the trunk costs ``|M| * hidden^2`` per row against
the ICNN's single ``|M| * width`` matmul — measured 5-11 h per organism against
the ICNN's 5 minutes.
"""

from __future__ import annotations

import jax
from jax import Array

from cfs.surrogate import deepset
from cfs.surrogate.deepset import (  # noqa: F401  (the arch interface, re-exported)
    DeepSetHead,
    batched_value,
    batched_value_and_grad,
    organism,
)
from cfs.surrogate.picnn import _softplus_inv
from cfs.surrogate.picnn_u import INPUT_TRANSFORM, W_CAP, to_diag  # noqa: F401


class TrunkU(deepset.Trunk):
    """:class:`cfs.surrogate.deepset.Trunk` with ``phi`` reading ``w``, not ``x``."""

    def __init__(self, key, n_in: int, emb_dim: int, k: int, hidden: int, depth: int):
        super().__init__(key, n_in, emb_dim, k, hidden, depth)
        # Same scale-aware init as `picnn_u`, and for the same measured reason: the
        # parent sizes `phi`'s slopes at sqrt(2) for a scalar in [0, 1], and on `w`
        # (up to `W_CAP`, with ~45% of cells at the far end) that starts the head
        # saturated into softplus's linear regime — initial loss 1.7e6 and a median
        # Hessian condition of exactly 0, i.e. no curvature to train against.
        #
        # A slope here is `softplus(wx + cx @ h)`, so dividing it by `scale` is only
        # exact for the `wx` term. `cx` starts at 0.1 * normal and `h` is a tanh, so
        # the conditioning term is a small perturbation at init and `cx / scale`
        # carries it along; this is an initial point, not a constraint, and the
        # constraint (the sign of the whole pre-softplus sum) is untouched.
        scale = W_CAP / 2.0
        shrink = lambda p: _softplus_inv(jax.nn.softplus(p) / scale)  # noqa: E731
        self.wx = [shrink(p) for p in self.wx]
        self.cx = [c / scale for c in self.cx]
        self.ox = shrink(self.ox)
        self.cox = self.cox / scale

    def __call__(self, x: Array, mask: Array) -> Array:
        return super().__call__(to_diag(x), mask)


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
    """As :func:`cfs.surrogate.deepset.stack_heads`, with the ``u``-space trunk."""
    return deepset.stack_heads(
        key,
        n_organisms,
        n_in,
        mask,
        width,
        depth,
        emb_dim,
        shared=shared,
        phi_hidden=phi_hidden,
        k_code=k_code,
        trunk_cls=TrunkU,
    )


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


def head_in_diag(head: DeepSetHead):
    """The head as a function of ``w``: the trunk's map, bypassed.

    Diagnostics must not reach ``w`` by round-tripping through ``x`` — ``1 - x``
    cancels in float32 across the replete far field. Calling the *parent* trunk
    skips :func:`to_diag` instead of inverting it.
    """
    return lambda w: head.rho(deepset.Trunk.__call__(head.trunk, w, head.mask))


def batched_value_diag(heads: DeepSetHead, w: Array) -> Array:
    """``batched_value`` in the concavity coordinate — takes ``w``, not ``x``."""
    return deepset._map(heads, w, head_in_diag)
