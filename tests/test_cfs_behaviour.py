"""Head B (§4/M4) unit checks."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cfs.surrogate import behaviour as B


def test_mm_floor_recovers_the_uptake_bound_in_the_head_s_own_units():
    """``x = u/(u+s)`` inverts exactly, and the head emits ``z/(mu*z_scale)``."""
    G, N, M = 2, 4, 3
    x = jnp.full((G, N, M), 0.5)  # u = x/(1-x) * s = s
    lo = B.mm_floor(x, jnp.full((G, N), 2.0), jnp.full((G, M), 1e-3), jnp.full((G, M), 5.0))
    assert np.allclose(np.asarray(lo), -B.VMAX * 1e-3 / (2.0 * 5.0))


def test_mm_hinge_is_silent_on_compliant_rows_and_pays_on_violations():
    """§3.3's uptake bound is a *provable* violation, so the term must be one-sided.

    Every label satisfies ``z_m >= -Vmax_m * u_m``, so ``--w-mm`` must add exactly
    nothing when the prediction is inside the bound and grow only with the part
    that is outside -- the same shape, and the same reason, as Head A's
    ``--w-under``.
    """
    G, N, M = 2, 6, 4
    mask = np.ones((G, M), bool)
    heads = B.stack_heads(jax.random.PRNGKey(0), G, M, mask, width=8, depth=2)
    x = jnp.asarray(np.random.default_rng(0).uniform(0.1, 0.9, (G, N, M)), jnp.float32)
    a = jnp.zeros((G, N))
    zn = jnp.zeros((G, N, M))
    z = np.asarray(B.batched_z(heads, x, a))

    plain = float(B._loss(heads, x, a, zn, 0.0, 0.0))
    # A bound the head already satisfies everywhere: the hinge must be exactly 0.
    slack = jnp.asarray(np.full_like(z, z.min() - 1.0))
    assert float(B._loss(heads, x, a, zn, slack, 10.0)) == pytest.approx(plain, rel=1e-6)
    # ...and one it violates everywhere: monotone in the weight, and in the gap.
    tight = jnp.asarray(np.full_like(z, z.max() + 1.0))
    small = float(B._loss(heads, x, a, zn, tight, 1.0))
    big = float(B._loss(heads, x, a, zn, tight, 10.0))
    assert big > small > plain
    worse = float(B._loss(heads, x, a, zn, jnp.asarray(np.full_like(z, z.max() + 2.0)), 1.0))
    assert worse > small
