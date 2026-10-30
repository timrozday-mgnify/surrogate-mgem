"""Head A conditioned on a **reaction-presence mask** — §14.2, option A.

Everything in :mod:`cfs.surrogate.groupmax` conditions ``mu_max`` on the medium for
a *fixed* network. This head adds a second input, ``z in {0,1}^R`` over the
organism's own internal reactions, so an internal edit — a candidate annotation, a
knockout, a gap-fill candidate — can be screened without rebuilding the GEM.

**A reaction is not a hyperplane.** A hyperplane of the value function is one
optimal basis, i.e. a vertex of the dual polytope: a metabolic *strategy* whose
gradient is its shadow-price vector. A reaction is a *column* of ``S``, which in the
dual is one *constraint*. A dual vertex is the intersection of many such
constraints, so each reaction affects many hyperplanes and each hyperplane is fixed
by a combination of reactions — neither direction is one-to-one. The consequence
that sets the sign convention: deleting a column deletes a dual constraint, which
*enlarges* the dual feasible set, so the min runs over *more* vertices and ``mu``
falls. **A knockout adds pieces to the min.** Here ``out`` is convex and
``mu_hat = -out``, so a knockout must *raise* a plane's offset to push it into the
group max. Conditioning that only ever removes capacity has the mechanism backwards.

**The mask enters only through the plane parameters, never the max itself:**

```
b_k(z)  = b_k + Bb_k . e(z)                 # per-unit offset, pre-activation
wx_k(z) = -softplus(wx_k + Bw_k . e(z))     # per-unit, inside the softplus
e(z)    = tanh(E z + e_b) @ W2               # zero-init: e(z) == 0 at init
```

so three invariants hold **structurally, for every** ``z``:

- **concave in** ``w`` — a min of affine-in-``w`` functions is concave however its
  coefficients were produced, so ``concavity_violation_rate`` must still read
  exactly 0, still measured in ``w``/``u`` and never in the model's input space;
- **monotone non-decreasing in** ``w`` — the signs are structural (``softplus`` on
  the pass-through, ``-softplus`` on the input skips) and an additive offset
  *inside* the ``softplus`` cannot change one;
- **the Sobolev target is unchanged** — ``grad_w mu_hat`` is still the active piece's
  ``pi``, so §7's per-row norm-relative term needs no modification.

``e(z) == 0`` at init (the final hypernetwork layer is zero), so the head is
**bit-identical to** :class:`~cfs.surrogate.groupmax.GroupMaxHead` before training
and inherits every measured result in the plan — the ``--gm-init labels`` tangent
init, ``DEFAULT_TEMP = 0.01``, ``W_CAP = 1e5``, the monotone output. That is the
reason for this parameterisation rather than a redesign: the measured-good head is
the starting point, not something to be re-earned.

**Deliberately restricted (§14.2).** Offsets are per *unit* (per plane), not per
(plane, input): ``wx`` is ``(width*group, M)`` per layer, so per-input offsets would
need ~450k hypernetwork outputs per layer. A mask can lift, drop or rescale a
strategy but not freely rotate it; rotation is §14.3's job. This is an assumption to
measure (M20), not a saving.

**Not structural, so it is a gate, not a guarantee.** ``mu_max`` is monotone
non-increasing under reaction removal and this family does not enforce it — see
:func:`mask_monotonicity_violation_rate`, which joins ``concavity_violation_rate``
in the M18 gate. §14.3's learned dual polytope gets that for free, which is the main
argument for eventually building it.

**P18, and it governs how this head may be quoted.** A uniform mask draw either hits
redundant reactions (``mu`` unchanged) or essential ones (``mu`` = 0); both are cheap
to predict and teach nothing. Labels must come from the *thin shell* — masks
differing from a carved model by a handful of reactions, drawn from the candidate set
actually being screened (:func:`cfs.sampling.masks.sample_knockout_masks`). So this
head is valid **near** the models it was trained around, not "network-editable" in
general, and that limit belongs beside every number it produces.

**P22**: only *internal* reactions are masked. An exchange edit must go through the
medium input, or the two inputs contradict each other.
"""

from __future__ import annotations

import logging

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from cfs.surrogate.groupmax import DEFAULT_GROUP, DEFAULT_TEMP, GroupMaxHead
from cfs.surrogate.picnn_u import to_diag

LOGGER = logging.getLogger(__name__)

DEFAULT_EMBED = 16


class MaskedGroupMaxHead(eqx.Module):
    """``mu_max(w, z)`` — :class:`GroupMaxHead` with its planes conditioned on ``z``.

    ``head`` carries the whole unconditioned head, so slicing, temperature surgery
    and the label-tangent init keep working on it unchanged.
    """

    head: GroupMaxHead
    embed_w: Array  # (R, embed) mask -> hidden
    embed_b: Array  # (embed,)
    out_w: Array  # (embed, embed) zero-init: e(z) == 0 at init
    bb: list[Array]  # (embed, width*g) per-layer bias offsets
    bw: list[Array]  # (embed, width*g) per-layer input-skip offsets
    n_reactions: int = eqx.field(static=True)

    def __init__(
        self,
        key,
        n_in: int,
        mask,
        n_reactions: int,
        width: int = 128,
        depth: int = 3,
        group: int = DEFAULT_GROUP,
        temp: float = DEFAULT_TEMP,
        embed: int = DEFAULT_EMBED,
        head: GroupMaxHead | None = None,
    ):
        keys = jax.random.split(key, 2 * depth + 3)
        self.head = (
            head
            if head is not None
            else GroupMaxHead(keys[0], n_in, mask, width, depth, group, temp)
        )
        h = width * group
        # The embedding reads a 0/1 vector of length R, so scale by 1/sqrt(R) to keep
        # the pre-tanh at O(1) however many reactions are present.
        scale = jnp.sqrt(1.0 / max(n_reactions, 1))
        self.embed_w = jax.random.normal(keys[1], (n_reactions, embed)) * scale
        self.embed_b = jnp.zeros((embed,))
        # Zero: `e(z) == 0` for every `z` at init, so this head starts bit-identical
        # to its unconditioned parent and inherits the measured init (§14.2, P21).
        self.out_w = jnp.zeros((embed, embed))
        # Offsets are small at init for the same reason -- `out_w` is zero so these
        # are inert until trained, but a large init here makes the first gradient
        # step able to kill every plane a row needs (P21).
        self.bb = [jax.random.normal(keys[2 + i], (embed, h)) * 0.01 for i in range(depth)]
        self.bw = [jax.random.normal(keys[2 + depth + i], (embed, h)) * 0.01 for i in range(depth)]
        self.n_reactions = int(n_reactions)

    def embedding(self, z: Array) -> Array:
        """``e(z)``, ``(embed,)``. Exactly zero at init.

        No clip anywhere: M17's rule is that a clip in an input map is a modelling
        assumption, because ``min(., cap)`` preserves concavity and destroys
        injectivity, and an upper-bound invariant needs the second.
        """
        return jnp.tanh(z @ self.embed_w + self.embed_b) @ self.out_w

    def on_w(self, w: Array, z: Array) -> Array:
        """The head in its own coordinate -- no input map applied.

        Diagnostics must reach ``w`` this way rather than by inverting ``to_diag``:
        ``1 - x`` cancels catastrophically in float32 in the replete far field, and a
        correctly concave head then reads as violating.
        """
        hd = self.head
        e = self.embedding(z)
        y = w * hd.mask
        wx = -jax.nn.softplus(hd.wx[0] + (e @ self.bw[0])[:, None])
        a = wx @ y + hd.b[0] + e @ self.bb[0]
        zz = hd._gmax(a)
        for k, (wx_k, wz_k, b_k) in enumerate(zip(hd.wx[1:], hd.wz, hd.b[1:], strict=True), 1):
            wx_e = -jax.nn.softplus(wx_k + (e @ self.bw[k])[:, None])
            a = jax.nn.softplus(wz_k) @ zz + wx_e @ y + b_k + e @ self.bb[k]
            zz = hd._gmax(a)
        out = jax.nn.softplus(hd.out_z) @ zz - jax.nn.softplus(hd.out_x) @ y + hd.out_b
        return -out

    def __call__(self, x: Array, z: Array) -> Array:
        return self.on_w(to_diag(x), z)


def head_in_diag(head: MaskedGroupMaxHead):
    """The head as a function of ``(w, z)``. See :meth:`MaskedGroupMaxHead.on_w`."""
    return head.on_w


def stack_heads(
    key,
    n_organisms: int,
    n_in: int,
    mask,
    n_reactions: int,
    width: int = 128,
    depth: int = 3,
    group: int = DEFAULT_GROUP,
    temp: float = DEFAULT_TEMP,
    embed: int = DEFAULT_EMBED,
) -> MaskedGroupMaxHead:
    """One PyTree with a leading organism axis (§6.1).

    ``n_reactions`` is the width of the **frozen, hashed** mask index (P13), so it is
    one number for the roster, not per organism; an organism that lacks a reaction
    simply never has that bit set.
    """
    keys = jax.random.split(key, n_organisms)
    make = eqx.filter_vmap(
        lambda k, m: MaskedGroupMaxHead(k, n_in, m, n_reactions, width, depth, group, temp, embed),
        in_axes=(0, 0),
    )
    return make(keys, jnp.asarray(mask, dtype=bool))


def organism(heads: MaskedGroupMaxHead, i: int) -> MaskedGroupMaxHead:
    """Slice organism ``i`` out of a stacked head."""
    return jax.tree.map(lambda p: p[i] if eqx.is_array(p) else p, heads)


@eqx.filter_vmap(in_axes=(0, 0, 0))
def batched_value(heads: MaskedGroupMaxHead, x: Array, z: Array) -> Array:
    """``(G, B, M), (G, B, R) -> (G, B)``."""
    return jax.vmap(heads)(x, z)


@eqx.filter_vmap(in_axes=(0, 0, 0))
def batched_value_and_grad(heads: MaskedGroupMaxHead, x: Array, z: Array):
    """``-> ((G, B), (G, B, M))`` — the gradient in ``x`` IS the shadow price.

    Differentiated in ``x`` only: ``z`` is discrete, and its "gradient" is an
    artefact of the embedding rather than a quantity the LP has.
    """
    return jax.vmap(jax.value_and_grad(heads))(x, z)


def mask_monotonicity_violation_rate(
    head: MaskedGroupMaxHead,
    w: Array,
    z: Array,
    key,
    n_drop: int = 1,
    tol: float = 1e-6,
) -> Array:
    """Fraction of nested mask pairs where removing a reaction *raised* ``mu_hat``.

    ``mu_max`` is monotone non-increasing under reaction removal — deleting a column
    can only shrink the primal feasible set — but §14.2's family does **not** enforce
    that, so it is a measured gate rather than a guarantee (§14.6, M18). For each row
    this drops ``n_drop`` of the *present* reactions and checks
    ``mu_hat(w, z') <= mu_hat(w, z) + tol``.

    ponytail: one dropped subset per row, not an exhaustive sweep; raise ``n_drop`` or
    call it with several keys if the rate needs a tighter confidence interval.
    """
    base = jax.vmap(head.on_w)(w, z)
    # Rank only the present bits so the drop is always a real removal: a uniform
    # choice over R would mostly "remove" reactions that are already absent, and the
    # rate would read artificially clean.
    noise = jax.random.uniform(key, z.shape)
    ranked = jnp.where(z > 0, noise, 2.0)
    kth = jnp.sort(ranked, axis=-1)[:, n_drop - 1][:, None]
    dropped = jnp.where(ranked <= kth, 0.0, z)
    lowered = jax.vmap(head.on_w)(w, dropped)
    # Rows with fewer than n_drop present reactions have nothing to drop: `dropped`
    # equals `z` there, so the comparison is trivially satisfied and does not bias
    # the rate.
    return jnp.mean(lowered > base + tol)
