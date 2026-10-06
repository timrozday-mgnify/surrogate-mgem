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

**Within that window ``T`` is set by the low-``mu`` bias, and 0.01 wins — measured
2026-08-29, which is why ``DEFAULT_TEMP`` moved from 0.03.** The smoothing sits
``~T*ln(K_active)`` *below* the hard min: an absolute offset, so it is ~4% of a
plateau ``mu`` and >100% of a starving one, and the absolute value MSE (74% of rows
are plateau) then lifts the bottom past the target rather than fixing it. On 21
organisms, 0.03 -> 0.01 takes the median bias below 5% of max ``mu`` from **+0.978
to +0.442** and the §8.1 composition's worst community from 0.322 to 0.051 log-X
error, ~5% at every community size, for worst gradient cosine 0.958 -> 0.928 (one
seed). Outside the window it reverses: **T=0.003 is worse than both on every
axis** (bias +1.305, worst cosine 0.880) — an optimisation failure, not a
representation one, and annealing into it (``--gm-temp-final``) does not rescue it
at roster scale either.

The design **nests max-affine exactly**: ``width=1, depth=1, group=K`` is
``min_k(a_k . w + c_k)`` and nothing else. Wider and deeper generalises it.

Input coordinate is ``w = min(u/s, W_CAP)`` from :mod:`cfs.surrogate.picnn_u`,
for the reason documented there — the target is concave in ``u`` and *not* in
``x``, so a head constrained in ``x`` cannot represent it.
"""

from __future__ import annotations

import logging

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from cfs.surrogate.picnn import _softplus_inv
from cfs.surrogate.picnn_u import INPUT_TRANSFORM, W_CAP, to_diag  # noqa: F401

LOGGER = logging.getLogger(__name__)

DEFAULT_GROUP = 8
DEFAULT_TEMP = 0.01


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


def with_temp(heads: GroupMaxHead, temp: float) -> GroupMaxHead:
    """Same head, different temperature.

    ``temp`` is a static field, so it lives in the treedef: ``dataclasses.replace``
    re-enters the custom ``__init__`` (which rebuilds the weights from a key),
    ``eqx.tree_at`` only reaches leaves, and ``copy.copy`` re-runs the vmapped
    constructor. Rebuilding the treedef and re-hanging this head's leaves on it is
    the one route that touches neither. Retraces the step, so call it a few times
    per run, not every epoch."""
    # `None` is a leaf here: Adam's moments are the *filtered* head, whose static
    # leaves are None, and dropping them would shift every remaining leaf.
    leaves = jax.tree_util.tree_flatten(heads, is_leaf=lambda z: z is None)[0]
    # A treedef carries the static fields but no shapes, so a one-input head of the
    # same depth donates a structurally identical one with the new temperature.
    like = GroupMaxHead(
        jax.random.PRNGKey(0),
        1,
        jnp.ones((1,), bool),
        width=1,
        depth=len(heads.wx),
        group=heads.group,
        temp=temp,
    )
    return jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(like), leaves)


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


def rank_by_territory(
    g_w: np.ndarray,
    w: np.ndarray,
    mu: np.ndarray,
    mask: np.ndarray,
    w_eval: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Row indices ordered by **Level 1 dominance** — SDDP's cut-selection rule.

    Our tangent model ``mu_hat(w) = min_j [mu_j + pi_j.(w - w_j)]`` is exactly the
    outer approximation SDDP maintains for a Bellman value function, sign-flipped:
    their *cuts* are our label tangents, their *trial points* are our media, and
    ``--gm-group K`` is their cut budget. That field has settled how to choose
    which cuts to keep, and this is their rule.

    A cut is **useless** when dropping it changes the approximation nowhere on the
    domain. Deciding that exactly costs one LP per cut (Pfeiffer, Apparigliato &
    Auchapt 2012, the *test of usefulness*) and is too slow to run routinely. The
    **territory algorithm** -- identical in its selection to de Matos, Philpott &
    Finardi's *Level 1 dominance* -- replaces "everywhere on the domain" with "at
    the trial points actually visited": each cut owns the points where it is the
    active one, and a cut whose territory is empty is dropped. Pure evaluations, no
    LP. It can drop a cut that would be useful on a region containing no visited
    point; in SDDP that is safe because the cut can be recomputed, and here because
    :func:`reanchor` reinstalls tangents mid-training.

    Because we score every cut at every point in one pass and keep, per point, only
    the index of its active cut, this is the *limited memory* variant (Guigues
    2017): memory is O(points), not O(cuts x points).

    **The point set is the whole lever, and it is ours to choose.** Ranking over
    the training rows reproduces the design's own distribution; ranking over
    community-regime media puts planes where §8.1 evaluates, which is why K above
    1000 was inert -- more planes, still in the wrong place. Measured in SDDP
    (Pfeiffer et al., 19-dimensional state, 500 iterations): territory selection
    cuts the model from 490 to 220 cuts per stage, and the combination with the
    exact test to 55, with the forward cost decreasing *at the same rate* as with
    no selection at all. The count was never the lever there either.

    ``w_eval`` is ``(P, M)`` trial points in the head's own ``w`` coordinate.
    Returns ``(order, territory)``: row indices sorted by territory size, largest
    first and empty territories last, plus the per-row point count.
    """
    ok = np.flatnonzero(valid)
    territory = np.zeros(len(w), dtype=int)
    if ok.size == 0 or len(w_eval) == 0:
        return ok, territory
    a = (g_w * mask)[ok]  # (J, M) non-negative slopes
    c = mu[ok] - np.einsum("km,km->k", a, w[ok])  # (J,) intercepts
    # min_j over cuts at every trial point, in chunks: (P, J) is P*J floats.
    step = max(1, int(2e7 // max(len(ok), 1)))
    for lo in range(0, len(w_eval), step):
        active = np.argmin(w_eval[lo : lo + step] @ a.T + c, axis=1)
        np.add.at(territory, ok[active], 1)
    # Largest territory first; ties by row order so the result is deterministic.
    order = ok[np.argsort(-territory[ok], kind="stable")]
    return order, territory


def valid_cuts(g_w, w, mu, mask, valid, tol: float = 1e-5) -> np.ndarray:
    """Which label tangents are genuine outer approximations over the training rows.

    A supporting hyperplane of a concave function is >= that function everywhere,
    so a tangent that reads *below* a labelled row is proof its dual is not a
    supergradient. That happens here for a measured reason: at a kink the stored
    dual is one-sided (§13.11 finite-differenced `theta = 0` duals at 0.16-0.54 of
    the true derivative), and a too-shallow slope falls away from its own anchor.
    On `labels_i3` **34% of CP000139.1's 3981 usable tangents are invalid**, by up
    to 2.85 `mu_scale` units.

    They are not harmless, because :func:`repair_intercepts` restores validity by
    *lifting* the offending plane -- which is correct globally and ruins it in its
    own territory. That is the whole of M17: the seeded head's hard min in that
    organism's mid-`mu` band is exact (0.9845 against a true 0.9845) and the
    repaired one reads **2.85**, with 239 planes lifted and 87% of the total lift
    forced by **five** rows.

    **Dropping them is SDDP's own rule (never add an invalid cut), it is free
    (2628 valid cuts against a budget of 1000) -- and it is REFUTED, 2026-09-09.**
    ``--gm-valid-cuts`` does exactly what it was built to do: the repair's
    per-plane lift vanishes (drop uniform 0.000489-0.000494, i.e. the smoothing
    alone, against a mean of 0.59 before). The band moves by **0.0001**, this
    organism goes cosine 0.895 -> **0.847** and value R2 0.318 -> **0.229**, and
    the roster worst 0.8955 -> **0.8234**.

    The reason is that no subset of these tangents is right everywhere, and it is
    one measurement (`20hm_bands/m17_bands.py`, median relative error per `mu/max`
    band on CP000139.1):

    ======================  ===============  ===============
    cut model over          `mu/max` <= 0.25   `mu/max` 0.5-0.75
    ======================  ===============  ===============
    all 3981 tangents       -0.0000          **-0.6707**
    the 2628 valid ones     **+0.1527**      +0.0000
    ======================  ===============  ===============

    So the invalid planes are *load-bearing*: they are the only reason the head is
    tight at high `mu`, and lifting them is the only reason it is tight there
    after selection. E1's "the labels are sufficient" verdict for this organism
    was taken in the mid-`mu` band alone -- **read the cut model over every band.**

    Neither the violation rate nor its size predicts the gate over the roster
    (Spearman +0.18 / -0.24 against cosine, p >= 0.29): DACTBY01 violates by 1.71
    and scores 0.9896 / R2 0.9992. Ninth refuted proxy.

    Kept, default off, for the negative result and because the mask itself is the
    cheapest statement of how far the labels are from concave.

    ``tol`` is *relative* to ``1 + |mu|``: the planes are evaluated as a float32
    matmul whose rounding differs by backend, and an absolute 1e-6 rejected 0.3%
    of exact tangents on macOS and 11% on the x86 CI runner (|mu| ~ 13, so the
    float32 noise is ~1e-5). Genuine violations are 0.1-2.85 here.

    Returns a bool mask over rows, ``valid`` AND outer-approximating.
    """
    ok = np.flatnonzero(valid)
    out = np.zeros(len(w), dtype=bool)
    if ok.size == 0:
        return out
    a = (g_w * mask)[ok]
    c = mu[ok] - np.einsum("km,km->k", a, w[ok])
    step = max(1, int(2e7 // max(len(w), 1)))
    for lo in range(0, len(ok), step):
        blk = np.asarray(jnp.asarray(a[lo : lo + step]) @ jnp.asarray(w).T)
        slack = blk + c[lo : lo + step, None] - mu[None, :]
        out[ok[lo : lo + step]] = (slack + tol * (1.0 + np.abs(mu))[None, :]).min(1) >= 0
    return out


def _softplus_inv_np(a: np.ndarray) -> np.ndarray:
    """``softplus^-1``, stable: ``expm1`` overflows at ``a ~ 88`` in float32 and the
    head is then seeded with inf weights and NaN curvature. Real label tangents
    reach it -- 3/21 organisms on ``20hm_bands`` -- and softplus is the identity to
    float precision well before that."""
    return np.where(a > 30.0, a, np.log(np.expm1(np.clip(a, 1e-9, 30.0))))


def _tangent_planes(g_w, w, mu, mask):
    """Rows -> ``(a, c)`` with ``mu ~ min_k(a_k . w - (-c_k))``.

    pre-act_j = ``-softplus(wx_j).y + b_j`` and the group max is ``max_j``, so
    ``mu = softplus(out_z) * min_j(a_j.y - b_j) + ...`` and ``b_j = -c_j``.
    """
    a = g_w * mask  # >= 0
    return a, mu - np.einsum("km,km->k", a, w)


def init_from_tangents(
    heads: GroupMaxHead,
    ds,
    seed: int = 0,
    select: str = "active-set",
    trial_points: np.ndarray | None = None,
    only_valid: bool = False,
) -> GroupMaxHead:
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

    ``select`` picks which tangents fill the budget. ``"active-set"`` buckets rows
    by their dual's support pattern (:func:`rank_by_active_set`), a *proxy* for
    which regimes occur. ``"level1"`` is SDDP's own answer
    (:func:`rank_by_territory`): keep the cuts that are the active minimum at some
    trial point, which is the exact version of what that proxy approximates.

    ``trial_points`` is ``(G, P, M)`` in the ``w`` coordinate, one point set per
    organism — the medium distribution the head will be *evaluated* on, which for
    §8.1 is community-regime media rather than the design's own. Default: the
    organism's training rows, which is the closest Level 1 analogue of SDDP's
    "points the forward pass actually visited".
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
        gv = ds.gvalid_train[i]
        if only_valid:
            gv = valid_cuts(g_w, w, mu, ds.mask[i], gv)
            LOGGER.info(
                "%s: %d of %d usable tangents are valid outer approximations",
                ds.genome_ids[i],
                int(gv.sum()),
                int(ds.gvalid_train[i].sum()),
            )
        if select == "level1":
            pts = w if trial_points is None else np.asarray(trial_points[i])
            idx, territory = rank_by_territory(g_w, w, mu, ds.mask[i], pts, gv)
            LOGGER.info(
                "%s: level1 keeps %d cuts with a non-empty territory of %d usable, "
                "over %d trial points; budget %d",
                ds.genome_ids[i],
                int((territory > 0).sum()),
                int(gv.sum()),
                len(pts),
                n_slots,
            )
        else:
            idx = rank_by_active_set(g_w, gv)
        if idx.size == 0:
            continue
        if idx.size < n_slots:  # too few usable rows: cycle, then jitter the rest
            idx = np.concatenate([idx, rng.choice(idx, n_slots - idx.size)])
        idx = idx[:n_slots]
        a, c = _tangent_planes(g_w[idx], w[idx], mu[idx], ds.mask[i])
        wx0[i] = _softplus_inv_np(a)
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


def _unit_usage(head: GroupMaxHead, w: np.ndarray) -> np.ndarray:
    """Mean softmax weight of each first-layer unit — how often it is its group's max.

    ``(n_slots,)``, summing to ``width`` over the whole layer. A unit at ~0 never
    reaches the output, so nothing it holds is being used.
    """
    a = -jax.nn.softplus(head.wx[0]) @ (jnp.asarray(w) * head.mask).T + head.b[0][:, None]
    p = jax.nn.softmax(a.reshape(-1, head.group, a.shape[-1]) / head.temp, axis=1)
    return np.asarray(p.reshape(a.shape[0], -1).mean(-1))


def reanchor(heads: GroupMaxHead, ds, frac: float = 0.1, seed: int = 0) -> tuple:
    """Re-seed the least-used first-layer planes from the worst-fit rows' tangents.

    :func:`init_from_tangents` fixes *initialisation*; it does not stop a plane
    going dead during training, and that is the failure that sets the gate.
    Measured on the 20000-media labels (seeded ``groupmax-u``, width 1 depth 1
    K=1000, held-out media): on the cells that fail, the model gets the limiting
    metabolite right on ~92% of rows but predicts ``d(mu)/dw`` of **1e-9** against
    a true 1.3e-3, while fitting the *value* on those same rows to 0.1%. Planes
    with the right slope are present -- 59-210 of the 1000, seeded from those rows'
    own duals -- but they sit **0.2-1.5 above the active minimum**, i.e. 15-40x the
    temperature, so the softmax gives them weight ~1e-7. The Sobolev term cannot
    reach them: its only path is that same exponentially-closed softmax.

    Nor is it coverage or capacity -- train and held-out cosine agree on the failing
    cells (0.988 / 0.987 and 0.521 / 0.484) -- and it *roves*: at fixed
    hyperparameters over five seeds the collapse lands on ``EX_o2_e`` in one run
    (0.500) and ``EX_thr__L_e`` in another (0.613), both well covered, both fine in
    the other seeds. Organism cosine varies by 0.036 across those seeds.

    So this is the alternation step of LSPA/CAP-style max-affine fitting, which the
    module docstring wrongly dismissed as unnecessary "because we have the duals".
    Having the duals makes it *cheaper*, not redundant: the planes to install are
    the labels' own tangents rather than a least-squares refit.

    Rows are picked worst-cosine-first in ``u`` space (the gate's coordinate) and
    de-duplicated by active set, so a budget of slots is not spent on one regime.
    Returns ``(heads, slots)``; ``slots`` is ``(G, n)`` of the indices overwritten,
    for the caller to clear the optimiser moments on.
    """
    n_slots = heads.wx[0].shape[1]
    n = max(1, int(round(frac * n_slots)))
    G = ds.x_train.shape[0]
    wx0 = np.asarray(heads.wx[0]).copy()
    b0 = np.asarray(heads.b[0]).copy()
    slots = np.zeros((G, n), dtype=int)
    rng = np.random.default_rng(seed)

    for i in range(G):
        head = organism(heads, i)
        x = ds.x_train[i]
        s = ds.x_scale[i]
        w = np.asarray(to_diag(jnp.asarray(x)))
        g_w = ds.g_train[i] * (1.0 - x) ** 2 / ds.mu_scale[i]
        mu = ds.mu_train[i] / ds.mu_scale[i]
        pred = np.asarray(jax.vmap(jax.grad(head.on_w))(jnp.asarray(w))) * ds.mask[i]
        # The gate is scored in u space, so rank the rows there: dmu/du = dmu/dw / s.
        t, p = (g_w * ds.mask[i]) / s, pred / s
        tn, pn = np.linalg.norm(t, axis=-1), np.linalg.norm(p, axis=-1)
        usable = np.asarray(ds.gvalid_train[i]) & (tn > 0)
        cos = np.where(usable, (p * t).sum(-1) / np.maximum(pn * tn, 1e-30), np.inf)

        # Worst first, one row per active set while distinct patterns last, then
        # worst-first regardless: on a dense-support organism the pattern dedup can
        # collapse to a single pick, and spending the whole budget on copies of one
        # row installs one plane where the caller asked for `n`.
        order = [int(j) for j in np.argsort(cos) if usable[j]]
        if not order:
            continue
        seen: set[tuple] = set()
        pick, rest = [], []
        for j in order:
            k = tuple(np.flatnonzero(g_w[j] > 0))
            (rest if k in seen else pick).append(j)
            seen.add(k)
            if len(pick) == n:
                break
        idx = np.asarray((pick + rest + order)[:n])
        # Overwrite the units that are doing the least work. Ties are common at
        # init, so break them randomly rather than always taking the low indices.
        dead = np.lexsort((rng.random(n_slots), _unit_usage(head, w)))[:n]
        a, c = _tangent_planes(g_w[idx], w[idx], mu[idx], ds.mask[i])
        wx0[i, dead] = _softplus_inv_np(a)
        b0[i, dead] = -c
        slots[i] = dead

    heads = eqx.tree_at(lambda h: (h.wx[0], h.b[0]), heads, (jnp.asarray(wx0), jnp.asarray(b0)))
    return heads, slots


def repair_intercepts(heads: GroupMaxHead, ds, local: bool = False) -> GroupMaxHead:
    """Restore the outer-approximation invariant: no plane below any training label.

    A min of *supporting* hyperplanes of a concave function is an upper bound
    everywhere, so a trained head that reads **low** is proof it has left the
    family: Adam moves both slope and intercept, and nothing re-imposes validity.
    (Measured: on `p4` training rows in the bottom 5% of `mu` the head
    under-predicts 53-68%, i.e. it is below labels it was fit on. The other
    candidate mechanism, the softmin's `T*ln(K)` downward gap, is refuted -- at the
    failing n=21 medium, re-evaluating at `T -> 1e-6` moves `mu_hat` by 0.008.)

    This is SDDP's cut-validity invariant, which that literature keeps by never
    modifying a cut once added. We do modify them, so we restore it afterwards.
    With the slopes held, the tightest valid intercept per plane is a closed form:
    the head is ``mu_hat(y) = min_j[(c a_j + q).y - c b_j - out_b]``, and plane `j`
    is valid iff ``c b_j <= min_r[(c a_j + q).y_r - out_b - mu_r]``. Taking that
    bound with equality gives the *lowest* valid upper bound for those slopes --
    so it is the exact optimum of the intercept LP, not a heuristic, and it
    tightens over-predicting planes in the same pass.

    **Exact only at width=1, depth=1**, the production config, where the head
    really is a min of affine functions; wider or deeper it is a non-negative sum
    of group-wise minima and no per-plane intercept has this meaning. Other shapes
    are returned untouched.

    ``local=True`` repairs each plane over **its own territory** — the training
    rows where it is currently the active minimum — instead of over every row.
    The global rule lifts a plane above labels it was never meant to bind at: on
    `CP000139.1` a plane anchored in the `mu ~ 1` band has to clear labels at
    `mu ~ 4.4`, which took that band from a true 0.98 to a predicted 2.85 (+1.87)
    while every other band stayed exact to 1e-4 — 22.5% of its held-out rows, and
    the whole of its 0.895 cosine / 0.318 value R2. The repair is global where
    the cut is local.

    **Measured and REFUTED, 2026-09-09; default off, kept for the negative
    result.** It does what it was built for — `CP000139.1` cosine 0.895 -> 0.964 —
    and is a roster disaster: median grad cosine **0.9859 -> 0.7764**, median
    value R2 **0.9997 -> -0.665**, worst R2 -3.45.

    The reason is structural and kills the idea rather than the implementation.
    Training-row validity is restored by the *uniform* lift below, so a locally
    repaired plane that sits under the truth off its own territory is paid for by
    raising **every** plane: the lift goes from **0.00047** (global rule, where it
    only ever compensates the softmin gap) to **1.49 median / 2.54 max** — 3000x,
    in `mu_scale` units. A global correction cannot preserve a local repair.
    Rescuing it would need the lift to be local too, and then nothing enforces
    validity between territories.
    """
    if len(heads.wx) != 1 or heads.out_z.shape[1] != 1:
        LOGGER.warning("repair_intercepts: head is not width=1 depth=1 — skipped")
        return heads

    b0 = np.asarray(heads.b[0]).copy()
    for i in range(ds.x_train.shape[0]):
        head = organism(heads, i)
        w_rows = np.asarray(to_diag(jnp.asarray(ds.x_train[i])))  # (N, M)
        y = w_rows * ds.mask[i]
        mu = ds.mu_train[i] / ds.mu_scale[i]  # (N,)
        c = float(jax.nn.softplus(head.out_z)[0])
        if c <= 1e-12:  # a collapsed output gain carries no planes to repair
            continue
        s = c * np.asarray(jax.nn.softplus(head.wx[0])) + np.asarray(
            jax.nn.softplus(head.out_x)
        )  # (K, M)
        # `min_r` is over rows *jointly* with `mu_r`, so the label cannot be split
        # out of the matmul: (K, N) is the whole point.
        pre = np.asarray(jnp.asarray(s) @ jnp.asarray(y).T) - float(head.out_b)  # (K, N)
        slack = pre - mu[None, :]
        if local:
            # Each plane is constrained only by the rows it actually binds at. A
            # plane that is nowhere the active minimum has no territory and keeps
            # the global rule, which is the conservative choice: it is exactly the
            # plane that could start binding once its neighbours move.
            owner = pre.argmin(axis=0)  # (N,) the active plane at each row
            own = np.full_like(slack, np.inf)
            rows = np.arange(len(mu))
            own[owner, rows] = slack[owner, rows]
            new = own.min(axis=1) / c
            empty = ~np.isfinite(new)
            new[empty] = slack.min(axis=1)[empty] / c
        else:
            new = slack.min(axis=1) / c

        # The head is the *smoothed* min, which sits below the hard one by up to
        # `c*T*ln(K)` -- so per-plane validity is necessary and not sufficient, and
        # training had been paying for that gap in the intercepts. Lowering every
        # `b_j` by the same delta shifts all pre-activations together, hence lifts
        # the smoothed head by exactly `c*delta`, so the minimal uniform lift that
        # restores validity is one evaluation away. Doing it this way rather than
        # assuming the worst-case `T*ln(K)` keeps the bound tight: the realised gap
        # is `T*ln(#near-active)`, which is far smaller.
        shifted = eqx.tree_at(lambda h: h.b[0], head, jnp.asarray(new))
        v = float(np.max(mu - np.asarray(jax.vmap(shifted.on_w)(jnp.asarray(w_rows)))))
        if v > 0:
            new = new - v / c
        LOGGER.info(
            # The median hides this: on CP000139.1 the global repair's median drop
            # is 0.0005 and its *mean* is 0.59, with 299 of 1000 planes moved by
            # >0.1 and one by 2.86 -- which is the whole of that organism's
            # +1.87 band error. Report the tail.
            "%s: repaired %d of %d planes (%s), drop median %.3g / mean %.3g / "
            "max %.3g, smoothing lift %.3g",
            ds.genome_ids[i],
            int((new < b0[i] - 1e-9).sum()),
            b0[i].size,
            "territory" if local else "global",
            float(np.median(b0[i] - new)),
            float(np.mean(b0[i] - new)),
            float(np.max(b0[i] - new)),
            max(v, 0.0) / c,
        )
        b0[i] = new
    return eqx.tree_at(lambda h: h.b[0], heads, jnp.asarray(b0))
