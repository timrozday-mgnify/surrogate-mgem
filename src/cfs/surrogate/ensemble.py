"""Deep ensemble over Head A, scored by **gradient** disagreement (§4.6).

The third and last acquisition signal, and the narrowest. Use it only for what
the other two cannot see.

Where the label budget should go can be read off the labels themselves, twice
over, without any model:

- :func:`cfs.sampling.design.limiting_scales` — the saturation ``u*`` at which a
  metabolite was observed to limit, which is where its sampling band belongs;
- :func:`cfs.sampling.design.topup_weights` — the *measured* held-out cosine per
  limiting metabolite, which is where the surrogate is actually wrong.

Both are blind to the same thing: a metabolite that **never limited** in the
previous run has no ``u*`` and no held-out error, so neither signal can point at
it. On the 21-genome roster that is 4-10 of each organism's 16-30 active
metabolites. Ensemble disagreement is the only signal left for those, because it
needs no labels at the point of interest — which is exactly the situation active
learning is for, and exactly where it is worth its cost.

**Disagreement on the gradient, not the value** — the legacy loop
(``surrogate_mgem.active``) scores ``std.mean(axis=1)`` of predicted growth, and
that is the wrong quantity here. The M3 gate is on the shadow-price *direction*,
and the two decouple in this data: value R2 sits at 0.62-0.72 while gradient
cosine sits at 0.66, and raising ``w_grad`` moves them in opposite directions. A
medium can have members agreeing perfectly on ``mu_max`` and disagreeing totally
on which metabolite binds. :func:`gradient_disagreement` scores mean pairwise
angle between members' predicted dual vectors, in ``u`` space — the same
coordinate as the gate, so the acquisition and the milestone measure the same
thing.

**Replete media are excluded, not ranked.** 23% of rows have an all-zero dual;
nothing pins a direction there, so members disagree maximally on a quantity that
carries no information. Ranking by raw angle would spend the whole batch on them.
Rows whose predicted gradients are all near-zero score 0 here.

An ensemble is cheap in this architecture: members are one more ``vmap`` axis on
:func:`~cfs.surrogate.picnn.stack_heads`, so ``E`` members over ``G`` organisms
cost about what ``G`` alone does. Concavity and monotonicity are per-member
structural properties, so every member is still a valid Head A.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from cfs.surrogate.data import ValueDataset
from cfs.surrogate.picnn import ValueHead
from cfs.surrogate.train import _du, train_value_heads


def train_ensemble(
    ds: ValueDataset, *, n_models: int = 5, seed: int = 0, **kwargs
) -> list[ValueHead]:
    """``n_models`` independently-seeded stacked heads (§4.6 acquisition ensemble).

    Deliberately a plain list rather than a further ``vmap`` axis: members differ
    only by seed, and keeping them separate means :func:`evaluate` and the
    checkpoint format apply to a member unchanged.
    """
    return [train_value_heads(ds, seed=seed + i, **kwargs) for i in range(n_models)]


def gradient_disagreement(members: list[ValueHead], x, x_scale, mask, floor: float = 1e-12):
    """``(G, B)`` mean pairwise angle between members' predicted duals, in ``u`` space.

    0 where the members agree on direction (or all predict ~no gradient), rising
    to ``pi`` for total disagreement. This is ``1 - cos`` averaged over member
    pairs, so it is on the same scale as the gate's shortfall.
    """
    du = _du(jnp.asarray(x), jnp.asarray(x_scale))
    m = jnp.asarray(mask)[:, None, :]
    g = jnp.stack([_grads(h, x) * du * m for h in members])  # (E, G, B, M)
    n = jnp.linalg.norm(g, axis=-1, keepdims=True)
    # Replete media: every member predicts ~nothing, so there is no direction to
    # disagree about. Normalising would amplify numerical dust into a top score.
    live = jnp.squeeze(n, -1) > floor
    u = g / jnp.maximum(n, floor)
    e = len(members)
    total = jnp.zeros(g.shape[1:3])
    pairs = 0
    for i in range(e):
        for j in range(i + 1, e):
            cos = jnp.sum(u[i] * u[j], axis=-1)
            ok = live[i] & live[j]
            total = total + jnp.where(ok, 1.0 - cos, 0.0)
            pairs += 1
    return total / max(pairs, 1)


def _grads(heads: ValueHead, x):
    """``(G, B, M)`` gradients of one stacked head — the shadow prices."""
    from cfs.surrogate.picnn import batched_value_and_grad

    return batched_value_and_grad(heads, jnp.asarray(x))[1]


def unmeasured_metabolites(
    diagnostics: dict, subspaces: dict[str, list[str]]
) -> dict[str, list[str]]:
    """Per organism, the ``A_i`` metabolites with no held-out evidence at all.

    These never limited in the previous run, so they carry neither a ``u*`` for
    :func:`~cfs.sampling.design.limiting_scales` nor an error for
    :func:`~cfs.sampling.design.topup_weights`. They are the ensemble's job.
    """
    out = {}
    for gid, d in diagnostics["per_organism"].items():
        seen = set(d.get("per_limiting_metabolite", {}))
        out[gid] = [ex for ex in subspaces.get(gid, []) if ex not in seen]
    return out


def rank_media(members: list[ValueHead], x, x_scale, mask, k: int = 256) -> list[np.ndarray]:
    """Per organism, indices of the ``k`` most disagreed-on media in ``x``.

    Candidates are scored, not proposed, here: generate them with
    :func:`cfs.sampling.design.sample_media` so they come from the same
    distribution as the labelled set, then solve only the returned rows.
    """
    d = np.asarray(gradient_disagreement(members, x, x_scale, mask))
    return [np.argsort(-row)[:k] for row in d]
