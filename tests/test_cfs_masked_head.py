"""§14 mask-conditioned Head A: the M18 invariants, and the mask label path.

The three structural invariants (§14.2) are what make this parameterisation worth
having rather than a redesign, so they are checked rather than asserted in prose:
reduction to the unconditioned head at init, concavity in ``w``, and monotonicity in
``w`` — each for *random masks*, not just the all-present one.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", False)

from cfs.sampling.masks import (  # noqa: E402
    index_hash,
    mask_to_knockouts,
    sample_knockout_masks,
)
from cfs.surrogate.groupmax import GroupMaxHead  # noqa: E402
from cfs.surrogate.masked import (  # noqa: E402
    MaskedGroupMaxHead,
    mask_monotonicity_violation_rate,
    stack_heads,
)

M = 6  # exchanges
R = 12  # maskable reactions


def _head(seed: int = 0, **kw):
    return MaskedGroupMaxHead(
        jax.random.PRNGKey(seed),
        M,
        jnp.ones((M,), bool),
        R,
        width=8,
        depth=2,
        group=4,
        **kw,
    )


def _rows(n: int, seed: int = 1):
    key = jax.random.PRNGKey(seed)
    kw, kz = jax.random.split(key)
    w = jax.random.uniform(kw, (n, M), minval=0.0, maxval=50.0)
    z = (jax.random.uniform(kz, (n, R)) > 0.3).astype(jnp.float32)
    return w, z


def test_reduces_to_unconditioned_head_at_init():
    """`e(z) == 0` at init, so the head IS its parent — the whole point of §14.2.

    Without this the measured `--gm-init labels` / `DEFAULT_TEMP` / `W_CAP` results
    would have to be re-earned on the conditioned head.
    """
    head = _head()
    parent = head.head
    w, z = _rows(16)
    got = jax.vmap(head.on_w)(w, z)
    want = jax.vmap(parent.on_w)(w)
    assert jnp.allclose(got, want, atol=1e-6)
    # ...and it is the mask channel that is inert, not the mask input being ignored:
    # every mask gives the parent's value at init.
    flipped = 1.0 - z
    assert jnp.allclose(jax.vmap(head.on_w)(w, flipped), want, atol=1e-6)


def _trained_like(head):
    """Non-zero `out_w`, i.e. a head whose mask channel is live.

    Standing in for training: the invariants must hold for *any* parameters, so
    perturbing is a stronger check than training and needs no labels.
    """
    key = jax.random.PRNGKey(7)
    return jax.tree.map(
        lambda p: p + 0.5 * jax.random.normal(key, p.shape) if hasattr(p, "shape") else p,
        head,
    )


def test_mask_changes_the_value_once_live():
    head = _trained_like(_head())
    w, z = _rows(32)
    base = jax.vmap(head.on_w)(w, z)
    other = jax.vmap(head.on_w)(w, 1.0 - z)
    assert not jnp.allclose(base, other, atol=1e-4)


@pytest.mark.parametrize("live", [False, True])
def test_concave_in_w_for_random_masks(live):
    """Structural: a min of affine-in-`w` pieces is concave however the coefficients
    were produced. Measured in `w`, never in the model's input space (M3's rule)."""
    head = _trained_like(_head()) if live else _head()
    key = jax.random.PRNGKey(3)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    n = 256
    w1 = jax.random.uniform(k1, (n, M), minval=0.0, maxval=50.0)
    w2 = jax.random.uniform(k2, (n, M), minval=0.0, maxval=50.0)
    z = (jax.random.uniform(k3, (n, R)) > 0.3).astype(jnp.float32)
    t = jax.random.uniform(k4, (n, 1))
    mid = t * w1 + (1 - t) * w2
    f = jax.vmap(head.on_w)
    chord = t[:, 0] * f(w1, z) + (1 - t[:, 0]) * f(w2, z)
    # Concave: f(mid) >= chord. Float32 slack only.
    assert jnp.all(f(mid, z) >= chord - 1e-4)


@pytest.mark.parametrize("live", [False, True])
def test_monotone_non_decreasing_in_w(live):
    """Relaxing an uptake bound can only enlarge the LP's feasible set."""
    head = _trained_like(_head()) if live else _head()
    w, z = _rows(128, seed=5)
    f = jax.vmap(head.on_w)
    bumped = w + jax.random.uniform(jax.random.PRNGKey(9), w.shape, minval=0.0, maxval=5.0)
    assert jnp.all(f(bumped, z) >= f(w, z) - 1e-4)


def test_mask_monotonicity_is_measured_not_guaranteed():
    """§14.2: monotonicity in `z` is a gate, not a structural property.

    At init the rate must be 0 (the mask channel is inert). The diagnostic has to
    actually look at dropped bits, so a live head is allowed to violate it — that is
    the number M18 reports and the argument for §14.3.
    """
    w, z = _rows(64, seed=11)
    at_init = mask_monotonicity_violation_rate(_head(), w, z, jax.random.PRNGKey(2))
    assert float(at_init) == 0.0
    live = mask_monotonicity_violation_rate(_trained_like(_head()), w, z, jax.random.PRNGKey(2))
    assert 0.0 <= float(live) <= 1.0


def test_stacked_heads_slice_per_organism():
    heads = stack_heads(jax.random.PRNGKey(0), 3, M, np.ones((3, M), bool), R, width=8, depth=2)
    assert heads.embed_w.shape == (3, R, 16)
    assert isinstance(heads.head, GroupMaxHead)


# --------------------------------------------------------------------------- #
# mask sampling (P18) and the knockout label path
# --------------------------------------------------------------------------- #


def test_thin_shell_sampling_only_touches_candidates():
    ids = [f"R{i}" for i in range(R)]
    cands = ids[:4]
    masks = sample_knockout_masks(ids, cands, n_masks=200, seed=0, k_max=3, p_unmasked=0.1)
    assert masks.shape == (200, R)
    # Nothing outside the candidate set is ever turned off.
    assert np.all(masks[:, 4:] == 1.0)
    # k in 1..k_max, and some rows are the unedited network.
    removed = (masks == 0).sum(axis=1)
    assert removed.max() <= 3
    assert (removed == 0).any() and (removed > 1).any()


def test_unknown_candidate_raises():
    with pytest.raises(ValueError, match="not in the mask index"):
        sample_knockout_masks([f"R{i}" for i in range(R)], ["nope"], n_masks=4)


def test_index_hash_is_order_sensitive():
    ids = [f"R{i}" for i in range(R)]
    assert index_hash(ids) == index_hash(list(ids))
    assert index_hash(ids) != index_hash(list(reversed(ids)))


def _toy_model():
    """Growth needs uptake of 'a' through transport T — so T is essential."""
    from cobra import Metabolite, Model, Reaction

    m = Model("toy")
    a_e, a_c, bio = (
        Metabolite(i, compartment=c) for i, c in [("a_e", "e"), ("a_c", "c"), ("bio_c", "c")]
    )
    ex_a = Reaction("EX_a_e", lower_bound=-10.0, upper_bound=1000.0)
    ex_a.add_metabolites({a_e: -1})
    t = Reaction("T", lower_bound=0.0, upper_bound=1000.0)
    t.add_metabolites({a_e: -1, a_c: 1})
    shunt = Reaction("T2", lower_bound=0.0, upper_bound=1000.0)  # a redundant route
    shunt.add_metabolites({a_e: -1, a_c: 1})
    grow = Reaction("BIOMASS", lower_bound=0.0, upper_bound=1000.0)
    grow.add_metabolites({a_c: -1, bio: 1})
    ex_bio = Reaction("EX_bio_e", lower_bound=0.0, upper_bound=1000.0)
    ex_bio.add_metabolites({bio: -1})
    m.add_reactions([ex_a, t, shunt, grow, ex_bio])
    m.objective = "BIOMASS"
    return m


def test_knockout_labels_are_monotone_and_index_excludes_boundaries():
    """V9 in miniature: the label side must rank knockouts correctly by construction."""
    pytest.importorskip("cobra")
    from cfs.groundtruth.solve import solve
    from cfs.sampling.masks import maskable_reactions

    m = _toy_model()
    ids = maskable_reactions(m)
    assert ids == ["T", "T2"]  # exchanges and biomass excluded (P22)

    conc = {"EX_a_e": 1e3}
    base = solve(m, conc, alpha=1.0, eps=1e-3).mu_max
    # Redundant single knockout: unchanged. Both routes gone: lethal. This pair is
    # the pathway-AND structure that k=1 sampling alone cannot show (P18).
    one = solve(m, conc, alpha=1.0, eps=1e-3, knockouts=["T"]).mu_max
    both = solve(m, conc, alpha=1.0, eps=1e-3, knockouts=["T", "T2"]).mu_max
    assert base > 0
    assert one == pytest.approx(base, rel=1e-6)
    assert both == pytest.approx(0.0, abs=1e-9)
    # The invariant the head is held to: removal never raises mu.
    assert one <= base + 1e-9 and both <= one + 1e-9
    # Bounds are restored by the context manager, so the model is reusable.
    assert m.reactions.get_by_id("T").upper_bound == 1000.0


def test_mask_to_knockouts_round_trips():
    ids = [f"R{i}" for i in range(R)]
    mask = np.ones(R, dtype=np.float32)
    mask[[2, 5]] = 0.0
    assert mask_to_knockouts(ids, mask) == ["R2", "R5"]
