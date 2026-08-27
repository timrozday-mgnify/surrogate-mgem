"""M3 Head A: the structural checks, on synthetic data (no parquet, no solver).

Fits the stacked value head to a known concave function whose gradient is known
analytically, then asserts the milestone gate on it. This is what fails if the
ICNN weight constraint, the negation, the mask, or the autodiff gradient wiring
breaks — none of which shows up as a bad training loss.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
pytest.importorskip("equinox")

from cfs.surrogate.data import ValueDataset, _organism_arrays  # noqa: E402
from cfs.surrogate.picnn import (  # noqa: E402
    batched_value,
    batched_value_and_grad,
    stack_heads,
)
from cfs.surrogate.train import evaluate, train_value_heads  # noqa: E402

G, M, N = 3, 6, 400


def _synthetic() -> ValueDataset:
    """mu = sum_m w_m x_m / (x_m + a_m) — concave, *non-decreasing* (which the head
    now is by construction), gradient known exactly.

    The half-saturation ``a_m`` is a property of the **metabolite**, shared across
    organisms, while the weight ``w_m`` is per organism. That is the assumption the
    shared DeepSet trunk encodes (one ``phi`` per metabolite for the whole roster,
    a per-organism ``rho``), so this target is expressible by every architecture in
    the registry and a failure here is a wiring fault rather than a capacity one.
    """
    rng = np.random.default_rng(0)
    mask = np.ones((G, M), dtype=bool)
    mask[1, -2:] = False  # one organism missing two metabolites
    x = rng.uniform(0, 1, (G, N, M)).astype(np.float32) * mask[:, None, :]
    a = rng.uniform(0.2, 0.8, (1, 1, M)).astype(np.float32)
    w = rng.uniform(0.5, 2.0, (G, 1, M)).astype(np.float32) * mask[:, None, :]
    mu = np.sum(w * x / (x + a), axis=-1)
    g = (w * a / (x + a) ** 2) * mask[:, None, :]
    gvalid = np.ones((G, N), dtype=bool)
    n_val = N // 5
    return ValueDataset(
        genome_ids=[f"g{i}" for i in range(G)],
        exchanges=[f"EX_{m}" for m in range(M)],
        mask=mask,
        x_train=x[:, n_val:],
        mu_train=mu[:, n_val:],
        g_train=g[:, n_val:],
        gvalid_train=gvalid[:, n_val:],
        x_val=x[:, :n_val],
        mu_val=mu[:, :n_val],
        g_val=g[:, :n_val],
        gvalid_val=gvalid[:, :n_val],
        mu_scale=mu.std(axis=1).astype(np.float32),
        x_scale=np.ones((G, M), np.float32),
        index_hash="test",
        rounds_present=[0],
    )


@pytest.mark.parametrize(
    "arch",
    [
        "icnn",
        "icnn-u",
        "deepset",
        "deepset-private",
        "deepset-u",
        "deepset-u-private",
        "groupmax-u",
    ],
)
def test_gate_on_a_known_concave_target(arch):
    ds = _synthetic()
    # The sign-constrained head converges slower than the unconstrained one did:
    # 600 epochs at lr 1e-2 stops at cosine 0.84 on this target, which is a
    # training budget, not a wiring fault.
    heads = train_value_heads(ds, arch=arch, width=64, depth=2, epochs=2000, batch=64, lr=3e-2)
    diag = evaluate(heads, ds, arch=arch)
    assert diag["passed"], diag["per_organism"]
    for gid, d in diag["per_organism"].items():
        # Structural for every concave arch: non-zero means a sign
        # reparameterisation is broken, not that the fit is poor.
        assert d["concavity_violation_rate"] == 0.0, gid


def test_unconstrained_mlp_trains_but_is_not_concave():
    """The ceiling measurement is wired up — and is *not* a usable head."""
    ds = _synthetic()
    heads = train_value_heads(ds, arch="mlp", width=64, depth=2, epochs=200, batch=64, lr=3e-3)
    diag = evaluate(heads, ds, arch="mlp")
    assert diag["arch"] == "mlp"
    assert all(np.isfinite(d["grad_cosine"]) for d in diag["per_organism"].values())


def test_untrained_head_is_concave_monotone_and_masked():
    """Concavity and monotonicity are structural, so they hold at initialisation."""
    mask = np.ones((2, M), dtype=bool)
    mask[0, 0] = False
    heads = stack_heads(jax.random.PRNGKey(1), 2, M, mask, width=16, depth=3)
    rng = np.random.default_rng(1)
    xa, xb = (rng.uniform(0, 1, (2, 64, M)).astype(np.float32) for _ in range(2))
    lam = 0.37
    mid = batched_value(heads, lam * xa + (1 - lam) * xb)
    chord = lam * batched_value(heads, xa) + (1 - lam) * batched_value(heads, xb)
    assert np.all(np.asarray(mid) >= np.asarray(chord) - 1e-5)

    # More nutrient never lowers mu_max: the input transform is concave, so the
    # composition is concave only if this holds.
    _, grad = batched_value_and_grad(heads, xa)
    assert np.all(np.asarray(grad * heads.mask[:, None, :]) >= 0.0)

    # A masked metabolite cannot move the output.
    x0 = np.zeros((2, 1, M), dtype=np.float32)
    x1 = x0.copy()
    x1[0, 0, 0] = 1.0
    assert np.allclose(batched_value(heads, x0)[0], batched_value(heads, x1)[0])


def test_icnn_u_is_concave_in_u_and_not_forced_concave_in_x():
    """The whole point of `icnn-u`: the constraint moves to ``w = u/s``.

    ``mu_max`` is concave in ``u`` and *not* in ``x`` (32-57% of label row pairs
    violate the ``x`` tangent test on real GEMs), so a head locked to concavity in
    ``x`` cannot represent it. This asserts the new head is concave where it should
    be — and that it has genuinely given up the ``x`` constraint rather than
    silently keeping it, which would make the arch a no-op.
    """
    from cfs.surrogate import picnn_u

    mask = np.ones((2, M), dtype=bool)
    mask[0, 0] = False
    heads = picnn_u.stack_heads(jax.random.PRNGKey(3), 2, M, mask, width=16, depth=3)
    rng = np.random.default_rng(3)
    # Sample in w, not x: mapping x -> w -> x back cancels in float32 near x = 1.
    wa, wb = (rng.uniform(0, 20, (2, 256, M)).astype(np.float32) for _ in range(2))
    lam = 0.37
    mid = picnn_u.batched_value_diag(heads, lam * wa + (1 - lam) * wb)
    chord = lam * picnn_u.batched_value_diag(heads, wa) + (1 - lam) * picnn_u.batched_value_diag(
        heads, wb
    )
    assert np.all(np.asarray(mid) >= np.asarray(chord) - 1e-4)

    # `x` is what the loss and the gate see; `to_diag` is the head's own map, so
    # the two entry points must agree.
    x = (wa / (1.0 + wa)).astype(np.float32)
    assert np.allclose(
        np.asarray(picnn_u.batched_value(heads, x)),
        np.asarray(picnn_u.batched_value_diag(heads, picnn_u.to_diag(x))),
        atol=1e-5,
    )

    # Monotone in every masked coordinate — still true, still required.
    _, grad = picnn_u.batched_value_and_grad(heads, x)
    assert np.all(np.asarray(grad * heads.mask[:, None, :]) >= -1e-6)

    # Not concave in x: some pair must violate, or the constraint never moved.
    xa, xb = (rng.uniform(0, 1, (2, 512, M)).astype(np.float32) for _ in range(2))
    midx = picnn_u.batched_value(heads, lam * xa + (1 - lam) * xb)
    chordx = lam * picnn_u.batched_value(heads, xa) + (1 - lam) * picnn_u.batched_value(heads, xb)
    assert np.any(np.asarray(midx) < np.asarray(chordx) - 1e-4)


def test_groupmax_nests_max_affine_and_keeps_curvature():
    """The two properties the head exists for.

    1. ``width=1, depth=1, group=K`` must BE ``min_k(a_k . w + c_k)`` — the exact
       form of an LP value function — and not merely resemble it.
    2. The smoothed max must keep a non-zero Hessian. A hard max has none inside a
       piece, which is P3 and the one thing §8's Newton cannot follow; ``temp`` is
       what buys the curvature back, so curvature must actually rise as T rises.
    """
    from cfs.surrogate import groupmax

    K, n_in = 6, 4
    mask = np.ones((1, n_in), dtype=bool)
    head = groupmax.organism(
        groupmax.stack_heads(
            jax.random.PRNGKey(7), 1, n_in, mask, width=1, depth=1, group=K, temp=1e-3
        ),
        0,
    )
    # Read the affine pieces straight off the parameters: out = softplus(out_z) *
    # gmax(-softplus(wx) w + b) - softplus(out_x) w, and mu = -out.
    a = np.asarray(jax.nn.softplus(head.wx[0]))  # (K, n_in), the slopes
    b = np.asarray(head.b[0])  # (K,)
    oz, ox = float(jax.nn.softplus(head.out_z)[0]), np.asarray(jax.nn.softplus(head.out_x))
    rng = np.random.default_rng(7)
    w = rng.uniform(0, 5, (32, n_in)).astype(np.float32)
    # mu = -oz * max_k(-a_k.w + b_k) + ox.w = min_k(oz*a_k.w - oz*b_k) + ox.w
    want = (oz * (w @ a.T) - oz * b).min(axis=1) + w @ ox
    got = np.asarray(jax.vmap(head.on_w)(w))
    assert np.allclose(got, want, atol=2e-3), np.abs(got - want).max()

    # Curvature is real and is controlled by `temp`, not incidental to it.
    def curv(t):
        h = groupmax.organism(
            groupmax.stack_heads(
                jax.random.PRNGKey(7), 1, n_in, mask, width=4, depth=2, group=4, temp=t
            ),
            0,
        )
        H = np.asarray(jax.vmap(jax.hessian(h.on_w))(w[:8]))
        return float(np.abs(np.linalg.eigvalsh(H)).max())

    lo, hi = curv(0.05), curv(1.0)
    assert lo > 0.0 and hi > lo, (lo, hi)


def _min_affine_dataset(K=5, n=300, M=4, seed=0):
    """A target that IS an LP value function: mu = min_k(a_k . w + c_k) in w space.

    Built directly so the label-tangent init has a known right answer — every row's
    dual is the active piece's coefficient vector, exactly as in a real solve.
    """
    from cfs.surrogate.picnn_u import to_diag

    rng = np.random.default_rng(seed)
    x = rng.uniform(0.05, 0.9, (1, n, M)).astype(np.float32)
    w = np.asarray(to_diag(jnp.asarray(x)))
    a = rng.uniform(0.1, 2.0, (K, M)).astype(np.float32)  # slopes >= 0: monotone
    c = rng.uniform(0.5, 3.0, (K,)).astype(np.float32)
    planes = w[0] @ a.T + c
    k = planes.argmin(1)
    mu = planes[np.arange(n), k][None]
    g_w = a[k][None]
    g_x = g_w / (1.0 - x) ** 2  # labels are d(mu)/dx
    mask = np.ones((1, M), dtype=bool)
    ok = np.ones((1, n), dtype=bool)
    nv = n // 3
    return ValueDataset(
        genome_ids=["g0"],
        exchanges=[f"EX_{j}" for j in range(M)],
        mask=mask,
        x_train=x[:, nv:],
        mu_train=mu[:, nv:],
        g_train=g_x[:, nv:],
        gvalid_train=ok[:, nv:],
        x_val=x[:, :nv],
        mu_val=mu[:, :nv],
        g_val=g_x[:, :nv],
        gvalid_val=ok[:, :nv],
        mu_scale=np.ones(1, np.float32),
        x_scale=np.ones((1, M), np.float32),
        index_hash="test",
        rounds_present=[0],
    )


def test_label_tangent_init_recovers_the_target_before_training():
    """`--gm-init labels` on the case it is exact for: width 1, depth 1, group K.

    The head then *is* min_k(a_k.w + c_k), and the labels *are* that function's
    supporting hyperplanes — so an untrained head must already be right. If this
    drifts, the init is silently a warm start rather than a reproduction.
    """
    from cfs.surrogate import groupmax
    from cfs.surrogate.picnn_u import to_diag

    ds = _min_affine_dataset()
    M = ds.x_train.shape[-1]
    heads = groupmax.stack_heads(
        jax.random.PRNGKey(0), 1, M, ds.mask, width=1, depth=1, group=64, temp=1e-4
    )
    seeded = groupmax.init_from_tangents(heads, ds)
    wv = to_diag(jnp.asarray(ds.x_val))
    before = np.asarray(groupmax.batched_value_diag(heads, wv))[0]
    after = np.asarray(groupmax.batched_value_diag(seeded, wv))[0]
    want = ds.mu_val[0]
    assert np.abs(after - want).max() < 1e-2, np.abs(after - want).max()
    # And it is the init doing it, not the target being easy to hit by accident.
    assert np.abs(before - want).max() > 10 * np.abs(after - want).max()

    # The gradient is the active plane's dual, which is what the gate scores.
    _, g = groupmax.batched_value_and_grad(seeded, jnp.asarray(ds.x_val))
    du = (1.0 - ds.x_val) ** 2  # d(mu)/dx -> d(mu)/dw
    cos = np.sum(np.asarray(g) * ds.g_val, -1) / (
        np.linalg.norm(np.asarray(g), axis=-1) * np.linalg.norm(ds.g_val, axis=-1) + 1e-30
    )
    assert np.nanmean(cos) > 0.99, np.nanmean(cos)
    assert du.shape == ds.x_val.shape


def test_active_set_ranking_puts_the_commonest_basis_first():
    from cfs.surrogate.groupmax import rank_by_active_set

    g = np.array([[1.0, 0, 0], [0, 1.0, 0], [1.0, 0, 0], [1.0, 0, 0], [0, 0, 1.0]])
    order = rank_by_active_set(g, np.ones(5, bool))
    # Pattern (0,) has 3 rows, (1,) and (2,) one each: a representative of the
    # commonest basis must come first, and every row must appear exactly once.
    assert order[0] in (0, 2, 3)
    assert sorted(order.tolist()) == [0, 1, 2, 3, 4]


def test_deepset_u_moves_phi_to_u_and_keeps_everything_else():
    """`deepset-u` is `deepset` with one coordinate changed — assert exactly that.

    The trunk must be the `u`-space one and its concavity must hold in `w`; the
    `--phi-hidden`/`--k-code` knobs and the shared/private split must survive the
    subclassing, since those are the M3b arm-3 axes.
    """
    from cfs.surrogate import deepset, deepset_u

    mask = np.ones((2, M), dtype=bool)
    mask[0, 0] = False
    heads = deepset_u.stack_heads(
        jax.random.PRNGKey(5), 2, M, mask, width=32, depth=2, phi_hidden=8, k_code=12
    )
    assert isinstance(heads.trunk, deepset_u.TrunkU)
    assert heads.shared and heads.trunk.b[0].shape == (8,) and heads.trunk.ob.shape == (12,)
    priv = deepset_u.stack_heads_private(jax.random.PRNGKey(5), 2, M, mask, width=32, depth=2)
    assert not priv.shared

    # Concave in w, sampled in w: mapping x -> w -> x cancels in float32 near x = 1.
    rng = np.random.default_rng(5)
    wa, wb = (rng.uniform(0, 20, (2, 128, M)).astype(np.float32) for _ in range(2))
    lam = 0.42
    mid = deepset_u.batched_value_diag(heads, lam * wa + (1 - lam) * wb)
    chord = lam * deepset_u.batched_value_diag(heads, wa) + (
        1 - lam
    ) * deepset_u.batched_value_diag(heads, wb)
    assert np.all(np.asarray(mid) >= np.asarray(chord) - 1e-4)

    # The two entry points agree: `x` is what the loss sees, `w` what concavity does.
    x = (wa / (1.0 + wa)).astype(np.float32)
    assert np.allclose(
        np.asarray(deepset_u.batched_value(heads, x)),
        np.asarray(deepset_u.batched_value_diag(heads, deepset_u.to_diag(x))),
        atol=1e-4,
    )

    # Still monotone, and the plain `deepset` trunk is untouched by the subclassing.
    _, grad = deepset_u.batched_value_and_grad(heads, x)
    assert np.all(np.asarray(grad * heads.mask[:, None, :]) >= -1e-6)
    plain = deepset.stack_heads(jax.random.PRNGKey(5), 2, M, mask, width=32, depth=2)
    assert type(plain.trunk) is deepset.Trunk


def test_deepset_phi_knobs_override_the_width_derivation():
    """`--phi-hidden` / `--k-code` are the M3b sweep's arm-3 axes.

    Default must stay exactly the old derivation (hidden ``width // 8``, code 16), or
    the trial's deepset numbers stop being reproducible.
    """
    from cfs.surrogate.deepset import batched_value as ds_value
    from cfs.surrogate.deepset import stack_heads as ds_stack

    mask = np.ones((2, M), dtype=bool)
    default = ds_stack(jax.random.PRNGKey(0), 2, M, mask, width=64, depth=2)
    assert default.trunk.b[0].shape == (64 // 8,)
    assert default.trunk.ob.shape == (16,)

    wide = ds_stack(jax.random.PRNGKey(0), 2, M, mask, width=64, depth=2, phi_hidden=32, k_code=48)
    assert wide.trunk.b[0].shape == (32,)
    assert wide.trunk.ob.shape == (48,)
    # `rho` reads the pooled code, so k_code must have moved its input width too.
    assert ds_value(wide, np.zeros((2, 3, M), np.float32)).shape == (2, 3)


def test_organism_arrays_signs_and_scatter(tmp_path):
    """The loader's saturation transform, dual sign, and zero-growth masking."""
    import json

    ex = ["EX_glc__D_e", "EX_o2_e"]
    (tmp_path / "g0.exchanges.json").write_text(json.dumps({"exchanges": ex}))
    shard = tmp_path / "g0" / "eps_0.001"
    shard.mkdir(parents=True)
    pd.DataFrame(
        [
            # a growing medium and a dead one; alpha != 1 rows must be dropped
            {
                "genome_id": "g0",
                "index_hash": "h",
                "medium_id": 0,
                "alpha": 1.0,
                "eps": 1e-3,
                "mu_max": 2.0,
                "status": "optimal",
                "medium": [0.02, 0.0],
                "z": [],
                "shadow": [-0.5, 0.0],
            },
            {
                "genome_id": "g0",
                "index_hash": "h",
                "medium_id": 0,
                "alpha": 0.5,
                "eps": 1e-3,
                "mu_max": 2.0,
                "status": "optimal",
                "medium": [0.02, 0.0],
                "z": [],
                "shadow": [-0.5, 0.0],
            },
            {
                "genome_id": "g0",
                "index_hash": "h",
                "medium_id": 1,
                "alpha": 1.0,
                "eps": 1e-3,
                "mu_max": 0.0,
                "status": "optimal",
                "medium": [0.0, 0.0],
                "z": [],
                "shadow": [-10000.0, 0.0],
            },
        ]
    ).to_parquet(shard / "part.parquet", index=False)

    km_cfg = {"classes": {"sugars": 0.01}, "default": 0.01, "keywords": {"sugars": ["glc"]}}
    col = {"EX_o2_e": 0, "EX_glc__D_e": 1, "EX_other_e": 2}
    x, mu, g, gvalid, mask, ihash, mid = _organism_arrays(tmp_path, "g0", 1e-3, col, km_cfg, 3)
    assert mid.tolist() == [0, 1]

    assert x.shape == (2, 3) and ihash == "h"
    assert mask.tolist() == [True, True, False]
    # saturation c/(Km+c) at c=0.02, Km=0.01 -> 2/3, scattered to the index column
    assert x[0, 1] == pytest.approx(2 / 3)
    # dual sign: stored shadow is the negated derivative, so dmu/dx > 0 here
    assert g[0, 1] == pytest.approx(0.5 * 1000.0)
    # the zero-growth row keeps its value label but its garbage duals are dropped
    assert gvalid.tolist() == [True, False] and np.all(g[1] == 0.0) and mu[1] == 0.0


def test_chunked_evaluation_matches_the_whole_set():
    """`_over_media` is an OOM guard, so it must change nothing but the peak.

    Evaluation used to call the head on the entire held-out set at once, which at
    D10 scale is 8x the training batch on the axis the deepset prices per
    metabolite. Slicing it is only safe if the diagnostics come out the same.
    """
    from cfs.surrogate.picnn import batched_value_and_grad, stack_heads
    from cfs.surrogate.train import _over_media

    ds = _synthetic()
    g, m = ds.mask.shape
    heads = stack_heads(jax.random.PRNGKey(0), g, m, ds.mask, width=16, depth=2)
    x = jax.numpy.asarray(ds.x_val)

    whole = batched_value_and_grad(heads, x)
    # A size that does not divide the media count, so the last slice is short.
    chunked = _over_media(lambda xx: batched_value_and_grad(heads, xx), x, size=7)

    for a, b in zip(whole, chunked, strict=True):
        assert a.shape == b.shape
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-5, atol=1e-6)
