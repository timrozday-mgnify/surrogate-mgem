"""The composition map itself, with a hand-written right-hand side.

`integrate` is the one piece of §8.1 that is neither a network nor an LP, and it
is shared by the surrogate and the ground-truth path — an error there would move
both trajectories the same way and hide itself in the comparison. So it is checked
against a case with a closed form: constant growth rate, constant consumption.

No JAX, no solver, no labels.
"""

from __future__ import annotations

import numpy as np

from cfs.compose.dfba import _cos, _rel, integrate


def test_exponential_growth_is_exact():
    # mu constant => X(t) = X0 exp(mu t) exactly, whatever the step size, because
    # the biomass update is the exponential map rather than an Euler step.
    mu = np.array([2.0, 0.5])
    traj = integrate(lambda c, X: (np.zeros(3), mu), np.ones(3), np.array([1.0, 4.0]), 0.1, 10)
    assert np.allclose(traj.x[-1], np.array([1.0, 4.0]) * np.exp(mu * 1.0))
    assert np.allclose(traj.c, 1.0)  # nothing consumed


def test_pool_is_clipped_at_zero():
    # A constant drain of 1/h from a pool of 0.5 must stop at 0, not go negative:
    # the LP's regime changes there (no uptake), so a negative concentration would
    # be fed straight into the MM bound.
    traj = integrate(
        lambda c, X: (np.array([-1.0]), np.zeros(1)), np.array([0.5]), np.ones(1), 0.1, 20
    )
    assert traj.c.min() == 0.0
    assert traj.c[-1, 0] == 0.0


def test_consumption_scales_with_biomass():
    # dc/dt = sum_i X_i z_i is the whole composition. Two organisms with the same
    # z drain twice as fast as one, and doubling biomass doubles the rate.
    z = np.array([-1.0])

    def rhs(c, X):
        return (X[:, None] * np.tile(z, (len(X), 1))).sum(0), np.zeros(len(X))

    one = integrate(rhs, np.array([10.0]), np.array([1.0]), 0.01, 100)
    two = integrate(rhs, np.array([10.0]), np.array([1.0, 1.0]), 0.01, 100)
    assert np.isclose(10.0 - one.c[-1, 0], 1.0)
    assert np.isclose(10.0 - two.c[-1, 0], 2.0)


def test_metrics():
    a = np.array([1.0, 0.0])
    assert _cos(a, a) == 1.0
    assert _rel(a, a) == 0.0
    assert np.isclose(_rel(np.array([2.0, 0.0]), a), 1.0)


# --------------------------------------------------------------------------- #
# Head B's loader: the alpha axis has to line up with the medium axis
# --------------------------------------------------------------------------- #


def test_behaviour_dataset_alpha_and_medium_alignment(tmp_path):
    """``z`` must land in the right (alpha, medium) cell, and share Head A's split.

    The shards store one row per (medium, alpha) in whatever order the generator
    wrote them, and the value arrays are sorted by ``medium_id``. Getting these two
    orderings out of step is silent: the shapes are right, the loss falls, and the
    head learns the wrong alpha response.
    """
    import json

    import pandas as pd
    import pytest

    pytest.importorskip("pyarrow")
    pytest.importorskip("yaml")

    from cfs.groundtruth.index import derive_index, write_index
    from cfs.surrogate.data import load_behaviour_dataset, load_value_dataset

    ex = ["EX_glc__D_e", "EX_o2_e"]
    alphas, mids = [0.0, 1.0], np.arange(10)
    index_path = tmp_path / "metabolite_index.json"
    digest = write_index(derive_index(["g0"], [set(ex)]), index_path)

    root = tmp_path / "labels"
    shard = root / "g0" / "eps_0.001"
    shard.mkdir(parents=True)
    (root / "g0.exchanges.json").write_text(json.dumps({"exchanges": ex}))
    # z is a function of both axes and of nothing else, so a misalignment shows up
    # as an exact value mismatch rather than as a worse fit.
    pd.DataFrame(
        [
            {
                "genome_id": "g0",
                "index_hash": digest,
                "medium_id": int(m),
                "alpha": a,
                "eps": 1e-3,
                "mu_max": 1.0,
                "status": "optimal",
                "medium": [0.02, 0.01],
                "z": [float(m), -100.0 * a],
                "shadow": [-0.5, 0.0],
            }
            # Deliberately not in (alpha, medium) order.
            for m in mids[::-1]
            for a in alphas
        ]
    ).to_parquet(shard / "part.parquet", index=False)

    ds = load_behaviour_dataset(root, index_path, eps=1e-3, seed=0)
    va = load_value_dataset(root, index_path, eps=1e-3, seed=0)
    assert ds.alphas.tolist() == alphas
    # z[:, 0] is the medium id and z[:, 1] is -100 * alpha, everywhere.
    assert np.allclose(ds.z_train[0][:, 1], -100.0 * ds.a_train[0])
    assert np.allclose(ds.z_val[0][:, 1], -100.0 * ds.a_val[0])
    # Same held-out media as Head A, each appearing once per alpha.
    assert ds.x_val.shape[1] == va.x_val.shape[1] * len(alphas)
    val_mids, train_mids = set(ds.z_val[0][:, 0]), set(ds.z_train[0][:, 0])
    assert not (val_mids & train_mids)
    assert val_mids | train_mids == {float(m) for m in mids}
    # Each held-out medium appears once per alpha and no more.
    assert len(ds.z_val[0]) == len(val_mids) * len(alphas)
    assert np.allclose(ds.x_scale, va.x_scale)


def test_behaviour_target_is_specific_flux(tmp_path):
    """``mu_train`` is the floored ``mu_max``, and ``z_scale`` is taken on ``z/mu``.

    Head B emits flux per unit growth and :mod:`cfs.compose.dfba` multiplies Head
    A's ``mu`` back in. If the loader's divisor drifts from that floor the two
    halves disagree by a factor that looks like a mediocre fit, not like a bug.
    """
    import json

    import pandas as pd
    import pytest

    pytest.importorskip("pyarrow")
    pytest.importorskip("yaml")

    from cfs.groundtruth.index import derive_index, write_index
    from cfs.surrogate.data import _MU_FLOOR_FRAC, load_behaviour_dataset

    ex = ["EX_glc__D_e", "EX_o2_e"]
    index_path = tmp_path / "metabolite_index.json"
    digest = write_index(derive_index(["g0"], [set(ex)]), index_path)
    root = tmp_path / "labels"
    shard = root / "g0" / "eps_0.001"
    shard.mkdir(parents=True)
    (root / "g0.exchanges.json").write_text(json.dumps({"exchanges": ex}))
    # One medium is starving (mu_max = 0), which is what the floor exists for.
    mus = {int(m): (0.0 if m == 3 else 1.0 + m) for m in range(10)}
    pd.DataFrame(
        [
            {
                "genome_id": "g0",
                "index_hash": digest,
                "medium_id": int(m),
                "alpha": 1.0,
                "eps": 1e-3,
                "mu_max": mus[int(m)],
                "status": "optimal",
                "medium": [0.02, 0.01],
                "z": [2.0 * mus[int(m)], -3.0],
                "shadow": [-0.5, 0.0],
            }
            for m in range(10)
        ]
    ).to_parquet(shard / "part.parquet", index=False)

    ds = load_behaviour_dataset(root, index_path, eps=1e-3, seed=0)
    floor = _MU_FLOOR_FRAC * np.mean(list(mus.values()))
    assert np.isclose(ds.mu_floor[0], floor)
    assert ds.mu_train.min() == pytest.approx(floor)
    # Glucose flux is exactly 2 * mu_max, so the specific flux is constant at 2
    # everywhere the floor does not bite -- i.e. a zero scale, floored to 1.
    spec = ds.z_train[0] / ds.mu_train[0][:, None]
    assert np.allclose(spec[ds.mu_train[0] > 1.001 * floor, 0], 2.0)
    assert np.allclose(ds.z_scale[0], spec.std(axis=0))


# --------------------------------------------------------------------------- #
# §13.1 — continuous culture
# --------------------------------------------------------------------------- #


def test_chemostat_washes_out_a_slow_member():
    # D > mu => X -> 0, D < mu => the member persists. This is the whole of
    # `with_chemostat`'s biomass half, and it is the coexistence question §13.4
    # is built on, so it is worth a closed form: X(t) = X0 exp((mu - D) t).
    from cfs.compose.dfba import with_chemostat

    mu = np.array([1.0, 0.2])
    base = lambda c, X: (np.zeros(1), mu)  # noqa: E731
    traj = integrate(with_chemostat(base, 0.5, np.zeros(1)), np.zeros(1), np.ones(2), 0.1, 100)
    assert np.allclose(traj.x[-1], np.exp((mu - 0.5) * 10.0))
    assert traj.x[-1, 0] > 100 and traj.x[-1, 1] < 0.05


def test_chemostat_pool_relaxes_to_the_feed():
    # No biomass: dc/dt = D (feed - c) => c -> feed with time constant 1/D.
    from cfs.compose.dfba import with_chemostat

    feed = np.array([3.0])
    base = lambda c, X: (np.zeros(1), np.zeros(1))  # noqa: E731
    traj = integrate(with_chemostat(base, 1.0, feed), np.zeros(1), np.ones(1), 0.01, 1000)
    assert np.isclose(traj.c[-1, 0], 3.0, atol=1e-3)
    assert np.isclose(traj.c[100, 0], 3.0 * (1 - np.exp(-1.0)), atol=1e-2)


def test_element_balance_projects_onto_the_secretion_bound():
    # §8.6g(2). You cannot secrete more carbon than you took up. The point of the
    # projection (over a uniform shrink, which also satisfies the constraint) is
    # that it cannot move `z` away from a point already inside the set.
    from cfs.compose.dfba import Surrogate

    sur = Surrogate.__new__(Surrogate)
    sur._E = np.array([[1.0, 6.0, 0.0], [0.0, 0.0, 0.0]])  # C, N atoms per mmol
    sur.mask = np.ones((2, 3), dtype=bool)
    sur.z_scale = np.ones((2, 3), dtype=np.float32)

    z = np.array([[10.0, -1.0, 5.0], [2.0, -3.0, 5.0]])  # row 0 secretes 10 C on 6
    out = Surrogate._element_balance(sur, z)
    assert (out @ sur._E.T <= 1e-9).all()
    assert np.allclose(out[1], z[1])  # a compliant row is not touched at all

    truth = np.array([3.0, -1.0, 4.0])  # some feasible z: 3 C out against 6 in
    assert (sur._E @ truth <= 0).all()
    assert np.linalg.norm(out[0] - truth) < np.linalg.norm(z[0] - truth)


def test_element_balance_matches_a_brute_force_projection():
    # Two elements binding at once, which is what the dual's active-set enumeration
    # is for -- and what a sign error in its feasibility test hides: the identity
    # is feasible for every subset when only one row of E is non-zero, so a
    # one-element case passes either way.
    from scipy.optimize import minimize

    from cfs.compose.dfba import Surrogate

    rng = np.random.default_rng(0)
    E = np.array([[1.0, 6.0, 2.0, 0.0], [1.0, 0.0, 3.0, 1.0]])  # C and N per mmol
    for _ in range(20):
        z = rng.normal(size=4) * np.array([5.0, 1.0, 3.0, 2.0])
        if (E @ z <= 0).all():
            continue
        sur = Surrogate.__new__(Surrogate)
        sur._E, sur.mask = E, np.ones((1, 4), dtype=bool)
        sur.z_scale = np.full((1, 4), 1.0, dtype=np.float32)
        out = Surrogate._element_balance(sur, z[None])[0]
        ref = minimize(
            lambda y, z=z: ((y - z) ** 2).sum(),
            z,
            constraints=[{"type": "ineq", "fun": lambda y, e=e: -e @ y} for e in E],
        ).x
        assert (E @ out <= 1e-8).all()
        assert np.allclose(out, ref, atol=1e-5)


def test_element_balance_is_the_identity_without_a_table():
    from cfs.compose.dfba import Surrogate

    sur = Surrogate.__new__(Surrogate)
    sur._E = None
    z = np.array([[1.0, -2.0]])
    assert np.allclose(Surrogate._element_balance(sur, z), z)


def test_traj_anchor_is_normalised_per_leaf():
    # §8.6g(3). The drift penalty must be per leaf: Head B's arrays differ in scale
    # by orders of magnitude, and `--w-prox` already showed that one global ratio
    # over a heavy-tailed quantity is set by its largest members and reads ~0 for
    # every other array — a penalty that silently does nothing.
    import equinox as eqx
    import jax.numpy as jnp
    import pytest

    from cfs.surrogate.traj import _anchor

    class P(eqx.Module):
        big: jnp.ndarray
        small: jnp.ndarray

    base = P(jnp.full((4,), 1000.0), jnp.full((4,), 0.001))
    # Same *relative* drift in each leaf: 10%.
    moved = P(base.big * 1.1, base.small * 1.1)
    assert float(_anchor(moved, base, 1.0)) == pytest.approx(2 * 0.01, rel=1e-4)  # float32
    # A drift in the small leaf alone still registers.
    small_only = P(base.big, base.small * 1.1)
    assert float(_anchor(small_only, base, 1.0)) == pytest.approx(0.01, rel=1e-4)  # float32
    assert float(_anchor(moved, base, 0.0)) == 0.0


def test_the_element_balance_cache_tolerates_a_head_a_only_surrogate():
    """§13.2 and §13.3 build a `Surrogate` with no behaviour checkpoint.

    Regression: caching the element-balance operators in `__init__` dereferenced
    `z_scale`, which is None for those callers, and took `cfs minimal-medium` and
    `cfs maximise-growth` from working to an AttributeError at construction. Every
    unit test supplied a behaviour dir, so nothing caught it.
    """
    from cfs.compose import dfba

    E = np.array([[1.0, 6.0, 0.0], [0.0, 0.0, 0.0]])
    mask = np.ones((2, 3), dtype=bool)
    assert dfba._eb_operators(E, mask, None) is None  # Head A only
    assert dfba._eb_operators(None, mask, np.ones((2, 3))) is None  # no element data
    built = dfba._eb_operators(E, mask, np.ones((2, 3)))
    assert len(built) == 2 and built[0][2].shape == (2, 2)
