"""M12 / §13.4 — the chemostat fixed point, against a Monod chemostat's closed form.

Two organisms on one substrate: competitive exclusion says only the one with the
lower break-even concentration survives, and both the survivor's `c*` and its
abundance have an exact answer. That is enough to catch a sign error in the
residual, a broken active set, and a wrong implicit derivative.
"""

import numpy as np

from cfs.science.steady import sensitivity, solve_steady, stability

MUMAX = np.array([1.0, 0.4])
K, YIELD, D, FEED = 1.0, 1.0, 0.5, 10.0


def _rhs(c, X):
    """Monod growth on metabolite 0; metabolite 1 is inert (nobody exchanges it)."""
    mu = MUMAX * c[0] / (K + c[0])
    dc = np.zeros_like(c)
    dc[0] = -float((X * mu).sum()) / YIELD
    return dc, mu


def test_fixed_point_matches_the_closed_form_and_excludes_the_slow_member():
    feed = np.array([FEED, 3.0])
    free = np.array([True, False])
    scale = np.array([K, K])
    sol = solve_steady(_rhs, feed, D, free, scale, feed.copy(), np.array([1.0, 1.0]))

    c_star = K * D / (MUMAX[0] - D)  # mu_0(c*) = D
    assert sol["converged"]
    assert bool(sol["alive"][0]) and not bool(sol["alive"][1])  # competitive exclusion
    assert np.isclose(sol["c"][0], c_star, rtol=1e-6)
    assert np.isclose(sol["c"][1], feed[1])  # inert: c = c_feed solves its row exactly
    assert np.isclose(sol["X"][0], YIELD * (FEED - c_star), rtol=1e-6)
    assert sol["X"][1] == 0.0

    # Stability of a Monod chemostat's coexistence-free fixed point.
    J = sol["J"].copy()
    J[1:, :] *= sol["X"][sol["alive"]][:, None]
    assert stability(J)["stable"]


def test_the_implicit_derivative_matches_a_resolve():
    """V4, in miniature: dX*/d(feed) = Y exactly, and dc*/d(feed) = 0."""
    feed = np.array([FEED, 3.0])
    free = np.array([True, False])
    scale = np.array([K, K])
    sol = solve_steady(_rhs, feed, D, free, scale, feed.copy(), np.array([1.0, 1.0]))
    S = sensitivity(sol["J"], np.flatnonzero(free), scale, D)
    assert np.allclose(S[:, 0], [0.0, YIELD], atol=1e-6)


def test_the_dual_chain_rule_and_its_two_corrections():
    """`_lp_mu_rows`: the sign convention, the chain rule, and both label clamps.

    A sign error here is silent -- it makes the Jacobian's growth rows point the
    wrong way and shows up only as slow or absent convergence -- and the two
    corrections are the ones `cfs.surrogate.data._organism_arrays` documents:
    the dual is `d(mu)/d(uptake bound)` only where that bound *binds*, and half
    the "non-zero" duals are O(1e-14) solver dust.
    """
    from cfs.science.steady import _lp_mu_rows
    from cfs.surrogate.behaviour import VMAX

    ex = ["EX_a_e", "EX_b_e", "EX_c_e", "EX_d_e"]
    km = np.array([1e-3, 1e-2, 1e-3, 1e-3])
    c = np.array([3.0e-8, 5.0e-3, 1.0e-3, 1.0e-3])
    duals = {
        "c": c.copy(),
        0: {
            "EX_a_e": -5.123134538636119,  # binds: a real sensitivity
            "EX_b_e": +2.0,  # a waste product's network value, NOT a derivative
            "EX_c_e": -1e-14,  # solver dust
            # EX_d_e absent from the solve entirely
        },
    }
    g = _lp_mu_rows(duals, None, ex, 1e-3, km, c, 1)

    # d(mu)/dc = pi * (-Vmax) * Km/(Km+c)^2, and only on the binding entry.
    want = 5.123134538636119 * VMAX * km[0] / (km[0] + c[0]) ** 2
    assert np.isclose(g[0, 0], want, rtol=1e-12)
    assert g[0, 0] > 0.0  # more nutrient cannot lower growth
    assert (g[0, 1:] == 0.0).all()  # positive dual, dust and absent all clamp to 0


def test_a_stale_dual_cache_is_not_silently_reused():
    """The duals ride on the residual's solve, so evaluation order is a hazard.

    Asking for rows at a different `c` than the cache holds must not return the
    cached ones: a stale Jacobian row is invisible and wrong.
    """
    from cfs.science.steady import _mixed_mu_rows

    ex = ["EX_a_e"]
    km = np.array([1e-3])
    st = {"c": np.array([1.0]), "lp": np.array([True]), "duals": {0: {"EX_a_e": -1.0}}}
    assert np.isfinite(_mixed_mu_rows(st, ex, km, np.array([1.0]), 1)).all()
    # NaN means "keep the finite difference", which is the safe fallback.
    assert np.isnan(_mixed_mu_rows(st, ex, km, np.array([2.0]), 1)).all()


def test_pseudo_transient_continuation_finds_the_same_root():
    """PTC is globalisation, not a different problem: same root, still not washout.

    The damping is what a line search cannot supply -- it changes the *direction*,
    not just its length -- so the invariant worth testing is that it does not buy
    that by landing somewhere else. The trivial `X = 0, c = c_feed` state is a
    genuine root and is what every scipy method converges to here (§13.4).
    """
    feed = np.array([FEED, 3.0])
    free = np.array([True, False])
    scale = np.array([K, K])
    sol = solve_steady(_rhs, feed, D, free, scale, feed.copy(), np.array([1.0, 1.0]), ptc=1.0)

    c_star = K * D / (MUMAX[0] - D)
    assert sol["converged"]
    assert np.isclose(sol["c"][0], c_star, rtol=1e-6)
    assert np.isclose(sol["X"][0], YIELD * (FEED - c_star), rtol=1e-6)
