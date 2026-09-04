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
