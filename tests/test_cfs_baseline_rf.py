"""The RF baseline's finite-difference gradients, on a target with a known slope.

A forest's analytic gradient is zero everywhere, so the baseline probes it. The
probe has to invert the stored ``x = u / (u + s)`` map, step in ``u``, and map
back, and getting that chain wrong produces gradients that are plausibly scaled
and systematically wrong — which is exactly the sort of thing a cosine against
real labels would half-hide. So test it where the answer is known.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("sklearn")

from cfs.surrogate.baseline import _u_space_gradients  # noqa: E402


def test_finite_differences_recover_a_known_slope():
    """``mu = sum_m k_m u_m`` — linear in u, so d(mu)/du_m is exactly k_m.

    ``k_m = 1 / s_m`` so every metabolite contributes equally to ``mu``. With
    unequal contributions the forest simply ignores the small dimensions and the
    recovered slopes are zero — a true fact about forests, but it would test
    sklearn rather than this module's chain rule.
    """
    from sklearn.ensemble import RandomForestRegressor

    rng = np.random.default_rng(0)
    # Kink scales spanning four decades, as they do on the real roster.
    s = np.array([1e-4, 1e-2, 1e-1, 1.0])
    k = 1.0 / s
    u = rng.uniform(0, 5, (4000, 4)) * s  # each metabolite around its own scale
    mu = u @ k

    x = u / (u + s)
    rf = RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=-1).fit(x, mu)
    g = _u_space_gradients(rf, x, s, delta=0.25)

    # The forest smooths, so the slopes come back attenuated — but by the *same*
    # factor across four decades of scale, which is the statement that the u <-> x
    # round trip is scaled per metabolite correctly. A chain-rule error would show
    # up here as an attenuation that tracks `s`.
    ratio = np.median(g, axis=0) / k
    assert np.ptp(ratio) < 0.1 * ratio.mean(), ratio
    cos = np.mean((g @ k) / (np.linalg.norm(g, axis=1) * np.linalg.norm(k)))
    assert cos > 0.9, cos


def test_a_step_across_the_kink_reads_the_wrong_regime():
    """A ramp that plateaus: the probe must not step over the elbow.

    The linear test above cannot fail this way -- it has no kink -- and that gap
    is why ``delta`` was originally set to 0.25 on evidence that could not falsify
    it. On the real roster that step inverted whole cells (``EX_cytd_e`` 0.613 at
    0.05 to -0.729 at 0.5), which was misread as the forest being non-monotone.

    Here ``mu`` ramps over ``u < s`` and is flat above, so a probe centred just
    past the elbow reads ~0 with a small step and a large positive slope with a
    step that reaches back into the ramp. The slope is a fact about the step size,
    not about the model.
    """
    s = np.array([1.0])

    class Ramp:
        """mu = min(u, s): slope 1 below the elbow, 0 above."""

        def predict(self, x):
            u = x[:, 0] * s[0] / np.maximum(1.0 - x[:, 0], 1e-12)
            return np.minimum(u, s[0])

    # Sit at u = 2s, comfortably on the plateau, where the true slope is 0.
    u0 = 2.0 * s[0]
    x = np.array([[u0 / (u0 + s[0])]])
    small = _u_space_gradients(Ramp(), x, s, delta=0.05)[0, 0]
    large = _u_space_gradients(Ramp(), x, s, delta=2.0)[0, 0]
    assert small == pytest.approx(0.0, abs=1e-6), small
    # The large step reaches back below the elbow and averages in the ramp.
    assert large > 0.2, large


def test_step_is_clipped_at_zero_concentration():
    """u = 0 is the depletion corner; the step must not go negative."""
    class Identity:
        """d(mu)/du = 1 exactly, so the returned slope reads the step taken."""

        def predict(self, x):
            s = np.array([1e-3, 1.0])
            return (x * s / np.maximum(1.0 - x, 1e-12)).sum(axis=1)

    s = np.array([1e-3, 1.0])
    x = np.zeros((3, 2))  # every concentration at 0
    g = _u_space_gradients(Identity(), x, s, delta=0.1)
    assert np.allclose(g, 1.0), g
