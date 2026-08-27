"""The §8.4 Jacobian spectrum reader: does it separate rank from scaling?"""

import numpy as np

from cfs.validate.master_jacobian import CUTS, spectrum_report


def _psd(vecs: np.ndarray) -> np.ndarray:
    return vecs.T @ vecs


def test_rank_and_scaling_are_reported_separately():
    rng = np.random.default_rng(0)
    n, g = 40, 3
    # Each organism contributes a rank-2 Hessian; disjoint supports, so the sum is
    # rank 6 out of 40 -- the shape the real heads turn out to have.
    h = np.zeros((g, 1, n, n))
    for i in range(g):
        v = np.zeros((2, n))
        v[:, 4 * i : 4 * i + 4] = rng.normal(size=(2, 4))
        h[i, 0] = _psd(v)
    r = spectrum_report(h, np.full(g, 1 / g))
    assert r["n_shared"] == n
    assert r["dims_above_cut"]["raw"][CUTS.index(1e-9)] == 6

    # Now hide that rank under a per-metabolite scale spread, as `1/s^2` does with
    # `s` spanning 5107x. Jacobi must see through it; the raw spectrum must not.
    d = np.exp(rng.normal(scale=6.0, size=n))
    scaled = h * d[None, None, :, None] * d[None, None, None, :]
    rs = spectrum_report(scaled, np.full(g, 1 / g))
    assert rs["dims_above_cut"]["jacobi"] == r["dims_above_cut"]["jacobi"]
    assert (
        rs["dims_above_cut"]["raw"][CUTS.index(1e-3)] < r["dims_above_cut"]["raw"][CUTS.index(1e-3)]
    )


def test_non_finite_curvature_is_zeroed_not_propagated():
    h = np.zeros((2, 1, 5, 5))
    h[0, 0] = np.eye(5)
    h[1, 0, 2, 2] = np.inf
    r = spectrum_report(h, np.full(2, 0.5))
    assert np.isfinite(r["top_eigenvalue"]).all()
