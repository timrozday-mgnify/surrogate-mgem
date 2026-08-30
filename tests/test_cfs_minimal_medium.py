"""M11: the penalty solve returns a feasible medium and drops what nothing eats."""

import numpy as np

from cfs.science import minimal


def _stub(monkeypatch, a):
    """mu_i(c) = sum_m a[i,m] * (1 - exp(-c_m)): concave, increasing, analytic."""

    def mu_and_grads(sur, c, members):
        mu = a @ (1.0 - np.exp(-c))
        return mu, a * np.exp(-c)[None, :]

    monkeypatch.setattr(minimal, "mu_and_grads", mu_and_grads)


def test_minimal_medium_is_feasible_and_sparse(monkeypatch):
    # member 0 eats metabolites 0,1; member 1 eats 1,2; nobody eats 3.
    a = np.array([[1.0, 2.0, 0.0, 0.0], [0.0, 1.0, 3.0, 0.0]])
    _stub(monkeypatch, a)
    c_hi = np.array([2.0, 2.0, 2.0, 2.0])
    target = 0.5 * (a @ (1.0 - np.exp(-c_hi)))
    cost = np.ones(4)

    c, info = minimal.minimise(None, c_hi, [0, 1], target, cost=cost)

    assert info["feasible_surrogate"], info
    mu = a @ (1.0 - np.exp(-c))
    assert np.all(mu >= target - 1e-9)
    assert c[3] < 1e-6 * c_hi[3]  # nothing eats it, so it is not in the medium
    assert cost @ c < 0.6 * (cost @ c_hi)
