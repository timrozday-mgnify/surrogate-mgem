"""M11: the penalty solve returns a feasible medium and drops what nothing eats."""

import numpy as np

from cfs.science import minimal


def _stub(monkeypatch, a):
    """mu_i(c) = sum_m a[i,m] * (1 - exp(-c_m)): concave, increasing, analytic."""

    def mu_and_grads(sur, c, members, cuts=None):
        mu = a @ (1.0 - np.exp(-c))
        g = a * np.exp(-c)[None, :]
        for i, per_member in enumerate(cuts or ()):
            for mu_j, g_j, c_j in per_member:
                vj = mu_j + float(g_j @ (c - c_j))
                if vj <= mu[i]:
                    mu[i], g[i] = vj, g_j
        return mu, g

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


def test_a_cut_tightens_the_feasible_set_and_raises_the_cost(monkeypatch):
    """A true tangent that reads below the head must make the design pay more.

    `mu_true` is concave, so a tangent is an upper bound on it everywhere: adding
    one can only remove points the truth does not admit. The design that comes back
    therefore costs at least as much, and the member the cut belonged to must still
    clear its floor under the *cut* model.
    """
    a = np.array([[1.0, 2.0, 0.0, 0.0], [0.0, 1.0, 3.0, 0.0]])
    _stub(monkeypatch, a)
    c_hi = np.full(4, 2.0)
    target = 0.5 * (a @ (1.0 - np.exp(-c_hi)))
    cost = np.ones(4)

    base, _ = minimal.minimise(None, c_hi, [0, 1], target, cost=cost)
    # A pessimistic-but-valid tangent for member 0 at the rich medium: half the
    # head's value there, same slope.
    mu0 = float(0.5 * a[0] @ (1.0 - np.exp(-c_hi)))
    cut = [[(mu0, a[0] * np.exp(-c_hi), c_hi.copy())], []]
    cut_c, _ = minimal.minimise(None, c_hi, [0, 1], target, cost=cost, cuts=cut)

    assert float(cost @ cut_c) >= float(cost @ base) - 1e-9
    mu, _ = minimal.mu_and_grads(None, cut_c, [0, 1], cut)
    assert (mu >= target - 1e-6).all()
