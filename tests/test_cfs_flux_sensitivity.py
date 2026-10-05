"""`solve.flux_sensitivity` — the elastic-net QP's own dz/dc. No solver needed."""

import numpy as np

from cfs.groundtruth.solve import flux_sensitivity


def _chain():
    """R1: -> A, R2: A -> B, R3: B -> , R4: A -> (dead, carries no flux)."""
    s = np.array([[1.0, -1.0, 0.0, -1.0], [0.0, 1.0, -1.0, 0.0]])
    v = np.array([1.0, 1.0, 1.0, 0.0])
    lb = np.array([1.0, -10.0, -10.0, -10.0])
    ub = np.array([10.0, 10.0, 10.0, 10.0])
    return s, v, lb, ub


def test_uptake_bound_propagates_through_the_free_set():
    s, v, lb, ub = _chain()
    dbound = np.zeros((4, 1))
    dbound[0, 0] = 1.0  # relax R1's binding lower bound by 1
    dv = flux_sensitivity(s, v, lb, ub, dbound)
    # R1 tracks its bound; mass balance carries the whole increment down the chain.
    assert np.allclose(dv[:, 0], [1.0, 1.0, 1.0, 0.0])


def test_a_reaction_at_zero_stays_at_zero():
    s, v, lb, ub = _chain()
    dbound = np.zeros((4, 1))
    dbound[0, 0] = 1.0
    dv = flux_sensitivity(s, v, lb, ub, dbound)
    # R4 is off by the L1 term, not by a bound: its subgradient is interior, so it
    # does not switch on for an infinitesimal perturbation.
    assert dv[3, 0] == 0.0


def test_eps_cancels_and_the_free_block_is_minimum_norm():
    # Two parallel routes A -> B: the min-norm restoration splits the load evenly.
    s = np.array([[1.0, -1.0, -1.0, 0.0], [0.0, 1.0, 1.0, -1.0]])
    v = np.array([2.0, 1.0, 1.0, 2.0])
    lb = np.array([2.0, -10.0, -10.0, -10.0])
    ub = np.full(4, 10.0)
    dbound = np.zeros((4, 1))
    dbound[0, 0] = 1.0
    dv = flux_sensitivity(s, v, lb, ub, dbound)
    assert np.allclose(dv[:, 0], [1.0, 0.5, 0.5, 1.0])
