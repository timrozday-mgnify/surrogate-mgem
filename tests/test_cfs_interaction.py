"""§13.5's objective: the arithmetic that decides what an interaction is."""

import numpy as np

from cfs.science.interaction import distinguishing, exchange, exchange_batch, link_rows


def test_a_monoculture_has_no_interaction():
    """The reason §13.5 cannot be posed at the §13.4 steady state.

    With one organism every metabolite has either `z >= 0` or `z <= 0`, so one of
    the two sums is zero and the min with it. §13.4 measured `k = 1` and one
    survivor on every roster cell, so maximising `E` at the equilibrium would be
    maximising zero.
    """
    z = np.array([[5.0, -3.0, 0.0, 2.0]])
    assert exchange(z, np.ones(1)).sum() == 0.0


def test_exchange_is_the_smaller_side():
    # metabolite 0: A secretes 5, B takes up 2  -> 2 handed over
    # metabolite 1: A takes up 3, B takes up 1  -> nobody secretes, 0
    # metabolite 2: both secrete                -> nobody eats, 0
    z = np.array([[5.0, -3.0, 4.0], [-2.0, -1.0, 1.0]])
    assert np.allclose(exchange(z, np.ones(2)), [2.0, 0.0, 0.0])


def test_exchange_scales_with_abundance():
    z = np.array([[6.0, 0.0], [-2.0, 0.0]])
    assert np.allclose(exchange(z, np.array([1.0, 1.0])), [2.0, 0.0])
    # Doubling the consumer doubles what it can absorb, up to the supply.
    assert np.allclose(exchange(z, np.array([1.0, 2.0])), [4.0, 0.0])
    # ...and past the supply the donor is what binds.
    assert np.allclose(exchange(z, np.array([1.0, 9.0])), [6.0, 0.0])


def test_batch_matches_the_single_path():
    rng = np.random.default_rng(0)
    z = rng.normal(size=(3, 5, 7))
    X = rng.uniform(0.5, 2.0, size=3)
    got = exchange_batch(z, X)
    want = np.array([exchange(z[:, b, :], X) for b in range(5)])
    assert np.allclose(got, want)


def test_link_rows_names_both_sides():
    z = np.array([[4.0, 0.0], [-1.0, 0.0], [-2.0, 0.0]])
    rows = link_rows(z, np.ones(3), ["EX_ac_e", "EX_o2_e"], ["A", "B", "C"])
    assert len(rows) == 1  # the all-zero metabolite is not a link
    assert rows[0]["metabolite"] == "EX_ac_e"
    assert rows[0]["rate"] == 3.0  # min(4 secreted, 3 taken up)
    assert set(rows[0]["donors"]) == {"A"}
    assert set(rows[0]["recipients"]) == {"B", "C"}


def test_distinguishing_reports_the_moved_components():
    ref = np.array([1.0, 1.0, 1.0])
    c = np.array([10.0, 1.0, 0.1])
    got = distinguishing(c, ref, ["a", "b", "c"], top=3)
    assert [g["metabolite"] for g in got] == ["a", "c"]  # "b" did not move
    assert np.isclose(got[0]["log10_fold"], 1.0)
    assert np.isclose(got[1]["log10_fold"], -1.0)


def test_buffered_species_are_not_interactions():
    """A chemostat's buffer absorbs the protons, so they are not a handover.

    Without this `E` is literally proton exchange: `EX_h_e` alone was 97.6% of
    one community's true interaction rate.
    """
    from cfs.science.interaction import keep_mask

    ex = ["EX_h_e", "EX_ac_e", "EX_h2o_e", "EX_co2_e"]
    keep = keep_mask(ex)
    assert list(keep) == [False, True, False, True]  # CO2 is real cross-feeding
    z = np.array([[900.0, 4.0, 500.0, 2.0], [-900.0, -4.0, -500.0, -2.0]])
    e = exchange(z, np.ones(2), keep)
    assert e.sum() == 6.0  # not 1406


class _FakeSur:
    km = np.array([1.0, 2.0, 4.0])


def test_buffered_species_are_pinned_at_saturation():
    """Abundance the vessel supplies: u = c/(Km+c) = 0.999 at the default."""
    from cfs.science.interaction import buffer_medium, keep_mask

    sur = _FakeSur()
    keep = keep_mask(["EX_ac_e", "EX_h_e", "EX_h2o_e"])
    c = buffer_medium(sur, np.array([0.5, 0.5, 0.5]), keep)
    assert c[0] == 0.5  # a designed metabolite is untouched
    assert np.allclose(c[1:], [2e3, 4e3])  # 1e3 * Km
    u = c[1:] / (sur.km[1:] + c[1:])
    assert np.all(u > 0.999)
