"""§13.5's objective: the arithmetic that decides what an interaction is."""

import numpy as np
import pytest

from cfs.science import interaction

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


def test_candidate_links_are_a_secretion_meeting_an_uptake(tmp_path):
    """The enumeration §13.5 seeds from: donor secretes, recipient takes up.

    Built from the labels alone -- no LP -- so the check is that the direction,
    the dust threshold and the recorded secretion medium are right. `B` secretes
    metabolite 1 and `A` takes it up, so 1 is a candidate; metabolite 0 is
    secreted by nobody and metabolite 2 only ever moves at dust level. `B`'s
    second row is where it secretes hardest, so that row's medium is the one the
    start has to reproduce.
    """
    import json

    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    from cfs.science.interaction import candidate_links

    ex = ["EX_a_e", "EX_b_e", "EX_c_e"]
    z = {"A": [[-4.0, -2.0, 1e-9], [-4.0, -2.0, 1e-9]],
         "B": [[-1.0, 3.0, -1e-9], [-1.0, 8.0, -1e-9]]}
    med = {"A": [[1.0, 1.0, 1.0]] * 2, "B": [[9.0, 9.0, 9.0], [7.0, 5.0, 3.0]]}
    for g in z:
        (tmp_path / f"{g}.exchanges.json").write_text(json.dumps({"exchanges": ex}))
        d = tmp_path / g / "eps_0.001"
        d.mkdir(parents=True)
        pq.write_table(
            pa.table({
                "z": pa.array(z[g], type=pa.list_(pa.float64())),
                "medium": pa.array(med[g], type=pa.list_(pa.float64())),
            }),
            d / "part.parquet",
        )

    links, donor_media, donor_box = candidate_links(tmp_path, ["A", "B"], ex)
    assert set(links) == {"EX_b_e"}
    assert links["EX_b_e"]["donors"] == ["B"]
    assert links["EX_b_e"]["recipients"] == ["A"]
    assert links["EX_b_e"]["best_donor"] == "B"
    # The medium of B's *hardest* secretion row, not its first.
    assert np.allclose(donor_media["EX_b_e"], [7.0, 5.0, 3.0])
    # ...and the box spans *both* rows where B secreted it, which is the region
    # the starts are drawn from rather than the single best point.
    lo, hi = donor_box["EX_b_e"]
    assert np.allclose(lo, [7.0, 5.0, 3.0])
    assert np.allclose(hi, [9.0, 9.0, 9.0])


def test_ceq_map_completes_the_layer_and_skips_buffered():
    """§13.11 stage 2': a "default" must reach every exchange except the buffered."""
    from cfs.science.interaction import ceq_map, keep_mask

    ex = ["EX_ac_e", "EX_glc__D_e", "EX_h_e", "EX_h2o_e"]
    keep = keep_mask(ex)

    m = ceq_map({"default": 1.0, "EX_ac_e": 3.0}, ex, keep)
    assert m == {"EX_ac_e": 3.0, "EX_glc__D_e": 1.0}  # P30: no silent uninhibited gap
    # A buffered species pinned at 1e3*Km would get ub = 0 under any finite c^eq,
    # i.e. a community that cannot excrete a proton. Only the default skips them.
    assert ceq_map({"default": 1.0, "EX_h_e": 5.0}, ex, keep)["EX_h_e"] == 5.0
    assert ceq_map({"EX_ac_e": 3.0}, ex, keep) == {"EX_ac_e": 3.0}  # no default, no fill


def test_interference_media_isolates_the_partners():
    """Self-depletion is in both arms and cancels; only the partner term differs."""
    c = np.array([1.0, 1.0])
    z = np.array([[-1.0, 0.0], [0.0, -2.0]])  # member 0 eats m0, member 1 eats m1
    X = np.array([1.0, 1.0])
    dt, alone, joint = interaction.interference_media(c, z, X, frac=0.1)
    assert dt == pytest.approx(0.05)  # first depletion is m1, at t = 0.5
    assert alone[0] == pytest.approx([0.95, 1.0])
    assert alone[1] == pytest.approx([1.0, 0.9])
    assert joint == pytest.approx([0.95, 0.9])
    # partner effect on member 0 = joint - alone[0], i.e. m1 only
    assert (joint - alone[0]) == pytest.approx([0.0, -0.1])


def test_interference_media_never_exhausts_a_metabolite():
    """The step is `frac` of the *first* depletion, so nothing reaches zero."""
    _, alone, joint = interaction.interference_media(
        np.array([1.0, 1e-9]), np.array([[-1.0, -1e-6]]), np.array([1.0])
    )
    assert (alone > 0).all() and (joint > 0).all()


def test_interference_media_never_goes_negative():
    dt, alone, joint = interaction.interference_media(
        np.array([1.0]), np.array([[-100.0]]), np.array([1.0]), frac=10.0
    )
    assert (alone >= 0).all() and (joint >= 0).all()


def test_resupplementation_keeps_secretions_and_restores_uptake():
    """The control that separates "your waste inhibits me" from "you ate my food"."""
    c = np.array([1.0, 1.0])
    z = np.array([[-1.0, +1.0]])  # the donor eats m0 and secretes m1
    _, spent, _ = interaction.interference_media(c, z, np.array([1.0]))
    resup = np.maximum(spent[0], c)
    assert spent[0][0] < c[0] and spent[0][1] > c[1]  # depleted, and conditioned
    assert resup[0] == c[0]  # what the donor ate is restored
    assert resup[1] == spent[0][1]  # what it secreted is kept


def test_target_level_balances_the_two_halves():
    """`E` is a min, so the best level is where uptake and secretion are equal."""
    lvl, km = interaction.target_level, 0.01
    assert lvl(km, None)[0] == pytest.approx(1000 * km)  # uninhibited: unchanged
    assert lvl(km, 1e6)[0] == pytest.approx(1000 * km)  # a c^eq that never binds
    for ceq in (1.0, 0.1, km, km / 100):
        c, f = lvl(km, ceq)
        assert c / (km + c) == pytest.approx(1.0 - c / ceq)  # the two halves meet
        assert f == pytest.approx(c / (km + c))
    # Not a hard window: `c^eq = Km` still hands over, at 38% of both capacities.
    assert lvl(km, km)[1] == pytest.approx(0.382, abs=1e-3)
    assert lvl(km, km / 100)[1] < lvl(km, km)[1] < lvl(km, 10 * km)[1]


def test_buffered_species_do_not_set_the_step():
    """The vessel holds them, so they can neither deplete nor scale `dt`.

    Without the mask a proton or water term is ordinarily the fastest-draining
    entry in `dc`, so it sets the whole step and every reported rate is scaled
    by a species the buffer is holding constant.
    """
    c = np.array([1.0, 1.0])
    z = np.array([[-1.0, -100.0]])  # m1 drains 100x faster and is buffered
    X = np.array([1.0])
    keep = np.array([1.0, 0.0])
    dt, alone, joint = interaction.interference_media(c, z, X, frac=0.1, keep=keep)
    assert dt == pytest.approx(0.1)  # set by m0 (t=1), not by m1 (t=0.01)
    assert alone[0][1] == 1.0 and joint[1] == 1.0  # the buffered species is held


class _ToySur:
    """Two members over two metabolites, both eating m0; only member 1 eats m1.

    `z` is constant so the step is hand-computable, and `mu` is each member's
    own substrate concentration so the growth response is too.
    """

    km = np.array([1.0, 1.0])
    genome_ids = ["a", "b"]
    members = np.array([0, 1])
    Z = np.array([[-1.0, 0.0], [-1.0, -2.0]])

    def mu_batch(self, C):
        return np.stack([np.asarray(C)[:, 0], np.asarray(C)[:, 1]])

    def mu_and_z_batch(self, C, alpha):
        return self.mu_batch(C), np.repeat(self.Z[:, None, :], len(C), axis=1)

    def mu_and_z(self, c, alpha):
        mu, z = self.mu_and_z_batch(np.asarray(c)[None], alpha)
        return mu[:, 0], z[:, 0]


def test_interference_objective_sees_competition_E_cannot():
    """Two members sharing m0: `E` is 0, the interference objective is not.

    Nobody secretes anything here, so `min(secretion, uptake) = 0` on every
    metabolite -- the handover objective is blind to a pure substrate
    competition by construction, which is the whole reason for the second mode.
    """
    sur, c, X = _ToySur(), np.array([1.0, 1.0]), np.array([1.0, 1.0])
    assert interaction.objective(sur, c, X)[0] == 0.0

    # dt = 0.1 * min(1/2, 1/2) = 0.05; member 0 sees m0 at 0.95 alone and 0.90
    # with its partner, member 1's own substrate m1 is untouched by member 0.
    v, d = interaction.objective_interference(sur, c, X, frac=0.1)
    assert d[0] == pytest.approx(1.0 * 0.05 / 0.05)  # X_0 * (0.95 - 0.90) / dt
    assert d[1] == pytest.approx(0.0)
    assert v == pytest.approx(d.sum()) and v > 0  # positive == suppression

    vr, dr = interaction.objective_interference(sur, c, X, frac=0.1, relative=True)
    assert dr[0] == pytest.approx(0.05 / (0.95 * 0.05))  # divided by mu_alone
    assert vr == pytest.approx(dr.sum()) and vr > 0


def test_the_absolute_loss_vanishes_where_the_relative_rate_saturates():
    """Why the objective is absolute: starving everyone must not be the optimum.

    In the scarce regime uptake is proportional to `c`, so the depletion time
    `c/|dc|` cancels `c` and the *relative* rate saturates at a model constant
    -- measured at `(4/19) Vmax/Km` on 10 of 10 design runs, reached by starving
    a trace metal (§13.5). The absolute loss keeps the numerator only, so it goes
    to zero on the same path and the degenerate optimum eliminates itself.
    """
    sur, X = _MMSur(), np.array([1.0, 1.0])
    out = [
        (
            interaction.objective_interference(sur, np.array([c0, 1.0]), X, frac=0.1)[0],
            interaction.objective_interference(
                sur, np.array([c0, 1.0]), X, frac=0.1, relative=True
            )[0],
        )
        for c0 in (1.0, 1e-3, 1e-6)
    ]
    absolute, relative = zip(*out, strict=True)
    # Starve the shared substrate over six decades: the absolute loss follows it
    # down, the relative rate holds at its cap.
    assert absolute[0] > absolute[1] > absolute[2]
    assert absolute[2] < absolute[0] / 1e5
    assert all(r > 0.5 for r in relative) and relative[2] > relative[0]


class _MMSur(_ToySur):
    """`_ToySur` with §3.3's uptake bound, so `z` falls with `c` as the LP's does.

    `_ToySur`'s constant `z` cannot show the saturation the objective choice
    turns on -- uptake has to be proportional to `c` in the scarce limit for the
    depletion time to cancel it.
    """

    def mu_and_z_batch(self, C, alpha):
        C = np.asarray(C, dtype=np.float64)
        u = C / (self.km + C)  # (B, M)
        return self.mu_batch(C), self.Z[:, None, :] * u[None]


def test_interference_objective_batch_matches_the_single_path():
    sur, X = _ToySur(), np.array([1.0, 1.0])
    C = np.array([[1.0, 1.0], [2.0, 0.5]])
    v, d = interaction.objective_interference_batch(sur, C, X)
    for b, c in enumerate(C):
        v1, d1 = interaction.objective_interference(sur, c, X)
        assert v[b] == pytest.approx(v1) and d[b] == pytest.approx(d1)


def test_objective_spec_handover_truth_is_the_exchange_sum():
    z = np.array([[+1.0, -2.0], [-1.0, +1.0]])
    X = np.array([1.0, 1.0])
    spec = interaction.objective_spec("handover")
    got = spec.truth(None, ["EX_a_e", "EX_b_e"], np.ones(2), z, X, ["a", "b"])
    assert got == pytest.approx(interaction.exchange(z, X).sum())
    with pytest.raises(ValueError):
        interaction.objective_spec("nope")
