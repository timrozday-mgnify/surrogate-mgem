"""§4.6 resampling: error-directed budget, then ensemble gradient disagreement.

The three acquisition signals in the order they should be spent (see
:mod:`cfs.surrogate.ensemble`): band placement from ``u*``, budget split from
*measured* held-out error, and — only for metabolites neither of those can see —
ensemble disagreement on the gradient.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from cfs.groundtruth.solve import load_km_defaults
from cfs.sampling.active_subspace import ActiveSubspace
from cfs.sampling.design import SamplingConfig, sample_media, topup_weights

jax = pytest.importorskip("jax")
pytest.importorskip("equinox")


def test_topup_weights_follow_measured_error():
    diag = {
        "EX_good_e": {"rows": 900, "grad_cosine": 0.99},
        "EX_mid_e": {"rows": 300, "grad_cosine": 0.70},
        "EX_bad_e": {"rows": 40, "grad_cosine": 0.10},
    }
    w = topup_weights(diag)
    assert sum(w.values()) == pytest.approx(1.0)
    assert w["EX_bad_e"] > w["EX_mid_e"] > w["EX_good_e"]
    # The floor keeps the already-correct metabolite in the design: it only stays
    # correct while it keeps getting media.
    assert w["EX_good_e"] > 0.0

    # A perfect model splits evenly rather than dividing by zero.
    even = topup_weights({"a": {"grad_cosine": 1.0}, "b": {"grad_cosine": 1.0}})
    assert even == pytest.approx({"a": 0.5, "b": 0.5})


def test_focus_weights_move_media_to_the_weak_metabolite():
    ex = [f"EX_m{i}_e" for i in range(4)]
    sub = ActiveSubspace("g", ex, [], dict.fromkeys(ex, 1.0), 10.0)
    km_cfg = load_km_defaults()
    cfg = SamplingConfig(n_media=2048, seed=1)

    def counts(fw):
        media = sample_media(sub, km_cfg, cfg, None, fw)
        c = Counter(min(ex, key=lambda e: m[e]) for m in media)
        return {e: c.get(e, 0) for e in ex}

    even = counts(None)
    # EX_m0_e is the one the surrogate got wrong.
    skewed = counts(
        topup_weights(
            {
                "EX_m0_e": {"grad_cosine": 0.1},
                "EX_m1_e": {"grad_cosine": 0.95},
                "EX_m2_e": {"grad_cosine": 0.95},
                "EX_m3_e": {"grad_cosine": 0.95},
            }
        )
    )
    assert skewed["EX_m0_e"] > even["EX_m0_e"]
    assert all(skewed[e] > 0 for e in ex)  # the floor keeps everyone represented


def test_gradient_disagreement_is_zero_on_agreement_and_ignores_replete():
    """Disagreement must score *direction*, and must not rank rows where every
    member predicts ~nothing — 23% of real rows are replete and have no direction
    to disagree about."""
    from cfs.surrogate.ensemble import gradient_disagreement
    from cfs.surrogate.picnn import stack_heads

    G, M, B = 2, 5, 32
    mask = np.ones((G, M), dtype=bool)
    a = stack_heads(jax.random.PRNGKey(0), G, M, mask, width=8, depth=2)
    b = stack_heads(jax.random.PRNGKey(1), G, M, mask, width=8, depth=2)
    x = np.random.default_rng(0).uniform(0, 1, (G, B, M)).astype(np.float32)
    x_scale = np.ones((G, M), np.float32)

    same = np.asarray(gradient_disagreement([a, a], x, x_scale, mask))
    assert np.allclose(same, 0.0, atol=1e-5)  # a member cannot disagree with itself

    diff = np.asarray(gradient_disagreement([a, b], x, x_scale, mask))
    assert diff.shape == (G, B)
    assert np.all(diff >= -1e-6) and diff.max() > 0.0

    # A head whose predicted gradient is ~0 everywhere (all inputs masked off)
    # scores 0, not the top of the ranking.
    dead = stack_heads(jax.random.PRNGKey(2), G, M, np.zeros((G, M), bool), width=8, depth=2)
    off = np.asarray(gradient_disagreement([dead, dead], x, x_scale, np.zeros((G, M), bool)))
    assert np.allclose(off, 0.0)


def test_unmeasured_metabolites_are_the_ones_with_no_evidence():
    from cfs.surrogate.ensemble import unmeasured_metabolites

    diag = {"per_organism": {"g0": {"per_limiting_metabolite": {"EX_a_e": {"grad_cosine": 0.9}}}}}
    out = unmeasured_metabolites(diag, {"g0": ["EX_a_e", "EX_b_e", "EX_c_e"]})
    # EX_a_e limited and so has both a u* and an error; the other two have neither.
    assert out == {"g0": ["EX_b_e", "EX_c_e"]}


def test_topup_cli_weights_every_metabolite_including_the_never_measured(tmp_path):
    """`cfs topup`: measured error where there is one, the floor where there is not."""
    import json

    from cfs.cli import main
    from cfs.sampling.active_subspace import write_subspaces

    ex = ["EX_seen_e", "EX_bad_e", "EX_blind_e"]
    write_subspaces(
        [ActiveSubspace("g0", ex, [], dict.fromkeys(ex, 1.0), 5.0)], tmp_path / "g0.subspace.json"
    )
    diag = {
        "per_organism": {
            "g0": {
                "per_limiting_metabolite": {
                    "EX_seen_e": {"grad_cosine": 0.95},
                    "EX_bad_e": {"grad_cosine": 0.1},
                }
            }
        }
    }
    (tmp_path / "diagnostics.json").write_text(json.dumps(diag))

    out = tmp_path / "weights.json"
    assert (
        main(
            [
                "topup",
                "--diagnostics",
                str(tmp_path / "diagnostics.json"),
                "--labels",
                str(tmp_path),
                "--out",
                str(out),
            ]
        )
        == 0
    )

    w = json.loads(out.read_text())["g0"]
    assert set(w) == set(ex) and sum(w.values()) == pytest.approx(1.0)
    assert w["EX_bad_e"] > w["EX_seen_e"]
    # EX_blind_e never limited, so it has no error to read — it must still get a
    # share, and the §4.7 probe has now given it a band worth sampling.
    assert w["EX_blind_e"] > 0.0
