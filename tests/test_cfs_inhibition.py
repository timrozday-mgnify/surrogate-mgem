"""§13.11 stage 4': the inhibition channel, in the labels and in the head's input.

Three things that would each be silent failures:

* with no ``inhibition.json`` every array keeps its old width and its old values,
  so a plain-FBA label root is bit-identical across this change;
* with one, Head A's input is ``[u | theta]`` and the *secretion* dual lands in
  the second block -- read off the same ``shadow`` column the uptake side uses,
  with the mirrored clamp (verified against finite differences of the true LP at
  20 binding cases, ratio 1.0000; cobra's `reduced_costs` came out at exactly 2x);
* Head B keeps the ``u`` half alone, because §13.11 gives it an inference-time
  clamp rather than an architecture change -- and its ``x_scale`` therefore still
  composes with Head A's under the P14 check.

No solver, no JAX: parquet and the frozen index only.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("yaml")

from cfs.groundtruth.index import derive_index, write_index  # noqa: E402
from cfs.groundtruth.solve import mm_upper_bound  # noqa: E402
from cfs.surrogate.data import (  # noqa: E402
    load_behaviour_dataset,
    load_value_dataset,
    theta,
)

EX = ["EX_glc__D_e", "EX_o2_e", "EX_mg2_e"]  # shard order
IDX = sorted(EX)  # the frozen index sorts; x/g/mask columns are in *this* order
GIDS = ["g0", "g1"]
CEQ = 0.05
VMAX = 1000.0
N = 24


def _rows(gid: str, mids: np.ndarray) -> pd.DataFrame:
    n = len(mids)
    # Medium sweeps EX_glc__D_e across c^eq so theta spans (0, 1) and pins at 0;
    # the other two sit fixed. Row m secretes glucose exactly at its own inhibited
    # capacity, so the secretion bound *binds* and its dual is a real sensitivity.
    c0 = np.linspace(0.0, 2 * CEQ, n)
    ub = np.array([mm_upper_bound(VMAX, c, CEQ) for c in c0])
    return pd.DataFrame(
        {
            "genome_id": gid,
            "index_hash": "",
            "medium_id": mids,
            "alpha": 1.0,
            "eps": 1e-3,
            "mu_max": np.linspace(0.5, 2.0, n),
            "status": "optimal",
            "medium": [[float(c), 0.02, 0.01] for c in c0],
            "z": [[float(u), -1.0, -0.5] for u in ub],
            # +0.25 on glucose: positive is the secretion side's sign (raising the
            # bound can only enlarge the feasible set). -0.5 on O2 is the uptake
            # side's. Mg is dust on both.
            "shadow": [[0.25, -0.5, 0.0] for _ in mids],
        }
    )


@pytest.fixture
def labels(tmp_path):
    result = derive_index(GIDS, [set(EX), set(EX)])
    index_path = tmp_path / "metabolite_index.json"
    digest = write_index(result, index_path)
    root = tmp_path / "labels"
    for gid in GIDS:
        (root / gid / "eps_0.001").mkdir(parents=True)
        (root / f"{gid}.exchanges.json").write_text(json.dumps({"exchanges": EX}))
        df = _rows(gid, np.arange(N))
        df["index_hash"] = digest
        df.to_parquet(root / gid / "eps_0.001" / "part.parquet", index=False)
    return root, index_path


def _inhibit(root):
    (root / "inhibition.json").write_text(json.dumps({"ceq": dict.fromkeys(EX, CEQ)}))


def test_no_inhibition_file_leaves_everything_at_its_old_width(labels):
    root, index_path = labels
    ds = load_value_dataset(root, index_path, eps=1e-3, seed=0)
    assert ds.exchanges == IDX
    assert ds.n_metabolites == len(EX)
    assert ds.ceq is None
    assert ds.x_train.shape[-1] == len(EX)
    assert ds.mask.shape == (len(GIDS), len(EX))


def test_theta_is_the_second_input_block_and_carries_the_secretion_dual(labels):
    root, index_path = labels
    plain = load_value_dataset(root, index_path, eps=1e-3, seed=0)
    _inhibit(root)
    ds = load_value_dataset(root, index_path, eps=1e-3, seed=0)

    m = len(EX)
    assert ds.n_metabolites == m
    assert ds.exchanges == IDX + [f"theta:{e}" for e in IDX]
    assert ds.x_train.shape[-1] == 2 * m
    assert ds.ceq == dict.fromkeys(IDX, CEQ)

    # The uptake half is untouched: adding a channel must not move the old one.
    assert np.allclose(ds.x_train[..., :m], plain.x_train)
    assert np.allclose(ds.g_train[..., :m], plain.g_train)

    # The theta block spans the range the design actually samples -- pinned at 0
    # above equilibrium, interior below it. Without both, the channel would be a
    # constant and the head could not learn the corner at all.
    th_raw = theta(np.linspace(0.0, 2 * CEQ, N), np.array([CEQ]))
    assert (th_raw == 0.0).any() and ((th_raw > 0) & (th_raw < 1)).any()

    # d(mu)/d(theta) = Vmax * shadow, and only where the secretion bound binds and
    # the dual is the right sign. Glucose binds by construction (z == ub); O2's
    # dual is the uptake side's, so it must not leak into the theta block.
    gt = ds.g_train[..., m:]
    glc, o2, mg = (IDX.index(e) for e in ("EX_glc__D_e", "EX_o2_e", "EX_mg2_e"))
    # The x -> x/(x+s) chain rule scales the target, so compare on the sign and
    # on which coordinates are non-zero, which is what the clamp decides.
    assert (gt[..., glc] > 0).any()
    assert np.all(gt[..., o2] == 0.0)
    assert np.all(gt[..., mg] == 0.0)


def test_a_never_secreted_metabolite_gets_no_theta_dual(labels):
    """The clamp that took held-out value R2 from -0.014 to 0.953 on a real GEM.

    At ``theta = 0`` the secretion bound is ``ub = 0``, so every metabolite the
    organism simply does not produce sits at ``z = 0 = ub`` and reads as binding,
    while its ``shadow`` is positive for the ordinary reason a nutrient's is. On
    AAXE02 that selected 7.7% of entries -- 99.9% of them at ``theta = 0``, over
    157 of 181 exchanges the network never secretes -- and seeding a max-affine
    head from those tangents makes it rise steeply in a direction the truth is
    flat in, so the min over planes over-predicts wherever ``theta > 0``.
    """
    root, index_path = labels
    _inhibit(root)
    m = len(EX)
    glc = IDX.index("EX_glc__D_e")

    ds = load_value_dataset(root, index_path, eps=1e-3, seed=0)
    assert (ds.g_train[..., m + glc] > 0).any()  # glucose is secreted in the shard

    # Same rows, but the organism never secretes anything: z <= 0 everywhere.
    for gid in GIDS:
        shard = root / gid / "eps_0.001" / "part.parquet"
        df = pd.read_parquet(shard)
        df["z"] = [[-1.0, -1.0, -0.5] for _ in range(len(df))]
        df.to_parquet(shard, index=False)
    none = load_value_dataset(root, index_path, eps=1e-3, seed=0)
    assert np.all(none.g_train[..., m:] == 0.0)
    # ...and the uptake channel is untouched by the capability test.
    assert np.array_equal(none.g_train[..., :m], ds.g_train[..., :m])


def test_head_b_keeps_the_uptake_half_so_the_two_heads_still_compose(labels):
    root, index_path = labels
    before = load_behaviour_dataset(root, index_path, eps=1e-3, seed=0)
    _inhibit(root)
    after = load_behaviour_dataset(root, index_path, eps=1e-3, seed=0)
    value = load_value_dataset(root, index_path, eps=1e-3, seed=0)

    assert after.exchanges == IDX and after.x_train.shape[-1] == len(EX)
    assert np.allclose(after.x_train, before.x_train)
    assert np.allclose(after.z_train, before.z_train)
    # P14: `dfba.Surrogate` compares Head B's x_scale against Head A's u half.
    assert np.allclose(value.x_scale[:, : len(EX)], after.x_scale)


def test_mm_upper_bound_is_affine_below_equilibrium_and_clipped_above():
    # Affine is the whole reason for this form: mu_max is concave in the LP's
    # bound vector, so it stays concave in theta only while the bound is affine.
    for c in (0.0, 0.01, 0.02, 0.04):
        assert mm_upper_bound(VMAX, c, CEQ) == pytest.approx(VMAX * (1 - c / CEQ))
    assert mm_upper_bound(VMAX, CEQ, CEQ) == 0.0
    assert mm_upper_bound(VMAX, 10 * CEQ, CEQ) == 0.0
