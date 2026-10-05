"""A1 (design spec §8.5): a community-regime held-out label set.

Held-out media are drawn from the *same* design as the training media, so a
change to the design moves the ruler with the model (P24). Three label roots in
a row improved every held-out metric while §8.1 regressed 17x, and each cost a
~5 h relabel to find out. This set is drawn instead from the regime §8.1 runs
in -- a §4.3 draw over the **union** of a community's members' active subspaces,
the same object :func:`cfs.compose.dfba.community_medium` draws one of -- solved
once and reused forever. It is the direct instrument for that gap, and it turns
a 5 h blind loop into a minutes-long one.

``mu_max`` only: the failure it exists to catch is Head A's level at a
multi-limited medium (+153% on one member), which needs one FBA per
(organism, medium) and no elastic-net QP, no alpha grid and no duals.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

LOGGER = logging.getLogger(__name__)


def community_media(
    labels_dir: Path, gids: list[str], exchanges: list[str], n: int, seed: int, scales_path=None
) -> np.ndarray:
    """``(n, M)`` §4.3 draws over the union of ``gids``' active subspaces.

    Deliberately *not* built on :func:`cfs.compose.dfba.community_medium`, which
    draws one medium and whose RNG stream every M5 number on record depends on.
    """
    from cfs.groundtruth.solve import load_km_defaults
    from cfs.sampling.active_subspace import ActiveSubspace, load_subspaces
    from cfs.sampling.design import SamplingConfig, sample_media

    subs = {}
    for gid in gids:
        subs |= load_subspaces(Path(labels_dir) / f"{gid}.subspace.json")
    active = sorted({e for gid in gids for e in subs[gid].active})
    background = sorted({e for gid in gids for e in subs[gid].background} - set(active))
    union = ActiveSubspace("+".join(gids), active, background, {}, 0.0)
    scales = json.loads(Path(scales_path).read_text()) if scales_path else None
    cfg = SamplingConfig(n_media=len(active) + n, probe=False, seed=seed)
    # `sample_media` emits one all-but-one-depleted corner per active metabolite
    # first; those are a legitimate design point but always the same shape.
    media = sample_media(union, load_km_defaults(), cfg, scales=scales)[len(active) :]
    col = {ex: j for j, ex in enumerate(exchanges)}
    out = np.zeros((len(media), len(exchanges)))
    for i, m in enumerate(media):
        for ex, v in m.items():
            if ex in col:
                out[i, col[ex]] = v
    return out


def make(
    roster_path: Path,
    labels_dir: Path,
    index_path: Path,
    out: Path,
    *,
    communities: list[list[str]],
    n_media: int = 200,
    eps: float = 1e-3,
    seed: int = 0,
) -> dict:
    """Draw + solve the held-out set. Writes ``community_holdout.npz``."""
    import cobra

    from cfs.groundtruth.index import index_hash, load_index
    from cfs.groundtruth.solve import load_km_defaults, solve
    from surrogate_mgem.data import read_roster

    idx = load_index(Path(index_path))
    exchanges = list(idx.index)
    roster = {gm.genome_id: gm for gm in read_roster(Path(roster_path))}
    km_cfg = load_km_defaults()
    gids = sorted({g for c in communities for g in c})

    media, owner = [], []  # owner[k] = the community index medium k came from
    for n, c in enumerate(communities):
        m = community_media(labels_dir, c, exchanges, n_media, seed + n)
        media.append(m)
        owner += [n] * len(m)
    media = np.concatenate(media)

    mu = np.full((len(media), len(gids)), np.nan)
    for j, gid in enumerate(gids):
        model = cobra.io.read_sbml_model(str(roster[gid].model_path))
        # Only the media of communities this organism is in: the rest are a
        # different organism's regime and would dilute the measurement.
        rows = [k for k in range(len(media)) if gid in communities[owner[k]]]
        for k in rows:
            conc = dict(zip(exchanges, media[k].tolist(), strict=True))
            sol = solve(model, conc, 1.0, eps, km_cfg)
            mu[k, j] = sol.mu_max if sol.status == "optimal" else 0.0
        LOGGER.info("%s: %d media solved", gid, len(rows))

    Path(out).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        Path(out) / "community_holdout.npz",
        media=media,
        mu=mu,
        owner=np.asarray(owner),
        genome_ids=np.asarray(gids),
        exchanges=np.asarray(exchanges),
        index_hash=index_hash(Path(index_path)),
        communities=np.asarray([",".join(c) for c in communities]),
    )
    return {
        "n_media": int(len(media)),
        "n_organisms": len(gids),
        "n_solves": int(np.isfinite(mu).sum()),
    }


def score(holdout: Path, value_dir: Path, out: Path | None = None) -> dict:
    """Head A's ``mu`` against the held-out set. Signed, because the failure is bias."""
    from cfs.compose.dfba import Surrogate

    z = np.load(Path(holdout), allow_pickle=True)
    media, mu_true, gids = z["media"], z["mu"], [str(g) for g in z["genome_ids"]]
    sur = Surrogate(Path(value_dir))
    if sur.index_hash != str(z["index_hash"]):
        raise ValueError("checkpoint and holdout disagree on the metabolite index (P13)")

    from cfs.surrogate import calibrate

    jnp = sur._jnp
    cols = [sur.genome_ids.index(g) for g in gids]
    u = media / (sur.km + media)  # (N, M)
    x = (u[None] / (u[None] + sur.x_scale[:, None, :])).astype(np.float32)  # (G, N, M)
    raw = np.asarray(sur.mod.batched_value(sur._vheads, jnp.asarray(x)))  # (G, N)
    hat = (calibrate.apply(raw, sur.value_cal) * sur.mu_scale[:, None]).T[:, cols]
    hat = np.maximum(hat, 0.0)
    # Relative to the truth, floored at 1% of that organism's own held-out max --
    # a starving medium divides by ~0 and the failure is over-prediction at
    # mid-`mu`, not the bottom.
    floor = 0.01 * np.nanmax(mu_true, axis=0)
    rel = (hat - mu_true) / np.maximum(mu_true, floor)

    per = {}
    for j, gid in enumerate(gids):
        r = rel[np.isfinite(rel[:, j]), j]
        if not r.size:
            continue
        per[gid] = {
            "n": int(r.size),
            "median_signed": float(np.median(r)),
            "median_abs": float(np.median(np.abs(r))),
            "p90_abs": float(np.percentile(np.abs(r), 90)),
            "max_abs": float(np.max(np.abs(r))),
        }
    worst = max(per, key=lambda g: per[g]["median_abs"])
    rep = {
        "value_dir": str(value_dir),
        "holdout": str(holdout),
        "median_abs": float(np.median([v["median_abs"] for v in per.values()])),
        "worst_organism": [worst, per[worst]["median_abs"]],
        "worst_p90": float(max(v["p90_abs"] for v in per.values())),
        "median_signed": float(np.median([v["median_signed"] for v in per.values()])),
        "per_organism": per,
    }
    if out:
        Path(out).mkdir(parents=True, exist_ok=True)
        (Path(out) / "community_holdout_score.json").write_text(json.dumps(rep, indent=2))
    return rep
