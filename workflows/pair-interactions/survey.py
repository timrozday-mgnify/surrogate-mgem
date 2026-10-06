#!/usr/bin/env python
"""Screen media with the true LP alone: which handovers the pair has, and how often.

The search's own screen is 64 draws plus the candidate media, and its handover
list was still growing at 64 (CLAUDE.md, pair-interactions). This draws media
from the same prior (§4.3 over the union of the members' active subspaces,
buffered species saturated -- exactly what `cfs interactions` draws) and solves
each one per member, FBA + elastic net. No surrogate, so it is the survey to
trust; the report turns the shards into frequencies, accumulation curves and an
estimate of how many handovers are still unseen.

Each shard is seeded from (seed, shard) only, so adding shards adds new media and
`-resume` keeps the old ones: raise --survey_shards until the report says the
list has stopped growing.

Writes --out: an npz with `c` (media x exchanges, mM), `z` (media x members x
exchanges, mmol/gDW/h, float32), `mu` (media x members), `exchanges`, `gids`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cfs.compose.dfba import _km_vector, community_medium
from cfs.groundtruth.solve import load_km_defaults, solve
from cfs.science.interaction import _BUFFER_SAT, ceq_map, keep_mask, subseed

_TAG = 2  # subseed tag: 0/1/3 are the search's draws, candidates, inhibited links


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--gems", type=Path, required=True)
    ap.add_argument("--value", type=Path, required=True, help="checkpoint: the exchange index")
    ap.add_argument("--pair", required=True, help="comma-separated genome ids")
    ap.add_argument("--ceq", type=float, default=None, help="c^eq (mM); omit for plain FBA")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n", type=int, default=250, help="media in this shard")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    import cobra

    gids = a.pair.split(",")
    models = [cobra.io.read_sbml_model(str(a.gems / f"{g}.xml")) for g in gids]
    ex = [
        e
        for e in json.loads((a.value / "value_heads.json").read_text())["exchanges"]
        if not e.startswith("theta:")
    ]  # the inhibited heads' second input block
    keep = keep_mask(ex)
    km = _km_vector(ex)
    ceq = None if a.ceq is None else ceq_map({"default": a.ceq}, ex, keep)
    km_cfg = load_km_defaults()
    col = {e: j for j, e in enumerate(ex)}

    C = np.zeros((a.n, len(ex)))
    Z = np.zeros((a.n, len(gids), len(ex)), dtype=np.float32)
    MU = np.zeros((a.n, len(gids)))
    for d in range(a.n):
        c = community_medium(a.labels, gids, ex, subseed(a.seed, a.shard, _TAG, d), None)
        c[~keep] = _BUFFER_SAT * km[~keep]  # interaction.buffer_medium, without a Surrogate
        C[d] = c
        conc = dict(zip(ex, c.tolist(), strict=True))
        for i, m in enumerate(models):
            sol = solve(m, conc, 1.0, 1e-3, km_cfg, ceq)
            if sol.status != "optimal":
                continue  # P2: no growth, no fluxes
            MU[d, i] = sol.mu_max
            for e, v in sol.z.items():
                Z[d, i, col[e]] = v
    np.savez_compressed(
        a.out,
        c=C,
        z=Z,
        mu=MU,
        exchanges=np.array(ex),
        gids=np.array(gids),
        seed=a.seed,
        shard=a.shard,
    )


if __name__ == "__main__":
    main()
