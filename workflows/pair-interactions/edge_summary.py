#!/usr/bin/env python
"""Every directed metabolite handover a community's pairs can have, as graph edges.

The survey (`survey/<arm>/shard_*.npz`) solves **every member** at each drawn
medium, so one set of shards answers the question for all `G(G-1)` ordered pairs:
member `a` hands metabolite `m` to member `b` at the rate
`min(a secretes, b takes up)`. That is why the survey is run over the whole
community and not per pair -- `G` solves a medium instead of `2` per pair.

Four tables, written as CSVs. Every row stands alone -- no run, seed or design is
referenced -- so rows from different communities concatenate:

``edges.csv``         one (arm, metabolite, producer, consumer). Potential
                      (`reach` = % of survey media, with a Wilson interval),
                      magnitude (`rate_typical` median where present,
                      `rate_capacity` at the best design) and support
                      (`forced` FVA fraction, `seeds`), kept apart because they
                      disagree: the largest handovers are often the solver's
                      tie-break while a small one is forced.
``graph_edges.csv``   one (arm, producer, consumer): the graph's directed edge.
                      Totals are summed over metabolites **at one medium** and
                      then aggregated over media -- never as a sum of the
                      per-metabolite maxima, which come from different media and
                      add up to a rate nothing delivers.
``relationships.csv`` one (arm, unordered pair): both directions on a row, so
                      `both_ways_pct` can say whether an exchange is mutual or
                      two one-way flows in different media.
``pairs.csv``         the same, ranked, for deciding which pairs are worth a
                      search: `cfs interactions` costs ~10 core-h a pair and the
                      survey costs nothing extra per pair.

Rates are mmol/gDW/h at equal biomass (`X = 1` each) and inherit the GEMs'
`|Vmax| = 1000`, so they are ~30x physiological: comparable with each other, not
with a stopwatch (CLAUDE.md, "Per-organism `Vmax`").

With `--robust crosseval/*/robust.csv` the designed-medium columns are filled in;
without it the tables are survey-only, which is what the ranking runs on before
any search.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

BUFFERED = ("EX_h_e", "EX_h2o_e")  # held by the vessel: never an interaction (§13.5)
_TOL = 1e-6  # mmol/gDW/h: below this the LP is reporting solver dust


def name(m: str) -> str:
    return m.removeprefix("EX_").removesuffix("_e")


def wilson(k: np.ndarray, n: int, z: float = 1.96):
    """95% interval for a proportion. Normal-approximation CIs go negative here:
    most handovers sit at a handful of media in thousands."""
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def load_survey(survey: Path) -> dict[str, dict]:
    """`{arm: {Z, mu, exchanges, gids, shards}}` from a survey directory.

    Two layouts, because the pipeline stages every shard into one directory and
    publishes them into per-arm ones: `<arm>/shard_<k>.npz` as published, and
    flat `<arm>__shard_<k>.npz` as staged.
    """
    groups: dict[str, list[Path]] = {}
    for f in sorted(survey.rglob("*shard_*.npz")):
        if not f.stat().st_size:
            continue
        arm = f.name.split("__")[0] if "__" in f.name else f.parent.name
        groups.setdefault(arm, []).append(f)
    out = {}
    for arm, fs in sorted(groups.items()):
        fs = sorted(fs, key=lambda f: int(f.stem.rsplit("_", 1)[1]))
        zs = [np.load(f, allow_pickle=True) for f in fs]
        out[arm] = {
            "Z": np.concatenate([z["z"] for z in zs]).astype(float),
            "mu": np.concatenate([z["mu"] for z in zs]),
            "exchanges": [str(x) for x in zs[0]["exchanges"]],
            "gids": [str(x) for x in zs[0]["gids"]],
            "shards": len(fs),
        }
    return out


def pair_rates(Z: np.ndarray, a: int, b: int, cols: np.ndarray) -> np.ndarray:
    """`(media, kept metabolites)`: what `a` hands to `b`, metabolite by metabolite."""
    return np.minimum(np.clip(Z[:, a, cols], 0, None), np.clip(-Z[:, b, cols], 0, None))


def survey_edges(S: dict[str, dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-metabolite edges and per-ordered-pair totals, over the survey media."""
    ed, pt = [], []
    for arm, s in S.items():
        ex, gids, Z = s["exchanges"], s["gids"], s["Z"]
        cols = np.array([j for j, e in enumerate(ex) if e not in BUFFERED])
        mets = [name(ex[j]) for j in cols]
        n = len(Z)
        for a, ga in enumerate(gids):
            for b, gb in enumerate(gids):
                if a == b:
                    continue
                R = pair_rates(Z, a, b, cols)
                tot = R.sum(1)  # the pair's whole handover at each medium
                live = tot > _TOL
                pt.append(
                    {
                        "arm": arm,
                        "producer": ga,
                        "consumer": gb,
                        "survey_media": n,
                        "pair_media_pct": 100 * float(live.mean()),
                        "pair_rate_typical": float(np.median(tot[live])) if live.any() else 0.0,
                        "pair_rate_survey_max": float(tot.max()),
                    }
                )
                k = (R > _TOL).sum(0)
                for j in np.flatnonzero(k):
                    r = R[:, j]
                    hit = r > _TOL
                    lo, hi = wilson(k[j], n)
                    # share *within this ordered pair*, so the number means the
                    # same thing whatever else the community is doing
                    ed.append(
                        {
                            "arm": arm,
                            "metabolite": mets[j],
                            "producer": ga,
                            "consumer": gb,
                            "survey_media": n,
                            "reach_pct": 100 * k[j] / n,
                            "reach_lo": 100 * lo,
                            "reach_hi": 100 * hi,
                            "rate_typical": float(np.median(r[hit])),
                            "rate_survey_max": float(r.max()),
                            "share_typical": 100 * float(np.median(r[hit] / tot[hit])),
                        }
                    )
    return pd.DataFrame(ed), pd.DataFrame(pt)


def designed(robust: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Designed-medium evidence: per metabolite edge, and per ordered pair.

    `robust.csv` is already one row per handover at each run's **best** design
    under that run's **own** model, so no filtering is needed here. The pair total
    is summed within a run first -- one run's best design is one medium -- and the
    best run then taken.
    """
    # The columns the merge needs, so a survey-only run (no searches yet -- what the
    # ranking is for) still produces the same table with those entries empty.
    met_cols = [
        "arm",
        "metabolite",
        "producer",
        "consumer",
        "rate_capacity",
        "share_designed",
        "forced",
        "seeds",
        "objectives",
    ]
    fs = [pd.read_csv(f) for f in robust if f.stat().st_size > 1]
    if not fs:
        return (
            pd.DataFrame(columns=met_cols),
            pd.DataFrame(columns=["arm", "producer", "consumer", "pair_rate_capacity"]),
        )
    R = pd.concat(fs, ignore_index=True)
    R["metabolite"] = R.metabolite.map(name)
    per_met = (
        R.groupby(["arm", "metabolite", "producer", "consumer"])
        .agg(
            rate_capacity=("rate", "max"),
            share_designed=("share", "max"),
            forced=("forced_frac", "median"),
            seeds=("seed", "nunique"),
            objectives=("objective", "nunique"),
        )
        .reset_index()
    )
    per_pair = (
        R.groupby(["arm", "run", "producer", "consumer"])
        .rate.sum()
        .reset_index()
        .groupby(["arm", "producer", "consumer"])
        .rate.max()
        .rename("pair_rate_capacity")
        .reset_index()
    )
    return per_met, per_pair


def tables(S: dict[str, dict], robust: list[Path]) -> dict[str, pd.DataFrame]:
    sv, pt = survey_edges(S)
    per_met, per_pair = designed(robust)
    key = ["arm", "metabolite", "producer", "consumer"]
    ED = sv.merge(per_met, on=key, how="outer") if not sv.empty else per_met.copy()
    # An outer merge against an all-empty designed frame leaves object columns, and
    # `fillna` on those is deprecated and would silently stop downcasting.
    for c in (
        "reach_pct",
        "reach_lo",
        "reach_hi",
        "survey_media",
        "rate_typical",
        "share_typical",
        "rate_survey_max",
        "rate_capacity",
        "share_designed",
        "forced",
        "seeds",
        "objectives",
    ):
        ED[c] = pd.to_numeric(ED.get(c), errors="coerce")
    ED["reach_pct"] = ED.reach_pct.fillna(0.0)
    ED["reach"] = np.select(
        [ED.reach_pct >= 50, ED.reach_pct >= 1, ED.reach_pct > 0],
        ["common", "occasional", "rare"],
        default="designed only",
    )
    # One number for an edge's width. Never reach x magnitude: they are the axes
    # the summary keeps apart, and the biggest edges here are the least forced.
    ED["weight"] = ED[["rate_capacity", "rate_survey_max"]].max(axis=1).fillna(0.0)
    for c in ("seeds", "objectives", "survey_media"):
        ED[c] = ED[c].fillna(0).astype(int)
    ED = ED[
        [
            "arm",
            "metabolite",
            "producer",
            "consumer",
            "reach",
            "reach_pct",
            "reach_lo",
            "reach_hi",
            "survey_media",
            "rate_typical",
            "share_typical",
            "rate_survey_max",
            "rate_capacity",
            "share_designed",
            "forced",
            "seeds",
            "objectives",
            "weight",
        ]
    ]

    GE = (
        ED.groupby(["arm", "producer", "consumer"])
        .agg(
            metabolites=("metabolite", lambda s: ", ".join(sorted(s))),
            n_metabolites=("metabolite", "nunique"),
            n_common=("reach", lambda s: int((s == "common").sum())),
            n_occasional=("reach", lambda s: int((s == "occasional").sum())),
            n_forced=("forced", lambda s: int((s >= 0.99).sum())),
            max_reach_pct=("reach_pct", "max"),
        )
        .reset_index()
        .merge(pt, on=["arm", "producer", "consumer"], how="outer")
        .merge(per_pair, on=["arm", "producer", "consumer"], how="left")
    )
    for c in (
        "pair_media_pct",
        "pair_rate_typical",
        "pair_rate_survey_max",
        "pair_rate_capacity",
        "max_reach_pct",
        "survey_media",
    ):
        GE[c] = pd.to_numeric(GE.get(c), errors="coerce")
    # No run optimises one direction on its own -- the searches maximise the whole
    # of E, or a different objective -- so the best total *seen* is the ceiling we
    # can defend, and on some edges that is the survey's.
    GE["weight"] = GE[["pair_rate_capacity", "pair_rate_survey_max"]].max(axis=1).fillna(0.0)
    for c in ("n_metabolites", "n_common", "n_occasional", "n_forced"):
        GE[c] = GE[c].fillna(0).astype(int)
    # a pair with no handover still gets a row, from the per-pair totals
    GE["metabolites"] = GE.metabolites.fillna("")
    GE["max_reach_pct"] = GE.max_reach_pct.fillna(0.0)

    rec = []
    for arm, g in GE.groupby("arm"):
        w = {(r.producer, r.consumer): r for r in g.itertuples()}
        for a, b in sorted({tuple(sorted(k)) for k in w}):
            f, r_ = w.get((a, b)), w.get((b, a))
            wf, wr = (f.weight if f is not None else 0.0), (r_.weight if r_ is not None else 0.0)
            pf = f.pair_media_pct if f is not None else 0.0
            pr = r_.pair_media_pct if r_ is not None else 0.0
            nc = (f.n_common if f is not None else 0) + (r_.n_common if r_ is not None else 0)
            nmet = (f.n_metabolites if f is not None else 0) + (
                r_.n_metabolites if r_ is not None else 0
            )
            rec.append(
                {
                    "arm": arm,
                    "member_a": a,
                    "member_b": b,
                    "pct_a_to_b": pf,
                    "pct_b_to_a": pr,
                    "both_ways_pct": min(pf, pr),
                    "weight_a_to_b": wf,
                    "weight_b_to_a": wr,
                    "n_metabolites": nmet,
                    "n_common": nc,
                    "asymmetry": (wf - wr) / (wf + wr) if wf + wr else 0.0,
                    "relationship": (
                        "none"
                        if max(pf, pr) < 1
                        else "mutual"
                        if min(pf, pr) >= 50
                        else "one-way"
                        if min(pf, pr) < 1
                        else "mostly one-way"
                    ),
                }
            )
    RE = pd.DataFrame(rec)

    # Ranked for spending searches on. `n_common` first: a pair with several
    # handovers that happen at most media is worth a design, and a big rate at
    # one medium in a thousand is a lead. Ties by mutuality, then by rate.
    if RE.empty:
        PA = RE
    else:
        PA = (
            RE.groupby(["member_a", "member_b"])
            .agg(
                n_common=("n_common", "max"),
                n_metabolites=("n_metabolites", "max"),
                both_ways_pct=("both_ways_pct", "max"),
                rate=("weight_a_to_b", "max"),
                arms=("arm", "nunique"),
                relationships=("relationship", lambda s: ", ".join(sorted(set(s)))),
            )
            .reset_index()
            .sort_values(["n_common", "both_ways_pct", "rate"], ascending=False)
            .reset_index(drop=True)
        )
        PA.insert(0, "rank", PA.index + 1)
        PA["pair"] = PA.member_a + "," + PA.member_b
    return {"edges": ED, "graph_edges": GE, "relationships": RE, "pairs": PA}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--survey", type=Path, required=True, help="the survey/ directory")
    ap.add_argument(
        "--robust",
        type=Path,
        nargs="*",
        default=[],
        help="crosseval robust.csv files; omit for survey-only tables",
    )
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    S = load_survey(a.survey)
    if not S:
        raise SystemExit(f"no survey shards under {a.survey}")
    a.out.mkdir(parents=True, exist_ok=True)
    for stem, df in tables(S, [f for f in a.robust if f.exists()]).items():
        df.to_csv(a.out / f"{stem}.csv", index=False)
        print(f"{stem}.csv: {len(df)} rows")


if __name__ == "__main__":
    main()
