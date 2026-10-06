#!/usr/bin/env python
"""Score every designed medium under every model, with the true LP (no surrogate).

Each `cfs interactions` run reports its design under the model it searched with,
so two arms differ both by the model and by where the search went. Here every
start and designed medium of every run is re-solved under plain FBA and under each
c^eq the arms used, at the same uniform abundance: the fixed-medium comparison
(CLAUDE.md, "Compare at a fixed medium"). The best design of each run also gets
the directional spent-medium assay under each model, which splits a partner's
effect into depletion (competition) and conditioning (product inhibition).

Writes, into --out:
  media.csv     one row per (run, design, medium kind, model): E, mu per member
  links.csv     one row per (..., metabolite): true rate and per-member flux
  fluxes.csv    one row per (run, medium, model, member, exchange) at each best
                design and its start: the member's full true-LP exchange profile
                (z > 0 secreted, z < 0 taken up, mmol/gDW/h), with that
                metabolite's handover rate, buffered species (H+, H2O) excluded
  media_composition.csv.gz
                one row per medium a run screened (draws + candidate media) or
                designed: its variant, target, true-LP E (own model), and the
                concentration of every exchange (mM). No solves: from media.npz
  spent.csv     one row per (run, model, donor -> recipient), best designs only
  robust.csv    one row per (run, model, handover) at each best design: is the
                handover forced or a tie-break among alternative optima (FVA on
                that exchange at mu_max and at 0.95 mu_max), does it survive the
                elastic-net eps (1e-2, 1e-4), and its rate and share of E at
                abundance ratios 1:10..10:1 (from the same z, no solves)
  revert.csv    one row per (run, metabolite the best design moved): how much E
                drops when that one metabolite goes back to its start value, under
                the run's own model. A design moves ~150-250 metabolites and E is
                flat along nearly all of them, so the fold change says where the
                search drifted and this says what the design is made of.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cfs.groundtruth.solve import apply_mm_bounds, load_km_defaults, solve
from cfs.science.interaction import ceq_map, exchange, keep_mask, spent_medium_assay

_MOVED_DECADES = 0.05  # below this a metabolite counts as unmoved
_TOL = 1e-6  # link threshold, as in `links.csv`
_EPS_ALT = (1e-2, 1e-4)  # the label set's other elastic-net levels
_FRACS = (1.0, 0.95)  # FVA at mu_max, and at near-optimal growth
_RATIOS = (0.1, 0.3, 1.0, 3.0, 10.0)  # X_first / X_rest, total biomass held at G


def true_solve(models, exchanges, c, ceq, eps=1e-3):
    """(mu per member, z per member) from the true FBA + elastic-net solve."""
    km = load_km_defaults()
    conc = dict(zip(exchanges, c.tolist(), strict=True))
    col = {ex: j for j, ex in enumerate(exchanges)}
    mu, z = np.zeros(len(models)), np.zeros((len(models), len(exchanges)))
    for i, m in enumerate(models):
        sol = solve(m, conc, 1.0, eps, km, ceq)
        if sol.status != "optimal":
            continue
        mu[i] = sol.mu_max
        for ex, v in sol.z.items():
            z[i, col[ex]] = v
    return mu, z


def designs(run: Path):
    """[(design index, is_best, c_start, c_design)] and the run's metadata."""
    meta = json.loads((run / "run.json").read_text())
    cell = json.loads((run / "interactions.json").read_text())["cells"][0]
    npz = np.load(run / "media.npz", allow_pickle=True)
    e_hat = [d["E_hat"] for d in cell["designs"]]
    best = int(np.argmin([abs(e - cell["E_designed"]) for e in e_hat]))
    rows = [(k, k == best, npz["c_start"][k], npz["c_design"][k]) for k in range(len(e_hat))]
    return meta, cell["community"], [str(x) for x in npz["exchanges"]], rows


def fva(models, exchanges, c, ceq, cols, frac):
    """(min, max) of each member's flux on exchanges `cols` over the optima at
    `frac * mu_max`, same bounds as :func:`true_solve`; (G, len(cols)) each.
    A member without the exchange is pinned at 0."""
    from cobra.flux_analysis import flux_variability_analysis

    km = load_km_defaults()
    conc = dict(zip(exchanges, c.tolist(), strict=True))
    lo, hi = np.zeros((len(models), len(cols))), np.zeros((len(models), len(cols)))
    for i, m in enumerate(models):
        ids = [exchanges[j] for j in cols if exchanges[j] in m.reactions]
        if not ids:
            continue
        with m:
            apply_mm_bounds(m, conc, km, ceq)
            try:  # processes=1: a worker pool re-pickles the GEM per call (CLAUDE.md)
                f = flux_variability_analysis(m, ids, fraction_of_optimum=frac, processes=1)
            except Exception:  # infeasible / no growth: no optimum to vary over
                continue
        for k, j in enumerate(cols):
            if exchanges[j] in f.index:
                lo[i, k], hi[i, k] = f.loc[exchanges[j], ["minimum", "maximum"]]
    return lo, hi


def robust(models, ex, keep, c, z, ceq, key):
    """Each handover at one medium: forced or a tie-break, eps, abundance ratio.

    The handover is read off the elastic-net QP's unique optimum, but the LP
    behind it is degenerate on ~69% of exchanges (M1), so a link can be the
    tie-break the QP picked rather than something the network must do. FVA gives
    the rate *every* optimum delivers (`forced_rate`) and the most any does.
    Abundance only rescales members' z (each member's LP is its own), so the
    ratio columns need no solves.
    """
    G = len(models)
    e = exchange(z, np.ones(G), keep)
    cols = list(np.flatnonzero(e > _TOL))
    if not cols:
        return []
    alt = {
        eps: exchange(true_solve(models, ex, c, ceq, eps)[1], np.ones(G), keep) for eps in _EPS_ALT
    }
    lim = {f: fva(models, ex, c, ceq, cols, f) for f in _FRACS}
    rows = []
    for k, j in enumerate(cols):
        r = key | {
            "metabolite": ex[j],
            "producer": int(np.argmax(z[:, j])),
            "consumer": int(np.argmin(z[:, j])),
            "rate": e[j],
            "share": e[j] / e.sum(),
        }
        for f, (lo, hi) in lim.items():
            # guaranteed: what every optimum secretes / takes up; possible: the most
            sec_min, upt_min = np.maximum(lo[:, k], 0).sum(), np.maximum(-hi[:, k], 0).sum()
            sec_max, upt_max = np.maximum(hi[:, k], 0).sum(), np.maximum(-lo[:, k], 0).sum()
            r[f"forced_rate@{f}"] = min(sec_min, upt_min)
            r[f"max_rate@{f}"] = min(sec_max, upt_max)
        # share of the reported rate that every optimum delivers
        r["forced_frac"] = f = r["forced_rate@1.0"] / e[j]
        r["class"] = "forced" if f > 0.99 else "partly forced" if f > 0.01 else "tie-break"
        for eps, ea in alt.items():
            r[f"rate@eps={eps:g}"] = ea[j]
        for q in _RATIOS:
            X = np.ones(G)
            X[0] = q
            X *= G / X.sum()
            ex_q = exchange(z, X, keep)
            r[f"rate@X={q:g}"] = ex_q[j]
            r[f"share@X={q:g}"] = ex_q[j] / ex_q.sum()
        rows.append(r)
    return rows


def reverts(models, ex, keep, X, c0, c1, meta, key):
    """Per moved metabolite: E(design) - E(design with that one at its start value)."""
    ceq = None if meta["ceq"] is None else ceq_map({"default": float(meta["ceq"])}, ex, keep)

    def e_of(c):
        return exchange(true_solve(models, ex, c, ceq)[1], X, keep).sum()

    e1 = e_of(c1)
    fold = np.log10((c1 + 1e-30) / (c0 + 1e-30))
    rows = []
    for j in np.flatnonzero(np.abs(fold) > _MOVED_DECADES):
        c = c1.copy()
        c[j] = c0[j]
        d = e1 - e_of(c)
        rows.append(
            key
            | {
                "metabolite": ex[j],
                "log10_fold": fold[j],
                "E_design": e1,
                "contribution": d,
                "contribution_rel": d / e1 if e1 else np.nan,
            }
        )
    return rows


def composition(runs) -> pd.DataFrame:
    """Every screened and designed medium of every run, one row each, all exchanges."""
    frames = []
    for run in sorted(runs):
        z = np.load(run / "media.npz", allow_pickle=True)
        if "pool_c" not in z:
            continue
        ex = [str(x) for x in z["exchanges"]]
        starts = {int(s) for s in z["start_draw"]}
        n, nd = len(z["pool_c"]), len(z["c_design"])
        meta = pd.DataFrame(
            {
                "run": run.name,
                "medium": [f"pool#{k}" for k in range(n)] + [f"design#{k}" for k in range(nd)],
                "variant": list(z["pool_variant"]) + ["design"] * nd,
                "target": list(z["pool_metabolite"]) + [f"from pool#{s}" for s in z["start_draw"]],
                "refined": [k in starts for k in range(n)] + [True] * nd,
                "E_true_screen": list(np.nansum(z["pool_ex_true"] * keep_mask(ex), 1))
                + [np.nan] * nd,
            }
        )
        conc = pd.DataFrame(np.vstack([z["pool_c"], z["c_design"]]), columns=ex)
        frames.append(pd.concat([meta, conc], axis=1))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--gems", type=Path, required=True)
    ap.add_argument("--ceq", default="", help="comma-separated c^eq values (mM) to score under")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    import cobra

    models_by = {}
    media, links, spent, revert, fluxes, rob = [], [], [], [], [], []
    for run in sorted(a.runs):
        meta, gids, ex, rows = designs(run)
        for g in gids:
            if g not in models_by:
                models_by[g] = cobra.io.read_sbml_model(str(a.gems / f"{g}.xml"))
        models = [models_by[g] for g in gids]
        keep = keep_mask(ex)
        X = np.ones(len(gids))
        arms = {"fba": None} | {
            f"ceq={v}": ceq_map({"default": float(v)}, ex, keep) for v in a.ceq.split(",") if v
        }
        for k, is_best, c0, c1 in rows:
            for kind, c in (("start", c0), ("design", c1)):
                for model, ceq in arms.items():
                    mu, z = true_solve(models, ex, c, ceq)
                    e = exchange(z, X, keep)
                    key = dict(
                        run=run.name, **meta, design=k, best=is_best, medium=kind, model=model
                    )
                    media.append(
                        key
                        | {"E_true": e.sum(), "n_links": int((e > 1e-6).sum())}
                        | {f"mu:{g}": m for g, m in zip(gids, mu, strict=True)}
                    )
                    for j in np.flatnonzero(e > 1e-6):
                        links.append(
                            key
                            | {"metabolite": ex[j], "rate": e[j]}
                            | {f"z:{g}": z[i, j] for i, g in enumerate(gids)}
                        )
                    if is_best:
                        for i, j in zip(*np.nonzero(np.abs(z) > 1e-6), strict=True):
                            if keep[j]:
                                fluxes.append(
                                    key
                                    | {
                                        "member": gids[i],
                                        "metabolite": ex[j],
                                        "z": z[i, j],
                                        "handover": e[j],
                                        "mu": mu[i],
                                    }
                                )
                    if is_best and kind == "design":
                        rob += [
                            x | {"producer": gids[x["producer"]], "consumer": gids[x["consumer"]]}
                            for x in robust(models, ex, keep, c, z, ceq, key)
                        ]
                        sm = spent_medium_assay(models, ex, c, z, X, gids, ceq=ceq, keep=keep)
                        for p in (sm or {}).get("pairs", []):
                            spent.append(key | p)
            if is_best:
                revert += reverts(models, ex, keep, X, c0, c1, meta, dict(run=run.name, **meta))
        print(run.name, "done", flush=True)

    a.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(media).to_csv(a.out / "media.csv", index=False)
    pd.DataFrame(links).to_csv(a.out / "links.csv", index=False)
    pd.DataFrame(spent).to_csv(a.out / "spent.csv", index=False)
    pd.DataFrame(revert).to_csv(a.out / "revert.csv", index=False)
    pd.DataFrame(fluxes).to_csv(a.out / "fluxes.csv", index=False)
    pd.DataFrame(rob).to_csv(a.out / "robust.csv", index=False)
    composition(a.runs).to_csv(a.out / "media_composition.csv.gz", index=False)


if __name__ == "__main__":
    main()
