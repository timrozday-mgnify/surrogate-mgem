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

from cfs.groundtruth.solve import load_km_defaults, solve
from cfs.science.interaction import ceq_map, exchange, keep_mask, spent_medium_assay

_MOVED_DECADES = 0.05  # below this a metabolite counts as unmoved


def true_solve(models, exchanges, c, ceq):
    """(mu per member, z per member) from the true FBA + elastic-net solve."""
    km = load_km_defaults()
    conc = dict(zip(exchanges, c.tolist(), strict=True))
    col = {ex: j for j, ex in enumerate(exchanges)}
    mu, z = np.zeros(len(models)), np.zeros((len(models), len(exchanges)))
    for i, m in enumerate(models):
        sol = solve(m, conc, 1.0, 1e-3, km, ceq)
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
    media, links, spent, revert, fluxes = [], [], [], [], []
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
    composition(a.runs).to_csv(a.out / "media_composition.csv.gz", index=False)


if __name__ == "__main__":
    main()
