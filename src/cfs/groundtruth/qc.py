"""M0 pre-flight QC for CarveMe models (plan §3.0, gate V0).

CarveMe gap-fills, and gap-filled models often contain thermodynamically
infeasible energy-generating cycles (EGCs) that make ATP from nothing. Any such
model makes every downstream growth prediction fiction, so this is a **hard
gate**: :func:`qc_roster` raises if any model has an EGC. MEMOTE runs alongside
as the defensible automated-QC answer to "you did no curation".

COBRApy only — no JAX, no MICOM. ``cobra`` lives in the ``data`` extra; imports
are function-local so the module loads solver-free.
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

LOGGER = logging.getLogger("cfs.qc")

_ATPM_CANDIDATES = ("ATPM", "ATPM_c", "NGAM")  # common maintenance-ATP reaction ids
_EGC_TOL = 1e-6


def has_egc(model) -> bool:
    """Whether ``model`` can generate ATP with all uptake closed (plan §3.0).

    Closes every exchange's uptake, maximises the maintenance-ATP reaction, and
    returns whether the optimum exceeds ``1e-6`` — i.e. ATP from nothing. Returns
    ``False`` (and warns) if the model has no recognisable maintenance reaction,
    since the test cannot be posed.
    """
    atpm = next((rid for rid in _ATPM_CANDIDATES if rid in model.reactions), None)
    if atpm is None:
        LOGGER.warning("%s has no ATPM/NGAM reaction; cannot test for EGC", model.id)
        return False
    with model:  # context manager restores bounds/objective on exit
        for ex in model.exchanges:
            ex.lower_bound = 0.0  # close all uptake
        model.objective = atpm
        sol = model.optimize()
    value = sol.objective_value
    return value is not None and value > _EGC_TOL


def run_memote(model_path: Path, out_html: Path) -> Path:
    """Write a MEMOTE snapshot HTML report for one model (plan §3.0).

    Shells out to the ``memote`` CLI rather than driving its Python API — the CLI
    is the supported entry point and reimplementing it buys nothing.

    ponytail: subprocess to `memote report snapshot`; if memote is missing or
    fails, warn and return the (absent) path rather than aborting the whole QC
    run — the EGC gate is the hard one, MEMOTE is advisory.
    """
    out_html = Path(out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "memote",
        "report",
        "snapshot",
        "--filename",
        str(out_html),
        str(model_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        detail = getattr(exc, "stderr", "") or str(exc)
        LOGGER.warning("MEMOTE failed for %s: %s", model_path, detail)
    return out_html


def qc_roster(roster, out_dir: Path, *, memote: bool = True) -> dict:
    """Run EGC + MEMOTE over every roster model; write ``qc_summary.json``.

    ``roster`` is a list of ``surrogate_mgem.data.GenomeModel``. Raises
    ``RuntimeError`` if any model has an EGC — the plan's hard gate. Returns the
    summary dict ``{genome_id: {egc, memote_report}}``.
    """
    from cobra.io import read_sbml_model

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict] = {}
    offenders: list[str] = []
    for gm in roster:
        model = read_sbml_model(str(gm.model_path))
        egc = has_egc(model)
        report = None
        if memote:
            report = str(run_memote(gm.model_path, out_dir / f"{gm.genome_id}.memote.html"))
        summary[gm.genome_id] = {"egc": bool(egc), "memote_report": report}
        if egc:
            offenders.append(gm.genome_id)
        LOGGER.info("QC %s: egc=%s", gm.genome_id, egc)

    (out_dir / "qc_summary.json").write_text(json.dumps(summary, indent=2))
    if offenders:
        raise RuntimeError(
            "EGC gate failed (plan §3.0/P0). Models generating ATP from nothing: "
            f"{sorted(offenders)}. Remove the cycle (loopless FBA) or drop them."
        )
    return summary
