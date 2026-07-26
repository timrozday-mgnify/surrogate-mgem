"""Command-line entry point: ``cfs {qc, freeze-index, degeneracy, active-subspace, generate}``.

M0-M2 of the v2 plan (``docs/design/community-fba-surrogates-plan-v2.md``). All
subcommands need the ``data`` extra (cobra; ``qc`` also needs the ``memote`` CLI;
``generate`` also needs ``pyarrow``). The Nextflow ``QC_MODELS`` process runs
``qc`` then ``freeze-index``; ``DEGENERACY_SURVEY`` runs ``degeneracy``. §4
sampling is ``active-subspace`` (the sensitivity sweep) then ``generate`` (label
shards -> parquet by organism × eps).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    """Return the top-level parser with one subparser per M0/M1 command."""
    parser = argparse.ArgumentParser(prog="cfs", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    qc = sub.add_parser("qc", help="M0: EGC pre-flight + MEMOTE over the roster.")
    qc.add_argument("--roster", type=Path, required=True, help="CSV: genome_id, model_path.")
    qc.add_argument("--outdir", type=Path, required=True, help="Output directory.")
    qc.add_argument("--no-memote", action="store_true", help="Skip MEMOTE (EGC gate only).")

    fi = sub.add_parser("freeze-index", help="M0: derive + freeze the metabolite index.")
    fi.add_argument("--roster", type=Path, required=True, help="CSV: genome_id, model_path.")
    fi.add_argument(
        "--out", type=Path, default=Path("config/metabolite_index.json"),
        help="Index JSON path (default config/metabolite_index.json).",
    )

    dg = sub.add_parser("degeneracy", help="M1: exchange-FVA degeneracy survey (decides D4).")
    dg.add_argument("--roster", type=Path, required=True, help="CSV: genome_id, model_path.")
    dg.add_argument("--outdir", type=Path, required=True, help="Output directory.")
    dg.add_argument("--alpha", default="1.0,0.7", help="Comma-separated growth fractions.")
    dg.add_argument("--n-media", type=int, default=50, help="Media sampled per organism.")
    dg.add_argument("--seed", type=int, default=0)

    asp = sub.add_parser("active-subspace", help="§4.2: per-organism sensitive-metabolite sweep.")
    asp.add_argument("--roster", type=Path, required=True, help="CSV: genome_id, model_path.")
    asp.add_argument("--out", type=Path, required=True, help="Subspaces JSON (fed to generate).")
    asp.add_argument("--tol", type=float, default=1e-3, help="Relative mu_max-drop threshold.")

    gen = sub.add_parser("generate", help="§4.5: solve label shards -> parquet by (organism, eps).")
    gen.add_argument("--roster", type=Path, required=True, help="CSV: genome_id, model_path.")
    gen.add_argument("--index", type=Path, required=True, help="Frozen metabolite_index.json.")
    gen.add_argument("--outdir", type=Path, required=True, help="Parquet shard root.")
    gen.add_argument("--n-media", type=int, help="Override media per organism (default 20000).")
    gen.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)

    # Imports are deferred so `--help` works without the data extra installed.
    from surrogate_mgem.data import read_roster

    roster = read_roster(args.roster)

    if args.command == "qc":
        from cfs.groundtruth.qc import qc_roster

        summary = qc_roster(roster, args.outdir, memote=not args.no_memote)
        print(json.dumps(summary, indent=2))
        return 0

    if args.command == "freeze-index":
        from cfs.groundtruth.index import derive_index_from_roster, write_index

        result = derive_index_from_roster(roster)
        digest = write_index(result, args.out)
        print(
            f"wrote {args.out} — {len(result.index)} exchanges "
            f"({len(result.shared)} shared, {len(result.private)} private) "
            f"sha256={digest}"
        )
        return 0

    if args.command == "degeneracy":
        from cobra.io import read_sbml_model

        from cfs.validate.degeneracy import sample_media, survey_roster

        alphas = tuple(float(a) for a in args.alpha.split(","))
        media_per_model = {
            gm.genome_id: sample_media(
                read_sbml_model(str(gm.model_path)), args.n_media, seed=args.seed
            )
            for gm in roster
        }
        rec = survey_roster(roster, media_per_model, args.outdir, alphas=alphas)
        print(json.dumps(rec, indent=2))
        return 0

    if args.command == "active-subspace":
        from cobra.io import read_sbml_model

        from cfs.sampling.active_subspace import active_subspace, write_subspaces

        subspaces = [
            active_subspace(read_sbml_model(str(gm.model_path)), gm.genome_id, tol=args.tol)
            for gm in roster
        ]
        write_subspaces(subspaces, args.out)
        print(f"wrote {args.out} — {sum(len(s.active) for s in subspaces)} active "
              f"metabolites across {len(subspaces)} organisms")
        return 0

    if args.command == "generate":
        from dataclasses import replace

        from cfs.sampling.design import SamplingConfig
        from cfs.sampling.generate import generate_roster

        cfg = SamplingConfig(seed=args.seed)
        if args.n_media is not None:
            cfg = replace(cfg, n_media=args.n_media)
        shards = generate_roster(roster, args.index, args.outdir, cfg)
        print(json.dumps(
            {s.genome_id: {"n_media": s.n_media, "shards": [str(p) for p in s.paths]}
             for s in shards}, indent=2))
        return 0

    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
