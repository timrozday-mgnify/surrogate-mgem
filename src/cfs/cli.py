"""``cfs {qc, freeze-index, degeneracy, active-subspace, generate, train-value, ...}``.

M0-M3 of the v2 plan (``docs/design/community-fba-surrogates-plan-v2.md``). The
M0-M2 subcommands need the ``data`` extra (cobra; ``qc`` also needs the ``memote``
CLI; ``generate`` also needs ``pyarrow``). The Nextflow ``QC_MODELS`` process runs
``qc`` then ``freeze-index``; ``DEGENERACY_SURVEY`` runs ``degeneracy``. §4
sampling is ``active-subspace`` (the sensitivity sweep) then ``generate`` (label
shards -> parquet by organism × eps).

``train-value`` and ``baseline-rf`` are M3 and are the odd ones out: they read
those label shards rather than any model, so they need the ``jax`` extra (the
forest only needs sklearn) and no solver stack, and take ``--labels``/``--index``
instead of ``--roster``. CLI-only for now — the heads train in minutes on a
laptop, so there is no Nextflow stage until M4 wants a cluster.
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
        "--out",
        type=Path,
        default=Path("config/metabolite_index.json"),
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
    gen.add_argument(
        "--scales",
        type=Path,
        help="JSON {genome_id: {exchange: scale}} from "
        "design.limiting_scales — fallback bands for the metabolites the LP "
        "demand probe finds no limiting regime for (§4.7).",
    )
    gen.add_argument(
        "--no-probe",
        action="store_true",
        help="Skip the §4.7 demand probe; "
        "band placement then falls back to --scales, roster median, 1.0.",
    )
    gen.add_argument(
        "--focus-weights",
        type=Path,
        help="JSON {genome_id: {exchange: weight}} "
        "from `cfs topup` — skews the focus budget toward the metabolites the "
        "trained head gets measurably wrong (§4.6).",
    )
    gen.add_argument(
        "--round",
        type=int,
        default=0,
        dest="round_idx",
        help="Top-up round index; >0 writes part.round<n>.parquet alongside the "
        "base shards instead of overwriting them.",
    )
    gen.add_argument("--seed", type=int, default=0)

    tu = sub.add_parser(
        "topup", help="§4.6: held-out diagnostics -> focus weights for the " "next generate round."
    )
    tu.add_argument(
        "--diagnostics", type=Path, required=True, help="diagnostics.json from train-value."
    )
    tu.add_argument(
        "--labels",
        type=Path,
        required=True,
        help="Label root holding the <id>.subspace.json sidecars.",
    )
    tu.add_argument("--out", type=Path, required=True, help="Focus-weights JSON.")
    tu.add_argument(
        "--floor",
        type=float,
        default=0.25,
        help="Share reserved for metabolites already predicted well.",
    )

    tv = sub.add_parser("train-value", help="M3: train Head A (concave mu_max) on label shards.")
    tv.add_argument("--labels", type=Path, required=True, help="Label shard root (§4.5).")
    tv.add_argument("--index", type=Path, required=True, help="Frozen metabolite_index.json.")
    tv.add_argument("--out", type=Path, required=True, help="Checkpoint + diagnostics dir.")
    tv.add_argument("--eps", type=float, default=1e-3, help="Which eps family level to train on.")
    tv.add_argument(
        "--arch",
        default="icnn",
        choices=[
            "icnn",
            "icnn-u",
            "deepset",
            "deepset-private",
            "deepset-u",
            "deepset-u-private",
            "groupmax-u",
            "mlp",
        ],
        help="Head architecture. `mlp` is unconstrained — a ceiling "
        "measurement, not a usable head.",
    )
    tv.add_argument(
        "--emb-dim", type=int, default=8, help="Metabolite embedding width (deepset only)."
    )
    tv.add_argument(
        "--gm-group",
        type=int,
        default=None,
        help="groupmax-u only: units per max group. width=1 depth=1 group=K is "
        "plain max-affine. Default 8.",
    )
    tv.add_argument(
        "--gm-temp",
        type=float,
        default=None,
        help="groupmax-u only: softmax temperature. Curvature scales as 1/T, so "
        "this is the Newton conditioning knob (§8). Default 0.1.",
    )
    tv.add_argument(
        "--gm-init",
        choices=["random", "labels"],
        default=None,
        help="groupmax-u only: `labels` seeds the first layer from the label "
        "tangents ranked by active-set frequency, instead of from noise.",
    )
    tv.add_argument(
        "--gm-reanchor",
        type=int,
        default=0,
        help="groupmax-u only: number of re-anchor passes -- re-seed the least-used "
        "planes from the worst-fit rows' tangents, evenly spaced over the run. "
        "Seeding fixes initialisation; this fixes planes that go dead during it.",
    )
    tv.add_argument(
        "--phi-hidden",
        type=int,
        default=None,
        help="deepset only: `phi` trunk width. Default derives it from "
        "--width (width // 8), which was a laptop-runtime choice.",
    )
    tv.add_argument(
        "--k-code", type=int, default=None, help="deepset only: pooled code width (default 16)."
    )
    tv.add_argument("--width", type=int, default=128)
    tv.add_argument("--depth", type=int, default=3)
    tv.add_argument("--epochs", type=int, default=400)
    tv.add_argument("--batch", type=int, default=512)
    tv.add_argument("--lr", type=float, default=3e-3)
    tv.add_argument("--w-grad", type=float, default=1.0, help="Sobolev term weight (§7.1).")
    tv.add_argument("--seed", type=int, default=0)
    tv.add_argument(
        "--organisms",
        help="Comma-separated genome_ids to stack (default: every shard under "
        "--labels). One organism per job is what the sweep fans out; only the "
        "shared-trunk `deepset` pools anything across the stack.",
    )

    rf = sub.add_parser("baseline-rf", help="Random-forest baseline on the same split and gate.")
    rf.add_argument("--labels", type=Path, required=True, help="Label shard root (§4.5).")
    rf.add_argument("--index", type=Path, required=True, help="Frozen metabolite_index.json.")
    rf.add_argument("--out", type=Path, required=True, help="Diagnostics dir.")
    rf.add_argument("--eps", type=float, default=1e-3)
    rf.add_argument("--n-estimators", type=int, default=100)
    rf.add_argument(
        "--delta",
        type=float,
        default=0.05,
        help="Finite-difference step, in units of each metabolite's kink scale.",
    )
    rf.add_argument("--seed", type=int, default=0)
    rf.add_argument(
        "--organisms",
        help="Comma-separated genome_ids to stack (default: every shard under "
        "--labels). One organism per job is what the sweep fans out; only the "
        "shared-trunk `deepset` pools anything across the stack.",
    )
    mj = sub.add_parser(
        "master-jacobian",
        help="V-for-§8: spectrum of the master Jacobian sum_i X_i H_i at real media.",
    )
    mj.add_argument("--labels", type=Path, required=True, help="Label shard root (§4.5).")
    mj.add_argument("--index", type=Path, required=True, help="Frozen metabolite_index.json.")
    mj.add_argument("--out", type=Path, required=True, help="Report dir.")
    mj.add_argument(
        "--checkpoint",
        type=Path,
        help="A `train-value` output dir. Omitted: seed a width-1 `groupmax-u` from "
        "the label tangents at each --gm-temp, which needs no training.",
    )
    mj.add_argument("--eps", type=float, default=1e-3)
    mj.add_argument("--gm-temp", default="0.01,0.03,0.1,0.3,1.0", help="Temperatures to scan.")
    mj.add_argument("--gm-group", type=int, default=250, help="Affine pieces K when seeding.")
    mj.add_argument("--n-media", type=int, default=6, help="Held-out media to evaluate at.")
    mj.add_argument("--seed", type=int, default=0)
    mj.add_argument("--organisms", help="Comma-separated genome_ids (default: every shard).")

    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")
    # cobra logs one INFO line per solve ('Compartment `C_e` sounds like an
    # external compartment'); at 20k media x 21 organisms that is ~60 MB of
    # stderr per task and nothing else.
    logging.getLogger("cobra").setLevel(logging.WARNING)
    args = build_parser().parse_args(argv)
    organisms = (
        [g for g in getattr(args, "organisms", None).split(",") if g]
        if getattr(args, "organisms", None)
        else None
    )

    if args.command == "master-jacobian":
        from cfs.validate.master_jacobian import run

        run(
            args.labels,
            args.index,
            args.out,
            checkpoint=args.checkpoint,
            eps=args.eps,
            gm_temp=[float(v) for v in args.gm_temp.split(",") if v],
            gm_group=args.gm_group,
            n_media=args.n_media,
            organisms=organisms,
            seed=args.seed,
        )
        return 0

    if args.command == "train-value":
        # The only subcommand that reads labels rather than models: no roster, no
        # solver stack, and the jax extra instead of the data one.
        from cfs.surrogate.train import run

        diagnostics = run(
            args.labels,
            args.index,
            args.out,
            eps=args.eps,
            arch=args.arch,
            width=args.width,
            depth=args.depth,
            epochs=args.epochs,
            batch=args.batch,
            lr=args.lr,
            w_grad=args.w_grad,
            emb_dim=args.emb_dim,
            phi_hidden=args.phi_hidden,
            gm_group=args.gm_group,
            gm_temp=args.gm_temp,
            gm_init=args.gm_init,
            gm_reanchor=args.gm_reanchor,
            k_code=args.k_code,
            seed=args.seed,
            organisms=organisms,
        )
        print(json.dumps(diagnostics, indent=2))
        return 0 if diagnostics["passed"] else 1

    if args.command == "baseline-rf":
        # A measurement, not a gate: it always exits 0, however it scores.
        from cfs.surrogate.baseline import run as run_rf

        print(
            json.dumps(
                run_rf(
                    args.labels,
                    args.index,
                    args.out,
                    eps=args.eps,
                    n_estimators=args.n_estimators,
                    delta=args.delta,
                    seed=args.seed,
                    organisms=organisms,
                ),
                indent=2,
            )
        )
        return 0

    if args.command == "topup":
        # Also label-side, no roster: last run's held-out error -> next run's focus
        # budget (§4.6). Metabolites that never limited have no error to read, so
        # they get the floor share rather than nothing — the probe (§4.7) has by
        # now given them a band worth sampling.
        from cfs.sampling.active_subspace import load_subspaces
        from cfs.sampling.design import topup_weights
        from cfs.surrogate.ensemble import unmeasured_metabolites

        diagnostics = json.loads(args.diagnostics.read_text())
        subspaces = {}
        for path in sorted(Path(args.labels).glob("*.subspace.json")):
            subspaces.update({gid: s.active for gid, s in load_subspaces(path).items()})
        unmeasured = unmeasured_metabolites(diagnostics, subspaces)

        weights = {}
        for gid, d in diagnostics["per_organism"].items():
            w = topup_weights(d.get("per_limiting_metabolite", {}), floor=args.floor)
            blind = unmeasured.get(gid, [])
            if blind:
                # Give the never-measured metabolites the same per-metabolite share
                # the floor gives a well-predicted one, then renormalise.
                share = args.floor / max(len(w) + len(blind), 1)
                w = {**w, **dict.fromkeys(blind, share)}
                total = sum(w.values())
                w = {ex: v / total for ex, v in w.items()}
            weights[gid] = w
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(weights, indent=2, sort_keys=True))
        print(
            f"wrote {args.out} — {len(weights)} organisms, "
            f"{sum(len(v) for v in unmeasured.values())} never-measured metabolites"
        )
        return 0

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
        print(
            f"wrote {args.out} — {sum(len(s.active) for s in subspaces)} active "
            f"metabolites across {len(subspaces)} organisms"
        )
        return 0

    if args.command == "generate":
        from dataclasses import replace

        from cfs.sampling.design import SamplingConfig
        from cfs.sampling.generate import generate_roster

        cfg = SamplingConfig(seed=args.seed, probe=not args.no_probe)
        if args.n_media is not None:
            cfg = replace(cfg, n_media=args.n_media)
        scales = json.loads(args.scales.read_text()) if args.scales else None
        focus = json.loads(args.focus_weights.read_text()) if args.focus_weights else None
        shards = generate_roster(
            roster,
            args.index,
            args.outdir,
            cfg,
            scales=scales,
            focus_weights=focus,
            round_idx=args.round_idx,
        )
        print(
            json.dumps(
                {
                    s.genome_id: {"n_media": s.n_media, "shards": [str(p) for p in s.paths]}
                    for s in shards
                },
                indent=2,
            )
        )
        return 0

    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
