"""Command-line entry point: ``surrogate-mgem {generate,train,validate,search}``.

Subcommands are wired up as the corresponding phases land; the parser itself is
stable so ``--help`` and the entry point work from the first commit.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from surrogate_mgem import __version__


def build_parser() -> argparse.ArgumentParser:
    """Return the top-level argument parser with one subparser per command."""
    parser = argparse.ArgumentParser(prog="surrogate-mgem", description=__doc__)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    gen = subparsers.add_parser(
        "generate", help="Generate surrogate training data (needs the 'data' extra)."
    )
    gen.add_argument(
        "--roster", type=Path, required=True, help="CSV with genome_id, model_path columns."
    )
    gen.add_argument(
        "--out", type=Path, required=True, help="Output directory for the training tables."
    )
    gen.add_argument("--n-communities", type=int, default=50)
    gen.add_argument("--size-min", type=int, default=2)
    gen.add_argument("--size-max", type=int, default=6)
    gen.add_argument("--media-per-community", type=int, default=20)
    gen.add_argument("--max-uptake", type=float, default=1000.0)
    gen.add_argument("--tradeoff", type=float, default=0.35)
    gen.add_argument(
        "--sampler", choices=["perturb", "sparse", "dirichlet", "lhs"], default="perturb"
    )
    gen.add_argument(
        "--n-active", type=int, default=20, help="sparse sampler: active components per medium."
    )
    gen.add_argument("--solver", default="hybrid")
    gen.add_argument("--seed", type=int, default=0)
    gen.add_argument("--workers", type=int, default=1)
    gen.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Total HPC shards; this run solves communities where ci %% num_shards == shard_index.",
    )
    gen.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="This shard's index in [0, num_shards). Shard 0 also writes exchange_universe.json.",
    )

    tr = subparsers.add_parser("train", help="Train a fixed-community growth-surrogate ensemble.")
    tr.add_argument(
        "--data-dir", type=Path, required=True, help="Directory of tables from `generate`."
    )
    tr.add_argument(
        "--out", type=Path, required=True, help="Output directory for model + report inputs."
    )
    tr.add_argument(
        "--community-id", default=None, help="Community to train on (default: most-sampled)."
    )
    tr.add_argument("--n-models", type=int, default=5, help="Ensemble size.")
    tr.add_argument(
        "--hidden",
        default="256,256",
        help="MLP hidden layers as comma-separated widths (e.g. '512,512,512').",
    )
    tr.add_argument("--epochs", type=int, default=300)
    tr.add_argument("--test-size", type=float, default=0.2)
    tr.add_argument(
        "--n-train",
        type=int,
        default=None,
        help="Cap the training split to this many rows (learning-curve sweeps).",
    )
    tr.add_argument("--seed", type=int, default=0)
    # Active learning (off unless --active-rounds > 0).
    tr.add_argument(
        "--active-rounds",
        type=int,
        default=0,
        help="Rounds of active learning (0 = static). Needs --roster to build the solver oracle.",
    )
    tr.add_argument(
        "--roster", type=Path, default=None, help="Roster CSV (required for active learning)."
    )
    tr.add_argument("--batch-size", type=int, default=16, help="Active: real solves per round.")
    tr.add_argument(
        "--n-candidates", type=int, default=2000, help="Active: media proposed per round."
    )
    tr.add_argument(
        "--sampler",
        choices=["perturb", "sparse", "dirichlet", "lhs"],
        default="perturb",
        help="Active: candidate proposal.",
    )
    tr.add_argument(
        "--n-active", type=int, default=20, help="Active: sparse proposer components per medium."
    )
    tr.add_argument("--max-uptake", type=float, default=1000.0)
    tr.add_argument("--tradeoff", type=float, default=0.35)
    tr.add_argument("--solver", default="hybrid")

    ar = subparsers.add_parser(
        "active-round",
        help="Run one active-learning round for a community; write augmented tables.",
    )
    ar.add_argument("--data-dir", type=Path, required=True, help="Current dataset dir.")
    ar.add_argument("--community-id", required=True, help="Community to augment.")
    ar.add_argument("--roster", type=Path, required=True, help="Roster CSV (solver oracle).")
    ar.add_argument("--out", type=Path, required=True, help="Output augmented dataset dir.")
    ar.add_argument("--round", type=int, default=0, help="Round index (seeds proposals).")
    ar.add_argument("--batch-size", type=int, default=16, help="Real solves this round.")
    ar.add_argument("--n-candidates", type=int, default=2000, help="Media proposed this round.")
    ar.add_argument(
        "--sampler", choices=["perturb", "sparse", "dirichlet", "lhs"], default="perturb"
    )
    ar.add_argument("--n-active", type=int, default=20)
    ar.add_argument("--n-models", type=int, default=5, help="Acquisition ensemble size.")
    ar.add_argument("--hidden", default="256,256", help="Acquisition MLP hidden widths.")
    ar.add_argument("--epochs", type=int, default=300)
    ar.add_argument("--max-uptake", type=float, default=1000.0)
    ar.add_argument("--tradeoff", type=float, default=0.35)
    ar.add_argument("--solver", default="hybrid")
    ar.add_argument("--seed", type=int, default=0)

    rp = subparsers.add_parser(
        "report", help="Render the Quarto performance report for a train run."
    )
    rp.add_argument("--results-dir", type=Path, required=True, help="A `train` output directory.")
    rp.add_argument(
        "--template", type=Path, default=None, help="Override the report .qmd template."
    )

    subparsers.add_parser("validate", help="validate (not yet implemented)")
    subparsers.add_parser("search", help="search (not yet implemented)")
    return parser


def _parse_hidden(spec: str) -> tuple[int, ...]:
    """Parse a ``"256,256"`` hidden-layer spec into a tuple of widths."""
    return tuple(int(w) for w in spec.split(",") if w.strip())


def _run_generate(args: argparse.Namespace) -> int:
    """Dispatch the ``generate`` subcommand (imports the micom-backed module here)."""
    from surrogate_mgem.data import GenerateConfig, generate, read_roster

    config = GenerateConfig(
        out_dir=args.out,
        n_communities=args.n_communities,
        size_range=(args.size_min, args.size_max),
        media_per_community=args.media_per_community,
        max_uptake=args.max_uptake,
        tradeoff=args.tradeoff,
        sampler=args.sampler,
        n_active=args.n_active,
        solver=args.solver,
        seed=args.seed,
        workers=args.workers,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    )
    generate(read_roster(args.roster), config)
    return 0


def _run_train(args: argparse.Namespace) -> int:
    """Dispatch the ``train`` subcommand (static ensemble or active learning)."""
    from surrogate_mgem.train import (
        load_fixed_community_dataset,
        train_fixed_community,
        train_fixed_community_active,
    )

    dataset = load_fixed_community_dataset(args.data_dir, args.community_id)
    if args.active_rounds <= 0:
        train_fixed_community(
            dataset,
            args.out,
            n_models=args.n_models,
            hidden=_parse_hidden(args.hidden),
            epochs=args.epochs,
            test_size=args.test_size,
            n_train=args.n_train,
            seed=args.seed,
        )
        return 0

    if args.roster is None:
        raise SystemExit("--roster is required when --active-rounds > 0.")
    from surrogate_mgem.active import ActiveConfig
    from surrogate_mgem.data import members_for_community, read_roster

    members = members_for_community(read_roster(args.roster), dataset.community_id)
    config = ActiveConfig(
        rounds=args.active_rounds,
        batch_size=args.batch_size,
        n_candidates=args.n_candidates,
        max_uptake=args.max_uptake,
        sampler=args.sampler,
        n_active=args.n_active,
        n_models=args.n_models,
        epochs=args.epochs,
        seed=args.seed,
    )
    train_fixed_community_active(
        dataset,
        members,
        args.out,
        active_config=config,
        solver=args.solver,
        tradeoff=args.tradeoff,
        test_size=args.test_size,
        seed=args.seed,
    )
    return 0


def _run_active_round(args: argparse.Namespace) -> int:
    """Dispatch the ``active-round`` subcommand (one round, augment the tables)."""
    from surrogate_mgem.active import ActiveConfig
    from surrogate_mgem.train import run_active_round

    config = ActiveConfig(
        batch_size=args.batch_size,
        n_candidates=args.n_candidates,
        max_uptake=args.max_uptake,
        sampler=args.sampler,
        n_active=args.n_active,
        n_models=args.n_models,
        hidden=_parse_hidden(args.hidden),
        epochs=args.epochs,
        seed=args.seed,
    )
    run_active_round(
        args.data_dir,
        args.community_id,
        args.roster,
        args.out,
        active_config=config,
        solver=args.solver,
        tradeoff=args.tradeoff,
        round_index=args.round,
    )
    return 0


def _run_report(args: argparse.Namespace) -> int:
    """Render the Quarto performance report into the results directory."""
    import os
    import shutil
    import subprocess
    import sys

    results = args.results_dir.resolve()
    template = args.template or (
        Path(__file__).resolve().parents[2] / "reports" / "model_report.qmd"
    )
    shutil.copyfile(template, results / "model_report.qmd")
    env = {**os.environ, "QUARTO_PYTHON": sys.executable, "SURROGATE_RESULTS_DIR": str(results)}
    subprocess.run(
        ["quarto", "render", "model_report.qmd", "--to", "html"],
        cwd=results,
        env=env,
        check=True,
    )
    print(f"Wrote {results / 'model_report.html'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Parse args and dispatch to the selected subcommand."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    args = build_parser().parse_args(argv)
    if args.command == "generate":
        return _run_generate(args)
    if args.command == "train":
        return _run_train(args)
    if args.command == "active-round":
        return _run_active_round(args)
    if args.command == "report":
        return _run_report(args)
    raise SystemExit(f"'{args.command}' is not implemented yet.")


if __name__ == "__main__":
    raise SystemExit(main())
