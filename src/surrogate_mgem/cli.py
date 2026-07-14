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
    gen.add_argument("--sampler", choices=["lhs", "dirichlet"], default="lhs")
    gen.add_argument("--solver", default="hybrid")
    gen.add_argument("--seed", type=int, default=0)
    gen.add_argument("--workers", type=int, default=1)

    tr = subparsers.add_parser("train", help="Train a fixed-community growth surrogate.")
    tr.add_argument(
        "--data-dir", type=Path, required=True, help="Directory of tables from `generate`."
    )
    tr.add_argument("--out", type=Path, required=True, help="Output directory for model + metrics.")
    tr.add_argument(
        "--community-id", default=None, help="Community to train on (default: most-sampled)."
    )
    tr.add_argument("--epochs", type=int, default=300)
    tr.add_argument("--lr", type=float, default=1e-3)
    tr.add_argument("--test-size", type=float, default=0.2)
    tr.add_argument("--seed", type=int, default=0)

    for name in ("validate", "search"):
        subparsers.add_parser(name, help=f"{name} (not yet implemented)")
    return parser


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
        solver=args.solver,
        seed=args.seed,
        workers=args.workers,
    )
    generate(read_roster(args.roster), config)
    return 0


def _run_train(args: argparse.Namespace) -> int:
    """Dispatch the ``train`` subcommand."""
    from surrogate_mgem.train import load_fixed_community_dataset, train_fixed_community

    dataset = load_fixed_community_dataset(args.data_dir, args.community_id)
    train_fixed_community(
        dataset, args.out, epochs=args.epochs, lr=args.lr, test_size=args.test_size, seed=args.seed
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """Parse args and dispatch to the selected subcommand."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    args = build_parser().parse_args(argv)
    if args.command == "generate":
        return _run_generate(args)
    if args.command == "train":
        return _run_train(args)
    raise SystemExit(f"'{args.command}' is not implemented yet.")


if __name__ == "__main__":
    raise SystemExit(main())
