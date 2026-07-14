"""Command-line entry point: ``surrogate-mgem {generate,train,validate,search}``.

Subcommands are wired up as the corresponding phases land; the parser itself is
stable so ``--help`` and the entry point work from the first commit.
"""

from __future__ import annotations

import argparse

from surrogate_mgem import __version__

_COMMANDS = ("generate", "train", "validate", "search")


def build_parser() -> argparse.ArgumentParser:
    """Return the top-level argument parser with one subparser per command."""
    parser = argparse.ArgumentParser(prog="surrogate-mgem", description=__doc__)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in _COMMANDS:
        subparsers.add_parser(name, help=f"{name} (not yet implemented)")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse args and dispatch to the selected subcommand."""
    args = build_parser().parse_args(argv)
    raise SystemExit(f"'{args.command}' is not implemented yet.")


if __name__ == "__main__":
    main()
