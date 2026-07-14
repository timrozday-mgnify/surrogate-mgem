"""Scaffold smoke tests: the package imports and the CLI parser is wired up."""

import pytest

import surrogate_mgem
from surrogate_mgem.cli import build_parser


def test_version_is_a_string():
    assert isinstance(surrogate_mgem.__version__, str)


def test_parser_exposes_all_commands():
    parser = build_parser()
    # generate has required args; the rest parse bare.
    assert parser.parse_args(["generate", "--roster", "r.csv", "--out", "o"]).command == "generate"
    for command in ("train", "validate", "search"):
        assert parser.parse_args([command]).command == command
    # A missing subcommand is a usage error (exit 2).
    with pytest.raises(SystemExit):
        parser.parse_args([])
