#!/usr/bin/env bash
# Create the repo's Python environment for the pipeline: the solver stack (cobra,
# GLPK, Clarabel) and the surrogate stack (jax/equinox) together, which
# `cfs interactions` needs and neither published image carries.
# Needs uv (https://docs.astral.sh/uv/); it fetches Python 3.11 itself.
set -euo pipefail
cd "$(dirname "$0")/../.."
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e ".[data,jax]"
.venv/bin/cfs interactions --help > /dev/null && echo "cfs ok"
