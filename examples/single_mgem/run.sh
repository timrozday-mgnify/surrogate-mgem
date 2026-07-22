#!/usr/bin/env bash
# Run the surrogate-mgem sweep on the single-mGEM example. Extra args pass through,
# e.g. `./run.sh -stub` for a solver-free wiring check.
# Override the container engine with NF_PROFILE=singularity ./run.sh
set -euo pipefail
cd "$(dirname "$0")"

nextflow run ../../main.nf \
    -profile "${NF_PROFILE:-docker}" \
    -c params.config \
    --roster roster.csv \
    --outdir results \
    -resume "$@"
