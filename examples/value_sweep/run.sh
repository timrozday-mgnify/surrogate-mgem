#!/usr/bin/env bash
# Run the M3b Head A sweep (--stage sweep). Extra args pass through, e.g.
# `./run.sh -stub` for a solver-free, container-free wiring check.
#
#   SWEEP=sweep_full.csv INDEX=/path/to/qc/metabolite_index.json ./run.sh
#   NF_PROFILE=singularity ./run.sh -c site.config      # on HPC
set -euo pipefail
cd "$(dirname "$0")"

nextflow run ../../main.nf \
    -profile "${NF_PROFILE:-docker}" \
    -c params.config \
    --stage sweep \
    --sweep "${SWEEP:-sweep.csv}" \
    --index "${INDEX:-index_stub.json}" \
    --outdir results \
    -resume "$@"
