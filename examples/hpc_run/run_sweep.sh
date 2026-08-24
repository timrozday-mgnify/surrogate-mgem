#!/usr/bin/env bash
# Run the M3b Head A sweep (--stage sweep). Extra args pass through, e.g.
# `./run_sweep.sh -stub` for a solver-free, container-free wiring check.
#
#   SWEEP=sweep_full.csv ./run_sweep.sh
#   NF_PROFILE=singularity ./run_sweep.sh -c site.config      # on HPC
set -euo pipefail
cd "$(dirname "$0")"

nextflow run ../../main.nf \
    -profile "${NF_PROFILE:-docker}" \
    -c sweep.config \
    --stage sweep \
    --sweep "${SWEEP:-sweep_smoke.csv}" \
    --index "${INDEX:-reference/metabolite_index.json}" \
    --outdir "${OUTDIR:-sweep_out}" \
    -resume "$@"
