#!/usr/bin/env bash
# Run the M3b Head A sweep (--stage sweep). Extra args pass through, e.g.
# `./run_sweep.sh -stub` for a solver-free, container-free wiring check.
#
#   ./run_sweep.sh                                        # sweep_full.csv
#   SWEEP=sweep_smoke.csv ./run_sweep.sh -stub            # wiring check
#   NF_PROFILE=singularity ./run_sweep.sh -c site.config  # on HPC
set -euo pipefail
cd "$(dirname "$0")"

# sweep_smoke.csv points at the empty labels_stub/, so it only means anything
# under -stub; defaulting to it for a real run just fails 3 tasks on "no
# per-genome shard dirs".
default_sweep=sweep_full.csv
for a in "$@"; do [[ $a == -stub* ]] && default_sweep=sweep_smoke.csv; done

nextflow run ../../main.nf \
    -profile "${NF_PROFILE:-docker}" \
    -c sweep.config \
    --stage sweep \
    --sweep "${SWEEP:-$default_sweep}" \
    --index "${INDEX:-reference/metabolite_index.json}" \
    --outdir "${OUTDIR:-sweep_out}" \
    -resume "$@"
