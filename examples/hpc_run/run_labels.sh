#!/usr/bin/env bash
# Generate the §4.5 ground-truth labels at D10 scale (20000 media/organism).
# Extra args pass through, e.g. `./run_labels.sh -stub` for a solver-free wiring check.
#
#   ./make_roster.sh ~/Documents/20hm_carveme_models > roster.csv   # once
#   ./run_labels.sh                       # uses reference/metabolite_index.json
#   NF_PROFILE=singularity ./run_labels.sh -c site.config   # on HPC
#
# A smaller trial run first is cheap and worth it:
#   OUTDIR=trial ./run_labels.sh --label_media 200
set -euo pipefail
cd "$(dirname "$0")"

nextflow run ../../main.nf \
    -profile "${NF_PROFILE:-docker}" \
    -c labels.config \
    --stage labels \
    --roster "${ROSTER:-roster.csv}" \
    --index "${INDEX:-reference/metabolite_index.json}" \
    --outdir "${OUTDIR:-labels_out}" \
    -resume "$@"
