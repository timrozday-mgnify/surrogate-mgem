#!/usr/bin/env bash
#SBATCH --job-name=hpc-labels
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=7-00:00:00
#SBATCH --output=hpc-labels-%j.log
# Nextflow head job (README.md, "Run"):
#   sbatch workflows/hpc-labels/run_hpc.sh
#   PROFILE=slurm,test sbatch workflows/hpc-labels/run_hpc.sh --outdir <dir> \
#       --organisms A,B                            # smoke test, minutes
#
# `--outdir` on the command line for the test profile: the repo-root nextflow.config
# sets `outdir` and overrides what a profile sets.
#
# Run from the repo root, after workflows/pair-interactions/setup.sh, with hpc.config
# next to this script.
set -euo pipefail
here=workflows/hpc-labels
# module load nextflow   # or whatever your site provides
export NXF_OPTS='-Xms1g -Xmx3g'
nextflow run "$here" -profile "${PROFILE:-slurm}" -c "$here/hpc.config" -resume "$@"
