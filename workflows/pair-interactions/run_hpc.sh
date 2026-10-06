#!/usr/bin/env bash
#SBATCH --job-name=pair-interactions
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=2-00:00:00
#SBATCH --output=pair-interactions-%j.log
# Nextflow head job (README.md, "Run"):
#   sbatch workflows/pair-interactions/run_hpc.sh [extra nextflow options]
#   ... --members A,B,C,D --max_pairs 6          # a community: all pairs surveyed, 6 searched
#   ... --max_pairs 12                           # extend the searches; -resume adds only the new
#   PROFILE=slurm,test sbatch ... --outdir <dir> # smoke test (sbatch exports PROFILE)
#
# `--outdir` on the command line, always, for the test profiles: the repo-root
# nextflow.config sets `outdir` and overrides what a profile sets.
#
# Run from the repo root, after workflows/pair-interactions/setup.sh, with hpc.config
# next to this script.
set -euo pipefail
here=workflows/pair-interactions
# module load nextflow   # or whatever your site provides
export NXF_OPTS='-Xms1g -Xmx3g'
nextflow run "$here" -profile "${PROFILE:-slurm}" -c "$here/hpc.config" -resume "$@"
