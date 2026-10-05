#!/usr/bin/env bash
#SBATCH --job-name=pair-interactions
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=2-00:00:00
#SBATCH --output=pair-interactions-%j.log
# Nextflow head job (README.md, "HPC"):
#   sbatch workflows/pair-interactions/run_hpc.sh [extra nextflow options, e.g. --seeds 0]
# Run from the repo root, after workflows/pair-interactions/setup.sh, with hpc.config
# next to this script.
set -euo pipefail
here=workflows/pair-interactions
# module load nextflow   # or whatever your site provides
export NXF_OPTS='-Xms1g -Xmx3g'
nextflow run "$here" -profile slurm -c "$here/hpc.config" -resume "$@"
