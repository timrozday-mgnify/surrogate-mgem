#!/usr/bin/env bash
#SBATCH --job-name=gapseq-20hm
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=3-00:00:00
#SBATCH --output=gapseq-20hm-%j.log
# Gap-filled GEMs for the 20HM v2 panel, one taxon per invocation:
#
#   TAXON=bacteria sbatch run_hpc.sh      # 23 genomes
#   TAXON=archaea  sbatch run_hpc.sh      # M. smithii, with -t Archaea
#
# Run setup_hpc.sh first, make the samplesheets with
# `GENOME_DIR=<panel dir> bash make_samplesheets.sh`, and copy
# hpc.<taxon>.example.config to hpc.<taxon>.config.
# Resume-safe: re-run after any failure.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
pipe=${PIPE:-$HOME/genome-annotation}
taxon=${TAXON:?set TAXON to bacteria or archaea}
cd "$here"

[ -f samplesheet.$taxon.csv ] || { echo "run make_samplesheets.sh first" >&2; exit 1; }
cfg=hpc.$taxon.config
[ -f "$cfg" ] || { echo "copy hpc.$taxon.example.config to $cfg and fill it in" >&2; exit 1; }
grep -q '<shared storage>' "$cfg" && { echo "$cfg still has placeholders" >&2; exit 1; }

# module load nextflow singularity   # or whatever your site provides
export NXF_OPTS='-Xms1g -Xmx3g'

# The taxon rides in hpc.$taxon.config as GAPSEQ_DOALL's ext.args, beside the
# reference database it has to match. `ext.args` is one value for every task,
# which is the whole reason this is two runs rather than one.
nextflow run "$pipe/main.nf" -profile codon -c "$cfg" -resume \
    -params-file "params.$taxon.yml" "$@"
