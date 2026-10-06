#!/usr/bin/env bash
# Build the input bundle the pipeline reads (--data), from a local run root.
#   bash workflows/pair-interactions/stage_data.sh <run_root> <bundle> [genome_id ...]
# then copy it to the cluster, e.g. rsync -a <bundle>/ hpc:<shared storage>/pair-data/
#
# List every member of the community, not just a pair -- the pipeline derives the pairs
# from `--members` and the survey needs all of their subspaces and GEMs. Only the listed
# genomes' label shards and GEMs are copied; the checkpoints hold all 21 organisms and
# load a subset. ~0.27 GB of checkpoints plus ~65 MB a genome, so ~0.4 GB for a pair and
# ~1.6 GB for a 21-member roster.
set -euo pipefail
src=${1:?run root, e.g. ~/Documents/surrogate-mgems_runs/20hm_bands}
out=${2:?bundle dir}
shift 2
gids=("$@")
[ ${#gids[@]} -eq 0 ] && gids=(CP040530.1 CP070062.1)
gem_dir=${GEM_DIR:-$HOME/Documents/20hm_carveme_models}

mkdir -p "$out/gems"
for ck in value_p4r2 behaviour_p4r2 value_i3_cap behaviour_i3; do
    rsync -a "$src/$ck/" "$out/$ck/"
done
for root in labels_p4 labels_i3; do
    mkdir -p "$out/$root"
    for g in "${gids[@]}"; do
        rsync -a "$src/$root/$g" "$src/$root/$g".*.json "$out/$root/"
    done
done
for g in "${gids[@]}"; do cp "$gem_dir/$g.xml" "$out/gems/"; done
du -sh "$out"
