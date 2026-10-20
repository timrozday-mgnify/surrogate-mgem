#!/usr/bin/env bash
# Build the input bundle the pipeline reads (--data), from a local CarveMe run.
#   bash workflows/hpc-labels/stage_inputs.sh <models dir> <index.json> <roster.csv> <bundle>
# then copy it to the cluster, e.g. rsync -a <bundle>/ hpc:<shared storage>/labels-data/
#
# The index must be the one `cfs freeze-index` froze on THIS roster: a new GEM set has
# a different exchange set, so an old index is a different model (P13). Only the
# roster's own genomes are copied, and model_path is rewritten to the bundle -- the
# pipeline re-points it at its own staged copy anyway, but a bundle that is wrong on
# its own is a trap.
set -euo pipefail
models=${1:?dir of CarveMe *.xml}
index=${2:?frozen metabolite_index.json}
roster=${3:?roster.csv (genome_id,model_path)}
out=${4:?bundle dir}

mkdir -p "$out/gems"
cp "$index" "$out/metabolite_index.json"
{
    echo genome_id,model_path
    tail -n +2 "$roster" | cut -d, -f1 | while read -r gid; do
        cp "$models/$gid.xml" "$out/gems/"
        printf '%s,gems/%s.xml\n' "$gid" "$gid"
    done
} > "$out/roster.csv"
echo "$(($(wc -l < "$out/roster.csv") - 1)) genomes"
du -sh "$out"
