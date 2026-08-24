#!/usr/bin/env bash
# Write roster.csv from a directory of CarveMe GEMs: genome_id = the filename stem.
#
#   ./make_roster.sh ~/Documents/20hm_carveme_models > roster.csv
#
# model_path is absolute here on purpose. The pipeline also accepts paths relative
# to the roster file, but on a cluster the models usually live on a different
# filesystem from the run dir.
set -euo pipefail
dir="${1:?usage: make_roster.sh <dir of *.xml>}"

echo "genome_id,model_path"
for f in "$dir"/*.xml "$dir"/*.xml.gz; do
    [ -e "$f" ] || continue
    id=$(basename "$f"); id=${id%.gz}; id=${id%.xml}
    printf '%s,%s\n' "$id" "$(cd "$(dirname "$f")" && pwd)/$(basename "$f")"
done
