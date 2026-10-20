#!/usr/bin/env bash
# After both runs: gather the gap-filled models and build the roster, to the same
# shape as ../20hm_v2_models (the CarveMe panel).
#
#   MODELS=~/20hm_v2_gapseq_models bash collect.sh
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
models=${MODELS:-$HOME/20hm_v2_gapseq_models}
cd "$here"

mkdir -p "$models"
n=0
for d in results-bacteria results-archaea; do
    [ -d "$d" ] || { echo "missing $d -- has that taxon's run finished?" >&2; continue; }
    for f in "$d"/*/gapseq/*.xml; do
        [ -e "$f" ] || continue
        cp "$f" "$models"/ && n=$((n + 1))
    done
done
[ "$n" -gt 0 ] || { echo "no gapseq .xml found under results-*/*/gapseq/" >&2; exit 1; }

"$here/../../surrogate-mgem_runs/hpc_run/make_roster.sh" "$models" > roster_all.csv
{ head -1 roster_all.csv
  grep -F -f <(awk -F'\t' '$2=="core"{print $1","}' roles.tsv) roster_all.csv
} > roster.csv
echo "models: $n  core roster: $(($(wc -l < roster.csv) - 1)) (expect 24 and 21)"
echo
echo "Then, as for the CarveMe panel -- a different model set is a different"
echo "exchange set, so it needs its own frozen index (P13):"
echo "  cfs qc --roster roster.csv --outdir qc --no-memote"
echo "  cfs freeze-index --roster roster.csv --out qc/metabolite_index.json"
