#!/usr/bin/env bash
# Two samplesheets from the panel manifest, because `gapseq doall` takes its taxon
# from `-t` and the module has one `ext.args` for every task -- so a single run
# would gap-fill the archaeon against Bacteria reference sequences. 23 + 1.
#
#   GENOME_DIR=/hps/.../20hm-genomes bash make_samplesheets.sh
#
# GENOME_DIR is where the panel FASTAs live ON THE CLUSTER.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
: "${GENOME_DIR:?set GENOME_DIR to the panel FASTA directory on the cluster}"
src=$here/samplesheet.csv   # the laptop samplesheet: sample,fastx,universe

ext=$(awk -F, 'NR==2{n=split($2,a,"/"); print a[n]}' "$src" | sed 's/^[^.]*//')
for taxon in bacteria archaea; do
    out=$here/samplesheet.$taxon.csv
    { echo sample,fastx
      awk -F, -v d="$GENOME_DIR" -v e="$ext" -v t="$taxon" '
        NR == 1 { next }
        { u = ($3 == "" ? "bacteria" : $3) }
        u == t  { printf "%s,%s/%s%s\n", $1, d, $1, e }' "$src"
    } > "$out"
    echo "$out: $(($(wc -l < "$out") - 1)) genomes"
done
