#!/usr/bin/env bash
# One-time setup on the cluster, before run_hpc.sh: download the two gapseq
# reference databases. It needs the SIF to already be in place, because the
# download runs `gapseq update-sequences` inside it.
#
#   PIPE=/path/to/genome-annotation REF=/shared/gapseq_ref bash setup_hpc.sh
#
# The SIF cannot be built here from the Dockerfile -- singularity/apptainer
# cannot read a Dockerfile, and codon has no docker daemon. Build it once where
# docker is available and copy it in:
#
#   docker build -t gapseq:2.0.1 $PIPE/docker/gapseq      # a few GB, needs disk
#   docker save gapseq:2.0.1 -o gapseq-v2.0.1.tar
#   scp gapseq-v2.0.1.tar codon:/tmp/
#   apptainer build $PIPE/containers/gapseq/gapseq-v2.0.1.sif \
#       docker-archive:///tmp/gapseq-v2.0.1.tar
#
# The path is not negotiable: GAPSEQ_DOALL's singularity container is
# `projectDir + '/containers/gapseq/gapseq-v2.0.1.sif'`, a path inside the
# checkout rather than a registry reference. The image has to be the repo's own
# build -- it patches gapseq's `prepare_batch_alignments.R` so the user
# sequences come from the external `-D` database, which is what makes the
# `-D ./gapseq_db_local` in the module work at all.
set -euo pipefail
: "${PIPE:?set PIPE to the genome-annotation checkout on the cluster}"
: "${REF:?set REF to where the gapseq reference databases should live}"
sif=$PIPE/containers/gapseq/gapseq-v2.0.1.sif

[ -f "$sif" ] || { sed -n '6,20p' "$0" >&2; echo "missing: $sif" >&2; exit 1; }
mkdir -p "$REF"

for taxon in Bacteria Archaea; do
    out=$REF/$(echo "$taxon" | tr '[:upper:]' '[:lower:]')
    if [ -d "$out/gapseq_db" ]; then
        echo "$taxon present: $out/gapseq_db"
    else
        echo "=== downloading the $taxon reference set"
        bash "$PIPE/scripts/download_gapseq_db.sh" --singularity --outdir "$out" -t "$taxon"
    fi
    printf '%-10s %s  %s\n' "$taxon" "$(du -sh "$out/gapseq_db" | cut -f1)" "$out/gapseq_db"
done

cat <<'MSG'

=== Done. Put the matching path in each hpc.<taxon>.config, and read the sizes
above as the maxForks budget: every running GAPSEQ_DOALL task copies the whole
database into its own work directory, so peak scratch is maxForks x that size.
MSG
