# Gap-filled (GapSeq) GEMs for the 20HM v2 panel — codon

The same 24 genomes as the CarveMe panel (`20hm_v2_models` in the runs
directory), reconstructed with GapSeq instead. Set up for the cluster; nothing
has been run.

This directory is the thing to copy to the cluster. `make_samplesheets.sh` and
the filled-in `hpc.<taxon>.config` are per-cluster, so they are gitignored --
everything needed to regenerate them is here.

Laptop was the wrong place for this: GapSeq is one to several hours per genome
against CarveMe's few minutes, and `GAPSEQ_DOALL` copies the whole reference
database into *every running task's* work directory, so concurrency multiplies
the database on disk. The laptop has 13 GiB free.

## Two runs, not one

`gapseq doall` takes its taxonomy from `-t`, and the module passes one
`task.ext.args` to every task — so a single run would gap-fill
*Methanobrevibacter smithii* against **Bacteria** reference sequences. It is
in the core 21 roster, so that would quietly corrupt a twenty-first of the
surrogate's ground truth.

`make_samplesheets.sh` splits the panel 23 + 1 off the `universe` column the
CarveMe samplesheet already carries. The taxon and the reference database live
in the same `hpc.<taxon>.config`, because they have to agree and a run against
the wrong reference set produces output that looks fine.

The alternative is a `taxon` samplesheet column — the GapSeq analogue of PR #2's
`universe` — which would make this one run. Worth doing if this is ever repeated.

## Order of operations

```bash
# 0. once, where docker exists (a few GB of disk):
docker build -t gapseq:2.0.1 <checkout>/docker/gapseq
docker save gapseq:2.0.1 -o gapseq-v2.0.1.tar
scp gapseq-v2.0.1.tar codon:/tmp/
# on codon:
apptainer build <checkout>/containers/gapseq/gapseq-v2.0.1.sif \
    docker-archive:///tmp/gapseq-v2.0.1.tar

# 1. the reference databases (prints their sizes -- that is the maxForks budget)
PIPE=<checkout> REF=<shared>/gapseq_ref bash setup_hpc.sh

# 2. samplesheets, with the panel's path on the cluster
GENOME_DIR=<shared>/20hm-genomes bash make_samplesheets.sh

# 3. configs
cp hpc.bacteria.example.config hpc.bacteria.config   # and edit
cp hpc.archaea.example.config  hpc.archaea.config    # and edit

# 4. run
TAXON=bacteria PIPE=<checkout> sbatch run_hpc.sh
TAXON=archaea  PIPE=<checkout> sbatch run_hpc.sh

# 5. gather + QC, exactly as the CarveMe panel
MODELS=~/20hm_v2_gapseq_models bash collect.sh
```

`run_hpc.sh` refuses to start on a missing samplesheet, a missing config or one
with placeholders left in, so a half-configured run fails in a second.

## Why the SIF cannot be built on the cluster

`GAPSEQ_DOALL`'s singularity container is `projectDir +
'/containers/gapseq/gapseq-v2.0.1.sif'` — a path inside the checkout, not a
registry reference — and apptainer cannot read a Dockerfile. It has to be the
repo's own build, too: the Dockerfile patches gapseq's
`prepare_batch_alignments.R` so user sequences resolve from the external `-D`
database, which is what makes the module's `-D ./gapseq_db_local` work.

## Two things to verify before the real run

- **`-t Archaea` on `doall`.** Confirm against `gapseq doall -h` inside the
  image. It is gapseq's taxonomy flag for the find step; if `doall` spells it
  differently, `hpc.archaea.config`'s `ext.args` is the one line to change.
- **`params.input` vs `params.samplesheet`.** `params.*.yml` here says `input:`,
  which is the `nf-core-compliance` branch's rename. Against today's
  `carveme-universe` or `master` these need `samplesheet:`. Mismatch fails
  fast rather than annotating nothing.

## After the models exist

A different model set has a different exchange set, so the gap-filled panel
needs **its own** `cfs freeze-index` output — the CarveMe panel's
`metabolite_index.json` is a different model and its hash is stamped into every
label shard (P13). `collect.sh` prints the two commands.
