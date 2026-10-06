# pair-interactions

The first application of the surrogate to a use case: which metabolic interactions
one pair of genomes can have, and in which media (design spec §13.5/M13, §13.11/M16).

```text
INTERACTIONS  one `cfs interactions` search per (arm, objective, seed):
              arm       = the model: plain FBA, or FBA + product inhibition at c^eq
              objective = handover (cross-feeding), interference (competition +
                          inhibition), conditioning (inhibition alone; inhibited arms)
              -> runs/<arm>__<objective>__s<seed>/{interactions.json, media.npz, run.json}
SURVEY        the true-LP survey: media from the same prior as the searches,
              independent of their seeds, one FBA + QP per member, sharded
              (--survey_shards x --survey_media, default 8 x 500 per arm, ~2 min
              a shard). The report turns it into per-handover frequencies with
              95% intervals, an accumulation curve and a Chao1 estimate of unseen
              handovers. Shard k is always the same media, so `-resume` with more
              shards only adds media
              -> survey/<arm>/shard_<k>.npz
PRUNE         per run: the best design reduced, with the true LP on the run's own
              objective, to the moves that keep it within 1% (cheapest reverted
              first), and the mechanism there -- handovers >= 1% of E and each
              member's limiting exchanges. Designs are grouped by that signature
              across seeds. ~1-15 min a run
              -> prune/<run>.json
CROSSEVAL     every start and designed medium re-solved with the true LP under every
              model, the spent-medium assay at each best design, and per handover
              at each best design: FVA (forced vs tie-break), the other
              elastic-net eps, and abundance ratios 1:10..10:1
              -> crosseval/{media,links,spent,fluxes,robust}.csv
```

Each search proposes media with the surrogate and accepts them with the true LP, so
every reported interaction is LP-verified. Measured on a laptop (2 cores per task,
5 at once): handover 16-22 min, interference 4-7 min, conditioning <1 min (it screens
constructed media without an ascent, so its seeds usually pick the same constructed
start). Seeds draw independent random media; the candidate media target the same
metabolites in every seed (from the labels) on an independently drawn background.
Before 2026-10-06 seed s reused seed 0's draws shifted by s, so the seeds of earlier
runs are not replicates. Three seeds' 15 searches + cross-evaluation took 47 min
wall; the default is now five seeds (25 searches).

## Plan

1. **One pair, both models, five seeds** (this pipeline, defaults). The pair is
   `CP040530.1 + CP070062.1`: earlier work found a reciprocal glycerol /
   glyceraldehyde exchange and contended disposal routes for both members, so it
   has both positive and negative interactions to find. Read the report for:
   which mechanisms the seeds agree on (the pruned designs), how often each handover
   happens over the survey and whether it has converged, whether
   they survive inhibition at a fixed medium, and who suppresses whom.
2. **Check the instruments before the biology.** In the report's Diagnostics:
   surrogate vs LP link rates, the rank correlations, and the seed spread. A link
   found in one seed only, or a designed medium whose rate the LP does not confirm,
   is a lead, not a result.
3. **Firm up what reproduces**: more seeds (`--seeds 0,1,2,3,4,5,6,7`) where no
   mechanism has a majority, and more survey shards (`--survey_shards 16 -resume`)
   where the accumulation curve is still rising; then a `c^eq` sweep (add arms with
   `ceq: 0.1` / `10`) — only 1.0 mM has its own inhibited heads, so other values put
   inhibition in the true LP only (the stage 3' design).
4. **Other pairs**: stage their genomes (`stage_data.sh <src> <out> A B`) and run
   with `--pair A,B`. Rank pairs by the consensus link count before spending on any.
5. **What not to read off it**: absolute rates and times (`Vmax` is ~30x
   physiological, so only orderings and ratios carry over), and anything about the
   assembled community's abundances — `E` is a capacity of the medium at equal
   biomass.

## Run

Needs Java 17+, Nextflow and the project's venv (`bash workflows/pair-interactions/setup.sh`,
which needs `uv`); everything runs on the host in that venv, as in
kmer-functional-profiler, so `-profile singularity` is not needed.

```bash
# 1. on your laptop: build the input bundle (~0.35 GB for a pair) and copy it over
bash workflows/pair-interactions/stage_data.sh ~/Documents/surrogate-mgems_runs/20hm_bands pair-data
rsync -a pair-data/ <hpc>:<shared storage>/pair-data/
# 2. on the cluster, from the repo root (git pull first if the checkout exists)
bash workflows/pair-interactions/setup.sh
cp workflows/pair-interactions/hpc.example.config workflows/pair-interactions/hpc.config  # fill in <...>
# 3. smoke test on the cluster: the test profile through Slurm, ~5 min with queueing
#    (--outdir on the command line, because hpc.config's outdir overrides the profile's)
PROFILE=slurm,test sbatch workflows/pair-interactions/run_hpc.sh --outdir <shared storage>/pair-interactions/test
# 4. the full run: 25 searches, 25 prunes, 16 survey shards, cross-evaluation
sbatch workflows/pair-interactions/run_hpc.sh
# 5. back on the laptop: pull the outdir (~50 MB) and render the report
rsync -a <hpc>:<shared storage>/pair-interactions/v2/ pair-results/v2/
task pair:report RUN_DIR=pair-results/v2
```

Cost on Slurm, one task per CPU: handover searches ~20 min each (10 of them, the
critical path), interference ~6 min, conditioning ~1 min, PRUNE 1-15 min, a survey
shard ~2 min, CROSSEVAL ~15 min after the last search. Expect ~1 h wall if the
queue is quiet; ~5 core-hours in all. Everything is `-resume`-safe: re-submitting
`run_hpc.sh` with more `--seeds` or `--survey_shards` only runs what is new.

`cfs interactions` exits 1 when V5 does not pass; that is recorded in
`interactions.json` and is not a task failure. A task fails only if no
`interactions.json` was written, and then prints the log's tail.

## Test

```bash
nextflow run workflows/pair-interactions -profile test --data pair-data
```

One seed, a tiny search, three runs, two 3-medium survey shards per arm: every
process end to end in ~1 minute on a laptop. Checks the plumbing, not the science;
render it with `task pair:report RUN_DIR=results-test` to check the report too.
