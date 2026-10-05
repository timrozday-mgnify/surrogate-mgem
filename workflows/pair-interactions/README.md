# pair-interactions

The first application of the surrogate to a use case: which metabolic interactions
one pair of genomes can have, and in which media (design spec §13.5/M13, §13.11/M16).

```text
INTERACTIONS  one `cfs interactions` search per (arm, objective, seed):
              arm       = the model: plain FBA, or FBA + product inhibition at c^eq
              objective = handover (cross-feeding), interference (competition +
                          inhibition), conditioning (inhibition alone; inhibited arms)
              -> runs/<arm>__<objective>__s<seed>/{interactions.json, media.npz, run.json}
CROSSEVAL     every start and designed medium re-solved with the true LP under every
              model, plus the spent-medium assay at each best design
              -> crosseval/{media,links,spent}.csv
```

Each search proposes media with the surrogate and accepts them with the true LP, so
every reported interaction is LP-verified. Measured on a laptop (2 cores per task,
5 at once): handover 16-22 min, interference 4-7 min, conditioning <1 min (it screens
constructed media without an ascent, so its seeds give identical answers); the
default 15 searches + cross-evaluation took 47 min wall. On Slurm, expect ~25 min.

## Plan

1. **One pair, both models, three seeds** (this pipeline, defaults). The pair is
   `CP040530.1 + CP070062.1`: earlier work found a reciprocal glycerol /
   glyceraldehyde exchange and contended disposal routes for both members, so it
   has both positive and negative interactions to find. Read the report for:
   which links reproduce across seeds and objectives (the consensus table), whether
   they survive inhibition at a fixed medium, and who suppresses whom.
2. **Check the instruments before the biology.** In the report's Diagnostics:
   surrogate vs LP link rates, the rank correlations, and the seed spread. A link
   found in one seed only, or a designed medium whose rate the LP does not confirm,
   is a lead, not a result.
3. **Firm up what reproduces**: more seeds (`--seeds 0,1,2,3,4,5`) and a larger
   survey (`--args '--draws 128'`) for the pair; then a `c^eq` sweep (add arms with
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
# 2. on the cluster, from the repo root
bash workflows/pair-interactions/setup.sh
cp workflows/pair-interactions/hpc.example.config workflows/pair-interactions/hpc.config  # fill in <...>
sbatch workflows/pair-interactions/run_hpc.sh
# 3. back on the laptop: pull the outdir (small) and render the report
rsync -a <hpc>:<shared storage>/pair-interactions/v1/ pair-results/
task pair:report RUN_DIR=pair-results
```

`cfs interactions` exits 1 when V5 does not pass; that is recorded in
`interactions.json` and is not a task failure. A task fails only if no
`interactions.json` was written, and then prints the log's tail.

## Test

```bash
nextflow run workflows/pair-interactions -profile test --data pair-data
```

One seed, a tiny search, three runs: checks the plumbing in ~2 minutes, not the
science.
