# pair-interactions

Which metabolic interactions a community's **pairs** can have, and in which media
(design spec §13.5/M13, §13.11/M16). One pair or twenty-one members: the pairs are
derived from `--members`.

```text
SURVEY        the true-LP survey, ONCE for the whole community: media drawn over the
              union of every member's active subspace (§4.3, buffered species
              saturated), every member solved at each one -- FBA + elastic net, no
              surrogate. So one set of shards answers the question for every ordered
              pair: `G` solves a medium, not `2` per pair. Sharded
              (--survey_shards x --survey_media, default 8 x 500 per arm; ~0.11 s a
              solve, so ~2 min a shard for a pair and ~20 min for 21 members). Shard
              k is always the same media, so `-resume` with more shards only adds media
              -> survey/<arm>/shard_<k>.npz
RANK_PAIRS    edge_summary.py over the survey alone: the edge tables, and the pairs
              ranked by how many handovers reach most media. No solves
              -> survey_edges/{edges,graph_edges,relationships,pairs}.csv
INTERACTIONS  one `cfs interactions` search per (pair, arm, objective, seed), for the
              top --max_pairs pairs only -- a search costs ~20 core-h a pair at ten
              seeds, and the survey costs nothing extra per pair
              arm       = the model: plain FBA, or FBA + product inhibition at c^eq
              objective = handover (cross-feeding), interference (competition +
                          inhibition), conditioning (inhibition alone; inhibited arms)
              -> runs/<A>__<B>__<arm>__<objective>__s<seed>/
PRUNE         per run: the best design reduced, with the true LP on the run's own
              objective, to the moves that keep it within 1% (cheapest reverted
              first), and the mechanism there -- handovers >= 1% of E and each
              member's limiting exchanges. ~1-15 min a run
              -> prune/<run>.json
CROSSEVAL     per pair (it caches a cobra model per genome, and one task over a whole
              community's runs is the documented OOM): every start and designed medium
              re-solved with the true LP under every model, the spent-medium assay at
              each best design, and per handover FVA (forced vs tie-break), the other
              elastic-net eps, and abundance ratios 1:10..10:1
              -> crosseval/<A>__<B>/{media,links,spent,fluxes,robust}.csv
EDGES         edge_summary.py again, with the designed media folded in: the tables both
              reports read, for every pair whether or not it was searched
              -> {edges,graph_edges,relationships,pairs}.csv
```

Each search proposes media with the surrogate and accepts them with the true LP, so
every reported interaction is LP-verified.

Seeds draw independent random media; the candidate media target the same metabolites
in every seed (from the labels) on an independently drawn background. Before
2026-10-06 seed `s` reused seed 0's draws shifted by `s`, so **the seeds of earlier
runs are not replicates**. The default is **ten** seeds (50 searches a pair): at
five, `ceq1mM__handover` split 2/2/1 across three mechanisms, too few to call a
majority.

`cfs interactions` exits 1 when V5 does not pass; that is recorded in
`interactions.json` and is not a task failure. A task fails only if no
`interactions.json` was written, and then prints the log's tail.

## The edge tables

`EDGES` is the contract, and it is what scales. Rows reference no run, seed or
design, so rows from different communities concatenate:

| file | one row per | what it is for |
| --- | --- | --- |
| `edges.csv` | arm, metabolite, producer, consumer | the directed metabolite edge: `reach` + Wilson interval, `rate_typical` / `rate_capacity`, the FVA `forced` fraction, `seeds` |
| `graph_edges.csv` | arm, producer, consumer | the graph's directed edge: totals summed **at one medium** then aggregated, and `weight` for the edge width |
| `relationships.csv` | arm, unordered pair | `both_ways_pct`, `asymmetry`, `mutual` / `one-way` — what colours a network edge |
| `pairs.csv` | unordered pair, ranked | which pairs are worth a search |

Three things to keep right when reading or extending them.

**A pair total must be summed at one medium, then aggregated** — never as a sum of
each metabolite's own maximum, since those come from different designed media and
their sum is a rate nothing delivers. `edge_summary.survey_edges` does the
per-medium sum and takes `G` members unchanged.

**Reach and magnitude are separate axes.** On the first pair measured, the
largest-capacity edge (`glyc`, 330-610 mmol/gDW/h) has an FVA-forced fraction of
~0 — the QP's tie-break — while `pi` is fully forced at a sixth of the rate.
`weight` is magnitude alone and `reach`/`forced`/`seeds` qualify it.

**A pair with no search is not a pair with no interaction.** Only the top
`max_pairs` get `rate_capacity`, `forced` and `seeds`; every pair gets `reach` and
`rate_typical`.

## Reports

```bash
task community:report RUN_DIR=<outdir>              # every pair, as a network
task pair:report      RUN_DIR=<outdir> [PAIR=A,B]   # one pair, in full
```

The community report is the all-pairs view: a producer x consumer matrix (which
stays readable where a node-link plot is a hairball), the relationships, which
metabolites the community trades, whether the survey has converged, what each
model arm changes, and where a designed medium beat a drawn one. The pair report
is the deep dive on one pair — the pruned mechanisms, the seed agreement, the
spent-medium assay, every screened medium's composition. `PAIR` defaults to the
top-ranked pair in the outdir.

## Plan

1. **One pair, both models, ten seeds** (this pipeline, defaults). The pair is
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
3. **Firm up what reproduces**: more seeds (`--seeds 0,...,15`) where no
   mechanism has a majority, and more survey shards (`--survey_shards 16 -resume`)
   where the accumulation curve is still rising; then a `c^eq` sweep (add arms with
   `ceq: 0.1` / `10`) — only 1.0 mM has its own inhibited heads, so other values put
   inhibition in the true LP only (the stage 3' design).

### Steps 1-3, measured — `CP040530.1 + CP070062.1`, 2026-10-06

The first full run: 67/67 tasks, 25 searches x 5 seeds, 16 survey shards
(8000 media), V5 passing on all 25. Three things it settles.

**The survey has converged, so do not buy more shards.** 8 shards = 4000 media an
arm already gives Chao1 = the number seen (9.0 of 9 under FBA, 10.0 of 10 under
inhibition), nothing first seen in the last quarter, every frequency to within
~1% at 95%, and the accumulation curve flat from ~1000 media. The pair has **9-10
handovers and that is the whole list** at this medium prior.

**Inhibition does not perturb the frequencies, it creates a handover.** The
headline, at *independently drawn* media so the arms differ only in the bound:
`no2: 062 -> 530` **4.4% -> 90.6%** and `nh4: 062 -> 530` **0% -> 84.8%** (never
once in 4000 FBA media), with `glyc`/`pi` near-universal in both. The per-design
`inhibition_only_links` answers the same question at each run's own designed
medium and **contradicts itself across seeds** — no2 reads "only" in `handover__s0`
and "suppressed" in `s3`/`s4` — because the arms' optima are different media. The
survey is the comparison to quote; the report now tabulates it with a
disjoint-interval test.

**The mechanism reproduces, and the old grouping hid it.** Designs were grouped by
the full signature (handovers | each member's limiters), which put FBA handover at
five groups of one seed each and read as total irreproducibility. The limiters
shift with the seed — ties at the 25% cut are common — and they already have their
own table, so including them double-counted them. Grouped on the handovers alone:

| arm | objective | top mechanism | share |
| --- | --- | --- | --- |
| fba | handover | glyc + no2 + pi | **4/5** |
| fba | interference | glyc + no2 + pi | **4/5** |
| ceq1mM | interference | nh4 + no2 + pi | 3/5 |
| ceq1mM | conditioning | glyc + pi | 3/5 |
| ceq1mM | handover | glyc + no2 + pi / glyc + pi | 2/5, 2/5 |

So **step 3's seed spend is now one cell, not five**: `ceq1mM__handover` is the
only one without a majority, and its split is exactly whether `no2` is present —
the same link inhibition promotes. The pruned designs are also tiny: 228 moves
proposed, **0-2 kept**, so the search's design is one or two concentrations.

**The instruments, in one line each.** `obj_rank_spearman` is 0.75-0.91 for
interference and **0.47-0.69** for handover; at the designed medium the surrogate's
*level* is within +-1% for interference and **-50% to +324%** for handover, and it
proposes 2-13x too many links (`n_links_hat` vs `n_links_true`). The LP acceptance
test carries the handover searches entirely — and under FBA the ascent adds little
over its own LP-screened start (`true_gain_rel` 0.014-0.111, one seed exactly 0),
while under inhibition it adds 6-115%. Conditioning does not ascend, so its
`true_gain_rel` is 1e-13 by construction and its rank rho is `nan`; that is not a
failure.
4. **Other pairs**: stage their genomes (`stage_data.sh <src> <out> A B`) and run

### Scaling to a community's pairs

The report's "Interaction potential and magnitude" section is the contract for
that, and it writes three CSVs into the outdir so a community-scale report is a
concatenation rather than a rewrite:

| file | one row per | what it is for |
| --- | --- | --- |
| `edges.csv` | arm, metabolite, producer, consumer | the directed metabolite edge: reach + Wilson interval, typical and best rate, FVA-forced fraction, seeds |
| `graph_edges.csv` | arm, producer, consumer | the graph's directed edge: totals summed **at one medium** then aggregated, and `weight` for the edge width |
| `relationships.csv` | arm, unordered pair | `both_ways_pct`, `asymmetry`, and `mutual` / `one-way` — what a community graph colours an edge by |

Every row stands alone: no run, seed or design is referenced, so rows from
different pairs concatenate directly. Two things to keep right when they do.

**A pair total must be summed at one medium, then aggregated** — never as a sum
of each metabolite's own maximum, since those maxima come from different designed
media and their sum is a rate nothing delivers. `survey_edges` does
the per-medium sum; at community scale the same function takes `G` members
instead of 2 and needs no other change.

**Reach and magnitude are separate axes and must stay separate.** On this pair
the largest-capacity edge (`glyc`, 330-610 mmol/gDW/h) has an FVA-forced fraction
of ~0, i.e. it is the QP's tie-break, while `pi` is fully forced at a sixth of the
rate. Multiplying them into one score would hide that; `weight` is magnitude
alone and `reach`/`forced`/`seeds` qualify it.

For this pair the relationship summary is the whole result in one line: under
plain FBA it is **mostly one-way** (`CP070062.1 -> CP040530.1` at 4.6% of media),
and under inhibition at `c^eq` = 1 mM it is **mutual** (93.6%) — the same
`no2`/`nh4` creation the survey comparison found, as a graph edge.
5. **What not to read off it**: absolute rates and times (`Vmax` is ~30x
   physiological, so only orderings and ratios carry over), and anything about the
   assembled community's abundances — `E` is a capacity of the medium at equal
   biomass.

## Run

Needs Java 17+, Nextflow and the project's venv (`bash workflows/pair-interactions/setup.sh`,
which needs `uv`); everything runs on the host in that venv, as in
kmer-functional-profiler, so `-profile singularity` is not needed.

```bash
# 1. on your laptop: build the input bundle and copy it over. List EVERY member of
#    the community, not just a pair (~0.27 GB of checkpoints + ~65 MB a genome)
bash workflows/pair-interactions/stage_data.sh ~/Documents/surrogate-mgems_runs/20hm_bands \
    pair-data CP040530.1 CP070062.1 CR626927.1
rsync -a pair-data/ <hpc>:<shared storage>/pair-data/
# 2. on the cluster, from the repo root (git pull first if the checkout exists)
bash workflows/pair-interactions/setup.sh
cp workflows/pair-interactions/hpc.example.config workflows/pair-interactions/hpc.config  # fill in <...>
# 3. smoke test through Slurm, ~5 min with queueing. --outdir on the command line
#    always: the repo-root nextflow.config's outdir overrides what a profile sets
PROFILE=slurm,test sbatch workflows/pair-interactions/run_hpc.sh \
    --outdir <shared storage>/pair-interactions/test
# 4. the survey first, so the ranking decides where the search budget goes
sbatch workflows/pair-interactions/run_hpc.sh --members CP040530.1,CP070062.1,CR626927.1 \
    --max_pairs 0
# 5. then the searches on the top pairs; -resume keeps the survey
sbatch workflows/pair-interactions/run_hpc.sh --members CP040530.1,CP070062.1,CR626927.1 \
    --max_pairs 3
# 6. back on the laptop: pull the outdir and render
rsync -a <hpc>:<shared storage>/pair-interactions/v3/ pair-results/v3/
task community:report RUN_DIR=pair-results/v3
task pair:report RUN_DIR=pair-results/v3 PAIR=CP040530.1,CP070062.1
```

Cost on Slurm, one task per CPU. Per searched pair at ten seeds: 10 handover
searches at ~20 min each (the critical path), 10 interference at ~6 min, 5
conditioning at ~1 min, 25 prunes at 1-15 min and one CROSSEVAL at ~15 min —
**~20 core-h a pair**. The survey is the cheap half and does not scale with the
pair count: `survey_shards x survey_media x G` solves at ~0.11 s, so 8 x 500 is
~0.25 core-h for a pair and ~2.6 core-h for 21 members. RANK_PAIRS and EDGES are
minutes.

So budget `0.25 * G/2 + 20 * max_pairs` core-hours, and pick `max_pairs` from the
ranking rather than from the pair count: a 21-member community has 210 pairs
(4200 core-h for all of them) and its whole survey costs under 3.

Everything is `-resume`-safe: re-submitting with more `--seeds`, more
`--survey_shards` or a larger `--max_pairs` only runs what is new. Raising
`--members`, though, changes the medium prior — the draws are over the union of
the members' subspaces — so it re-runs the survey and everything after it.

## Test

```bash
# one pair: one seed, a tiny search, three runs, two 3-medium survey shards an arm
nextflow run workflows/pair-interactions -profile test --data pair-data --outdir results-test
# three members: the pairwise fan-out, the ranking, per-pair CROSSEVAL. Needs all
# three genomes staged (stage_data.sh ... CP040530.1 CP070062.1 CR626927.1)
nextflow run workflows/pair-interactions -profile community_test --data pair-data \
    --outdir results-community-test
```

Every process end to end in ~1 minute (one pair) or ~10 (three members). Checks
the plumbing, not the science — the survey is 6 media, so every frequency in the
report is meaningless. Render both reports against the outdir to check them too.
`--outdir` is needed because the repo-root `nextflow.config` overrides the
profile's.
