# Example: one mGEM

Smallest useful run of the pipeline: a single CarveMe GEM (`FNPN01`, copied from
`~/Documents/carveme_models/20HM+/FNPN01.xml`), one "community" of one member,
a few hundred sampled media, three active-learning rounds, and a 4-cell sweep.

```
roster.csv          the samplesheet: genome_id, model_path (relative to this file)
models/FNPN01.xml   the mGEM. This is the only required input — no genome FASTA
                    is consumed anywhere in the pipeline.
params.config       every knob, with comments on what to change
run.sh              launcher
results/            outputs (gitignored)
```

## Run

Needs Nextflow ≥25 and Docker (images pull from GHCR).

```bash
./run.sh              # real run
./run.sh -stub        # wiring check: no containers, no solver, seconds
```

Add `-profile singularity` on HPC, plus `-c site.config` for your executor.

## Outputs

```
results/leaderboard.csv     one row per sweep cell (community x arch x ensemble x n_train) with metrics
results/models/<cell>/      checkpoint + train_metrics.json per cell,
                            cell = FNPN01__h128x128__k3__n100 etc.
results/pipeline_info/      execution trace, report, timeline, DAG
```

Intermediate tidy tables (`samples.csv`, `media.csv`, `member_growth.csv`,
`membership.csv`, `exchange_universe.json`) stay in Nextflow's `work/` — they are
not published. `-resume` reuses them.
