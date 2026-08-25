# HPC run: ground-truth labels, then the M3b Head A sweep

Everything needed to take the 21-genome roster from CarveMe GEMs to a scored
leaderboard of value-head architectures, in two `nextflow run` invocations. Both
stages use the same containers (GHCR, Docker locally / Singularity on HPC) and the
same `site.config`.

The question the run answers: Head A's gate is **`worst_grad_cosine > 0.99`** and the
best measured to date is **0.733**, with value R² 0.538 against a random forest's
**0.979** on the same rows. That gap is an *architecture* deficit, not a label
ceiling — but every measurement so far is one laptop at 4000 media/organism, width
128, depth 3, and the models underfit. Stage 1 produces D10-scale labels (5× the
rows); stage 2 sweeps width, depth and architecture over them. If R² does not move
with 5× rows and 8× width, localisation is confirmed and the answer is a
per-metabolite architecture rather than scale.

```
make_roster.sh      roster.csv from a directory of GEMs (run this first)
reference/
  metabolite_index.json   THE frozen index — see "Reference data" below
labels.config       stage 1 params: 20000 media/organism, probe-anchored bands
run_labels.sh       stage 1 launcher   -> labels_out/labels/
make_sweep.py       writes a sweep samplesheet from axis flags
sweep_smoke.csv     3 cells against labels_stub — -stub only (no real data)
sweep_full.csv      30 cells: the real sweep
sweep.config        stage 2 params: xla_devices, per-process resources
run_sweep.sh        stage 2 launcher   -> sweep_out/sweep_leaderboard.csv
site.config         EXAMPLE slurm + singularity config, shared by both stages
labels_stub/        empty stand-in so sweep_smoke.csv resolves with no data
```

## Before you start

Copy `site.config` and edit three things: the queue, the account, and a **shared**
`singularity.cacheDir` so parallel tasks don't race converting the same image.

Build the roster, then check the wiring without touching a solver or a container —
seconds, and it catches a broken path or a malformed samplesheet before you burn
queue time:

```bash
./make_roster.sh /path/to/carveme_models > roster.csv
./run_labels.sh -stub && ./run_sweep.sh -stub
```

`-stub` still resolves every input path, so the roster has to point at GEMs that
really exist — which is most of what this check is for.

## Stage 1 — labels

```bash
NF_PROFILE=singularity ./run_labels.sh -c site.config
```

One task per organism: active subspace (§4.2) → demand probe (§4.7) → stratified
Sobol design (§4.3–4.4) → elastic-net QP solves (§5.4), sharded to
`labels_out/labels/<id>/eps_<e>/part.parquet` with `<id>.subspace.json`
and `<id>.exchanges.json` sidecars beside them.

**Cost**, measured through this pipeline on a laptop and scaled: ~5 h and ~150 MB
per organism at 20000 media, single-core, under 1 GB RSS. 21 organisms in parallel
is one afternoon and ~3.2 GB published (plus as much again in `work/` until you
delete it). `labels.config` sets a 16 h wall so the first attempt does not hit
`process_low`'s 4 h default, which is sized for the 4000-media run.

Do a trial first — it costs minutes and proves the whole chain including the image
pull:

```bash
OUTDIR=trial ./run_labels.sh --label_media 200
```

Then check the trial's `*.subspace.json`: `bands` should be mostly
`"source": "probe"`. That is §4.7 anchoring each metabolite's sampling band on its
own limiting regime, which is what separates this from the pre-band design — where
every metabolite shared one band and nearly every limiting medium went to whichever
metabolite had the smallest scale.

**One core per task, deliberately.** Never let cobra parallelise inside a task: an
exchange-FVA that takes ~1 s single-threaded went to *not finishing in 30 minutes*
with a worker pool, because each worker re-pickles the GEM. HiGHS is likewise pinned
to `threads=1` so labels are repeatable. The parallelism is the 21-way fan-out.

**If a task dies mid-shard** (a full disk is the usual cause) it leaves a partial
parquet that must be deleted, not resumed. Nextflow handles this for you — a failed
task re-runs in a fresh work dir — but if you ever run `cfs generate` by hand,
delete the partial shard first.

## Stage 2 — the sweep

```bash
SWEEP=sweep_full.csv NF_PROFILE=singularity ./run_sweep.sh -c site.config
```

One task per samplesheet row. `sweep_full.csv` is 30 cells:

| arm | cells | what it tests |
| --- | --- | --- |
| ICNN width {128,256,512,1024} × depth {3,4,6} | 12 | does scale close the R² gap? |
| deepset + deepset-private, `phi` hidden {32,64,128} × `k_code` {16,64} | 12 | the localisation bias, at a trunk width a laptop could not afford |
| `mlp` (unconstrained) and `rf` | 5 | ceiling and floor — both changed how the ICNN's numbers read |
| the 4000-media set, ICNN + rf | 3 | the rows axis: same architectures, 1/5 the labels |

27 rows point at `labels_out/labels`, so stage 2 chains off stage 1 with no editing.
The 3 rows tagged `m4k` point at `/path/to/20hm_bands/labels` — **edit that path** to
your existing 4000-media set, or drop those rows.

Regenerate with different axes:

```bash
./make_sweep.py --labels m20k=labels_out/labels --arch icnn --width 128,512 --depth 3,4 --epochs 1500 -o sweep_full.csv
```

The samplesheet is `cell_id,arch,labels,args`, and `args` is the literal flag string
for that cell — so a cell's exact command stays greppable, and `rf` rows can carry
`--n-estimators/--delta` while deepset rows carry `--phi-hidden/--k-code`. Only
*swept* axes appear in the cell id. `./make_sweep.py --demo` self-checks the
generator.

**Cost**: the ICNN cells are ~9 min each at width 128 (0.35 s/epoch × 1500) and grow
with width; the deepsets are the expensive arm — `phi` is priced *per metabolite*, so
at trunk width 256 it measured 47 s/epoch, i.e. ~19 h for one cell. That arm is the
reason this needs a cluster.

**Memory**: measured on 4000 media (1/5 of this run's rows), `--arch deepset`, by
peak RSS.

| cell | load | peak |
| --- | --- | --- |
| `--phi-hidden 64 --k-code 64 --batch 512` | 2.1 GB | 16.8 GB |
| `--phi-hidden 128 --k-code 64 --batch 128` | 2.2 GB | 14.0 GB |
| `--phi-hidden 128 --k-code 64 --batch 512` | 2.2 GB | >24 GB (OOM-killed) |

Essentially all of it is live autodiff activations inside one training step —
(organisms × batch × 444 metabolites × `phi_hidden`) through a second-order (Sobolev)
tape. Evaluation adds **0.00 GB** on top and nothing accumulates across epochs, so
there is nothing to spill to disk; the levers are `--batch` (linear) and
`--phi-hidden`. Going from 4000 to 20000 media only adds the resident label arrays,
about 5 GB. `TRAIN_VALUE` therefore requests **48 GB for deepset cells and 16 GB for
the rest**, off `task.tag`, rather than one blanket number.

That is also what the original failure was: a 16 GB request against a ~22 GB cell.
Two things hid it, both fixed here. `params.max_*` is `process.resourceLimits`, which
clamps the `* task.attempt` retry ladder too, so a ceiling below what a cell needs is
a cap and not a safety net. And `TRAIN_VALUE`'s `|| true` — there because
`cfs train-value` exits 1 on an unmet gate — also swallowed the kernel's 137, the one
status `conf/base.config` retries on; it now tolerates exit 1 only.

## Reference data

`reference/metabolite_index.json` is **the frozen metabolite index** — 444 exchanges,
365 shared / 79 private, produced by `--stage qc` on this 21-genome roster (21/21
EGC-free, V0 passing). It is committed here because every label row is hashed
against it and every downstream run must pass the *same* one: a different index
silently invalidates every label. Both run scripts default `--index` to it.

Only regenerate it if the roster changes:

```bash
nextflow run ../.. -profile singularity --stage qc --roster roster.csv --outdir qc_out
```

## Outputs

```
labels_out/labels/<id>/eps_<e>/part.parquet   the label shards
labels_out/labels/<id>.{subspace,exchanges}.json        bands + exchange order
sweep_out/sweep_leaderboard.csv                         one row per cell
sweep_out/sweep/<cell_id>__<genome_id>/                 diagnostics.json + weights
*/pipeline_info/                                        trace, report, timeline, DAG
```

Every arch but the shared-trunk `deepset` is fanned out **one organism per task** —
the organism axis is a vmap axis those heads do not share anything across, so 21
short jobs replace one long one. `COLLECT_VALUE_METRICS` merges a cell's tasks back
into a single leaderboard row, so the CSV is unchanged: one row per cell.

`sweep_leaderboard.csv` carries worst/mean grad cosine, min/median value R², the
worst organism, Hessian conditioning, and `n_val_media`/`n_train_media`/
`rounds_present` — that last group is the split provenance, and two runs are only
comparable if it matches.

## Two things that will bite

- **`cfs train-value` exits 1 whenever the gate is unmet**, and the best cell to date
  is 0.733 — an honest exit code would fail every task in the sweep. The module
  tolerates it and gates on `diagnostics.json` existing instead. Read `passed` in the
  leaderboard, not the pipeline's exit status.
- **`xla_devices` must divide the organism count**, and the sweep now stacks **one
  organism per task**, so leave it at 0. `train._shard_organisms` logs "do not
  divide" and silently runs unsharded otherwise — a slow run that looks like a fast
  one. It only ever applied to a multi-organism stack, i.e. the `deepset` cells,
  where 21 organisms → 3 or 7.
