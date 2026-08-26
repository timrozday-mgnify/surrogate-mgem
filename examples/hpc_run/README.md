# HPC run: ground-truth labels, then the M3b Head A sweep

Everything needed to take the 21-genome roster from CarveMe GEMs to a scored
leaderboard of value-head architectures, in two `nextflow run` invocations. Both
stages use the same containers (GHCR, Docker locally / Singularity on HPC) and the
same `site.config`.

The question the run answers: Head A's gate is **`worst_grad_cosine > 0.99`**.

The 2026-08 run of this pipeline found the previous ceiling (0.733) was not a label
or capacity limit but a **coordinate** one: `mu_max` is an LP value function in its
RHS and the uptake bound is `lb = -Vmax * u`, so it is concave and piecewise *linear*
in `u` — while the heads imposed concavity in `x = u/(u+s)`, which is a strictly
smaller class that excludes the target (`x` tangent test violated on 32-57% of label
row pairs, `u` on 0.0%). That is why 37× the parameters moved paired per-organism
cosine and R² by ±0.001.

`--arch icnn-u` / `deepset-u` are the correction — the same heads over
`w = min(u/s, 300)`, affine in `u`. On one organism that took R² 0.477 → 0.802, and
a `w_grad` sweep on top of it reached **cosine 0.903 / R² 0.756** at `w_grad = 10`,
beating the forest (0.817), the unconstrained MLP (0.812) and the old ICNN (0.715)
from *inside* the concave class. This run asks whether that holds on all 21
organisms, where the frontier's optimum actually sits, and whether capacity is a
live axis once the gradient term is the one binding.

```
make_roster.sh      roster.csv from a directory of GEMs (run this first)
reference/
  metabolite_index.json   THE frozen index — see "Reference data" below
labels.config       stage 1 params: 20000 media/organism, probe-anchored bands
run_labels.sh       stage 1 launcher   -> labels_out/labels/
make_sweep.py       writes a sweep samplesheet from axis flags
make_sweep_full.sh  regenerates sweep_full.csv as five named arms
sweep_smoke.csv     4 cells against labels_stub — -stub only (no real data)
sweep_full.csv      36 cells: the real sweep
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

One task per samplesheet row. `sweep_full.csv` is 36 cells in seven arms, written by
`make_sweep_full.sh` (which carries the measurement behind each arm in its comments):

| arm | cells | what it tests |
| --- | --- | --- |
| A `icnn-u`, `w_grad` {1,3,10,15,20} | 5 | **the axis that moves the gate.** 10 was the best of six points on one organism; 10-20 was never probed |
| B `icnn-u` width {512,1024} × depth {3,6}, at `w_grad` 10 | 4 | capacity, re-tested where the gradient term actually binds — the last run measured it at `w_grad` 1 and called it inert |
| C `icnn`, `mlp`, `rf` at the same `w_grad` | 3 | the x-space head being replaced, plus an unconstrained ceiling and a non-parametric floor |
| D `deepset-u-private`, `phi` {32,64} × `k_code` {16,64} | 4 | the per-metabolite head with its coordinate fixed. **This arm is most of the cost** |
| F `groupmax-u`, group {4,8,16} × temperature {0.01,0.03,0.1} | 9 | **kinks as the primitive.** Also the accuracy-vs-conditioning frontier: curvature scales as 1/T |
| G `groupmax-u` seeded from label tangents, K {100,1000} × T {0.01,0.03} × init {labels,random} | 8 | **initialisation, not architecture.** Both inits at matched K and T |
| E the 4000-media set: `icnn-u`, `icnn`, `rf` | 3 | the rows axis |

**Both label roots must exist.** Arm E reads `labels_out_4k/labels`; generate it by
running stage 1 a second time:

```bash
OUTDIR=labels_out_4k NF_PROFILE=singularity ./run_labels.sh --label_media 4000 -c site.config
```

Or point `LABELS_4K` at an existing 4000-media root and regenerate the sheet. This
matters: the previous sweep shipped a `/path/to/...` placeholder here, every task
silently read the 20000-media root instead, and `m4k__rf` came back **bitwise
identical** to `m20k__rf` — the rows axis was reported as measured when it had never
run. Check `n_train_media` in two cells' `diagnostics.json` differ before believing
any rows conclusion.

Arm E has a prediction on file, which is what makes it a test rather than a fishing
trip: **rows will not help.** The train-vs-held-out gap at the `w_grad` optimum is
0.005 cosine / 0.013 R², so the head underfits, and rows only buy variance.

**Arm F is the architecture bet.** Every concave head so far builds a corner out of
a non-negative sum of smooth softplus ridges, and `mu_max` is piecewise *linear*
with ~1700 distinct active sets per organism — depth compounds smoothness rather
than making corners. `groupmax-u` makes the max the activation, so one corner is one
unit, and it nests plain max-affine exactly at `--width 1 --depth 1 --gm-group K`.
`--gm-temp` is not only an accuracy knob: a hard max has zero Hessian inside a piece
(P3, the one thing §8's Newton cannot follow) and curvature scales as `1/T`, so this
arm measures the frontier Newton has to buy from. It costs about what Arm A does.

**Arm G is the one to read first.** Every labelled row is an exact supporting
hyperplane of `mu_max`, so the head's first layer can simply be *told* its affine
pieces rather than discovering them. At `--width 1 --depth 1` the head IS
`min_k(a_k.w + c_k)` and the seed reproduces the pruned tangent model outright,
which scores held-out cosine 0.95-0.96 and R² 0.996 **before any training**.
Ranking the tangents by active-set frequency is what makes a small K enough: 100
ranked planes match ~2000 random ones.

It matters because random init measurably does not get there. At identical class
and identical `T = 0.1`, the tangent model scores 0.923 and the same head trained
from noise scores 0.712 — a 0.21 optimisation gap, which is the failure that
LSPA/CAP-style max-affine fitting exists to fix. The arm sweeps both inits at
matched K and T so that comparison is inside one arm.

Drop Arm D first if the queue budget is tight — Arms A-C answer the gate question on
their own. The shared-trunk `deepset-u` is deliberately absent: it OOM-killed in
every cell last run, and sharing across organisms measured *worse* than private on
every metric.

Regenerate the whole sheet, or a variant:

```bash
./make_sweep_full.sh > sweep_full.csv
```

Regenerate with different axes:

```bash
./make_sweep.py --labels m20k=labels_out/labels --arch icnn-u --width 128,512 --w-grad 1,10 --epochs 1500 -o sweep_full.csv
```

The samplesheet is `cell_id,arch,labels,args`, and `args` is the literal flag string
for that cell — so a cell's exact command stays greppable, and `rf` rows can carry
`--n-estimators/--delta` while deepset rows carry `--phi-hidden/--k-code`. Only
*swept* axes appear in the cell id. `./make_sweep.py --demo` self-checks the
generator.

**Cost**, measured on the 2026-08 run of this pipeline at 20000 media (324 cpu-h for
its 21 completed cells): ICNN cells ~5 min per organism at width 128, ~17 min at 512,
~47-96 min at 1024; `mlp` ~12 min; `rf` ~1 min. Arms A-C and E are therefore ~20
cpu-h all in. **Arm D is 420-920 cpu-h on its own** — 5-11 h per organism per cell —
so budget the run around it or drop it. One more warning from last time: its
`ph64/kc64` cells all died at exactly 11 h 59 m, i.e. a **12 h queue wallclock**, not
the 48 h `max_time` in `site.config`. Check the queue's own limit before launching.

The deepsets are the expensive arm — `phi` is priced *per metabolite*, so
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

Every arch but the shared-trunk `deepset`/`deepset-u` is fanned out **one organism per task** —
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
