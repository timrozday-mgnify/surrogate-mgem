# Example: the M3b Head A sweep

The plan's §7.4 sweep, wired for the cluster. Everything measured on Head A so far
is one laptop — 4000 media/organism (1/5 of D10), width 128, depth 3 — and the models
**underfit**: a random forest reaches R² 0.979 on the same rows the ICNN reads at
0.538. This stage is what answers whether that deficit is scale or localisation:
one training cell per samplesheet row, run in parallel, collected into a leaderboard.

```
make_sweep.py     writes the samplesheet: the cross product of the axes, one row per cell
sweep.csv         a 3-cell smoke sweep (icnn, mlp, rf) against labels_stub — runs with -stub as-is
sweep_full.csv    the real sweep: 54 cells over rows x width x depth x architecture
params.config     run-level knobs (xla_devices, per-process resources), with comments
site.config       EXAMPLE executor config to copy and edit — slurm + singularity
run.sh            launcher
labels_stub/      an empty stand-in label root, so the committed sweep.csv resolves
index_stub.json   likewise for --index
results/          outputs (gitignored)
```

## The samplesheet

`cell_id,arch,labels,args`. `args` is the literal flag string for that cell, and
`arch=rf` routes to `cfs baseline-rf` instead of `cfs train-value`:

```
m4k__icnn__w128__d3,icnn,/path/to/20hm_bands/labels,--width 128 --depth 3 --epochs 1500 ...
m4k__rf,rf,/path/to/20hm_bands/labels,--eps 1e-3 --n-estimators 100 --delta 0.05 --seed 0
```

Literal flags rather than a param per axis, because the axes are not a clean cross
product — `--phi-hidden`/`--k-code` are deepset-only, `--n-estimators`/`--delta` are
rf-only, and the "5× rows" arm is a different label root — and because a cell's exact
command stays greppable afterwards. `labels` is absolute or relative to the
samplesheet, the same convention as `model_path` in a roster.

Regenerate:

```bash
./make_sweep.py --labels m4k=/path/to/20hm_bands/labels --arch icnn --width 128,256,512,1024 --depth 3,4,6 --epochs 1500 -o sweep.csv
```

Only *swept* axes appear in the cell id (`__w512__d4`); the fixed ones are still in
`args`. `./make_sweep.py --demo` self-checks the cross product.

`sweep_full.csv` is the four arms CLAUDE.md calls for: (1) D10's 20000 media/organism
as a second `--labels` root, (2) ICNN width {128…1024} × depth {3,4,6}, (3) deepset
at real trunk width — `--phi-hidden {32,64,128}` × `--k-code {16,64}`, shared and
private — and (4) `mlp` and `rf` as ceiling and floor. Edit its two `/path/to/...`
label roots before running.

## Run

Needs Nextflow ≥25 and Docker (images pull from GHCR).

```bash
./run.sh -stub
```

That is a wiring check against the stubs — no containers, no jax, seconds. A real run
needs a real label root and the frozen index:

```bash
SWEEP=sweep_full.csv INDEX=~/Documents/surrogate-mgems_runs/20hm/results/qc/metabolite_index.json ./run.sh
```

On HPC, copy `site.config`, edit the queue/account/cacheDir, then:

```bash
NF_PROFILE=singularity SWEEP=sweep_full.csv INDEX=/path/to/qc/metabolite_index.json ./run.sh -c site.config
```

The label sets themselves come from `--stage labels` (which anchors its own bands via
the §4.7 demand probe, so a new roster needs no previous run):

```bash
nextflow run ../.. -profile singularity --stage labels --roster roster.csv --index /path/to/metabolite_index.json --label_media 20000 --outdir d10
```

## Outputs

```
results/sweep_leaderboard.csv   one row per cell: worst/mean grad cosine, min/median value R²,
                                worst organism, Hessian cond, and the split provenance
results/sweep/<cell_id>/        diagnostics.json + value_heads.{eqx,json} per cell
results/pipeline_info/          execution trace, report, timeline, DAG
```

The gate is `worst_grad_cosine > 0.99`, measured in `u = c/(Km+c)` space. The sweep's
own success criterion is looser: any cell with value R² ≥ 0.9 *and* worst cosine ≥
0.9. If R² does not move with 5× rows and 8× width, localisation is confirmed and the
answer is a per-metabolite architecture, not scale.

## Two things that will bite

- **`cfs train-value` exits 1 whenever the gate is unmet**, and the best cell measured
  to date is 0.733 — an honest exit code would fail every task in the sweep. The
  module tolerates it and gates on `diagnostics.json` existing instead, so read
  `passed` in the leaderboard, not the pipeline's exit status.
- **`xla_devices` must divide the roster's organism count.** `train._shard_organisms`
  logs "do not divide" and silently runs unsharded otherwise — a slow run that looks
  like a fast one. 21 organisms → 3 or 7.
