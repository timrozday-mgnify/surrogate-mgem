# CLAUDE.md — surrogate-mgem

> ## Design north star — read this first
>
> The project is **pivoting** to the v2 design spec:
> [`docs/design/community-fba-surrogates-plan-v2.md`](docs/design/community-fba-surrogates-plan-v2.md).
> That document is the authoritative map — milestones **M0…M8**, locked
> decisions **D1…D10**, validation gates **V0…V8**, pitfalls **P0…P15**. Orient
> to it before starting work.
>
> The target architecture is **JAX/Equinox** with **per-organism CarveMe FBA**
> ground truth (not MICOM community solves): a concave value head (`mu_max`), a
> behaviour head (exchange fluxes), shadow-price Sobolev training, elastic-net
> label uniqueness, and Newton composition. New work lands under **`src/cfs/`**.
>
> The existing **`src/surrogate_mgem/`** package (PyTorch + MICOM growth
> surrogate, documented below) is **legacy/reference** — kept, not deleted.
>
> ### Progress — M0–M2 and §4 done on the real roster; M3 built, gate not yet met
>
> | Milestone | State | Code |
> | --- | --- | --- |
> | M0 QC + frozen index | **done, V0 passes** — 21/21 EGC-free; index = 444 exchanges, **365 shared / 79 private** (so the Newton Jacobian is 365×365) | `src/cfs/groundtruth/{qc,index}.py` |
> | M1 degeneracy → D4 | **done, V1 complete** — **68.9%** of 371k exchange-FVA observations degenerate roster-wide (59.7–82.8% per genome; 88% at α=0.7 vs 50% at α=1.0) ⇒ **D4 = elastic net** | `src/cfs/validate/degeneracy.py` |
> | M2 solve interface | **done, §3.4 gate verified on a real GEM** — MM uptake bounds (§3.3), FBA for `mu_max` + duals, then the **Clarabel** elastic-net QP for `z` | `src/cfs/groundtruth/solve.py` |
> | §4 sampling + bulk labels | **done, generated** — active subspace, stratified-Sobol design, parquet driver | `src/cfs/sampling/`, `--stage labels` |
> | M3 Head A (value) | **built and training on the real roster; gate NOT met** — held-out gradient cosine **worst 0.40 / mean 0.72 / best 0.94** against the required 0.99. Concavity violations ~0 (7e-5), so the structural prior holds | `src/cfs/surrogate/`, `cfs train-value` |
>
> **The label set** (M3/M4 train on this): `~/Documents/surrogate-mgems_runs/20hm/`
> from `~/Documents/20hm_carveme_models` (21 CarveMe GEMs). 4000 media/organism —
> 1/5 of the D10 budget, laptop-sized; `SamplingConfig.n_media` still defaults to
> 20000 for HPC. **940 800 rows, 573 MB, 21/21 organisms, 0 failures, 100% optimal
> solves**, one `index_hash` throughout (P13). Per organism: 32 000 rows at the
> primary `eps=1e-3` and 6400 at each of `1e-2`/`1e-4`. `|A_i|` = 11–32, median 24.
> The run dir also holds `check_v2.py` (the §3.4 gate) and `check_labels.py`
> (shard sanity), plus `local.config` — the external `-c` site config for this box.
>
> **Two label-interpretation bugs M3 uncovered — read before touching the duals.**
> The stored `shadow` is `d(mu_max)/d(uptake bound)` *only where that bound
> binds*. Elsewhere it is the metabolite's value in the network, which for waste
> like CO2 is **positive** — i.e. it claims more nutrient lowers growth, which no
> LP can do. Finite differences: 12/12 positive-dual cases returned
> `d(mu_max)/d(supply) = 0.000000` exactly. That is ~12% of the non-zero duals, on
> metabolites present in half the media. Second, **half the "non-zero" duals are
> solver dust** (O(1e-14)); a row whose whole target is dust dominates any
> norm-relative loss by ~1e14. Both are handled in
> `cfs.surrogate.data._organism_arrays` (clamp at 0, `_DUAL_TOL`) — do not
> "simplify" that back to a plain `-shadow`.
>
> **What shapes M3 and will shape M4.** On a real GEM `mu_max` is a *linear ramp*
> in saturation `x = c/(Km+c)` that reaches its plateau by `x* ~ 5e-3`: 99% of the
> input range carries no signal, and `d(mu)/dx` spans five decades within one
> organism. Three fixes each bought a large jump and are load-bearing —
> per-metabolite kink rescale of the inputs (`_kink_scale`, affine so concavity is
> untouched), a **per-row norm-relative** Sobolev term instead of §7.1's absolute
> `||grad - pi||^2` (absolute gave cosine 0.05: a few ion-limited rows absorbed the
> whole term), and ICNN init at `softplus^-1(1/width)`.
>
> **Next: finish M3.** Cosine is still budget-bound (0.50 → 0.56 → 0.72 as capacity
> and steps grew), and at `w_grad=10` the value head collapses late in training
> (R2 -0.61) while cosine climbs — balance the two terms first. The worst gradient
> errors are the ion-limited metabolites (`EX_mg2_e`, `EX_k_e`, `EX_cl_e`,
> `EX_ca2_e` lead in 17–19 of 21 organisms), exactly where the ramp is steepest, as
> §7.3 predicts. Checkpoint + diagnostics:
> `~/Documents/surrogate-mgems_runs/20hm/value/`. Hessian condition number is 7e11
> — recorded, not gated, and the early warning for M6.

## Legacy package (surrogate_mgem)

Two layers: a Python package (`src/surrogate_mgem/`, the surrogate model + CLI)
and a Nextflow pipeline (`main.nf` + `workflows/` + `modules/`) that scales
training across an HPC cluster. This page is the map; read the linked code for
detail.

## Python package

`surrogate-mgem <subcommand>` (`cli.py`), consumed by the pipeline:

| Subcommand | Does | Needs |
| --- | --- | --- |
| `generate` | Sample communities + media, solve MICOM, write tidy CSVs. Shardable via `--num-shards/--shard-index` (shard 0 writes `exchange_universe.json`). | `data` extra (micom/cobra) |
| `train` | Fit a fixed-community ensemble. Sweep knobs: `--hidden` (layers×width), `--n-models` (ensemble size), `--n-train` (training-row cap), `--n-features` (input width). Writes `train_metrics.json`. | torch only |
| `active-round` | One active-learning round for one community: train acquisition ensemble → solve a diverse high-uncertainty batch → append to the tidy tables (single-community output dir). | `data` extra |
| `report` | Quarto performance report (local, not in the HPC path). | `report` extra + quarto |

Model: `model.py` `GrowthSurrogate` (standardising ReLU MLP; `hidden` architecture
is **persisted in the checkpoint** so a sweep can vary it). `ensemble.py`
`GrowthEnsemble` (deep ensemble → predictive std = acquisition signal).
`active.py` `active_round` / `active_learning_loop`. `train.py`
`run_active_round` does the tidy-table writeback.

## Nextflow pipeline

House style mirrors `../subspecies-phylogeny`: DSL2, meta maps, `conf/base.config`
labels + retry, `conf/modules.config` for `ext.args`/publishDir, nf-test stub
tests, per-process container ternary.

`main.nf` has three stages via `--stage` (default `train`):

- **`qc`** — M0+M1 ground-truth QC (`workflows/groundtruth_qc.nf`, the v2 pivot).
  `QC_MODELS` (whole roster: EGC gate + MEMOTE, freeze
  `${outdir}/qc/metabolite_index.json`) → `DEGENERACY_SURVEY` (per organism,
  exchange-FVA — the FVA is the cost, hence the per-genome fan-out)
  → `COLLECT_D4` (roster-wide `d4_recommendation.json`; advisory — the human
  records D4). CLI: `cfs {qc, freeze-index, degeneracy}` (`src/cfs/cli.py`).
  Run this first, before any `train`. Stub: `tests/qc.nf.test`.
- **`labels`** — §4.5 bulk ground-truth labels (`workflows/label_generation.nf`).
  `GENERATE_LABELS`, one task per organism: active subspace (§4.2) → stratified
  Sobol design (§4.3-4.4) → elastic-net solves, sharded to
  `${outdir}/labels/genome_id=<id>/eps=<e>/part.parquet` plus `<id>.subspace.json`
  / `<id>.exchanges.json` sidecars. Needs `--index` (the `metabolite_index.json`
  the `qc` stage freezes) and `--label_media` (4000 here; the D10 scale is 20000).
  CLI: `cfs generate`. Stub: `tests/labels.nf.test`.
- **`train`** — the legacy sweep below.

DAG (`workflows/surrogate_training.nf`):

```
GENERATE_DATA (per shard) ─┐
                           ├─ MERGE_DATA ─ pick top communities ─┐
                           ┘                                     │
   ACTIVE_LEARN (per community: N discrete active-round calls    │
     folded in one task, dataset grows each round) ──────────────┤
                                                                 │
   TRAIN_SURROGATE (per cell = community × hidden × n_models      │
     × n_train × n_features) ── COLLECT_METRICS ── leaderboard ──┘
```

| Module | Image | Label |
| --- | --- | --- |
| `GENERATE_LABELS` | `surrogate-mgem-data` | process_low |
| `GENERATE_DATA` | `surrogate-mgem-data` | process_high |
| `MERGE_DATA` | `surrogate-mgem-train` | process_low |
| `ACTIVE_LEARN` | `surrogate-mgem-data` | process_medium |
| `TRAIN_SURROGATE` | `surrogate-mgem-train` | process_low |
| `COLLECT_METRICS` | `surrogate-mgem-train` | process_single |

### Conventions / things easy to get wrong

- **Iteration lives inside a process, not the DAG.** Nextflow forbids invoking a
  process more than once, so `ACTIVE_LEARN` folds `params.active_rounds` discrete
  `active-round` calls in a bash loop (like the reference's `accumulating_merge`),
  rather than unrolling per-round Nextflow tasks. Each round is still a distinct
  CLI invocation that grows the dataset.
- **Data-size sweep = `--n-train` cap** on a fixed dataset (a learning curve), not
  active-round snapshots.
- **Containers only** (no bioconda package) — the modules are container-only with
  no `environment.yml`; the `conda` profile won't cover them. Two images, built
  out-of-repo via `docker/{train,data}.Dockerfile`, referenced by GHCR convention
  (`ghcr.io/timrozday-mgnify/surrogate-mgem-{train,data}:0.1.2`). Bump the tag in
  all five modules together. No `-sif` ORAS artifacts are published, so the
  modules name the Docker image plainly (no nf-core `oras://` ternary) and
  singularity/apptainer converts on first pull.
- **Media sampling is the whole ballgame — use `titrate`.** A random nutrient
  subset (`sparse`) practically never contains the organism's essential set, so
  growth is 0 for every sample; and an uptake bound far above saturation makes
  every viable medium grow at the same rate. `data.medium_spec` fixes both: it
  bisects for the limiting bound, scans for essential exchanges, and reads each
  nutrient's **own uptake demand** off the LP (`estimate_demand`). Demands span
  orders of magnitude, so a shared sampling band leaves the small ones saturated
  and growth cannot respond to them — that is a data-generation defect no
  rescaling or extra data can undo. All three go to `medium_spec.json` for the
  active loop to reuse. `params.min_growth_frac` makes `MERGE_DATA` abort when
  the target is flat.
- **Limit a few nutrients per medium, not all of them.** `titrate` gives every
  offered nutrient its own demand-relative bound but only makes `n_limiting`
  (default 3) of them scarce; the rest sit replete at 2-5x demand. Titrating all
  ~110 at once makes growth a minimum over everything: the target's spread
  collapses (std 5.2 -> 0.2) and even a random forest falls from 0.90 to 0.75.
- **A wide medium space needs `--n-features`.** Growth is set by a handful of
  limiting nutrients; the rest are dimensions a dense net memorises noise in.
  `model.select_features` (RF importances, training rows only) picks the input
  view, and it is persisted in the checkpoint alongside the `log1p` input
  transform, so `predict` and the acquisition loop still take full-width media.
  It is a sweep axis (`params.n_features_list`, cell suffix `__f<n>`) because the
  best width moves with dataset size and community: on one community at 2400 rows,
  8/16/32 features gave R2 0.56/0.86/0.70 against 0.44 for all 96.
- **Never let cobra parallelise inside a task.** `flux_variability_analysis`
  defaults to a worker pool; each worker re-pickles the GEM, and one exchange-FVA
  on a CarveMe model went from ~1 s to *not finishing in 30 minutes*. Everything
  in `src/cfs/` is single-threaded on purpose (FVA `processes=1`, HiGHS
  `threads=1` for label repeatability) — parallelism is the per-organism Nextflow
  fan-out.
- **HiGHS backs the default `hybrid` solver** — no CPLEX/Gurobi licence (`highspy`
  is in the `data` extra). **But not the QP**: the M2 elastic-net labels go
  through **Clarabel**, because HiGHS's QP active-set method failed on ~30% of
  real CarveMe solves at the primary `eps` and stalled for minutes at `eps=1e-4`
  (design doc §5.4). Do not "simplify" it back to one solver.
- **No slurm/test profile in-repo** — layer the executor via an external
  `-c site.config`; `max_cpus/max_memory/max_time` cap `process.resourceLimits`.
- **Community fan-out** picks the top `n_communities_augment` communities by
  feasible-sample count (channel algebra on the merged `samples.csv`).

### Dev commands

```bash
pip install -e ".[dev]"            # + ".[dev,data]" for the solver stack
pytest                             # solver-free units (incl. active-round writeback)
nf-test test tests/default.nf.test # stub pipeline (no solver, no containers)
nf-test test tests/e2e.nf.test --profile docker  # real solves+training, ~3 min, pulls both images
task report RUN_DIR=/path/to/run  # interactive run report -> rendered_reports/<run>_report.html
```
