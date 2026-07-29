# CLAUDE.md — surrogate-mgem

> ## Design north star — read this first
>
> The project is **pivoting** to the v2 design spec:
> [`docs/design/community-fba-surrogates-plan-v2.md`](docs/design/community-fba-surrogates-plan-v2.md).
> That document is the authoritative map — milestones **M0…M8**, locked
> decisions **D1…D10**, validation gates **V0…V8**, pitfalls **P0…P17**. Orient
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
> | M3 Head A (value) | **built; gate NOT met; four architectures measured** — best is still the ICNN at **worst 0.733 / mean 0.800** against the required 0.99, in `u` space. Value R² 0.54 (a forest gets **0.979** on the same split — the deficit is architecture, not labels), concavity violations exactly 0 (structural), Hessian cond 3.4e7. **4.29M parameters**, 0.35 s/epoch | `src/cfs/surrogate/`, `cfs train-value`, `cfs baseline-rf` |
>
> **The label set** (M3/M4 train on this): `~/Documents/surrogate-mgems_runs/20hm_bands/`
> from `~/Documents/20hm_carveme_models` (21 CarveMe GEMs). 4000 media/organism —
> 1/5 of the D10 budget, laptop-sized; `SamplingConfig.n_media` still defaults to
> 20000 for HPC. 21/21 organisms, 63/63 shards, **100% optimal solves**, one
> `index_hash` throughout (P13). Per organism: 32 000 rows at the primary
> `eps=1e-3` and 6400 at each of `1e-2`/`1e-4`. `|A_i|` = 11–32, median 24.
>
> It supersedes `20hm/` (same design, one shared sampling band) and differs only
> in that each metabolite's focus stratum is centred on **its own** limiting
> regime — `design.limiting_scales` → `cfs generate --scales`. Over `A_i`, roster
> medians: the median metabolite's limiting media **143 → 174**, metabolites with
> ≥50 media 15 → 17, the top metabolite's share 0.58 → 0.55. The run dir holds
> `make_scales.py` (labels → `scales.json`), `check_coverage.py` (the number that
> predicts the gate), `run_labels.sh` + `one_organism.sh` (21-way local fan-out —
> **not** nextflow: the `0.1.2` data image predated the band code and would have
> silently regenerated the old design. Fixed at `0.1.3` — the image rebuilds from
> `src/`, and `GENERATE_LABELS` now passes the band flags via
> `params.label_{probe,scales,round,focus_weights}`, so `--stage labels` reproduces
> the banded design). `20hm/` still holds `check_v2.py` (the §3.4
> gate), `check_labels.py`, `local.config` (the external `-c` site config) and —
> load-bearing — `results/qc/metabolite_index.json`, the frozen index every run
> passes as `--index`. Its **label shards were deleted** (disk, 2026-07-27); so
> were the `value/` and `value_v3/softmin_*` weights, whose `diagnostics.json`
> (the measured record cited below) were kept.
>
> Generating a set costs ~30 MB/organism and ~1 h/organism at 10-way concurrency;
> it dies mid-shard on a full disk, and the resulting partial shard must be
> deleted, not resumed.
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
> **The gate is measured in `u = c/(Km+c)` space, never in the network's input
> space** (`cfs.surrogate.train._du`). A cosine taken in the model's own input
> coordinate is a different number for every input transform: it cannot be
> compared between runs, and it improves for free when you change the transform.
> The first M3 checkpoint reported 0.72 in its own coordinate and scores **0.09**
> in `u` — the number the master problem and HMC will actually feel. Both the gate
> and the Sobolev term now live in `u`.
>
> **What shapes M3 and will shape M4.** On a real GEM `mu_max` is a *linear ramp*
> in saturation `u = c/(Km+c)` that reaches its plateau by `u* ~ 1.4e-4` for the
> ions: 99.9% of the input range carries no signal, and `d(mu)/du` spans five
> decades within one organism. What is load-bearing, in the order it was found:
>
> - a **saturating** per-metabolite input rescale `x = u/(u+s)`, `s` = median `u`
>   where that metabolite limits (`_kink_scale`, no floor). A *linear* rescale
>   cannot work: reaching the ions' ramp at `u ~ 1.4e-4` sends replete dims to
>   `x ~ 1e4`, and the floor that stopped that pinned **45%** of dims short of
>   their kink. Composed with the MM map it is just a smaller effective `Km`;
> - the head is **monotone non-decreasing** (`wx`, `out_x` = `-softplus(param)`).
>   Required — the input map is concave, so `f(h(x))` is concave only if `f` is
>   non-decreasing — and true of the target: relaxing an uptake bound can only
>   enlarge the LP's feasible set. It converges ~3x slower than the unconstrained
>   head, so a short run looks like a regression when it is a budget;
> - a **per-row norm-relative** Sobolev term instead of §7.1's absolute
>   `||grad - pi||^2` (absolute gave cosine 0.05: a few ion-limited rows absorbed
>   the whole term), with the all-zero-target rows (23%) *included* — dropping
>   them is what let the model spend most of its gradient magnitude on
>   non-limiting metabolites for free;
> - ICNN init at `softplus^-1(1/width)`.
>
> Measured in `u` space, worst/mean cosine: old linear-rescale checkpoint
> 0.06/0.09 → saturating transform + monotone head 0.41/0.69 → same, trained on
> the `u`-space objective for 1500 epochs 0.63/0.77 → **same model, same
> hyperparameters, banded labels 0.733/0.800**. Predicted gradient magnitude
> landing on zero-target metabolites fell 98% → 53% across that sequence, and the
> Hessian condition number 7e11 → 3.4e7.
>
> **The gate was label coverage before it was architecture.** Held-out cosine per
> (organism, metabolite) cell tracks that cell's *training row count* at Spearman
> 0.72 (<50 rows → 0.49, >400 → 0.95), and the old design's counts were skewed
> ~1137 : 9 : 1 per organism because every metabolite got the same
> `log10(c/Km) ∈ [-4, 1]` band while real limiting points span 5107×. Fixing that
> alone bought +0.10 worst cosine and 60× conditioning — but the tail did not move
> (p05 0.21 → 0.20), the same ions still lead the error (`EX_mg2_e` 21/21 organisms,
> `EX_cl_e` 19, `EX_ca2_e` 17), and value R² fell 0.72 → 0.54. Also rejected: a
> soft-min ("Liebig") head that matches the target's sparsity exactly scored
> *worse*, 0.55, with 6–16 orders worse conditioning (`20hm/value_v3/`).
>
> **§4.7 is done — bands anchor themselves.** `active_subspace.demand_probe`
> bisects `log10(c/Km)` per (organism, metabolite in `A_i`) for the point where
> `mu_max` has recovered `target_frac` of that metabolite's own range, holding
> every other uptake at the *design's* rich level (`Km * 10**log10_hi`, not
> `_C_RICH`). `design.band_scales` then resolves probe → previous `u*` (`--scales`,
> now a fallback) → roster median → 1.0 and records the choice per metabolite in
> `<id>.subspace.json`. On by default (`SamplingConfig.probe`), ~2 s and ~350 LPs
> per organism, and it needs no labels — a new GEM anchors itself.
>
> `target_frac=0.1` is **calibrated against the measured `u*`** (median `log10`
> difference over 4 organisms / 64 metabolites: −0.15 at 0.05, +0.09 at 0.1, +0.77
> at the range midpoint) — do not "simplify" it to the midpoint, which shifts every
> band 0.8 decades above the regime the labels actually found. At 200 media the
> probe path reproduces the `--scales` path's coverage exactly (`top_share` 0.463
> vs 0.466), with no previous run.
>
> **§4.6 top-up loop, in the order the signals are worth spending.** `cfs topup`
> reads `train-value`'s held-out `per_limiting_metabolite` → focus weights (
> `design.topup_weights`, plus the floor share for the metabolites
> `ensemble.unmeasured_metabolites` finds never limited); `cfs generate
> --focus-weights --round N` spends them into `part.round<N>.parquet` beside the
> base shards. `load_value_dataset` globs every parquet in an `(organism, eps)`
> dir, and round `N` offsets `medium_id` by `N * 1e6` because the train/val split
> is by medium. Loop: train → topup → generate round → retrain, ~20% of the base
> budget per round, stop when the worst cell stops moving.
>
> ### The gate is **not** a sampling problem — measured, 2026-07-28
>
> The full roster ran on probe-anchored bands (`20hm_probe/`: 21/21, 63/63 shards,
> 940 800 rows, 100% optimal, one `index_hash`; band sources **377 probe / 16
> roster-median / 3 previous / 100 default**). Coverage matched or beat the
> two-pass set — roster median metabolite 174.5 vs 173.5 media, `A_ge50` 17 both,
> `top_share` 0.597 vs 0.606, `EX_mg2_e` 122 vs 110 — **with no previous run**.
> Then two top-up rounds:
>
> | run | worst cosine | mean | R² |
> | --- | --- | --- | --- |
> | `value_b1` two-pass bands (baseline) | 0.733 | 0.800 | 0.538 |
> | `value_p1` probe bands | 0.695 | 0.783 | 0.548 |
> | `value_p2` + round 1 (20% budget, `1-cos` over ~35 mets) | 0.696 | 0.780 | 0.517 |
> | `value_p3` + round 2 (40% budget, **80% on the 5 worst cells**) | **0.648** | 0.771 | 0.524 |
>
> ~~**More media where the model is worst makes it worse.**~~ **RETRACTED,
> 2026-07-29 — the p1→p3 table above was scored against a moving ruler.**
> `load_value_dataset` permuted *all* media into the train/val split, and
> `_organism_arrays` globs every parquet in an `(organism, eps)` dir — including
> each `part.round<N>.parquet`. So §4.6 top-up media, drawn *deliberately* from the
> regions the model is worst at, landed in the **validation** set too and the
> held-out set got harder every round. Measured on one organism's usable held-out
> rows: **p1 491 → p2 595 → p3 781**. Three different tests, three numbers, read as
> a trend. The conclusion that top-up hurts is **not established**; neither is the
> claim that the coverage↔cosine relation is merely correlational.
>
> Fixed in `cfs.surrogate.data.load_value_dataset`: **val is a seeded sample of
> round-0 media only**, top-up media are appended to train, and `diagnostics.json`
> now records `n_val_media` / `n_train_media` / `rounds_present` so two runs can be
> checked for comparability. Top-up media are appended rather than mixed into the
> permutation, so a round-free label set reproduces the old split exactly —
> verified: a re-run of `value_b1` reproduces 0.7328/0.8005/0.5383 and every
> per-organism `EX_mg2_e` cell bit-identically. The round shards were deleted in
> the 2026-07-27 cleanup, so p2/p3 cannot be retro-scored; the fix goes forward.
> Regression test: `tests/test_cfs_data_split.py`. **Re-running the top-up
> experiment on the fixed ruler is open work, not a closed question.**
>
> ### Architecture trial — measured, 2026-07-29
>
> All on `20hm_bands/`, identical knobs (128/3/1500/512/3e-3/`w_grad` 1/`eps` 1e-3/
> seed 0) and the *same* 800 held-out round-0 media, so only the architecture
> varies. `cfs train-value --arch {icnn,deepset,deepset-private,mlp}`, plus
> `cfs baseline-rf`.
>
> | run | worst | mean cos | R² | Hessian cond | train loss (value) |
> | --- | --- | --- | --- | --- | --- |
> | `value_b1` icnn (baseline) | **0.733** | **0.800** | 0.538 | 3.4e7 | 0.62 (0.375) |
> | `value_icnn_b64` icnn, batch 64 | 0.666 | 0.793 | 0.547 | 5.6e9 | — |
> | `value_ds1` deepset, shared trunk | 0.259 | 0.432 | 0.111 | **4.0e3** | 1.35 (0.895) |
> | `value_dsp1` deepset, private trunks | 0.374 | 0.601 | 0.388 | 1.2e5 | 0.99 (0.626) |
> | `value_mlp1` unconstrained MLP | 0.243 | 0.768 | 0.797 | 7.1e10 | 0.37 (0.216) |
> | `value_rf1` random forest | (probe-limited, see below) | | **0.979** | n/a | n/a |
>
> **The ICNN is still the best head. Four things were learned, in order of how much
> they should change what happens next.**
>
> 1. **The value function is almost perfectly learnable, and the ICNN is nowhere
>    near it.** The forest scores R² **0.959–0.996 on every one of the 21
>    organisms** against the ICNN's 0.30–0.80. R² 0.538 is an *architecture*
>    deficit — not a label ceiling, not solver noise, not sampling. Nothing before
>    this trial could distinguish those.
> 2. **The concave family is not the ceiling either.** Dropping both constraints
>    (`mlp`) moves the median cosine +0.009 — nothing — while the worst organism
>    collapses 0.733 → 0.243 and 43.8% of Hessians go non-concave. The constraints
>    are close to free on gradient accuracy and are what holds the worst case
>    together. **P11's difference-of-convex escape hatch does not look worth its
>    complexity**, and relaxing structure is not the route to 0.99.
> 3. **The ions are an architecture failure, not an intractable cell.** The forest
>    reads `EX_mg2_e` at ~1.000 on 20/21 organisms where the ICNN manages 0.21–0.98
>    (and the MLP goes *negative* on six). Mg limitation is an axis-aligned kink in
>    one coordinate: native to a tree split, and evidently not localisable by a
>    dense smooth net of width 128 spread over 444 inputs. The remaining error is
>    **localisation**, which is the thing to attack.
> 4. **Sharing across organisms hurts — D1's supersession is not worth taking.**
>    `deepset-private` beats `deepset` on every metric (R² 0.111 → 0.388, worst
>    0.259 → 0.374). The 21 organisms' ramps are not one function; a shared `phi`
>    spends its capacity reconciling them rather than pooling evidence.
>
> **The deepset caveat, stated because it is a confound and not a footnote.** `phi`
> is priced *per metabolite*, so at the design width (`width // 2` = 64) it cost
> 123x the ICNN's FLOPs — 47 s/epoch against 0.35, i.e. 19 h a run. It was cut to
> `width // 8` = 16 **for speed**, and both deepsets then *underfit the training
> set* (value loss 0.895 and 0.626 against the ICNN's 0.375). More trunk capacity
> demonstrably helps (that is result 4's mechanism), so **the deepset numbers above
> are a lower bound and do not cleanly refute the architecture.** Restoring
> hidden=64 is ~11 h/run even sharded. It was not judged worth a day given the
> ICNN reaches 0.538 with 4.29M parameters at 0.35 s/epoch — but the honest status
> is *not measured at full width*, not *refuted*. The one clear deepset win is
> **conditioning: 4.0e3 against the ICNN's 3.4e7**, four orders better and directly
> what §8's Newton composition needs; worth revisiting if M4 turns out to be
> conditioning-limited rather than accuracy-limited.
>
> **The forest's gradients are probe-limited — do not quote an aggregate.** It has
> no analytic gradient, so `baseline.py` uses central finite differences at
> `delta * s_m` in `u`. The worst-organism cosine is **U-shaped in delta** (0.398 /
> 0.451 / 0.374 / 0.209 / −0.102 / −0.023 at 0.01/0.02/0.05/0.25/0.5/1.0) and the
> two ends hit different metabolites: carbon sources invert with a large step
> (`EX_cytd_e` 0.883 → −0.729), ions fall off with a small one (`EX_mg2_e` 0.824 at
> 0.01). An earlier reading here — "the forest loses the carbon sources to
> non-monotonicity" — was **wrong**: those negatives are the probe, not the model.
> Result 3 survives only because `EX_mg2_e` sits on a *plateau* (0.998/0.996/0.993/
> 0.982 across 0.05→1.0). Quote a per-cell number only where it is flat in delta.
> Default is now 0.05; R² 0.979 is delta-independent throughout.
>
> **Next — M3b, the HPC sweep (plan §7.4).** Everything above is one laptop, 4000
> media/organism (1/5 of D10), width 128, depth 3. The models **underfit**: the
> deepsets on training loss outright, the ICNN by inference (a forest reaches R²
> 0.98 on the same rows). Neither more rows nor more width has been tried, so that
> is the sweep — (1) D10's 20000 media/organism, (2) ICNN width {128…1024} × depth
> {3,4,6}, (3) **deepset at full width and beyond** (`phi` hidden {32,64,128},
> `k_code` {16,64,128}, `--emb-dim` {8,16,32}) — that arm was cut to hidden=16 for
> laptop runtime, it has the localisation bias result 3 says is worth having, and
> its per-metabolite cost is exactly what needs a cluster — (4) keep `mlp` and
> `baseline-rf` in as ceiling and floor; both changed how the ICNN's numbers read.
> Gate: any cell with R² ≥ 0.9 *and* worst cosine ≥ 0.9. If R² does not move with
> 5× rows and 8× width, localisation is confirmed and the answer is a
> per-metabolite architecture, not scale.
>
> **The pipeline for it is built and end-to-end verified** (`--stage sweep`,
> `workflows/value_sweep.nf`, `examples/value_sweep/`). One task per row of a
> samplesheet (`cell_id,arch,labels,args`, where `args` is the cell's literal flag
> string and `arch=rf` routes to `baseline-rf`), collected into
> `sweep_leaderboard.csv`. `examples/value_sweep/sweep_full.csv` is the four arms
> above as 54 cells; `make_sweep.py --demo` self-checks the generator. Two traps
> baked into the modules: `cfs train-value` **exits 1 whenever the gate is unmet**,
> so `TRAIN_VALUE` tolerates that and gates on `diagnostics.json` existing instead
> (read `passed`, not the exit status); and `params.xla_devices` must **divide** the
> organism count. Arm 3 needed `--phi-hidden`/`--k-code` on the CLI — they were
> hardcoded at `width // 8` and `16`, and the defaults still are.
>
> Sharding note for the cluster: the organism axis splits cleanly via
> `train._shard_organisms` (1.8× on CPU with
> `XLA_FLAGS=--xla_force_host_platform_device_count=N`, N must divide the organism
> count); on GPU the `filter_vmap` stacking is the §6.1 win. The §4.7/§4.6
> machinery stays: it removed the two-pass dependency and it is how a *new* genome
> gets labelled at all.
>
> Loss weights are still not the answer: `w_grad` only trades the heads (1 → 0.63
> cosine / 0.72 R², 10 → 0.66 / 0.62) and neither end reaches the "R² ≥ 0.9"
> balance rule. Checkpoints: `20hm_bands/value_b1/` (best worst-cosine to date,
> 0.733), `20hm_bands/value_{icnn_b64,ds1,dsp1,mlp1,rf1,rf_d*}/` (this trial),
> `20hm_probe/value_p{1,2,3}/` (probe bands + top-up rounds — **scored on the
> moving ruler; not comparable to anything above**), `20hm/value_v2/{u_w1,u_w10}/`
> (pre-band baseline).

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

`main.nf` has four stages via `--stage` (default `train`):

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
  CLI: `cfs generate`. Stub: `tests/labels.nf.test`. `--label_probe` (§4.7 demand
  probe, on by default), `--label_scales`, `--label_round` and
  `--label_focus_weights` (§4.6) are all wired; the last two are staged files, so
  `GENERATE_LABELS` builds those two flags itself rather than from `ext.args`.
- **`sweep`** — M3b/§7.4 Head A sweep (`workflows/value_sweep.nf`). One task per row
  of `--sweep <sweep.csv>` (`cell_id,arch,labels,args`): `TRAIN_VALUE`
  (`cfs train-value`) or, for `arch=rf`, `BASELINE_RF` (`cfs baseline-rf`) →
  `COLLECT_VALUE_METRICS` → `${outdir}/sweep_leaderboard.csv`. Needs `--index`;
  needs **no `--roster`** — the only stage that reads labels rather than GEMs, which
  is why the roster check in `main.nf` is stage-aware. The per-cell knobs live in the
  samplesheet, not in params, so there is no param per sweep axis; `params.sweep` and
  `params.xla_devices` are the only two. Worked example plus generator:
  `examples/value_sweep/`. Stub: `tests/sweep.nf.test`.
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
| `TRAIN_VALUE` | `surrogate-mgem-train` | process_high |
| `BASELINE_RF` | `surrogate-mgem-train` | process_medium |
| `COLLECT_VALUE_METRICS` | `surrogate-mgem-train` | process_single |
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
  (`ghcr.io/timrozday-mgnify/surrogate-mgem-{train,data}:0.1.3`). Bump the tag in
  every module together. The **train image carries `.[jax]`** (M3 Head A) as well as
  torch — including `pyyaml`, which `load_value_dataset` needs for
  `km_defaults.yaml` and which was only in the `data` extra until the sweep hit it.
  No `-sif` ORAS artifacts are published, so the
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
  (design doc §5.4). Do not "simplify" it back to one solver. The **LP is GLPK**
  (cobra's default here, despite the above) and GLPK's simplex can cycle forever
  on a near-degenerate medium — one organism burned 4.5 h at 100% CPU with a
  frozen log. Both LP and QP carry `_QP_TIME_LIMIT`; the LP's needs `int()`
  because optlang feeds it to glpk's integer `tm_lim`.
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
