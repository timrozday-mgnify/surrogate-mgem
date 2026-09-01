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
> ### Progress — M0–M2 and §4 done on the real roster; M3 at 0.958 of a 0.99 gate
>
> | Milestone | State | Code |
> | --- | --- | --- |
> | M0 QC + frozen index | **done, V0 passes** — 21/21 EGC-free; index = 444 exchanges, **365 shared / 79 private** (so the Newton Jacobian is 365×365) | `src/cfs/groundtruth/{qc,index}.py` |
> | M1 degeneracy → D4 | **done, V1 complete** — **68.9%** of 371k exchange-FVA observations degenerate roster-wide (59.7–82.8% per genome; 88% at α=0.7 vs 50% at α=1.0) ⇒ **D4 = elastic net** | `src/cfs/validate/degeneracy.py` |
> | M2 solve interface | **done, §3.4 gate verified on a real GEM** — MM uptake bounds (§3.3), FBA for `mu_max` + duals, then the **Clarabel** elastic-net QP for `z` | `src/cfs/groundtruth/solve.py` |
> | §4 sampling + bulk labels | **done, generated** — active subspace, stratified-Sobol design, parquet driver | `src/cfs/sampling/`, `--stage labels` |
> | M3 Head A (value) | **gate not met; roster-wide worst is 0.958 against 0.99.** Three causes, each measured and each fixed: concavity imposed in the wrong coordinate (`x`, not `u`), random initialisation collapsing the max-affine head, and planes going *dead during training* and being unable to revive. Seeded `groupmax-u` + `--gm-reanchor 3`, 21 organisms: worst cosine **0.958**, median 0.978, median R² 0.995 | `src/cfs/surrogate/{picnn_u,deepset_u,groupmax}.py` |
> | M4 Head B (behaviour) | **done** — `z_i(c, alpha)` masked MLP predicting **specific** flux `z / mu_max`, worst held-out R2 **0.883** / median 0.941, worst flux cosine 0.995. The `z` parametrisation cost the M5 tail: see below | `src/cfs/surrogate/behaviour.py` |
> | M5 dFBA composition | **built and measured over replicates; the 1% gate is not met and the regression is open.** Median log-X over n=5 replicates x 10 communities: **0.009 / 0.018 / 0.076 / 0.027 / 0.027** at sizes 2/3/5/10/21 on the pre-relabel labels (`r1`), but **0.014 / 0.067 / 0.029 / 0.093 / 0.466** after the `probe_lo` relabel — and three design fixes have not moved it (`p3` 0.783, `p4` 0.706 at n=21). Error does not grow with community size; the failure is one over-predicted member at one medium. Stock-take and ranked plan: **design spec §8.5** | `src/cfs/compose/dfba.py`, `src/cfs/surrogate/calibrate.py` |
> | M3b HPC sweep | **two runs. 2026-08-25** (350 tasks, 324 cpu-h) refuted its own "scale closes the gap": width/depth inert. **2026-08-27** (442/442 tasks, 21 cells x 21 organisms) is the source of the roster numbers below. The rows arm has now failed to run three times | `examples/hpc_run/`, `--stage sweep` |
>
> | M9-M14 Phase 7 applications | **spec written; M9, M10 and M11 done** — `cfs simulate` integrates a community forward, batch or chemostat (`--dilution`), surrogate only, no LP; `cfs maximise-growth` is §13.2's convex medium design, 19/20 V5 round-trips at the default trust region, median true gain +2.2%, median optimism 0.3%. `cfs minimal-medium` is §13.3's convex program, and V6 does not pass: Head A cannot represent essentiality (below). The rest (steady state, interaction maximisation, the inverse-problem posterior) is specced in **§13** with a per-use-case accuracy table: most need less than M3's and M5's unmet gates | `src/cfs/compose/dfba.py` (`simulate`, `with_chemostat`), `src/cfs/science/growth.py` |
>
> **The label set** (M3/M4 train on this): `~/Documents/surrogate-mgems_runs/20hm_bands/`
> from `~/Documents/20hm_carveme_models` (21 CarveMe GEMs). 4000 media/organism —
> 1/5 of the D10 budget, laptop-sized; `SamplingConfig.n_media` still defaults to
> 20000 for HPC. 21/21 organisms, 63/63 shards, **100% optimal solves**, one
> `index_hash` throughout (P13). Per organism: 32 000 rows at the primary
> `eps=1e-3` and 6400 at each of `1e-2`/`1e-4`. `|A_i|` = 11–32, median 24.
>
> Four label roots now live under it: `labels` (= `r1`, pre-relabel), `labels_p2`
> (`probe_lo = -12` + the low-`mu` stratum), `labels_p3` (P24 budget, reverted in code)
> and `labels_p4` (`focus_bg_decades`, kept). Each has base + a round-1 community pass,
> 63/63 shards, 100% optimal, one `index_hash`; every §8.1 comparison in this file is on
> the identical community list and media. See the design spec §8.5.
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
> varies. `cfs train-value --arch {icnn,icnn-u,deepset,deepset-private,mlp}`, plus
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
> **The deepset caveat above was real, and it is now discharged — see "Deepset is
> measured at full width, and cut" below.** `phi` is priced *per metabolite*, so at
> the design width (`width // 2` = 64) it cost 123x the ICNN's FLOPs — 47 s/epoch
> against 0.35, i.e. 19 h a run. It was cut to `width // 8` = 16 **for speed**, and
> both deepsets then *underfit the training set* (value loss 0.895 and 0.626
> against the ICNN's 0.375), so the two rows in this table are a lower bound and
> never refuted the architecture. The full-width rerun did, on 21 organisms.
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
> ### The M3b sweep ran, and located the deficit — measured, 2026-08-26
>
> `~/Documents/surrogate-mgem_runs/hpc_run/export/`: 350 completed tasks over 21 of
> the 30 planned cells, one task per (cell, organism), 324 cpu-h. No
> `sweep_leaderboard.csv` — aggregate `sweep/*/diagnostics.json` directly. All 6
> shared-`deepset` cells OOM-killed at ~30 s (exit 137) and `deepset-private
> ph64/kc64` hit the 12 h walltime (exit 140); 72 tasks aborted downstream. The
> four `deepset-private` cells were rerun and **all four now hold 21/21 organisms**
> (462 task dirs in `export/sweep_out/sweep/`), which is what the deepset verdict
> below rests on.
>
> **The rows axis was not tested.** The three `m4k` samplesheet rows still carry
> the literal `/path/to/20hm_bands/labels` placeholder, every task reports
> `n_train_media=16000`, and `m4k__rf` is bitwise identical to `m20k__rf` on all 21
> organisms. Use the local `20hm_bands` runs as the 4000-media reference.
>
> **ICNN capacity is a null axis.** Width 128→1024 × depth 3→6 is 37x the
> parameters; paired per-organism deltas against `w128 d3` are ±0.001 on *every*
> one of the 12 cells (same organism, w128→w1024: cos 0.71503→0.71514, R²
> 0.47738→0.47747). Width buys only conditioning: median log10 Hessian cond
> 9.8 → 3.9. Paired 4k → 20k, the ICNN gains +0.01 R² while `mlp w512` gains
> **+0.16** (0.752 → 0.915) and `deepset-private` +0.16 — so the ICNN is neither
> data- nor capacity-limited.
>
> **Why: concavity was imposed in the wrong coordinate.** `mu_max` is an LP value
> function in its RHS and `lb = -Vmax * u`, so it is exactly concave and piecewise
> *linear* in `u` — but `u = s*x/(1-x)` is **convex** in `x`, so a head locked to
> concavity in `x` can only fit each ramp with a chord. Tangent test on the labels
> themselves (1500 row pairs/organism): violated in `x` on **32.3 / 39.7 / 56.9%**
> of pairs (CR626927.1 / CP001820.1 / ABCC02), in `u` on **0.0%** of all three.
> `{concave in x}` is a proper subset of `{concave in u}` and the target sits in
> the gap — every ICNN variant converges to the same projection, which is what
> capacity cannot move. The 0.0% column also validates the dual handling.
>
> **The class is right once stated in `u`.** The parameter-free cutting-plane model
> `mu_hat(u) = min_j [mu_j + pi_j.(u - u_j)]` over the training rows, scored by
> `train.score` on the same held-out media: cosine **0.969-0.996**, R²
> **0.997-0.999**, top1-share 0.82-0.97 on 6/6 organisms, `GCA_000007325.1`
> clearing the 0.99 gate outright. Against the sweep's best on the same media —
> ICNN 0.715/0.477, `mlp w512` 0.824/0.909, rf 0.817/0.987. It wins on both axes at
> once, and unlike the forest it is concave, monotone and analytically
> differentiable. **This is the new ceiling measurement; retire the forest for it.**
> **It is not usable in §8 as it stands** — a min of affine functions is exactly
> concave, but its Hessian is identically 0 inside every piece and undefined at the
> kinks, which is P3 verbatim ("zero Hessian → Newton stalls or NaNs, gradients look
> fine"). §8.4 wants a Jacobian that is a *sum of PSD Hessians* under
> `lx.positive_semidefinite_tag`; from this model that sum is the zero matrix, and
> the kinks break the smoothness HMC needs downstream. The fix is the log-sum-exp
> smoothing `-T * logsumexp(-z_j / T)`: still exactly concave and monotone, but
> C-infinity, with Hessian `A^T (diag(p) - p p^T) A / T` — PSD, and with curvature
> ~1/T, so `T` is an *explicit* conditioning knob rather than an emergent number,
> the same homotopy pattern §5.4 already uses for `eps`. The accuracy/conditioning
> trade-off over `T` is unmeasured.
>
> **`cfs train-value --arch icnn-u`** (`src/cfs/surrogate/picnn_u.py`) is that
> correction: the identical ICNN fed `w = min(u/s, 300) = min(x/(1-x), 300)`, which
> is *affine* in `u` per metabolite, so the class is the full concave-in-`u` one.
> Measured on CR626927.1 at the sweep's exact knobs (w128/d3/1500/512/3e-3/w_grad 1):
> **R² 0.477 → 0.802**, training loss 0.62 → 0.358, cosine flat at 0.710, concavity
> violations 0. Three things are load-bearing and were each measured:
>
> - **`W_CAP = 300`, and capping lower is actively harmful.** A limiting cell above
>   the cap gets zero predicted gradient on the one metabolite that matters:
>   cutting-plane cosine on ABCC02 is 0.432 at cap 30, 0.793 at 100, 0.9816 at 300,
>   0.9817 uncapped. 300 costs ≤0.001 against uncapped and shrinks the range 25x.
>   Only an *affine* rescale is admissible — any concave squash of `w` (the `x` map
>   included) reintroduces the original defect.
> - **A scale-aware init is required.** The parent's `sqrt(2/n_in)` assumes `x` in
>   [0,1]; on `w` it starts the head at initial loss **1.7e6** with median Hessian
>   condition exactly 0 — softplus is affine out there, so the net begins *linear*
>   and Adam spends the run walking the bias down. Dividing the input weights by
>   `W_CAP/2` gives initial loss 41.6 and is what turns R² 0.66 into 0.80.
> - **The diagnostics must move with the constraint.** `concavity_violation_rate`
>   and `hessian_cond_median` are taken in the head's own coordinate via the
>   optional `to_diag`/`batched_value_diag`/`head_in_diag` hooks; read in `x`,
>   a correctly concave `icnn-u` reports 98% violating and cond ~1e32. And the
>   diagnostic must not reach `w` by round-tripping through `x` — `1-x` cancels in
>   float32 exactly across the replete far field (~45% of cells). Heads without the
>   hooks see the identity, so every existing arch is byte-identical.
>
> **`icnn-u`: capacity is still inert, but the `w_grad` frontier moved wholesale.**
> Width 128 → 512 at `w_grad` 1 changes nothing (cosine 0.7095 → 0.7114, R² 0.8024
> → 0.8025), so the hypothesis that the `x` constraint was what suppressed the
> capacity axis is **wrong** — something second and independent pins width. But the
> `w_grad` sweep (CR626927.1, w128/d3/1500/512/3e-3, dataset loaded once,
> `w_grad` 1 reproduces the standalone run exactly):
>
> | `w_grad` | cosine | R² | p05 | top1 | Hessian cond |
> | --- | --- | --- | --- | --- | --- |
> | 0 | 0.390 | **0.821** | 0.000 | 0.375 | 9.6e11 |
> | 0.3 | 0.537 | 0.811 | 0.000 | 0.516 | 3.6e14 |
> | 1 | 0.710 | 0.802 | 0.002 | 0.692 | 1.7e15 |
> | 3 | 0.816 | 0.787 | 0.128 | 0.796 | 6.2e15 |
> | **10** | **0.903** | 0.756 | **0.353** | 0.864 | 1.7e18 |
> | 30 | 0.753 | 0.511 | 0.003 | 0.710 | 2.0e19 |
>
> **An earlier note in this file — "`icnn-u` fixes the value head and nothing else"
> — is retracted.** It was measured at `w_grad` 1 only. Against the x-space ICNN at
> the *same* weight, 10 → 0.66 cosine / 0.62 R², `icnn-u` gives **0.903 / 0.756**:
> the coordinate fix moved the whole frontier, not just its value end. At the
> optimum it beats every model measured on this organism — `rf` 0.817, `mlp w512`
> 0.812, `icnn` 0.715 — from *inside* the concave class, violations still 0, and is
> closing on the cutting-plane's 0.969.
>
> **Everything in this subsection is retracted at roster scale — see "The second
> sweep ran" below.** `icnn-u` at `w_grad` 10 scores **0.494 worst over 21
> organisms**, and its whole `w_grad` x width grid is flat at 0.47-0.49. The
> coordinate fix stands; the frontier measured on top of it does not generalise.
>
> Three things this did **not** establish, written before that was known. It is
> **one organism**; the gate is the worst over 21. `w_grad` 30 collapses on both axes, and whether that is a real
> trade or an optimisation failure at fixed `lr` is untested (10-30 is unprobed).
> And the price is conditioning — 9.6e11 → 1.7e18 — so the accuracy the gradient
> term buys is being paid for in exactly the currency §8's Newton spends. The
> "loss weights are not the answer" verdict below was measured in `x` and does not
> carry over.
>
> **Capacity re-tested at the `w_grad` optimum, and it is still not the lever.** The
> width test above was at `w_grad` 1; redone at 10, where the gradient term binds
> (CR626927.1): width 128 → 0.9031 cos / 0.7563 R² / p05 0.353, 512 → 0.9088 /
> 0.7599 / 0.429, 1024 → 0.9067 / 0.7615 / 0.366. It peaks at 512 and turns over —
> +0.006 cosine for 16x the parameters. Train-vs-held-out gap is **0.005 cosine /
> 0.013 R²** throughout, so the head underfits with no variance to trade: this is an
> optimisation/inductive-bias limit, not a capacity or a label one.
>
> **Rows scale the *nonparametric* family and nothing else.** The cutting-plane
> model is coverage-limited — each labelled row contributes one dual vertex — and it
> does improve with K: CR626927.1 0.9305 (K=100) → 0.950 (1k) → 0.9621 (4k) → 0.9690
> (all 16k); ABCC02 0.8983 → 0.959 → 0.9711 → 0.9817. But `1-cos` only falls ~0.89x
> per doubling on CR626927.1 (~800x rows to reach 0.99) against ~0.75x on ABCC02
> (~4x). The gate is the worst organism, so **rows alone do not get there** — though
> p05 climbs 0.32 → 0.79 over 160x and is nowhere near saturated, so they do buy
> tail. K=250 tangents already reach 0.94, i.e. the tangent set is hugely redundant
> and a *parametric* max-affine should not need one piece per vertex. Better fitting
> dominates more rows.
>
> **`--arch groupmax-u`** (`src/cfs/surrogate/groupmax.py`) is the architecture bet
> that follows: the same `u`-coordinate ICNN with the activation changed from
> `softplus` to a **smoothed group max**, `T * logsumexp(a/T)` over groups of the
> pre-activation. A max of affine functions *is* the target's form, so one corner
> costs one unit instead of a sum of many soft bends — and depth compounds
> smoothness, which is why width/depth were inert. It nests plain max-affine exactly
> at `--width 1 --depth 1 --gm-group K` (asserted in `tests/test_cfs_value_head.py`).
> `--gm-temp` is the conditioning knob, not just an accuracy one: a hard max has zero
> Hessian inside a piece (P3), and curvature scales as `1/T`, so the accuracy-vs-
> conditioning frontier §8 must buy from becomes a swept axis. It is fixed, never
> learned — a learned temperature collapses toward the hard max (measured, cond
> 1e32). Related work: GroupMax (arXiv 2206.06622, motivated by Bellman *cuts*),
> Maxout (1302.4389), and LSPA/CAP/AMAP for max-affine fitting, which is the known
> fix for the dead-piece collapse a plain Adam max-affine hits (K=256 and K=1024
> scored identically).
>
> **Initialisation is the binding constraint, measured in a matched A/B.**
> `groupmax-u`, width 1 / depth 1 / K=250 / T=0.03 / `w_grad` 10, CR626927.1, same
> seed, 1500 epochs — *only* `--gm-init` differs:
>
> | init | cosine | R² | p05 | Hessian cond |
> | --- | --- | --- | --- | --- |
> | `labels` | **0.9733** | **0.996** | **0.834** | 1.9e24 |
> | `random` | 0.5982 | 0.4795 | 0.000 | **0.0** |
>
> The `random` run's Hessian condition of exactly 0 is the diagnosis: zero curvature
> everywhere means the head collapsed to a *single* affine piece — 249 of 250 planes
> never became active, so never got useful gradient. That is the dead-piece failure
> LSPA/CAP-style max-affine fitting exists to fix, and it is the same pathology as
> the earlier hand-rolled max-affine probe where K=256 and K=1024 scored identically.
> **The reasoning recorded earlier — that a fixed-temperature softmax gives every
> piece non-zero weight and therefore removes the problem — is wrong.** The remedy
> is not LSPA *for the initialisation*: those algorithms infer the planes from
> values, and we have the duals, so seeding them directly is simpler and strictly
> better. **But the claim that this makes the alternation step unnecessary is also
> wrong** — planes still die during training and cannot revive. See
> `--gm-reanchor` below.
>
> At K=250 the seeded head beats the **full 16 000-tangent cutting-plane model**
> (0.9733 vs 0.969, p05 0.834 vs 0.791) on 111k parameters, while staying concave
> (violations 0), monotone and analytically differentiable. Against the 0.99 gate it
> is the closest anything has come — on **one organism**.
>
> ~~**The bill is conditioning, and it is now the open problem.**~~ **RETRACTED,
> 2026-08-26 — 1.9e24 is real and does not reach §8.** It is `train._hessian_cond`,
> *one organism's* Hessian on its own dims; §8.4 inverts `J = sum_i X_i H_i +
> supply'`, and that is a different matrix. See "The conditioning bill is not §8's"
> below. Sharp pieces still buy gradient accuracy and cost per-organism curvature —
> the trade is just not one §8 pays, so `T` is chosen on accuracy alone.
>
> **Not the answer, on this evidence:** per-metabolite heads. `deepset` already is
> one (shared `phi` per metabolite, pooled, per-organism `rho`) and is under the
> same coordinate defect — `phi` is concave in `x_m`. Its mean-pooling also makes
> `d(mu)/dx_m = <rho'(S), dphi_m/dx_m>`, so the medium reaches the gradient
> *pattern* only through a `k_code`-wide vector, a narrow channel for what is an
> argmin across metabolites. It costs 5-11 h/organism against the ICNN's 5 min.
> Full-width numbers for it now exist and close the question — see **"Deepset is
> measured at full width, and cut"** below.
>
> **The second sweep has since run** — 442/442 tasks, and everything above that is
> one laptop organism is superseded by the roster numbers in "The second sweep ran"
> below. `examples/hpc_run/sweep_full.csv` is now 36 cells (arms A-C, E-G as run,
> plus H for `--gm-reanchor` and the seed axis and H' for `--w-rel`), regenerated by
> `make_sweep_full.sh`, which carries the measurement motivating each arm.
>
> ### The second sweep ran, and the gate is now a plane-death problem — 2026-08-27/28
>
> `~/Documents/surrogate-mgem_runs/hpc_run/export/`: 442/442 tasks, 21 cells x 21
> organisms, one `n_train_media=16000` throughout, `sweep_leaderboard.csv` present.
> Worst / median over the 21 organisms, held-out round-0 media:
>
> | cell | worst | med cos | med p05 | med R² |
> | --- | --- | --- | --- | --- |
> | `groupmax-u` seeded, w1/d1 K=1000, T 0.1 | **0.937** | 0.975 | 0.878 | 0.988 |
> | same, T 0.01 / T 0.03 | 0.934 / 0.930 | 0.975 | 0.87 | 0.997 |
> | `groupmax-u` seeded, w128/d3 grp8, T 0.03 | 0.911 | **0.984** | **0.943** | 0.990 |
> | `groupmax-u` seeded, w1/d1 K=100, T 0.01 | 0.744 | 0.956 | 0.692 | 0.972 |
> | `icnn` (x-space) | 0.691 | 0.817 | 0.049 | 0.435 |
> | `mlp w512` | 0.617 | 0.893 | 0.187 | 0.908 |
> | `icnn-u`, best of 9 cells | 0.494 | 0.864 | 0.050 | 0.736 |
> | `rf` | 0.357 | 0.664 | 0.000 | 0.994 |
>
> 1. **Seeded `groupmax-u` generalises; the one-organism 0.973 was not a fluke.**
>    19/21 organisms are >= 0.95 and value R² >= 0.975 on every one, concave,
>    violations 0.
> 2. **`icnn-u` at `w_grad` 10 does NOT transfer, and that claim is retracted.**
>    0.903 on CR626927.1 became **0.494 worst** roster-wide, and its whole
>    `w_grad` x width grid is flat at 0.47-0.49. The coordinate fix is real; the
>    `w_grad` frontier it was measured on is one organism.
> 3. **`K` is the lever, `T` is not.** K 100 -> 1000 is +0.19 worst cosine; T over
>    0.01-0.1 moves it 0.007 at K=1000. Pick T=0.03 and stop sweeping it.
> 4. **The rows arm never ran** — no `labels_out_4k` was staged, so all three `m4k`
>    cells are absent from the trace. Third attempt, third miss.
>
> **The residual is planes dying during training, not coverage, capacity or labels.**
> On a failing cell the model gets the limiting metabolite right on ~92% of rows but
> predicts `d(mu)/dw` of **1e-9** against a true 1.3e-3, while fitting the *value* on
> those same rows to 0.1%. Planes with the right slope are present — 59-210 of the
> 1000, seeded from those rows' own duals — but sit **0.2-1.5 above the active
> minimum**, i.e. 15-40x the temperature, so their softmax weight is 1e-3 to 1e-7.
> Since a plane's gradient *is* that weight, death is an absorbing state, and the
> Sobolev term's only channel to a dead plane is the same closed softmax. Train and
> held-out cosine agree on those cells (0.988/0.987, 0.521/0.484), so more media
> cannot help.
>
> **It roves with the seed, which makes the sweep's cell ranking noise.** At fixed
> hyperparameters over 5 seeds the collapse lands on `EX_o2_e` in one run (0.500) and
> `EX_thr__L_e` in another (0.613) — both well covered, both fine in the others.
> Per-organism cosine sd is **0.015**, against 0.007 between the leaderboard's top
> three cells. Never quote a single-seed `groupmax-u` cell.
>
> Two dead ends, so nobody re-chases them: `EX_12ppd__R_e` as a co-limiting partner
> (its dual is exactly 0 — it was argsort's first tie among zeros), and `W_CAP`
> clipping (26% of one failing cell's rows are clipped, but clipped and unclipped
> rows score the same, 0.688 vs 0.721).
>
> ### `--gm-reanchor` — the alternation step, measured 2026-08-28
>
> `cfs.surrogate.groupmax.reanchor` overwrites the lowest-softmax-weight first-layer
> slots with the worst-fit rows' label tangents, mid-training, Adam moments cleared
> for those slots. `--gm-reanchor N` spaces N passes over the run (default 0, off);
> costs <2% of runtime. A dead plane cannot revive itself, but an *installed* one is
> tangent at its anchor row and therefore at or below every other plane there, so it
> is alive on arrival.
>
> **This retracts the note that LSPA/CAP-style alternation is unnecessary "because we
> have the duals".** Having the duals makes the alternation step cheap — write the
> exact tangent instead of a least-squares refit — not redundant. Seeding fixed
> initialisation only; nothing was fixing drift.
>
> | measurement | baseline | `--gm-reanchor 3` |
> | --- | --- | --- |
> | AAXE02 x 5 seeds, worst / mean / sd | 0.927 / 0.945 / 0.0149 | **0.962 / 0.966 / 0.0029** |
> | AAXE02 x 5 seeds, worst p05 | 0.500 | **0.724** |
> | 21 organisms, seed 0: worst / median | 0.930 / 0.977 | **0.958** / 0.978 |
>
> **It removes the downside tail; it does not raise the typical organism.** 19/21
> organisms move by <0.003 and the median is flat — the roster gain is one organism
> (DACTBY01 0.930 -> 0.968, `EX_leu__L_e` **0.488 -> 0.969**, `EX_glu__L_e` 0.718 ->
> 0.982). That is the point: on any given seed only one or two organisms have a dead
> plane, and which ones is the seed's business. The variance collapse (sd 0.0149 ->
> 0.0029) is the more useful number than the mean. Value R² is unchanged to three
> decimals everywhere, and cells that were never dead are untouched.
>
> **The new worst organism is a different problem.** `GCA_000209935.1` sits at 0.958
> in both arms, led by `EX_ham_e` (1495 held-out rows, 0.945) — well covered, no dead
> plane, unmoved by re-anchoring. Same for `EX_arg__L_e` on AAXE02 (184 rows, 0.888
> in all ten runs). Whatever closes the last 0.03 is not this.
>
> **Next:** `examples/hpc_run/sweep_full.csv` is 36 cells; Arm H is
> `grp1000 T=0.01` x `--gm-reanchor {0,3}` x seed {0,1,2}, which is the smallest
> design that separates the pass from the seed noise on 21 organisms, and H' adds
> `--w-rel {0,0.3}` at `--gm-reanchor 3` over the same seeds. `T` moved 0.03 ->
> 0.01 for the reason in the low-mu section below.
>
> ### M4 + M5: the heads compose, and community error does not grow with size — 2026-08-28
>
> Head B and the §8.1 composition are built: `src/cfs/surrogate/behaviour.py`
> (`cfs train-behaviour`) and `src/cfs/compose/dfba.py` (`cfs community`).
> Checkpoints on `20hm_bands`: `value_ra3` (seeded `groupmax-u` w1/d1 K=1000
> T=0.03 `--gm-reanchor 3`, worst cosine **0.947**, median 0.966, value R2 median
> 0.986) and `behaviour_b1` (600 epochs, lr 1e-3, worst **R2 0.856**, median
> 0.921, worst median flux cosine 0.993, worst sign agreement 0.941).
>
> **Head B's binding constraint was output scaling, not capacity.** The head emits
> `z / z_scale` and the label scale is applied outside it. Predicting raw
> mmol/gDW/h instead scores held-out **R2 0.017**, worse than predicting the
> per-alpha mean; the same net on the normalised target scores **0.885** on the
> same organism. Exchange fluxes run to O(400) on the gases and O(1e-3) on the
> ions, so a `sqrt(2/n_in)` init starts ~400x short on the dimensions carrying the
> variance and Adam spends the run walking biases — the same failure
> `picnn_u`'s scale-aware init exists for. `z_scale` is in the checkpoint;
> `behaviour.flux` is the accessor. Do not "simplify" the head to raw units.
>
> **10 communities, sizes 2-21, `20hm_bands` media, 40 Euler steps, per-organism
> FBA as ground truth on the identical integrator/step/inoculum:**
>
> | size | dc/dt cosine | mu rel | log-X final | overgrowth (V5) | cross-feed |
> | --- | --- | --- | --- | --- | --- |
> | 2 (x5, median) | 0.983 | 0.016 | 0.055 | <=0.066 | 6/8 |
> | 3 (x2) | 0.847 | 0.083 | 0.122 | <=0.210 | 8/9 |
> | 5 | 0.915 | 0.133 | 0.322 | 0.182 | 10/11 |
> | 10 | 0.996 | 0.013 | 0.041 | -0.002 | 27/27 |
> | **21** | **0.997** | **0.014** | **0.044** | **-0.001** | **38/38** |
>
> 1. **Size is not the error axis.** The 21-member community is the *second most*
>    accurate run in the set and recovers every one of its 38 cross-feeding links
>    (a metabolite one member secretes and another consumes). Errors are
>    per-organism and largely independent, so they partially cancel in the pool
>    sum rather than compounding — which is the central bet of D1(a)+§8.1 and it
>    holds. 89/93 links recovered overall.
> 2. **A slow member is.** Every bad cell contains an organism with `mu0 < 1.3
>    h^-1` (GCA_000007325.1 at 0.42, AAXE02 at 1.23 on its drawn medium): the
>    2-member 0.866/0.215 row, the 3-member 0.695/0.219 row and the 5-member row.
>    Head A's held-out R2 is taken over each organism's *own* `mu` spread, so a
>    near-starving organism is a small absolute error and a large relative one,
>    and the composition integrates the relative one. **The next Head A signal is
>    accuracy at low `mu`, not the roster-worst cosine.**
> 3. **M5's 1% gate is not met, and the shortfall is the value head's, not the
>    integrator's.** Median log-X error is 4-5%: `d(log X)/dt = mu`, so a 1.4%
>    `mu` error over ~2 doublings integrates to ~4%. Closing it needs a better
>    `mu`, not a better ODE solver.
> 4. **P4 does not bite.** Re-solving the true LP at the state the surrogate
>    walked *itself* to (V5) gives `overgrowth <= 0.21` of initial `mu` and ~0 on
>    the large communities. The composition does not run away to a fictitious
>    fast-growing state.
>
> **Two traps this cost time to find, both now in the code.** A batch culture has
> two independent clocks — members doubling and the pool emptying — and a horizon
> set by the growth clock alone killed the true community at step 2 of 40, leaving
> two live points to score. `run` now solves for the *inoculum* instead
> (`dc/dt` is linear in `X`, so one probe solve fixes it) so the pool empties at
> the end of `--doublings`. And metrics are scored only while the true community
> is alive and normalised by fixed initial scales: a dead culture has `mu = 0`
> everywhere, where a per-step relative error divides by zero — the first version
> reported `nan` and a 237% `mu` error for a run whose live phase agreed to 2%.
> Concentration error is per metabolite relative to its own `c0`; a plain L2 over
> the pool is 0.7% on a trajectory where the limiting ion is gone in the truth and
> untouched in the surrogate.
>
> **Not measured yet:** §8.2 SteadyCom and §8.3 MICOM (this is §8.1 only, and
> deliberately — a joint community LP is a *different model*, so mixing it in
> would make a Head B error and a modelling choice indistinguishable); the
> `--steps` refinement check; abundances other than equal-split; and M6's implicit
> gradients through the Newton form.
>
> ### Head A over-predicts at low `mu`, and `T` is the lever — measured, 2026-08-29
>
> The M5 note above ("the next Head A signal is accuracy at low `mu`") is now
> measured. Held-out media binned by each organism's own `mu / max(mu)`, `value_ra3`
> (seeded `groupmax-u`, T=0.03, `--gm-reanchor 3`), 21 organisms:
>
> | band | rows | median rel err | median bias | grad cosine |
> | --- | --- | --- | --- | --- |
> | < 5% of max mu | 52 | 0.978 | **+0.978** | 1.000 |
> | 5-10% | 26 | 0.427 | +0.427 | 1.000 |
> | 10-25% | 29 | 0.188 | +0.188 | 1.000 |
> | 25-50% | 42 | 0.117 | +0.117 | 0.994 |
> | 50-75% | 33 | 0.065 | +0.065 | 0.943 |
> | > 75% | 586 | 0.012 | -0.012 | 0.959 |
>
> **100% of held-out rows below 75% of max mu are over-predicted**, and the error is
> pure bias — median |rel| equals median signed rel in every low band. It is
> invisible to every existing diagnostic: value R² is 0.986, and the *gradient*
> cosine in those bands is 1.000, better than on the plateau. `score` now reports
> `value_rel_err_low_mu` / `value_bias_low_mu` (median over rows below 25% of max).
>
> **It is not labels, coverage, or the plane budget.** The parameter-free
> cutting-plane model over the same organism's training tangents has bias
> **-0.000 in every band including <5%**, and so does a K=1000 subset picked by
> `rank_by_active_set` — i.e. the head's own *initialisation*. Training creates the
> bias. Two absolute offsets do it: (1) the softmin smoothing sits `~T*ln(K_active)`
> **below** the hard min — ~0.2 in `mu_scale` units at T=0.03, which is 4% at the
> plateau and >100% at a starving medium (the 1-epoch seeded head reads -1.207 at
> <5% and -0.040 at the plateau, exactly that shape); and (2) `_loss`'s value term
> is an absolute MSE with 74% of rows on the plateau, so Adam removes the offset
> where the rows are and lifts the bottom straight past the target.
>
> **`--gm-temp 0.01` fixes more of it than anything in the loss, and it is what §8
> feels.** Roster (21 organisms) and the same 10 communities as `community_c1`,
> per-organism FBA truth, seed 0 throughout:
>
> | run | low-mu bias | plateau rel | worst cos | med R² | log-X err, sizes 2/3/5/10/21 |
> | --- | --- | --- | --- | --- | --- |
> | `value_ra3` T=0.03 | +0.978 | 0.012 | **0.958** | 0.986 | 0.055 / 0.122 / **0.322** / 0.041 / 0.044 |
> | **`value_T01`** T=0.01 | +0.442 | **0.005** | 0.928 | **0.990** | **0.034 / 0.050 / 0.051 / 0.048 / 0.047** |
> | `value_T01_wrel03` `--w-rel 0.3` | +0.100 | 0.014 | 0.924 | 0.989 | 0.058 / 0.082 / 0.060 / 0.065 / 0.064 |
> | `--w-rel 1` | -0.002 | 0.031 | 0.944 | 0.986 | 0.095 / 0.101 / 0.085 / 0.140 / 0.138 |
> | T=0.003 | +1.305 | 0.018 | 0.880 | 0.979 | 0.150 / 0.555 / 0.293 / 0.394 / 0.352 |
>
> 0. **`groupmax.DEFAULT_TEMP` is now 0.01** (was 0.03), and `--gm-temp`'s help
>    text said 0.1, which was stale. Re-running an old checkpoint's settings needs
>    an explicit `--gm-temp`; the sweep cells all set it.
> 1. **T=0.01 is a sweet spot, not a direction.** It cuts the composition's worst
>    community from 0.322 to 0.051 log-X error and flattens M5 to ~5% at every size;
>    T=0.003 is worse than either on *every* axis, so do not read this as "sharper is
>    better". The price is worst grad cosine 0.958 -> 0.928 — ~2x the seed sd, one
>    seed, unrepeated.
> 2. **`--w-rel` (new) buys the bottom by selling the plateau.** It adds the same
>    value error measured relatively, denominator floored at 0.1 of the organism's
>    mean `mu` (no floor => plateau unweighted => R² -1.66). It removes the low-mu
>    bias outright at **no cost in grad cosine** (0.928 -> 0.924 -> 0.944 across
>    0/0.3/1), but the plateau carries most of what `d(log X)/dt = mu` integrates in
>    a big community, so composition gets monotonically worse there. Default 0.
>    Use it when slow members dominate; leave it off for large pools.
> 3. **The M5 conclusion is unchanged and sharpened**: with T=0.01 the log-X error
>    is ~5% at *every* community size, and the two bad cells from 2026-08-28 (the
>    3- and 5-member communities with a slow member) were that slow member's
>    relative error, not composition.
>
> **Two fixes that look obvious and are not.** *More low-mu labels*: the
> information is already there -- the cutting-plane model over the *existing*
> training tangents, and the K=1000 seeded init itself, have bias -0.000 in every
> band. Extra rows would enter through the same absolute MSE and only change the
> low-mu row *share*, i.e. a reweighting, which `--w-rel` does directly instead of
> ~1 h/organism of solves. *A harder softmax*: `--gm-temp-final` anneals `T` in 3
> geometric stages (`groupmax.with_temp`; `temp` is static, so Adam's moments have
> their metadata rewritten too). On 3 organisms it looked decisive -- 0.03 -> 0.003
> took the low-mu bias +1.845 -> **-0.009** with the plateau intact, beating
> `--w-rel` on both. **On 21 organisms it loses to fixed T=0.01 on every axis**:
> bias +0.315, worst cosine 0.899, community log-X 0.102 vs 0.047 at size 21
> (0.03 -> 0.01 is in between: bias -0.101, cosine 0.928, log-X 0.099). Kept and
> off by default, with the negative result on file. Third time a 3-organism
> frontier has failed to survive the roster -- do not promote one again.
>
> And the temperature cannot be made *per row*: `T*logsumexp(a/T)` is concave in
> `a` only for constant `T`, and exact concavity in `u` is what the head is for.
> Per-epoch is free; per-prediction is not. `--w-rel` already is the per-row
> growth-rate weighting -- `1/(mu + 0.1*mean mu)^2` -- and its scalar plus that 0.1
> floor are the shape knobs.
>
> Not done: multi-seed confirmation of the 0.958 -> 0.928 cosine cost, Head B
> retrained at T=0.01 (`behaviour_b1` is reused as-is above, which is fair since
> only Head A changed), and `--w-rel` on the HPC sweep.
>
> ### The low-mu bias is an output calibration, and it buys M5 — 2026-08-30
>
> `src/cfs/surrogate/calibrate.py`. Head A's low-`mu` over-prediction is a function
> of the **predicted value alone**: an isotonic map fit on the train rows and applied
> to held-out media drives every band's median bias to <=0.005 and *raises* R2
> (0.9898 -> 0.9901). So it is correctable after the fact, and nothing is missing
> from the labels — which is a second, independent refutation of "more low-`mu`
> media would help".
>
> `g(m) = a*m - d0*exp(-m/beta)` is increasing (`a, d0 >= 0`) and concave
> (`g'' < 0`), so `g(head(u))` stays exactly concave and non-decreasing in `u` —
> §8.4's PSD Hessian tag and `concavity_violation_rate` both survive — and the
> gradient is scaled by a positive per-row scalar, so **`grad_cosine` is
> bit-identical**. It is fit at the end of `train.run`, stored in the checkpoint JSON
> beside `mu_scale`, and applied where `mu_scale` is (`train.evaluate`,
> `compose.dfba.Surrogate.mu_and_z`), so every existing checkpoint deserialises
> unchanged and reads the identity.
>
> **The fit weight is the whole result, and it is set on the plateau — not on the
> band that motivated the work.** Residuals are divided by
> `max(|mu|, _W_FLOOR * max|mu|)`. Same 10 communities and media as
> `community_T01`, per-organism FBA truth, `value_T01` throughout:
>
> | median log-X error | n=2 | n=3 | n=5 | n=10 | n=21 | low-mu bias | plateau |
> | --- | --- | --- | --- | --- | --- | --- | --- |
> | uncalibrated | 0.034 | **0.050** | 0.051 | 0.048 | 0.047 | +0.446 | -0.005 |
> | `--w-rel 0.3` | 0.058 | 0.082 | 0.060 | 0.065 | 0.064 | +0.100 | +0.014 |
> | `_W_FLOOR` 0 (pure relative) | 0.046 | 0.074 | 0.061 | 0.098 | 0.098 | **-0.033** | -0.009 |
> | **`_W_FLOOR` 0.3 (default)** | **0.024** | 0.072 | **0.044** | **0.014** | **0.016** | -0.250 | **-0.002** |
>
> `median_mu_rel` at size 21 goes 0.009 -> 0.005 and R2/cosine do not move. **Sizes
> 10 and 21 are now 1.4% / 1.6% against M5's 1% gate**, from 4.7%. Size 3 is the one
> regression (0.050 -> 0.072).
>
> The `_W_FLOOR` 0 row is the trap: it removes the bias *outright* on every band and
> doubles the composition error. The map is downward-only, the plateau was already
> at -0.005, and `d(log X)/dt = mu` integrates the plateau, not the bottom — the same
> trade `--w-rel` makes, moved after training. **Tune this by the composition, never
> by `value_bias_low_mu`.**
>
> **Two things this retracts.** The mechanism in "Head A over-predicts at low mu"
> blamed the softmin offset; re-evaluating the *same trained head* at `T -> 1e-6`
> (`groupmax.with_temp`) makes the bias **worse** — +0.860 against +0.446 below 5%
> of max `mu` — so the smoothing gap is a *downward* offset that partially cancels
> it, and what is left is plane placement. That is what the class predicts: a min of
> tangents to a concave function is an upper bound everywhere, so positive bias is
> the only bias a max-affine head can have unless a plane sits tangent at that row.
> And the hope of running `--gm-temp 0.03` plus calibration to recover worst cosine
> 0.928 -> 0.947 is dead: `community_ra3_cal` is worse than `value_T01` on every
> axis (log-X 0.081 at size 21). T=0.01 stays.
>
> **Not the lever:** re-anchoring on relative error. `reanchor` ranks rows by
> gradient cosine and the low-`mu` rows score **1.000**, so they are never picked;
> re-ranked by relative over-prediction, one post-hoc pass moves +0.446 -> 0.313 at
> 30% of the planes and costs worst cosine 0.928 -> 0.901. Untested *inside*
> training, where planes can still settle.
>
> ### M5's residual is Head B's magnitude, and it is a reparametrisation — 2026-08-30

The M5 summary is a median per size, and it hides the shape of the failure. Across
all 10 communities the final log-X error tracks **`dc_rel`** — Head B's pool
derivative, scored on the *true* path — and not `mu_rel`, which is <= 3% everywhere:

| `dc_rel_median` | 0.19-0.79 (6 cells) | 1.1-2.2 (4 cells) |
| --- | --- | --- |
| `x_log_err_final` | 0.003-0.024 | 0.044-0.589 |

So M5 was Head A's problem only up to the point `--gm-temp 0.01` and the output
calibration fixed it. **The remaining error is Head B's**, and the size story is
cancellation: per-organism `z` errors are largely independent, so the 21-member
pool sum averages them away and a 2-member one does not.

**The failure is magnitude, not pattern, and it is medium-specific.** Per-organism
`z` cosine at the community media is 0.40-0.87 on the bad cells for organisms whose
*held-out* p05 is >= 0.93 — CP070062.1 scores 0.399 in one 2-member community and
1.000 in a 3-member one. At the failing point it predicted `|z| = 2780` against a
true `1040`: a replete organism's fluxes at a scarce medium (`mu` 25 where its
plateau is 39), while Head A had `mu` right to 1%. The community medium is **not**
out of distribution — every masked coordinate of it is inside that organism's own
training range on every failing cell.

**Exchange flux is nearly proportional to growth rate, and Head A already knows the
growth rate.** Measured on the labels: one constant per (metabolite, alpha) times
`mu_max` — a model with no inputs at all — explains a median **0.807** of the
held-out `z` variance (0.63-0.86 over the 21 organisms), against the trained
256x3 net's 0.921. Most of what Head B was learning was a magnitude it had to infer
from `x` and Head A predicts directly.

So Head B now emits **specific flux `z / mu_max`** and `compose.dfba` multiplies
Head A's `mu` back in, floored at `data._MU_FLOOR_FRAC` = 5% of the organism's mean
`mu` (1% of media have `mu_max` below 1% of the median, and dividing by those turns
the target into noise — the same trade `calibrate._W_FLOOR` makes). The floor is in
the checkpoint as `mu_floor`; a checkpoint without it reads as flux directly, so
`behaviour_b1` still composes.

`behaviour_zmu` (600 epochs, lr 1e-3, seed 0, otherwise identical to
`behaviour_b1`), same `value_T01_cal`, same 10 communities, same media:

| | n=2 worst two | n=3 med / worst | n=5 | n=10 | n=21 |
| --- | --- | --- | --- | --- | --- |
| `behaviour_b1` | 0.327 / 0.589 | 0.072 / 0.141 | 0.044 | 0.014 | 0.016 |
| **`behaviour_zmu`** | **0.064 / 0.351** | **0.041 / 0.078** | 0.073 | 0.014 | 0.016 |

Held-out Head B (scored against the labels' own `mu`, so this is pattern only):
worst R2 0.856 -> **0.883**, median 0.921 -> 0.941, worst sign agreement 0.941 ->
0.957. `mu_rel` is bit-identical everywhere — Head A did not move.

1. **It buys the tail and nothing else.** The four `dc_rel > 1` cells improve, the
   six good ones are unchanged to three decimals, and sizes 10/21 stay at 1.4% /
   1.6%. Worst community over the whole set: 0.589 -> 0.351.
2. **Size 5 regresses, 0.044 -> 0.073**, on the one cell whose `dc_rel` is 1.08 —
   right at the boundary. Its cross-feeding recall goes 0.91 -> 1.00 at the same
   time, so this is not a straight loss. Unexplained; one cell, one seed.
3. The residual is then the **direction** of `dc`, not its size — see the MM clamp
   below, which is the free half of it.

### Head B was violating §3.3's own uptake bound — 2026-08-30

The LP that made the labels cannot take up faster than `-Vmax_m * u_m`
(`solve.mm_lower_bound`), and **every exchange of every roster GEM has
`|lower_bound| = 1000`**, so that bound is a constant times the head's own input
saturation — no fit, no per-organism data, nothing to store. Head B has no such
constraint and breaks it: at M5's worst community, **28 of CP040530.1's 213
exchanges** are below the floor at once, by up to 186x (`EX_acald_e` -260 against
-1.4, `EX_glyc3p_e` -329 against -14) — and those are the same entries that lead
the `dc` error. `compose.dfba.Surrogate.mu_and_z` now clamps.

It is a projection onto a convex set the true `z` is already inside, so the
right-hand-side error cannot rise, and it does not: `dc_rel_median` falls on
**10/10 communities**. Same 10 communities, media and checkpoints throughout:

| median log-X | n=2 | n=3 | n=5 | n=10 | n=21 | worst cell |
| --- | --- | --- | --- | --- | --- | --- |
| `behaviour_b1` | 0.024 | 0.072 | 0.044 | 0.014 | 0.016 | 0.589 |
| `behaviour_zmu` (specific flux) | 0.024 | 0.041 | 0.073 | 0.014 | 0.016 | 0.351 |
| **+ MM clamp** | **0.011** | **0.035** | 0.086 | 0.014 | 0.016 | 0.408 |

**A strictly better right-hand side does not give a monotonically better
trajectory.** `dc_rel` improves on all ten, but two cells integrate worse (the
0.351 -> 0.408 worst cell, and size 5 0.073 -> 0.086) because the clamp changes
*which* metabolite empties first, and a batch culture's endpoint turns on that.
Seven of ten improve, sizes 2 and 3 halve, cross-feeding recall and `overgrowth`
(<= 0.028 everywhere) are unaffected. Keep the clamp: the rhs is the thing being
modelled, and the trajectory flips are a property of the map.

**Not done:** applying the clamp during Head B *training* (the targets already
satisfy it, so the net is currently spending capacity on outputs the LP cannot
produce), and re-fitting `calibrate` against the new Head B.

### Head B's remaining error is label coverage of the community medium

Measured on all 21 organisms: Spearman(distance to the nearest training medium in
`x`, `1 - flux cosine`) is **0.33-0.84, median 0.65**, and the top NN-distance
quintile's median cosine falls from ~0.999 to 0.89-0.99. Over the 10 communities
the split is clean — every cell with a member at NN distance >= 1.67 has
`dc_rel >= 0.90`, every cell whose members are all <= 0.31 has `dc_rel <= 0.84`,
and the four worst-log-X cells are exactly the four far ones (the worst is at
**2.92** against a held-out median of 0.10).

**It is a joint gap, not a marginal one.** Every coordinate of the failing medium
is inside that organism's own training range, and its count of scarce dimensions
(110) is typical (train median 103). What is missing is the *combination*: §4.3
samples `A_i` and holds the background at one rich level, while a community medium
is a draw over the **union** of the members' active subspaces, so a member sees
~30 of its background metabolites in bands at once.

**`SamplingConfig.frac_bg_perturb` was already the knob, and it was degenerate.**
`design.sample_media` perturbed the background all-or-nothing — 10% of media with
*every* held metabolite redrawn, 90% with none — so the design is bimodal in the
one axis the composition moves along, with nothing in between. It now draws a
random *share* per medium, which spans a 2-member community's ~20% and the whole
roster's ~all in one design, and `cfs generate --bg-perturb` exposes it. Test:
`tests/test_cfs_sampling.py::test_background_is_perturbed_over_a_random_share`.

### The community-regime round closes the tail and costs the two largest cells

`cfs generate --n-media 800 --bg-perturb 0.9 --round 1 --seed 1` on all 21
organisms (~1 h wall at 10-way; 21/21, 63/63 shards, 100% optimal, one
`index_hash`, 98.9% of media growing — the §4.3 "titrate everything at once"
collapse does **not** happen with a random share). Round media go to train only, so
the held-out set is bit-identical to every earlier run. Both heads had to be
retrained: `x_scale` is `_kink_scale` over the *training* rows, so a round changes
it and `Surrogate.__init__`'s P14 check fires if only one head is rebuilt — which
is exactly what it is for.

| held out, same 800 media | before | after round 1 |
| --- | --- | --- |
| Head B worst R2 / median | 0.883 / 0.941 | **0.907 / 0.952** |
| Head A worst grad cosine | 0.928 | **0.956** |
| Head A low-`mu` bias, median | -0.083 | -0.071 |

| median log-X | n=2 | n=3 | n=5 | n=10 | n=21 | worst cell |
| --- | --- | --- | --- | --- | --- | --- |
| session start (`behaviour_b1`) | 0.024 | 0.072 | 0.044 | 0.014 | 0.016 | 0.589 |
| + specific flux + MM clamp | 0.011 | 0.035 | 0.086 | 0.014 | 0.016 | 0.408 |
| **+ round 1 (both heads)** | **0.006** | **0.015** | 0.107 | 0.027 | 0.029 | **0.023*** |

*worst 2- or 3-member cell; the size-5 cell at 0.107 is now the worst overall.

1. **The coverage diagnosis is confirmed.** The two cells the whole investigation
   started from go 0.327 -> 0.023 and 0.589 -> 0.006, and their `dc_rel` 1.60 ->
   0.74 and 1.78 -> 0.22. `dc_rel` falls on 8/10.
2. **Sizes 10 and 21 read as a regression (1.4%/1.6% -> 2.7%/2.9%), and the
   comparison cannot support one.** See the next section: a single M5 cell is one
   Head A seed *and* one medium draw, and both move it more than this. Every M5
   number this file has carried, the 1.4% / 1.6% headline included, is n=1 on both
   axes.
3. **It is not the calibration.** Stripping `value_cal` from the round-1 head makes
   all ten communities worse (size 21 0.029 -> 0.043), so the refit is still
   earning its keep.
4. **Size 5 keeps drifting** (0.044 -> 0.086 -> 0.107) across all three changes
   while its `dc_rel` falls monotonically (1.089 -> 1.013 -> 0.897). Same
   rhs-vs-trajectory decoupling as the clamp's two regressions.

Cross-feeding recall is 1.00 at sizes 3, 5, 10 and 21 (0.91-0.89 before) and
`overgrowth` <= 0.038 throughout, so V5/P4 still do not bite.

### One M5 cell is n=1 on two axes, and the medium is the bigger one — 2026-08-30

Every community number this file has ever carried is a single Head A seed at a
single medium draw. Both were measured: 3 Head A seeds at a fixed medium
(retraining `value_r1` at seeds 0/1/2, Head B and the medium held), and 3 medium
draws at fixed heads (`cfs community --seed {0,100,200}`), over the same 10
communities. Max/min per cell:

| axis | median ratio | worst cell |
| --- | --- | --- |
| Head A seed | **1.8x** | 6.0x |
| medium draw | **6.1x** | **448x** (0.002 -> 0.739) |

**The medium is the dominant source of variance, by 3x, and the earlier claim that
the seed was is retracted.** That claim came from a run where `--communities` was
cut to the two large cells to save time: `dfba.run` draws each community's medium
from `seed + n` with `n` its *index in the list*, so shortening the list silently
re-drew every medium. The 0.029 -> 0.149 attributed to seed 1 was a different
medium; seed 1 on the full list gives 0.029. **Two `cfs community` runs are
comparable only if the community list is identical, in the same order** — and a
single draw is not worth quoting whatever the order.

Pooling both axes, n=5 replicates per community, `value_r1` + `behaviour_r1`:

| size | median | p25 | p75 | max | n |
| --- | --- | --- | --- | --- | --- |
| 2 | **0.009** | 0.005 | 0.018 | 0.739 | 25 |
| 3 | **0.018** | 0.007 | 0.064 | 0.110 | 10 |
| 5 | 0.076 | 0.028 | 0.085 | 0.107 | 5 |
| 10 | 0.027 | 0.022 | 0.029 | 0.125 | 5 |
| 21 | **0.027** | 0.027 | 0.029 | 0.029 | 5 |

1. **Sizes 2 and 3 are within 2x of M5's 1% gate on the median**; nothing passes it.
2. **The 21-member community is the most reproducible cell in the set** — 1.7x
   across seeds and **1.1x** across media, against 12x and 448x for 2-member ones.
   That extends "size is not the error axis": large pools are not just as accurate,
   they are far more *stable*, because the same independence that lets per-organism
   errors cancel in the sum also averages away the medium draw.
3. **The tail is a medium, not a community.** The 0.739 outlier is one 2-member
   community on one draw where the same heads score 0.002 on another. Chasing a
   worst-cell number without replicates is chasing the draw.

**What this means for the M5 gate:** it has to be stated over replicates. A single
`cfs community` invocation has a 6x sampling error on a small community, which is
larger than every model change measured today.

### M10: the medium designer walks out of the design unless it is stopped — 2026-08-30

`cfs maximise-growth` (`src/cfs/science/growth.py`) is §13.2: projected gradient
ascent on `mu_k(c)` under `cost . c <= B` and a box, with the head's own analytic
gradient chained through `dx/du . du/dc`, then every optimum round-tripped through
the true LP (V5). Head B is not needed, so `compose.dfba.Surrogate` now accepts
`behaviour_dir=None`.

**The result is P21, and it is severe.** Left with the budget and a box, the
designer pays for carbon by zeroing ~50 cheap metabolites at once, and the LP at
its "optimum" does not grow at all — `mu_true` 12.1 -> 0.0 on one case, 38.8 -> 0.0
on another, while the head reports an improvement. It is always the same
mechanism: one zeroed essential trace metabolite takes `mu` to 0 however good the
rest of the medium is.

The trust region is in `x`, the head's own input coordinate and the one §6.3's
nearest-training-medium distance is measured in, centred on the §4.3 start draw,
and **multiplicative** (`--trust-decades`, default 0.5) — an *additive* radius 0.2
still lets a metabolite at `x ~ 0.1` reach exactly zero and still loses 3 of 20
cases. 20 cases, one per roster organism, `value_r1`:

| trust region | improved | median gain | max gain | worst | median abs optimism |
| --- | --- | --- | --- | --- | --- |
| additive, radius 0.2 | 15/20 | — | — | **-100%** (x3) | — |
| 0.25 decades | 17/20 | +2.2% | 1.2x | -0.0% | 0.30% |
| **0.5 (default)** | 17/20 | +2.2% | 3.6x | **-14.8%** | 0.30% |
| 1.0 decades | 18/20 | +2.2% | **106x** | -0.0% | 0.28% |

1. **The gradient is good enough for this use case, as §13.7 said it would be.**
   Median optimism `mu_hat(c*) - mu_true(c*)` is **0.3%** of the LP's own value,
   and 17-18 of 20 ascents improve the *true* growth rate. The unimproved cases are
   start media already at the plateau.
2. **Median gain is 2.2% at every radius; the tail is what widens** (1.2x -> 3.6x
   -> 106x), and the big gains are near-starving start media rescued by
   reallocating the same total budget.
3. **The one loss is a `mu = 2.0` start medium** and it is not monotone in the
   radius (0.25 and 1.0 both pass) — one optimistic point in Head A's known weak
   low-`mu` band, not a trust-region trend. Do not tune the radius on it.
4. The region designs by *reallocation*: a metabolite the start medium has none of
   stays at zero. Seed the start medium to ask "should I add X".

Not done: non-uniform cost vectors, the selective-medium DC program (§13.2's
sign-flipped version), and the community version, which needs M12's steady state.

### The band floor was hiding a fifth of the design — measured, 2026-08-31

`SamplingConfig.log10_lo = -4` bounded `demand_probe`'s bracket *and* the focus
stratum's floor together, so a metabolite whose limiting onset is below
`c/Km = 1e-4` was invisible to the probe (`mu_lo == mu_hi` -> omitted by contract)
and unreachable by its own band. Roster-wide, **100 of 496 active (organism,
metabolite) bands were anchored at `"default"`** and therefore replete in every
medium. `probe_lo = -12` separates the two knobs; `log10_lo` stays at -4 for the
*unfocused* strata, where widening it is the measured "everything starves
together" collapse.

A second stratum, `frac_low_mu = 0.15`, draws 1-3 metabolites at once below their
**own** anchors with the rest replete. The design otherwise lands on the plateau
(76% of held-out media above 75% of max `mu`), which is where Head A is accurate
and M5's slow members are not.

`labels_p2` (4000 media + the round-1 community pass, 21/21, 63/63 shards):
band sources **494 probe / 2 previous / 0 default**, from 377/100/16/3.

| held out | `value_r1` worst / median | `value_p2` worst / median |
| --- | --- | --- |
| grad cosine | 0.9558 / 0.9674 | **0.9633 / 0.9813** |
| grad cosine p05 | 0.711 / 0.797 | **0.756 / 0.917** |
| top-1 share | 0.749 / 0.894 | **0.826 / 0.938** |
| value R2 | 0.974 / 0.9905 | **0.989 / 0.9994** |
| Head B R2 | 0.907 / 0.952 | **0.935 / 0.960** |

20/21 organisms improve on cosine, and it reproduces across Head A seeds: worst
cosine **0.9558 / 0.9500 / 0.9545** (r1, seeds 0/1/2) against **0.9633 / 0.9645 /
0.9662** (p2), i.e. +0.011 with the seed sd halved (0.003 -> 0.0015).

**M11's blocker is closed.** `n_missed_essential` **6 -> 0**. Unrestricted
(`--no-keep-essential`), where the old head took a 273-component medium to 41 with
`mu_true` 0/0/0 on every case, the new one gives 2/3, 0/3, **3/3** members clearing
the floor and `worst_true_frac` 0.436 against 0.000. V6 still does not pass at a 0.5
floor (0.491 / 0.436 / 0.512) but it is now a few-percent question. The minimal
media are co-limited, as they should be: all three members land on the same `mu`
(34.77 / 34.77 / 34.77).

**`value_rel_err_low_mu` reads worse (median 0.077 -> 0.242) and is not
comparable.** A new design means a new held-out set, and this one contains many
more genuinely slow media. Same for any per-band row count.

### ...and it made the §8.1 composition worse at large community size — 2026-08-31

The same 10 communities, **n=5 replicates each** (3 medium draws x Head A seed 0,
plus seeds 1 and 2 at draw 0 — matching the r1 replicate set exactly). The media
are **identical** between the two: `community_medium` only uses band scales when
`--scales` is passed, and the active-subspace lists did not change, so `p2` heads
on the `r1` label root reproduce `p2` on `p2` bit for bit. Same ruler.

| median log-X | n=2 | n=3 | n=5 | n=10 | n=21 | overall | `mu_rel` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `r1` | 0.009 | 0.018 | 0.076 | 0.027 | **0.027** | 0.017 | 0.0045 |
| `p2` | 0.014 | 0.067 | **0.029** | 0.093 | **0.466** | 0.031 | 0.0087 |

1. **This is not seed noise, and it is not the medium draw.** At size 21 the p2
   cells are bimodal: all three Head A seeds fail at medium draw 0
   (0.482 / 0.466 / 0.469, `dc_cos` 0.913) where all three r1 seeds are fine
   (0.029 / 0.029 / 0.017), and draws 100/200 give 0.078 / 0.081 against r1's
   0.027. So there is a ~3x regression everywhere at size 21 plus a reproducible
   blow-up on one medium.
2. **It is a tail, not a level.** At the failing medium the median |rel| over the
   21 members is 0.055 (r1) vs 0.071 (p2); what changed is the worst member —
   p2 over-predicts AAXE02 by **+153%** (44.6 against a true 17.6) and
   GCA_000151225.1 by +79%, where r1's worst is -28%.
3. **Two hypotheses tested and refuted.** *Plane budget* — `--gm-group 2000` on
   the same labels changes nothing at all: n=21 log-X 0.482 -> 0.482, `mu_rel`
   0.1044 -> 0.1056, `dc_cos` 0.9133 -> 0.9134 (and held-out worst cosine 0.9633 ->
   0.9596). K is not the lever above 1000, as it was not for width or depth.
   *Out-of-distribution in `x`* — the §6.3 nearest-training-medium distance for the
   failing members is **unchanged or slightly better** under the new coordinate
   (AAXE02 4.218 -> 3.961, GCA_000151225.1 3.635 -> 3.471, CP002109.1 4.590 ->
   4.614). The community medium is not further from the design than it was.
4. **The limiter is a carbon source whose band did not move.** At that medium
   AAXE02's true `mu` is set by `EX_g3pg_e` (10x it, `mu` +23.8 to 41.4; every
   other active metabolite gives 0.000), and its band is `probe` in both runs at
   7.43e-3 -> 7.34e-3. The surrogate's 44.6 is roughly the `mu` of a medium with
   **10x** the limiting carbon — it is not resolving how scarce that one
   metabolite is, at a point where ~30 others are also in bands.
5. ~~**So the suspect is the budget reallocation, not the new anchors.**~~
   **REFUTED — two relabels, see the next section.** `frac_low_mu` does come out
   of `n_rest`, and the unfocused "below Km" stratum *is* the only one that
   produces co-limited media. Both were fixed and neither changed the composition.

**The verdict is not "revert".** The label fix is a clean, reproducible win on
every label-level metric and it is what closes M11's essentiality blocker; the
composition regression is a separate, newly exposed weakness of the head at
multi-limited media. Do not read the old `r1` composition numbers as evidence the
old labels were better — they were better *at hiding* those dimensions.

### The size-21 regression: three design fixes, none of them it — 2026-08-31

Full stock-take and the ranked plan are **§8.5 of the design spec**. Read that
before touching the sampling design again. Summary:

| median log-X, n=5 replicates | n=2 | n=3 | n=5 | n=10 | n=21 | overall |
| --- | --- | --- | --- | --- | --- | --- |
| `r1` pre-relabel | 0.009 | 0.018 | 0.076 | 0.027 | **0.027** | 0.017 |
| `p2` relabel | 0.014 | 0.067 | 0.029 | 0.093 | 0.466 | 0.031 |
| `p3` stratum budget out of `frac_focus` | 0.024 | 0.049 | 0.022 | 0.082 | 0.783 | 0.030 |
| `p4` `focus_bg_decades = (0.0, 1.5)` | 0.032 | 0.048 | 0.026 | 0.095 | 0.706 | 0.035 |

1. **Reproducible, and a tail.** At n=21, medium draw 0 fails on all three Head A
   seeds (0.47/0.79/0.71) where all three `r1` seeds are fine; `mu_rel_median` on
   the true path is 0.005 (`r1`) vs 0.104-0.144. Median |rel| over the 21 members
   barely moves (0.055 -> 0.068) — it is **AAXE02 at +147%** (`mu_hat` 43.6 vs a
   true 17.6) and GCA_000151225.1 at +74%.
2. **Every label metric improved over the same relabel** — worst cosine 0.956 ->
   0.963, R² 0.974 -> 0.989, Head B R² 0.907 -> 0.937, per-metabolite limiting
   rows p10 **1 -> 100**, M11 misses 6 -> 0. So it is not fit and not coverage:
   it is distribution shift the held-out protocol **cannot see**, because
   held-out media come from the design that changed (P24).
3. **Five refuted predictors (P25).** Co-limitation count, near-onset count,
   NN-distance in `x`, per-metabolite limiting rows, held-out cosine/R². Each
   moved as designed with no downstream effect. Also refuted: plane budget
   (K 1000 -> 2000, nothing) and the limiter's band (never moved). **Do not spend
   a 5 h relabel on a proxy that has not first been shown to correlate with §8.1
   on runs already on disk.**
4. **`p3` was reverted; `p4` (`focus_bg_decades`) was kept** as the more
   defensible definition of "replete", not as a fix.
5. **Structural reason max-affine over-predicts here:** a min of tangents to a
   concave function is an *upper bound* everywhere and tight only near a tangent
   point. At a medium with no nearby anchor the min of the rest sits high. Any
   fix must put a plane in the community regime or bound the head from below.

**E1 has run, and it decided the branch: the labels are insufficient.** The
parameter-free cutting-plane model over `p4`'s *own* 3985 training tangents
over-predicts AAXE02 at the failing medium by **+153%** — slightly worse than the
trained head's +148% — while `r1`'s tangent set is **exact there (-0.000)**. So
the head is at its label ceiling and every architecture branch (C1/C1b/C2/E2/D1)
is refuted for this failure; the B branch is live. The mechanism is the mid-`mu`
band: `p4`'s binding plane is anchored at a `mu = 50.6` plateau row where `r1`'s
is at `mu = 17.9` against a truth of 17.63, and rows with `mu/mu_max` in 0.3-0.6
fell **261 -> 90** (AAXE02) across the relabel. Nearest-row distance in `w` is
*worse* for `r1`, so it is the growth regime, not proximity — a sixth refuted
proxy. Numbers and the script (`20hm_bands/e1_cutting_plane.py`): **design spec
§8.5 / "E1: the labels are insufficient"**.

### M5's 1% gate is met at every size but 21, and n=21 is the whole remaining failure — 2026-09-01

Median final log-X, 3 medium draws x 10 communities, identical list and media
throughout. `p4` labels; `_nc` = the output calibration stripped.

| run | n=2 | n=3 | n=5 | n=10 | n=21 | overall | max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `r1` (pre-relabel) | 0.006 | 0.016 | 0.028 | 0.027 | **0.027** | 0.014 | 0.739 |
| `p4` calibrated | 0.032 | 0.048 | 0.026 | 0.095 | 0.706 | 0.035 | 1.192 |
| `p5` (B2 mid-`mu` stratum) | 0.028 | 0.026 | 0.028 | 0.269 | 0.569 | 0.038 | 1.698 |
| `p4` no cal | 0.005 | 0.002 | 0.004 | 0.093 | 0.272 | 0.007 | 1.698 |
| `+ --w-under 1` | 0.006 | 0.002 | 0.005 | **0.055** | 0.346 | **0.006** | 0.462 |
| `+ level1` (rows) | 0.006 | 0.004 | 0.004 | 0.110 | 0.342 | **0.006** | 0.447 |
| `+ level1` (community pts) | 0.005 | 0.005 | 0.009 | 0.106 | 0.340 | 0.009 | **0.418** |

**Sizes 2/3/5 are at 0.4-0.9%, under M5's 1% gate. n=10 is 5.5-11%. n=21 sits at
0.27-0.35 and has not moved for five independent interventions** — four label
designs (`p2`/`p3`/`p4`/`p5`), the output calibration, `--w-under`, the plane
budget, and cut selection. It is the entire remaining M5 failure.

1. **The output calibration's sign flips with the label design.** It helps on
   `r1` (A1 median abs 0.0022 vs 0.0054 uncalibrated) and is the *dominant*
   error on the relabelled roots (`p4`: 0.0052 -> **0.0007**, worst organism
   0.039 -> **0.0027**). `calibrate` weights residuals at `_W_FLOOR = 0.3` of
   max `mu` — tuned when 72% of media sat above 0.8 of max `mu`; after
   `probe_lo = -12`, **62% sit below 0.2**, so the fit is dominated by the bottom
   and over-corrects the plateau, which is what `d(log X)/dt = mu` integrates.
   The old note "it is not the calibration" was measured on `r1` only. **Re-measure
   whether to calibrate on every new label root; it is a property of the design's
   `mu` distribution, not of the head.** V5 improves too (`overgrowth` max +0.262
   -> +0.097), so nothing runs away.
2. **B2 (the mid-`mu` stratum) is refuted.** It hit its label-level target
   exactly — rows in [0.3, 0.8] median 183 -> 555, **min 13 -> 470** — and made
   both A1 and §8.1 worse, calibrated or not. Seventh refuted proxy, but caught
   by A1 in seconds instead of a 5 h loop. Kept in the code, default 0.15;
   **set `--mid-mu 0` for a new label root** until something re-motivates it.
3. **`--w-under` (new): the one-sided hinge, and the diagnosis behind it.** At the
   failing n=21 medium the head read `mu` = 0.055 against a true 0.363 — an
   *under*-prediction, which a max-affine head cannot do from valid tangents. On
   training rows in the bottom 5% of `mu`, `p4` under-predicts **53-68%** against
   `r1`'s **0.2-0.5%**: `probe_lo` made the design bottom-heavy, so the absolute
   MSE now has thousands of low-`mu` rows and straddles them, and near zero that
   reads as a dead member. `_loss` now adds
   `w_under * mean(relu(mu - mu_hat)/den)^2` — a **provable violation** (every
   labelled point is one a min-of-supporting-hyperplanes must sit on or above),
   not an accuracy trade like `--w-rel`. `w=1` beats `w=10` on every axis; it costs
   worst held-out cosine 0.921 -> 0.899 and cuts A1's worst p90 0.311 -> 0.051.
4. **Level 1 cut selection (`--gm-select level1`) — see below.** It gives that
   cosine back (0.899 -> 0.919) and buys the worst cell, not `n=21`.
5. **Bug fixed:** `calibrate.apply` returned **NaN** for a negative raw
   prediction under the identity calibration (`d0 * exp(-m/1e-12)` = `0 * inf`),
   which took a dFBA trajectory to NaN at step 0. Every pre-2026-08-30 checkpoint
   was exposed.

### C4 (min over Head A seeds) is refuted — the sign is wrong — 2026-09-01

`cfs community --value a,b,c` now takes the **pointwise min** over several Head A
checkpoints (`dfba.Surrogate._mu`; the first dir supplies all metadata, and
`growth`/`minimal` still use it alone). Valid for free: a max-affine head is an
upper bound off-distribution, so a min of seeds stays in the family.

`p4` uncalibrated, seeds 0/1/2, 3 medium draws x the same 10 communities:

| median log-X | n=2 | n=3 | n=5 | n=10 | n=21 | overall |
| --- | --- | --- | --- | --- | --- | --- |
| `p4` no cal (seed 0) | 0.005 | 0.002 | 0.004 | 0.093 | 0.272 | 0.007 |
| **C4 min over 3 seeds** | 0.006 | 0.004 | 0.005 | 0.102 | **0.259** | 0.007 |

**Null, and structurally it had to be.** The min binds on 14 of 21 members at the
failing medium (median ratio 0.995), so it *is* doing something — but the n=21
failure is an **under**-prediction: `mu_rel_worst_member` is GCA_000007325.1 at
**-0.857** in both arms, and the community's `mu_rel_median` gets *worse*
(0.127 -> 0.143). Pushing predictions down cannot fix a member the head already
reads too low. This is the second E1 verdict ("trained-head deficit", the head
reading 0.055 against a true 0.363) restated as a composition metric: **read
`mu_rel_worst_member`'s sign before picking a fix.** C4 is the right tool for the
*calibrated* `p4` failure (AAXE02 at +148%), which is a different failure.

Kept in the code — it costs one comma and it is the cheap fix if an
over-prediction ever leads again.

### An under-prediction is a validity failure, and `--gm-repair` proves it — 2026-09-01

A min of *supporting* hyperplanes of a concave function is an upper bound
everywhere, so a head that reads **low** has left the family. Two mechanisms can do
that and both are separable in seconds, before any retrain:

| mechanism | test | verdict on `p4` |
| --- | --- | --- |
| the softmin gap, `<= T*ln(K_active)` | re-evaluate at `T -> 1e-6` (`groupmax.with_temp`) | **refuted** — moves `mu_hat` 0.008 at the failing medium (the `T*ln K` bound, 0.75 in `mu` units, is loose: ~1 plane is near-active) |
| planes no longer valid tangents | `mu_hat >= mu` on the head's **own training rows** | **confirmed** — 48.1% of rows under-predicted |

`groupmax.repair_intercepts` / `cfs train-value --gm-repair` is the fix, and it
applies **post hoc to an existing checkpoint** (`20hm_bands/repair_posthoc.py`, no
refit): hold the slopes, set each plane's intercept to the tightest value keeping
it above every training label, then apply the one uniform shift that covers the
smoothing gap. It is SDDP's cut-validity invariant, which that literature keeps by
never modifying a cut — we do, so we restore it after. With slopes fixed it is the
exact optimum of the intercept LP, not a heuristic. Training rows under-predicted
**48.1% -> 0.0%**, at a +4% median over-prediction.

**Per-plane validity is necessary and NOT sufficient — this cost a cycle.** The
head is the *smoothed* min and sits up to `c*T*ln(K)` below the hard one, and
training had been paying for that gap in the intercepts. Repairing planes without
restoring it left **96%** of rows under-predicted, worse than doing nothing. The
uniform shift is exact: lowering every `b_j` by the same delta moves all
pre-activations together, so it lifts the smoothed head by exactly `c*delta`. The
regression test runs at a production temperature on purpose; at `temp=1e-4` the gap
hides under any tolerance and the bug does not show.

3 medium draws x the same 10 communities:

| median log-X | n=2 | n=3 | n=5 | n=10 | n=21 | overall | max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `p4` no cal | 0.005 | 0.002 | 0.004 | 0.093 | **0.272** | 0.007 | 1.698 |
| + `--gm-repair` | 0.009 | 0.008 | 0.007 | **0.067** | 0.401 | 0.014 | 2.842 |

**The under-prediction is gone and the head is now loose instead.** Every member at
the failing n=21 medium flips sign — GCA_000007325.1 **-0.857 -> +0.998** — and
`mu_rel_worst_member` becomes DACTBY01 at **+2.436**. So the §8.6 under-prediction
and the §8.5 over-prediction are two ends of one thing, not two problems.

**It localises the deficit to the slopes.** The projection moves intercepts only,
and E1 found the cutting-plane model over the *same* labels is exact at that medium
(0.363) — so a valid model can be tight there and this one cannot be at any
intercept. Training moved the slopes off the label tangents. Do not read the n=21
regression as the repair failing; it removed the thing hiding a slope error. Off by
default. **Next is a chosen quantile (`--w-under` made explicit as `tau`), which
moves the slopes too — the ranked options and their literature are design spec
§8.6 and `docs/reading-map.md` §3c.**

### `--w-tau`: the expectile works, and the hinge still beats it — 2026-09-01

Option 2 of §8.6. `--w-tau` makes the value loss an **expectile** (asymmetric least
squares): the under-predicting side gets weight `tau`, the rest `1-tau`, scaled so
`tau = 0.5` is the plain MSE **bit for bit** — verified, `value_p4_tau0.5` matches
`value_p4` on all 21 organisms' `grad_cosine` to 1e-9, so `lr`/`w_grad` keep their
meaning. `score` now also reports `value_under_rate` / `value_under_rate_low_mu`,
the one-sided invariant as a rate, so a knob like this is chosen from a checkpoint
in seconds instead of a 5 h composition run.

| 21 organisms | worst cos | med R2 | low-`mu` under-rate | A1 worst p90 | n=21 log-X |
| --- | --- | --- | --- | --- | --- |
| `tau 0.5` (= `value_p4`) | 0.9211 | 0.9994 | 0.558 | 0.4493 | **0.272** |
| `tau 0.7` | 0.9237 | 0.9993 | 0.539 | 0.3045 | — |
| **`tau 0.9`** | 0.9076 | 0.9993 | **0.517** | **0.1298** | 0.327 |
| `tau 0.99` | 0.9228 | 0.9989 | 0.508 | 0.5322 | — |
| `--w-under 1` | 0.8985 | 0.9993 | — | **0.0509** | 0.346 |

1. **A prediction made here that it would be inert is retracted.** The value term
   is 0.3% of the objective at `w_grad 10` (`loss 0.248, value 0.00071,
   grad 0.02473`), and the inference that a tilt inside 0.3% cannot matter was
   wrong: A1's `worst_p90` falls 3.5x monotonically over `tau` 0.5 -> 0.9, and
   `tau 0.9` flips the failing n=21 member's sign exactly as the hinge does
   (GCA_000007325.1 **-0.857** -> DACTBY01 **+0.713**). A small term can decide a
   tail.
2. **There is an optimum and it is interior.** `tau 0.99` turns over hard (A1 p90
   0.130 -> 0.532) — at weight 0.02 nothing holds the plateau down. "More
   one-sided is better" is false.
3. **The hinge wins, and the mechanism is why.** The expectile tilts *every* row,
   so it buys one-sidedness by degrading the fit everywhere — community
   `mu_rel_median` 0.254 against `--w-under`'s 0.052 — while the hinge is exactly
   zero on compliant rows. For a **provable violation**, paying only at the
   violation is the right shape.
4. **A1's p90 reproduced the composition's `max` ordering a third time**
   (0.051 / 0.130 / 0.449 -> 0.462 / 1.658 / 1.698) and again did not predict
   n=21. It is a tail-over-media instrument; n=21 is one member at one community.

5. **The combination is refuted — they compete.** `--w-under 1 --w-tau 0.9`
   lands on the expectile's behaviour and slightly worse, not between the two:
   worst cosine **0.8870** (worst of the four arms), A1 p90 0.117, n=21 **0.362**,
   composition max 1.755. Both act on the same residuals, so once every row is
   tilted the hinge has no separate signal left. **Use `--w-under` alone.**

| median log-X, 3 draws | n=2 | n=3 | n=5 | n=10 | n=21 | max | A1 p90 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `tau 0.5` (baseline) | 0.005 | 0.002 | 0.004 | 0.093 | **0.272** | 1.698 | 0.449 |
| **`--w-under 1`** | 0.006 | 0.002 | 0.005 | **0.055** | 0.346 | **0.462** | **0.051** |
| `tau 0.9` | 0.014 | 0.001 | 0.005 | 0.077 | 0.327 | 1.658 | 0.130 |
| `--w-under 1 --w-tau 0.9` | 0.015 | 0.001 | 0.004 | 0.084 | 0.362 | 1.755 | 0.117 |

**Also fixed:** `train.run` wrapped `calibrate.fit` in a try/except. A finished
1500-epoch run was discarded by an import error inside that post-hoc 1-D fit; the
identity is a valid calibration, so it must never be able to throw away training.

### E1, run twice, gives opposite answers — and both are right

§8.5's cutting-plane check scores the parameter-free `min_j` model over a root's
**own** training tangents at a failing medium. It has now been run on two
different n=21 failures and it separates them cleanly:

| failure | trained head | cutting-plane over the same labels | verdict |
| --- | --- | --- | --- |
| AAXE02, `p4` **calibrated** | +148% | **+153%** (`r1`: -0.000) | labels insufficient |
| GCA_000007325.1, `p4` **uncalibrated** | 0.055 vs a true 0.363 | **0.363, exact** | trained-head deficit |

So "the labels are insufficient" was correct for the over-prediction the
calibration was masking, and is **wrong for the failure that remains**. Run E1
before choosing a branch; it costs minutes and it has flipped once already.

### SDDP's Level 1 cut selection, adopted — and ~90% of tangents never bind

The cutting-plane model *is* SDDP's outer approximation, sign-flipped: label
tangents are **cuts**, media are **trial points**, `--gm-group K` is the cut
budget. `groupmax.rank_by_territory` implements **Level 1 dominance** (de Matos,
Philpott & Finardi 2015) = the **territory algorithm** (Pfeiffer, Apparigliato &
Auchapt 2012 — provably the same selection): each cut owns the trial points where
it is the active minimum, and a cut with an **empty territory** is dropped. Scoring
all cuts at all points in one pass, keeping only the active index per point, makes
this the **limited-memory** variant (Guigues 2017): O(points), not O(cuts x points).
We **store and select** rather than prune, which is free because the tangents live
in the label shards. `--gm-select level1`, `--gm-trial-media <community_holdout.npz>`.

**The diagnostic is the durable result, and it needed no training.** Of ~3985
usable tangents per organism, those with a non-empty territory over the 4000
training media number **174-686, median 419 — about 10%**; over community-regime
trial points, **44-170**. The budget K=1000 **exceeds the useful count on 21/21
organisms**, so K 1000 -> 2000 could not have helped: there were never 2000 binding
cuts. That reproduces Pfeiffer et al.'s own ratio (490 -> 220 -> 55 cuts/stage with
the forward cost falling at the same rate). **K was never the lever.**

Level 1 wins the label metrics and the worst cell — worst grad cosine 0.899 ->
0.919, A1 worst organism 0.0083 -> 0.0022, composition `max` 1.698 -> 0.418 across
the four arms, an ordering that tracks A1's `worst_p90` exactly — and **loses at
n=10** (0.055 -> 0.110). `rank_by_active_set` stays the default: it is a proxy for
the same thing and every number on file was measured with it.

**A1 is a tail instrument and behaved as one.** Its `worst_p90` reproduced the
composition's `max` ordering; its median correctly said the four heads are
equivalent in bulk. Neither predicts the n=21 median, because that is not a tail
over media — it is one member at one community.

### Where the plan stands, and what is next

The staged plan is §8.5's "The progression for improving the training rows":
Stage 0 A1 **done**, Stage 1 B2 **run and refuted**, Stage 2 (retune from the `mu`
histogram) **done — the histogram was right and it did not help**. Stages 3-5 are
*not* the live question for n=21, because the second E1 says the labels are
sufficient at that medium. What is open:

- **n=21 is one member at one community.** Report it that way: `cfs community`
  emits `mu_rel_per_member` / `mu_rel_worst_member` (A2). The worst member is
  DACTBY01 at +0.68 to +1.22 — an over-prediction, which is max-affine's
  structural one-sided error at a point with no nearby tangent.
- **C4 is done and refuted** (see below): the min over seeds is the wrong sign
  for an under-prediction. **Untried and cheap:** `--gm-trial-media` pointed at a
  *bigger* community-regime pool than A1's 2000.
- **Do not** re-run `cfs topup` against held-out media from the design being
  changed; that is the naive Stage 4 and it has failed twice.
- **Literature map:** `docs/reading-map.md` (also an artifact). Read §3a before
  touching cut selection again.

**Caveat that affects all of it:** `x = u/(u+s)` takes `s` from the training
rows, so every relabel silently changes the input coordinate and two label roots
are never strictly comparable.

### M11: the minimal medium is blocked on essentiality, not on the program — 2026-08-30

`cfs minimal-medium` (`src/cfs/science/minimal.py`) is §13.3: minimise `cost . c`
subject to `mu_i(c) >= target_i` for every member and a box. Each floor is a
concave function >= a constant, so the feasible set is convex and the objective is
linear. Solved as a smooth quadratic penalty with `rho` continuation, a feasibility
restoration step, then a greedy **cardinality prune** — the convex program
minimises `cost . c` and V6 counts *components*, and a metabolite already pushed to
1% of its rich level still costs nothing to keep. The prune is what makes the count
mean anything.

**Head A does not know that removing an essential metabolite stops growth, and the
optimiser finds that immediately.** Single knockouts from a §4.3 rich medium,
3-member community, 37 free metabolites (`knockout_audit`, in the report):

| KO | `mu_hat` | `mu_true` |
| --- | --- | --- |
| `EX_cobalt2_e`, `EX_cu2_e`, `EX_mn2_e`, `EX_zn2_e` | 54.5 / 69.2 / 38.2 (rich: 54.5 / 69.2 / 38.2) | **0 / 0 / 0** |
| `EX_abg4_e`, `EX_bz_e` | unchanged | 55.1 / 70.2 / **0** |
| `EX_ca2_e`, `EX_cl_e` (one organism) | 1.33, 1.08 (rich 1.91) | **0**, **0** |

6 of 37 are lethal knockouts the head misses outright. Left free, the program plus
the prune takes 273 components to **41** with every surrogate floor satisfied and
the true LP growing **none** of the three members. That is P21 in a use case where
`growth.trust_box` cannot help: the region there is *multiplicative* precisely so
nothing reaches zero, and reaching zero is this program's job.

**The cause is `SamplingConfig.log10_lo = -4.0`, and the sidecar already names the
victims.** Measured by sweeping one metabolite with the rest held at the draw
(CP070062.1): `EX_cobalt2_e`'s true ramp lives between `c/Km` **1e-9 and 1e-6** —
`mu_true` is 0 at zero, 0.039 at `c/Km = 4e-9` and fully recovered by `4e-7`. The
probe brackets `log10(c/Km)` in `[lo, hi] = [-4, 1]`, sees `mu_lo == mu_hi`, and
by its own contract omits the metabolite ("never limits inside the band"); the
fallback chain then hands it scale **1.0**. The band's lower end is the *same*
parameter (`design.sample_media`, and the focus stratum is clamped at
`max(log10_lo, a - 1.5)`), so no sampled medium is ever cobalt-limited either.

Three consequences, in order:

1. **The dual is exactly 0 on every training row**, so `_kink_scale` takes its
   documented "never limits — nothing to resolve" fallback, `x_scale = 1.0`.
2. **The head's input then has no resolution there.** Across the entire design
   `w = u/x_scale` for cobalt spans `[0, 3.9e-3]`; separating `mu = 0` from
   `mu = 1.386` inside that needs a plane of slope ~350, and the only evidence for
   one is the single all-but-one-depleted corner row (23 such rows in 32 000).
3. **The four missed essentials are exactly the four with `"source": "default"` in
   `<id>.subspace.json`.** `EX_ca2_e`/`EX_cl_e` got `"probe"` at 1.1e-4 — itself
   the bracket floor — and the head *does* respond to them (1.91 -> 0.79 at zero).
   §4.7's promise that "a band anchored at the default is a known blind spot, not a
   silent one" held; nothing was reading it.

So this is a *label* defect, not a loss or architecture one, and restricting the
design to the active subspaces cannot help — the misses are active metabolites.

**So the support is pinned from the models** (`--keep-essential`, default off with
`--no-keep-essential`): one FBA per free metabolite per member, a static property of
the GEM that no medium search has to discover. 3 draws, 3-member community, floor at
0.5 of each member's `mu` on the rich medium, `value_r1`:

| | components | free / pinned / dropped | members clearing the floor under the LP | worst |
| --- | --- | --- | --- | --- |
| all metabolites free | 273 -> **41** | 273 / 0 / 232 | **0/3** | 0.000 |
| active subspace only | 273 -> 244 | 37 / 0 / 29 | 0/3 | 0.000 |
| **+ essentials pinned** | 273 -> **251** | 37 / 13 / 22 | **2/3** | 0.489 / 0.485 / **0.334** |

1. **V6 does not pass, and two of the three misses are ~2%** (0.489 and 0.485
   against 0.5). The real failure is the third, at 0.334, on the draw where a member
   starts at `mu_true` **3.5** against the others' 55 and 70 — the slow-member axis
   M5 found, in a new use case.
2. **The calibration belongs in the constraint.** §13.2 evaluates the head raw
   because an increasing map cannot move an argmax; here the constraint is on the
   *level*, so `calibrate.apply` and its derivative `g' = a + (d0/beta)e^{-m/beta}`
   are in both the value and the chain rule. Adding them moved the worst case
   0.190 -> **0.334** and dropped 7 more components.
3. **The MILP reference is not comparable and is off by default in spirit.**
   `cobra.medium.minimal_medium` is free over every exchange while the design is
   restricted to the active subspace with essentials pinned, and it is per organism,
   so the union is an upper bound on the joint optimum and `max_i` a lower one.

**What unblocked it: `probe_lo`, not the loss — done 2026-08-31.** Reweighting the
corner rows would teach a step at exactly zero and still leave the ramp unsampled
and unresolvable in the input coordinate; `--w-rel` was the wrong knob. The band
floor was the knob, and after the relabel `n_missed_essential` is **0** — see "The
band floor was hiding a fifth of the design" below for the numbers and for the
composition regression it cost. Count `"source": "default"` bands in the sidecars
to check any future label root: 100 of 496 before, **0 of 496** after.

### The conditioning bill is not §8's — measured, 2026-08-26
>
> `cfs train-value` reports `hessian_cond_median` from `train._hessian_cond`: **one
> organism's** Hessian, on its own dims, in the head's own coordinate. §8.4 inverts
> something else — `J = sum_i X_i (-d2 mu_i/du2) + supply'(u)` over the 365 shared
> exchanges. `cfs master-jacobian` (`src/cfs/validate/master_jacobian.py`) measures
> that object at real held-out media. 21 seeded `groupmax-u` heads, K=250, uniform
> abundances, `20hm_bands`; eigenvalues of `sum_i X_i H_i` above a fraction of the
> top, median over media, out of 365:
>
> | T | 0.01 | 0.03 | 0.3 | `value_b1` icnn (trained) |
> | --- | --- | --- | --- | --- |
> | raw, > 1e-12 | 98 | 147 | 194 | 365 |
> | **Jacobi-preconditioned, > 1e-12** | **22** | **20** | **13** | **365** |
>
> 1. **The Hessian sum is singular** — ~10-25 of 365 directions carry curvature.
>    `optx.Newton` under `positive_semidefinite_tag` on `sum_i H_i` alone is
>    ill-posed, not merely ill-conditioned. §8.1's `inflow(c)` is what makes the
>    solve well-posed, and once `lam I` is present `cond(J) = 1 + top_ev/lam`
>    **exactly**: the supply model sets the conditioning, the head does not.
> 2. **`T` buys no conditioning.** 0.01 → 0.3 is 30x blunter, costs held-out cosine
>    0.951 → 0.833 on the `DEFAULT_TEMP` table, and moves the curvature rank from 22
>    to 13. The sharpness-vs-Newton trade **does not appear in `J`**. So `--gm-temp`
>    is an accuracy knob, `hessian_cond_median` is not a Phase-5 predictor, and Arm
>    G's range moved to the sharp half (0.01-0.1) where accuracy actually varies.
> 3. **The smooth `icnn`'s ill-conditioning was per-metabolite scaling, not
>    curvature.** Jacobi-preconditioned it is full rank at every cut; raw it is not.
>    `s` spans 5107x across the index, so **precondition `J` diagonally in §8.4
>    whatever the head is** — which also absorbs the still-unwritten chain rule from
>    `u` to the price coordinate, so the result does not depend on it.
>
> That leaves the smooth head giving a well-conditioned Jacobian that is *wrong*
> (cosine 0.73) and the sharp head an accurate gradient on a rank-20 one. ~10-40
> curvature dims is an **active-set** picture, arrived at from the spectrum rather
> than from theory: §8.4 wants a reduced-space or semismooth Newton (Qi–Sun) plus
> the diagonal preconditioner and the supply term, not a blunter head. P11's
> difference-of-convex escape hatch and P9's damping are unaffected.
>
> Caveats it does not clear: abundances are uniform (a positive diagonal reweighting
> cannot change which directions carry curvature, but the *scale* moves), the heads
> are seeded rather than trained, and no supply model exists yet, so `lam` has no
> physical scale. Re-run with `--checkpoint` once Arm G lands.
>
> ### Deepset is measured at full width, and cut — 2026-08-27
>
> The full-width rerun (`export/sweep_out/sweep/`, four `deepset-private` cells x
> 21 organisms, held-out round-0 media, `w_grad` 1, x-space) discharges the
> "lower bound, not refuted" caveat. Roster medians, worst over 21 organisms:
>
> | cell | worst cos | med cos | med R2 | med p05 | log10 Hess cond | h/organism |
> | --- | --- | --- | --- | --- | --- | --- |
> | `deepset-private ph32/kc64` | **0.742** | **0.810** | 0.564 | 0.137 | 11.3 | 7.5 |
> | `deepset-private ph64/kc64` | 0.733 | 0.809 | 0.564 | 0.132 | 11.7 | 12 |
> | `deepset-private ph64/kc16` | 0.665 | 0.811 | 0.560 | 0.138 | 12.5 | 11 |
> | `deepset-private ph32/kc16` | 0.653 | 0.797 | 0.554 | 0.137 | 12.1 | 5 |
> | `icnn w128/d3` | 0.678 | 0.755 | 0.541 | 0.106 | 9.8 | 0.075 |
> | `mlp w512` | 0.681 | 0.873 | 0.915 | 0.168 | 7.5 (60.7% non-concave) | 0.2 |
> | `rf` | 0.357 | 0.664 | 0.994 | 0.000 | n/a | 0.01 |
>
> 1. **At full width it beats the x-space ICNN, and by little.** 18/21 organisms
>    on cosine, median +0.055, worst +0.064 — and value R2 is *identical* (0.564 vs
>    0.541, worst organism 0.352 vs 0.350). A small gradient lift, not an
>    architecture change, and nowhere near the 0.99 gate. Cost is ~157 cpu-h for one
>    21-organism cell against 1.6 for the whole `icnn w128/d3` cell.
> 2. **The conditioning win is gone.** It was the reason to revisit deepset at all
>    (the underfit shared-trunk run reported 4.0e3). At full width the median
>    Hessian condition is **1e11-1e12, worse than the ICNN's 1e10**. And per "The
>    conditioning bill is not §8's", that number does not reach Phase 5 anyway.
> 3. **It does not fit the per-metabolite cells better — which was the whole bet.**
>    Per (organism, metabolite) cell against `icnn w128/d3` on the same media, the
>    lift is small and roughly uniform, *not* concentrated where coverage is thin:
>    <25 rows +0.012 (n=357), 25-100 +0.062 (215), 100-400 +0.020 (141), >=400
>    +0.010 (31). Split by difficulty instead of coverage: on the 284 cells where
>    the ICNN scores <0.5, deepset scores **0.312 against 0.286**; on the 72 cells
>    where the ICNN is already >=0.9, 0.958 vs 0.954. The two heads fail on the
>    *same cells* by nearly the same amount. Same three ions lead the error on
>    ~20/21 organisms in both (`EX_mg2_e` median 0.407 vs ICNN 0.322 vs forest
>    0.995; `EX_cl_e` 0.105 / 0.037 / 0.990; `EX_ca2_e` 0.079 / 0.037 / 0.985).
>    Private per-metabolite trunks did **not** localise the kink.
> 4. **Both deepset capacity axes are inert, exactly like the ICNN's.** Paired
>    per-organism medians: `k_code` 16->64 at ph32 **-0.001** (range -0.037 to
>    +0.100), `phi` 32->64 at kc16 **+0.001** (-0.018 to +0.044), against `icnn`
>    w128->w1024 d3 **-0.000**. The 0.653 -> 0.742 worst-cosine move is one
>    organism's tail, not a trend — do not read it as a `k_code` effect.
>
> **Why `deepset-u` is not worth building on top of this.** The rerun is x-space,
> so it measures the per-metabolite bet *under* the coordinate defect, and
> `deepset_u.TrunkU` would inherit the `w` fix and presumably the ICNN's gain
> (R2 0.477 -> 0.802). But the **larger** lever cannot be applied to it at all:
> `groupmax.init_from_tangents` writes the labels' duals straight into layer 1 as
> affine planes over the whole metabolite vector, worth cosine 0.598 -> 0.973 on
> its own. Deepset's first layer is a per-metabolite *scalar* map `phi_m: R -> R^k`;
> there are no planes over the metabolite vector to seed, and the duals do not
> factor into per-metabolite ramps. So `deepset-u` would land near `icnn-u`, not
> near seeded `groupmax-u`, at ~100x the compute. Result 3 above is the empirical
> half of the same point: the cells it fails are the cells every smooth head fails.
>
> Structurally: `d(mu)/dw_m = <rho'(S), phi_m'(w_m)> / |M|` is a rank-`k_code`
> bilinear form standing in for an argmin over 444 metabolites. The exact
> log-sum-exp softmin factorisation through a *mean* pool needs `rho` convex and
> decreasing, which the enforced concave-non-decreasing constraint excludes; an
> approximation through a `k>=2` code is not ruled out, but it is a detour to
> something `groupmax-u` computes natively, its max being over planes that span all
> metabolites at once.
>
> **What would reopen it:** one cell, one organism — `deepset-u` ph32/kc64 on
> CR626927.1 at `w_grad` 10, ~7.5 h, against `icnn-u` 0.903 and seeded
> `groupmax-u` 0.973 on the same held-out media. Below 0.903 the arm is dead; above
> it, the ~157 cpu-h roster cell becomes arguable. The arch stays built, tested and
> registered (`--arch deepset-u{,-private}`) so that test costs no new code.
>
> **Cut, with reasons on file** (`make_sweep_full.sh` keeps the invocations):
> `deepset-u` (was 80% of the run at 5-11 h/organism/cell; the per-metabolite bet
> is now refuted at full width — see the section directly above — and its one
> claimed advantage, conditioning, is both gone and known not to reach §8);
> the `--gm-init random` control cells (the
> single-organism A/B is decisive, so the sweep no longer carries its own control
> and the attribution rests on one laptop measurement); and Arm F's random-init
> group/temperature grid (it would have re-measured collapse on 21 organisms).
>
> **What would change the picture:** a roster-wide worst-organism cosine well below
> 0.973 (the one-organism result does not generalise), or a supply model whose `lam`
> is too small to regularise a rank-20 `J` (then §8 needs the reduced-space or
> semismooth Newton above, not a better head).
>
> **Pipeline traps that still bite** (`--stage sweep`, `workflows/value_sweep.nf`):
> `cfs train-value` **exits 1 whenever the gate is unmet**, so `TRAIN_VALUE`
> tolerates that and gates on `diagnostics.json` existing instead (read `passed`,
> not the exit status); `params.xla_devices` must **divide** the organism count; and
> the shared-trunk fan-out check is `arch in ['deepset','deepset-u']` — fanning a
> shared-trunk cell out per organism silently makes its trunk private. The `m4k`
> rows shipped a `/path/to/...` placeholder last time and every task read the 20k
> root instead, so **check `n_train_media` differs between two cells before
> believing any rows conclusion**. It happened again in the 2026-08-27 run — no
> `labels_out_4k` was staged at all, so the three `m4k` cells simply never ran.
>
> Sharding note for the cluster: the organism axis splits cleanly via
> `train._shard_organisms` (1.8× on CPU with
> `XLA_FLAGS=--xla_force_host_platform_device_count=N`, N must divide the organism
> count); on GPU the `filter_vmap` stacking is the §6.1 win. The §4.7/§4.6
> machinery stays: it removed the two-pass dependency and it is how a *new* genome
> gets labelled at all.
>
> Loss weights are not the answer **for the x-space ICNN**: `w_grad` only trades
> the heads (1 → 0.63 cosine / 0.72 R², 10 → 0.66 / 0.62) and neither end reaches
> the "R² ≥ 0.9" balance rule. This is coordinate-specific — see the `icnn-u`
> `w_grad` sweep above, where 10 gives 0.903 / 0.756. Checkpoints: `20hm_bands/value_b1/` (best worst-cosine to date,
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
- **`sweep`** — M3b/§7.4 Head A sweep (`workflows/value_sweep.nf`). One task per
  (row of `--sweep <sweep.csv>` (`cell_id,arch,labels,args`), organism):
  `TRAIN_VALUE` (`cfs train-value`) or, for `arch=rf`, `BASELINE_RF`
  (`cfs baseline-rf`) → `COLLECT_VALUE_METRICS` → `${outdir}/sweep_leaderboard.csv`.
  **The stack is any number of organisms** (`--organisms`, a subset of the label
  root's shards; `load_value_dataset` splits by `medium_id`, which is identical
  across organisms, so a 1-wide stack's held-out set is the 21-wide stack's). The
  workflow fans out one organism per task for every arch except the shared-trunk
  `deepset` — the only one that pools across the organism axis — and
  `COLLECT_VALUE_METRICS` merges a cell's tasks back into one leaderboard row by
  stripping the `__<genome_id>` suffix. `xla_devices` therefore only ever applies to
  the `deepset` cells. Needs `--index`;
  needs **no `--roster`** — the only stage that reads labels rather than GEMs, which
  is why the roster check in `main.nf` is stage-aware. The per-cell knobs live in the
  samplesheet, not in params, so there is no param per sweep axis; `params.sweep` and
  `params.xla_devices` are the only two. Worked example plus generator:
  `examples/hpc_run/`. Stub: `tests/sweep.nf.test`.
- **`train`** — the legacy sweep below.

M4/M5 are CLI-only (no Nextflow stage yet): `cfs train-behaviour` trains Head B on
the same shards and the *same* held-out media split as `cfs train-value`
(`cfs.surrogate.data._stack` is shared, so the two heads cannot silently disagree
on `x_scale` or on which media are held out — `Surrogate.__init__` re-checks both),
and `cfs community` composes a value + behaviour checkpoint pair into §8.1's dFBA
and scores it against per-organism FBA. `cfs community` needs the `data` extra
(cobra) *and* the `jax` extra, the only subcommand that needs both.

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
  (`ghcr.io/timrozday-mgnify/surrogate-mgem-{train,data}:0.1.7`). Bump the tag in
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
