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
> | M3 Head A (value) | **gate not met, but the deficit is located and mostly closed.** Two independent causes, each measured: concavity was imposed in the wrong coordinate (`x`, not `u`), and then *initialisation* collapsed the max-affine head. Best to date on one organism: `groupmax-u` seeded from label tangents, **cosine 0.973 / R² 0.996 / p05 0.834**, concave, 111k parameters — against the old ICNN's 0.715/0.477. Roster-wide numbers do not exist yet | `src/cfs/surrogate/{picnn_u,deepset_u,groupmax}.py` |
> | M3b HPC sweep | **ran 2026-08-25 (350 tasks, 324 cpu-h) and is the source of the above.** Its own conclusion — "scale closes the gap" — was refuted: width/depth are inert, rows are inert. A second 24-cell sweep is built and unlaunched | `examples/hpc_run/`, `--stage sweep` |
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
> Three things this does **not** establish. It is **one organism**; the gate is the
> worst over 21. `w_grad` 30 collapses on both axes, and whether that is a real
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
> is not LSPA though: those algorithms *infer* the planes from values, and we have
> the duals, so seeding them directly is simpler and strictly better.
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
> **Next — the second sweep.** Everything
> above is CR626927.1 on a laptop; the gate is the worst of 21 organisms at 0.99.
> `examples/hpc_run/sweep_full.csv` is **24 cells in six arms, ~150 cpu-h** (half
> the last run, answering more), regenerated by `make_sweep_full.sh`, which carries
> the measurement motivating each arm:
>
> | arm | cells | asks |
> | --- | --- | --- |
> | A `icnn-u` `w_grad` {1,3,10,15,20} | 5 | where the frontier's optimum sits, roster-wide |
> | B `icnn-u` width {512,1024} × depth {3,6} at `w_grad` 10 | 4 | capacity, at the operating point |
> | C `icnn` / `mlp` / `rf` | 3 | the x-space head, plus ceiling and floor |
> | F `groupmax-u` seeded, width 128 × depth 3, T {0.01,0.03,0.1} | 3 | does depth help *once the pieces are seeded*? |
> | G `groupmax-u` seeded, width 1 × depth 1, K {100,1000} × T {0.01,0.03,0.1} | 6 | K, and where the accuracy knee in `T` sits |
> | E the 4000-media set | 3 | the rows axis, wired to a real second label root |
>
> ### The conditioning bill is not §8's — measured, 2026-08-26
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
> believing any rows conclusion**.
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
  (`ghcr.io/timrozday-mgnify/surrogate-mgem-{train,data}:0.1.6`). Bump the tag in
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
