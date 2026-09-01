# Composable Surrogate Models for Community Flux Balance Analysis

Implementation specification, v2. Decisions locked except D4.
Written to be handed to Claude Code.

---

## 0. Locked decisions

| # | Decision | Choice | Consequence |
|---|---|---|---|
| D1 | Organism representation | **One surrogate per organism, fixed roster of 20** | No generalisation to unseen organisms. Use `eqx.filter_vmap` over stacked parameters — 20 surrogates cost roughly the same as 1 |
| D2 | Shared metabolite universe | **100+, automated derivation, no curation** | Drives the sampling redesign in §4. Requires the partition scheme in §2.1 |
| D3 | Model source | **CarveMe** | BiGG namespace aligns automatically. Mandatory EGC pre-flight (§3.0) |
| D4 | Uniqueness scheme | **DECIDED — elastic net (§5.4)** | Diagnostic returned "genuine". See §5.4 decision note |
| D5 | Concentration → uptake bound | **Michaelis–Menten** | Km values are not available for 100+ metabolites. See §3.3 |
| D6 | Framework | **JAX** — Equinox, Optimistix, Lineax, Diffrax, BlackJAX | Do not use JAXopt (unmaintained) |
| D7 | Smoothing scale `τ` | **Three-level family** | Unified with D4 — see §5.5 |
| D8 | Ecological framing | **Build all, validate dFBA first** | |
| D9 | First scientific target | **Minimal medium design** | Drives sampling toward low concentrations (§4.3) |
| D10 | Scale | **N=20, 20k media, K=8** | Compute budget in §4.5 — read this before generating |

---

## 1. What we are building

```
Layer 0   Ground truth       COBRApy + QP solver     ~50 ms per solve
Layer 1   Surrogate          20 frozen nets          microseconds
Layer 2   Composition        Newton on a master      microseconds
Layer 3   Science            minimal medium / HMC    many Layer-2 calls
```

### Per-organism learned object

**Head A — value function.** `mu_max_i(c) -> scalar`. Concave in uptake bounds.
Gradient is the shadow-price vector. Partially input-convex network, negated.

**Head B — behaviour map.** `z_i(c, alpha) -> R^{M_i}`. Net exchange fluxes per
unit biomass at normalised growth rate `alpha = mu / mu_max_i(c) ∈ [0,1]`.

Feasibility falls out of Head A: `feasible(c, mu) <=> mu <= mu_max_i(c)`.

### The key structural point given D1(a) and D2

Each organism's surrogate operates in **its own exchange subspace** `M_i`
(typically 40–70 reactions for a CarveMe model), not the full shared universe
`M ≈ 200`. The 200-dimensional coordinate system exists only at the
composition layer, where masked vectors are summed.

This matters enormously for sample efficiency. 20k samples in 200 dimensions
is hopeless coverage; 20k samples in the ~20-dimensional *active* subspace of
one organism (§4.2) is respectable.

---

## 2. Automated metabolite universe (D2)

No curation, so this must be a deterministic procedure.

### 2.1 Derivation

```python
# 1. Union of exchange reactions across all 20 CarveMe models.
#    BiGG IDs align by construction — this is why D3 = CarveMe matters.
all_ex = set().union(*(set(m.exchanges) for m in models))

# 2. Partition by how many organisms can exchange each metabolite
shared  = {m for m in all_ex if n_organisms_with(m) >= 2}   # coupling
private = {m for m in all_ex if n_organisms_with(m) == 1}   # medium input only

# 3. Only `shared` enters the composition coupling.
#    Newton Jacobian is |shared| x |shared|, not |all_ex| x |all_ex|.
```

Expect `|all_ex| ≈ 200–260`, `|shared| ≈ 80–130` for 20 gut or environmental
organisms. Private metabolites still enter each organism's own surrogate; they
just never need clearing.

### 2.2 Freeze it

Write the resulting index map to `config/metabolite_index.json` and version it.
Every dataset, checkpoint, and result must record the hash of this file.
Silently changing the index later invalidates every trained model with no
error message.

---

## 3. Phase 1 — ground truth pipeline

### 3.0 Pre-flight: energy-generating cycles

**Do this before anything else.** CarveMe gap-fills, and gap-filled models
frequently contain thermodynamically infeasible energy-generating cycles that
produce ATP from nothing. If present, every growth prediction downstream is
fiction and the L2 regularisation in §5 will happily distribute flux around
the cycle.

```python
def has_egc(model):
    with model:
        for ex in model.exchanges:
            ex.lower_bound = 0.0          # close all uptake
        model.objective = model.reactions.ATPM
        return model.optimize().objective_value > 1e-6
```

Run for all 20. Any model returning True must have the cycle removed (loopless
FBA, or `cobra.flux_analysis.loopless_solution` to identify it) or be dropped
from the roster. Do not proceed with a model that fails.

Run MEMOTE on all 20 as well and store the reports. It is the standard
automated QC and it is the defensible answer to "you did no curation".

### 3.1 Standardisation

Map every model's exchanges onto the frozen index from §2.2. Build a boolean
mask matrix `(20, M)` recording which organism can exchange which metabolite.
Store it alongside the index.

### 3.2 Solve interface

```python
def solve(organism_id, c, alpha, eps) -> Solution:
    """Returns mu_max, z (masked to M_i), shadow prices, solver status."""
```

Two stages: solve for `mu_max` and capture exchange duals; then fix
`v_biomass = alpha * mu_max` and solve the §5 uniqueness problem for `z`.

### 3.3 Michaelis–Menten parameters (D5)

You will not have `Km` for 100+ metabolites and should not pretend otherwise.

```
lb_m = -Vmax_m * c_m / (Km_m + c_m)
```

- `Vmax_m`: take the model's existing default uptake bound (CarveMe sets these).
  This absorbs the scale and avoids inventing a second unknown.
- `Km_m`: use a single default per transporter class (sugars, amino acids,
  ions, gases) from literature order-of-magnitude values. Four numbers, not
  200.

**State this as a limitation explicitly in any writeup.** The `Km` values are
not measured, so any result whose ranking depends on their relative magnitudes
is not supported. Results that depend only on which metabolites are limiting
(the topology of the shadow-price structure) are robust to this.

Longer term this is a natural extension of your HMC work: put priors on `Km`
and infer them jointly. Out of scope for v1.

### 3.4 Acceptance criteria

- Identical inputs give bitwise-identical `z` across repeated runs.
- Perturbing `c` by 1e-6 changes `z` by O(1e-6), not O(1).
- `dmu_max/dc` agrees with returned shadow prices to finite-difference
  tolerance.
- No model has an EGC.

**Hard gate.** These labels are inherited by every later phase, and a
degenerate labelling problem does not show up in training loss.

---

## 4. Phase 2 — sampling design

D2 at 100+ dimensions makes this the part of the plan that changed most.

> **Implementation status (2026-07-26).** §4.1–4.5 built in `src/cfs/sampling/`:
> `active_subspace.py` (§4.2 sweep), `design.py` (§4.3–4.4 stratified-Sobol +
> `SamplingConfig` carrying the K=8 alpha grid and the 3-level `eps` family), and
> `generate.py` (§4.5 driver → parquet sharded by organism × `eps`, full media at
> the primary `eps`, 20% subset at the others; stores `mu_max`/`z`/shadow/status/
> medium, records `index_hash` per row per P13). CLI: `cfs active-subspace`,
> `cfs generate`. `pyarrow` added to the `data` extra. Tested (`tests/test_cfs_sampling.py`).
>
> **Run at scale (2026-07-26), 21-genome roster** (`~/Documents/20hm_carveme_models`,
> run dir `~/Documents/surrogate-mgems_runs/20hm`). Nextflow `GENERATE_LABELS` +
> `--stage labels` now exist (`workflows/label_generation.nf`), one task per
> organism, publishing `labels/<id>/eps_<e>/part.parquet` plus
> `<id>.subspace.json` / `<id>.exchanges.json` sidecars. First bulk set generated
> at **4000 media/organism** (1/5 of the D10 budget — the design is unchanged, the
> count is laptop-sized; 20000 is still the `SamplingConfig` default for HPC).
>
> Result: **21/21 organisms, 0 failures, 940 800 label rows, 573 MB parquet**,
> ~58 min for the slowest organism (12 concurrent, 12 cores). Per organism and
> `eps` the shard shapes are exactly the §4.5 design — 32 000 rows at the primary
> `eps=1e-3` (4000 media × 8 α) and 6400 at each of `1e-2`/`1e-4` (the 20% subset).
> **100% of solves optimal**, one `index_hash` across every row (P13), `mu_max`
> std 7–28 per organism with 0–0.5% zero-growth media — neither flat nor
> mostly-infeasible. **`|A_i|` = 11–32, median 24** across the roster, confirming
> §4.2's 15–30 estimate on real models (two organisms fall just outside).
>
> Three things the scale-up found and fixed (all were latent, none visible on toy
> models): `km_defaults.yaml` had to move into the package (`src/cfs/config/`) —
> the repo-root lookup resolved to `site-packages/../config` in the container;
> cobra's FVA had to be pinned to `processes=1` (its default worker pool turned a
> 1-second exchange-FVA into a >30-minute one); and **the elastic-net QP moved
> from HiGHS to Clarabel** (§5.4 note).
>
> **Not yet:** §4.6 active-learning reserve (needs Phase 5); `config/sampling.yaml`
> (defaults live on `SamplingConfig`).
>
> **Second run (2026-07-27), per-metabolite bands** (run dir
> `~/Documents/surrogate-mgems_runs/20hm_bands`, same roster, same 4000
> media/organism, same frozen index). §4.3's single shared band is wrong on real
> models — see the correction there — so this run centred each metabolite's focus
> stratum on its own limiting regime via `design.limiting_scales` (`cfs generate
> --scales`), read off the first run's labels. 63/63 shards, 21/21 organisms,
> 100% optimal. Over the sampled set `A_i`, roster medians: the **median
> metabolite's limiting media 143 → 174**, metabolites with ≥50 media 15 → 17,
> the top metabolite's share of media 0.58 → 0.55; the ions gained most
> (AAXE02 `EX_mg2_e` 84 → 155, `EX_ca2_e` 38 → 108) and `EX_k_e` came down
> 559 → 411. Downstream effect on M3 in the §7 status block.
>
> Two operational findings: the run needs ~30 MB/organism and dies mid-shard on a
> full disk (`OSError: [Errno 28]`); and GLPK — cobra's default LP solver here —
> can cycle indefinitely on a near-degenerate medium (one organism burned 4.5 h at
> 100% CPU inside `glp_simplex` on a single solve). The FBA now carries the same
> `_QP_TIME_LIMIT` the QP has had since M2.

### 4.1 Per-organism designs, not shared media

Surrogates are trained independently and only composed at inference, so each
organism gets its own sampling design over its own `M_i`. Do not use a common
media set — it wastes most samples on metabolites the organism cannot use.

### 4.2 Active subspace reduction

For each organism, before bulk sampling:

1. Solve on a rich medium, record the shadow-price vector.
2. Do a coarse one-at-a-time sweep: which metabolites, when reduced, change
   `mu_max`?
3. Define `A_i` = metabolites with non-negligible sensitivity anywhere in the
   sweep. Expect `|A_i| ≈ 15–30`.

Sample densely over `A_i`; hold the rest at a fixed background level with
occasional randomised perturbation (10% of samples) so the surrogate learns
they are inert rather than never seeing them vary.

This is what makes 20k samples viable. Without it you are sampling a
60-dimensional box with 20k points and the surrogate will be accurate nowhere
in particular.

### 4.3 Sampling distribution — biased for D9

Minimal medium design drives concentrations toward zero, so the optimiser will
spend all its time in the low-concentration regime. That is also where MM is
steepest and where feasibility flips.

> **Correction (2026-07-27, measured).** The band below is written *relative to
> `Km`* and shared by every metabolite. That assumes metabolites start limiting at
> comparable `c/Km`, and on the 21-genome roster they do not: measured medians
> span **5107×**. `EX_mg2_e`/`EX_ca2_e`/`EX_cl_e` begin limiting 3–4% into the
> band, `EX_o2_e`/`EX_malt_e`/`EX_h_e` at 82–88%. The consequence is a label set
> whose per-metabolite coverage is skewed ~**1137 : 9 : 1** per organism, and
> held-out gradient cosine per (organism, metabolite) cell tracks that cell's
> training-row count at Spearman **0.72** — so the shared band, not the
> architecture, was the first thing holding the M3 gate down. The literature `Km`
> is not the right anchor; the nutrient's own demand is (the legacy pipeline
> reached the same conclusion via `surrogate_mgem.data.estimate_demand`). Read the
> band placement below as *per metabolite*, anchored per §4.7.

- Sample `log10(c)` uniformly over roughly `[-4, 1]` relative to `Km`.
- **Weight 40% of samples below `Km`.** A uniform log design underweights
  exactly the region D9 cares about.
- Include the all-but-one-depleted corners explicitly: for each `m ∈ A_i`,
  a batch with `c_m` swept to zero and others rich. These pin down the
  single-limitation facets of the value function.
- Sobol sequences within each stratum, not uniform random.

> **Correction (2026-08-30, measured) — the design must cover the *community*
> regime, and it did not.** §4.2 fixes each organism's background at one rich
> level and varies only `A_i`. In a community the shared pool is drawn over the
> **union** of the members' active subspaces, so a member routinely sees a large
> share of its own background metabolites off replete at once — ~20% of it for a
> pair, most of it for the whole roster. `SamplingConfig.frac_bg_perturb` was
> already the knob for this and was degenerate: it perturbed the background
> all-or-nothing, 10% of media with *every* held metabolite redrawn and 90% with
> none. The design was therefore bimodal in exactly the axis the composition moves
> along, with nothing in between.
>
> That hole is where §8.1's failing communities sat. Head B's held-out flux cosine
> tracks distance to the nearest training medium on **all 21 organisms** (Spearman
> 0.33–0.84, median 0.65), the worst community medium is **2.92** from its nearest
> neighbour against a held-out median of 0.10 — and it is a *joint* gap, not a
> marginal one: every coordinate of it is inside that organism's own training
> range and its count of scarce dimensions is typical.
>
> `sample_media` now draws a random **share** of the background per medium, which
> spans every community size in one design; `cfs generate --bg-perturb` exposes the
> fraction. An 800-media round at `--bg-perturb 0.9` on 21 organisms leaves 98.9%
> of media growing, so the "titrate everything at once" collapse (which is real —
> see the legacy pipeline's `n_limiting`) does not happen with a random share.
> Test: `tests/test_cfs_sampling.py::test_background_is_perturbed_over_a_random_share`.

> **Addition (2026-08-31, measured).** The design lands on the growth plateau:
> **76%** of held-out media above 75% of the organism's max `mu`, 7% below 5%. That
> is the band Head A over-predicts in (§7) and the band §8.1's slow members live
> in, and no reweighting puts rows there. `SamplingConfig.frac_low_mu = 0.15` draws
> **1–3 metabolites at once below their own anchors**, the rest replete —
> anchor-relative, because an absolute band leaves a metabolite whose onset is
> `1e-6` replete at `1e-3`. True-LP counts over 3 organisms × 400 media:
> alive-but-slow rows 84 → 176, 116 → 193, 140 → 203, with dead media flat
> (7 → 8, 5 → 8, 17 → 15).
>
> **P24, and the budget was not it — the whole loop is measured, 2026-08-31.**
> Composition damage scales with community size: median log-X 0.009 → 0.014 at
> n=2 and **0.027 → 0.466 at n=21** (n=5 matched replicates, identical media).
>
> Co-limitation *did* collapse. Counting metabolites with a non-dust negative dual
> per medium (`alpha=1`, `eps=1e-3`, 21 organisms, 84 000 base media each):
>
> | | mean | ≥3 | ≥5 | ≥10 |
> | --- | --- | --- | --- | --- |
> | pre-relabel (`labels`) | 2.46 | 27.8% | 14.1% | 5.1% |
> | post-relabel (`labels_p2`) | 1.88 | 20.2% | 8.6% | 1.7% |
> | `labels_p3`, `frac_low_mu` paid out of `frac_focus` | 1.92 | 21.6% | 9.9% | 2.0% |
>
> By stratum (CP070062.1), `below Km` is the only source of co-limited media —
> 22.5% of it has ≥5, against 2.7% of `focus`, 3.4% of `above Km` and 0.5% of the
> low-mu stratum — and `frac_low_mu` takes 30% of it. So the obvious fix is to pay
> for the low-mu stratum out of `frac_focus`. **That was relabelled, retrained over
> 3 Head A seeds and scored over 5 replicates, and it does nothing:**
>
> | median log-X | n=2 | n=3 | n=5 | n=10 | n=21 | overall |
> | --- | --- | --- | --- | --- | --- | --- |
> | `r1` (pre-relabel) | 0.009 | 0.018 | 0.076 | 0.027 | **0.027** | 0.017 |
> | `p2` | 0.014 | 0.067 | 0.029 | 0.093 | 0.466 | 0.031 |
> | `p3` (budget fixed) | 0.024 | 0.049 | 0.022 | 0.082 | **0.783** | 0.030 |
>
> It costs Head A's worst organism (0.9633/0.9645/0.9662 over seeds → 0.9591/
> 0.9536/0.9350) and is **reverted in code**, with the measurement kept here.
>
> **Three things this refutes, so nobody re-runs them.** (a) The budget split is
> not the cause. (b) **Co-limitation count is not the predictor** — it moved the
> right way and the composition did not follow. (c) Scarcity does not compose:
> the low-mu stratum averages 1.0 binding uptakes per medium whatever band or
> subset size it gets (four bands, two budgets), so widening its subset to a drawn
> share of `A_i` — the `frac_bg_perturb` correction, one stratum over — is also
> rejected.
>
> **What it actually is: "replete" is now 2–6 decades above onset, and the strata
> that set it are still absolute.** The focus strata hold the non-focused columns
> at `log10(c/Km) ∈ [0, 1]` and the unfocused ones at `[-4, 1]` — neither is
> anchor-relative. Once `probe_lo` moved the onsets down, a background metabolite
> stopped being *near* its limit and became super-replete. Counting active
> metabolites within ±0.5 decades of their own anchor, at the 21-member community
> medium that fails on every Head A seed:
>
> | | community medium | training median | training p95 |
> | --- | --- | --- | --- |
> | `AAXE02`, pre-relabel | 1 of 16 | 7 | 11 |
> | `AAXE02`, post-relabel | 3 of 16 | **0** | 3 |
> | `GCA_000151225.1`, pre-relabel | 4 of 32 | 15 | 20 |
> | `GCA_000151225.1`, post-relabel | **10 of 32** | **0** | 7 |
>
> The community medium is inside the old design's distribution and outside the new
> one's, in the one coordinate that matters. It is not out of range per dimension
> (that was checked and is clean) — it is out of range in *how many dimensions sit
> near their onset at once*, which is exactly what a pool over the union of 21
> members' active subspaces looks like.
>
> **`SamplingConfig.focus_bg_decades = (0.0, 1.5)` is that fix**: the focus
> strata's non-focused columns are drawn that many decades above each metabolite's
> *own* anchor, capped at the rich level, instead of over an absolute `[0,
> log10_hi]`. This is not the "everything starves together" collapse (§4.3), which
> came from shifting whole bands so nothing was replete; here every background
> column stays at or above its own onset and only the focused one goes below.
> Probe, 2 organisms x 400 media, metabolites within ±0.5 decades of their anchor:
>
> | | near-onset median | p95 | dead media | colim ≥5 |
> | --- | --- | --- | --- | --- |
> | `r1` (pre-relabel target) | 11 / 15 | 15 / 20 | 0.3% | 6.8% / 63.5% |
> | `p3`, absolute background | 0 / 0 | 4 / 7 | 0.3% | 5.0% / 49.2% |
> | **`(0.0, 1.5)`** | **4 / 4** | **10 / 14** | 3.0% | 3.0% / 26.0% |
> | `(0.5, 1.5)` | 0 / 0 | 4 / 7 | 3.0% | 3.0% / 26.3% |
>
> `(0.5, 1.5)` cannot reach the window by construction and is dropped. `(0.0, 1.5)`
> covers the failing community medium's 3-10 near-onset metabolites at its p95 and
> keeps 97% of media growing. Note co-limitation *falls* — which is consistent with
> it not being the predictor.
>
> **It was relabelled and scored (`labels_p4`), and the composition did not move.**
> n=21 median log-X 0.466 (`p2`) → 0.783 (`p3`) → **0.706** (`p4`); overall 0.031 →
> 0.030 → 0.035. Head A's held-out metrics stay good throughout. The change is kept
> — `(0.0, 1.5)` is the more defensible definition of "replete" and costs nothing
> measurable — but it is **not** the fix, and the third proxy in a row to move as
> designed with no downstream effect. See §8.5 for the stock-take that follows.

### 4.4 Growth-rate grid

`alpha ∈ {0, 0.25, 0.5, 0.7, 0.85, 0.93, 0.97, 1.0}` — K=8, densified near 1
where dFBA lives and where `z` moves fastest.

### 4.5 Compute budget — read before generating

```
20 organisms × 20k media × (1 mu_max + 8 alpha) = 3.6M solves
at 3 epsilon levels (D7)                        = 10.8M solves
at ~50 ms                                       = 150 CPU-hours
```

That is 5 hours on 32 cores. Feasible, but:

**Reduce it.** Generate the full 20k at the middle `eps` only. Generate the
other two levels on a stratified 20% subset (4k media). The smoothing family
needs to span the same region, not resolve it at equal density. This cuts the
budget to ~64 CPU-hours.

Shard to parquet by `(organism, eps)`. Store `mu_max`, `z`, shadow prices,
solver status, and the medium vector. Do not store internal fluxes — they are
large and you do not need them.

### 4.6 Reserve for active learning

Hold back 20% of budget. After the first composition experiments, generate
samples at the allocations the master problem actually visits. Communities
create metabolite concentration profiles that no single-organism design will
have sampled — this is the P4 failure mode and passive sampling cannot fix it.

### 4.7 Band placement must be measured per (organism, metabolite)

> **Built (2026-07-27).** `cfs.sampling.active_subspace.demand_probe` +
> `design.band_scales`, run inside `generate_organism` and on by default
> (`SamplingConfig.probe`). On AAXE02 the probe anchors the same 12/16 metabolites
> the two-pass `u*` did, in **2 s** for the whole organism, and agrees within 1.5
> decades on all 12. Its target — the point where `mu_max` has recovered
> `target_frac` of that metabolite's own range — is **calibrated, not chosen**:
> median `log10` difference against the measured `u*` anchors over 4 organisms /
> 64 metabolites is −0.15 at `target_frac=0.05`, **+0.09 at 0.1** (the default),
> +0.46 at 0.25, +0.77 at the midpoint. At 200 media on 2 organisms the resulting
> coverage matches the `--scales` path metabolite for metabolite (roster median
> `top_share` 0.463 vs 0.466, `A_med` 9.75 both), with **no previous labels**.
> Provenance is in the sidecar: `probe` 12/16 and 25/30, the rest `default`.

> **Correction (2026-08-31, measured).** The probe's bracket was
> `log10(c/Km) ∈ [lo, hi]` with `lo = SamplingConfig.log10_lo = -4`, the *same*
> field that floors the sampling band. A metabolite whose onset is below `c/Km =
> 1e-4` therefore returns `mu_lo == mu_hi`, is omitted by the contract above, falls
> through to scale 1.0 — and its band could not have reached the onset even with a
> correct anchor. Roster-wide that was **100 of 496 active (organism, metabolite)
> bands**, sampled replete in every medium: the trace metals (`EX_cobalt2_e`,
> `EX_cu2_e`, `EX_mn2_e`, `EX_zn2_e`) whose ramps live at `c/Km ~ 1e-9…1e-6`.
> Downstream, `_kink_scale` takes its "never limits" fallback (`x_scale = 1`) and
> the head has ~4e-3 of its own input coordinate in which to separate `mu = 0` from
> the plateau — which is why M11's designer could zero an essential for free.
>
> `probe_lo = -12` is now a separate field. `log10_lo` stays at -4 for the
> *unfocused* strata, where widening it is the measured "everything starves
> together" collapse. On CP070062.1 the probe then anchors **23/23** instead of
> 16/23 for 0.5 s more probing, and anchors that already worked move ≤0.06 decades;
> on the full relabel, band sources are **494 probe / 2 previous / 0 default**.

**The bar is that it holds for any metabolite on any GEM without hand-tuning.**
The mechanism that produced the second run —
`limiting_scales` reading `u*` off the previous run's labels — works, but it is
two-pass and therefore not a design: a genome with no labels yet falls back to
scale 1.0 and its first pass is as skewed as the original. A roster-median prior
is not a substitute either; `u*` is stable across organisms for the ions (2–5×)
but spans 2585× for `EX_arg__L_e`.

What it has to be instead, in order:

1. **A pre-sampling probe, not a previous run.** Per organism, per metabolite in
   `A_i`, bisect the uptake bound for the point where `mu_max` starts to fall —
   the same LP probe as legacy `estimate_demand`, a few solves per metabolite
   against the ~32 000 the organism's labels cost. It needs no labels, no model
   and no human, so it runs inside `cfs generate` for a genome the roster has
   never seen. This is what makes the design self-anchoring.
2. **Fall back in a stated order:** probe → previous run's `u*` → roster-median
   `u*` for that metabolite → 1.0. Record which was used per metabolite in the
   `<id>.subspace.json` sidecar; a band anchored at the default is a known blind
   spot, not a silent one.
3. **Verify by coverage, not by eye.** The acceptance number is per-metabolite,
   over `A_i`: the *median* metabolite's share of media and the count of
   metabolites below ~50 media. Coverage is the quantity that predicts held-out
   gradient cosine (Spearman 0.72), so it is checkable before any training runs.
   `check_coverage.py` in the run dir is the current form of this.

Two things deliberately **not** in that list. Per-metabolite budget weighting by
measured held-out error (`design.topup_weights`) is a refinement on top of a
correct anchor, not a replacement for one — it needs a trained model and a
val split, and it inherits whatever the anchor got wrong. Ensemble gradient
disagreement is for the 4–10 metabolites per organism that never limited at all
and so have neither a probe result nor a measurable error; it is the last resort,
not the first (see the ordering rationale in §4.6 and P4).

Both now exist as the §4.6 top-up loop, *after* the anchor: `cfs topup` turns
`train-value`'s held-out `per_limiting_metabolite` into focus weights (and gives
the never-measured metabolites the floor share rather than nothing), and
`cfs generate --focus-weights --round N` spends them, writing
`part.round<N>.parquet` beside the base shards — `load_value_dataset` reads every
parquet in an `(organism, eps)` directory, and round `N`'s `medium_id`s are
offset so the by-medium train/val split cannot fuse two rounds' media.

---

## 5. D4 — the uniqueness decision, expanded

You asked to keep this open. Here is the decision procedure rather than a
decision, because the right answer depends on a measurement you have not yet
taken.

### 5.1 What uniqueness you actually need

This is the crux and it is narrower than it first appears.

- **`mu_max` is always unique.** The optimal objective value of an LP is
  unique regardless of degeneracy. Head A's *values* are safe under any choice.
- **Shadow prices are not.** Primal degeneracy means multiple dual solutions,
  so `pi` can be solver-arbitrary. This threatens your Sobolev term.
- **Internal fluxes are not, and you do not care.** You never predict them.
- **Exchange fluxes `z` are the question.** Many alternate optima differ only
  in internal routing and produce *identical* exchange profiles.

So the real question is: **are your exchange fluxes already unique at the
optimum?** If yes, you need no regularisation for Head B at all.

### 5.2 The diagnostic that decides it

Run this before choosing. It is cheap.

```python
def exchange_degeneracy(model, alpha=1.0):
    """FVA restricted to exchange reactions at fixed growth."""
    sol = model.optimize()
    with model:
        model.reactions.get_by_id(BIOMASS).bounds = (alpha*sol.objective_value,)*2
        fva = cobra.flux_analysis.flux_variability_analysis(
            model, reaction_list=model.exchanges, fraction_of_optimum=1.0)
    return (fva.maximum - fva.minimum)
```

Run across a stratified sample of ~200 media per organism, at `alpha ∈ {1.0,
0.7}`. Record the distribution of ranges.

**Interpretation:**

| Result | Meaning | Action |
|---|---|---|
| Ranges < 1e-6 almost everywhere | Exchange fluxes already unique | Use plain FBA for Head B. No regularisation needed |
| Ranges large on a few metabolites | Localised degeneracy | Regularise, but only weakly. Check *which* metabolites — often it is a redundant transporter pair |
| Ranges large on many metabolites | Genuine degeneracy | Full elastic-net scheme (§5.4) |

Expect the third case at `alpha < 1` — a sub-maximal growth rate leaves slack,
which is exactly what creates alternate optima. So even if `alpha = 1` is
clean, the K=8 grid probably is not.

### 5.3 The option space, honestly

**pFBA.** Minimise total absolute flux subject to optimal growth. Biologically
motivated (minimal enzyme investment), fast, standard.
*Problem:* still an LP, so still has vertex solutions and can still tie. It
reduces alternate optima substantially but does not eliminate them, and the
remaining ties are exactly the symmetric-pathway cases most likely to differ
in exchange profile. **Not sufficient alone for your purposes.**

**L2-regularised QP.** Minimise `||v||²` subject to growth ≥ target. Strictly
convex, so the primal is unique and the value function is differentiable.
*Problem:* L2 spreads flux across parallel pathways, which is not what cells
do. At moderate weight the flux distributions become unbiological. It also
destroys sparsity, which matters if you ever want to interpret the fluxes.

**Elastic net — `||v||₁ + (ε/2)||v||²`.** Sparse like pFBA, strictly convex
like L2. Unique primal, unique-enough duals, smooth value function, and the
sparsity structure survives at small `ε`.
*Cost:* a QP rather than an LP, so ~2–5× slower. Given §4.5 this is affordable.

**Loopless FBA.** Adds thermodynamic constraints. MILP, an order of magnitude
slower, and infeasible at your solve count. Use it once during the EGC
pre-flight, not in bulk generation.

**Flux sampling and averaging.** Gives a "typical" flux but the mean is not an
FBA solution, and the cost is prohibitive at 10M solves.

### 5.4 Recommendation

**Elastic net with `ε` small, contingent on §5.2 coming back non-clean.**

```
minimise   ||v||_1 + (eps/2) * ||v||^2
s.t.       S v = 0
           lb(c) <= v <= ub
           v_biomass = alpha * mu_max
```

If §5.2 returns clean exchange fluxes at all `alpha`, drop to plain FBA and
save yourself the QP. Let the diagnostic decide.

#### D4 decision — elastic net (2026-07-26)

The M1 diagnostic (`cfs degeneracy`, exchange-FVA at `alpha ∈ {1.0, 0.7}`) came
back firmly in the **genuine-degeneracy** regime, so Head B labels are generated
with the elastic-net QP above.

Evidence, on the CarveMe model `FNPN01` (259 exchanges, biomass `Growth`, EGC-free):

- **69% of exchange-flux observations were degenerate** (FVA range `> 1e-6`)
  across the surveyed media × `alpha` grid — far above the "few metabolites"
  localised threshold. `recommend_d4` → `"genuine"`.
- The degeneracy concentrated at `alpha < 1`, exactly as §5.2 predicts:
  sub-maximal growth leaves slack that opens alternate optima, and the K=8 grid
  (D7) spends most of its mass there. Plain FBA and pFBA (§5.3) would hand the
  network label noise on the majority of samples.

Consequence: labels are the `ε`-family (D7/§5.5) `ε ∈ {1e-2, 1e-3, 1e-4}`, the
middle level primary. This is the same knob as the smoothing scale `τ` (§5.5),
so there is one regularisation parameter, not two.

Caveat (P15-adjacent): this is a single-model read. Re-run `cfs degeneracy` on
the full roster before generating the bulk label set; the decision holds unless
the roster-wide survey is dramatically cleaner.

##### Roster-wide confirmation (2026-07-26) — the caveat is closed

`--stage qc` over all 21 CarveMe models (50 media × α ∈ {1.0, 0.7} each,
371 400 exchange-FVA observations):

- **68.9% of observations degenerate** roster-wide; `recommend_d4` → `"genuine"`.
- Per organism the range is **59.7% – 82.8%** — not one outlier, the whole roster.
- Split by growth fraction: **88% at α = 0.7 vs 50% at α = 1.0**, exactly the §5.2
  prediction that sub-maximal growth opens alternate optima. The K=8 grid spends
  most of its mass at α < 1.
- V0 also passes: **21/21 models EGC-free**. Frozen index: 444 exchanges,
  **365 shared / 79 private**, so the Newton Jacobian is 365 × 365.

Elastic net stands as D4.

##### QP backend: Clarabel, not HiGHS (2026-07-26)

The scheme above is unchanged; the solver behind it is. HiGHS's QP active-set
method returned `"solve error"` / `"not set"` on **~30% of real CarveMe solves at
the primary `eps = 1e-3`**, and at `eps = 1e-4` it stalled for minutes on media
where growth is zero. Clarabel (sparse interior point, MIT, `pip install
clarabel`) solved the same batch **48/48 at every `eps` level and ~6× faster**;
`mu_max` is identical and `z` agrees with HiGHS's successful solves to its looser
convergence (median difference exactly 0). Zero-growth solves now short-circuit
to `v = 0` analytically — with growth fixed at 0 and the origin feasible, that is
the exact elastic-net optimum, not an approximation.

§3.4 re-verified on a real GEM after the swap (`check_v2.py` in the run dir):
repeat solves agree to 5e-13; a 1e-6 concentration perturbation moves `z` by
3e-8; finite-differenced `dmu_max/d(uptake bound)` matches the returned duals to
2e-12 (`corr = -1.0000` — the dual is the negated derivative).

### 5.5 D4 and D7 are the same knob

This is worth stating explicitly because it simplifies the implementation.

Tikhonov regularisation of the primal smooths the value function. At `ε > 0`
the primal solution is unique and continuous in `c`, so `mu_max_ε(c)` is
continuously differentiable and its gradient — the shadow-price vector —
varies smoothly rather than jumping at vertices. As `ε → 0` you recover the
true LP.

That is exactly what D7's smoothing scale `τ` was for. **They are the same
parameter.** Set `τ = ε` and generate the three-level family by solving at
three regularisation weights.

This is better than smoothing only the network, because the smoothing is now
in the *labels*. The network is learning a genuinely smooth function rather
than being forced to blur a kinked one.

Recommended levels: `ε ∈ {1e-2, 1e-3, 1e-4}`.

Use them as: large `ε` for well-conditioned Newton solves and homotopy
continuation; small `ε` for final accuracy; the middle level as the primary
training set.

### 5.6 Consequence for network activations

Because the labels are now smooth, the ReLU-Hessian problem (P3) still
applies but for a different reason: you need the *network* to have curvature
because Newton differentiates the network, not the labels. Softplus or ELU in
the convex pathway remains mandatory.

---

## 6. Phase 3 — architecture

### 6.1 Stacked parameters for N=20

With D1(a) and identical architectures, do not write a loop over 20 models.

```python
# Stack parameters with a leading organism axis, vmap the forward pass.
@eqx.filter_vmap
def batched_value_head(model, log_c):
    return model(log_c)

# models: a single PyTree with leading dim 20
mu = batched_value_head(models, log_c_batch)   # (20,)
```

20 organisms then cost approximately what 1 costs on GPU. This is the single
biggest engineering win available at your scale and it should be in from the
start — retrofitting a loop-based implementation to vmap is painful.

Organism-specific masks are applied as a `(20, M)` boolean array, not as
different architectures.

### 6.2 Head A

```python
class ValueHead(eqx.Module):
    """mu_max as a concave function of log-concentrations."""
    # partially input-convex, negated for concavity:
    #   - non-negative pass-through weights on the log_c pathway
    #   - softplus activations (NOT relu)
    #   - unconstrained pathway for any conditioning inputs
    def __call__(self, log_c: Float[Array, "M_i"]) -> Float[Array, ""]:
        ...
```

Input is the organism's own `M_i`, masked from the shared vector.

### 6.3 Head B

```python
class BehaviourHead(eqx.Module):
    """Net exchange flux per unit biomass at normalised growth rate."""
    def __call__(self, log_c, alpha) -> Float[Array, "M_i"]:
        # predict uptake and secretion as separate non-negative heads,
        # return the difference — makes sign structure explicit
        ...
```

No convexity constraint. Output masked to `M_i`, scattered to `M` at
composition time.

> **Implementation status (2026-08-28) — M4 built.**
> `src/cfs/surrogate/behaviour.py`, CLI `cfs train-behaviour`. A masked softplus
> MLP over `(x, alpha)`, `x` the same saturation coordinate Head A uses, trained
> on the alpha grid the §4.4 labels already carry (8 levels x 4000 media). Both
> heads come off `cfs.surrogate.data._stack`, so they cannot silently disagree on
> `x_scale` or on which media are held out; `compose.dfba.Surrogate` re-checks
> both plus `index_hash` before composing (P13/P14).
>
> 21 organisms, 600 epochs, lr 1e-3, held-out round-0 media: worst **R² 0.856**,
> median 0.921; worst per-organism median flux cosine **0.993**; worst sign
> agreement 0.941.
>
> **Two departures from the sketch above, both deliberate.**
>
> The separate non-negative uptake and secretion heads are *not* built. A
> difference of two non-negative outputs is any real number, so it constrains
> nothing — it is presentational, and one signed output is the same function
> class in half the parameters.
>
> The head emits `z / z_scale`, **not** `z`, with the per-(organism, metabolite)
> label scale stored in the checkpoint and applied by `behaviour.flux`. This is
> load-bearing, not cosmetic: raw-unit output scores held-out **R² 0.017** —
> worse than predicting the per-alpha mean — against 0.885 for the identical net
> on the normalised target. Exchange fluxes span O(400) on the gases to O(1e-3)
> on the ions *within one organism*, so a `sqrt(2/n_in)` init starts ~400x short
> on the dimensions carrying the variance and Adam spends the run walking biases.
> Same failure as §7's input-side scale problem, in the output layer.

> **Update (2026-08-30) — the head predicts *specific* flux `z / mu_max`, and the
> magnitude comes from Head A.** Exchange flux is very nearly proportional to how
> fast the organism is growing. Measured on these labels, a model with no inputs at
> all — one constant per (metabolite, alpha) times `mu_max` — explains a median
> **0.807** of the held-out `z` variance (0.63–0.86 over the 21 organisms) against
> the trained net's 0.921. So most of what Head B was learning was a magnitude
> Head A predicts directly and accurately (`mu_rel` <= 3% on every §8.1 community).
>
> Leaving it inside Head B is what broke §8.1's small communities: at a scarce
> community medium the head produced a replete organism's fluxes (|z| 2780 against
> a true 1040, flux cosine 0.40, on an organism whose held-out p05 is 0.93) while
> Head A had `mu` right to 1%. The divisor is floored at
> `data._MU_FLOOR_FRAC` = 5% of the organism's mean `mu` — 1% of media have
> `mu_max` below 1% of the median and dividing by those turns the target into
> noise, the same trade `calibrate._W_FLOOR` makes. The floor is in the checkpoint
> as `mu_floor`; a checkpoint without it reads as flux directly.
>
> **A second, free constraint: §3.3's own uptake bound.** The LP that made the
> labels cannot take up faster than `-Vmax_m * u_m`, and every exchange of every
> roster GEM has `|lower_bound| = 1000`, so the bound is a constant times the
> head's own input saturation — no fit, nothing stored. Head B has no such
> constraint and broke it badly: at the worst community, **28 of one member's 213
> exchanges** were below the floor at once, by up to 186x, and those were the same
> entries leading the `dc` error. `compose.dfba.Surrogate.mu_and_z` clamps. It is a
> projection onto a convex set the true `z` is already inside, so the right-hand-side
> error cannot rise — and does not: `dc_rel` fell on 10/10 communities. **A strictly
> better right-hand side is not a monotonically better trajectory**, though: the
> clamp changes which metabolite empties first, and a batch culture's endpoint turns
> on that, so two of ten trajectories got worse while all ten right-hand sides
> improved.
>
> After both changes plus the §4.3 community-regime round, held out on the same 800
> media: worst **R² 0.907**, median 0.952, worst sign agreement 0.957.

---

## 7. Phase 4 — training

> **Implementation status (2026-07-27) — M3 Head A built, gate not met.**
> `src/cfs/surrogate/` (`picnn.py`, `data.py`, `train.py`), CLI `cfs train-value`.
> Measured held-out, in `u = c/(Km+c)` space (never in the network's own input
> coordinate — a cosine there is a different number for every input transform),
> 21/21 organisms, 1500 epochs, `w_grad = 1`:
>
> | | 20hm labels | 20hm_bands labels |
> |---|---|---|
> | worst gradient cosine (the gate) | 0.629 | **0.733** |
> | mean | 0.765 | 0.800 |
> | per-row p05 | 0.214 | 0.200 |
> | value R² | 0.722 | 0.538 |
> | Hessian cond, median | 2.0e9 | 3.4e7 |
> | concavity violations | 0 | 0 |
>
> The gate is 0.99. Same architecture, same hyperparameters, *only the label
> design changed* (§4.3 correction / §4.7): +0.10 worst cosine and 60× better
> conditioning came from coverage alone. What that leaves:
>
> - **The tail did not move** (p05 0.21 → 0.20) and the leading error metabolites
>   are unchanged — `EX_mg2_e` in 21/21 organisms, `EX_cl_e` 19, `EX_ca2_e` 17,
>   with `EX_k_e` receding 16 → 11 exactly as its share of media fell. 84% of the
>   587 (organism, metabolite) cells are still below 0.9. Coverage was *a*
>   constraint, not the only one.
> - **R² fell 0.72 → 0.54** at fixed `w_grad`: harder gradient targets now eat the
>   value head's budget. This is the same trade the `w_grad` sweep showed
>   (1 → cosine 0.63 / R² 0.72; 10 → 0.66 / 0.62), reached via the labels instead,
>   and it is not something retuning `w_grad` settles — neither end satisfies the
>   "R² ≥ 0.9" balance rule. Expect this from **architecture and scale** (the run
>   is 4000 media, 1/5 of the D10 budget), not from loss weights.
> - Architecture already tried and rejected: a soft-min ("Liebig") head matching
>   the target's sparsity (54.7% of rows have exactly one non-zero dual) scored
>   *worse* — 0.55 worst — with Hessian conditioning 6–16 orders worse.
>
> Load-bearing pieces of the current head, in the order they were found: a
> saturating per-metabolite input rescale `x = u/(u+s)` (a linear one cannot work —
> reaching the ions' ramp at `u ~ 1.4e-4` sends replete dims to `x ~ 1e4`); a
> **monotone non-decreasing** head, required for `f(h(x))` to stay concave under a
> concave input map, and true of the target; a **per-row norm-relative** Sobolev
> term instead of §7.1's absolute one, with the all-zero-target rows (23%)
> included; ICNN init at `softplus^-1(1/width)`.
>
> Two label-interpretation bugs this uncovered, handled in
> `cfs.surrogate.data._organism_arrays`: the stored `shadow` is
> `d(mu_max)/d(uptake bound)` **only where that bound binds** — elsewhere it is the
> metabolite's value in the network, positive for waste like CO2, which no LP can
> mean (12/12 finite-difference checks returned exactly 0), so it is clamped at 0;
> and half the "non-zero" duals are solver dust at O(1e-14), which would dominate
> any norm-relative loss by ~1e14.

> **Architecture trial (2026-07-29) — the ICNN survives; the deficit is located.**
> `cfs train-value --arch {icnn,icnn-u,deepset,deepset-private,mlp}` and `cfs baseline-rf`,
> all on `20hm_bands/` at identical knobs and the *same* 800 held-out media.
>
> | | worst | mean | R² | Hess. cond | train loss (value) |
> |---|---|---|---|---|---|
> | icnn | **0.733** | **0.800** | 0.538 | 3.4e7 | 0.62 (0.375) |
> | deepset, shared trunk | 0.259 | 0.432 | 0.111 | **4.0e3** | 1.35 (0.895) |
> | deepset, private trunks | 0.374 | 0.601 | 0.388 | 1.2e5 | 0.99 (0.626) |
> | unconstrained MLP | 0.243 | 0.768 | 0.797 | 7.1e10 | 0.37 (0.216) |
> | random forest | probe-limited | | **0.979** | n/a | n/a |
>
> - **The value function is nearly perfectly learnable and the ICNN is far from
>   it.** The forest scores R² 0.959–0.996 on *every* organism. R² 0.538 is an
>   architecture deficit — not a label ceiling, not solver noise, not sampling.
> - **The concave family is not the ceiling.** Removing both constraints moves the
>   median cosine +0.009 while the worst organism collapses 0.733 → 0.243 and 43.8%
>   of Hessians go non-concave. The constraints are ~free on accuracy and hold the
>   worst case together, so **P11's difference-of-convex escape hatch is not worth
>   its complexity**.
> - **The ions are an architecture failure, not an intractable cell.** The forest
>   reads `EX_mg2_e` at ~1.000 on 20/21 organisms where the ICNN manages 0.21–0.98.
>   Mg limitation is an axis-aligned kink in one coordinate: native to a tree split,
>   evidently not localisable by a dense smooth net of width 128 over 444 inputs.
>   The remaining error is **localisation**.
> - **Sharing `phi` across organisms hurts** (private beats shared on every metric),
>   so D1's supersession is not worth taking.
>
> **Caveat, since discharged.** `phi` is priced *per metabolite*, so at the design
> width (`width // 2` = 64) the deepset cost 123× the ICNN's FLOPs — 47 s/epoch vs
> 0.35. It was cut to `width // 8` = 16 **for speed**, and both variants then
> *underfit the training set* (value loss 0.895 / 0.626 vs the ICNN's 0.375), so
> the two rows above are a lower bound and never refuted the architecture. The
> full-width rerun did — see below.
>
> **Deepset at full width — measured 2026-08-27, and cut.** Four `deepset-private`
> cells × 21 organisms on the M3b cluster run, x-space, `w_grad` 1. Best cell
> (`ph32/kc64`): worst cosine **0.742**, median 0.810, median R² 0.564 — against
> `icnn w128/d3`'s 0.678 / 0.755 / 0.541. It wins on 18/21 organisms by a median
> +0.055 cosine, at identical R², for ~157 cpu-h a cell against 1.6.
>
> - **The conditioning advantage is gone**, and it was the reason to revisit the
>   arch. At full width median Hessian condition is 1e11–1e12, *worse* than the
>   ICNN's 1e10. (And §8 does not pay that bill — see the master-Jacobian result.)
> - **It does not fit the per-metabolite cells better**, which was the bet. The
>   lift is uniform, not concentrated where coverage is thin: <25 rows +0.012,
>   25–100 +0.062, 100–400 +0.020, ≥400 +0.010. Where the ICNN scores <0.5 (284
>   cells) deepset scores 0.312 vs 0.286; where it is ≥0.9 (72 cells), 0.958 vs
>   0.954. The same three ions lead the error on ~20/21 organisms in both.
> - **Both capacity axes are inert**, like the ICNN's: paired per-organism median
>   Δcosine is −0.001 for `k_code` 16→64 and +0.001 for `phi` 32→64.
> - **`deepset-u` inherits the coordinate fix but not the bigger lever.**
>   `groupmax.init_from_tangents` seeds layer 1 with the labels' duals as affine
>   planes over the whole metabolite vector (cosine 0.598 → 0.973). Deepset's first
>   layer is a per-metabolite *scalar* `phi_m: R → R^k` — there are no planes to
>   seed. It would land near `icnn-u`, at ~100× the compute.
>
> Reopening test, if wanted: one organism (CR626927.1), `deepset-u` ph32/kc64 at
> `w_grad` 10, ~7.5 h, against `icnn-u` 0.903 and seeded `groupmax-u` 0.973 on the
> same held-out media. The arch stays built and registered, so it costs no code.
>
> **The forest's gradients are probe-limited.** No analytic gradient, so
> `baseline.py` uses central differences at `delta * s_m` in `u`. Worst-organism
> cosine is U-shaped in delta (0.398/0.451/0.374/0.209/−0.102/−0.023 at
> 0.01/0.02/0.05/0.25/0.5/1.0) and the ends hit different metabolites — carbon
> sources invert with a large step, ions fall off with a small one. Do not quote an
> aggregate; quote a cell only where it is flat in delta (`EX_mg2_e` is:
> 0.998/0.996/0.993/0.982 across 0.05→1.0, which is why the ion result stands).
> R² 0.979 is delta-independent.
>
> **Also fixed: the held-out set used to move.** `load_value_dataset` permuted all
> media including §4.6 top-up rounds, so top-up media — drawn where the model is
> *worst* — entered validation and made the test harder each round (one organism:
> 491 → 595 → 781 usable rows over p1→p3). Val is now round-0 media only, and
> `diagnostics.json` records `n_val_media` / `n_train_media` / `rounds_present`.
> **The earlier finding that top-up rounds hurt is retracted** — it was never a
> controlled experiment. Re-running it on the fixed ruler is open work.

> **M3b ran (2026-08-26) and both of the bullets above about *where* the deficit
> lives are superseded — see `CLAUDE.md` for the full read.** The short form: the
> ICNN's capacity axis is inert (width 128→1024 × depth 3→6 moves paired
> per-organism cosine and R² by ±0.001 on all 12 cells) because **concavity was
> imposed in the wrong coordinate**. `mu_max` is an LP value function in its RHS
> and `lb = -Vmax * u`, so it is concave and piecewise *linear* in `u`; but
> `u = s*x/(1-x)` is **convex** in `x`, so a concave-in-`x` head fits each ramp
> with a chord. Tangent test on the labels: violated in `x` on 32.3/39.7/56.9% of
> row pairs (3 organisms), in `u` on **0.0%**. `{concave in x} ⊊ {concave in u}`
> and the target is in the gap — every variant converges to the same projection,
> which no amount of capacity moves.
>
> Consequences for the decisions recorded above:
>
> - **"The remaining error is localisation" is withdrawn.** The ions are the
>   metabolites whose ramps are steepest in `u`, i.e. the ones the `x`-space chord
>   approximates worst. A per-metabolite architecture does not address this; the
>   `deepset`'s `phi` is concave in `x_m` and carries the identical defect.
> - **The forest is no longer the ceiling.** The parameter-free cutting-plane model
>   `mu_hat(u) = min_j [mu_j + pi_j.(u − u_j)]` scores cosine **0.969–0.996** and R²
>   **0.997–0.999** on 6/6 organisms — beating the forest on gradients and the MLP
>   on both, while being concave, monotone and analytically differentiable. One
>   organism clears the 0.99 gate. Use it as the ceiling measurement; it needs no
>   finite-difference probe, so the delta caveat below no longer applies to it.
> - **P11 stays parked, but for a new reason.** The MLP's advantage was read as
>   +0.009 cosine when the constraint was in `x`; it is now clear the constraint was
>   never the problem, the coordinate was.
>
> `cfs train-value --arch icnn-u` (`src/cfs/surrogate/picnn_u.py`) is the
> correction — the same ICNN over `w = min(u/s, 300)`, affine in `u`, so the class
> is the full concave-in-`u` one. R² 0.477 → **0.802** on one organism at identical
> knobs, concavity violations still 0. Its cosine has not moved yet.

> **Second finding, 2026-08-26: initialisation, not architecture.** With the
> coordinate corrected, the remaining gap is where the optimiser *starts*. Matched
> A/B on CR626927.1 — `groupmax-u`, width 1 / depth 1 / K=250 / T=0.03 / `w_grad`
> 10, same seed, only `--gm-init` differing:
>
> | init | cosine | R² | p05 | Hessian cond |
> |---|---|---|---|---|
> | `labels` | **0.9733** | **0.996** | **0.834** | 1.9e24 |
> | `random` | 0.5982 | 0.4795 | 0.000 | **0.0** |
>
> Condition exactly 0 means no curvature anywhere: the head collapsed to a *single*
> affine piece, so 249 of 250 planes never became active and never received useful
> gradient. This is the dead-piece failure LSPA/CAP-style max-affine fitting exists
> to fix, and it reproduces at width 128 / depth 3 too (cosine 0.76, cond 0.0 at
> both T=0.01 and T=0.03), so it is what random init does to a group-max head
> generally rather than a quirk of the narrow configuration.
>
> The remedy is **not** those fitting algorithms: they infer the planes from values,
> and §5's labels give the duals directly, so every row is an exact supporting
> hyperplane and the first layer can simply be told what its pieces are. Ranked by
> active-set frequency (the dual's support is the LP basis), 100 planes match ~2000
> drawn at random. `cfs train-value --arch groupmax-u --gm-init labels`.
>
> At K=250 the seeded head beats the full 16 000-tangent cutting-plane model
> (0.9733 vs 0.969) on 111k parameters, concave and monotone with violations 0 —
> the closest anything has come to the 0.99 gate, on one organism.
>
> **This makes conditioning the binding constraint, and it is now V-gate material.**
> 1.9e24 is the worst number measured and it is precisely what §8.4's Newton spends
> (`tags=lx.positive_semidefinite_tag`, `rtol=1e-10`). Sharp affine pieces buy
> gradient accuracy and cost curvature; the log-sum-exp temperature is the knob, and
> the second sweep's Arm G traces that frontier. If no temperature is both accurate
> and conditionable, the answer is P9's remedy — damped Newton / trust region — not a
> better head. **§7.3's diagnostic set should treat `hessian_cond_median` as an
> objective, not a warning light.**
>
> Related work, for the record: GroupMax (arXiv 2206.06622, motivated by Bellman
> *cuts*, i.e. exactly this), Maxout (1302.4389), Magnani & Boyd's LSPA and
> Hannah & Dunson's CAP/AMAP for max-affine fitting.

> ### 7.4 M3b — HPC sweep (next)
>
> Everything above is one laptop, 4000 media/organism (1/5 of the D10 budget), one
> width, one depth. Two of the four findings point the same way — **the models are
> underfitting, not overfitting** (the deepsets on training loss outright; the ICNN
> because a forest reaches R² 0.98 on the same rows) — and neither more labels nor
> more width has actually been tried. That is the sweep.
>
> Scale, in the order it matters:
>
> 1. **Rows.** D10's full 20000 media/organism, 5× the current set. Generate with
>    `--stage labels` (§4.5); the §4.7 probe means no previous run is needed.
> 2. **Width/depth.** The laptop never went above 128×3. Sweep width
>    {128, 256, 512, 1024} × depth {3, 4, 6} for the ICNN.
> 3. **DeepSet at full width and beyond** — `phi` hidden {32, 64, 128}, `k_code`
>    {16, 64, 128}. This is the arm that was cut for laptop runtime and it is the
>    one with the localisation inductive bias the forest showed is worth having,
>    plus 4 orders better conditioning. `--emb-dim` {8, 16, 32}. Its cost is
>    per-metabolite, so it is the arm that most needs the cluster.
> 4. Keep `mlp` and `baseline-rf` in the sweep as the ceiling and the floor: both
>    are cheap and both changed the reading of the ICNN's numbers here.
>
> Mechanics: the organism axis shards cleanly (`train._shard_organisms`, 1.8× on
> CPU with `XLA_FLAGS=--xla_force_host_platform_device_count=N`); on GPU the
> existing `filter_vmap` stacking is the win §6.1 describes. A `train` stage in
> `main.nf` fanning out over (arch × width × depth × n_rows) cells with
> `COLLECT_METRICS` on `diagnostics.json` mirrors the legacy sweep's shape.
>
> Gate for M3b: does **any** cell reach R² ≥ 0.9 with worst cosine ≥ 0.9? If the
> ICNN's R² does not move with 5× rows and 8× width, the localisation story is
> confirmed and the answer is a per-metabolite architecture, not scale.

### 7.1 Loss

```
L = w_v * (mu_hat - mu)^2
  + w_g * ||grad_c mu_hat - pi||^2        # Sobolev — do not skip
  + w_z * ||z_hat - z||^2
  + w_m * ||sum_m z_hat_m * atoms_m||^2   # optional elemental balance
```

Take `grad_c mu_hat` by autodiff of Head A, never as a separate output head.
A separate head is not constrained to be the derivative and breaks concavity.

The gradient term is the one that matters. Both the master problem and HMC
follow slopes and never look at values.

**`w_v` alone is measured to be the wrong value term (2026-08-29).** It is
absolute, and 74% of held-out rows sit on the plateau, so the media below a
quarter of max `mu` carry no weight and the head over-predicts *every one of
them*. `cfs train-value --w-rel` adds the same error measured relatively,

```
+ w_rel * ((mu_hat - mu) / (mu + 0.1 * mean_media(mu)))^2
```

with the denominator floored at 0.1 of the organism's mean `mu` — without the
floor the plateau goes unweighted and value R² collapses to −1.66. It removes
the low-`mu` bias outright at **no cost in gradient cosine**, but it buys the
bottom by selling the plateau, and a large community's log-X error is mostly
plateau. Default 0; see the low-`mu` status block below for when to raise it — and
prefer the output calibration (`cfs.surrogate.calibrate`, recorded in the same
section), which makes the same correction *after* training at no cost to the gate.

### 7.2 Schedule

1. Head A alone, value + gradient loss, until gradient cosine plateaus.
2. **Freeze Head A.** Train Head B.
3. Optional joint fine-tune at low LR.

Freezing is not optional at step 2: Head B's `alpha` is defined relative to
Head A's output, so a moving Head A makes Head B's targets non-stationary.

### 7.3 Diagnostics

- Gradient cosine similarity to true shadow prices, per organism.
- Concavity violation rate on random convex combinations.
- Hessian condition number of Head A. Exploding condition number predicts
  Newton failure in Phase 5.
- Per-metabolite gradient error — expect the worst errors on metabolites in
  `A_i` near their limitation boundary, and check that is where they are.
- **Relative value error and bias below 25% of max `mu`** (`value_rel_err_low_mu`,
  `value_bias_low_mu`). Nothing else here can see it: value R² and the MSE are
  absolute, so a head reads R² 0.99 while over-predicting a starving medium by
  +98%, and its *gradient* cosine on those same rows is 1.000. §8 integrates
  `d(log X)/dt = mu`, so this is the number a slow community member feels.

> **Head A over-predicts at low `mu` — measured 2026-08-29, and the temperature
> is the lever.** This is §8.1's "the next Head A signal is accuracy at low `mu`",
> carried out. Held-out media binned by each organism's own `mu / max(mu)`,
> `value_ra3` (seeded `groupmax-u`, K=1000, T=0.03, `--gm-reanchor 3`, `w_grad`
> 10), 21 organisms:
>
> | band | rows | median rel err | median bias | grad cosine |
> |---|---|---|---|---|
> | < 5% of max `mu` | 52 | 0.978 | **+0.978** | 1.000 |
> | 5–10% | 26 | 0.427 | +0.427 | 1.000 |
> | 10–25% | 29 | 0.188 | +0.188 | 1.000 |
> | 25–50% | 42 | 0.117 | +0.117 | 0.994 |
> | 50–75% | 33 | 0.065 | +0.065 | 0.943 |
> | > 75% | 586 | 0.012 | −0.012 | 0.959 |
>
> **100% of held-out rows below 75% of max `mu` are over-predicted**, and the
> error is pure bias — median |rel| equals median signed rel in every low band. It
> is invisible to every diagnostic that existed before this: value R² is 0.986 and
> the *gradient* cosine in those bands is 1.000, better than on the plateau. §7.3
> now reports it.
>
> **It is not labels, coverage, or the plane budget.** The parameter-free
> cutting-plane model over the same organism's training tangents has bias
> **−0.000 in every band including <5%**, and so does the K=1000 subset picked by
> `rank_by_active_set` — that is, the head's own *initialisation*. Training
> creates the bias. Two absolute offsets do it:
>
> 1. the smoothed group max sits `~T*ln(K_active)` **below** the hard min — ~0.2 in
>    `mu_scale` units at T=0.03, which is 4% of a plateau `mu` and >100% of a
>    starving one. The 1-epoch seeded head reads −1.207 at <5% and −0.040 at the
>    plateau, exactly that shape;
> 2. §7.1's value term is an absolute MSE with 74% of rows on the plateau, so Adam
>    removes the offset where the rows are and lifts the bottom straight past the
>    target.
>
> **`--gm-temp 0.01` fixes more of it than anything in the loss, and it is what §8
> feels.** Roster (21 organisms) plus the same 10 communities as the §8.1 block,
> per-organism FBA truth, seed 0, everything else identical:
>
> | run | low-`mu` bias | plateau rel | worst cos | med R² | log-X err, sizes 2/3/5/10/21 |
> |---|---|---|---|---|---|
> | T=0.03 (`value_ra3`) | +0.978 | 0.012 | **0.958** | 0.986 | 0.055 / 0.122 / **0.322** / 0.041 / 0.044 |
> | **T=0.01 (`value_T01`)** | +0.442 | **0.005** | 0.928 | **0.990** | **0.034 / 0.050 / 0.051 / 0.048 / 0.047** |
> | T=0.01, `--w-rel 0.3` | +0.100 | 0.014 | 0.924 | 0.989 | 0.058 / 0.082 / 0.060 / 0.065 / 0.064 |
> | T=0.01, `--w-rel 1` | −0.002 | 0.031 | 0.944 | 0.986 | 0.095 / 0.101 / 0.085 / 0.140 / 0.138 |
> | T=0.003 | +1.305 | 0.018 | 0.880 | 0.979 | 0.150 / 0.555 / 0.293 / 0.394 / 0.352 |
> | anneal 0.03 → 0.01 | −0.101 | 0.026 | 0.928 | 0.989 | 0.089 / 0.146 / 0.137 / 0.098 / 0.099 |
> | anneal 0.03 → 0.003 | +0.315 | 0.022 | 0.899 | 0.986 | 0.094 / 0.198 / 0.166 / 0.095 / 0.102 |
>
> 1. **T=0.01 is a sweet spot, not a direction.** It cuts §8.1's worst community
>    from 0.322 to 0.051 log-X error and flattens M5 to ~5% at *every* size;
>    T=0.003 is worse than either neighbour on every axis, so "sharper is better"
>    is refuted. The price is worst gradient cosine 0.958 → 0.928 — about 2x the
>    seed sd, on one seed, unrepeated. **This supersedes T=0.03 as the operating
>    point**; the earlier reading that `T` is an accuracy-only knob stands, but its
>    accuracy is the low-`mu` half, which nothing was measuring.
> 2. **`--w-rel` trades the bottom against the plateau.** It removes the bias with
>    no cosine cost (0.928 → 0.924 → 0.944 across 0 / 0.3 / 1), but composition gets
>    monotonically worse on the large communities, which is where the plateau is.
>    Raise it only when slow members dominate the question being asked.
> 3. **Two obvious fixes that do not work.** *More low-`mu` labels*: the
>    information is already in the labels — the cutting-plane model over the
>    existing tangents is unbiased — so extra rows change only the low-`mu` row
>    *share*, i.e. a reweighting, which `--w-rel` does directly instead of ~1 h per
>    organism of solves. *A harder softmax reached by annealing* (`--gm-temp-final`,
>    3 geometric stages): decisive on 3 organisms (0.03 → 0.003 took the bias
>    +1.845 → **−0.009** with the plateau intact) and **beaten by fixed T=0.01 on
>    all 21**, on every axis. Kept, off by default, with the refutation on file.
>    That is the third one-organism/three-organism frontier not to survive the
>    roster — the `icnn-u` `w_grad` sweep and the seeded-head cosine were the
>    others. **Do not promote a sub-roster frontier again.**
> 4. **The temperature cannot be made per-row.** `T*logsumexp(a/T)` is concave in
>    `a` only for constant `T`, and exact concavity in `u` is what the head is for;
>    per-epoch is free, per-prediction is not. `--w-rel` *is* the per-row
>    growth-rate weighting — `1/(mu + 0.1*mean mu)^2` — and its scalar plus that
>    0.1 floor are its shape knobs.
>
> Not done: multi-seed confirmation of the 0.958 → 0.928 cosine cost, Head B
> retrained at T=0.01 (`behaviour_b1` is reused above, which is fair since only
> Head A changed), and `--w-rel` / `--gm-temp` on the HPC sweep.

> **Implementation status (2026-08-30) — the low-`mu` bias is an output
> calibration, and calibrating for the *plateau* is what buys M5.**
> `src/cfs/surrogate/calibrate.py`. The bias is a function of the **predicted value
> alone**: an isotonic map fit on the train rows and applied to held-out media
> drives every band's median bias to ≤ 0.005 and *raises* R² (0.9898 → 0.9901). That
> is a second, independent refutation of "more low-`mu` media would help" — nothing
> is missing from `mu_hat`.
>
> `g(m) = a*m − d0*exp(−m/beta)` is increasing (`a, d0 ≥ 0`) and concave
> (`g'' < 0`), so `g(head(u))` stays exactly concave and non-decreasing in `u` —
> §8.4's PSD Hessian tag and `concavity_violation_rate` both survive — and the
> gradient is scaled by a positive per-row scalar, so **`grad_cosine` is
> bit-identical**. Unlike `--w-rel`, nothing is traded on the gate. It is fit at the
> end of `train.run` on the training rows, stored in the checkpoint JSON beside
> `mu_scale`, and applied where `mu_scale` is (`train.evaluate`,
> `compose.dfba.Surrogate.mu_and_z`), so every existing checkpoint deserialises
> unchanged and reads the identity. Bound `beta ≥ 0.05 ×` the prediction range: a few
> organisms predict slightly negative `mu`, where an unbounded fit explodes
> (worst R² −7).
>
> **The fit weight is the result.** Residuals are divided by
> `max(|mu|, _W_FLOOR * max|mu|)`. Same 10 communities and media as the §8.1 block,
> `value_T01` throughout, per-organism FBA truth:
>
> | median log-X error | n=2 | n=3 | n=5 | n=10 | n=21 | low-`mu` bias | plateau |
> |---|---|---|---|---|---|---|---|
> | uncalibrated | 0.034 | **0.050** | 0.051 | 0.048 | 0.047 | +0.446 | −0.005 |
> | `--w-rel 0.3` | 0.058 | 0.082 | 0.060 | 0.065 | 0.064 | +0.100 | +0.014 |
> | `_W_FLOOR` = 0 (pure relative) | 0.046 | 0.074 | 0.061 | 0.098 | 0.098 | **−0.033** | −0.009 |
> | **`_W_FLOOR` = 0.3 (default)** | **0.024** | 0.072 | **0.044** | **0.014** | **0.016** | −0.250 | **−0.002** |
>
> `median_mu_rel` at size 21 goes 0.009 → 0.005; R² and cosine do not move.
> **Sizes 10 and 21 are 1.4% / 1.6% against M5's 1% gate**, from 4.7%. Size 3 is the
> one regression (0.050 → 0.072).
>
> The `_W_FLOOR` = 0 row is the trap: it removes the bias on *every* band and
> doubles the composition error. The map is downward-only, the plateau was already
> at −0.005, and `d(log X)/dt = mu` integrates the plateau, not the bottom — the same
> trade `--w-rel` makes, moved after training. **Tune this by the composition, never
> by `value_bias_low_mu`.**
>
> **This retracts the softmin-offset mechanism above (point 1 of the block).**
> Re-evaluating the *same trained head* at `T → 1e−6` (`groupmax.with_temp`) makes
> the bias **worse** — +0.860 against +0.446 below 5% of max `mu`. The smoothing gap
> is a *downward* offset that partially cancels the bias; what is left is plane
> placement, which is what the class predicts: a min of tangents to a concave
> function is an upper bound everywhere, so positive bias is the only bias a
> max-affine head can have unless a plane sits tangent at that row. The smoothing is
> still what *creates* it during training (point 2 stands, and T=0.01 still beats
> 0.03), but it is not what the checkpoint carries.
>
> **Also dead:** running `--gm-temp 0.03` plus calibration to recover worst cosine
> 0.928 → 0.947. `community_ra3_cal` loses to uncalibrated `value_T01` on every
> axis (log-X 0.081 at size 21). T=0.01 stays.
>
> **Not the lever:** re-anchoring on relative error. `reanchor` ranks rows by
> gradient cosine and the low-`mu` rows score **1.000**, so they are never picked.
> Re-ranked by relative over-prediction, one post-hoc pass moves +0.446 → 0.313 at
> 30% of the planes and costs worst cosine 0.928 → 0.901. Untested *inside*
> training, where a re-anchored plane still has epochs to settle.

---

## 8. Phase 5 — composition

Surrogates frozen. The master problem is the composition operator.

### 8.1 dFBA (build first)

```python
def dfba_rhs(c, X):
    mu = batched_value_head(models_A, log(c))          # (20,)
    z  = batched_behaviour(models_B, log(c), ones(20)) # (20, M) masked
    return (X[:, None] * z).sum(0) + inflow(c), X * mu
```

For the equilibrium, Newton-solve `rhs = 0` rather than integrating. Trajectory
gradients are badly conditioned; a one-shot root-find is not.

> **Implementation status (2026-08-28) — M5 built and measured; the 1% gate is
> not met.** `src/cfs/compose/dfba.py`, CLI `cfs community`. Both frozen heads
> into the right-hand side above (`inflow = 0`, batch culture), integrated with
> explicit Euler and the pool clipped at zero. The ground truth is per-organism
> FBA — `cfs.groundtruth.solve.solve`, the same call that made the labels — at
> the community's shared medium, on the **identical** integrator, step size,
> inoculum and MM bounds, so the comparison isolates the surrogates. A joint
> community LP is deliberately *not* used here: SteadyCom's equal-growth
> constraint and MICOM's tradeoff are different models, and mixing that in would
> make a Head B error and a modelling choice indistinguishable. §8.2/§8.3 are
> where those belong.
>
> 10 communities, `20hm_bands` media over the union of the members' active
> subspaces, 40 steps, equal-split abundances:
>
> | size | dc/dt cosine | mu rel err | log-X final err | overgrowth (V5) | cross-feeding links |
> |---|---|---|---|---|---|
> | 2 (x5, median) | 0.983 | 0.016 | 0.055 | <=0.066 | 6/8 |
> | 3 (x2) | 0.847 | 0.083 | 0.122 | <=0.210 | 8/9 |
> | 5 | 0.915 | 0.133 | 0.322 | 0.182 | 10/11 |
> | 10 | 0.996 | 0.013 | 0.041 | -0.002 | 27/27 |
> | **21** | **0.997** | **0.014** | **0.044** | **-0.001** | **38/38** |
>
> 1. **Community size is not the error axis.** The 21-member run is the second
>    most accurate in the set and recovers every one of its 38 cross-feeding
>    links — a metabolite one member secretes and another consumes, which is the
>    behaviour no member's labels contain, since each organism was solved alone.
>    89/93 links recovered overall. Per-organism errors are largely independent,
>    so they partially cancel in `sum_i X_i z_i` rather than compounding: this is
>    the central bet of D1(a) plus this section, and it holds.
> 2. **A slow member is the error axis.** Every bad cell contains an organism
>    with `mu0 < 1.3 h^-1` on its drawn medium. Head A's held-out R² is taken
>    over each organism's *own* mu spread, so a near-starving organism is a small
>    absolute error and a large relative one, and `d(log X)/dt = mu` integrates
>    the relative one. The next Head A signal to chase is accuracy at low `mu`,
>    not the roster-worst gradient cosine of §7.
> 3. **M5's 1% gate is missed by ~4x, and the shortfall is Head A's.** A 1.4%
>    `mu` error over ~2 doublings *is* a 4% log-X error. Closing it needs a
>    better `mu`, not a better ODE solver.
> 4. **P4 does not bite.** Re-solving the true LP at the state the surrogate
>    walked *itself* to (V5) gives overgrowth <= 0.21 of initial `mu`, and ~0 on
>    the large communities. The composition does not run away to a fictitious
>    fast-growing state.
>
> **Two measurement traps, both now handled in the code.** A batch culture has
> two independent clocks — members doubling, and the pool emptying — and a
> horizon set by the growth clock alone killed the true community at step 2 of
> 40, leaving two live points to score. `run` solves for the *inoculum* instead
> (`dc/dt` is linear in `X`, so one probe solve fixes it) so the pool empties at
> the end of the horizon. And the metrics are scored only while the true
> community is alive, normalised by fixed initial scales: a dead culture has
> `mu = 0` everywhere, where a per-step relative error divides by zero — the
> first version reported `nan` and a 237% `mu` error for a run whose live phase
> agreed to 2%. Concentration error is per metabolite relative to its own `c0`;
> a plain L2 over the pool reads 0.7% on a trajectory where the limiting ion is
> gone in the truth and untouched in the surrogate.
>
> **Update 2026-08-29 — points 2 and 3 are now acted on, and the numbers above
> are superseded.** Re-running the identical 10 communities against a Head A
> trained at `--gm-temp 0.01` (§7.3's low-`mu` block) gives median log-X error
> 0.034 / 0.050 / 0.051 / 0.048 / 0.047 at sizes 2 / 3 / 5 / 10 / 21: the 0.322
> cell collapses and **the error is ~5% at every community size**. Point 1 (size is
> not the error axis) is unchanged and now holds without exception; point 2 (a slow
> member is the axis) is confirmed by its fix — the slow-member cells were Head A's
> relative error at low `mu`, not composition. Point 3 stands: the residual ~5% is
> still `mu`, and the 1% gate still needs a better value head.
>
> **Update 2026-08-30 — the output calibration takes the large communities to
> ~1.5%.** The same 10 communities and media, `value_T01` plus the plateau-weighted
> calibration of §7.3: median log-X error **0.024 / 0.072 / 0.044 / 0.014 / 0.016**
> at sizes 2 / 3 / 5 / 10 / 21, against 0.034 / 0.050 / 0.051 / 0.048 / 0.047
> uncalibrated. `median_mu_rel` at size 21 is 0.005. Point 3 above is now only
> partly true: sizes 10 and 21 are within 1.6x of the 1% gate and the residual there
> is no longer dominated by the `mu` *bias*. Size 3 regressed (0.050 → 0.072) and is
> the open cell. Head A's held-out gradient cosine and R² are unchanged by the
> calibration, by construction.
>
> **Update 2026-08-30 (b) — the residual was Head B's, and it was label coverage.**
> Every number above is `dc_rel`-limited, not `mu`-limited: across the 10
> communities the final log-X error tracks Head B's pool-derivative error and not
> `mu_rel`, which is <= 3% everywhere (`dc_rel` 0.19–0.79 -> log-X 0.003–0.024;
> `dc_rel` 1.1–2.2 -> log-X 0.044–0.589). Point 3 above, "the shortfall is
> Head A's", was true when written and stopped being true once `--gm-temp 0.01`
> and the calibration landed.
>
> Three changes, in the order they were found: Head B predicts specific flux
> (§6.3), `mu_and_z` clamps at §3.3's uptake bound (§6.3), and the §4.3 design now
> covers the community background regime, spent as an 800-media round-1 top-up on
> all 21 organisms. Both heads had to be retrained for the round: `x_scale` is the
> kink scale over the *training* rows, so a round changes it and
> `Surrogate.__init__`'s P14 check fires if only one head is rebuilt — which is
> exactly what it is for. Held out on the identical 800 media, Head A's worst
> gradient cosine went 0.928 -> **0.956** and Head B's worst R² 0.856 -> **0.907**.
>
> **Update 2026-08-30 (c) — and every M5 number, above and elsewhere, is n=1 on
> two axes.** Both were then measured over the same 10 communities: 3 Head A seeds
> at a fixed medium, and 3 medium draws at fixed heads. Median max/min per cell is
> **1.8x** for the seed and **6.1x** for the medium, whose worst cell spans
> **448x** (0.002 -> 0.739). The medium dominates.
>
> A trap that caused a wrong attribution here before it was caught: `run` draws
> each community's medium from `seed + n` with `n` its *index in the list*, so
> cutting `--communities` down to a subset silently re-draws every medium. **Two
> `cfs community` runs are comparable only if the community list is identical and
> in the same order** — and a single draw is not worth quoting whatever the order.
>
> Pooling both axes, n=5 replicates per community, median log-X error:
>
> | size | median | p25 | p75 | max | n |
> |---|---|---|---|---|---|
> | 2 | **0.009** | 0.005 | 0.018 | 0.739 | 25 |
> | 3 | **0.018** | 0.007 | 0.064 | 0.110 | 10 |
> | 5 | 0.076 | 0.028 | 0.085 | 0.107 | 5 |
> | 10 | 0.027 | 0.022 | 0.029 | 0.125 | 5 |
> | 21 | **0.027** | 0.027 | 0.029 | 0.029 | 5 |
>
> Sizes 2 and 3 are within 2x of the 1% gate on the median; nothing passes it.
> **Point 1 gains a second, stronger form:** the 21-member community is the most
> *reproducible* cell in the set — 1.7x across seeds and 1.1x across media, against
> 12x and 448x for 2-member ones. The independence that lets per-organism errors
> cancel in `sum_i X_i z_i` also averages away the medium draw. Cross-feeding recall
> is 1.00 at sizes 3, 5, 10 and 21, and `overgrowth` <= 0.038, so V5/P4 still do not
> bite.
>
> **The gate has to be stated over replicates.** A single `cfs community`
> invocation carries ~6x sampling error on a small community, which is larger than
> every model change measured on 2026-08-30.
>
> **Not measured:** the `--steps` refinement check, abundances other than equal
> split, the Newton form of the equilibrium (M6) — this is the integrated
> trajectory only — a Head B seed axis (only Head A's was varied), and re-scoring
> the three 2026-08-30 changes over replicates now that the error bar is known.

### 8.2 SteadyCom

Bisect on common `mu`; `alpha_i = mu / mu_max_i(c)`; check a non-negative
abundance vector balances the shared pool. ~20 iterations to 1e-6.
Verify feasibility is monotone in `mu` under your surrogate before trusting it.

### 8.3 MICOM

Two-stage with a tradeoff parameter. Strictly convex, best-behaved gradients.

### 8.4 Price form

```python
sol = optx.root_find(excess_demand,
                     optx.Newton(rtol=1e-10, atol=1e-12),
                     p0, args=(c, X, supply),
                     adjoint=optx.ImplicitAdjoint(),
                     tags=lx.positive_semidefinite_tag)
```

Unknowns = `|shared|` from §2.1, not `M`. Jacobian is a sum of PSD Hessians.

---

### 8.5 The size-21 regression — stock-take and the plan (2026-08-31)

**The problem.** After the `probe_lo = -12` relabel, §8.1 regressed at large
community size, and **three design fixes have not moved it**. Median log-X over
5 replicates x 10 communities, identical community list and media throughout:

| | n=2 | n=3 | n=5 | n=10 | n=21 | overall |
| --- | --- | --- | --- | --- | --- | --- |
| `r1` pre-relabel | 0.009 | 0.018 | 0.076 | 0.027 | **0.027** | 0.017 |
| `p2` relabel | 0.014 | 0.067 | 0.029 | 0.093 | 0.466 | 0.031 |
| `p3` stratum budget | 0.024 | 0.049 | 0.022 | 0.082 | 0.783 | 0.030 |
| `p4` anchor-relative background | 0.032 | 0.048 | 0.026 | 0.095 | 0.706 | 0.035 |

It is **reproducible and localised**, not sampling noise. At n=21 medium draw 0
fails on all three Head A seeds (0.47 / 0.79 / 0.71) where all three `r1` seeds
are fine (0.017-0.029); draws 100/200 are ~3x worse than `r1` but not
catastrophic. On the true path at that medium Head A's `mu_rel_median` is
**0.005 (`r1`) against 0.104 / 0.144 / 0.144**.

**It is a tail, one member.** Median |rel| over the 21 members is 0.055 (`r1`) vs
0.068 (`p4`); what changed is **AAXE02, `mu_hat` 43.6 against a true 17.6
(+147%)**, and GCA_000151225.1 at +74%. `r1`'s worst member is -28%.

**And every label-level metric improved across the same relabel**: worst grad
cosine 0.956 → 0.963, value R² 0.974 → 0.989, Head B R² 0.907 → 0.937, rows per
(organism, metabolite) where that metabolite limits p10 **1 → 100**, M11's
`n_missed_essential` 6 → 0.

> So this is not a fit deficit and not a coverage deficit. It is **distribution
> shift the held-out protocol cannot see**, because held-out media are drawn from
> the same design that changed. That is P24, restated.

**Five metrics are refuted as predictors of §8.1** (P25). Each moved the right way
with no downstream effect: co-limitation count (`p3`: >=5 8.6% → 9.9%),
near-onset count (`p4`: median 0 → 4, p95 4 → 10), §6.3 nearest-training-medium
distance in `x` (unchanged or better at the failing members), per-metabolite
limiting-row coverage (p10 1 → 100), and held-out cosine / R². Also refuted:
the plane budget (K=1000 → 2000 changes nothing), and the limiter's own band
(`EX_g3pg_e` 7.43e-3 → 7.34e-3, it never moved).

**One structural observation.** At the failing medium both `r1` and `p4` put
essentially all of AAXE02's gradient on `EX_12ppd__R_e`, which limits in 1 and 0
training rows respectively — the argsort-tie-among-zeros artefact, now visible at
the gradient. The true limiter `EX_g3pg_e` (276 → 98 limiting rows) gets ~0. The
composition integrates the *value*, so this does not explain +147% by itself, but
the head has no correct local structure there.

**Why max-affine over-predicts here, structurally.** `mu_hat(u) = min_j [mu_j +
pi_j.(u - u_j)]` is a min of tangents to a concave function, so it is an **upper
bound everywhere**, and it is tight only near a tangent point. At a medium with
no nearby anchor the min of the remaining planes sits high — +147% is the
textbook form of that, not an anomaly. Any fix has to either put a plane near the
community regime or bound the head from the other side.

#### The plan, in order

The methodological point first: three design changes were chosen from proxies and
each cost a ~5 h relabel + 3-seed retrain + 5-replicate scoring. That loop is not
converging, and the measurement should be fixed before the next design change.

1. **E1 — cutting-plane check at the failing medium.** Score the parameter-free
   `min_j` model over `p4`'s *own* training tangents at that medium. Minutes, no
   training. If it also over-predicts AAXE02, the labels lack the information and
   no architecture change helps; if it does not, the deficit is the trained head.
   Everything below is conditional on this.
2. **A1 — a community-regime held-out label set.** ~2000 media per organism drawn
   over the **union** of the members' active subspaces, solved once and held out
   permanently. Converts a 5 h blind loop into a minutes-long one, and is the
   direct instrument for P24. This should have preceded `p3`.
3. **A2 — report per-member worst |rel| in `cfs community`.** The failure is a
   tail and the summary reports medians; it hid this for three runs.
4. **C4 — take the min over the Head A seeds already trained.** Valid for an
   upper-bound family, preserves concavity and monotonicity, directly attacks the
   over-prediction tail, and costs nothing: three seeds are already trained per
   run. The one-line version of E2.

Then, conditional on E1:

**If the labels are sufficient (cutting-plane is right at that medium):**

- **C1 — put planes in the community regime.** Re-rank `init_from_tangents` /
  `reanchor` to include community-style media, rather than `rank_by_active_set`
  over the training rows alone. This is why K above 1000 is inert: more planes in
  the wrong place.
- **C1b — bound from the other side.** Clip the head with the Liebig bound
  `min_m (a_m + b_m u_m)` fitted per metabolite from the labels: exactly
  computable, a valid upper bound, and much tighter than a generic tangent in the
  multi-limited regime.
- **E2 — a predictive quantile instead of a point estimate.** `d(log X)/dt = mu`,
  so systematic over-prediction compounds along the trajectory; a deep ensemble
  (already available) or a last-layer Laplace with a low quantile is the
  statistically correct object for composition.
- **C2 — a shared cross-organism residual head** trained only on community media.
  Keep it concave and non-decreasing and §8.4's PSD tag survives.
- **D1/D2** — `--w-rel` targeted at the mid-`mu` band with `calibrate` refitted
  (never tested in combination); `reanchor` ranked by relative over-prediction
  *inside* training and on community media (the post-hoc version hurt cosine).

**The labels ARE insufficient — E1 ran and the cutting-plane over-predicts by
+153%. This is the live branch.** The stages below are ordered cheapest-first and
each one is *gated on the previous one's A1 score*, because the whole reason this
loop stalled is that design changes were chosen from proxies and scored 5 h later
on a ruler that moved with them.

#### The progression for improving the training rows

**Stage 0 — a fixed ruler (A1). Done.** `cfs community-holdout`. Nothing below
means anything without it: three relabels improved every held-out metric while
§8.1 regressed 17x. It ranks `r1` above `p2`/`p4`, which is the §8.1 order and the
inverse of the held-out order. **Every stage below is scored here before anything
downstream is retrained.** Cost: 45 min once, seconds per checkpoint after.

**Stage 1 — a non-adaptive stratum (B2). RUN, AND REFUTED (2026-09-01).** It hit
its label target exactly (rows with `mu/mu_max` in [0.3, 0.8]: median 183 -> 555,
minimum **13 -> 470**) and made both A1 and §8.1 worse, calibrated or not. Kept in
the code at `--mid-mu` default 0.15; **pass `--mid-mu 0` for a new label root**
until something re-motivates it. The description below is what it does.

**Stage 1 (as designed).** `--mid-mu` (default
0.15): a community-sized share of `A_i` between each metabolite's own onset and
its 50%-recovery point, bounded by a second `demand_probe` bisection (~2 s per
organism, no labels needed). This is the cheapest thing that puts tangents in the
regime E1 found empty, and it is *not* adaptive — no model in the loop, so it
cannot chase its own errors. **Gate: A1 median abs and worst p90 at or below
`value_r1`'s 0.0022 / 0.163.** If it clears, stop: the remaining stages buy
nothing a fixed design already has.

**Stage 2 — retune the stratum, not the mechanism. DONE, AND IT WAS NOT A
PARAMETER MISS.** The first band drew each picked metabolite *between* its onset
and its 50%-recovery point and moved the target band the wrong way (AAXE02's
[0.3, 0.8] rows 29 -> 13); recalibrating it above the 50% point, by solving a few
hundred media per organism, put 0.64-0.68 of media in the band on three organisms.
The histogram was then right and §8.1 still did not improve — so the mechanism,
not the parameters, is what fails. This is the stage working: a 20-minute solve
caught the parameter error before the 8 h retrain, and the retrain then closed the
mechanism question.

**Stage 2 (as designed).** `mid_target_frac` (0.5),
`mid_mu_share` (0.2-1.0) and `frac_mid_mu` (0.15) are three scalars, and the
resulting `mu/mu_max` histogram is measurable on the labels **without training
anything**. If Stage 1 misses, check that histogram first: if the stratum is not
landing in [0.3, 0.8], this is a parameter miss, not a mechanism one, and a
re-generate is ~1 h with no retrain. Only if the histogram is right and A1 is
still bad does the mechanism need replacing.

**Stage 3 — B1, community media as a first-class stratum.** 30-50% of the budget
drawn over *unions* of active subspaces (what A1's own media are), rather than
the current 800-media round-1 afterthought at `--bg-perturb 0.9`. More expensive
than Stage 1 and it changes the design's centre of mass, so it is worth it only
if the mid-`mu` band turns out to be necessary but not sufficient.

**Stage 4 — adaptive search (B3), and only here.** "Add rows where the model is
inaccurate, until the space is explored" is the natural idea and it is also
`cfs topup`, which already exists and already failed twice — once because top-up
media leaked into the validation split (fixed, never re-run) and once, more
fundamentally, because **the acquisition signal was computed on held-out media
from the design being changed**, which is exactly what cannot see this failure.
So the adaptive version is admissible only with:

- **the pool drawn from the community regime**, not the training design — i.e.
  candidates sampled the way A1's media are;
- **an acquisition function that exploits the head's one-sidedness.** Head A is
  max-affine, so `mu_hat` is a *min of tangents to a concave function* and can
  only over-predict. The looseness of that bound is therefore computable at a
  candidate medium with **no ensemble, no training and no solve**: take the
  binding tangent and compare the `mu` of the row it is anchored at with the
  predicted value. Tight at the failing medium under `r1` (anchor 17.9, truth
  17.63); loose under `p4` (anchor 50.6, prediction 44.6, truth 17.63). Score a
  large candidate pool in milliseconds, solve only the top tail, and each solve
  installs a tangent exactly where the bound was loosest;
- **a stopping rule from the same quantity**: stop when the *max* bound-looseness
  over a fresh community-regime pool stops falling. That is the "sufficiently
  explored" criterion, and it is measured on candidates rather than on the model's
  own held-out error.

`ensemble.gradient_disagreement` is the alternative acquisition signal and is
worse here: it needs N trained heads, and it measures *variance* where the failure
is a *bias* that `d(log X)/dt = mu` compounds along the trajectory. Keep it as a
fallback for a failure the bound-looseness score cannot rank.

**Stage 5 — accept a label ceiling and change the head.** Only if a community-
regime pool that has been actively covered still leaves A1's tail high. Then the
deficit is genuinely representational and C1b (a Liebig lower bound) or E2 (a
predictive quantile rather than a point estimate) apply. Nothing before Stage 4
distinguishes that case from an unlabelled region, which is why they come last.

#### Where this stands after 2026-09-01, and why stages 3-5 are not the live question

E1 has now been run on **two different** n=21 failures and gives opposite answers,
which is the whole reason it comes first:

| failure | trained head | cutting-plane over the same labels | verdict |
| --- | --- | --- | --- |
| AAXE02, `p4` **calibrated** | +148% | **+153%** (`r1`: -0.000) | labels insufficient |
| GCA_000007325.1, `p4` **uncalibrated** | 0.055 vs a true 0.363 | **0.363, exact** | trained-head deficit |

So the B branch was correct for the over-prediction the output calibration was
masking, and is **wrong for the failure that remains**. Two head-side changes
followed from that, both in the code and both measured:

- **`--w-under`** — a one-sided relative penalty on under-prediction. `mu_max` is
  concave and the head is a min of affine pieces, so `mu_hat < mu` at a labelled
  row is a **provable violation**, not an accuracy trade. On `p4`'s bottom-5%-`mu`
  training rows the head under-predicts 53-68% (against `r1`'s 0.2-0.5%); the hinge
  takes that to 19-24%, cuts A1's worst p90 0.311 -> 0.051, and halves the n=21
  cell at draw 0 (0.776 -> 0.346). `w=1` beats `w=10` on every axis.
- **`--gm-select level1`** — SDDP's cut selection (Part 3a of `docs/reading-map.md`).
  Gives back the held-out cosine the hinge cost (0.899 -> 0.919) and takes the
  composition's worst cell 1.698 -> 0.418, but does not move `n=21`. Its durable
  contribution is diagnostic: **~90% of label tangents never bind anywhere**, and
  `K = 1000` exceeds the useful cut count on **21/21 organisms**, which is why the
  plane budget was inert.

**The state to hand on:** sizes 2/3/5 are under M5's 1% gate, n=10 is 5.5-11%, and
`n=21` is flat at 0.27-0.35 across five independent interventions. On the
*calibrated* head it is one member (DACTBY01, +0.68 to +1.22) at one community — an
over-prediction at a point with no nearby tangent, max-affine's structural one-sided
error. On the uncalibrated head, which is the better arm, the worst member is an
**under**-prediction and §8.6 applies instead. Cheap and untried:
`--gm-trial-media` against a larger community-regime pool than A1's 2000.
A1 is a *tail* instrument — its `worst_p90` reproduced the composition's `max`
ordering exactly — and no per-medium statistic can see a single member at a single
community, so **A2's `mu_rel_worst_member` is the reporting unit for this cell.**

**C4 has since run and is refuted, and it changed the reading of the cell.** The
min over three Head A seeds is a null (n=21 0.272 -> 0.259 over 3 medium draws) and
had to be: on the **uncalibrated** `p4` head — the one that is otherwise best —
`mu_rel_worst_member` is GCA_000007325.1 at **-0.857**, an *under*-prediction, and a
min can only push predictions down. The DACTBY01 over-prediction above is the
*calibrated* head's failure. **Read `mu_rel_worst_member`'s sign before picking a
fix**, because the two failures need opposite tools and the toolkit is almost
entirely built for the over-prediction one. The under-prediction branch is §8.6.

#### §8.6 — the under-prediction branch, in priority order

An under-prediction is not an accuracy shortfall; it is a **certificate that the
head has left the valid-outer-approximation family**. A min of supporting
hyperplanes of a concave function is an upper bound everywhere, so it cannot read
low. Only two mechanisms produce one, and both are separable before any retrain:

| mechanism | test | cost | verdict on `p4` |
| --- | --- | --- | --- |
| the softmin's downward gap, `<= T*ln(K_active)` | re-evaluate at `T -> 1e-6` (`groupmax.with_temp`) | seconds | **refuted** — moves `mu_hat` by 0.008 at the failing medium |
| planes no longer valid tangents | `mu_hat >= mu` on the head's own **training** rows | seconds | **confirmed** — 48.1% of rows under-predicted, 53-68% in the bottom-5% `mu` band |

Run both before spending anything. Literature for each option: `docs/reading-map.md`
§3c. The options are ordered cheapest-first and each is gated on A1 plus
`mu_rel_worst_member`, per Stage 0.

**Option 1 — validity projection (`--gm-repair`). Implemented.** Hold the slopes;
set each plane's intercept to the tightest value that keeps it above every training
label, then apply the one uniform shift that covers the smoothing gap. Closed form,
no refit, `O(K x N)` per organism, and it applies **post hoc to an existing
checkpoint** (`20hm_bands/repair_posthoc.py`). This is SDDP's cut-validity
invariant, which that literature maintains by never modifying a cut; we do modify
them, so we restore it afterwards. Measured on `value_p4_nc`: training rows
under-predicted **48.1% -> 0.0%**, at the cost of a +4% median over-prediction —
which is what a genuine outer approximation costs and is the currency `--w-under`
was buying in the same direction, but during training and only approximately.

*Trap that cost a cycle:* per-plane validity is necessary and **not sufficient**.
The head is the *smoothed* min and sits up to `c*T*ln(K)` below the hard one, and
training had been paying for that gap in the intercepts. Repairing the planes
without restoring it left **96%** of rows under-predicted — worse than doing
nothing. The uniform shift is exact because lowering every `b_j` by the same delta
moves all pre-activations together, so it lifts the smoothed head by exactly
`c*delta`. `tests/test_cfs_value_head.py::test_repair_restores_validity_without_loosening_the_fit`
runs at a production temperature specifically so this cannot regress; at
`temp=1e-4` the gap hides under any tolerance and the bug does not show.

**Measured on §8.1, and it is necessary but not sufficient — which is the useful
part.** 3 medium draws x the same 10 communities, `value_p4_nc` vs the repaired
checkpoint:

| median log-X | n=2 | n=3 | n=5 | n=10 | n=21 | overall | max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `p4` no cal | 0.005 | 0.002 | 0.004 | 0.093 | **0.272** | 0.007 | 1.698 |
| **+ `--gm-repair`** | 0.009 | 0.008 | 0.007 | **0.067** | 0.401 | 0.014 | 2.842 |

The invariant behaves exactly as designed and **the under-prediction is gone**: at
the failing n=21 medium every member flips sign, GCA_000007325.1 from **-0.857 to
+0.998**, and `mu_rel_worst_member` becomes DACTBY01 at **+2.436**. n=10 improves.
So a *valid* outer approximation at that medium is **loose** — which converts the
§8.6 failure back into the §8.5 one, and the two are now provably two ends of one
thing rather than two problems.

**This localises the remaining deficit to the slopes, and that is the handoff to
Option 2.** Validity projection moves intercepts only. E1 on this same failure
found the cutting-plane model over the *same* labels is exact there (0.363), so a
valid model can be tight at that medium — the trained head cannot be, with its
slopes, at any intercept. Training moved the slopes off the label tangents and no
projection recovers that. Do not read the n=21 regression as the repair failing;
read it as the repair removing the only thing that was hiding a slope error.
`--gm-repair` is off by default and worth carrying because it makes the head's
one-sidedness true rather than approximate, which is what §13's programs assume.

**Option 2 — a chosen quantile instead of a tuned weight.** `--w-under` is the
`tau -> 1` hinge of the asymmetric-loss family (Koenker & Bassett 1978; Newey &
Powell 1987) with `tau` taken implicitly. Make it explicit: fit a `tau`-quantile
envelope, so the one-sidedness comes with stated coverage rather than a weight
tuned against a 5 h composition run. Cheap — it is a loss change in `_loss`, one
retrain — and it subsumes Option 1's guarantee *during* training rather than
projecting onto it afterwards. Do it only if Option 1's post-hoc repair helps but
the head then drifts back on a fresh label root. Note the hard-constrained version
(fit subject to `f(u_r) >= mu_r`) is, with slopes fixed, the same LP as Option 1 —
so Option 1 is optimal for its slopes, and Option 2's value is that it moves the
*slopes* too.

**Option 3 — one-sided conformal on the community holdout.** Inflate by the
`(1-alpha)` quantile of the **signed** residual for a finite-sample guarantee in
whichever direction is failing. The caveat is the one that has bitten this project
twice: conformal validity is w.r.t. the calibration distribution, so it must be
calibrated on `community_holdout` media, **never** on held-out media from the
design being changed (P24). This is `calibrate.py`'s family with a guarantee
attached — and note `calibrate` is already known to be design-dependent, so a
conformal version inherits that and states it rather than hiding it.

**Option 4 — nothing symmetric exists; do not look for it.** min-over-ensemble is
pessimism (CQL), max is optimism (bootstrapped DQN/UCB), and **a concave family
admits only the min** — a max of concave functions is not concave. C4 is therefore
the *whole* ensemble toolkit available here, and it is the wrong sign for this
failure. Do not reach for "max over seeds": it leaves the family, and §8.4's PSD
Hessian tag goes with it.

**The mismatch worth stating if this is written up.** The offline-MBO and offline-RL
literatures are almost entirely about *conservatism*, because an optimistic
surrogate gets exploited by whatever optimises against it. This head is conservative
by construction and the failure that remains is the opposite sign, so most of that
machinery points the wrong way. The question it leaves open — how to keep a
structurally one-sided estimator **tight** rather than how to make an unconstrained
one safe — does not appear to be addressed anywhere.

**Sampling rather than optimising** (E2 above is the main one). Also open:
**E3**, turn the value call into an argmin — sample the head along each active
coordinate at each dFBA step to identify the limiter, which the head is
measurably good at (top-1 share 0.83-0.94), instead of trusting the level.
**E4**, HMC over the design, is the wrong tool for this; that is §13's inverse
problem, not this.

**Not worth spending on.** A different optimiser: the train/held-out gap is 0.005
cosine and 0.013 R², so nothing is being lost to optimisation — this is an
inductive-bias and distribution problem, not a convergence one.

**A coordinate caveat that affects all of the above.** `x = u/(u+s)` takes `s`
from the training rows, so **every relabel silently changes the input
coordinate** and no two label roots' distributions are strictly comparable. A
fixed coordinate (clipped `log u`) would make this class of question answerable;
it is a prerequisite if the distribution-shift hypothesis is to be tested
rigorously rather than by proxy.


### E1: the labels are insufficient — the cutting-plane check, 2026-08-31

§8.5's step 1, run (`20hm_bands/e1_cutting_plane.py`, minutes, no training). The
parameter-free `min_j [mu_j + pi_j.(w - w_j)]` model over each root's **own full
training tangent set**, at `community_p4_s0`'s failing 21-member medium:

| member | true `mu` | `p4` head | `p4` cut-plane | `r1` head | `r1` cut-plane |
| --- | --- | --- | --- | --- | --- |
| **AAXE02** | 17.63 | +148% | **+153%** | -4% | **-0.0%** |
| GCA_000151225.1 | 4.91 | +74% | +91% | +20% | +30% |
| CP002109.1 | 45.49 | +14% | +17% | +13% | +10% |

1. **The trained head is at its label ceiling here, and slightly better than it.**
   Every architecture branch of §8.5 (C1, C1b, C2, E2, D1/D2) is refuted for this
   failure: no head in the max-affine class can do better on `p4`'s tangents than
   the min over all 3985 of them, and that is +153%.
2. **`r1`'s tangent set is exact at the same medium** (-0.000). So the medium is
   not intrinsically hard, and the difference is entirely which media were
   labelled. This is the **B branch: B1/B2/B3**.
3. **The mechanism is the mid-`mu` band, and it is B2 verbatim.** The binding
   plane at that medium is anchored at a row with `mu` **17.9** under `r1`
   (the truth is 17.63) and at one with `mu` **50.6** under `p4` — a plateau row,
   whose tangent sits high everywhere below it. Rows with `mu/mu_max` in 0.3-0.6:
   AAXE02 **261 -> 90**, GCA_000151225.1 **231 -> 109**. The relabel moved mass
   into `<0.2` (0.17 -> 0.65 of rows) and took it from *both* the plateau and the
   middle; AAXE02 sits at 45% of its plateau at the failing medium.
4. Nearest-training-row distance in `w` is **worse for `r1`** (222 vs 154) at the
   same medium, so this is not proximity — it is having a tangent at the right
   *growth regime*. A sixth refuted proxy for §8.1 (P25).

**Next, in order:** B2 (a `mu/mu_max in [0.3, 0.8]` stratum) is now the specific,
measured design change; A1 (a community-regime held-out label set) is what makes
it scorable in minutes instead of 5 h and should be built with it. A2 is **done** —
`cfs community` now reports `mu_rel_per_member` and `mu_rel_worst_member`.


### A1 works: the community-regime ruler ranks the label roots §8.1's way

`cfs community-holdout make|score` (`src/cfs/validate/community_holdout.py`).
2000 media over the 10 communities' member-union active subspaces, seed 7000,
10 400 `mu_max` solves, ~45 min once. Scoring a checkpoint against it is seconds.

| scored on A1 | median abs rel | worst organism | worst p90 |
| --- | --- | --- | --- |
| `value_r1` | **0.0022** | 0.011 | **0.163** |
| `value_p4` | 0.0052 | 0.039 | 0.443 |
| `value_p2` | 0.0052 | 0.075 | 0.405 |

**It ranks `r1` above the relabels — which every held-out metric got backwards**
(worst cosine 0.956 -> 0.963, R² 0.974 -> 0.989, Head B R² 0.907 -> 0.937 all
favoured the relabel). That is P24 made measurable, and it turns a 5 h relabel +
3-seed retrain + 5-replicate scoring loop into a seconds-long one. Every future
sampling-design change is scored here **before** anything is retrained.

Two properties to keep in mind when reading it. The medians are small (0.2-0.5%)
because most community media are easy; the signal is in the **p90 and the max**
(1.26 for AAXE02 under `p4`), which is the same one-member tail §8.1 integrates.
And it scores `mu_max` only — one FBA per (organism, medium), no elastic-net QP,
no alpha grid, no duals — because the failure it exists to catch is Head A's
*level* at a multi-limited medium. Add duals to it only if a gradient question
turns up that it cannot answer.


### B2 is refuted, and the output calibration was the size-2-to-10 error — 2026-09-01

Two separable results, both on 5 replicates x 10 communities, identical community
list and media throughout. Median final log-X:

| run | n=2 | n=3 | n=5 | n=10 | n=21 | overall |
| --- | --- | --- | --- | --- | --- | --- |
| `r1` | 0.009 | 0.018 | 0.076 | 0.027 | **0.027** | 0.017 |
| `p4` | 0.032 | 0.048 | 0.026 | 0.095 | 0.706 | 0.035 |
| `p5` (B2, mid-`mu` stratum) | 0.028 | 0.026 | 0.028 | 0.269 | 0.569 | 0.038 |
| **`p4`, calibration stripped** | **0.005** | **0.002** | **0.004** | **0.007** | 0.729 | **0.006** |
| `p5`, calibration stripped | 0.007 | 0.005 | 0.023 | 0.311 | 0.808 | 0.016 |

**1. B2 did what it was designed to do and did not help.** Rows per organism with
`mu/mu_max` in [0.3, 0.8] went median 183 -> 555 and **minimum 13 -> 470** (against
`r1`'s 505/61), with `<0.2` held at 0.596 — i.e. `p4`'s low-`mu` coverage plus
`r1`'s middle, which is exactly the design goal. A1 got *worse* (median abs 0.0075
vs `p4`'s 0.0052, worst organism 0.145 vs 0.039) and so did §8.1, calibrated or
not. A seventh refuted proxy (P25) — but refuted in **seconds by A1**, not in 5 h,
which is Stage 0 of the progression doing its job. Stage 2's histogram check
passed, so this is not a parameter miss: the mechanism does not help.

**2. The output calibration's sign flips with the label design, and on the
relabelled roots it is the dominant error at every size below 21.** A1:

| | A1 median abs | A1 worst organism |
| --- | --- | --- |
| `value_r1` cal / no cal | **0.0022** / 0.0054 | **0.011** / 0.015 |
| `value_p4` cal / no cal | 0.0052 / **0.0007** | 0.039 / **0.0027** |
| `value_p5` cal / no cal | 0.0075 / 0.0010 | 0.145 / 0.187 |

`calibrate` fits a downward concave map with residuals weighted at
`_W_FLOOR = 0.3` of the organism's max `mu` — deliberately "set on the plateau,
not on the band that motivated the work", and tuned when 72% of held-out media sat
above 0.8 of max `mu`. After `probe_lo = -12`, **62% sit below 0.2**, the fit is
dominated by the bottom, and it over-corrects the plateau — which is precisely what
`d(log X)/dt = mu` integrates. The existing note "it is not the calibration" was
measured on `r1`, where it genuinely helps (and still does); it does not transfer.

Uncalibrated `p4` is better on every other axis too: `mu_rel_median` on the true
path 0.0098 -> **0.0015**, V5 `overgrowth` max +0.262 -> **+0.097** (so nothing
runs away), and worst-member |rel| median 0.002.

> **M5's 1% gate is met at sizes 2, 3, 5 and 10 — 0.5% / 0.2% / 0.4% / 0.7% —
> for the first time.** n=21 is untouched at 0.729 and is now the only cell that
> fails, on any label root or calibration setting.

**3. So the size-21 regression is neither the calibration nor the labels' mid-`mu`
band.** It sits at 0.57-0.81 across `p2`/`p3`/`p4`/`p5`, calibrated and not, and at
0.027 on `r1`. Every design change since `probe_lo = -12` has left it exactly
where it was. The next hypothesis to test is `probe_lo` itself — it is the one
change `r1` does not have, and it is also what closed M11's essentiality blocker,
so a revert is not free.

**Do not set `_W_FLOOR` (or decide to calibrate at all) without re-measuring on
the label root in use.** It is a property of the design's `mu` distribution, not
of the head. The cheapest correct form is probably to weight relative to the
design's own plateau share rather than a fixed 0.3; that is untested.

**A latent bug this uncovered.** `calibrate.apply` returns **NaN** for any negative
raw prediction under the identity calibration: an uncalibrated checkpoint stores
`beta = 0`, the divisor is floored at 1e-12, `exp(-m/1e-12)` overflows to `inf`
and `d0 * inf` is NaN rather than 0. Head A's raw output does go negative at a
scarce medium, so composing an uncalibrated head took the dFBA trajectory to NaN
at step 0 — every pre-2026-08-30 checkpoint was exposed. Fixed by clipping the
exponent at 700 and writing `beta = 1` in `_identity_cal`;
`tests/test_cfs_value_head.py::test_identity_calibration_is_finite_on_negative_predictions`.


### SDDP's Level 1 cut selection, adopted — 2026-09-01

The parameter-free cutting-plane model is SDDP's outer approximation, sign-flipped:
label tangents are *cuts*, media are *trial points*, `--gm-group K` is the cut
budget. That field settled cut selection fifteen years ago, and the rule transfers
directly. `groupmax.rank_by_territory` implements **Level 1 dominance** (de Matos,
Philpott & Finardi 2015) = the **territory algorithm** (Pfeiffer, Apparigliato &
Auchapt 2012 — the two provably select the same cuts): each cut owns the trial
points where it is the active minimum, and a cut with an **empty territory** is
dropped, because dropping it changes the approximation nowhere on those points.
Scoring every cut at every point in one pass and keeping only the active index per
point makes this the **limited-memory** variant (Guigues 2017): O(points), not
O(cuts x points). We **store and select** rather than prune — the other convention
Guigues distinguishes — which is free here because the tangents live in the label
shards. `--gm-select level1`, `--gm-trial-media <community_holdout.npz>`.

**It explains "K is inert above 1000" mechanically, before any training.** Of ~3985
usable tangents per organism, the number with a **non-empty territory** over the
4000 training media is **174-686, median 419 — about 10%**. The budget K=1000
**exceeds the useful count on 21/21 organisms**, so K 1000 -> 2000 could not have
helped: there were never 2000 binding cuts to find. That reproduces Pfeiffer et
al.'s own ratio (490 -> 220 -> 55 cuts per stage with the forward cost falling at
the same rate) on our labels. Over *community-regime* trial points the count falls
further, to **44-170 (median 88)**.

Same knobs as `value_p4_wu1` throughout, only the selection rule and its point set
differ; all scored uncalibrated on A1:

| checkpoint | A1 med abs | A1 worst org | A1 p90 | worst grad cos |
| --- | --- | --- | --- | --- |
| `p4` (active-set) | 0.0007 | 0.0027 | 0.311 | 0.9211 |
| `+ --w-under 1` | 0.0007 | 0.0083 | 0.051 | 0.8985 |
| **`+ level1`, training rows** | 0.0010 | **0.0022** | **0.050** | **0.9194** |
| `+ level1`, community media | **0.0005** | 0.0038 | 0.055 | 0.9209 |

**Level 1 gives back the held-out gradient cosine the hinge cost, at no cost to the
tail** (0.8985 -> 0.9194 against an unhinged 0.9211, with p90 still 0.05 against
0.311). It dominates the `--w-under`-only arm on every column. The community
point-set arm buys the best median and a slightly worse worst organism — the two
point sets are a real trade, not a strict ordering.

**But the composition does not follow, and n=21 is now flat across everything.**
Median final log-X, 3 medium draws x 10 communities, identical list and media:

| run | n=2 | n=3 | n=5 | n=10 | n=21 | overall | max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `r1` | 0.006 | 0.016 | 0.028 | 0.027 | **0.027** | 0.014 | 0.739 |
| `p4` no cal | 0.005 | 0.002 | 0.004 | 0.093 | 0.272 | 0.007 | 1.698 |
| `+ --w-under 1` | 0.006 | 0.002 | 0.005 | **0.055** | 0.346 | **0.006** | 0.462 |
| `+ level1` rows | 0.006 | 0.004 | 0.004 | 0.110 | 0.342 | **0.006** | 0.447 |
| `+ level1` community pts | 0.005 | 0.005 | 0.009 | 0.106 | 0.340 | 0.009 | **0.418** |

1. **Level 1 buys the label metrics and the worst cell, not the composition.** The
   max over all 30 runs falls 1.698 -> 0.418 across the four `p4` arms and the
   ordering tracks A1's `worst_p90` (0.311 / 0.051 / 0.050 / 0.055) — which is what
   A1's p90 was built to predict, and it does. But `n=10` is *worse* than the hinge
   alone (0.055 -> 0.110) and `n=21` is unmoved.
2. **`n=21` is flat at 0.27-0.35 across every arm**, and is now untouched by five
   independent interventions: four label designs (`p2`/`p3`/`p4`/`p5`), the output
   calibration, `--w-under`, the plane budget, and cut selection. Sizes 2/3/5 are
   at 0.4-0.9% — under M5's gate — and n=10 is 5.5-11%. **The whole remaining M5
   failure is one cell.**
3. **A1 is behaving as designed and should be read as a *tail* instrument.** It
   ranked the arms by `worst_p90` exactly as the composition's `max` came out, and
   its *median* (0.0005-0.0010 across all four) correctly says these heads are
   equivalent in bulk. Neither statistic predicts the n=21 median, because that is
   not a tail over media — it is one member at one community.

**`rank_by_active_set` stays the default.** It buckets rows by dual support pattern
as a *proxy* for which regimes occur; Level 1 is the exact version of what that
proxy approximates, and it wins on held-out cosine, A1 worst organism and the
composition's worst cell — but it loses at `n=10`, so the evidence does not yet
justify flipping a default every number on file was measured against.

---

## 9. Phase 6 — minimal medium (D9)

```
minimise    sum_m cost_m * c_m  +  lambda * ||c||_1
subject to  mu_community(c) >= mu_target
            c >= 0
```

`mu_community` comes from §8. Gradient by implicit differentiation. Solve with
projected gradient or interior point in log-`c` space.

### 9.1 You have exact ground truth here

Minimal medium is a classical FBA problem with an exact MILP formulation. Run
it for the true community model on a handful of cases and compare. This is a
much stronger validation than anything else in the plan and you should use it
as the headline result — "the surrogate finds the same minimal medium X times
faster" is a clean claim.

### 9.2 Report shadow prices, not just the medium

The interpretable output is which metabolites go to zero and what the shadow
price is on those that do not. The medium itself is one point; the shadow
prices tell you the structure.

---

## 10. Pitfalls

| ID | Pitfall | Symptom | Solution |
|---|---|---|---|
| P0 | CarveMe energy-generating cycles | Growth on nothing; L2 fills futile loops | §3.0 pre-flight. Mandatory |
| P1 | Exchange-flux degeneracy | `z` labels jump between nearby media | §5.2 diagnostic then §5.4 |
| P2 | Infeasible media | Loss dominated by garbage | Head A returns 0; separate feasibility model; exclude from Head B |
| P3 | ReLU network has zero Hessian | Newton stalls or NaNs, gradients look fine | Softplus/ELU. Monitor `cond(hessian)` |
| P4 | Master exploits surrogate error | Composed growth exceeds true LP | Re-solve true LP at composed optimum; active-learning reserve (§4.6); trust region |
| P5 | Warm-start from HMC history | Healthy-looking chain, wrong distribution | Deterministic amortised initialiser only |
| P6 | Multiple equilibria | Clustered NUTS divergences | Detect branch boundaries; report basin. Step size will not fix it |
| P7 | Loose Newton tolerance | Inflated rejection, biased posterior | `rtol=1e-10` |
| P8 | Extinction boundary `X=0` | Singular Jacobian, IFT violated | Sample in log-abundance |
| P9 | Non-convergence NaN | Hard wall in HMC | Damped Newton, trust region, `throw=False`, log failure rate |
| P10 | SteadyCom bilinearity | Bisection converges wrong | Verify monotonicity under surrogate |
| P11 | ICNN cannot fit non-convex recourse | Systematic bias in specific media | Concavity stress test early; difference-of-convex if violated |
| P12 | Sparse coverage in 200-D | Accurate nowhere in particular | §4.2 active subspace reduction |
| P13 | Metabolite index drift | Silent invalidation of all checkpoints | Hash `metabolite_index.json` into every artefact |
| P14 | Unit and scale mismatch | Silent | Assert units in loader; store normalisation stats with checkpoint |
| P15 | Km values are invented | Overconfident quantitative claims | §3.3. State the limitation; report topology-dependent results only |
| P16 | One sampling band for every metabolite | Training loss and mean accuracy look fine; the gradient is wrong on whichever metabolites the band missed, and no amount of extra data or rescaling fixes it | §4.7. Anchor each band on that metabolite's own limiting regime; check per-metabolite coverage, not row count |
| P17 | LP solver cycling on a degenerate medium | One shard hangs at 100% CPU with no output; looks like a slow organism | Wall-clock limit on the FBA as well as the QP; a non-optimal row is dropped downstream (P2) |
| P18 | Per-organism design never shows the community regime | Both heads look fine held out and fail *only* at composition, on media whose every coordinate is individually in range | §4.3's `frac_bg_perturb` over a random **share** of the background. Check distance to the nearest training medium, not per-coordinate bounds |
| P19 | A single composition run read as a measurement | ~6x sampling error on a small community swamps the model change you are attributing it to; and cutting `--communities` to a subset silently re-draws every medium (`seed + n`) | Replicates over both Head A seed and medium draw. Never compare two `cfs community` runs with different community lists |

### The four that will cost you time

**P0** invalidates everything upstream and is invisible until you look for it.
**P1** is why D4 is open — take the measurement.
**P4** and **P5** both produce results that look correct: a community that
grows impossibly fast reads as a discovery, and a path-dependent posterior
mixes beautifully while sampling the wrong thing.

---

## 11. Validation protocol

| V | Test | When | Catches |
|---|---|---|---|
| V0 | EGC + MEMOTE on all 20 models | Before anything | P0 |
| V1 | Exchange-FVA degeneracy survey | Before D4 is settled | P1 |
| V2 | Label repeatability and Lipschitz continuity | After Phase 1 | P1, P14 |
| V3 | Held-out accuracy, **gradient error reported separately** | After Phase 4 | P3, P11 |
| V4 | Finite-difference the full objective gradient at 20 points | Before any HMC | P3, adjoint errors |
| V5 | Round-trip: true LP at composed optimum, 100 cases, report the tail | After Phase 5 | P4 — **first 10 cases done 2026-08-28** (`round_trip_end` in `cfs community`): overgrowth <= 0.21 of initial `mu`, ~0 on the large communities. Not yet 100 cases |
| V6 | Exact MILP minimal medium comparison | Phase 6 | Everything, end to end |
| V7 | Simulation-based calibration | Before any posterior | P5, P6 |
| V8 | Published defined media, never trained on | Before writeup | P12, P15 |

---

## 12. Milestones

| M | Deliverable | Gate |
|---|---|---|
| M0 | 20 CarveMe models, QC'd, index frozen | V0 passes |
| M1 | Degeneracy survey, D4 decided | V1 complete, choice documented |
| M2 | Ground truth pipeline | V2 passes |
| M3 | Head A trained, all 20, vmapped | Gradient cosine > 0.99 held-out |
| M4 | Head B trained, alpha sweep validated | V3 passes — **built 2026-08-28**, worst held-out R² 0.856 / median 0.921; **specific-flux target + §3.3 uptake clamp + the §4.3 community-regime round (2026-08-30) take it to 0.907 / 0.952** on the same held-out media (§6.3) |
| M5 | dFBA composition | Trajectory matches COBRApy dFBA to 1% — **built 2026-08-28; gate not met. Now scored over n=5 replicates (3 Head A seeds x 3 medium draws): median log-X 0.9% / 1.8% / 7.6% / 2.7% / 2.7% at sizes 2/3/5/10/21, sizes 2-3 within 2x. A single run carries ~6x sampling error on a small community — larger than any model change measured — so state this gate over replicates only** (§8.1) |
| M6 | Newton equilibrium + implicit gradients | V4 passes |
| M7 | Minimal medium, surrogate vs exact MILP | V5, V6 pass |
| M8 | SteadyCom / MICOM framings | Agreement with reference implementations |

**M9–M14 are the applications layer and live in §13** — forward simulation
(batch and chemostat, built), convex medium design, minimal medium, chemostat
steady state, interaction maximisation, and the one place a posterior is the right
instrument. §13.7 states what accuracy each of them actually needs, which is less
than M3's and M5's gates for all but the last two.

M1 is new and comes before any training. It is a two-day job and it determines
the shape of your entire label set.

**M3 status (2026-07-29):** built, gate not met — worst held-out gradient cosine
0.733 against 0.99, best of four architectures (§7 status block). (1) §4.7
automatic per-metabolite band placement is **done**; (2) architecture has now been
*measured* rather than assumed, and the deficit is located: a random forest reaches
R² 0.979 on the same split, so the ceiling is neither the labels nor the concavity
constraint, and the residual error is per-metabolite **localisation**. What has not
been tried is **scale** — every run so far is one laptop, 4000 media, width 128,
and the models underfit. Hence M3b.

| M3b | HPC sweep: D10 rows × width/depth × arch, deepset at full width | Any cell with R² ≥ 0.9 and worst cosine ≥ 0.9 |

M3b is §7.4. It is the last cheap thing to try before concluding that Head A needs
a per-metabolite architecture rather than a bigger dense one. Anything that reads
as hand-tuning a metabolite is still not the deliverable.

---

---

## 13. Phase 7 — what the surrogates are *for*

Phases 1–6 build and measure the heads. This section is the applications layer:
the optimisation and sampling problems the frozen heads make cheap. It is written
after M0–M5 landed with **M3 and M5 both short of their gates** (worst held-out
gradient cosine 0.956 against 0.99; median community log-X error 0.9–7.6% against
1%), and it is deliberately ordered so that the use cases whose accuracy
requirement the heads *already meet* come first.

### 13.0 The interface everything below is built on

Two functions and one sum, all of them cheap and analytically differentiable:

```python
mu_i(c)          # concave, non-decreasing in u = c/(Km+c); Head A + calibration
z_i(c, alpha)    # exchange fluxes, mmol/gDW/h; Head B x mu_i, clamped at -Vmax u
dc/dt = sum_i X_i z_i(c) + D (c_feed - c)      dX_i/dt = X_i (mu_i(c) - D)
```

Every application is an objective or a likelihood on those. Nothing below needs a
new network, and nothing below should be allowed to require one: if an application
wants a quantity the heads do not emit, that is a Phase 3 change, not a Phase 7
one.

**The structural fact that decides which problems are easy.** `mu_i` is concave
and non-decreasing in `u`, and `u_m = c_m/(Km_m + c_m)` is a concave increasing
map of `c_m`. So **`mu_i` is concave in `c`**, and so is `min_i mu_i` and any
non-negative weighted sum. Anything expressible as *maximise a concave function of
the medium* or *minimise a linear cost subject to a growth floor* is therefore a
**convex program with a unique optimum** — no multistart, no HMC, no local optima.
That covers §13.2 and §13.3, which is why they are first.

`z` carries no such structure, and neither does anything integrated through a
trajectory. Those are §13.5 onward, and they are where sampling earns its place.

### 13.1 Forward simulation — batch and continuous — **built 2026-08-30**

`cfs simulate` (`compose.dfba.simulate`, `with_chemostat`). §8.1's map with the
LP removed: given members, a medium and an inoculum, integrate. `--dilution D`
makes it a chemostat — `inflow(c) = D (c_feed - c)` and net growth `mu - D` —
which needs no change to `integrate`, because the biomass update is already the
exponential map. `D = 0` is exactly the batch culture M5 measures.

Reports the trajectory, who washes out, which metabolite empties first, and the
cross-feeding links (a metabolite one member secretes and another consumes) at the
midpoint. This is the cheapest useful thing the heads do: seconds against one
LP per organism per step.

The inoculum is solved for, not given: `dc/dt` is linear in `X`, so one probe at
unit biomass fixes the biomass whose pool empties at the end of the horizon. This
is §8.1's two-clocks trap and it bites here too — an arbitrary 1e-3 gDW/L killed
the first batch run at 3% of its horizon. `--biomass` overrides. In a chemostat
the horizon is at least five residence times.

Measured on `value_r1` + `behaviour_r1`, 3 members, 100 steps, ~5 s: batch runs
the full horizon with 88 cross-feeding links; at `D = 0.5 h^-1` the two members
whose `mu` falls below `D` wash out (`X` 3e-10) and the third persists. `washed_out`
is reported only for `D > 0` — in a batch culture every `mu` is 0 at the end
because the pool is empty, not because anyone lost.

**What it is good for at the accuracy actually measured.** Ordering, structure and
qualitative dynamics — who dominates, what is exchanged, when the culture stops.
Cross-feeding recall is 1.00 at sizes 3–21 (§8.1). It is **not** good for
quantitative yield: 3–8% on log-biomass, and on a *batch* culture the endpoint
turns on which metabolite empties first, which flips under changes that improve
the right-hand side on every measure (§8.1, the MM-clamp result). Report rates and
structure; treat a batch endpoint as an estimate with a wide error bar, and prefer
a chemostat steady state (§13.4) when a number has to be quoted.

### 13.2 Maximise a member's growth rate over the medium — convex — **built 2026-08-30**

`cfs maximise-growth` (`src/cfs/science/growth.py`). Projected gradient ascent in
`c`: the head's own analytic gradient chained through `dx/du . du/dc`, and one
bisection on the budget multiplier for the projection onto
`{c_lo <= c <= c_hi, cost . c <= B}`. The output calibration is an increasing
scalar map, so it cannot move the argmax and is applied only when a `mu` is
reported. Head B is not needed, so `compose.dfba.Surrogate` now takes
`behaviour_dir=None`.

**P21 is not a footnote here, it is the result.** With the box alone the designer
pays for carbon by zeroing ~50 cheap metabolites at once, and the LP at its
"optimum" does not grow at all — `mu_true` 12.1 -> 0.0 while the head reports an
improvement. That is 2 of the first 4 cases with no trust region, and still 3 of
20 under an additive one; every one of those had zeroed something. The fix is a
trust region in `x`, the head's own coordinate and the one §6.3's
nearest-training-medium distance is measured in, centred on the §4.3 start draw,
and **multiplicative** (`--trust-decades`, default 0.5) because §4.3's bands are:
an additive radius still lets a trace metabolite at `x ~ 0.1` reach exactly zero,
and one missing essential takes `mu` to 0 however good the rest of the medium is.

| trust region | improved | median gain | max gain | worst | median abs optimism |
| --- | --- | --- | --- | --- | --- |
| additive, radius 0.2 | 15/20 | — | — | **-100%** (x3) | — |
| 0.25 decades | 17/20 | +2.2% | 1.2x | -0.0% | 0.30% |
| **0.5 (default)** | 17/20 | +2.2% | **3.6x** | **-14.8%** | 0.30% |
| 1.0 decades | 18/20 | +2.2% | **106x** | -0.0% | 0.28% |

1. **The gradient direction is good enough, as §13.7 predicted.** Median optimism —
   `mu_hat(c*) - mu_true(c*)` relative to the LP — is **0.3%**, and the ascent
   improves the *true* LP on 17-18 of 20 cases. The rest are start media already at
   the plateau, where the correct answer is that there is nothing to buy.
2. **The tail is where the value is.** Median gain is 2.2% at every radius, but the
   best case goes 1.2x -> 3.6x -> 106x as the region widens: the large gains are
   near-starving start media rescued by reallocating the same total budget.
3. **The single loss is a low-`mu` start** (`CP001726.1`, `mu = 2.0` -> 1.71 at 0.5
   decades), the band Head A is documented worst in (§7), and it is *not* monotone
   in the radius — 0.25 and 1.0 both pass. Read it as one point where the head is
   optimistic, not as a trust-region trend.
4. The region is a reallocation: a metabolite the start medium has none of stays at
   zero. Seeding the start medium is how "should I add X" gets asked.

Not done: cost vectors other than uniform, the selective-medium DC program, and
the community version at a steady state (that is M12).

```
maximise    mu_k(c)        subject to    sum_m cost_m c_m <= B,   0 <= c <= c_max
```

Concave objective, linear constraints: projected gradient in `u`-space converges
to the global optimum, and the gradient is the head's own analytic one. Head A's
worst *gradient* cosine is 0.956, and the argmax depends only on the gradient
direction, so this is the use case whose requirement the current heads most nearly
meet. The community version — maximise member `k` while the others are present —
is the same program with `mu_k` evaluated at the §13.4 steady state, and is no
longer convex; do the static one first and use it as the warm start.

**Selective media are the same problem, one sign flipped:** `mu_k(c) - max_{j!=k}
mu_j(c)` is a difference of concave functions, so it is *not* convex — DC
programming (convex–concave procedure) or §13.6's sampler. Worth doing: "a defined
medium on which member `k` outgrows the rest of this community" is a directly
testable wet-lab claim.

**Always round-trip the answer through the true LP (V5/P4).** An optimiser's whole
job is to find where the surrogate is most optimistic.

### 13.3 Minimal medium — convex, and this is §9

§9's program with the structure made explicit: `mu_community(c) >= mu_target` with
`mu_community = min_i mu_i` is a *concave* constraint, so the feasible set is
convex and the L1-penalised objective is a convex program. Two versions, and the
difference matters:

* **Static** — the growth floor is on `mu_i(c)` at `t = 0`. Convex, unique, and
  the one to build. It answers "what is the smallest defined medium on which this
  community all grows at rate `mu_target`".
* **Dynamic** — the floor is on the §13.4 steady state or on a batch yield. Not
  convex, because the equilibrium map is not. Needs §13.6 or a homotopy from the
  static answer.

V6 (exact MILP on the true community model) validates the static one, and §9.1 is
right that this is the headline claim. §9.2's shadow prices come free from the
same solve.

**The known risk is exactly where Head A is weakest.** A minimal medium is
determined by the *binding set* — which metabolites are limiting — and the
roster's persistent worst cells are the ions (`EX_mg2_e`, `EX_cl_e`, `EX_ca2_e`;
§7). A minimal-medium answer that drops an ion is the failure mode to look for
first, and the MILP comparison is what finds it.

### 13.4 Chemostat steady state, coexistence and stability — needs M6

Newton-solve `dc/dt = 0, mu_i(c) = D` rather than integrating (§8.1 already says
this, and §8.4 is the price form). The steady state is the right place to quote
numbers, to define objectives, and to differentiate through: trajectory gradients
are badly conditioned, an equilibrium's are not, and the implicit-function
derivative gives sensitivity to every medium component, every abundance and every
`Km` at the cost of one linear solve.

What falls out for free once it exists:

* **Coexistence** — which members survive at dilution `D`, i.e. the classical
  competitive-exclusion question, on a real community rather than a toy model.
* **Stability** — eigenvalues of the Jacobian at the fixed point.
* **Invasion / colonisation resistance** — `mu_new(c*) - D` at the resident steady
  state: one head evaluation per candidate invader.
* **Keystone members** — leave-one-out over the community, `N` simulations, no new
  machinery at all. Cheap enough to be worth doing before anything clever.

Conditioning is measured (§7, "the conditioning bill is not §8's"): the Hessian
sum is rank ~10–25 of 365, so the supply term is what makes the solve well-posed
and `J` must be **diagonally preconditioned** whatever the head is. P9's damping
and `throw=False` apply; log the failure rate.

### 13.5 Maximise metabolic interaction — non-convex, and the weakest use case

Define the interaction rate as the mass actually handed between members:

```
E(c, X) = sum_m min( sum_i X_i max(z_im, 0),  sum_i X_i max(-z_im, 0) )
```

— per metabolite, the smaller of total secretion and total uptake, so a metabolite
everyone excretes and nobody eats scores zero. Maximise over `c` at the §13.4
steady state. Non-concave (a min of two objectives that are neither), so:
multistart projected gradient, or §13.6 at a temperature, which additionally
answers "how many *different* media achieve this" — a more useful answer than one
point.

**This is the use case with the weakest support from the measurements**, and it
should be labelled as exploratory in any writeup. It depends on flux *magnitude*,
which is Head B's worst axis (worst held-out R² 0.907 against a worst flux cosine
of 0.995 — the direction is far better than the size), and on cross-feeding
structure, which the per-organism labels never contain directly. That structure is
recovered at 1.00 recall (§8.1), which is the reason to attempt it at all.

### 13.6 Where HMC is the right tool — and where it is not

Three of the four use cases above are optimisation, and a sampler is the wrong
instrument for them: it is slower, it needs the surrogate's error model, and for
§13.2/§13.3 it would sample a distribution whose mode a convex solver already
returns exactly. Sampling earns its place in exactly two places:

**(a) The inverse problem — infer the medium from the community.** Given observed
abundances (16S, plating, OD) and optionally measured metabolite depletion, sample

```
p(c | X_obs) proportional to  p(X_obs | c) p(c)
```

with the forward map from §13.1/§13.4. This is a genuine posterior with no
optimisation counterpart, it is the scientifically strongest thing in this
section, and it is what the differentiability of the heads is *for* — 365
dimensions is far past anything a gradient-free sampler will do. It is also where
the whole Phase-5 accuracy story becomes a likelihood width rather than a gate.

**(b) Design ensembles.** `p(c) ∝ exp(objective(c) / T)` on a budget-constrained
support: instead of one designed medium, the family of media that work. That
directly gives which components are pinned and which are free — §9.2's point,
generalised — and it is robust where a single argmax of an imperfect surrogate is
not.

**The prerequisite is an error model, and we do not have one.** A deterministic
surrogate plugged into a likelihood produces a posterior that is confidently
wrong: all of the error lands in the prior's tails and none in the data's. The
held-out residuals are already measured per organism (§7.3, §6.3), so the cheap
version is a per-organism Gaussian on `log mu` and on `z`, widened by the M5
replicate spread. Do that before any chain. V7 (simulation-based calibration) is
the check, and P5 (no warm-starting from chain history) and P6 (multiple
equilibria show up as clustered divergences, and no step size fixes them) both
apply as written. **V4 — finite-differencing the full objective gradient at 20
points — comes before any of this**, per the validation table.

### 13.7 Accuracy required, per use case

The honest version of "M3 and M5 did not meet their gates": the gates were set for
the hardest downstream use, and most uses need less.

| Use case | Depends on | Currently | Verdict |
|---|---|---|---|
| §13.1 simulation, structure and ordering | `z` direction, cross-feeding | flux cosine 0.995, recall 1.00 | usable now |
| §13.1 quantitative yield | integrated `mu` | 3–8% log-X, batch endpoint unstable | rates only; prefer §13.4 |
| §13.2 growth maximisation | `d(mu)/dc` direction | worst cosine 0.956 | usable now, round-trip every answer |
| §13.3 minimal medium | the binding set | ions are the worst cells | build it; V6 decides |
| §13.4 steady state | `mu` level + Jacobian | `mu` rel err <= 3%; `J` rank-deficient | needs M6 + preconditioning |
| §13.5 interaction | `z` magnitude | worst R² 0.907 | exploratory |
| §13.6 posterior | a calibrated error model | none exists | blocked until one does |

### 13.8 New pitfalls

| ID | Pitfall | Symptom | Solution |
|---|---|---|---|
| P20 | Deterministic surrogate inside a likelihood | Posterior far too narrow; SBC ranks pile up at the edges | Per-organism residual model from the held-out set, widened by the M5 replicate spread. No chain before it exists |
| P21 | The designer walks out of the design | Spectacular objective, LP disagrees; medium far from any training medium | Trust region on the §6.3 nearest-training-medium distance — the same metric that diagnosed P18 — plus V5 at every reported optimum |
| P22 | An objective on flux *magnitude* | Inherits Head B's weakest axis while the diagnostics (cosine, sign agreement) look fine | Prefer direction- and structure-valued objectives; label magnitude-valued results exploratory |
| P23 | Optimising a batch-culture endpoint | The answer flips under changes that improve the right-hand side on every measure — the endpoint turns on which metabolite empties first | Optimise rates, or a chemostat steady state. Never a batch endpoint |
| P24 | A relabel that improves every held-out metric and breaks composition | Worst grad cosine, value R², per-metabolite coverage and M11 all improve; §8.1 regresses 17x at n=21 | Held-out media come from the *same design that changed*, so they cannot see it. Score every design change on a **community-regime held-out set** (§8.5). The stratum-budget reading of P24 was measured and is wrong — see §4.3 |
| P25 | Tuning a training distribution against a proxy metric | The proxy moves exactly as designed, three times, and the downstream number does not follow | Co-limitation count, near-onset count, NN-distance in `x` and per-metabolite limiting rows are all refuted as predictors of §8.1 (§8.5). Do not spend a 5 h relabel on a metric that has not first been shown to correlate with the composition on runs already on disk |

### 13.9 Milestones

| M | Deliverable | Gate |
|---|---|---|
| M9 | `cfs simulate`, batch + chemostat | **done 2026-08-30**; agrees with `cfs community`'s surrogate path on `D = 0` |
| M10 | §13.2 growth maximisation, convex solver | Optimum survives V5 round-trip on 20 cases — **built 2026-08-30; 19/20 at the default trust region, 20/20 at 0.25 and at 1.0 decades.** Median true gain +2.2%, median optimism 0.3%. The one failure is a `mu = 2.0` start medium, the head's known weak band; it is not monotone in the trust radius. **Under an additive trust region 3/20 collapse to `mu_true = 0`, and under none at all 2 of the first 4** — P21, and the mechanism is zeroing an essential trace metabolite |
| M11 | §13.3 static minimal medium | **built 2026-08-30; the essentiality blocker is closed 2026-08-31, V6 still short.** `cfs minimal-medium`: convex penalty solve + a greedy cardinality prune, one case per medium draw. **Head A cannot represent essentiality** — knocking a trace metal (`EX_cobalt2_e`, `EX_cu2_e`, `EX_mn2_e`, `EX_zn2_e`) out of a rich medium takes the true LP to `mu = 0` and moves the head by <1%, 6 of 37 free metabolites on a 3-member community. Unrestricted, the program exploits exactly that: 273 -> **41** components with every surrogate floor satisfied and `mu_true` 55/70/38 -> **0/0/0**. With the lethal singles pinned from the models (`--keep-essential`, default; one FBA per free metabolite, a static property of the GEM), 273 -> 251 and 2 of 3 members clear a 0.5 floor under the LP, the misses being 0.489/0.485 — i.e. ~2% short — and one real failure at 0.334 on the community's slow member (`mu_true` 3.5 against 55 and 70), Head A's known weak low-`mu` band. **The cause is `SamplingConfig.log10_lo = -4`**: the trace metals' limiting regime is at `c/Km ~ 1e-9..1e-6`, outside the probe's bracket, so the probe omits them, `band_scales` defaults them to 1.0, the design never makes them scarce, `_kink_scale` defaults `x_scale` to 1.0 and the head has no resolution left in that coordinate. The four missed essentials are exactly the four `"source": "default"` bands in the sidecar. **Fixed by `probe_lo = -12` (§4.7) and a relabel: `n_missed_essential` 6 -> 0**, and unrestricted the design no longer collapses the LP (2/3, 0/3, 3/3 members clearing the floor, worst true fraction 0.436 against 0.000). V6 still does not pass at a 0.5 floor — 0.491 / 0.436 / 0.512 — so what remains is a few-percent accuracy question, not a structural one |
| M12 | §13.4 steady state + stability + invasion | V4 passes; Newton failure rate logged and < 1% |
| M13 | §13.5 interaction maximisation | Reported with the V5 round-trip and labelled exploratory |
| M14 | Error model + §13.6(a) posterior | V7 (SBC) passes |

M9–M11 need nothing that does not already exist. M12 is M6. M13 and M14 are the
research half, and M14 is blocked on a piece of work — the error model — that is
small and has not been started.

## Appendix — repository layout

```
community-fba-surrogates/
├── config/
│   ├── organisms.yaml
│   ├── metabolite_index.json      # frozen, hashed
│   ├── km_defaults.yaml           # four transporter classes
│   └── sampling.yaml
├── src/cfs/
│   ├── groundtruth/
│   │   ├── qc.py                  # EGC check, MEMOTE driver
│   │   ├── index.py               # §2.1 universe derivation
│   │   ├── uniqueness.py          # §5 — elastic net / FBA switch
│   │   └── solve.py
│   ├── sampling/
│   │   ├── active_subspace.py     # §4.2
│   │   ├── design.py              # §4.3 stratified Sobol
│   │   └── generate.py            # parallel driver -> parquet
│   ├── surrogate/
│   │   ├── picnn.py               # Head A
│   │   ├── calibrate.py           # §7.3 concave output calibration
│   │   ├── behaviour.py           # Head B
│   │   ├── stacked.py             # §6.1 vmap harness
│   │   └── train.py
│   ├── compose/
│   │   ├── dfba.py                # §8.1 — built; `cfs community`, `cfs simulate` (§13.1)
│   │   ├── framings.py            # §8.2/§8.3 — not built
│   │   └── solve.py               # Optimistix wrappers — not built
│   ├── science/
│   │   └── minimal_medium.py
│   └── validate/
│       ├── degeneracy.py          # V1
│       ├── gradcheck.py           # V4
│       ├── roundtrip.py           # V5
│       └── exact_milp.py          # V6
├── tests/
└── notebooks/
```
