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

> **Implementation status — M5 built; the 1% gate is met at n=2/3/5 and n=21 as
> of 2026-09-02 (§8.6c), with n=10 at 2.5% and one cell of 30 at 0.318. The
> paragraphs below record the state at 2026-08-28, when it was not met; read them
> as history and §8.6c/§8.6d/§8.6e for where it stands. §8.6e also shows the gate
> **under-samples its own failure regime** — 42 of 780 member-states reach the
> depletion depth where Head B is 3300x wrong — so re-state it over depletion
> depth as well as over scarcity-matched media.** `src/cfs/compose/dfba.py`, CLI `cfs community`. Both frozen heads
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

#### §8.6a — cut selection beats cut fitting on n=21

`--epochs 0 --gm-init labels` is a supported mode: the head is the selected
label-tangent model with the slopes left as the duals wrote them. With Level 1
selection over community-regime trial media **and** the validity repair, `n=21`
is **0.175** — against 0.272-0.36 for every trained head across nine
interventions, and the first time that cell has moved at all. A1 median 0.00033
(2x better than the trained head) and worst p90 0.218 (vs 0.311).

Three things it establishes. **The trial-point set is the lever** — the identical
frozen model selected by `active-set` instead gives 0.336, so which 1000 of ~3985
cuts are kept is worth a factor of two on that cell. **The repair is mandatory
for a frozen head** — untrained cuts have never absorbed the softmin's
`T*ln(K_active)` offset, so A1 `median_signed` is -0.021 and the bulk is 10x
worse (n=2 0.058) until the uniform shift fixes it. **Training still buys the
middle** — n=3/n=5 stay better trained, `overall` 0.007 vs 0.013.

**And the selection axis is now closed, from both ends.** A 10x trial pool (20 000
community-regime media, free of solves — `_trial_points` never reads `mu`) keeps
2.4x more cuts and moves n=21 by 0.006 (0.175 -> 0.169) — which is one medium draw
wobbling, not a trend: paired per draw it is 0.1748/0.0229/0.3429 ->
0.1686/0.0232/0.3431, and 200 000 points returns to 0.175 exactly. In the other
direction, **1010 dFBA trajectory states** — `c_true`, i.e. SDDP's forward pass
taken literally, and already on disk from any `cfs community` run — keep just
**3-10 cuts per organism (median 6)** and reproduce the 45-103-cut model cell for
cell (n=21 0.178, A1 median identical). Subsampling those states 5x selects the
same cuts. **The useful cut count is single digits**, against a K=1000 budget. With "~90% of tangents
never bind" and "K 1000 -> 2000 is inert" already on file, more points, more
budget and better ranking are all exhausted. What remains is the cuts themselves.

So the trade is now separated rather than conflated: the label tangents are right
for the tail, gradient training is right for the bulk. Interpolating them is what
proximal / level bundle methods (Lemarechal-Nemirovskii-Nesterov 1995; Kiwiel)
are for — a stability centre at the seeded tangents with a trust radius on how far
the slopes may move. **That is the next arm, and unlike every previous guess it
has a measured trade-off behind it.** The cheap ML shorthand is a
`||a_j - a_j^0||^2` penalty in `_loss`; the principled version is the bundle
method proper.

#### §8.6b — what n=21 actually was, and what the error actually is

**n=21 was structurally n=1** — one full roster — so it could never separate
"large community" from "this medium". 16 distinct **15-member** subsets x 2 medium
draws do, and they settle the M5 tail:

1. **The composition metric reduces exactly to Head A.** Over 480 (community,
   member) points, `logX_err = |mu_rel| x that member's true growth` at
   correlation **0.9935**, median residual 0.0000. The integrator, Head B and the
   pool sum contribute nothing.
2. **The failing regime is a member below ~2x its own `mu_scale`**: median
   `|mu_rel|` 0.072 (<1) and 0.041 (1-2) against **0.0004** above 2.
3. **So the n=21 cell is a medium draw.** The same community scores **0.023** on
   draw 100 (0 of 21 members scarce), 0.175 on draw 0 (21 of 21) and 0.343 on
   draw 200 (20 of 21). Every head-side fix in §8.6/§8.6a failed on n=21 because
   the failure was the draw.
4. **Size and scarcity are confounded by the benchmark.** `community_medium`
   draws one §4.3 medium over the *union* of the members' active subspaces and
   §4.3 bands a fixed *fraction* of it, so a larger union starves every member.
   Fraction of members below 2x `mu_scale` runs 0.13 / 0.33 / 0.33 / 0.67 / 0.65
   over sizes 2/3/5/10/21 — while n=15 on a replete draw scores 0.0185, like a
   3-member community. **Match `mu0/mu_scale` before comparing sizes**; both "error
   does not grow with size" and "n=21 is the whole remaining failure" were reading
   this confound.

**And the error itself is one number.** Contriving the regime at n=1
(`titrate_n1b.py`: hold everything replete, titrate one metabolite, 46 limiting
pairs, 1288 rows, `mu/mu_scale` to 1.2e-6):

- it is a **constant additive over-prediction of ~0.0105 `mu_scale`**, flat across
  five decades of `mu` and falling to 0.0006 only above 2x. The relative error
  explodes solely because the denominator vanishes — **it is not a low-`mu`
  accuracy problem**, which is why `--w-rel`, `--w-under` and `--w-tau` all failed
  on it;
- the offset **is the head's floor**: it never predicts below 0.005-0.015
  `mu_scale`, and on deeply starved rows offset and floor agree to **7e-5**
  (corr 0.894 over 18 organisms). A min of planes with no tangent anchored near
  `mu = 0` cannot descend. **100.0% of 1288 rows are over-predictions**;
- **7.1x spread across genomes** (worst: GCA_000007325.1, the organism that led the
  n=21 failures), and a *larger* spread across limiters within one genome —
  ABFX02 maltose **0.0002** vs chloride 0.0159, **75x**. Carbon sources cheap, ions
  expensive, **O2 worst** (0.0246). Same ranking the gradient cosine has always
  shown for mg2/cl/ca2, now on the value.

So `logX_err = (0.0105 * mu_scale / mu_true) x growth`, and the `mu0/mu_scale < 2`
boundary is where that crosses ~0.5%. `mu0/mu_scale` needs no LP, so it is a
**runtime** predictor of a 100x error and belongs in `cfs community` / `simulate`
output. Solutions are deliberately not proposed here.

#### §8.6c — the floor IS the smoothing, and `--gm-eval-temp` removes it (2026-09-02)

§8.6b characterised the offset and deliberately proposed nothing. It has a cause,
it is arithmetic, and it is a one-flag fix.

**The n=1 titration is now a harness, not a measurement.** The media are a
deterministic function of `(organism, limiter, dilution)` and the sidecar, so
`20hm_bands/n1_bench.py <value_dir>` rebuilds them with **no LP solves** and scores
any Head A checkpoint on the same 1288 rows in seconds. It reports the signed error
on the plateau and on the ramp, the ramp's **sd**, the local smoothing gap
`c*T*ln(n_active)` and the softmin support size `n_active`. Two readings crack it:

- **the ramp sd is 0.00000 on 44 of 46 pairs** — the offset is constant to five
  decimals over five decades of `mu`. One plane active, *correct slope*, wrong
  intercept. Not a floor, an intercept;
- **the offset tracks `-T*ln(n_active)`**: `n_active` 34 -> offset 0.0180,
  `n_active` 126 -> 0.0054, predicted difference 0.0131 against 0.0126 observed.
  O2 is the worst limiter (§8.6b) precisely because `n_active` ~ 3 there.

**`n1_decompose.py` splits it three ways.** `GCA_000007325.1 / EX_k_e`, all 28
dilutions, `mu_scale` units:

| model | value |
| --- | --- |
| hard min over **all 3985** label tangents | **exact to 5 dp at every row**, down to `mu` = 5e-6 |
| the head's own hard min over its selected 1000 planes | truth **+0.05330**, constant everywhere |
| smoothing gap, plateau (`n_active` 34-200) | -0.05313 |
| smoothing gap, ramp (single plane active) | -0.03526 |
| **net** | plateau **+0.0002**, ramp **+0.0180** |

The `+0.0533` *is* `repair_intercepts`' uniform smoothing lift. Per-plane repair is
exact (selection error 1e-5); the lift is then sized by the **max** local gap over
the training rows and applied uniformly, so wherever fewer planes are active than
at that maximum the lift is uncancelled — and a starved medium is the case with
*one* plane active, i.e. the smallest gap and the largest residual. **Selection is
not lossy, the labels are not insufficient, the slopes of the frozen head are
fine.** Cf. §8.5's E1: the all-tangent model being exact here is E1's "trained-head
deficit" verdict localised to a single scalar.

**`--gm-eval-temp` (new).** Training needs a soft argmax so gradient reaches every
plane; *inference* does not. The flag ships the head at a colder temperature —
`groupmax.with_temp` before the repair, and `gm_temp` in the checkpoint becomes the
shipped value so `train.load` reconstructs what was scored (`gm_train_temp` records
the other). Frozen level-1 head on `p4`, identical in every other respect:

| T | n=1 median ramp | worst | worst cos | med cos | low-`mu` bias | under-rate | A1 med / p90 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| unrepaired | -0.0041 | 0.037 | 0.9091 | 0.9367 | -0.171 | **1.000** | — |
| 0.01 (was default) | 0.01074 | 0.0387 | 0.9091 | 0.9352 | +0.0457 | 0.000 | 0.00033 / 0.218 |
| 0.001 | 0.00103 | 0.0039 | 0.9469 | 0.9767 | +0.0044 | 0.000 | 0.000044 / 0.0223 |
| **0.0001** | **0.00011** | **0.0004** | **0.9522** | **0.9807** | **+0.0004** | 0.000 | — |

**Strictly better on every held-out axis, with no trade** — because on a frozen head
`T` is pure evaluation smoothing with no optimiser interacting with it. The floor
falls 100x. The only cost is curvature -> 0 (P3), and `cfs master-jacobian` already
showed `T` does not reach §8.4: 0.01 -> 0.3 moved the curvature rank 22 -> 13 while
the supply term sets the conditioning.

**§8.1, 3 medium draws x the same 10 communities, uncalibrated throughout:**

| median log-X | n=2 | n=3 | n=5 | n=10 | n=21 | overall | max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| frozen l1 + repair, T=0.01 | 0.006 | 0.007 | 0.009 | 0.060 | 0.175 | 0.013 | 0.508 |
| **T=0.001** | 0.003 | 0.004 | 0.002 | 0.026 | **0.022** | **0.004** | 0.320 |
| **T=0.0001** | 0.003 | 0.004 | **0.000** | 0.025 | **0.009** | **0.004** | 0.318 |
| trained `p4` nc (reference) | 0.005 | 0.002 | 0.004 | 0.093 | 0.272 | 0.007 | 1.698 |
| `r1` (reference) | 0.006 | 0.016 | 0.028 | 0.027 | 0.027 | 0.014 | 0.739 |

Paired at n=21: draw 0 **0.175 -> 0.009**, draw 100 **0.023 -> 0.0008**, draw 200
**0.343 -> 0.318, unmoved**. **Sizes 2/3/5 and n=21 are under M5's 1% gate**; n=10
is 2.5%. §8.6b's identity is confirmed quantitatively: a 100x smaller offset gave a
19-29x smaller trajectory error on the two draws whose members were starved.

**Draw 200 is a different failure and the floor fix could not have touched it.**
`mu_rel_worst_member` is GCA_000151225.1 at **+0.254** with `mu_true` = 11.7 — a
*mid-`mu`* over-prediction, i.e. §8.5's class (a max-affine head loose where no
tangent is near), not §8.6b's. `dc_rel_median` is 0.878 there too. Run E1 on it
before choosing anything.

**`--gm-repair` now forces the identity calibration, and this cost a cycle.** The
two post-hoc corrections fight: the repair guarantees `mu_hat >= mu` on the
training rows, and `calibrate` is a least-squares fit on those same rows whose map
is downward, so it pulls the head straight back under them. End to end on an
otherwise-exact repaired head, `value_under_rate_low_mu` went **0.000 -> 0.977**.
A repaired head is already unbiased (n=1 residual 1e-4), so there is nothing for a
1-D output map to buy. This is a third instance of "the calibration is
design-dependent" — now with a structural reason to switch it off rather than a
measurement.

**What this retracts.** §8.6b's "solutions are deliberately not proposed" stands,
but its framing — a floor set by the absence of tangents near `mu = 0` — is wrong:
the tangents are there and are exact. And the whole low-`mu` branch is retired.
`--w-rel`, `--w-under`, `--w-tau`, the mid-`mu` stratum B2, the output calibration
and re-anchoring on relative error were all compensating for this one arithmetic
error; that is the **eighth refuted proxy**, and the cheapest to have avoided.

**What is left, measured.** The same treatment on a *trained* head
(`value_p4_nc`, repair at T=1e-4) fixes the median but floors at low-`mu` bias
**+0.023 against the frozen head's +0.0004**, n=1 worst ramp **0.027 vs 0.0004**.
With the smoothing removed that residual is **slope drift alone**, quantified for
the first time and per (organism, limiter) in seconds. That is where the
proximal/bundle interpolation belongs — "label tangents win the tail, gradient
training wins the bulk" now has a target, not a community cell.

Reproduce: `cfs train-value --epochs 0 --gm-init labels --gm-select level1
--gm-trial-media holdout_community/community_holdout.npz --gm-group 1000
--gm-temp 0.01 --gm-eval-temp 0.0001 --gm-repair --width 1 --depth 1`.

**Step 4: with the smoothing gone, the frozen head dominates the trained one
everywhere — "training wins the bulk" is retracted.** The same repair at T=1e-4
applied to the *trained* head (`value_p4_nc`), same 3 draws, same 10 communities:

| median log-X | n=2 | n=3 | n=5 | n=10 | n=21 | overall | max | A1 med / p90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **frozen l1 + repair, T=1e-4** | **0.003** | **0.004** | **0.000** | **0.025** | **0.009** | **0.004** | **0.318** | **0.000004 / 0.0093** |
| trained + repair, T=1e-4 | 0.004 | 0.005 | 0.004 | 0.057 | 0.322 | 0.005 | 0.837 | 0.000266 / 0.0491 |
| trained, T=0.01 (was the bulk winner) | 0.005 | 0.002 | 0.004 | 0.093 | 0.272 | 0.007 | 1.698 | — |

1. **The frozen head is now better at every size and loses nowhere**, where at
   T=0.01 the trained head won `overall` (0.007 vs 0.013) and sizes 2/3/5. That
   trade was an artifact of the uniform lift: the trained head's intercepts had
   absorbed the smoothing gap during training, so it paid less of the penalty the
   frozen head paid in full. Remove the gap and the ordering flips.
2. **Slope drift is worth ~90x at a scarce medium, not the ~50x the held-out rows
   suggested.** Paired at n=21: draw 0 **0.0091 (frozen) vs 0.8368 (trained)**,
   and the trained head gets *worse* there under the colder temperature
   (0.7761 -> 0.8368) while the frozen head goes 0.1748 -> 0.0091. The held-out
   low-`mu` bias gap (+0.0004 vs +0.023) understates it because the failure is one
   member at one medium, which is exactly what A1's `worst_p90` catches
   (0.0093 vs 0.0491) and its median does not.
3. **So the proximal/bundle arm loses its motivation.** It existed to interpolate
   between "label tangents win the tail, gradient training wins the bulk"; there
   is no bulk left for training to win. `--w-prox` was already refuted directly
   (monotonically harmful). Do not re-open it without a new measurement showing
   gradient training buying something the selected tangents do not.
4. **What is actually left is two cells, both §8.5's class, not §8.6b's.** n=10 at
   0.025 (draws 0.0006 / 0.0249 / 0.0533) and the n=21 draw-200 cell at 0.318,
   whose worst member is a **mid-`mu` over-prediction** (+0.254 at `mu_true` 11.7)
   with `dc_rel_median` 0.878 — Head B is implicated there too. Run E1 on that
   medium before choosing anything.

#### §8.6d — with Head A exact, M5's residual is Head B's coverage (2026-09-02)

After §8.6c, `mu_rel_median` is <= 0.0005 on **all 30 cells** (10 communities x 3
medium draws). §8.6b's identity `logX = |mu_rel| x growth` is therefore retired:
the term it was built from has gone to zero, and what remains tracks `dc_rel` /
`dc_cosine`. Per-member flux cosine is 0.9998 on the best cell and **0.74-0.96** on
the failing ones (`20hm_bands/dc_decompose.py`, which reconstructs the pool sum
from one FBA per member per state).

**A free predictor, and a refuted fix built on it.** `compose.dfba` projects onto
§3.3's uptake bound `z_m >= -Vmax_m * u_m` at inference. How hard that projection
works (`clamp_bite.py`, no LP) ranks the ten communities by trajectory error:

| cell | 4 | 2 | 5 | 7 | 1 | 0 | 6 | 3 | **8** | **9** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| logX | 0.0001 | 0.0030 | 0.0007 | 0.0005 | 0.0109 | 0.0126 | 0.0258 | 0.0336 | **0.0533** | **0.3181** |
| `\|dz\|/\|z\|` | 0.000 | 0.000 | 0.004 | 0.011 | 0.011 | 0.033 | 0.082 | 0.237 | **0.412** | **0.279** |

Individual predictions sit **13x** outside a bound every label satisfies, so
`--w-mm` adds the one-sided hinge on it — same shape and same justification as Head
A's `--w-under` (`behaviour.mm_floor`; `VMAX` now has one definition, in
`behaviour`, which `dfba` reads, so the training bound and the inference projection
cannot drift). **It is refuted, twice over:**

| median log-X | n=2 | n=3 | n=5 | n=10 | n=21 | overall | Head B worst cos / sign |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `behaviour_p4` | 0.003 | 0.004 | 0.000 | **0.025** | **0.009** | **0.004** | **0.9853 / 0.9629** |
| `--w-mm 1` | 0.003 | 0.004 | 0.001 | 0.041 | 0.009 | 0.004 | 0.9845 / 0.9613 |
| `--w-mm 10` | 0.004 | 0.006 | 0.001 | 0.055 | 0.015 | 0.005 | 0.9827 / 0.9557 |

1. **The violation is off-distribution and the loss cannot reach it.** On its own
   *training* rows Head B violates the bound on **0.053%** of entries (labels
   0.000%); `--w-mm 1` moves that to 0.046% and `10` to 0.037%. The 13x violations
   exist only at community-regime media the head never saw.
2. **Where the hinge did work, the composition did not follow.** At `w=10`, cells
   0/1/3/6 go bite 0.033 -> 0.0085 and worst ratio 0.12 -> 1.000, with their
   trajectory error unchanged. That refutes causation independently of (1): the
   bite is a **symptom of being off-distribution**, which is why it predicts so
   well, and not a cause.

Kept in the code, default 0, with the negative result on file — the discipline that
`--w-prox` and `--gm-temp-final` are kept under.

**The coverage proxy passes P25's gate, which none of the seven refuted ones did.**
`nn_proxy.py`, no LP, on the 30 cells already on disk: Spearman(median NN distance
in `x` to the member's own training media, `dc_rel`) = **+0.673** (p=4.6e-5), and
`+0.515` against `logX` itself. The six worst cells are the six farthest (median NN
1.33-2.63) and the best sit at 0.11. So the fix is coverage of the media §8.1
actually visits — `make_traj_pool.py` already extracts those states from any
`cfs community` run, and unlike the Level 1 trial pool, labelling them costs real
solves. **`cfs generate --media <npz> --round N` takes that pool directly** — it
skips the design, the probe and the sidecars and labels exactly those states.

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

**Option 2 — a chosen quantile instead of a tuned weight. IMPLEMENTED
(`--w-tau`), REAL, AND DOMINATED BY `--w-under`.** `--w-tau` makes the value term
an expectile (asymmetric least squares, Newey & Powell 1987): residuals on the
under-predicting side get weight `tau`, the rest `1 - tau`, with the `2x` chosen
so `tau = 0.5` reproduces the plain MSE **bit for bit** — verified,
`value_p4_tau0.5` matches `value_p4` on the `grad_cosine` of all 21 organisms to
1e-9, so `lr` and `w_grad` keep the scale every number on file was measured at.

| 21 organisms, held out | worst cos | med cos | med R2 | low-`mu` under-rate | A1 worst p90 |
| --- | --- | --- | --- | --- | --- |
| `tau = 0.5` (= `value_p4`) | 0.9211 | 0.9630 | 0.9994 | 0.558 | 0.4493 |
| `tau = 0.7` | 0.9237 | 0.9595 | 0.9993 | 0.539 | 0.3045 |
| **`tau = 0.9`** | 0.9076 | 0.9578 | 0.9993 | **0.517** | **0.1298** |
| `tau = 0.99` | 0.9228 | 0.9579 | 0.9989 | 0.508 | 0.5322 |
| `--w-under 1` (the hinge) | 0.8985 | 0.9626 | 0.9993 | — | **0.0509** |

3 medium draws x the same 10 communities, all uncalibrated:

| median log-X | n=2 | n=3 | n=5 | n=10 | n=21 | overall | max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `tau = 0.5` (baseline) | 0.005 | 0.002 | 0.004 | 0.093 | **0.272** | 0.007 | 1.698 |
| `--w-under 1` | 0.006 | 0.002 | 0.005 | **0.055** | 0.346 | **0.006** | **0.462** |
| `tau = 0.9` | 0.014 | 0.001 | 0.005 | 0.077 | 0.327 | 0.015 | 1.658 |

1. **It works, and an earlier prediction that it would be inert is retracted.** At
   `w_grad = 10` the value term is 0.3% of the objective (`loss = 0.248,
   value = 0.00071, grad = 0.02473`), and the reasoning that a tilt inside 0.3% of
   the gradient cannot matter is **wrong**: A1's `worst_p90` falls 3.5x
   monotonically over `tau` 0.5 -> 0.9, and `tau = 0.9` flips the failing n=21
   member's sign exactly as the hinge does (GCA_000007325.1 **-0.857** ->
   DACTBY01 **+0.713**). A small term can still decide a tail.
2. **There is an optimum and it is not at the end.** `tau = 0.99` turns over hard
   (A1 p90 0.130 -> 0.532): with the over-prediction side at weight 0.02 nothing
   holds the plateau down. Do not read "more one-sided is better".
3. **The hinge still wins, and the mechanism is the difference between them.** The
   expectile tilts **every** row, so it buys one-sidedness by degrading the fit
   everywhere — its community `mu_rel_median` is 0.254 against the hinge's 0.052 —
   while `--w-under` is an additive term that is exactly zero on rows already in
   compliance. For a *provable violation*, paying only at the violation is the
   right shape. A1's p90 ordering reproduced the composition's `max` ordering for
   a third time (0.051/0.130/0.449 -> 0.462/1.658/1.698) and again failed to
   predict n=21, which is one member at one community, not a tail over media.
4. **The durable win is the diagnostic.** `value_under_rate` and
   `value_under_rate_low_mu` are now in `score`, so any future one-sided knob is
   chosen from a checkpoint in seconds instead of a 5 h composition run.

5. **The combination is refuted: they do compete.** `--w-under 1 --w-tau 0.9`
   together lands on the *expectile's* behaviour and slightly worse than it, not
   between the two — worst cosine **0.8870** (the worst of the four arms), A1 p90
   0.1174, and the composition's worst numbers everywhere that matters:

| median log-X | n=2 | n=3 | n=5 | n=10 | n=21 | max | A1 p90 | worst cos |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `tau = 0.5` (baseline) | 0.005 | 0.002 | 0.004 | 0.093 | **0.272** | 1.698 | 0.449 | 0.9211 |
| **`--w-under 1`** | 0.006 | 0.002 | 0.005 | **0.055** | 0.346 | **0.462** | **0.051** | 0.8985 |
| `tau = 0.9` | 0.014 | 0.001 | 0.005 | 0.077 | 0.327 | 1.658 | 0.130 | 0.9076 |
| `--w-under 1 --w-tau 0.9` | 0.015 | 0.001 | 0.004 | 0.084 | 0.362 | 1.755 | 0.117 | 0.8870 |

   The tilt is applied to the *same* residuals the hinge acts on, so once every row
   is reweighted the hinge has no separate signal left to contribute — it is not
   two independent constraints, and the guess that it was is retracted. **Use
   `--w-under` alone.**

**Option 2 as originally specced.** `--w-under` is the
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

#### §8.6e — the coverage round, the attribution, and the depletion sweep (2026-09-02)

§8.6d ended blocked on code: `cfs generate` could not be told which media to
label. It can now (`--media <npz> --round N`), and three things followed.

**1. The coverage round ran, and coverage transfers in the bulk only.**
`label_pool_n15.npz` is 672 forward-pass states from the **16 n=15 communities x 2
medium draws**, strided 5 — deliberately *not* the 10 benchmark communities, since
`make_traj_pool.py`'s Level 1 selection pool may reuse them (no label is involved)
while labelling them would train on the M5 benchmark. 63/63 shards, 158 256 rows,
100% optimal, one `index_hash`, ~8.5 min/organism. Both heads retrained (P14).

| median log-X, 3 draws x 10 communities | n=2 | n=3 | n=5 | n=10 | n=21 | overall | max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| frozen l1 + repair, T=1e-4 (§8.6c) | 0.003 | 0.004 | 0.000 | 0.025 | **0.009** | 0.004 | 0.318 |
| + round 2 | **0.002** | **0.002** | 0.000 | 0.026 | 0.013 | **0.002** | 0.318 |

Paired per draw, **21 of 30 cells improve** and `dc_rel` falls on 21 of 30. The
mechanism check is the durable part: `nn_proxy.py` measures the distance at **`t=0`
only** — the one state a design draw already covers — so it barely moves. Measured
over *every* step (`nn_delta.py`, no solves) the round **halves the typical distance
and leaves the tail exactly where it was**: median over 30 cells **0.252 -> 0.119**,
p90 2.108 -> 2.129, max 2.315 -> 2.312. **One community's forward path does not
reach another's.** Covering the tail needs many more communities in the pool, or a
benchmark on fresh communities so the visited states can be labelled directly.
Held-out saw none of it (P24): Head A's worst cosine unchanged to 4 dp, Head B
+0.005 R2, against an overall composition halving.

**2. Attribution: rank metabolites, not genomes.** `attribute_b.py` +
`attrib_report.py` score Head B at 5 states along the true path of every cell of
every run given, one FBA per (cell, state, member), aggregating `X_i (z_hat -
z_true)`. Across two independent community sets — the 10 benchmark communities x 3
draws (780 member-states) and the 16 n=15 x 2 (2400):

| axis | Spearman between the sets |
| --- | --- |
| per-**metabolite** share of squared pool error | **+0.879** (p=3e-144) |
| per-**genome** median relative flux error | **+0.719** (p=2.4e-4) |
| per-**genome** share of squared pool error | +0.413 (p=0.06) |

A member's share is its difficulty **times** the biomass it reached in that
community, so it barely reproduces (`CP001820.1` is 54% of one set and 15% of the
other). Genome *difficulty* does. Same confound as §8.6b's "n=21 is scarcity, not
size", one level down. The metabolites are **not Head A's**: `EX_h2o_e`, `EX_h_e`,
`EX_akg_e`, `EX_succ_e`, `EX_acald_e`, `EX_nh4_e` — by-products and central carbon,
where Head A is led by `EX_mg2_e`/`EX_cl_e`/`EX_ca2_e`. The two heads fail on
different metabolites, so a coverage fix aimed at Head A's ions never applied.

**3. The depletion sweep — the Head B analogue of `titrate_n1`.** Head A's n=1
titration works because `mu` is a scalar min over limiters, so one-scarce-rest-
replete is a complete parametrisation. Head B's failing states are end-of-batch
media drawn down **at once**, which that design cannot construct. A **monoculture
batch is the sweep**: the medium walks down the organism's own consumption
direction with the true LP defining the path — and it is `cfs community` with 21
single-member communities, so it needed no new code (`monocultures.txt`,
`mono.sh`, `monodeep.sh` at 8 doublings, 3 draws each).

| depth (`mu_true` / `mu_true` at t=0) | n | med flux cosine | p05 cosine | med `\|z_hat\|/\|z_true\|` |
| --- | --- | --- | --- | --- | --- |
| 0.90-1.0 | 366 | 0.991 | 0.781 | 1.00 |
| 0.50-0.90 | 135 | 0.903 | 0.519 | 1.01 |
| 0.10-0.50 | 46 | 0.843 | -0.065 | 1.24 |
| 0.01-0.10 | 18 | 0.863 | **-0.440** | **3318** |

The cause is arithmetic, in `dfba.Surrogate.mu_and_z`: inference multiplies
specific flux by `max(mu, mu_floor)`, `mu_floor = _MU_FLOOR_FRAC * mean mu`. At
those states median `mu_true` is **1.9e-4** against a floor of 0.77 and **89% have
`mu_hat` below the floor**, so the floor sets the flux. `max(mu_hat, floor)/mu_true`
= **4172** against `mu_hat/mu_true` = **52** — the floor dominates, Head A's
residual is the smaller term.

**And removing it is refuted.** One line, same 30 cells: `dc_rel` **2.047 -> 0.747**
on the worst cell and 0.594 -> 0.428 on another, while their `x_log_err_final` goes
**0.0041 -> 0.1241** and 0.0676 -> 0.1124; 25 of 30 cells are bit-identical and no
size median moves. **Third instance of "a strictly better rhs is not a better
trajectory"** (§8.6d's MM clamp gave two) and the starkest: 2.7x better derivative,
30x worse endpoint. An over-predicted uptake empties the pool at roughly the right
*time*; correct it and a starving member keeps the pool alive too long, which
changes **which metabolite empties first**, and a batch endpoint turns on that.
Reverted; the floor stays.

**Two things this fixes for future work.** The benchmark **under-samples the
regime** — only 42 of 780 member-states are below depth 0.1 — so a gate that rarely
visits the failure cannot reward fixing it; state M5 over scarcity-matched media
(§8.6b) *and* over depletion depth. And normalise a depleting trajectory by a
**fixed** scale with dead states dropped: the first pass reported a median relative
error of **3e31**, the same divide-by-a-dead-culture trap §8.1 already guards
against. Quote cosine beside it — a fixed denominator flatters the deep end exactly
as a local one explodes, and the median hides the failure entirely (0.99 -> 0.86)
where the p05 shows it (0.78 -> -0.44).

#### §8.6f — Head B's target is ~25-dimensional, and the ranked plan (2026-09-02)

§8.6e localised Head B's error and refuted its obvious fix. This section asks what
*structure* the head is missing, measures three candidates against the labels
before building any of them (P25: a proxy earns a retrain by first predicting
something on runs already on disk), and refutes two of them in minutes.

Three scripts, no LP solves, all under `20hm_bands/`: `bound_binding.py` (labels
only), `gate_check.py` and `flux_rank.py` / `offmanifold.py` (labels plus a frozen
head pair; `value_p4_ncrep_T0.0001` + `behaviour_p4` throughout).

**1. Complementary slackness holds exactly on the labels — and the clamp already
collects it.** Wherever a stored dual is non-zero, that exchange's flux sits on
§3.3's uptake bound: `dual => tight` is **0.996-1.000** on every organism in every
`mu` band. So the limiting metabolites' fluxes are a closed-form function of the
medium, and Head A's gradient — which §8.6c leaves at `mu_rel <= 5e-4` — is an
active-set classifier for them. That is Bertsimas & Stellato's "predict the active
set, recover the solution exactly", available for free.

**It buys nothing, because `mu_and_z`'s projection already lands there.** Head B
over-predicts uptake magnitude, so on truly-tight entries the clamp is what
answers, and the head's relative error on them is **0.000 in every depth bin**.
The tight set also carries only **1-10% of the squared flux error**; 48-69% is on
**secretion** entries, which are positive, unbounded above, and reachable by no
projection. Refuted before it was built.

**2. The label flux set is 12-39 dimensional out of 138-259 exchanges.** SVD of
each organism's training specific-flux matrix `z / mu`:

| explained | rank, roster range |
| --- | --- |
| 99% | **3-12** |
| 99.9% | 8-23 |
| 99.99% | **12-39** |

And the basis generalises: projecting the **held-out** true flux onto the
training basis costs a median relative error of **0.0009-0.0046**, i.e. 30-100x
below the head's own 0.09-0.26. This is the LP's geometry showing through — the
optimum is a vertex, `c -> z` is piecewise affine over critical regions (mpLP),
and the reachable set is the flux cone's inner description. A conservation law
(elemental balance, a left-null-space moiety) is by definition a direction the
label matrix has zero variance in, so the SVD absorbs every one of them without
needing formulas from the GEM.

**But projecting the existing predictions is not the fix.** Post hoc, `z := V V^T z`
moves the held-out relative error 0.1238 -> 0.1228 — the head's error is already
almost entirely *inside* the subspace. The off-manifold component is 0.4-1.9% at
held-out media and **0.9-8.7% at the dFBA trajectory states §8.1 visits** (median
ratio **2.5x**, `offmanifold.py`, no truth needed), so it does grow exactly where
the composition fails, but a few percent of the norm cannot explain a 12-26%
error. **The basis is worth having as a reparametrisation, not as a projection**:
a head that emits ~25 coefficients instead of ~200 fluxes cannot leave the
manifold at all, and its output layer is 10x smaller.

##### The ranked series, cheapest first

| # | attempt | evidence it rests on | cost |
| --- | --- | --- | --- |
| **B1** | **Low-rank basis head.** `z / mu = w . V`, `V` the top-`r` right singular vectors of the training specific flux, `r` by a 99.99% cutoff; `V` in the checkpoint, clamp applied after. | rank 12-39; held-out oracle **0.001-0.005**; off-manifold 2.5x at trajectory states | SVD in `data.py`, one matmul in the head, one retrain |
| **B2** | **Weight the loss by what `dc` feels.** 48-69% of the squared error is on secretion, and the composition consumes `sum_i X_i z_i`, not a per-metabolite MSE. Check the weighting on training rows before building it. | error split uptake/secretion/tight = 0.21-0.46 / 0.48-0.69 / 0.01-0.10 | one line in `_loss` |
| **B3** | **Drop sub-floor rows at training** instead of flooring them (`data._MU_FLOOR_FRAC`). The target is `z / mu`; below the floor the division is noise and the head is fitting it. Inference keeps the floor — removing it there is refuted (§8.6e). | 89% of the failing states have `mu_hat` below the floor | one line in `load_behaviour_dataset` |
| **B4** | **Re-state M5 over depletion depth**, not only over scarcity-matched media. | 42 of 780 benchmark member-states reach the regime | measurement only |
| **B5** | **Conservation / elemental balance as a DC3-style completion.** Predict the free coordinates, solve the balance for the rest. | *probably subsumed by B1* — a conservation law is a null direction the SVD already removes | needs formulas from the GEMs; verify `E z = mu b` on the labels first |
| **B6** | **Piecewise-affine / active-set head.** mpLP says `c -> z` is affine on critical regions and discontinuous across degenerate ones; Head A's gradient already names the active set. | the class is right; nothing measured | a different architecture |

**Refuted today, on file so nobody re-runs them:** the complementarity gate (1
above) and the post-hoc subspace projection (2 above).

##### B1 as built — `cfs train-behaviour --basis-var`

`behaviour.flux_basis` takes the SVD of each organism's training specific flux,
keeps the leading directions to an explained-variance cutoff (`--basis-var`,
default 0.9999, `0` restoring the old full-width head), and the head's output
layer emits those coordinates instead of one free flux per exchange. The basis is
a non-trainable leaf of the module, so it serialises and round-trips with `mask`;
`arch.basis_rank` is what `load` rebuilds the skeleton at. Everything downstream
is untouched — the loss, the `--w-mm` hinge, `evaluate`, and `compose.dfba`'s MM
clamp all still see a full-width `z`.

**The coordinate the SVD is taken in is load-bearing, and the obvious choice is
the wrong one.** Building the basis in the head's own `z_scale`d units — which is
where the loss lives, so it looks like the natural place — gives rank **59-101**
instead of **12-39** on the same labels at the same cutoff. `z_scale` divides each
metabolite by its own std, which promotes every ion to the same footing as the gas
exchanges and puts the discarded directions back. The compression is a property of
the space the *composition* consumes (`dc = sum_i X_i z_i`, in mmol/gDW/h), so the
basis is built there and mapped into the head's units afterwards, at the cost of
the rows no longer being orthonormal — which nothing depends on, only the span.
Same family of error as measuring a gradient cosine in the network's own input
coordinate (§7.2): a rank is not invariant to a rescaling of the axes.

##### B1 is measured, and it is null — default off (2026-09-02)

`behaviour_p4r2_basis`: rank 12-39 (max 39), 600 epochs, lr 1e-3, seed 0,
otherwise identical to `behaviour_p4r2`; Head A untouched (`value_p4r2`), so this
is a clean A/B on the same labels, the same 10 communities and the same 3 medium
draws.

| median log-X | n=2 | n=3 | n=5 | n=10 | n=21 | overall | max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `behaviour_p4r2` (control) | 0.002 | 0.002 | 0.000 | 0.026 | 0.013 | **0.002** | 0.318 |
| **+ B1 basis** | 0.002 | 0.002 | 0.000 | 0.028 | 0.012 | **0.002** | 0.318 |

Held out: worst R2 **0.9354 -> 0.9199**, median 0.9638 -> 0.9564, worst flux
cosine 0.9826 -> 0.9793.

1. **Null on the trajectory** — identical to three decimals at every size, 11 of
   30 cells better, `max` unchanged. Not a regression either.
2. **It trades exactly as the restriction predicts, and the trade nets to
   nothing.** Paired by cell, on `dc_rel`: the 15 easy cells go **0.079 -> 0.147**
   (worse on 13 of 15) and the 10 hard ones **0.920 -> 0.840** (better on 5 of
   10). Discarding 0.01% of the flux variance costs bulk accuracy everywhere and
   buys a coin flip off-distribution.
3. **Fourth instance of "a strictly better rhs is not a better trajectory."** The
   0.318 cell's `dc_cosine` improves 0.9559 -> **0.9745** and its `x_log_err_final`
   moves by 0.0001.
4. **Why it could not have been the lever, in hindsight**: §8.6f already measured
   the off-manifold component at **0.9-8.7%** of the norm at trajectory states.
   Removing it exactly can at most buy that, against a 12-26% error. The
   post-hoc-projection refutation was the same number and should have been read as
   an upper bound on B1's ceiling, not merely as "the projection is not the fix".
   **A structural constraint is worth at most the violation it removes — measure
   the violation first.**

Kept in the code, default **0** (off), the discipline `--w-prox`, `--w-mm` and
`--gm-temp-final` are kept under. The basis object itself is still worth having:
it reconstructs held-out truth to 0.001-0.005 and is the cheapest available
statement of what Head B's target actually is.

**What this leaves for Head B**: B2, B3 and B4, taken in that order below. B5 is
now firmly subsumed -- conservation relations are inside the subspace B1 restricted
to, and restricting to it changed nothing.

##### B2 is refuted before it was built — the relative error is flat in flux scale

The premise was that the loss divides every metabolite by its own std (`z_scale`),
so a 1e-3 ion counts as much as a 400 mmol/gDW/h gas, while `dc = sum_i X_i z_i`
is consumed in raw units where the gases dominate -- and §8.6e's attribution found
the error on exactly those (`EX_h2o_e`, `EX_h_e`, `EX_akg_e`, `EX_succ_e`).

**The attribution was reading a scale effect.** Raw squared error is dominated by
big fluxes whatever the model does, so the question is whether the *relative*
error is worse there. Held out, per (organism, metabolite), 2272 cells
(`20hm_bands/loss_alloc.py`, no solves):

| `z_scale` quintile | median `z_scale` | median relative error | share of raw squared error |
| --- | --- | --- | --- |
| 1 | 2.5e-4 | 0.034 | 0.000 |
| 2 | 0.020 | 0.662 | 0.000 |
| 3 | 0.076 | 0.311 | 0.002 |
| 4 | 0.388 | 0.688 | 0.093 |
| 5 | 3.5 | 0.219 | **0.905** |

**Spearman(`z_scale`, relative error) = +0.018.** 90.5% of the raw error sits in
the top quintile purely because those fluxes are ~1e4x bigger. Re-weighting toward
raw units would chase error that is already proportionally as accurate as the
rest, and would sell the small metabolites -- which are the ones that decide *which
metabolite empties first*, and a batch endpoint turns on that. Not built. This is
the check that would have saved B1, applied first this time.

##### B3 is the wrong sign — the floor is self-consistent, and it is not learned

The premise was that flooring `mu` corrupts the target. **It does not.** Training
divides by `max(mu_label, floor)` and `compose.dfba` multiplies by
`max(mu_hat, floor)`, so the reparametrisation is consistent end to end: with the
head right and `mu_hat ~ mu_label`, `z` is recovered exactly however far below the
floor the state is. Dropping those rows would remove the only supervision the
floored regime has -- and there is real supervision to remove: **12% of training
rows (9-20% per organism) are sub-floor**, carrying 3% of the squared target
(`20hm_bands/floor_rows.py`). The p4 design is bottom-heavy after `probe_lo = -12`,
so this is far more than the "1% of media" the original note assumed.

**What §8.6e actually measured, re-read.** At the depletion-sweep states
`|z_hat| / |z_true|` = 3318 and `max(mu_hat, floor) / mu_true` = 4172. Those agree
only if the head is emitting `z_true / mu_true` -- the **un**floored specific flux
-- where its training target at such `mu` is `z_true / floor`. So the head
extrapolates the above-floor relation into the sub-floor region instead of the
floored one it was trained on. Not a shortage of sub-floor rows: a shortage of
sub-floor rows *at co-depleted media*, since §4.3's low-`mu` strata starve one or
a few metabolites while a batch endpoint draws the whole pool down at once. That
is §8.6d's coverage finding again, one regime deeper -- and it makes B4 the
prerequisite, not the afterthought.

##### B4: M5's gate was a property of the horizon (2026-09-03)

`20hm_bands/depth_gate.py` conditions the trajectory error on **depletion depth**
-- a member's true growth rate over its own rate at `t = 0` -- with **no LP
solves**: `integrate` advances biomass as `X * exp(dt * mu)`, so `mu_true` is
recoverable from the stored `x_true` exactly. Over the 30 benchmark cells at the
default 4 doublings, 15 446 member-steps:

| member depth | share of steps | median log-X err | p90 |
| --- | --- | --- | --- |
| 0.9-1.0 | **0.862** | 0.0000 | 0.0028 |
| 0.5-0.9 | 0.110 | 0.0058 | 0.0119 |
| 0.1-0.5 | 0.014 | 0.0084 | 0.0167 |
| 0.01-0.1 | **0.001** | 0.0672 | 0.0675 |
| < 0.01 (dead) | 0.003 | 0.0012 | 0.0023 |

**86% of the benchmark is spent within 10% of maximum growth, and 0.4% below
depth 0.1** -- 51 steps of 15 446, all of them in 2-member communities; sizes 3,
5, 10 and 21 never go below 0.1 at all. §8.6e's "42 of 780 member-states" was
right, and this is the same number at 20x the sample, for free.

Re-running the identical 10 communities and 3 draws at `--doublings 8`:

| median log-X | n=2 | n=3 | n=5 | n=10 | n=21 | overall | max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 4 doublings (the gate as stated) | 0.002 | 0.002 | 0.000 | 0.026 | 0.013 | **0.002** | 0.318 |
| **8 doublings** | 0.031 | 0.135 | 0.041 | 0.053 | 0.054 | **0.041** | 0.572 |

1. **"M5 met at n=2/3/5 and n=21" was a statement about the horizon.** Doubling it
   costs 20x overall and **every size fails the 1% gate**, by 3-13x. Nothing about
   the heads changed. The share of member-steps below depth 0.1 goes 0.017 -> 0.185
   at n=2 and 0.000 -> 0.146 at n=3.
2. **It is entirely Head B.** `mu_rel_median` is **3e-5 in both** and its max
   0.0387 in both -- the same single §8.5-class cell. `dc_rel` median goes
   0.190 -> **0.556** and max 2.05 -> **6.49**. V5 still does not bite
   (`overgrowth` max +0.071).
3. **The error peaks in the *transition*, not at the bottom.** Median log-X by
   depth at 8 doublings: 0.0001 (>0.9), 0.0101, **0.0304** (0.1-0.5), 0.0160
   (0.01-0.1), 0.0034 (dead). Once a member is fully starved both trajectories
   stop growing and the error freezes at whatever it accumulated on the way down,
   so the deepest bins flatter the model. **Quote the 0.1-0.5 band.** This refines
   §8.6e: its flux-magnitude blow-up is real, but it lands where `d(log X)/dt` has
   already gone to zero.
4. **State the gate with its horizon.** A batch community has two clocks (§8.1),
   and the inoculum is solved so the pool empties at the end of the horizon --
   which fixes *when* starvation happens but not how long the culture spends in
   it. Quote M5 at both 4 and 8 doublings, or with its depth distribution; a
   single number is a statement about the integration window.

**And B1 gets its fair test, and is still null.** The deeper gate is the sensitive
one, so the basis head was re-run on it: overall **0.041 -> 0.039**, sizes
0.024 / 0.160 / 0.040 / 0.057 / 0.051, better on **12 of 30** cells, `dc_rel`
median *worse* (0.556 -> 0.736). The refutation was not an artifact of a benchmark
that never visited the failure regime.

##### The depletion coverage round (round 3) is null, and the reach is why

B3's re-diagnosis pointed at coverage of *co-depleted* media: the head extrapolates
the above-floor relation into states §4.3's low-`mu` strata cannot build, because
they starve one or a few metabolites while a batch endpoint draws the whole pool
down at once. Round 2 covered the community regime at `t ~ 0`; this covers depth.

`make_depl_pool.py` takes the states an **8-doubling monoculture batch** visits
below depth 0.5 -- the only construction that reaches them without a community,
and disjoint from the M5 benchmark, so nothing trains on what it scores. 567 media,
`cfs generate --media label_pool_depl.npz --round 3`, 63/63 shards, both heads
retrained (P14).

| median log-X | n=2 | n=3 | n=5 | n=10 | n=21 | overall | max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 4 doublings, round 2 | 0.002 | 0.002 | 0.000 | 0.026 | 0.013 | 0.002 | 0.318 |
| 4 doublings, **+ round 3** | 0.003 | 0.002 | 0.000 | 0.036 | 0.015 | 0.002 | 0.415 |
| 8 doublings, round 2 | 0.031 | 0.135 | 0.041 | 0.053 | 0.054 | 0.041 | 0.572 |
| 8 doublings, **+ round 3** | 0.044 | 0.107 | 0.038 | 0.078 | 0.061 | 0.043 | 0.576 |

Held-out Head B is unchanged (worst R2 0.9354 -> 0.9346, median 0.9638 -> 0.9649).

1. **Null on the trajectory at both horizons**, 10/30 and 8/30 cells better --
   and by depth it is mildly *worse* in every band, the depleted ones it was built
   for included. The true trajectories are identical between arms, so the depth
   distribution is too and the bands compare directly: median log-X 0.0101 ->
   0.0120 (0.5-0.9), **0.0304 -> 0.0348** (0.1-0.5), 0.0160 -> 0.0269 (0.01-0.1),
   0.0034 -> 0.0141 (dead). A round is not free: it moves `x_scale`, so the whole
   input coordinate shifts for labels that bought nothing.
2. **And a fifth instance of "a strictly better rhs is not a better trajectory":**
   `dc_rel` improves on **21 of 30** cells at 8 doublings (median 0.556 -> 0.504)
   while the endpoint does not follow.
3. **The reach is the reason, and it is measured** (`depl_reach.py`, no solves).
   Median NN distance in `x` from the benchmark's own deep community states to
   the round-3 pool is **5.57**, against **5.22** to the existing training set --
   the new labels are *no closer*, and slightly farther. **A monoculture's
   depletion path does not reach a community's**, which is §8.6d's "one
   community's forward path does not reach another's" one regime deeper.
4. **The scale of the gap is the real finding.** Those NN distances are 4.1 (n=2)
   to 7.7 (n=21), where §8.6d's held-out median is **0.10** and its worst cells
   sit at 1.33-2.63. At 8 doublings the benchmark is 40-70x farther from the design
   than anything measured before, so a 567-medium round was never going to close
   it. Covering that regime needs community trajectories at depth -- the 16 n=15
   sets re-run at `--doublings 8` and labelled, a far larger spend -- or the deep
   gate accepted as an extrapolation test rather than a coverage one.

**Ranked list status: B1 built and null, B2 refuted, B3 re-diagnosed and its
coverage form measured and null, B4 done and the most informative of the four,
B5 subsumed, B6 untried.** What is left for Head B is either the large label spend
in 4 above, or B6 (the piecewise-affine / active-set head) -- and B6 is the only
remaining candidate that could improve *extrapolation* rather than coverage, which
is what the deep gate is actually testing.

##### B6 is refuted before it was built — the active set is not the missing variable

mpLP says `c -> z` is affine on critical regions, and complementary slackness makes
the region label free (§8.6f: `dual => flux on the bound` at 0.996-1.000, and Head A
predicts that gradient at cosine 0.95). B6 would condition Head B on it. Two
premise checks, both cheap, both negative.

**(a) The active set at the failing states is not novel.** Head A's own predicted
limiting set at the 208 deep (`depth < 0.5`) member-states of the 8-doubling
benchmark, against every training row's limiting set (`deep_active_set.py`, no
solves): mean `|A_pred|` = **1.1**, and the Hamming distance to the nearest
training active set has **median 0.0 and p90 0.0 at every community size**. Every
deep state's limiting set is one the design already produced. On held-out design
media the same is true (mean Hamming 0.1-0.3) and the novelty-vs-error correlation
is weak -- Spearman **+0.364** median, +0.136 to +0.564
(`active_set_novelty.py`), nowhere near the +0.673 that earned §8.6d its relabel.
**The failure is extrapolation *inside* a known region** -- the medium is 4-8 away
in `x` (§8.6f, round 3) with the same limiting metabolite -- **not a question of
which region it is in.**

**(b) A limiting set is not a critical region, so the affine structure is not
resolvable from these labels.** Inside each organism's commonest (train ∩ val)
limiting-set bucket, a standardised ridge affine fit of `z` (`affine_coord.py`):

| | in-sample R2 | held-out R2 |
| --- | --- | --- |
| affine in `u` | **0.921** | 0.589 |
| affine in `x` | 0.867 | 0.497 |

A genuine critical region would fit *exactly*; 0.92 in-sample says the bucket is
not one. And the existing MLP scores median held-out R2 **0.964** on all rows, so
a per-bucket affine model is far worse than what is already there. `u` beats `x`
on both axes and `x` throws wild negatives (-44, -7911) where `u` does not, so
there is a mild coordinate signal -- but nothing like §7.2's decisive one for
Head A, and it does not motivate a rebuild.

**Why (b) is a limit of the labels, not of the theory.** A critical region is
determined by the LP's full optimal basis, internal reactions included; the shards
record only the exchange duals. Testing mpLP structure properly means storing the
basis in `groundtruth.solve` and relabelling the roster -- a large spend with, as
of (a), no evidence behind it. **That is what would reopen B6.**

Incidentally measured: the biggest *training* active-set buckets have **zero**
held-out rows, because rounds 1-3 go to train only and occupy patterns the round-0
design never produces. P24 again, in a new instrument.

**The ranked list is now closed.** B1 built and null (twice), B2 refuted from the
labels, B3 re-diagnosed and its coverage form null, B4 done and the most
informative, B5 subsumed, B6 refuted on both legs. Head B's residual is
**extrapolation at states 40-70x farther from the design than anything the
held-out protocol contains**, and nothing on this list addresses that. The honest
options are: label the regime properly (the 16 n=15 sets at `--doublings 8`, a
large spend), state M5 at a horizon the design actually covers, or accept the deep
gate as an extrapolation benchmark and report it as such.

##### Round 4: the n=15 sets at 8 doublings, and the first coverage pool that reaches

The large label spend, run. The 16 n=15 communities (disjoint from the M5
benchmark) re-integrated at `--doublings 8`, 2 medium draws, ~65 min of LP truth.
Two things had to be learned before a single medium could be chosen.

**1. A 15-member community barely depletes, even at 8 doublings.** By
`depth_gate.py`, **94.8%** of its 48 000 member-steps sit above 0.9 of starting
growth, 1.2% in 0.1-0.5 and **0.1%** below 0.1 -- against the benchmark's 73.5% /
5.9% / 1.0% at the same horizon. The medium is drawn over the *union* of 15 active
subspaces, so it is rich, and the horizon ends on the fastest member's clock while
the slow ones are still growing. **Depletion is not something a large community
does; it is something a small one does** -- the same confound §8.6b found for
scarcity, one clock later. It also explains the deep gate's own shape: sizes 10
and 21 never go below depth 0.1 there either, so their 5-6% log-X error accrues
from many steps in the 0.5-0.9 and 0.1-0.5 bands, not from a starved tail.

Consequently the selection rule had to move from "the median member is below 0.5"
(**48** states over 32 trajectories) to "**any** member has left its starting rate"
(`max_depth 0.99`, stride 2, **598** states) -- `make_depl_pool.py` now selects on
the *slowest* member, since the state is hard for the member that is starving.

**2. It reaches, where round 3 did not.** Median NN distance in `x` from the
benchmark's own deep states (`depth < 0.9`) to each candidate pool
(`depl_reach.py`, no solves):

| size | to the training set | to the **n=15 deep pool** | to round 3's monoculture pool |
| --- | --- | --- | --- |
| 2 | 4.14 | 6.23 | — |
| 3 | 4.77 | 5.95 | — |
| 5 | 6.10 | **4.43** | — |
| 10 | 7.01 | **3.54** | 6.95 |
| 21 | 7.78 | **3.07** | 7.50 |
| all | 6.28 | **3.87** | 6.50 |

The new pool **halves** the distance at n >= 5 and is farther only for n=2/3,
whose media are much leaner than a 15-member draw. Round 3's monoculture pool was
no closer anywhere (6.50 against 6.28), which is exactly why it was null. This is
the §8.6d proxy -- the one predictor of eight to clear P25's bar -- used as a gate
*before* the spend rather than as a post-mortem after it.

##### The LP-fallback / self-labelling hybrid, and the trigger it needs

The combination worth building is a fallback that pays for itself: at each dFBA
step, score each member with a **runtime** trigger (no LP); above threshold, call
`solve()` for that member at that state, use the true `z` for the step **and keep
the row as a label**; retrain periodically. It is SDDP's forward pass with a
stopping rule -- the framing already adopted for Head A's cut selection -- and it
is explicitly **not** the Stage-4 active-learning failure (§8.5, Settles), whose
defect was an acquisition pool drawn from the training distribution. Here the pool
*is* the shifted distribution, because the trajectory chooses it.

Two numbers decide it and both are measurable on trajectories already on disk
(`fallback_roc.py`, no solves): the **fire rate** (cost) and the share of the
accumulated `|d log X|` landing on fired steps (benefit).

**The §8.6d reach proxy does not work as the trigger.** It is a *per-cell*
predictor -- median NN distance over a whole path, Spearman +0.673 across 30 cells
-- and it does not transfer to a per-step decision. At 8 doublings every community
step is far (distances 3-8 throughout), so any threshold below 4 fires on 100% of
steps, and the thresholds that do discriminate are **anti**-correlated with error:

| NN threshold | fire rate | error captured | lift |
| --- | --- | --- | --- |
| <= 3.0 | 1.000 | 1.000 | 1.0x |
| 4.0 | 0.915 | 0.823 | **0.9x** |
| 6.0 | 0.654 | 0.420 | **0.6x** |

Worse than random. The distance is set by where the *medium* is, which is far at
every step; the error accrues where the *member* is depleting, and a depleting
member can be closer to the design's low-`mu` media, not farther. **A proxy
validated across cells is not thereby a proxy within one.**

**Predicted depletion depth is the trigger.** `mu_hat(t) / mu_hat(0)` per member,
surrogate-only, free -- and B4 measured the error rising monotonically as it falls:

| depth < | fire rate | error captured | lift | (4 doublings) |
| --- | --- | --- | --- | --- |
| 0.99 | 0.447 | 0.807 | 1.8x | 0.329 / 0.412 |
| 0.95 | 0.309 | 0.769 | 2.5x | 0.191 / 0.354 |
| **0.90** | **0.239** | **0.719** | **3.0x** | 0.122 / 0.289 |
| 0.75 | 0.163 | 0.602 | 3.7x | 0.041 / 0.124 |
| 0.50 | 0.100 | 0.283 | 2.8x | 0.016 / 0.052 |

At `depth < 0.9` the fallback solves **24%** of member-steps -- a 4x saving against
the full LP before any retraining -- and those steps carry **72%** of the error.
The knee is 0.75-0.9; below 0.5 the rate falls faster than the error, because by
then the culture is dying and both trajectories have stopped growing (B4 point 3).

**Four things to get right, all of them already measured traps.**

1. **Pin `x_scale`.** A round moves it (P14) and every relabel silently changes the
   input coordinate, so incremental rounds are not comparable and both heads must
   be rebuilt. Freeze it at round 0 and pass it in; this is a change to
   `data._stack`, and it is the prerequisite for any online loop.
2. **Expect a discrete endpoint flip, not a smooth interpolation.** Substituting
   truth at the fired steps changes *which metabolite empties first*, and that is
   the mechanism behind five separate "a better rhs is not a better trajectory"
   results. The hybrid's error will not interpolate between surrogate and truth as
   the threshold moves; measure it, do not assume it.
3. **No validity guarantee, so store and re-run.** Head A's cuts are valid by
   construction and SDDP's convergence follows; Head B has no such property, so
   the loop can oscillate as retraining moves the visited set. Keep every label
   (Guigues's store-and-select), and re-run the trajectory after each retrain
   rather than trusting the old one.
4. **Score the fallback on the endpoint, not on `dc`.** The two disagree, five
   times over.

##### Depletion states can be explored pre-emptively, and generating them needs no LP

The fallback above spends LP solves online. Most of that is avoidable, because
**generating** a depletion state costs nothing -- only labelling one does. A
depleted medium is `c0` minus a conical combination of the members' consumption
vectors, and a surrogate that is *wrong* about those vectors can still land in the
right region: a generator does not have to be accurate, only to be in-distribution
for the target.

**Measured, and it holds exactly.** `cfs community` stores `c_surr` beside
`c_true`, so the two pools can be compared for free. Building the round-4 pool from
the **surrogate's own** n=15 paths instead of the LP truth's (`make_depl_pool.py
... surr <runs>`), then scoring reach against the benchmark's deep states:

| size | pool from LP-truth paths | pool from surrogate paths |
| --- | --- | --- |
| 5 | 4.427 | 4.428 |
| 10 | 3.540 | 3.532 |
| 21 | 3.068 | **3.058** |
| all | 3.872 | **3.808** |

Identical to three decimals. **The 48 000 LP solves behind the n=15 truth runs were
needed only to score them, not to build the pool** -- `cfs simulate` (§13.1,
surrogate-only, no LP) generates the same states for free.

**And the pool is ~24x redundant.** Median NN distance *within* round 4's 598
media is **0.006**, against 3.9 to the states they are meant to cover; a
farthest-point cover at radius 1.0 needs **25 of the 598**. Consecutive
integration steps are near-duplicates -- the same finding as the Level 1 trial
pool, where 5x subsampling selected the identical cuts.

**So the label budget should be spent the other way round.** Generate massively and
for free (thousands of surrogate-only communities, `cfs simulate`), filter by
predicted depth, **farthest-point subsample to a coverage radius**, and only then
label. At 25 points per composition, round 4's 598 solves would have covered ~24
different community compositions instead of two, which is exactly the axis §8.6d
found to be the limit ("one community's forward path does not reach another's").
The online fallback then handles what pre-emption misses, and its fire rate is the
measurement that says whether pre-emption worked.

**Cold start.** A genome with no head yet cannot run this. The analytic version
needs no integrator and no head: step `c <- c0 - lambda * sum_i X_i z_i` using the
organisms' *label* `z` vectors, over a ladder of `lambda` and random member
subsets, clipped at zero -- a one-step Euler cone through the same region. That
preserves §4.7's property that a new GEM anchors itself, and it belongs in
`design.sample_media` as a depletion stratum rather than as a separate pool.

##### Round 4's verdict: the coverage chain is real, and it stops at the endpoint

| median log-X | n=2 | n=3 | n=5 | n=10 | n=21 | overall | max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 4 doublings, round 2 | 0.002 | 0.002 | 0.000 | 0.026 | 0.013 | 0.002 | 0.318 |
| 4 doublings, **+ round 4** | 0.004 | 0.002 | 0.000 | **0.018** | 0.015 | 0.003 | 0.318 |
| 8 doublings, round 2 | 0.031 | 0.135 | 0.041 | 0.053 | 0.054 | 0.041 | 0.572 |
| 8 doublings, **+ round 4** | 0.049 | 0.137 | 0.039 | **0.027** | 0.063 | 0.042 | 0.699 |

Head B held out: worst R2 0.9354 -> 0.9294, median 0.9638 -> 0.9641 -- unchanged,
as every round has been (P24).

1. **The chain `reach -> dc_rel` is confirmed, and this is the first time it was
   predicted before the spend rather than fitted after it.** `dc_rel` improves on
   **21 of 30** cells (median 0.556 -> 0.482 at 8 doublings) and per size it tracks
   exactly where reach improved: n=5 0.362 -> 0.284, n=10 2.816 -> 2.484, n=21
   2.771 -> 2.481. The sizes whose reach got *worse* (n=2, n=3 -- a 15-member draw
   is much richer than a 2-member one) are the sizes whose `dc_rel` did not move.
   §8.6d's proxy has now earned its keep twice.
2. **And it stops there.** log-X is better on 12 of 30 and its median is flat
   (0.0411 -> 0.0417). n=10 halves (0.053 -> 0.027); n=21 *worsens* (0.054 ->
   0.063) on the largest reach gain in the set. **Sixth instance of "a strictly
   better rhs is not a better trajectory", and the decisive one**: the mechanism
   was stated in advance, the money was spent, the predicted metric moved, and the
   endpoint did not.
3. **So Head B's accuracy is no longer the binding constraint on the batch
   endpoint.** Two more coverage rounds would buy more `dc_rel` and, on this
   evidence, no log-X. What is left is the endpoint's own sensitivity: `d(log X)`
   integrates `mu`, but *when* the culture stops turns on **which metabolite
   empties first**, a discrete outcome that no norm on `dc` can see and no
   improvement in `dc` reliably fixes.

**What this redirects the work to**, in order:

- **Report what the heads are good at.** §13.7 already says "rates only; prefer
  §13.4" for quantitative yield. That is now measured rather than cautious:
  structure, ordering, cross-feeding recall and `dc` are all strong and improving;
  the batch endpoint is not, and is not the metric to tune on.
- **Trajectory-level training** (§8.6f solution 3) is the only remaining idea that
  optimises what the endpoint measures -- backprop the endpoint through the
  integrator instead of fitting `z` per state. Everything else on the list
  optimises `dc`, and `dc` has now been shown six times not to carry.
- **The LP fallback** at `depth < 0.9` (24% of member-steps, 72% of the error),
  which sidesteps the question by substituting truth exactly where the discrete
  flips happen.
- And if a coverage round *is* run, run it the cheap way: generate free, subsample
  by farthest point, label ~25 per composition across many compositions.

#### §8.6g — Head B, stock-take (2026-09-03)

Head B is accurate on its own distribution -- held-out worst R2 **0.929-0.935**,
median 0.964, worst flux cosine 0.983, sign agreement 0.961 -- and every §8.1
failure is somewhere else. Five distinct problems, in the order they bind.

**1. It is an extrapolation failure, not a fit failure.** The residual lives at
states whose NN distance in `x` is **4-8**, where held-out media sit at **0.10**.
Everything aimed at the model was null: the low-rank basis (B1, twice, including
on the sensitive gate), the loss reweighting (B2, refuted from the labels), the MM
hinge (`--w-mm`), and the active-set conditioning (B6, Hamming 0.0 at the failing
states). The only thing that ever moved the metric it was aimed at is coverage.

**2. There is no structure to degrade into.** Head A is concave, monotone and
one-sided *by construction*, so off-distribution it fails predictably and
`--gm-repair` restores the invariant in closed form. Head B has exactly one
constraint available -- §3.3's uptake bound -- and it binds on the wrong side:
**48-69% of the squared error is on secretion**, positive and unbounded above. The
subspace constraint that does exist is worth at most the violation it removes
(0.9-8.7% of the norm against a 12-26% error).

**3. The specific-flux floor is self-consistent and not learned.** `z = z_hat *
max(mu_hat, floor)` recovers `z` exactly at any depth *if* the head emits the
floored target; below depth 0.1 it emits the unfloored one, giving **3318x**
magnitude error and p05 cosine **−0.44**. Removing the floor makes the derivative
2.7x better and the trajectory 30x worse, so the error is load-bearing.

**4. The metric is not the objective.** Six instances of "a strictly better rhs is
not a better trajectory", the last of them predicted in advance and confirmed on
its own metric. A batch endpoint turns on **which metabolite empties first**.

**5. The benchmark was hiding all of it.** M5 passes at 4 doublings and fails at
every size at 8; 86% of the shallow gate sits above 0.9 of starting growth; and
depletion is confounded with size, since a 15-member community barely depletes at
all (94.8% above 0.9 even at 8 doublings).

##### What looks promising, ranked

**All four were built and measured, 2026-09-03/04. Verdicts first, detail below.**

| item | verdict |
| --- | --- |
| 1. runtime predictors | **shipped**, and item 4's trigger came out of it |
| 2. secretion bound `E z <= 0` | **kept**, small and free: 14/30 cells better, never worse |
| 3. trajectory-level training | **refuted** by its own zero-weight control |
| 4. LP fallback at `depth < 0.9` | **works**: 8-doubling 0.041 -> 0.028, n=21 -> 0.015, at a 24.6% fire rate. Self-labelling half null at 71 media |

Only item 4 moves the gate, and it moves it **without changing either head** —
which is consistent with everything else in this section: Head B's residual is
extrapolation, every model-side arm is refuted, and the useful responses are to
*detect* the regime (item 1) and *pay for truth* in it (item 4).

Three methodological results came out of this pass and generalise beyond it:

* **Enforcing a constraint is not projecting onto it.** A uniform secretion shrink
  satisfies the same four inequalities and made `dc_rel` worse on 10 of the 11
  cells it moved; only the min-norm projection inherits "a set containing the truth
  cannot increase the error".
* **Run the zero-weight ablation.** The joint label+trajectory loss beat its
  starting head on every metric until `--w-traj 0` reproduced the whole gain.
* **Score a label round against a matched retrain.** A fresh fit at the same seed,
  with no new rows, moves the 8-doubling mean 0.085 -> 0.143 and the max 0.572 ->
  1.866 — larger than any round-sized effect measured here. Rounds 3 and 4 were
  scored against their starting checkpoint instead, and carry that caveat.

1. **Ship the runtime predictors — done 2026-09-03.** `cfs simulate`'s report now
   carries `reach` (per member, at `t = 0`), `depth_final` and
   `frac_steps_below_depth_0.9`, all surrogate-only and all free. The reach proxy
   needs the training media at runtime, which only the checkpoint can supply, so
   `behaviour.save` writes **`reference_x.npz`** — 512 strided training rows per
   organism in `x`, 1.3 MB compressed. A checkpoint without it reports
   `reach: null`; `20hm_bands/ref_posthoc.py` adds one to an existing checkpoint
   with no refit. The subsample reads **5-11% above** the exact NN distance
   (2.61 vs 2.475, 1.63 vs 1.469 on benchmark cell 0), so read it against the
   thresholds — held-out media ~0.10, Head B's failures at 4-8 — and not against
   `nn_proxy.py` to three decimals. The original entry:

   Per member per step, free: the reach proxy
   (NN distance in `x`) and predicted depletion depth `mu_hat(t)/mu_hat(0)`. The
   first is a *per-cell* accuracy predictor (Spearman +0.673 on `dc_rel`) and
   **must not** be used per step (measured: lift 0.9x, worse than random); the
   second is the per-step one (lift 3.0x). Together they make the output honest
   and give §13.6 its missing nonconformity score.
2. **A secretion-side bound — premise check passed, 2026-09-03, and it is the
   first constraint to do so.** `20hm_bands/element_bound.py` (no LP solves; the
   elemental matrix is read off the GEMs' formulas, 444/444 exchanges covered)
   measures `E z <= 0` for C, N, P, S. Rows whose gross uptake of an element is
   dust are dropped — they have no bound to violate and divide by ~0, which is
   where a spurious 1e7 came from on the first pass.

   | violation rate | C | N | P | S | excess / element turnover (C) |
   | --- | --- | --- | --- | --- | --- |
   | **labels** (the LP's own `z`, 21k rows) | **0.0000** | 0.0000 | 0.0000 | 0.0000 | 0.000 |
   | Head B, 4200 held-out label media | 0.391 | 0.371 | 0.344 | 0.429 | 0.085 (p95 0.181) |
   | Head B, 936 trajectory member-states | 0.209 | 0.099 | 0.169 | 0.125 | 0.056 |
   | **Head B, the 198 states in cells with log-X > 0.05** | **0.532** | 0.180 | 0.279 | 0.398 | **0.110** (p95 0.356) |

   Both legs pass. The labels satisfy it **exactly**, so it is provable and not an
   approximation; and the excess is **8-11% of that element's turnover** against
   Head B's own 12-26% relative error, where B1's off-manifold component was
   0.9-8.7%. On the failing cells the median net carbon flux goes *positive*
   (+0.022 of gross uptake, p95 +1.11) — half those states secrete more carbon
   than they took up.

   **Unlike `--w-mm`, it is reachable in-distribution.** The MM hinge failed
   because the violation was 0.053% on training rows and the loss had nothing to
   grip; here 34-43% of *held-out label media* violate. So both mechanisms are
   available — a hinge in training and a DC3-style correction at inference — and
   this project's own measurement says to build the **correction** first (§8.6d's
   clamp bought the composition; §8.6f's projection bought 0.001). The trap to
   respect is [[better-rhs-is-not-a-better-trajectory]]: score it on the endpoint
   over 3 draws x 10 communities, not on `dc_rel`.

   **Built and kept, on by default (`dfba.Surrogate._element_balance`).** It is
   the weighted **minimum-norm projection** onto `E z <= 0`, in the head's own
   `z_scale` metric; four constraints, so the dual is a 4-D non-negative least
   squares whose active set is found by enumerating the 15 non-empty subsets. Same
   10 communities x 3 draws, only `mu_and_z` differing:

   | paired over 30 cells | better | worse | same | median | max |
   | --- | --- | --- | --- | --- | --- |
   | log-X endpoint | **14** | 4 | 12 | 0.0023 -> **0.0019** | 0.318 -> 0.318 |
   | `dc_rel` | 7 | 4 | 19 | 0.1898 -> 0.1896 | unchanged |
   | `dc_cos` | 8 | 2 | 20 | — | — |
   | `mu_rel`, cross-feeding | 0 | 0 | 30 | bit-identical | — |

   Modest and free: size medians do not move (0.002/0.002/0.000/0.026/0.013), the
   paired endpoint median falls 18%, and several small cells halve. The two bad
   cells are untouched, as they should be — they are §8.5's mid-`mu`
   over-prediction, which this bound says nothing about.

   **Three things this cost, worth not repeating.**

   1. **Enforcing a constraint is not projecting onto it.** The first build was a
      uniform shrink of the secretions — one scalar per organism, provably
      satisfying all four inequalities, ten lines. It left the endpoint unchanged
      (9 cells better, 4 worse) and made `dc_rel` **worse on 10 of the 11 cells
      that moved**, because it also shrinks the fluxes that were not implicated.
      Only a projection inherits §3.3's guarantee that a set containing the truth
      cannot increase the error.
   2. **Measure the violation in the norm the consumer uses.** 8-11% of *element
      turnover* sounds large; the same violation is a median **2.8% of `||z||`**
      where it fires and 0 over all states, which is the size of effect the
      composition then shows. `20hm_bands/proj_size.py` measures it directly, and
      is the honest premise number for any future projection.
   3. **A sign error in a feasibility test reads exactly like a null result.** The
      dual's primal-feasibility check was inverted, so the projection fired on
      **1.9%** of member-states against a measured ~30% violation rate, and the
      first composition run came back null. The diagnostic that caught it is the
      fire rate against the independently measured violation rate — always compare
      those two before believing a projection did nothing. The regression test
      checks the result against a brute-force `scipy.optimize.minimize`
      projection with **two** elements binding: a one-element case passes with the
      sign either way, because the identity is feasible for every subset.

   The original entry: **the last untested structural constraint, and it sits
   on the 48-69%.** You cannot secrete more carbon, nitrogen or electrons than you
   took up: `E z <= 0` element-wise against the biomass drain, a *provable*
   one-sided inequality on the unbounded side, the analogue of the MM bound on the
   uptake side. **It is not subsumed by B1**, which removed zero-variance
   directions, not inequality faces. Mandatory premise check first, per
   [[constraint-worth-at-most-the-violation]]: measure the violation on the
   labels (should be ~0) and on the failing predictions. If the violation is
   small, it is dead like B1.
3. **Trajectory-level training — premise check passed, 2026-09-03, and it is the
   strongest positive signal Head B work has had.** `20hm_bands/traj_sens.py` and
   `traj_sens2.py` perturb one member's `z` and re-integrate the surrogate from
   the run's own initial state (no LP solves), over 20 cells:

   * **Smooth and monotone on 20/20 cells.** The feared threshold — the endpoint
     turns on which metabolite empties first, so gradients might be blocked — does
     not appear in the response. `integrate` clips the pool at zero, and that
     clipping is a subgradient, not a wall.
   * **The gradient is ample.** Estimating `||g||` by random probing
     (`E|g.u| = ||g|| sqrt(2/pi/d)`), **17 of 20 cells could close their entire
     endpoint error with a <= 10% relative move in `z`**, and the linearisation is
     real: `||g||` measured at eps 0.01 and 0.03 agrees within 5% on 15 cells and
     within 1.6x on the rest — nonlinear where the sensitivity is huge, never a
     staircase.
   * **The endpoint is ~3 orders more sensitive to *direction* than to
     magnitude.** Uniform scaling of a member's `z` gives `||g||` 0.067 where the
     full-space estimate is 272 (n=10, draw 200). That is the mechanism behind six
     instances of "a better rhs is not a better trajectory": a per-state loss
     weighted by `z_scale` spends itself on magnitude, and the endpoint does not
     care about magnitude.
   * **It cannot reach the worst cell, and should not be expected to.** n=21 draw
     200 (error 0.318) has `||g||` 0.198, an order of magnitude short — consistent
     with its own diagnosis, Head A's mid-`mu` over-prediction (`mu_rel` 0.0388),
     not Head B's.

   **Design constraint the check hands over:** `||g||` reaches ~300 through 40
   Euler steps, so the training gradients will be stiff. Clip, and prefer a loss
   over the whole trajectory to an endpoint-only one.

   **Built (`cfs train-traj`, `src/cfs/surrogate/traj.py`) and measured, and it
   does what it says on its own objective without being a net win at the gate.**
   The JAX replica of the dFBA map reproduces `dfba.integrate` to **3e-6** in
   `log X` (`20hm_bands/traj_check.py`), so training and scoring are the same map.

   **(a) The community loss cannot identify per-organism behaviour.** `d(log X)/dt`
   is Head A's frozen `mu`, a function of `c` alone, so Head B reaches the loss
   *only* through the pool sum `sum_i X_i z_i`: 15 members' fluxes collapse into
   one vector per step. Measured on the 16 n=15 communities: 60 epochs buy 9% of
   the trajectory loss and take held-out label R2 from **0.577 to -31.98** (median
   0.937 -> -0.26), with both gates worse (4 doublings 0.002 -> 0.014 overall,
   8 doublings 0.041 -> 0.104). A per-leaf weight-drift anchor (`--w-anchor`) does
   not fix it -- at `w = 1.0`, where learning is nearly off (-2.3% loss), the worst
   organism is still at **-15.8**. An anchor cannot repair an identifiability
   problem.

   **(b) Monocultures identify it, and the loss then actually moves.** In a
   monoculture the pool sum *is* that organism. On `monodeep_s*` (21 organisms x 3
   draws, 8 doublings -- the depletion regime, and already on disk) the trajectory
   loss falls **0.0180 -> 0.0053, a 3.4x reduction** against 9% on community data,
   and the label damage is far milder: worst R2 -0.445, median 0.817, worst cosine
   0.911, sign agreement 0.918 (*better* than the 0.912 it started from).

   **(c) At the gate it is mixed, and splits cleanly by size.** Paired over 30
   cells, `behaviour_tjm_lr1e-5` against the same-medium controls:

   | median log-X | n=2 | n=3 | n=5 | n=10 | n=21 | better/30 |
   | --- | --- | --- | --- | --- | --- | --- |
   | 4 doublings, base -> tuned | 0.0020 -> 0.0050 | 0.0025 -> 0.0327 | 0.0002 -> 0.0006 | 0.0259 -> **0.0200** | 0.0129 -> **0.0094** | 10 |
   | 8 doublings, base -> tuned | 0.031 -> 0.034 | 0.135 -> **0.095** | 0.041 -> **0.037** | 0.053 -> **0.032** | 0.054 -> **0.039** | 13 |

   At the deep gate every size from 5 up improves by 10-40%; the small communities
   lose, and they dominate the count.

   **(d) The inverse of six earlier results, and the sharpest evidence yet that
   `dc_rel` is the wrong instrument.** `dc_rel` gets *much worse* — median
   0.190 -> 0.626 at 4 doublings, 0.556 -> 1.586 at 8, better on only 2-3 of 30 —
   while the endpoint at n >= 5 improves. Six times a better rhs failed to buy the
   endpoint; here a **3x worse rhs bought it** at the sizes that matter. The two
   are close to independent, and only one of them is the gate.

   **(e) One trap, and it is not specific to this module.** A single non-finite
   gradient is permanent: `optax.clip_by_global_norm` puts the NaN in the global
   norm and Adam's moments carry it forever, so every later cell reads non-finite
   and the run looks like a learning-rate divergence at any `lr`. It arises on
   starved states, where Head A's softmin at the shipped `gm_eval_temp` (1e-4)
   amplifies by ~1/T in float32. `traj.run` now skips the update on a non-finite
   *gradient*, not only a non-finite loss, and reports the skip count (12-13 of 63
   monoculture cells).

   **(f) The joint loss was built, controlled, and the trajectory term is null.**
   `--w-label` adds the per-state label loss on a random minibatch beside the
   trajectory term; `--w-traj 0` is the control that runs the label term alone
   through the identical optimiser, steps and data. On monocultures at
   `--w-label 10`:

   | | worst R2 | median R2 | worst cos | sign | 4-dbl overall | 8-dbl overall |
   | --- | --- | --- | --- | --- | --- | --- |
   | `behaviour_p4r2` (start) | 0.577 | 0.937 | 0.964 | 0.912 | **0.002** | **0.041** |
   | joint (label + trajectory) | 0.784 | 0.958 | 0.982 | 0.955 | 0.006 | 0.082 |
   | **control (label only)** | **0.816** | **0.959** | 0.982 | 0.953 | 0.007 | 0.070 |

   Joint and control are indistinguishable: paired over 30 cells the joint head is
   better on **18/30** at 4 doublings and **16/30** at 8 -- a coin flip -- with a
   median relative difference of 5-9%. And the control alone drives the *trajectory*
   loss down 0.0854 -> 0.0233 (3.7x), which is more than the trajectory-only arm's
   own 3.4x. **Everything attributed to the trajectory term is the label term.**
   §8.6g(3) is refuted; `cfs train-traj` stays in the tree, off any default path,
   with this result on file.

   **(g) A tenth instance of "held-out cannot see it", and the sharpest.** The
   control improves **every** held-out label metric over the head it started from
   -- worst R2 0.577 -> 0.816, median 0.937 -> 0.959, worst cosine 0.964 -> 0.982,
   sign 0.912 -> 0.953 -- and makes the composition **worse at both gates** (0.002
   -> 0.007, 0.041 -> 0.070). More label training, better label scores, worse
   composition. Whatever selects a Head B for §8.1, it is not the held-out label
   fit.

   The original entry: Backprop the endpoint through the integrator
   instead of fitting `z` per state -- the only idea that optimises what the gate
   measures, and the only one that can see which metabolite empties first. The
   stack has the pieces (JAX; `integrate` is an explicit Euler map).
4. **The LP fallback at `depth < 0.9` — built and it works, 2026-09-04.**
   `cfs community --fallback-depth 0.9` (`dfba.rhs_hybrid`): at each step, any
   member whose predicted depletion depth `mu_hat(t)/mu_hat(0)` is below the
   threshold gets its true LP solved and substituted for that step. The trigger is
   surrogate-only and free; `mu_hat(0)` is taken at the first call.

   | median log-X, 3 draws x 10 communities | n=2 | n=3 | n=5 | n=10 | n=21 | overall | better/30 |
   | --- | --- | --- | --- | --- | --- | --- | --- |
   | 4 doublings, base | 0.002 | 0.002 | 0.000 | 0.026 | 0.013 | 0.002 | — |
   | 4 doublings, fallback | 0.001 | 0.002 | 0.000 | 0.023 | 0.011 | **0.001** | **19** (4 worse) |
   | 8 doublings, base | 0.031 | 0.135 | 0.041 | 0.053 | 0.054 | 0.041 | — |
   | **8 doublings, fallback** | 0.029 | 0.127 | **0.027** | **0.029** | **0.015** | **0.028** | **26** (3 worse) |

   1. **It buys the deep gate, which nothing else has**: overall -32%, and the
      large communities most — n=21 **0.054 -> 0.015 (-72%)**, n=10 -45%. The
      4-doubling gate improves slightly (0.002 -> 0.001) because there is little
      depletion there to trigger on.
   2. **The offline ROC was accurate.** `fallback_roc.py` predicted a 24% fire rate
      at `depth < 0.9`; live, the pooled rate is **24.6%** (3835/15600 member-steps)
      at 8 doublings and 12.3% at 4. So it is a ~4x saving against solving every
      member every step, and the offline estimator can be trusted to price a
      threshold before running it.
   3. **The benefit tracks the trigger.** Over the 27 deep cells that fired,
      Spearman(fire rate, relative endpoint gain) = **+0.458** (p=0.016) — the
      cells that fire more are the cells that gain more, which is what a trigger
      aimed at the error should do.
   4. **Quote the pooled rate, not the per-cell median.** At 4 doublings the
      per-cell median fire rate is 1.8% against a pooled 12.3%: most cells never
      deplete and a few fire on 83% of their steps. The median describes a typical
      cell, not the cost.

   **The self-labelling half is built too, and its first pass is null — but the
   control is the result worth keeping.** Trap 1 is cleared:
   `load_{value,behaviour}_dataset(..., x_scale=...)` and
   `cfs train-{value,behaviour} --x-scale-from <checkpoint>` pin the input
   coordinate instead of recomputing it from the rows, so a round can extend an
   existing head **without rebuilding Head A** — `Surrogate`'s P14 check passes
   across the round, which is what it is for. Pinning costs nothing: retraining on
   the *unchanged* labels with the coordinate pinned gives worst R2 0.933 / median
   0.964 against `behaviour_p4r2`'s 0.935 / 0.964.

   The loop then runs end to end: `--fallback-media` writes the fired states in the
   layout `cfs generate --media` reads (no second label writer, and no need to
   solve every alpha online), 2 draws x 16 n=15 communities at 8 doublings gave
   **71 media** after dedup and striding, `--round 5` labelled them 21/21, and Head
   B was retrained with the pin.

   | median log-X | n=2 | n=3 | n=5 | n=10 | n=21 | overall | mean | max |
   | --- | --- | --- | --- | --- | --- | --- | --- | --- |
   | 8 dbl, `behaviour_p4r2` | 0.031 | 0.135 | 0.041 | 0.053 | 0.054 | 0.041 | 0.085 | 0.572 |
   | 8 dbl, **control** (pinned retrain, no round 5) | 0.026 | 0.054 | 0.031 | 0.057 | 0.059 | 0.035 | **0.143** | **1.866** |
   | 8 dbl, + round 5 | 0.032 | 0.201 | 0.031 | 0.057 | 0.068 | 0.040 | 0.126 | 0.573 |

   1. **The round is null against its own control** — better on 9/30 cells at 8
      doublings and 11/30 at 4, with median and mean disagreeing in both (the round
      fixes the control's tail and loses the typical cell). 71 media against a
      ~4700-media root is a small spend, so this refutes nothing bigger; it says
      one pass at this size does not transfer.
   2. **Retraining alone moves the composition more than the round does, and that
      is the finding.** The control differs from `behaviour_p4r2` only by a fresh
      600-epoch fit at the same seed and settings, and at 8 doublings it moves the
      mean **0.085 -> 0.143** and the max **0.572 -> 1.866**. So the noise floor
      for "did this round help" is larger than any round-sized effect measured
      here. **Score a label round against a matched retrain, never against the
      checkpoint it started from** — rounds 3 and 4 were both scored the older way.
   3. The online fallback remains the thing that works: 0.041 -> 0.028 on the same
      gate, with no retraining at all.
5. **Application-scoped coverage**, run the cheap way -- generate free from
   `c_surr`, farthest-point subsample, label ~25 per composition across many
   compositions. On round 4's evidence this buys `dc` and structure, not the
   endpoint.

##### What the literature says, and where it does not help

* **The framing** is amortized optimization (Amos 2023) and the argmin map of a
  parametric LP (mpLP; Borrelli/Bemporad/Morari). B6 measured that the active set
  does not discriminate the failure, and testing mpLP structure properly needs the
  LP's **optimal basis** stored at label time, which the shards do not record.
  That is the one change that would reopen it.
* **The offline-MBO conservatism literature points the wrong way here.** It exists
  for surrogates being *optimised against*; Head B is being *integrated*. Nothing
  in it addresses keeping an unconstrained estimator honest along a trajectory.
* **The right neighbours are hybrid and fallback methods.** Basis reuse in
  community dFBA (bioRxiv 2020) is the non-ML baseline any speed claim must
  acknowledge and the natural fallback in (4).
* **For the secretion bound**: elemental balancing and conserved moieties (Famili
  & Palsson 2003; Haraldsdottir & Fleming 2016) give the constraint, DC3 gives the
  completion/correction mechanism -- and this project's own measured lesson is
  that DC3's *correction* bought the composition (§8.6d's clamp) while its
  *projection* bought 0.001 (§8.6f's B1).
* **For (3)**: neural ODEs and differentiable simulators. The nearest applied
  precedent, the reactive-transport ANN (Sci Rep 2025), does **not** do it, and its
  own error analysis is about this same near-depletion regime.
* **For §13.6**: conformal prediction with a distance-aware nonconformity score,
  calibrated on community-regime states.

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
| M3 | Head A trained, all 20, vmapped | Gradient cosine > 0.99 held-out — **not met, worst 0.952 (2026-09-02), and no longer what §8.1 waits on.** The best head is *frozen* label tangents (`--epochs 0 --gm-init labels --gm-select level1 --gm-repair --gm-eval-temp 1e-4`): median cosine 0.981, median R² 0.9999, low-`mu` bias +0.0004, and `mu_rel_median <= 5e-4` on all 30 §8.1 cells (§8.6c) |
| M4 | Head B trained, alpha sweep validated | V3 passes — **built 2026-08-28**, worst held-out R² 0.856 / median 0.921; specific-flux target + §3.3 uptake clamp + the §4.3 community-regime round take it to **0.931 / 0.963** on `labels_p4` (§6.3). **Now M5's bottleneck (2026-09-02):** held-out is not the binding number — per-member flux cosine falls to 0.74-0.96 at community media, tracking NN distance to its own training media at Spearman +0.673 against `dc_rel`. `--w-mm`, the hinge on §3.3's bound, is refuted: the violation is off-distribution (§8.6d) |
| M5 | dFBA composition | Trajectory matches COBRApy dFBA to 1% — **met at n=2/3/5 and n=21; n=10 and one cell of 30 open (2026-09-02).** Median log-X over 3 medium draws x 10 communities, frozen Head A at `--gm-eval-temp 1e-4`: **0.003 / 0.004 / 0.000 / 0.025 / 0.009** at sizes 2/3/5/10/21, overall 0.004, max 0.318 (was 0.006/0.007/0.009/0.060/0.175). The residual is Head B's coverage, not Head A (§8.6c/§8.6d). **State this gate over replicates only** — a single run carries ~6x sampling error on a small community, larger than most model changes measured (§8.1) |
| M6 | Newton equilibrium + implicit gradients | V4 passes — **built 2026-09-04** as `cfs steady-state` (§13.4). Active-set Newton on the chemostat fixed point, `lstsq` after row/column equilibration because the Hessian sum is rank ~10-25 of 365. After the finite-difference step was corrected (it was 61x wrong on the limiting metabolite -- §13.4): 11 iterations to a scaled residual of **1.8e-8**, where the broken step took 159 to 9.3e-7. 2 of 5 2-member cells converge; the rest fail on globalisation, not on the model. Roster-wide failure rate not yet stated, and V4 not yet re-measured |
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

#### Per-organism `Vmax` is a required future input — noted 2026-09-06

**The GEM gives the yield, never the rate.** FBA returns `mu = uptake x yield`; the
stoichiometry is genuinely in the model, the uptake bound is imposed. Every
exchange of every roster GEM carries `|lower_bound| = 1000 mmol/gDW/h` -- a
numerical infinity, ~100x physiological -- so `mu_max` comes out at **0.77 to 57.6
/h**, doubling times down to **6 minutes**, and `--dilution-frac 0.2` then sets
`D` anywhere from 3.7 to **276 turnovers/day**. Every absolute timescale in a
dynamic result is inflated by that factor. It does not touch M5, which normalises
by doublings, and it does not touch R* orderings or `k`; it does mean "what happens
in 72 h" is currently not answerable.

The absolute scale is cheap to fix and needs **no retraining**: the LP sees the
bound only as `Vmax * u`, so a uniform rescale of `Vmax` is exactly a rescale of
`u`, the head's own input. Measured on cell 2, `mu` responds smoothly and
essentially linearly in the scarce regime -- a 10x medium change gives 9.9x `mu` --
and reaches 2.5 /h at 1e-4 of the feed and 0.26 /h at 1e-5.

**But a uniform scale is not the interesting knob, and this is the finding.**
Rescaling toward physiological rates moves the community *into* the supply-limited
regime, where members become **more** alike, not less: cell 2's two organisms are
3.6% apart in `mu_max` at full scale and **0.04% apart at 1e-4**. With identical
`Vmax` on every exchange of every genome, competition is decided only by internal
network yield, and in the scarce regime that difference nearly vanishes -- which is
where this session's 0.02-0.15% R* ties come from. Realistic growth rates make the
ties *worse*.

So **per-organism `Vmax` (ideally per organism x metabolite) is a required input
that the models cannot supply**, and it is the one that changes who wins rather
than merely how fast the clock runs. Transporter capacity is a primary axis of
real competition and is currently constant across the roster. Until it is
supplied, read every chemostat result as: orderings, `k`, R* structure and
steady-state accuracy are meaningful; absolute times, `D` in turnovers/day, and
the closeness of competitors are not.

#### The chemostat *transient* — `cfs simulate --stiff`, 2026-09-06

`--dilution` made `simulate` a continuous culture from the start, but on explicit
Euler, and the chemostat transient is **stiff**: the pool equilibrates fast while
biomass grows slowly, because the medium saturates `mu` at ~0.2% of the feed. §13.4
trap 3 records what that does to a warm start; it does the same to a simulation.

`--stiff` integrates with **BDF in `log X`** instead. `log X` keeps abundances
positive without a clip, makes five decades of biomass an O(1) range, and turns
washout into `w` drifting down rather than a root at `X = 0`. Two of the four
Jacobian blocks are exact -- `d(dc/dt)/d(log X_i) = X_i z_i` because `dc` is linear
in `X`, and `d(mu - D)/d(log X) = 0` -- the growth rows come from Head A
analytically, and only `d(dc/dt)/dc` is finite-differenced, through the batched
evaluator.

**Validated against the fixed point, which is the strong form of the test.** Cell
4, its own feed and `D = 2.4477`, 40 h:

| | explicit Euler | **BDF (`--stiff`)** | `cfs steady-state` |
| --- | --- | --- | --- |
| `X`, survivor | 2.356e-02 (**480x**) | **4.891e-05** | 4.891e-05 |
| pool, `norm(dc)/norm(c*)` | 4.2e-01 | **1.3e-10** | — |
| loser's `mu` | −2.4473 | **0.2870** | invasion margin −0.883 |
| wall clock | 7 s | 22 s | — |

The transient and the Newton solve **share no numerics**, so agreeing on the pool
to ten decimal places and on biomass to four significant figures is evidence for
both -- and the extinct member's `mu` reproduces the steady state's invasion
margin exactly.

**Euler's failure is the dangerous kind: it looks converged.** It ends with
`mu = D` to four decimals, so growth balances dilution and the run reads as
settled, while the biomass is 480x wrong and the pool 42% off. It has found a
state where `dX/dt = 0` without the pool being at steady state at all. Never read
`mu = D` as convergence.

**Why this earns its place beyond convenience.** For the cells whose R* values are
tied -- 0.02-0.15% on cells 1, 2 and 7, four-way at 0.015% on cell 10 -- the
equilibrium cannot say who wins, because the surrogate cannot resolve the gap. In
a tie the winner is set by initial abundances and by the transient, which is a
question only a time course can answer. **For exactly the cells where
`steady-state` is weakest, the transient is the more informative instrument.**

Regression test: `tests/test_cfs_steady.py`, the Monod chemostat's closed form,
with Euler as the failing control.

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

#### Built — `cfs steady-state`, 2026-09-04

`src/cfs/science/steady.py`. Surrogate only by default; no LP anywhere in the solve.

    D (c_feed - c) + sum_i X_i z_i(c) = 0        X_i (mu_i(c) - D) = 0

The second family is a complementarity condition, so it is a **square Newton
system over an assumed survivor set**, wrapped in an active-set loop: drop a
member whose abundance reaches zero, re-admit one whose `mu(c*)` exceeds `D`,
never re-admit a member already dropped in this solve (Bland's rule, and it is
needed). Only the metabolites some member exchanges are solved for; the rest have
`z = 0`, so `c = c_feed` solves their row exactly and carrying them would add
rank-deficient directions. `J` is finite-differenced in `c` with the `X` columns
exact, since `d(dc/dt)/dX_i = z_i` is what a unit-biomass probe already returns.
`lstsq` after **row and column equilibration**, per §7's measurement that the
Hessian sum is rank ~10-25 of 365. Coexistence, stability, invasion and
`dy*/dc_feed = -J^-1 D I` all come out of the same factorisation.

**The finite-difference step is the single most important line in the file, and
the first version of it was wrong by 61x.** The step was `1e-3 * (c + Km)`. The
limiting metabolite is by definition the scarce one, and at a real fixed point
`EX_k_e` sits at `c = 3.0e-8` against `Km = 1e-3` -- so that step is **33x `c`
itself** and secants straight across the Michaelis-Menten saturation:

| `d(mu)/dc` for `EX_k_e` | value |
| --- | --- |
| the LP's own shadow price, chain-ruled | **5 122 827.50** |
| finite difference at `h = 1e-4 c` | 5 122 827.5 |
| finite difference at `h = 1e-3 (c + Km)` | **84 153.7** |

It is 61x low on precisely the row that sets the answer. Fixed to
`1e-3 * max(c, 1e-3 Km)` -- relative to `c`, with `Km` only as a floor for a
metabolite at `c = 0`. Both bounds are real and were measured: too small and the
float32 heads return noise (`sqrt(eps_f32)` ~ 3e-4 is the floor), too large and it
crosses the kink. **Effect on the same cell: 159 Newton iterations and a residual
of 9.3e-7 became 11 iterations and 1.8e-8, 615 s -> 55 s.** Two of five 2-member
cells now converge outright where one did, and a third is within 8x of tolerance.

**LP residual, exact dual rows, and the mixed residual.** Three arms on the same
five cells, identical warm starts, `--fd-check 0`:

| cell | surrogate | pure LP residual | mixed at 1% |
| --- | --- | --- | --- |
| 1 | **1.8e-8**, 11 it, 55 s | 3.0, 6 it | **6.9e-9, 10 it**, `lp_frac` 0.03 |
| 2 | 10.0, 11 it | 10.0, 1 it | 10.0, `lp_frac` 0.88 |
| 3 | 10.0, 100 it (cap), 477 s | 10.0, 0 it | 10.0, `lp_frac` 0.82 |
| 4 | **5.3e-8**, 25 it, 113 s | 0.021, 13 it | **5.3e-8, 11 it, 69 s** |
| 5 | 7.8e-6, 23 it | **9.4e-7**, 8 it (`sur_res` 4.4) | 7.8e-6, `lp_frac` 0.00 |

1. **A pure LP residual with a surrogate Jacobian does not converge** -- 0-13
   iterations before the line search finds no descent. That is the textbook
   inexact-Newton failure, and `mu_rel` at these states (1e-4 to 9e-3) says why it
   is not worth it: Head A is already accurate here, so the LP buys little on `mu`
   while the residual/Jacobian inconsistency costs the direction.
2. **The mixed residual fixes that.** Solve both; keep the surrogate's `(mu, z)`
   for any member agreeing within `--mix-mu-rel`, substitute the LP only where
   they diverge, and give the Jacobian **exact dual rows for exactly those
   members** (NaN elsewhere means "keep the finite difference"). It matches or
   beats surrogate-only wherever it converges and needs **2.3x fewer iterations on
   cell 4** at 3-5% LP usage. It costs the LP either way -- divergence cannot be
   detected without it -- so what it buys is consistency, not solves.
3. **The mix triggers on `mu`, and Head B's error is in `z`.** Cell 5 fires on
   *nothing* at 1% and stays at 7.8e-6 while the pure LP converges to a different
   fixed point where the surrogate's own residual is **4.4**. A `z`-side trigger
   is the obvious extension and is not built.
4. **The exact dual rows are free and verified.** `d(mu)/dc = pi * (-Vmax) *
   Km/(Km+c)^2`, with **the same two corrections the label pipeline applies** --
   the dual is that derivative only where the bound binds, and clamping at zero
   also drops the solver dust that is half the non-zero duals. Verified against a
   properly-scaled finite difference to 8 significant figures.

**The remaining failures are the optimiser, not the model.** Cells 2 and 3 sit at
a scaled residual of *exactly* 10.0 in all three arms, and that is not a bad row:
it is `feed/Km ~ 10` on nearly every fed metabolite at once, because the iterate
has collapsed the whole pool to `c ~ 0` and `mu ~ 0.005` against `D ~ 12`. It has
driven metabolites the community *secretes* (`EX_h2_e`, `Xz = +0.029`, steady
state `c >= c_feed`) to zero, where the clip at zero holds them. A damped line
search is not enough globalisation for this; a real trust region, Jacobian-free
Newton-Krylov, or pseudo-transient continuation is the fix, and all three are
`scipy.optimize.root` one-liners that would *delete* the hand-rolled Newton here.

**And §13.7's caution is measured, not predicted.** `reach` at `c*` is **1.0-5.5**
against a held-out design median of ~0.10, and the two cells that fail are the two
deepest (5.2 and 5.5). An equilibrium really is a long way outside the design.
The cheap answer is that an equilibrium visits **one** state, so paying the LP
there costs nothing like §8.6g(4)'s 24.6% of member-steps along a trajectory.

**Traps, in the order they cost time.**

1. **`Surrogate.reach` did not exist.** §8.6g(1) recorded it as shipped and
   `cfs simulate` called it; the method was never written, so every `simulate`
   run raised `AttributeError`. Found only when a second caller reused it.
2. **The finite-difference step, above.** It also made every *other* diagnosis in
   this section look worse than it was, including the warm-start comparison.
3. **Do not warm-start by integrating** -- though the reason is narrower than it
   first appeared. The medium saturates `mu` at ~0.2% of the feed, so the dynamics
   are stiff at the kink and explicit Euler ratchets `X` upward (growing at
   `mu - D` while `c > 0`, decaying only at `D` after it clips `c` to zero) to
   `X ~ 1e8`. At a smaller step it washes out to the *spurious extinction* fixed
   point instead -- `X ~ 1e-9`, residual 1e-13, and a genuine root. The warm start
   is a bisection on a **partially** scaled feed: only what the community consumes
   is scaled down, because a secreted metabolite's steady state sits at or above
   the feed, and dragging it down with everything else makes the NNLS that matches
   abundances return `X = 0` on 8 of 9 cells (water and protons dominate the
   right-hand side and the community secretes both).
4. **Fraction to the boundary, or the anti-cycling rule eats the community.** One
   Newton overshoot takes an abundance through zero, the ban stops it returning,
   and the solve converges cleanly at residual 1e-13 on `X = 0, c = c_feed`.
5. **§8.4's `rtol=1e-10` is unreachable and the solver is not at fault.** float32
   heads plus a finite-differenced `J` floor the residual near 5e-8 relative.
   Default `tol` is `1e-6`, dimensionless (pool rows over `D*Km`, growth rows over
   `D`); read `residual_max_scaled`.
6. **Report the residual of the rhs you actually solved.** `residual_max` was
   computed from the surrogate even in LP mode, which made a converged LP run read
   as a failure at 4.4. The surrogate's residual at the LP's fixed point is a
   genuinely useful number -- it is how far the surrogate alone is from calling
   that state an equilibrium -- but it is not the convergence test, and it now has
   its own key.

##### The solver pass — 2026-09-04. One job refuted, one exact, one mixed

**Job 1, replacing the hand-rolled Newton with `scipy.optimize.root`, is
refuted, and the reason is worth keeping: none of those methods knows `X > 0`.**
On the toy chemostat, whose answer is closed-form, `hybr`, `df-sane`, `broyden1`
and `krylov` **all** converge to the *trivial washout root* `X = 0, c = c_feed` --
which is always present, is usually nearest, and is wrong. The hand-rolled Newton
avoided it only through its fraction-to-the-boundary step. Re-parametrising
abundances as `log X` removes that root to minus infinity and does not rescue
them either: they then converge nowhere on the toy. `--solver` keeps all four
selectable and defaults to `newton`.

**A matching fraction-to-the-boundary on `c` is also refuted**, and it was the
obvious reading of the diagnosis (a step collapses the pool, so cap the step).
Measured: cell 1 goes from converged (1.8e-8, 11 iterations) to failed (20.0, 21)
and cell 5 from 7.8e-6 to 14. The boundaries are **not symmetric** -- a
concentration reaching zero is a normal steady state (that metabolite is absent),
where an abundance reaching zero is a change of active set.

**Job 5, the analytic growth rows, is exact and is the durable result.** Head A is
analytically differentiable, so those rows never needed probing:
`dmu/dc = dmu/dx . dx/du . du/dc` times the output calibration's derivative
(`calibrate.deriv`, new -- `mu` is *reported* calibrated, so the Jacobian must be
too). Two cross-checks, both on the limiting metabolite of a real fixed point:

| `d(mu)/dc`, `EX_k_e` | value |
| --- | --- |
| Head A, analytic | **5 122 829** |
| the LP's shadow price, chain-ruled | **5 122 827.5** |
| finite difference, corrected step | 5 122 807 - 5 122 838 |

Agreement to **7 significant figures** between two independent derivations. And it
is not only about precision: at `EX_cu2_e` the analytic gradient is 5.6 where the
finite difference returns **0.0** -- FD was silently zeroing real entries.

**Job 2, a warmer Jacobian temperature, is real but not a default.** Head B's `z`
does not depend on Head A's temperature, so the growth rows are the *only* place
the shipped `gm_eval_temp = 1e-4` -- effectively a hard min, so piecewise-constant
derivative -- reaches the Jacobian. `--jac-temp` warms Head A there and nowhere
else, which costs nothing by construction. Measured at `T = 0.01`
(gradient cosine to the shipped one: 0.99992 at 1e-3, 0.9891 at 1e-2):

| cell | FD rows | analytic rows (default) | analytic + `--jac-temp 0.01` |
| --- | --- | --- | --- |
| 1 | **1.8e-8**, 11 it | **1.5e-8**, 11 it | 2.3, 7 it |
| 2 | 10.0, 11 it | 9.8, 100 it | 10.0, 37 it |
| 3 | 10.0, **100 it** | 10.0, **14 it** | **4.0**, 20 it |
| 4 | **5.3e-8**, 25 it | **5.3e-8**, 30 it | **1.9e-7, 9 it** |
| 5 | 7.8e-6, 23 it | **1.2e-6**, 13 it | 1.2e-6, 14 it |

Analytic rows leave the converged count at 2 of 5 but move cell 5 from 7.8e-6 to
**1.2e-6**, just outside a 1e-6 tolerance, at half the iterations, and make cell
3's failure 7x cheaper. Warming buys cell 4 (25 -> **9** iterations) and cell 3
(10.0 -> 4.0) and **loses cell 1 outright**, so it stays off by default. Timings
in that table are from two arms run in parallel and are contended; read
iterations, not seconds.

**Job 3, V4, re-measured with the corrected step.** The implicit feed derivative
against a full re-solve at a perturbed feed, 10 components per cell:

| | cell 1 | cell 4 |
| --- | --- | --- |
| median relative error | **3.1e-7** (6.0e-7 analytic) | **3.9e-6** (3.1e-6 analytic) |
| max | 1.4e-6 | 3.9e-5 |

Against the 2.0e-4 / 4.9e-3 measured with the broken step -- 50-500x better, and
**V4 passes** on every cell that converges.

**Job 6, a `z`-side trigger for the mix (`--mix-z-rel`).** The `mu`-only trigger
fires on nothing exactly where it is needed: Head A is the accurate head, so a
member can have `mu` right to four decimals and `z` badly wrong, which is §8.6g's
whole finding restated at an equilibrium. Cell 5 fired on **0%** of members at
`--mix-mu-rel 0.01` while the pure-LP residual converged to a different fixed
point at which the surrogate's own residual is **4.4**.

Measured on that cell, `--mix-mu-rel 0.01 --mix-z-rel 0.1`: the trigger fires on
**54% of members** against 0% for `mu` alone, and the solve takes **5 iterations
against 23**, ending at 4.9e-6. (Not comparable to the surrogate-only 1.2e-6 --
that is a different, harder residual -- but it is the first evidence the trigger
is firing on the right members.)

**Job 7**, regression tests for the dual chain rule -- the sign convention, both
label clamps (non-binding duals, and the O(1e-14) dust that is half the non-zero
ones), and that a stale dual cache returns NaN rather than a silently wrong row.

##### The globalisation pass — 2026-09-04. Both arms are cell-dependent, neither is a default

Job 1 of the list below was skipped in favour of the solver work, and the solver
work answered half of it anyway. Two globalisations, both off by default so every
number above reproduces bit for bit. Same five cells, same warm starts,
`--fd-check 10`, `value_p4r2`/`behaviour_p4r2`:

| cell | baseline | `--ptc 1e-3` (trust region) | `--d-steps 8` (continuation) |
| --- | --- | --- | --- |
| 1 | **1.5e-8**, 11 it | **1.5e-8**, 11 it | **2.3e-8, 5 it**, V4 1.7e-6 |
| 2 | 10.0, 100 it | 9.8, 100 it | 2.7e+02 |
| 3 | 10.0, 14 it | 10.0, 20 it | **4.5e-05** |
| 4 | **5.3e-8**, 30 it | **1.5e-7**, 31 it | 2.8e-01 |
| 5 | 1.2e-6, 13 it | 1.2e-6, 13 it | **1.2e-6, 11 it** |

**1. `--ptc`, a Levenberg-Marquardt trust region, is null -- and that is the
informative half.** When backtracking exhausts, it escalates the damping and takes
a *different* direction rather than returning "no descent". It fires on cells 2
and 3 (cell 3 goes 14 -> 20 iterations) and changes nothing: residual still
exactly 10.0. **So the failure diagnosed above -- "a damped line search is not
enough globalisation" -- is wrong.** Backtracking was not rejecting a good step
for being too long, and no reachable direction from that iterate helps. The
remaining failures are not the step.

Two dampings that look equivalent and are not, both measured on the Monod toy and
both now in the docstring. `A + damp I` **after** the row equilibration perturbs a
rank-deficient non-symmetric matrix arbitrarily and returns a step **4x larger**
than the undamped one. True pseudo-transient continuation, `A + damp diag(rscale)`
with no line search, is unbounded -- `X` reaches 1e80 -- because the damping alone
does not cap a step whose abundance column scaling is multiplicative. The
normal-equation form `(A^T A + damp I) w = A^T r` is the one that works: symmetric
positive definite for any `damp > 0`, so the step is always a descent direction
for `||r||^2` and shrinks monotonically in `damp`.

**2. `--d-steps`, natural-parameter continuation in `D`, is the first thing to
move cells 2 and 3 -- and it breaks a cell that was converging.** `D -> mu_max` at
the feed is the transcritical end, where the fixed point is known exactly
(`c = c_feed`, `X = 0`); the ladder walks `D` down from there, warm-starting each
rung on the last. It **halves cell 1** (11 -> 5 Newton iterations, V4 still
passing) and takes **cell 3 from the collapsed-pool 10.0 to 4.5e-5** -- five orders,
and just outside the 1e-6 tolerance. It also takes cell 4 from converged (5.3e-8)
to 2.8e-1 and cell 2 from 10.0 to 2.7e+02.

**3. Cell 3 nearly converging is the free half of Job 1.** A state at a scaled
residual of 4.5e-5 is *almost* a fixed point, which is evidence cell 3 has one and
that its old failure was the warm start, not the model. Cell 2's ladder, by
contrast, converges only on its **top** rung and fails on all eight below, ending
worse than where it started -- the top rung's answer is `X ~ 0` by construction, so
the ladder there is carrying a degenerate state down. Re-seeding the abundances at
each rung (the NNLS the single-`D` warm start already does) is the obvious next
variant and is untested.

**4. A free structural scan, and a negative result worth keeping.**
`branch_scan.py` walks the same partially-scaled feed the warm start bisects and
asks the two fixed-point conditions separately -- can `mu` reach `D`, can the pool
balance close with `X >= 0` -- in ~60 right-hand-side calls and no LP. On cells 2
and 3 `max mu / D` does cross 1 (at `theta` 4.6e-4 and 2.2e-4), so the growth
condition is satisfiable. But the pool residual is **10-16 at every `theta` on all
three cells scanned, including cell 1, which converges**. So the warm start's path
never closes the pool balance even where the solve succeeds: `c*` is not a uniform
scaling of the feed, and this scan cannot decide existence. It is cheap enough to
run before any future warm-start idea.

**Neither flag defaults on.** Like `--jac-temp`, both win some cells and lose
others, and the converged count over the five is 2 (baseline), 2 (`--ptc`), 1
(`--d-steps`). The per-cell decision is now three flags wide, which is itself a
finding: this solver has no single setting.

##### The M12 gate over the roster — 2026-09-05. 60% failure, and one survivor everywhere

Job 2. All ten §8.1 communities, sizes 2 to 21, one feed draw each (`seed 0`),
default solver, `value_p4r2`/`behaviour_p4r2`. V4 on the cells small enough to
afford it (it re-solves the whole fixed point per column).

| cell | n | conv | residual | it | surv | stable | `reach` | max `mu_j(c*) - D`, excluded | V4 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2 | **yes** | 1.5e-08 | 11 | 1 | yes | 2.7 | +2.5e-4 | **9.1e-07** |
| 2 | 2 | no | 9.8e+00 | 100 | 1 | — | 2.4 | -10.9 | — |
| 3 | 2 | no | 1.0e+01 | 14 | 1 | — | 5.7 | -14.3 | — |
| 4 | 2 | **yes** | 5.3e-08 | 30 | 1 | yes | 2.2 | -2.16 | **3.1e-06** |
| 5 | 2 | no | 1.2e-06 | 13 | 1 | — | **1.0** | -9.8e-5 | — |
| 6 | 3 | no | 4.2e-03 | 76 | 1 | — | 2.7 | +5.4e-4 | — |
| 7 | 3 | **yes** | 7.3e-08 | 32 | 1 | yes | 2.4 | -2.2e-4 | **5.6e-07** |
| 8 | 5 | **yes** | 9.2e-08 | 35 | 1 | yes | 2.7 | +5.6e-5 | — |
| 9 | 10 | no | 4.2e-03 | 76 | 1 | — | 2.7 | **+41.0** | — |
| 10 | 21 | no | 1.1e+01 | 15 | 1 | — | 4.5 | -11.9 | — |

**1. V4 passes on every cell that converges** -- 5.6e-7 to 3.1e-6 -- which is the
half of M12's gate that is met, and it now has four cells behind it rather than
two.

**2. The Newton failure rate is 60%, against a gate of 1%.** M12 does not pass.
Failure is **not** size-monotone: the 5-member cell converges in 35 iterations
where three of the five 2-member cells fail. Size was never the axis here either.

**3. `reach` does not separate converged from failed, and the earlier claim that
it did is retracted.** It was measured on five 2-member cells; over ten it is
2.2-2.7 on the converged and **1.0-5.7** on the failed. Cell 5 fails at
`reach = 1.0`, the *shallowest* cell in the set, and cell 8 converges at 2.7. The
head being off-distribution at `c*` remains true and remains §13.7's concern; it
is not the convergence predictor.

**4. Failure is bimodal, and only one mode is the pool collapse.** Cells 2, 3 and
10 sit at a scaled residual of 10-11, which is the `feed/Km` signature of a
collapsed pool. Cells 5, 6 and 9 stop at 1.2e-6 / 4.2e-3 / 4.2e-3 -- within one to
three orders of the tolerance, on a state that is nearly an equilibrium. Those are
two different problems and should stop being counted as one number.

**5. Exactly one survivor on every cell at every size -- and four of them are not
valid fixed points.** The `invasion_score` column is the check, and it is free:
an excluded member with `mu_j(c*) > D` could grow, so the active set is wrong.
Cells 1, 6 and 8 are within **0.2% of `D`** of a tie -- neutral coexistence at the
surrogate's own resolution, not clean exclusion -- and **cell 9 is outright
invalid at +41.0 against `D` = 11.1**, an excluded member growing 4.7x faster than
the dilution rate.

This is the anti-cycling ban doing exactly what it is documented to do. Bland's
rule guarantees the active-set loop terminates; it does not guarantee it
terminates on a state satisfying the complementarity condition, and on a near-tie
it will not. So "complete competitive exclusion at every community size" is the
*reported* answer and it is only trustworthy on the six cells whose excluded
members are decisively negative. **Read `invasion_score` before quoting a
coexistence result**, and treat any cell with `max invasion > 0` as unconverged
whatever `residual_max_scaled` says -- the report has both and the convergence
flag currently reflects only the second.

##### The near-ties are not solver failures — 2026-09-05

Job 1 of the list below. Three of the four invadable cells are ties the surrogate
cannot resolve, and the fix is a threshold, not an algorithm.

| cell | n | `D` | max `mu_j(c*) - D`, excluded | as a fraction of `D` |
| --- | --- | --- | --- | --- |
| 8 | 5 | 3.83 | +5.6e-5 | **1.5e-05** |
| 6 | 3 | 9.35 | +5.4e-4 | **5.8e-05** |
| 1 | 2 | 0.154 | +2.5e-4 | **1.6e-03** |
| 9 | 10 | 11.1 | **+41.0** | **3.7** |

**Head A's own `mu_rel` at these fixed points is 1e-4 to 9e-3** (measured in the
mixed-residual arm above). Three of the four margins sit at or below that, so
"this excluded member can invade" is a claim the model cannot support -- and the
fourth is **four orders larger**. The split is clean; there is nothing in between.

`--invade-rel` (default **1e-2**) is therefore the threshold for both the
re-admission test and the `invadable` report, and deliberately the *same* number
for both: the loop must never decline to chase a member it then calls an invader.
At the default, cells 1, 6 and 8 are exclusions and only cell 9 is a failure.

**A bounded re-admission budget was built first and it does not fix the ties.**
`--readmits` (default 1) replaces the permanent anti-cycling ban with a per-member
allowance -- termination only needs re-admissions to be *finite*, not forbidden.
On cell 1 it changes nothing: the member is re-admitted, the two-survivor Newton
fails to converge, the "inconsistent survivor set" branch drops it again, and the
budget is spent reaching the same state (11 iterations, identical residual). So at
a genuine tie there is **no two-survivor fixed point the solver can reach**, which
is the same answer the threshold gives, arrived at 75 seconds more expensively.
Kept at 1 because it costs nothing where it does not fire, and because cell 9's
margin is real and is exactly the case it was built for.

**What this does not settle.** Whether cells 1, 6 and 8 *actually* coexist is
beyond this surrogate: at 1.5e-5 to 1.6e-3 of `D` the question needs the LP, and
`--roster`/`--mix-mu-rel` is how to ask it -- an equilibrium is one state, so the
solves are affordable. Until then "one survivor at every size" should be read as
"one survivor, with three cells too close to call".

##### Cell 9, and keystone leave-one-out — 2026-09-05

**The re-admission budget fixes the one real invasion, which is what it was built
for.** Cell 9 (n=10), the only cell whose excluded member cleared `--invade-rel`:

| cell 9 | permanent ban | `--readmits 1` |
| --- | --- | --- |
| max `mu_j(c*) - D`, excluded | **+41.0** (3.7x `D`) | **+3.2e-05** (2.9e-6 x `D`) |
| `invadable` | yes | **no** |
| `residual_max_scaled` | 4.2e-03 | 3.2e-02 |
| iterations / active-set passes | 76 / 8 | 96 / 10 |

Reproduced exactly on a second run. **The residual gets worse and that is not a
regression**: the old number was a well-solved *wrong* active set, the new one is
the right active set solved less well. Read the two together or the arrow points
backwards.

So `--readmits` and `--invade-rel` divide the four invadable cells cleanly between
them -- the budget cannot manufacture a fixed point at a tie (cell 1 is unchanged),
and the threshold does not excuse a genuine invasion (cell 9 clears it by 370x).

##### Keystone members, at a fixed feed

The §13.4 job that needs no code -- except one line, because it was confounded:
the feed is drawn over the **union** of the members' active subspaces, so dropping
a member silently redraws the chemostat being compared against.
`steady_state.npz` now records `feed`, and every leave-one-out below runs at the
full community's own feed *and* its `D`, via `--medium` and `--dilution`.

**Cell 7 (n=3), and it is textbook.**

| removed | conv | survivors | `X` |
| --- | --- | --- | --- |
| — | yes | CP001726.1 | 2.14e-05 |
| **CP001726.1** | yes | **DACTBY01 + GCA_000007325.1** | 2.14e-05, 2.11e-05 |
| DACTBY01 | yes | CP001726.1 | 2.14e-05 *(identical)* |
| GCA_000007325.1 | yes | CP001726.1 | 2.14e-05 *(identical)* |

**This is the first coexistence anywhere in the project**, and it settles a
question the M12 roster left open: the active-set loop *can* return a two-survivor
fixed point, cleanly (residual 5.0e-7, not invadable). "One survivor at every
size" is therefore a fact about these communities under a dominant member, not a
limitation of the solver -- and it strengthens the near-tie reading above, since
here a reachable two-survivor state plainly exists and at cell 1's tie none does.

**Cell 8 (n=5), and it is not textbook at all.**

| removed | conv | survivors | `X` |
| --- | --- | --- | --- |
| — | yes | AAXE02 | 1.7e-05 |
| AAXE02 | yes | GCA_000209935.1 | 1.59e-05 |
| **CP027002.1** | yes | **AAXE02 + DACTBY01** | 1.14e-05, 5.08e-06 |
| **DACTBY01** | no (4.6e-03) | **CP027002.1** | 1.74e-05 |
| FNPN01 | yes | AAXE02 | 1.7e-05 *(identical)* |
| **GCA_000209935.1** | no (4.6e-03) | **CP027002.1** | 1.74e-05 |

Removing the dominant member is the expected result: a clean succession to the
runner-up. **The other three are not.** `CP027002.1`, `DACTBY01` and
`GCA_000209935.1` are all *extinct* in the full solve, and removing any of them
changes the answer -- one into coexistence, two into a different survivor
entirely. `FNPN01`, equally extinct, is inert to three significant figures.

**An extinct member contributes `X_i z_i = 0` to the pool balance and `X_i = 0`
satisfies its own growth row, so it cannot change the fixed-point equations.**
What it changes is the *path*: which member the inconsistent-set branch drops
first, and in what order re-admissions are spent. So the equilibrium this solver
returns on cell 8 is **path-dependent**, which means cell 8 has **multiple fixed
points** and the loop selects among them by history. That is P6 arriving in the
steady state rather than in the posterior.

**Consequences, in order.** (a) A keystone result is only interpretable where the
extinct members are inert -- cell 7 passes that check, cell 8 fails it -- so
**report the inert-removal control alongside every keystone claim**; it costs one
extra solve per excluded member and it is the difference between an ecological
finding and a solver artifact. (b) The cheap confirmation is to re-solve the
*full* community warm-started at each leave-one-out's answer: if it converges
there too, multiple fixed points are demonstrated rather than inferred. Untested.
(c) `--readmits` raises the number of reachable paths, so it may itself be what
made cell 8 path-sensitive; the `readmits 0` comparison is one flag and was not
run.

##### The chemostat steady state is not unique — confirmed, 2026-09-05

`--warm-start` takes another solve's `steady_state.npz` as the starting `(c, X)`;
members it does not name enter **dead**, so the active-set loop's own re-admission
test decides whether they can invade. That makes the multiplicity check exact:
hand each cell-8 leave-one-out's answer back to the **full five-member**
community, same feed, same `D` = 3.8316, and ask whether it is still an
equilibrium.

| warm-started from | that sub-community's survivors | full-community result | conv | invadable | residual |
| --- | --- | --- | --- | --- | --- |
| −AAXE02 | GCA_000209935.1 | **GCA_000209935.1** | yes | no | 1.7e-07 |
| −CP027002.1 | AAXE02 + DACTBY01 | **AAXE02 + DACTBY01** | yes | no | 2.8e-07 |
| −FNPN01 | AAXE02 | **AAXE02** | yes | no | 3.6e-07 |
| −DACTBY01 | CP027002.1 | CP027002.1 | no | no | 4.6e-03 |
| −GCA_000209935.1 | CP027002.1 | CP027002.1 | no | no | 4.6e-03 |
| default warm start, `--readmits` 0 **or** 1 | — | **AAXE02** | yes | no | 9.2e-08 |

**Three distinct, converged, non-invadable fixed points of the identical system**:
`AAXE02` alone, `GCA_000209935.1` alone, and `AAXE02 + DACTBY01` coexisting. Two
more warm starts land on a fourth candidate that does not converge (both at the
same 4.6e-03, so they are the same point). The previous section inferred
multiplicity from the extinct-member argument; this measures it.

**1. The coexistence is real, and it belongs to the full community.** It was found
by deleting `CP027002.1`, but restoring that member does not destroy it -- it
cannot invade. So cell 8's keystone reading was an artifact only in *which*
attractor the deletion moved the solver to; the state it found is a genuine
equilibrium the default warm start never reaches.

**2. It is not `--readmits`.** The budget was the obvious suspect, being this
session's own change and a way to reach more states. `--readmits 0` returns
`AAXE02` at 9.2e-08 exactly as before, so the path sensitivity predates it.

**3. What this costs the downstream sections.** §13.4 says the steady state "is
the right place to quote numbers, to define objectives, and to differentiate
through". It still is -- but **there is no such thing as *the* steady state of
these communities**, and every number quoted at one (coexistence, stability,
invasion, `dy*/dc_feed`, §13.5's interaction rate, §13.6's forward map) is
conditional on the warm start that selected it. P6 anticipated multiple equilibria
as "clustered divergences" in HMC; they are visible far earlier and for three
solves rather than a chain.

**4. The cheap instrument already exists.** A multiplicity scan is
`--warm-start` from a handful of structured starting points -- each single-member
monoculture's own equilibrium is the obvious basis -- counting distinct converged
non-invadable states. No LP, no new code. Quote a steady-state result with the
size of that set, or say it was not measured.

##### The multiplicity scan — and a retraction of "three fixed points" — 2026-09-05

Every member's own monoculture equilibrium, at the cell's own feed and `D`, handed
to the full community. The previous section read `invadable` at the shipped
`--invade-rel 1e-2` and concluded cell 8 had **three** fixed points. Reading the
*signed margin* instead says something sharper and different.

| cell | state | conv | residual | survivors | max `(mu_j - D)/D`, excluded |
| --- | --- | --- | --- | --- | --- |
| 1 | **default** | yes | 9.4e-08 | GCA_000151225.1 | **+1.64e-03** |
| 1 | from CR626927.1 | yes | 8.5e-07 | **CR626927.1** | **−1.64e-03** |
| 4 | default / either probe | yes | 5.3e-08 | CR626927.1 | **−8.83e-01** |
| 7 | default / all three probes | yes | 7.3e-08 | CP001726.1 | **−2.09e-04** |
| 8 | **default** | yes | 9.2e-08 | AAXE02 | **+1.47e-05** |
| 8 | from −CP027002.1 | yes | 2.8e-07 | **AAXE02 + DACTBY01** | **−3.42e-05** |
| 8 | from −AAXE02 | yes | 1.7e-07 | GCA_000209935.1 | +2.81e-04 |
| 8 | from −FNPN01 | yes | 3.6e-07 | AAXE02 | +1.47e-05 |

**1. RETRACTED: "three distinct converged non-invadable fixed points".** At the
1e-2 tolerance all three read as non-invadable; by sign, **only one of them is** —
the coexistence, at −3.42e-05. The other two are invadable by +1.5e-05 and
+2.8e-04. They are near-equilibria, not equilibria, and the tolerance was hiding
the distinction it was introduced to make. Multiple equilibria at the *surrogate's
resolution* is still real and still the operative problem, because those margins
sit inside Head A's own 1e-4 to 9e-3 error; multiple equilibria as a mathematical
claim about the system is **not established**.

**2. The default warm start returns a strictly invadable state on 2 of the 4
converging cells.** Cell 1's default gives `GCA_000151225.1` at **+1.64e-03** while
the other member's basin gives `CR626927.1` at **−1.64e-03** -- exactly antisymmetric,
as a two-member exclusion must be, and the default picks the wrong side. Cell 8's
default gives `AAXE02` at +1.47e-05 where the valid state is the coexistence. So
the earlier "one survivor at every size" was partly the warm start choosing an
invalid exclusion.

**3. So there is a free selection rule: take the most negative margin.** Every
`steady_state.json` already carries `invasion_score`. Across the scan it picks
`CR626927.1` on cell 1 and the coexistence on cell 8 -- in both cases the strictly
valid state over the default's invalid one -- and is unchanged where the answer is
unique (cells 4 and 7). **Rank scan states by margin; do not stop at the first
that converges.**

**4. Cells 2 and 3 have fixed points. The failure was the basin, not existence.**

| cell | default survivor | default residual | monoculture-seeded survivor | residual |
| --- | --- | --- | --- | --- |
| 2 | CP001726.1 | **9.8** | **CP001820.1** | **1.3e-05** |
| 3 | AAXE02 | **10.0** | **ABCC02** | **3.8e-05** |

Six orders, on the two cells that had defeated every method tried -- and cell 3's
3.8e-05 agrees with the continuation arm's 4.5e-05, so two independent routes find
the same state. In both, the good fixed point is the one where the *other* member
wins: the bisection warm start commits to the wrong survivor and collapses the
pool. **This closes the "do cells 2 and 3 have a fixed point at all" question that
has been open since §13.4 was written, and the answer is yes.**

**5. Every monoculture probe that mattered reported `mono=False`.** An unconverged
monoculture is still a far better basin seed than the bisection. That, not any of
the three globalisation flags, is the change worth making: **seed from the
monocultures**. It costs `G` extra solves, it is trivially parallel, and it is the
only thing that has moved cells 2 and 3.

**6. Uniqueness is predicted by the margin's sign, not its size.** Cell 4 at
−8.8e-01 and cell 7 at −2.1e-04 are both unique across every probe, four orders
apart in magnitude; cells 1 and 8, both positive, are not. A prediction made from
magnitude ("a near-tie means alternative states") was wrong on cell 7 and is
recorded here because it was made in advance.

##### Monoculture seeding is the default, and the M12 gate re-run on it — 2026-09-05

`--seed-mode monoculture` (default; `bisect` reproduces every earlier number).
Solve from the bisection, then from each member's own monoculture equilibrium,
and keep the state with the most negative **signed** invasion margin, stopping at
the first strictly valid one. Nine of ten roster cells, one feed draw, same
`value_p4r2`/`behaviour_p4r2`:

| cell | n | before | after | margin now | note |
| --- | --- | --- | --- | --- | --- |
| 1 | 2 | 9.4e-08, **+1.6e-03 invalid** | 8.5e-07 | **−1.64e-03** | valid, and *faster* (101 s vs 205 s) |
| 2 | 2 | **9.8** | **1.1e-05** | +7.0e-05 | different survivor (`CP001820.1`) |
| 3 | 2 | **10.0** | **2.1e-05** | **−2.49e-05** | different survivor (`ABCC02`), valid |
| 4 | 2 | 5.3e-08 | 5.3e-08 | −8.83e-01 | control: unchanged, V4 3.1e-06 |
| 5 | 2 | 1.2e-06 | 1.2e-06 | −8.11e-06 | unchanged |
| 6 | 3 | 4.2e-03, **+5.8e-05 invalid** | 1.9e-03 | **−5.49e-06** | now valid |
| 7 | 3 | 7.3e-08 | 7.3e-08 | −2.09e-04 | control: unchanged, V4 5.6e-07 |
| 8 | 5 | 9.2e-08, **+1.5e-05 invalid** | 9.4e-09 | +1.47e-05 | **still invalid** — see below |
| 9 | 10 | 4.2e-03, **invalid** | **2.4e-07** ✓ | **−3.09e-06** | converges now |

**Four cells improved, two unchanged, zero regressions**, and both unique cells
(4 and 7) are bit-identical including V4. Strictly valid states go from 4 of 9 to
**8 of 9**, and the two cells that had defeated every method reach 1e-05 from a
collapsed-pool 10.

**1. It does not find a coexistence, and cell 8 is the proof.** Cell 8's only
valid state is the two-member `AAXE02 + DACTBY01` equilibrium, and no
single-member monoculture basin reaches it -- it was found by the *leave-one-out*
probe, which removes a member from the system rather than starting it dead. So
monoculture seeding explores alternative **monoculture** equilibria only. Covering
coexistence needs pairwise probes, `G^2`, and one cell is not evidence enough.

**2. The cost is bounded by two knobs, and at n=21 that is still not enough.**
Probes run fastest-grower-at-the-feed first (a chemostat's survivor usually has
the lowest break-even concentration, which tracks `mu` at the feed) and stop at
the first strictly valid state; `--seed-probes` (default 4) caps them.

**Cell 10 (n=21) is not measured on the new default.** It was attempted at
uncapped, 2 and 1 probes and exceeded a one-hour wall each time -- the bisection
solve alone is ~15 min there, and even one probe adds a monoculture plus a second
full solve on a 21-member system. So the honest statement is that monoculture
seeding is affordable to about n=10 (cell 9 took 28 min) and **not measured
beyond it**. `--seed-probes 0` gives exactly the old bisection behaviour without
changing mode, and is the escape hatch for a large community; the roster-scale
cell should be re-run there or on better hardware before M12 is quoted at n=21.

**3. Two false starts, both worth keeping.** *Ranking on the margin alone prefers
the degenerate state*: a collapsed pool has `c ~ 0`, so `mu ~ 0` for everybody,
nobody can invade it, and it scores the most negative margin in the set (−0.948).
The margin is meaningful only on a converged state, so the key is
`(converged, margin, −residual)` -- the same trap as judging a root find by its
residual, one level up. And *zeroing the other abundances is not a monoculture*:
reusing the full community's `c0` with `X_j = 0` left cells 2 and 3 at 9.8. The
**bisection itself** has to be per-sub-community -- `consumed` from that member's
own consumption, the theta search targeting *its* `mu = D`.

**4. Unexplained, and not waved through: cell 1's V4 is 8.5e-05 against 9.1e-07.**
It is measured at a different (now valid) fixed point so it is not like-for-like,
but two orders is not noise and M12's V4 half should not be called met on cell 1
until it is understood. Cells 4 and 7 are unmoved at 3.1e-06 and 5.6e-07.

##### V4 was never established: it is a median at 5 components — 2026-09-05

Chasing cell 1's apparent V4 regression found something larger. `_fd_check`
reports `median_rel_error` and `max_rel_error`, and every V4 number this document
carries is the **median** at 5 or 10 components. At 20:

| cell | margin | `X` | V4 median | **V4 max** |
| --- | --- | --- | --- | --- |
| 4 | **−8.83e-01** (decisive) | 4.89e-05 | 3.1e-06 | **6.2e-05** |
| 7 | −2.09e-04 (tie) | 2.14e-05 | 5.6e-07 | **3.8e-01** |
| 1, bisect state | +1.64e-03 (tie, invalid) | 6.20e-07 | 6.8e-07 | **4.5e-02** |
| 1, monoculture state | −1.64e-03 (tie, valid) | 6.21e-07 | 8.5e-05 | **9.1e-03** |

**1. Only cell 4 passes V4 on the max, and it is the only cell with a decisive
invasion margin.** The medians are all 5.6e-07 to 8.5e-05 and say nothing about
this; the tail is four to six orders worse on the tie cells. **"V4 passes wherever
the solve converges" is retracted** -- it was measured on medians, at a quarter of
this depth.

**2. The mechanism is the tie, and it is not a solver defect.** A feed
perturbation at a near-tie can push the community across the survivor swap, so the
"finite difference" differences two states on *different branches* rather than
approximating a derivative. `_fd_check` already guards the visible half of this by
skipping any component whose active set changed -- which is exactly why cells 1 and
7 report `n=19` of 20 and cell 4 reports 20.

**The other half of that explanation was wrong and is retracted.** It said the
surviving error is a component that stayed on one branch while `J` went
near-singular in the swap direction. `J` and `dy*/dc_feed` are both stored in
`steady_state.npz`, so this cost no solves to check (`20hm_bands/jcond.py`), and
**the analytic object is well conditioned at every converged fixed point,
tie cells included**: column-equilibrated `cond(J)` is 8.7e+02 to 4.4e+04 and
`max |S|` is 1.0e+02 to 4.2e+03. Neither tracks the margin -- the *smallest*
margin in the set, cell 8's +1.47e-05, has the **best** conditioning of all
(8.66e+02) -- and the smallest singular vector puts only 0.20-0.41 of its mass in
the abundance block, so it is not a swap direction either.

**3. So the invasion margin's magnitude predicts V4, though it does not predict
multiplicity.** Cell 4 at −8.8e-01 is clean, cells 7 and 1 at −2.1e-04 and
±1.6e-03 are not. That is the opposite of the earlier reading, where magnitude
failed to predict uniqueness and only the sign worked -- the two questions want
different things from the same number, and both are free.

**4. Cell 1's "regression" was not one.** The new fixed point is 125x worse on the
median and **5x better on the max** (4.5e-02 -> 9.1e-03). The 9.1e-07 -> 8.5e-05
comparison reported earlier was a 5-component median against a 20-component one on
a cell whose V4 is dominated by its tail. Monoculture seeding did not degrade V4.

**What M12's V4 half can actually claim:** V4 passes on cells with a decisive
invasion margin (1 of 3 measured, max 6.2e-05) and does **not** pass on near-tie
cells (max 4.5e-02 to 3.8e-01). Quote V4 as a max at 20 components with the cell's
margin beside it, or do not quote it.

**5. What it means downstream, which is the reason to care.** §8.4 and §13.6 both
differentiate through `c*`, and the conditioning above says **they can**: `S` is
finite and well conditioned everywhere measured, so the implicit derivative is
usable *within* a branch. What fails at a near-tie is not the derivative but the
**function** -- `y*(c_feed)` is discontinuous across the survivor swap, so a feed
step that crosses it lands on a different branch and the gradient that was correct
up to the boundary predicts nothing beyond it. V4's max is detecting that
discontinuity, which is why it is four orders worse than the median while the
Jacobian is fine.

That makes it the same shape as P21, where §13.2's medium designer walks out of
the design and the true LP does not grow: a locally-correct gradient plus a step
that leaves the region it was valid in. The remedy is the same too -- a trust
region, in the coordinate the boundary lives in -- and the boundary here is
announced for free by the invasion margin going to zero. **A steady-state
optimiser or sampler should refuse, or shorten, a step that would change the
survivor set.** Not built.

##### Multiple steady states: the answer is competitive exclusion, not a better solver — 2026-09-05

The session had been treating "several fixed points" as a numerical problem to be
attacked with better globalisation and more probing. Two measurements, both free
and neither needing a solve, say it is mostly not one.

**1. `k = 1`: one metabolite carries the whole growth gradient, at every fixed
point.** Counted from Head A's own analytic gradient at `c*`
(`20hm_bands/kres.py`), over states from n=2 to n=10:

| state | survivors | `k` at 1% of the top | top share |
| --- | --- | --- | --- |
| cells 4, 7, 1, 9, 8 (single-survivor) | 1 | **1** | **1.000** |
| cell 8's coexistence | 2 | 1 and **2** | 1.000 / 0.972 |

Hsu, Hubbell & Waltman (1977) prove that `n` species on **one** limiting resource
with monotone growth admit exactly one survivor, *globally* -- no bistability, no
basin structure, and the winner is the lowest R*. Tilman's R* theory bounds
coexistence by the number of limiting resources. At `k = 1` there is nothing to
enumerate: the apparent multiplicity is one equilibrium the surrogate cannot
resolve, not several the solver must find. The single coexistence state needs its
second resource, and has it at **2.8%** of that member's gradient -- inside the
model's own error.

**2. R\* is the right ranking statistic and `mu` at the feed is the wrong one.**
R* here is `theta*`, the feed scaling at which a member alone reaches `mu = D`:
one bisection per member, 50 right-hand-side evaluations, **no steady-state
solve**. Against `mu(feed)`, which is what the probe order used:

| cell | R* (theta*) | gap | R* winner | `mu(feed)` winner |
| --- | --- | --- | --- | --- |
| 4 | 0.1995 / 1.000 | **5x** | CR626927.1 | CR626927.1 |
| 3 | 2.791e-4 / 2.860e-4 | 2.4% | ABCC02 | ABCC02 |
| 1 | 0.1985 / 0.1988 | 0.15% | CR626927.1 | CR626927.1 |
| 7 | 0.1996 / 0.1997 | 0.05% | CP001726.1 | CP001726.1 |
| 2 | 4.544e-4 / 4.545e-4 | **0.02%** | CP001726.1 | **CP001820.1** |

They agree except on cell 2, and the **gap** is the cell's difficulty measured
before any solve: 5x on the one cell with a decisive invasion margin, 0.02-0.15%
on every near-tie cell. `--seed-mode monoculture` now orders its probes by R*.

**A claim made here and immediately walked back:** that R* "gets cell 2 right where
the solver gets it wrong". Its gap there is **0.02%** -- R* is not resolving that
cell either, and neither is the invasion margin at 7e-05. Cell 2 is unresolvable
at every statistic available, and the solver returning either member is
defensible. Re-running it under R* order confirmed this: the R*-preferred member's
basin leads to a *degenerate* state (margin −1, unconverged) and selection
correctly falls back to the other on residual. **Cell 2 is unchanged and should
stop being treated as a fixable failure.**

**What this refutes, before it was built.** Five methods were on the plan for
enumerating multiple equilibria and `k = 1` removes the motivation for all of
them: deflation (Farrell, Birkisson & Funke 2015 -- the standard way to find
distinct roots from one start), convex pre-screening of candidate survivor sets
(available because `{c : mu_i(c) >= D}` is convex when Head A is concave),
a Fischer-Burmeister/semismooth Newton reformulation that would delete the
active-set loop and its path dependence (Qi & Sun 1993), pairwise seed probes, and
the `2G` monoculture probing that R* replaces with `G` bisections. **Pairwise
probes were the one that had already been built and run: 10 probes on cell 8, and
they found nothing the singles had not** -- consistent with the theory, and removed.

**The caveat that would reopen all of it.** `k = 1` is partly Head A's structure
rather than biology: a max-affine head's gradient at a point *is* a single active
plane, so a top share of 1.000 is what the model class produces whatever the
medium does. Separating "these media are singly limited" from "this head reports
one limiter at a time" needs the LP, and `--roster` is how to ask -- an equilibrium
is one state, so the solves are affordable. If the true `k` is 2-3, cell 8's
coexistence is real, deflation becomes the right tool, and the convex pre-screen
becomes worth its complexity.

**What is worth building regardless: the analytic pool-block Jacobian.**
`d(dc/dt)/dc = -D I + sum_i X_i dz_i/dc`, and Head B is a JAX MLP, so `jacfwd`
gives it exactly where the code currently spends **one right-hand-side evaluation
per free metabolite** -- 355 of them per Jacobian. The growth rows are already
analytic (`_head_mu_rows`); this is the other block, it is pure implementation,
and it is what makes n=21 affordable at all. Independent of every question above.

##### Making a solve cheap: profile first, and the batch is not the single — 2026-09-05

The plan called for an analytic pool-block Jacobian via `jacfwd` through Head B.
**It is not the small change it looked like**: `mu_and_z` interleaves numpy with
JAX -- `np.maximum`, and an active-set NNLS in `_element_balance` -- so it is not
traceable end to end, and `jacfwd` needs that projection rewritten first. Two
cheaper things were found by measuring instead.

**1. 27% of every right-hand side was an allocation storm.** `cProfile` on
`rhs_surrogate`: Head A's `_mu` 54%, `_element_balance` **27%**, Head B's flux --
which had been the assumed bottleneck -- inside the remaining 19%. Almost all of
the 27% was **1100 `np.asarray` calls per evaluation**: the masked element matrix,
the `z_scale` metric and the dual's `Q` were rebuilt on every call although they
depend only on the organism. Caching them per organism takes the right-hand side
from **23.9 ms to 17.1 ms** with `dc` and `mu` bit-identical, and it helps every
caller, not only the steady state.

Built in `__init__`, not lazily: a lazy cache of `_E`/`mask`/`z_scale` goes
silently stale if any of them is reassigned, which is exactly what the unit tests
do and how the first version was caught.

**2. Batching the Jacobian's finite differences, 2.3x.** Both heads already carry
a medium axis -- `mu_and_z` merely passes `B = 1` -- and evaluating them one
medium at a time is nearly all JAX dispatch. Measured on Head A: **11.6 ms for one
medium against 0.143 ms each for 64 at once, 81x**. The finite-differenced
Jacobian is exactly that shape, one medium per free metabolite differing in a
single coordinate, so `Surrogate.mu_and_z_batch` / `rhs_surrogate_batch` feed a
`residual_b` that `_jacobian` uses when it is available.

Per Jacobian **3.96 s -> 1.70 s**, and end to end on the three cells with known
answers: cell 1 **101 s -> 37 s**, cell 4 **448 s -> 104 s**, cell 7 **258 s ->
49 s**, every survivor, margin and convergence flag unchanged (cells 4 and 7 shift
residual 5.3e-08 -> 1.5e-07 and 7.3e-08 -> 1.3e-07, both far under tolerance).

**The trap, and it would have shipped as a 4.6x win.** XLA does **not** compute a
batch of `n` in float32 the way it computes a batch of 1: `z` differs by ~7e-5
between the two paths, and `mu` by ~9e-7. The first version differenced batched
perturbations against an `r0` computed by the *single* path, which put that
discrepancy in the numerator over a step of `1e-3 c` -- **the Jacobian was wrong
by a relative 7e+07**, larger than the derivative being measured, while running
4.6x faster. The unperturbed medium now goes in the same batch so both sides come
from one kernel. Same family as the finite-difference step bug in §13.4: the error
is manufactured by a small denominator, not by the model.

After the fix the two Jacobians agree at **cosine 0.9999999999**. Individual
entries still differ by up to 10% where the response sits at the float32 noise
floor -- those entries are ill-determined in the loop version too, so this is the
same accuracy rather than new error, and it is worth knowing before anyone reads a
single `J` entry as meaningful.

##### `k = 1` confirmed against the LP, and the fixed point is true to <1% — 2026-09-05

The `k = 1` measurement came with a caveat that could have reopened everything: a
max-affine head's gradient at a point **is** one active plane, so counting
limiters from Head A may measure the model class rather than the medium. The LP
settles it (`20hm_bands/true_k.py`): set §3.3's Michaelis-Menten bounds from the
converged `c*`, solve, and count exchanges whose bound binds with a non-dust
reduced cost.

| state | survivor | `mu_LP(c*)` | `D` | rel. error | `k_LP` | limiter |
| --- | --- | --- | --- | --- | --- | --- |
| cell 4 | CR626927.1 | 2.446 | 2.448 | **0.08%** | **1** | `EX_k_e` |
| cell 7 | CP001726.1 | 1.066 | 1.068 | **0.19%** | **1** | `EX_trp__L_e` |
| cell 1 | CR626927.1 | 0.1524 | 0.1538 | **0.9%** | **1** | `EX_k_e` |

**1. `k = 1` is the medium, not the head.** Two independent derivations agree, and
the top limiter carries the whole dual in each case. So competitive exclusion
applies for real: one limiting resource, one survivor, chosen by R*. The
enumeration branch -- deflation, convex pre-screening, the semismooth rewrite --
stays closed, and now on evidence rather than on a model artefact.

**2. The surrogate's fixed point is a fixed point of the true LP, to 0.08-0.9% in
growth rate.** This is the §13.4 accuracy statement that was missing, measured
where the use case actually evaluates rather than on held-out design media. §13.7
called the steady state "the most exposed use case" and `reach` at `c*` of
1.0-5.5, against a held-out ~0.10, made that look severe. It is not: at the state
itself the growth rate is right to under a percent. **`reach` measures distance
from the design, and this is a direct measurement of the thing `reach` was a proxy
for -- prefer it wherever a fixed point exists to solve.**

**A trap that returned zero rather than a wrong answer, which is the good kind.**
cobra's `solution.shadow_prices` is indexed by **metabolite**; the sensitivity of
growth to an *exchange bound* is `solution.reduced_costs`, indexed by reaction.
Using the former found no binding exchange at all and reported `k = 0` -- visibly
broken, rather than a plausible-looking count.

##### Cell 10 (n=21), finally measured — 2026-09-05

Three earlier attempts exceeded a one-hour wall (uncapped, 2 probes, 1 probe). At
~4x cheaper per solve it completes in **17 minutes** with the default four
R*-ordered probes:

| cell 10 | residual | survivor | margin | probes |
| --- | --- | --- | --- | --- |
| `--seed-probes 0` (the old bisection) | **9.9e+00** | — | −1.00 | 0 |
| R*-ordered monoculture seeding | **3.87e-03** | ABYJ02 | +2.26e-05 | 4 |

**Three orders, and the same story as cells 2 and 3**: the bisection commits to a
survivor and collapses the pool; a monoculture seed does not. It does not
converge, and the R* values say why --

    R* = [3.4771e-02, 3.4771e-02, 1.0000e+00, 3.4772e-02, 3.4776e-02, 1.0510e-01, ...]

**four members tied to within 0.015%.** The early stop never fires because no
strictly valid state exists to find, all four probes run, and the final margin is
a tie at +2.3e-05. So the roster-scale cell is **not** a size failure either: it is
the same sub-resolution tie as cells 1, 2 and 7, with more members inside it. That
is now the single explanation for every unconverged cell in the gate.

##### What is next, in order

| # | Job | Why here |
| --- | --- | --- |
| 1 | **The near-tie cells** — cells 1, 6, 8 are within 0.2% of `D` of coexistence and the anti-cycling ban decides them | Now that `converged` gates on `invadable`, these read as failures, which is honest but not useful. A tie is a real ecological answer (neutral coexistence); the loop needs a way to *return* it instead of banning one side. Smallest version: on a re-admission that is within tolerance of `D`, solve the two-survivor system once rather than banning |
| 2 | **Cell 9 at `mu - D` = +41** | Not a tie and not the pool collapse: an excluded member growing 4.7x faster than the dilution rate at the returned state. The active set is simply wrong there, and it is the one cell where the ban costs an order of magnitude rather than a rounding error |
| 3 | Per-cell `--jac-temp` / `--d-steps` / `--ptc` | Three flags now, each winning some cells and losing others, none a default. A short grid over the ten cells would at least say whether *some* setting converges each one |
| 4 | Keystone members | `--organisms` minus one, N runs, **no code**. Cheap, and it needs only the cells that converge |
| 5 | Re-seed abundances per continuation rung | Cell 2's ladder converges only on its top rung, whose answer is `X ~ 0` by construction, and carries that state down. The single-`D` warm start already has the NNLS that would fix it |
| 6 | Whether cells 2, 3 and 10 have a fixed point at all | Still open, and still the branch that would stop this being a numerical question. Deprioritised because integration is expensive (17 ms per right-hand side over a 357-dimensional stiff system) and because cell 3 reaching 4.5e-5 under continuation is evidence *for* existence |

**What would change this plan.** If cells 2 and 3 have no fixed point under the
surrogate, the question stops being numerical and becomes §13.7's: an equilibrium
is a drawn-down medium, `reach` there is 5.2 and 5.5 against a held-out ~0.10, and
Head B's `z` at that distance need not admit a steady state at all. The `--roster`
and `--mix-*` paths exist precisely so that one state can be bought exactly.

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

**Revised 2026-09-03, after §8.6f.** Head A is now exact where §8.1 evaluates it
(`mu_rel_median <= 5e-4` on all 30 cells, 3e-5 at both horizons), so every row
below is a Head B statement, and two things changed the verdicts:

| Use case | Head B in the loop? | Revised verdict |
|---|---|---|
| §13.2 growth maximisation | **no** — `Surrogate(behaviour_dir=None)` | **unaffected by all of §8.6f.** 17/20 true improvements, median optimism 0.3% |
| §13.3 minimal medium | **no** | unaffected; V6 is blocked on essentiality, not on Head B |
| §13.1 structure, ordering, cross-feeding | yes | **safe at a 4-doubling horizon** — recall 1.00, median flux cosine 0.995. **Not** safe per link in a starved culture: p05 cosine reaches **−0.44** below depth 0.1 |
| §13.1 quantitative yield / batch endpoint | yes | **not safe at a long horizon** — 4.1% overall at 8 doublings, worst cell 0.70 — and §8.6f's round 4 shows **more Head B accuracy does not fix it** |
| **§13.4 steady state / SteadyCom / §8.4** | yes | **the most exposed use case, and the next milestone.** A steady state *is* a drawn-down medium — the regime where Head B is worst — evaluated inside every Newton iteration. M5's shallow numbers must not be assumed to transfer; measure at the equilibrium |
| §13.5 interaction magnitude | yes | exploratory, unchanged |
| §13.6 posterior | yes | still blocked, but the error model now has a candidate: the reach proxy as a distance-aware nonconformity score, calibrated on community-regime states (never on held-out design media) |

**Amended 2026-09-04, after §8.6g.** Two rows move, and both move because the LP
fallback makes accuracy something a use case can *buy* rather than something the
head either has or lacks.

| Use case | Amendment |
| --- | --- |
| §13.1 quantitative yield / batch endpoint | **Reachable, at a price.** `--fallback-depth 0.9` takes the 8-doubling gate from 0.041 to 0.028 overall and n=21 from 0.054 to **0.015**, for 24.6% of the member-steps a full LP costs. Still not 1%, and P23 still stands — but "not safe at a long horizon" is now a statement about the LP budget, not about the head |
| §13.6 posterior | The nonconformity score has a **second** candidate, and it is the better one: predicted depletion **depth**, which is the only *per-step* predictor measured to beat random (lift 3.0x, against the reach proxy's 0.9x). Reach stays the per-cell instrument. Both ship in `cfs simulate` |
| §13.4 steady state | Unchanged and still the most exposed — but the fallback applies there too, and an equilibrium solve visits few states, so the price of truth at those states is low. Measure it at the equilibrium, as this row already says |

**And a caution the §8.6g pass adds to P25.** Every one of the label rounds in
this project has been scored against the checkpoint it started from, which
conflates the new rows with a fresh fit. Measured at the 8-doubling gate, a
matched retrain with **no new rows** moves the mean 0.085 -> 0.143 and the max
0.572 -> 1.866. Rounds 3, 4 and 5's "null" verdicts are safe in direction — none
of them helped — but their magnitudes are inside that noise, and any future round
must budget a matched control.

### 13.8 New pitfalls

| ID | Pitfall | Symptom | Solution |
|---|---|---|---|
| P20 | Deterministic surrogate inside a likelihood | Posterior far too narrow; SBC ranks pile up at the edges | Per-organism residual model from the held-out set, widened by the M5 replicate spread. No chain before it exists |
| P21 | The designer walks out of the design | Spectacular objective, LP disagrees; medium far from any training medium | Trust region on the §6.3 nearest-training-medium distance — the same metric that diagnosed P18 — plus V5 at every reported optimum |
| P22 | An objective on flux *magnitude* | Inherits Head B's weakest axis while the diagnostics (cosine, sign agreement) look fine | Prefer direction- and structure-valued objectives; label magnitude-valued results exploratory |
| P23 | Optimising a batch-culture endpoint | The answer flips under changes that improve the right-hand side on every measure — the endpoint turns on which metabolite empties first | Optimise rates, or a chemostat steady state. Never a batch endpoint |
| P24 | A relabel that improves every held-out metric and breaks composition | Worst grad cosine, value R², per-metabolite coverage and M11 all improve; §8.1 regresses 17x at n=21 | Held-out media come from the *same design that changed*, so they cannot see it. Score every design change on a **community-regime held-out set** (§8.5). The stratum-budget reading of P24 was measured and is wrong — see §4.3 |
| P26 | Reading a fine-tune or a label round against its starting checkpoint | Every metric improves and the change looks earned; the same numbers appear with the new term at weight zero or the new rows absent | Two ablations, both measured to matter here: run the new loss term at **weight 0** (the trajectory term's entire gain was the label term's), and score a label round against a **matched retrain** (a fresh fit alone moves the 8-doubling mean 68% and the max 3.3x) |
| P25 | Tuning a training distribution against a proxy metric | The proxy moves exactly as designed, three times, and the downstream number does not follow | Co-limitation count, near-onset count, NN-distance in `x` and per-metabolite limiting rows are all refuted as predictors of §8.1 (§8.5). Do not spend a 5 h relabel on a metric that has not first been shown to correlate with the composition on runs already on disk |

### 13.9 Milestones

| M | Deliverable | Gate |
|---|---|---|
| M9 | `cfs simulate`, batch + chemostat | **done 2026-08-30**; agrees with `cfs community`'s surrogate path on `D = 0` |
| M10 | §13.2 growth maximisation, convex solver | Optimum survives V5 round-trip on 20 cases — **built 2026-08-30; 19/20 at the default trust region, 20/20 at 0.25 and at 1.0 decades.** Median true gain +2.2%, median optimism 0.3%. The one failure is a `mu = 2.0` start medium, the head's known weak band; it is not monotone in the trust radius. **Under an additive trust region 3/20 collapse to `mu_true = 0`, and under none at all 2 of the first 4** — P21, and the mechanism is zeroing an essential trace metabolite |
| M11 | §13.3 static minimal medium | **built 2026-08-30; the essentiality blocker is closed 2026-08-31, V6 still short.** `cfs minimal-medium`: convex penalty solve + a greedy cardinality prune, one case per medium draw. **Head A cannot represent essentiality** — knocking a trace metal (`EX_cobalt2_e`, `EX_cu2_e`, `EX_mn2_e`, `EX_zn2_e`) out of a rich medium takes the true LP to `mu = 0` and moves the head by <1%, 6 of 37 free metabolites on a 3-member community. Unrestricted, the program exploits exactly that: 273 -> **41** components with every surrogate floor satisfied and `mu_true` 55/70/38 -> **0/0/0**. With the lethal singles pinned from the models (`--keep-essential`, default; one FBA per free metabolite, a static property of the GEM), 273 -> 251 and 2 of 3 members clear a 0.5 floor under the LP, the misses being 0.489/0.485 — i.e. ~2% short — and one real failure at 0.334 on the community's slow member (`mu_true` 3.5 against 55 and 70), Head A's known weak low-`mu` band. **The cause is `SamplingConfig.log10_lo = -4`**: the trace metals' limiting regime is at `c/Km ~ 1e-9..1e-6`, outside the probe's bracket, so the probe omits them, `band_scales` defaults them to 1.0, the design never makes them scarce, `_kink_scale` defaults `x_scale` to 1.0 and the head has no resolution left in that coordinate. The four missed essentials are exactly the four `"source": "default"` bands in the sidecar. **Fixed by `probe_lo = -12` (§4.7) and a relabel: `n_missed_essential` 6 -> 0**, and unrestricted the design no longer collapses the LP (2/3, 0/3, 3/3 members clearing the floor, worst true fraction 0.436 against 0.000). V6 still does not pass at a 0.5 floor — 0.491 / 0.436 / 0.512 — so what remains is a few-percent accuracy question, not a structural one |
| M12 | §13.4 steady state + stability + invasion | V4 passes; Newton failure rate logged and < 1% — **built 2026-09-04**, `cfs steady-state`: coexistence from the active set, stability from the `(c, X)` Jacobian's eigenvalues, invasion from `mu_j(c*) - D`, and `dy*/dc_feed` from one extra solve. `--roster` adds an LP residual with a surrogate Jacobian, and `--mix-mu-rel` the hybrid that actually converges. **Measured over the roster 2026-09-05: V4 does NOT pass in general — those 5.6e-7 to 3.1e-6 figures are *medians* at 5 components, and at 20 the max is 6.2e-05 on the one cell with a decisive invasion margin but 4.5e-02 to 3.8e-01 on the near-tie cells, where a feed perturbation crosses the survivor swap and the difference quotient spans two branches; the Newton failure rate is 60% against the 1% gate, 4 of 10 cells returned a state an excluded member can invade, and the default warm start returns a *strictly invadable* state on 2 of the 4 converging cells — seeding from each member's monoculture instead finds the valid one, and takes the two hardest cells from residual 10 to 1e-5** -- so M12 does not pass. The failures are **not** the line search (a trust region is null) and not size (the 5-member cell converges where three 2-member ones fail). **The `reach` at `c*` is 1.0-5.7 and does not separate converged from failed**, so §13.7 is right that this is the most exposed use case — but an equilibrium is *one* state, so `--fallback-depth`'s LP is cheap here in a way it is not along a trajectory |
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
