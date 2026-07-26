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

- Sample `log10(c)` uniformly over roughly `[-4, 1]` relative to `Km`.
- **Weight 40% of samples below `Km`.** A uniform log design underweights
  exactly the region D9 cares about.
- Include the all-but-one-depleted corners explicitly: for each `m ∈ A_i`,
  a batch with `c_m` swept to zero and others rich. These pin down the
  single-limitation facets of the value function.
- Sobol sequences within each stratum, not uniform random.

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

---

## 7. Phase 4 — training

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
| V5 | Round-trip: true LP at composed optimum, 100 cases, report the tail | After Phase 5 | P4 |
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
| M4 | Head B trained, alpha sweep validated | V3 passes |
| M5 | dFBA composition | Trajectory matches COBRApy dFBA to 1% |
| M6 | Newton equilibrium + implicit gradients | V4 passes |
| M7 | Minimal medium, surrogate vs exact MILP | V5, V6 pass |
| M8 | SteadyCom / MICOM framings | Agreement with reference implementations |

M1 is new and comes before any training. It is a two-day job and it determines
the shape of your entire label set.

---

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
│   │   ├── behaviour.py           # Head B
│   │   ├── stacked.py             # §6.1 vmap harness
│   │   └── train.py
│   ├── compose/
│   │   ├── residual.py
│   │   ├── framings.py
│   │   └── solve.py               # Optimistix wrappers
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
