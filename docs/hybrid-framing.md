# The LP/surrogate hybrid — what it is, who else built one, and whether to reframe it

Written 2026-09-06, after M9–M14's first pass. Companion to
[`reading-map.md`](reading-map.md), which maps the *heads*; this maps the
**composition of head and solver**, which is now a distinct object with its own
literature.

The prompt: three use cases shipped with an LP in the loop, so is the project
still "a surrogate that replaces FBA", or is it better described as *a derivative
estimator that makes gradient-based optimisation of a metabolic community
possible*, with the LP retained as the value oracle? The short answer is that the
derivative framing is **half right, and the wrong half is the headline** — but
naming the other half correctly unlocks three bodies of method that fix limits
this repo is currently paying for by hand.

---

## 1. What was actually built — three hybrids, three triggers, one pattern

| Where | Trigger | What the LP supplies | What the surrogate supplies | Measured |
| --- | --- | --- | --- | --- |
| `cfs community --fallback-depth 0.9` (§8.6g(4)) | predicted depletion depth `mu_hat(t)/mu_hat(0)` | the member's whole `(mu, z)` at that step | every other member-step | 8-doubling log-X 0.041 → **0.028**, n=21 → 0.015, at a **24.6%** fire rate |
| `cfs steady-state --roster --mix-mu-rel/--mix-z-rel` (§13.4) | disagreement between LP and head at the iterate | the residual, and **exact dual rows of `J`** for the mixed members | the rest of `J`, and every unmixed member | matches or beats surrogate-only; 2.3× fewer Newton iterations at 3–5% LP usage |
| `cfs minimal-medium --lp-repair` (M11) | post-hoc: any member below its growth floor | the true `mu` at the designed medium | the whole design | **V6 4/4**, worst case +2 components in 370, 94 LP solves |

Same pattern each time: **the surrogate proposes, the LP disposes**, on a
cheap trigger that fires on a minority of states. Three hand-rolled triggers,
three hand-rolled acceptance rules, no shared theory, no convergence statement.
That is precisely the gap the literature in §4 fills.

Note also what is *not* in that table. §13.2 growth maximisation and §13.3's
convex program run with **no Head B and no LP at all**, and they are the two use
cases that meet their gates. Whatever reframing is adopted has to keep them.

---

## 2. The reframe, tested: is this "primarily a Jacobian estimator"?

### 2a. Against — the LP's first derivative is free, and now so is its full Jacobian

`mu_max` is an LP value function in its RHS, so the optimal dual **is** the
gradient, at zero marginal cost from the same solve. This repo already uses that:
`steady._head_mu_rows`' analytic rows and the LP's chain-ruled duals agree to
**7 significant figures** (5 122 829 vs 5 122 827.5), and the exact dual rows were
"the durable win" of the M12 solver pass.

Worse for the framing, the *behaviour* Jacobian is no longer expensive either.
**Chapman, Kratochvíl, Ebenhöh & Wilken, _Algebraic differentiation for fast
sensitivity analysis of optimal flux modes in metabolic models_ (Bioinformatics
2025; `DifferentiableMetabolism.jl`)** implicitly differentiates the KKT system of a
pruned GEM and returns `∂v_i/∂p_j` — every flux against every parameter, uptake
bounds included — for **one linear solve** after the LP, ~4× faster than finite
differencing, benchmarked on 342 yeast models and explicitly supporting community
models. It requires a unique optimum, which their pruning theorem provides and
which **D4's elastic net provides here for the same reason**.

So a paper claiming "a neural estimator of metabolic Jacobians" is contested by a
2025 method that computes them exactly, faster than FD, in the same setting. Do
not lead with it.

### 2b. For — three things the exact derivative is not

1. **It is not defined across an active-set change.** The 2025 paper says so
   itself: pruned models "cannot capture larger metabolic switches", and the
   construction fails when "infinitesimal parameter changes cause switching
   between optimal solutions". That is exactly the regime this project's own
   measurements keep landing in — the batch endpoint turning on *which metabolite
   empties first* ([[rhs-accuracy-does-not-buy-the-endpoint]], six instances),
   and V4's max blowing up to 3.8e-01 on near-tie cells "because a feed
   perturbation crosses the survivor swap and the difference quotient spans two
   branches".
2. **There is no second derivative at all.** `mu_max` is piecewise linear in `u`,
   so its Hessian is zero inside every piece and undefined at the kinks (P3). The
   §13.4 Newton solve needs `sum_i X_i H_i`; the LP cannot supply one entry of it.
   The smoothed head can, and `--gm-temp` is an explicit knob on how much.
3. **It is local, and gives no global structure.** Concavity-by-construction is
   what makes §13.2 and §13.3 *convex programs with a unique optimum*. No amount
   of exact pointwise sensitivity buys that.

### 2c. The framing that survives both

> **A smooth, globally concave, everywhere-defined relaxation of a piecewise-linear
> LP value function, supervised on that LP's own duals — with the LP retained as a
> value oracle wherever the relaxation is untrusted.**

"Relaxation/smoothing", not "estimator". The distinction is not cosmetic: it names
the right competitors, and they are not ML papers.

* **Höffner, Harwood & Barton — DFBAlab / lexicographic LP** makes the embedded
  LP's exchange fluxes *unique* so the DAE is well posed. Same problem as D4,
  different answer (lexicographic tie-break vs elastic net); the reading map
  already flags Jones & Kvasnica's lexicographic perturbation for exactly this.
* **The interior-point reformulation of dFBA** (Comput. Chem. Eng. 2019) and the
  **NLP/KKT reformulation** (Comput. Chem. Eng. 2022) replace the embedded LP with
  a *smooth* algebraic system so an ODE/NLP solver can integrate and differentiate
  it. These are published, non-learned solutions to the same smoothness problem
  the surrogate solves by fitting — and the second one solves the whole
  design-over-a-trajectory problem as one MPCC. **Any speed or differentiability
  claim has to be stated against these, not against a cold simplex.**

The honest novelty claim moves accordingly. Not "we estimate the Jacobian"
(contested), and not only the reading map's three-way intersection, but:

> The relaxation is *concave and monotone by construction*, so the applications
> built on it are convex programs rather than local searches — and its
> approximation error is one-sided by the same structure, which makes it a
> **certificate**, not just a point estimate (§5).

---

## 3. What the LP-in-the-loop buys, priced against the right baseline

The fallback's headline is 24.6% of member-steps. That is a **4× saving against
solving everything**, and CLAUDE.md already states the comparison correctly: the
baseline is *Minimizing the Number of Optimizations for Efficient Community dFBA*
(bioRxiv 2020), which reuses the LP basis across timesteps, not a cold solve.

The reframing point is that a hybrid whose cheap component is only ever a
*veto-able guess* is strictly worse than one whose cheap component **makes the
expensive component cheaper**. Which brings the third framing:

---

## 4. Three framings that import published machinery, ranked by what they fix

### A. Trust-region model management — built and measured 2026-09-06

**Alexandrov, Dennis, Lewis & Torczon** (first-order-consistent corrections) and
the process-systems version, **Eason & Biegler, _A trust region filter method for
glass box/black box optimization_** (AIChE J 2016; 2018 sequel; Hameed et al. 2026
adds Hessian information). The result: if the cheap model is corrected to match the
true model's **value and gradient at the trust-region centre**, and the region is
managed by the usual ratio test, the loop **provably converges to a KKT point of
the true problem** — with no global accuracy requirement on the surrogate.

Built as `science.growth.trf`, `cfs maximise-growth --trf N`, default off. Two
things make it unusually cheap here: **the LP is a first-order oracle** (the dual is
the gradient, so one FBA per iteration supplies both halves of the consistency
condition), and the correction is **affine**, so the model stays concave and the
subproblem stays §13.0's convex program.

| 20 V5 cases | median true gain | max | better/equal/worse | max optimism | LPs |
| --- | --- | --- | --- | --- | --- |
| single ascent (baseline) | **+2.34%** | **22.8x** | — | 0.0730 | 0 |
| `--trf 60` | +2.12% | 6.8x | **7 / 10 / 3** | **0.0071** | 219 (median 6) |

**Two model forms were measured** (`--trf-mode`), and the difference is the result:

| 20 V5 cases | median true gain | max | better/eq/worse | max optimism | LPs |
| --- | --- | --- | --- | --- | --- |
| single ascent (baseline) | +2.34% | 22.79x | — | 0.0730 | 0 |
| `shift` — additive first-order correction | +2.12% | 6.79x | 7/10/3 | 0.00712 | 219 |
| **`bundle` — `min(head, LP tangents)`** | **+2.35%** | **23.15x** | **7/11/2** | **0.00686** | **70** |

**The bundle dominates the baseline on both axes at once** — same-or-better true
gain on 18/20, better median *and* better max, a 10.6x tighter optimism bound, at a
median of 3 LP solves per case. Every tangent is a supporting hyperplane of a
concave function, so the min is concave and the subproblem stays convex; cuts from
**rejected** steps are kept, which is what a bundle is for.

**Why the textbook additive correction fails, which is the more useful finding.**

1. **The LP's gradient at a kink is a subgradient *selection*.** `mu_max` is
   piecewise linear in `u`, so the dual is one element of the subdifferential. With
   the model matched to the LP in value *and* gradient at the centre, `rho -> 1` as
   the step shrinks — unless the function is not differentiable there. Instrumented
   under `shift`: radius shrank 16x, `predicted` tracked it exactly, `actual` stayed
   **pinned at 0.0046**. TRF correctly refuses and halts *at the kink* (`mu_true`
   9.29 against the ascent's 22.0). The smoothed head walks through because
   smoothing averages both sides. **This is §7's mollification argument as an
   optimiser failure, and it is the strongest single argument for the relaxation
   framing over the estimator framing: what the surrogate offers over the exact
   oracle is not accuracy, it is a usable direction at a corner.** The bundle fixes
   exactly those cases — 6.79 -> 23.15, and 0.0064 -> 0.0334.
2. **An additive correction is the wrong form at eight decades of gradient range.**
   `dmu/dc` reaches 1e8 on the ions, so `shift` oscillates between norm ~1 and ~5e6
   and swamps the concave head (`predicted` 129 against an actual 1.2). Alexandrov's
   multiplicative beta correction would be the classical alternative; the bundle
   sidesteps the issue entirely, and its rejects fall from 6-18 per case to 0-1.

**What still limits it — the inner solver, not the model.** The two residual losses
were first blamed on the head reading below the truth at a designed medium; §5's
measurement **refutes that** (the head is a valid upper bound at all 20 bundle
optima and all 20 baseline optima, both loss cases included). A cut is a valid upper
bound too, so `min(head, cuts)` is one everywhere and its maximum over the region is
at least the true maximum — the better point was *inside* the model's feasible set.
**The subproblem solver failed to find its own model's maximum**: projected
subgradient ascent with backtracking stalls at the nonsmooth `min`'s kinks. The
bundle fixes the *outer* kink and introduces an *inner* one, which is why bundle
methods solve their subproblem as an LP/QP over the epigraph. A softmin over the
cuts is the cheap fix in this codebase's idiom. Not built.

**One transferable trap.** The textbook expansion rule grows the radius only when
the step reaches the trust-region *face*. Here the **budget** binds, so steps are
interior, expansion never fires, and the radius ratchets down until the loop halts
with gains remaining (9 of 20 cases ended at exactly six halvings). Expand on the
ratio test alone and cap the radius.

**Verdict.** P21 becomes the mechanism rather than a pitfall, every reported optimum
is LP-verified at every step, and **the M3 gate no longer constrains §13.2**: the
model is exact at the centre by construction and the bundle keeps it honest away
from it.

**§13.3 inherits the bundle but not the trust region — `cfs minimal-medium --cuts N`,
measured 2026-09-06.** There the surrogate is in the *constraints* and the objective
is exact, so the right algorithm is Kelley cutting planes rather than a filter TRF:
each LP tangent is an upper bound on the concave `mu_true`, so requiring it to clear
the growth floor is necessary for the true constraint and the model tightens
monotonically toward the true feasible set. On 4 communities x 3 draws it is a
**second route to V6, not a replacement for `--lp-repair`**: cell 6 passes on cuts
alone with no repair at 247 components against the repair's 250/252 (the optimality
claim landing), cells 7 and 8 are the correct null, and cell 9 costs 406/404
components against 372/375. Off by default.

**And one transferable pitfall.** A cut carries no information at a **dead** member:
at `mu_true = 0` every dual is zero, so the tangent is `0 >= target` — flat and
satisfiable nowhere. The model goes infeasible and the design walks back toward rich
(cell 9: 369 -> 402 components, still failing). `cut_loop` skips dead members and
hands them to the repair. Anyone applying cutting planes to a constraint whose
function can reach zero needs that guard.

### B. Inexact Newton — the §13.4 failure is textbook, and already half-diagnosed

CLAUDE.md: "a pure LP residual with a surrogate Jacobian does not converge (0–13
iterations, then no descent direction) — the textbook inexact-Newton failure".
That is **Dembo, Eisenstat & Steihaug (1982)** and the theory says more than that
it fails:

* convergence is governed by the **forcing term** `eta_k` (Eisenstat & Walker
  1996), and the right design is to *loosen* it far from the solution and tighten
  it as the residual falls. `--mix-mu-rel` is a fixed forcing term. Making it a
  schedule is a few lines and is the standard answer to "this solver has no single
  setting" (currently three per-cell flags: `--jac-temp`, `--d-steps`, `--ptc`).
* **Jacobian-free Newton–Krylov** costs one rhs evaluation per Krylov iteration
  against 230 finite-difference columns — CLAUDE.md already calls it a
  `scipy.optimize.root` one-liner and has not done it. With the batched rhs
  (`rhs_surrogate_batch`, 2.3×) the arithmetic now strongly favours it.
* the same literature warns about exactly the bug that was found: mixing an exact
  residual with an inconsistent Jacobian. The fix used (exact dual rows for
  exactly the mixed members) is the principled one.

Note the interaction with §2b(2): JFNK needs only Jacobian-*vector* products, and
the surrogate supplies those analytically and smoothly where the LP cannot supply
them at all. **This, not "Jacobian estimation" in general, is the derivative
claim that holds up.**

### C. Semi-amortization — make the surrogate pay for the LP it triggers

**Amos, _Tutorial on Amortized Optimization_** (FnT ML 2023) already names Head A
"fully-amortized, regression-based". The fallback makes the system
**semi-amortized**: a learned initialisation plus a few true solver steps. The
literature's point is that a semi-amortized system should hand the solver
something, not merely stand aside — and the standard currency is a **warm start or
a predicted active set** (Bertsimas & Stellato; the 2025–26 GNN warm-start work
for active-set QP/LP solvers).

Here that is nearly free and untested:

* the fallback already knows *which* member-state it is about to solve;
* Head A's argmax over planes is a predicted limiting set, and Head B's clamp
  already lands on the MM bound for every tight entry (`dual ⇒ flux on the bound`
  0.996–1.000 on the labels);
* **the B6 negative result does not close this.** B6 asked whether the predicted
  limiting set *discriminates failing states* (Hamming 0.0 — it does not) and
  whether a limiting set is a critical region (it is not, "because the shards
  record exchange duals and not the LP's optimal basis"). Whether it is a good
  *warm start* is a different question with a different success criterion
  (iterations saved, not variance explained), and it was never asked.

Cheap experiment, no new modelling: store the optimal basis alongside the duals in
one label shard, then measure simplex iterations from a surrogate-derived basis
against a cold start on the fallback's own fired states. If it lands, the 24.6%
fire rate is priced at well under 24.6% of the LP cost, and the hybrid stops being
a compromise.

### D. (Weaker) Surrogate gradients / straight-through estimators

The ML name for "true value forward, smooth derivative backward". It is a real and
widely used pattern (spiking networks, quantised networks) and it describes
`--mix-*` accurately. It is listed last because it is a *heuristic* family with
thin guarantees, and A and B are the same idea with theorems. Use it for
exposition, not as the method's home.

---

## 5. The idea the reframe actually generates: the two heads bracket the truth

Framing the head as a **relaxation** rather than an estimator makes its error
one-sided, and one-sidedness is a certificate.

* A min of supporting hyperplanes of a concave function is an **upper bound
  everywhere** — this repo has already proved and repaired that invariant
  (`--gm-repair`: training rows under-predicted 48.1% → 0.0%). So the repaired
  Head A gives `mu_hat(c) >= mu_true(c)` — a **dual/outer bound**, exactly SDDP's
  upper bound, which the reading map's §3a already anchors.
* Any **feasible** flux vector gives a lower bound on `mu_true`. Head B predicts
  one that is nearly feasible, and the DC3-style machinery to make it feasible
  (the MM clamp, the `E z <= 0` min-norm projection) is already built and on by
  default.
* The gap between them is a **computable, certified error bar** on `mu` at any
  state — and SDDP's stopping rule is precisely "stop when upper minus lower is
  small".

That would replace all three hand-rolled fallback triggers (depth, `mu`
disagreement, floor violation) with one principled quantity, and it is the missing
piece of §13.6's error model (P20) — a bound, not a fitted residual distribution.

**Measured 2026-09-06 (`20hm_bands/bound_gap.py`), and the upper half holds where
it matters.** 182 (point, organism) pairs against the true LP:

| point set | n | valid (`mu_hat >= mu_true`) | median gap | median rel | worst rel |
| --- | --- | --- | --- | --- | --- |
| held-out design media | 72 | 0.986 | 0.00226 | 4.2e-04 | **-8.3e-03** |
| §4.3 community-regime draws | 26 | **1.000** | 2.3e-04 | 4.2e-06 | 1.6e-06 |
| §13.2 designed optima | 20 | **1.000** | 3.7e-04 | 7.8e-06 | 1.3e-08 |
| §13.3 minimal media | 63 | **1.000** | 0.464 | 1.3e-02 | 1.7e-05 |

1. **109 of 109 off-distribution points are valid**, and the single violation in the
   whole set is on *held-out design media* (-0.83%). The bound is **safer away from
   the design, not less safe** — which is what a max-affine upper bound should do,
   since it is loosest where no tangent is nearby.
2. **It underwrites both design programs.** §13.2's bundle TRF and §13.3's cut loop
   each need `min(head, cuts)` to be a valid upper bound, and it is, at exactly the
   point sets they generate. That premise is no longer an assumption.
3. **It explains both use cases' error directions from one fact.** For §13.2's
   *maximisation* an upper bound means the reported optimum is optimistic — measured
   optimism 0.3-0.7%. For §13.3's *constraint* `mu >= target` it is the unsafe
   direction: the model can be satisfied while the truth is not, which is V6's
   failure mode and why `--lp-repair` and the cut loop exist. The gap size grades
   it — the minimal-medium set is the loosest (1.3% median, up to 14% on one
   member), which is exactly where the design is most aggressive.

**The honest cost, stated up front:** Head B predicts *exchange* fluxes only, so
turning it into a certified primal bound needs a completion to a full flux vector
— an LP *feasibility* problem. Cheaper than the FBA optimisation and warm-startable
(§4C), but not free.

**The lower half is built and measured — 2026-09-08 (`cfs.science.growth.mu_lower`,
`20hm_bands/bracket.py`).** The completion does not have to be solved separately:
**restrict each exchange's uptake to what Head B predicts and hand the network back
to the LP**, which completes the vector itself. Tightening a bound can only shrink
the feasible set, so `mu_lower <= mu_true` **by construction** — validity is not a
measurement, only tightness is. Same LP size as a plain FBA, no QP, and it reuses
`apply_mm_bounds`; the bounds are tightened with `max`/`min` so a prediction beyond
§3.3's Michaelis-Menten bound cannot loosen anything.

98 (state, organism) points, `value_p4r2` + `behaviour_p4r2`:

| point set | n | `lo` valid | median width | median lower half | median upper half | p90 width | `lo = 0` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| held-out design media | 72 | 1.000 | **0.069** | 0.069 | 0.00044 | 0.50 | 0.04 |
| §4.3 community-regime draws | 26 | 1.000 | **0.0078** | 0.0077 | 4.0e-06 | 1.00 | 0.00 |

1. **The bracket is tight where it is used: 0.8% wide at community-regime media**
   and 6.9% on the design's own held-out set, which is the certified error bar P20
   has been blocked on — a bound, not a fitted residual distribution, and no chain
   is needed to justify it.
2. **It is Head B's bracket.** The upper half contributes 4e-06 to 4e-04 of the
   width; essentially all of it is the lower bound's slack, i.e. how much growth
   the network could still make from an uptake Head B under-predicted. So the
   width is an instrument for Head B, which is where every remaining §8.6g
   residual sits.
3. **It is tight typically and uninformative on a tail** — p90 width 0.50-1.00,
   with `mu_lower = 0` on 3-4% of design points (Head B predicts no uptake of
   something essential). An error bar that goes wide exactly where the surrogate
   is least trustworthy is the desired behaviour, not a defect, but it means the
   *width* is the trigger, never the midpoint.
4. **Capping secretion at the predicted rate as well is worse, as predicted, and
   the ablation is cheap** (`--secretion`): median width 6.9% -> 7.9% and
   0.78% -> 1.4%, with `mu_lower = 0` on **12%** of community points against 0%.
   The network needs secretions Head B under-predicts. Uptake-only is the default.

---

## 6. Recommendation

1. **Keep the framing as smoothing/relaxation + amortization, not Jacobian
   estimation.** The Jacobian claim is contested by `DifferentiableMetabolism.jl`
   (2025); the smoothing claim is contested only by Barton-group reformulations,
   which are not concave and not amortized.
2. **Adopt TRF for §13.2/§13.3 (framing A).** Highest value per line changed:
   it converts M10/M11 from "surrogate answer, round-tripped" into "true optimum,
   found cheaply", and it makes the unmet M3 gate irrelevant to them.
3. **Restate §13.4's flags as an inexact-Newton forcing schedule and try JFNK
   (framing B).** The diagnosis is already written down; only the vocabulary and
   a `scipy` call are missing.
4. **Run the warm-start experiment (framing C).** One label-shard change, one
   measurement, and it decides whether the hybrid is a compromise or a win.
5. **Measure the Head A / Head B bound gap (§5) on existing states before building
   anything on it.**

Do not: re-open the Head A architecture branch, or claim a speedup against a cold
LP.

---

## 7. Chapman et al. specifically — distinctness, and whether to just extend it

Asked directly after §2a: if the LP's Jacobian is exact and cheap, why not apply or
extend `DifferentiableMetabolism.jl` instead of training heads?

### 7a. The timing table settles it

Table 1 of the paper, single core of a Ryzen 9 5950X, geometric mean of 10
replicates, "sensitivity of all variables in the optimal model solutions ... to all
parameters":

| repr. | model | #vars | DiffMet | central FD |
| --- | --- | --- | --- | --- |
| full | yeastGEM | 436 | **7.48 ± 0.11 s** | 32.45 ± 0.32 s |
| full | iML1515 | 394 | **6.24 ± 0.08 s** | 27.06 ± 0.31 s |
| OFMs | yeastGEM | 2 | 3.81 ± 0.15 s | 144.90 ± 0.99 s |

**Seconds per Jacobian, per model, per parameter point.** Against this repo's own
profiling: a whole 21-member community rhs is **17.1 ms**, Head A is **0.143 ms**
per medium batched, and the full 365-column community Jacobian is **1.70 s** — for
all 21 organisms at once, i.e. ~**80 ms per organism**, against Chapman's ~6–7 s.
Two orders of magnitude, before counting that theirs presupposes the LP solve and
this one does not.

That is the distinction in one line: **theirs makes one derivative exact; this one
makes ten thousand derivatives affordable.** They are not competing for the same
job.

### 7b. Five differences that are not about speed

1. **Different parameter.** They differentiate w.r.t. **turnover numbers and
   enzyme capacities** in enzyme-constrained models — that is the paper's whole
   motivation, and the demonstrations are kcat control coefficients and knockout
   prediction. This project's parameter is the **medium**, i.e. the LP's RHS,
   where `d(mu)/d(bound)` is the dual and has always been free from any solve.
   The overlap is therefore *not* on Head A at all; it is only on `dz/dc`.
2. **They need a fixed active set; this project lives on the switches.** Their
   pruning theorem manufactures uniqueness by deleting inactive reactions, and the
   paper states the limits plainly: local/infinitesimal only, "cannot capture
   larger metabolic switches", and the construction fails when an infinitesimal
   change switches between optimal solutions. Every hard result in this repo is a
   switch: which metabolite empties first (six instances of
   [[rhs-accuracy-does-not-buy-the-endpoint]]), the survivor swap that breaks V4 on
   near-tie cells, the active-set changes the depletion sweep lives in.
3. **First order only.** §13.4's Newton needs `sum_i X_i H_i`. There is no second
   derivative in their framework, and none in the LP.
4. **No global structure.** Concavity-by-construction is what makes §13.2/§13.3
   convex programs with a unique optimum and a certificate. Pointwise exact
   sensitivity cannot produce it.
5. **Not composed.** It is per-model sensitivity analysis. Nothing in it addresses
   per-organism composition into a shared pool, which is D1(a) and §8.1.

### 7c. The uncomfortable corollary: two of the use cases do not need the surrogate

Taking the cost argument seriously cuts both ways. Order-of-magnitude from this
repo's own numbers (label generation runs ~45k solves/hour/organism, so an LP+QP is
~0.1 s):

| use case | evaluations needed | true-LP cost | verdict |
| --- | --- | --- | --- |
| §13.2 growth maximisation | ~10² (projected ascent, one member) | ~20 s | **the surrogate is a convenience, not an enabler** |
| §13.3 minimal medium | ~10³ (ascent + greedy prune) | ~2 min | same; it already fires 94 LP solves for `--lp-repair` |
| §13.1 dFBA, one trajectory | 40 steps × 21 members ≈ 10³ | ~1.5 min | borderline; ~10⁵ over the 30-cell benchmark |
| **§13.4 steady state** | **365 FD columns × 21 members × ~11 Newton iterations ≈ 10⁵** | **~2.4 h per solve** | **enabler** — the measured surrogate solve is 104 s |
| §13.6 HMC posterior | 10⁵–10⁶ | days | enabler |

`mu_max` is genuinely concave in `c`, and the LP hands you an exact supergradient,
so §13.2 could be solved to the global optimum by projected subgradient ascent on
the true model with a computable duality gap — no surrogate, no trust region, no
V5 round-trip, no P21. That is worth stating before claiming M10 as a surrogate
result. What the surrogate buys there is ~100× and independence from a solver, not
a capability.

### 7d. The arm was built and it is refuted — the QP's derivative is Head A's — 2026-09-06

The import was `dz*/dc` labels for Head B, the head with no derivative supervision.
**Built, validated against the true QP, and measured — and the answer is that there
is almost no information in it.**

**Built** (`solve.flux_sensitivity` / `solve.exchange_jacobian`, and it is short,
because D4 does the hard part). At the optimum every reaction is *at a bound*
(`dv = dbound`), *at zero by the L1 term* (`dv = 0`, the correct local lasso
behaviour), or *free*, where stationarity is `eps*v + sign(v) = S'y` with the sign
locally constant, so `eps*dv_F = S_F' dy`. With `S dv = 0` that is

    dv_F = argmin ||d||  s.t.  S_F d = -S_B dbound_B

— the minimum-norm restoration of mass balance, and **`eps` cancels entirely**. No
pruning step: the elastic net already supplies the unique optimum Chapman et al.'s
Theorem 1 has to manufacture. Two concentration routes are included: the
metabolite's own uptake bound, and the fixed biomass flux `alpha * mu_max`, whose
sensitivity is the FBA stage's dual under `data`'s sign convention and clamp.

**Validated** against central finite differences that re-solve the QP, 3 organisms
x 25 held-out media (`20hm_bands/dz_check.py`). Columns are gated on the FD's own
step-independence — a derivative is unchanged when the step triples, solver noise
divided by the step falls to a third — which is non-circular and rejects half the
attempted columns outright:

| organism | resolvable | median cosine | median magnitude ratio |
| --- | --- | --- | --- |
| AAXE02 | 6 of 6 | **0.999991** | 0.9998 |
| CR626927.1 | 6 of 7 | **1.000000** | 0.9999 |
| GCA_000007325.1 | 6 of 12 | **1.000000** | 1.0000 |

16 of those 18 agree to six significant figures. **Two do not** — `EX_zn2_e` and
`EX_bz_e`, both trace metabolites at `c ~ 1e-6`, where the analytic predicts a
derivative 1e4-1e6 times larger than the FD sees. Unexplained; they pass the gate,
so this is not a step-size artefact.

**And then it is null, for a structural reason.** Two measurements kill it:

1. **The Jacobian is zero in almost every direction.** `dz/dc` is nonzero only
   where an uptake bound actually binds or the LP dual is nonzero — **1 to 3
   columns of 167-181**, median 2. Perturbing a slack bound cannot move the
   optimum. That is an independent confirmation of `kres.py`'s `k = 1`, arrived at
   from the QP rather than from Head A's gradient.
2. **In the columns that are live, 96-99% of it is a proportional rescale.**
   Splitting `dz/dc` into `(z/mu) * dmu/dc` — the whole flux vector scaling with
   growth rate, which Head A supplies exactly — and the residual, the residual
   share is **median 0.037 / 0.011 / 0.024** over the three organisms. So the QP's
   derivative w.r.t. the medium is, to a few percent, a restatement of the dual
   Head A already has.

Consistent with both, Head B's *composed* Jacobian already matches: Frobenius
cosine median **0.973 / 0.955 / 0.973**, magnitude ratio 0.95-1.00. It scores well
because Head A carries it.

**So the premise change was real and the arm is still dead.** The argmin *is*
differentiable once the elastic net is in place — reading-map Part 3d's "no dual to
supervise it" is wrong as stated — but the derivative it yields is not new
supervision. This is the same fact as §8.6e's reparametrisation ("one constant per
metabolite times `mu_max` explains a median 0.807 of the held-out `z` variance"),
measured locally and much more sharply: 96-99%, not 81%.

It also explains B1-B6 and the coverage rounds from a new direction. Head B's
residual error was never a local-derivative deficit; it is the *level* of `z/mu` in
regions the design does not cover — which is exactly §8.6g's conclusion, now
reached without a single composition run.

**Two caveats, both material.** This is measured at **design** media, not at the
community-regime states where Head B actually fails; the residual share could be
larger there, and that is the one measurement that would reopen this. And the
comparison excludes `mu_and_z`'s inference-time projections (the MM clamp, the
element balance), which are corrections to the prediction rather than part of the
map.

**One thing worth keeping regardless.** `d(z/mu)/dc` — the pure Head B object — is
**not computable** at these media: it is a ~9-digit cancellation between two terms
of order `1e8` whose difference is order `1`. Neither the QP nor the head resolves
it. Anything that wants to supervise or regularise Head B's derivative must work in
raw `dz/dc`, where Head A dominates, or not at all.

### 7e. What is left of §7

Unchanged: `exchange_jacobian` is exact and cheap, so it remains available as the
`z`-side analogue of `steady._head_mu_rows` for the members the LP fallback already
fires on (§4B), and as a second independent check on the head. The framing
conclusions in §6 stand — the derivative claim that survives is Jacobian-*vector*
products for a JFNK steady-state solve, not Jacobian estimation in general.
