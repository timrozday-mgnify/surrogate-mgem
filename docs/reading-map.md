# Cuts, Duals and Surrogates — a reading map for `surrogate-mgem`

Head A is a smoothed max of affine pieces, seeded from LP duals, trained to match both a
value and its gradient, and composed into a dFBA integrator. Every one of those clauses
belongs to a literature with its own name, and three of them predict failures already
measured in this repo.

Ordered by the model's own dataflow — target, class, training, use, open — rather than by
field or date. Where a paper predicts something already measured here, the entry says so;
those are the ones to read first.

Also published as an artifact:
<https://claude.ai/code/artifact/4b694fd0-287b-402f-9a5f-7bd768fb07ac>

---

## If you read five things

1. **Warin, *The GroupMax neural network approximation of convex functions*** — literally this
   architecture, and it was invented to approximate Bellman values *by cuts*. Same object as
   the label tangents.
2. **Balázs, *Max-affine estimators for convex stochastic programming*** — max-affine fitting of
   an optimisation *value function*, which is what `mu_max(u)` is.
3. **Czarnecki et al., *Sobolev Training for Neural Networks*** — the formal case for why the
   duals are worth more than more media.
4. **Ghosh et al., *Max-Affine Regression*** — proves the alternating-minimisation story:
   initialisation determines the basin. The `--gm-init labels` result stated as theory.
5. **Reznik & Segrè, *Flux Imbalance Analysis*** — the metabolic reading of shadow prices,
   including the sign conventions behind the two label-interpretation bugs in CLAUDE.md.

---

## Part 1 — The estimation target: a concave value function and its subgradients

Not a regression surface: the optimal-value map of a parametric LP, with subgradients
supplied for free. That framing explains the monotonicity constraint, the concavity
constraint, the Sobolev term and the whole `u`-coordinate argument at once.

> `mu_max` is the optimal value of an LP as a function of its right-hand side, so it is
> concave and piecewise linear in that RHS, and the optimal dual is a supergradient at every
> point. Every labelled row is therefore an exact supporting hyperplane — not a noisy sample.
> That is why a max-affine class is the correct one, why label-tangent seeding works, and why
> concavity had to be imposed in `u` rather than `x`.

| Reference | Why it matters here |
| --- | --- |
| Amos, **Tutorial on Amortized Optimization**, [arXiv:2202.00665](https://arxiv.org/abs/2202.00665), FnT ML 2023 | The name for what this project is. Read the taxonomy: fully vs semi-amortized, regression-based vs objective-based. Head A is fully-amortized and regression-based; §8's Newton composition is the semi-amortized move. |
| Czarnecki, Osindero, Jaderberg, Świrszcz, Pascanu, **Sobolev Training for Neural Networks**, NeurIPS 2017, [arXiv:1706.04859](https://arxiv.org/abs/1706.04859) | Why a dual is worth more than a medium: derivative matching constrains the function in a neighbourhood of each sample, and the gain is largest in the low-data regime — 4000 media/organism. The value-vs-gradient loss trade is the `w_grad` frontier. |
| Lu et al., **Sobolev-Trained NN Surrogate Models for Optimization**, Comput. Chem. Eng. 2021 | The process-systems version: Sobolev surrogates inside an outer optimisation, i.e. M10/M11. Reports the same finding — gradient accuracy, not value accuracy, is what the outer solver consumes. |
| Reznik, Mehta, Segrè, **Flux Imbalance Analysis**, [PLoS Comput Biol 2013](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003195) | **Read before touching the duals.** Which sign means growth-limiting, why non-limiting metabolites carry near-zero or wrong-signed prices, how degeneracy muddies them. The `_DUAL_TOL` clamp and the positive-dual-on-CO₂ finding are this paper's territory. |
| Amos & Kolter, **OptNet**, ICML 2017, [arXiv:1703.00443](https://arxiv.org/abs/1703.00443); Agrawal et al., **Differentiable Convex Optimization Layers**, NeurIPS 2019 | The road not taken: keep the LP, differentiate its KKT conditions. Exact gradients, but a solve stays in the inner loop — the cost this project exists to remove. Also defines what the exact Jacobian *is*, the reference object for `cfs master-jacobian`. |
| Bertsimas & Tsitsiklis, **Introduction to Linear Optimization**, ch. 4–5 | Duality and sensitivity done properly, including **degeneracy making the dual non-unique** — the M1 finding (68.9% degenerate) restated. When the dual is not unique, an "exact tangent" is a selection from a set. |

---

## Part 2 — The hypothesis class: max-affine, ICNN, GroupMax

Three literatures converged on this architecture from different directions: statistics
(convex regression), deep learning (convex/monotone networks), and stochastic programming
(cuts). The statistics one has the sharpest results and is the least likely to be on the shelf.

| Reference | Why it matters here |
| --- | --- |
| Warin, **The GroupMax Neural Network Approximation of Convex Functions**, [arXiv:2206.06622](https://arxiv.org/abs/2206.06622), IEEE TNNLS 2023 | `--arch groupmax-u`. Read the motivation: designed because ICNNs approximate convex functions *smoothly* and Bellman value functions need approximating **by cuts**. The whole coordinate-and-activation finding, published before it was measured here. Also has the universal approximation theorem and the partial-convexity adaptation (the route to admitting abundances alongside `u`). |
| Amos, Xu, Kolter, **Input Convex Neural Networks**, [ICML 2017](https://proceedings.mlr.press/v70/amos17b/amos17b.pdf) | What this started with. Read now for contrast: the measured result that a softplus ICNN converges to a chord where the target has a kink is a statement about this class's smoothness prior. The PICNN half is still live. |
| Goodfellow et al., **Maxout Networks**, [arXiv:1302.4389](https://arxiv.org/abs/1302.4389); Anil, Lucas, Grosse, **Sorting Out Lipschitz Function Approximation** (GroupSort), [arXiv:1811.05381](https://arxiv.org/abs/1811.05381) | The original group-max unit, and the norm-constrained cousin GroupMax is named after. Maxout's section on units "dying" is the dead-plane pathology: a unit never the max of its group receives no gradient. |
| Ghosh, Pananjady, Guntuboyina, Ramchandran, **Max-Affine Regression**, [arXiv:1906.09255](https://arxiv.org/abs/1906.09255) | Alternating minimisation with guarantees **conditional on initialisation inside a basin**. The `--gm-init labels` A/B (0.598 → 0.973, everything else held) is this theorem's empirical shadow. Read the initialisation section and the spectral method they use when they don't have tangents — we do, which is why seeding beats it. |
| Balázs, **Max-Affine Estimators for Convex Stochastic Programming**, [arXiv:1609.06331](https://arxiv.org/abs/1609.06331) | Max-affine estimation of an optimisation value function — the exact object. Sample complexity, and the interaction between number of pieces and data, i.e. the K-vs-rows question treated theoretically. |
| Balázs, **Adaptively Partitioning Max-Affine Estimators for Convex Regression** (AMAP), [AISTATS 2022](https://proceedings.mlr.press/v151/balazs22a/balazs22a.pdf); Hannah & Dunson (CAP, JMLR 2013); Magnani & Boyd, **Convex piecewise-linear fitting** (LSPA, Optim. Eng. 2009) | The alternation family `--gm-reanchor` belongs to. AMAP cross-validates the model size and **adapts to intrinsic dimension** — the formal version of "444 inputs, ~30 active per organism". Our reanchor is a cheap AMAP: the least-squares refit is skipped because the exact tangent is in the labels. |
| Balázs, György, Szepesvári, **Near-Optimal Max-Affine Estimators for Convex Regression**, [AISTATS 2015](https://proceedings.mlr.press/v38/balazs15a.html) | The minimax rate. The honest expectation for how error falls with rows — the measured cutting-plane scaling (1−cos falling ~0.89×/doubling on one organism, ~0.75× on another) is this rate. |
| **An Elementary Proof of the Near Optimality of LogSumExp Smoothing**, [arXiv:2512.10825](https://arxiv.org/abs/2512.10825); Blanchard, Higham & Mary, **Accurate Computation of the Log-Sum-Exp and Softmax Functions** | The smoothing gap is bounded by `T·ln(K)` and no overestimating smoothing beats ~0.81·ln(K). **That bound is the measured low-`mu` offset**: at T=0.03 with a few hundred active planes, `T·ln(K_active)` ≈ 0.2 in `mu_scale` units — 4% at the plateau, >100% at a starving medium. It also shrinks only logarithmically in K, so more planes cannot buy it back, consistent with K 1000→2000 doing nothing. |
| **Advancing Constrained Monotonic Neural Networks**, [arXiv:2505.02537](https://arxiv.org/abs/2505.02537); Sill, **Monotonic Networks**, NIPS 1997 | Sill's min-max networks are a max-of-mins of sign-constrained affine functions — architecturally a sibling, and the origin of the `-softplus(param)` trick. The 2025 paper covers what the constraint costs in expressiveness: the question behind "3× slower convergence, not a regression". |
| Calafiore, Gaubert, Possieri, **Universal Approximation of Difference of Log-Sum-Exp Networks**, IEEE TNNLS 2020 | P11's difference-of-convex escape hatch, if ever needed. The measured result (unconstrained MLP moves the median +0.009 and collapses the worst organism) argues against it — read to close the question. |

---

## Part 3 — The training tweaks, and where each comes from

### 3a. Cut selection and pruning — the mature answer to "K is inert above 1000"

**This is the closest match in any literature to a problem currently open in the repo, so it
gets its own treatment.**

#### The mapping

SDDP builds a value function as an accumulating set of subgradient cuts taken from solved
subproblems, giving an outer approximation

```
V_bar(x) = max_j [ H_j·x + h_j ]        (convex minimisation; ours is the min, concave)
```

which is exactly `mu_hat(w) = min_j [mu_j + pi_j·(w − w_j)]`, the parameter-free cutting-plane
model in `cfs.surrogate.groupmax`. Their *cuts* are our *label tangents*; their *trial points*
are our *media*; their cut budget is our `--gm-group K`. The field has ~15 years of work on
exactly the question of which cuts to keep.

#### The two exact/heuristic criteria

**A cut is *useless*** if dropping it does not change the approximation anywhere on the domain:

```
H_j·x + h_j  ≤  max_{k≠j} [H_k·x + h_k]     for all x in X
```

**Test of usefulness** (Pfeiffer, Apparigliato & Auchapt 2012, eqns 19–21) decides this
*exactly* with one LP per cut:

```
max   y
s.t.  x ∈ X,  y ∈ R
      y ≤ (H_j − H_k)·x + (h_j − h_k)    for all k ≠ j
```

The cut is useless iff the optimum `y ≤ 0`. Exact, and too expensive to run every iteration —
one LP per cut, each with (number of cuts) constraints.

**Territory algorithm / Level 1 dominance** (de Matos, Philpott & Finardi 2015; the territory
formulation in Pfeiffer et al. 2012 — the two select the *same* cuts) is the cheap heuristic
that replaces "for all x in X" with "for the trial points actually visited". Each cut `j`
carries a **territory** `P_j`: the visited points at which `j` is the active one. When a new
cut `j` is computed at point `x̄`:

- `P_j = {x̄}`;
- find the cut `k` maximising at `x̄`; if `k` beats `j` there, `P_j = ∅` and `P_k` gains `x̄`;
- for every other cut `k` and every `x ∈ P_k`, if `j` beats `k` at `x`, move `x` from `P_k` to `P_j`;
- **any cut left with an empty territory is dropped.**

Pure function evaluations, no LP. A cut with empty territory is *potentially* useless, not
provably so — the territory test can delete a cut that is useful over a region containing no
visited point. That does not break convergence, because the algorithm can recompute it later,
and the test sharpens monotonically as points accumulate.

**Combination**: when a cut's territory empties, run the exact test of usefulness; delete only
if `y ≤ 0`, otherwise keep it with a fresh territory.

#### The numbers (Pfeiffer et al., 19-dimensional state, 365 stages, 500 iterations)

| method | mean cuts/stage | optimisation phase | simulation phase |
| --- | --- | --- | --- |
| no selection | 490 | 543 s | 798 s |
| territory | 220 | 690 s | **168 s** |
| territory + usefulness | **55** | 2181 s | 382 s |

Two conclusions transfer directly:

1. **Pruning to ~1/10 of the cuts does not degrade the value function.** "The forward cost is
   decreasing at similar rates for the three methods", and the territory algorithm "provides
   the most efficient representation of the value function, with the smallest number of cuts".
   That is the mature answer to the K-is-inert result: the problem was never the number of
   planes, it is which ones — and 10× fewer well-chosen ones perform the same.
2. **The exact test is not worth running online** (2181 s vs 543 s) but is affordable as a
   one-off audit.

#### Guigues (2017), *Dual Dynamic Programming with cut selection* — [arXiv:1705.08941](https://arxiv.org/abs/1705.08941)

Three things worth taking:

- **Store vs prune.** de Matos et al. *delete* non-relevant cuts; Philpott et al. *store all
  cuts and select* the relevant subset each time. Storing-and-selecting is strictly safer: a
  deleted cut must be recomputed, a stored one is free. For us, storage is trivial — the
  tangents are already on disk in the label shards.
- **Limited Memory Level 1 (LML1).** At each trial point store the index of only *one* cut —
  the oldest among those attaining the max there. Memory is then O(number of trial points)
  rather than O(cuts × points), with convergence preserved.
- **The convergence condition is weak.** All of Territory, Level 1, LML1 and "Level H" (keep
  the H highest cuts at each trial point) satisfy Assumption (H2): the approximation at
  previously visited trial points is non-decreasing across iterations. Any selection rule
  satisfying that inherits the convergence proof. A rule we invent can be checked against it.

#### What this says to do here, concretely

| Repo item | What the literature says |
| --- | --- |
| `rank_by_active_set` (buckets rows by dual support pattern, takes representatives of the commonest) | This is a *proxy* for "which regimes occur". **Level 1 is the exact version of what it approximates**: rank a tangent by whether it is the binding minimum at a point you care about. Same cost class — one (media × tangents) evaluation, a matmul we already do. |
| **C1, "put planes in the community regime"** (design spec §8.5) | Falls out for free: the point set in Level 1 is *chosen*. Score the 16 000 label tangents by territory size over **community-regime media** (the A1 holdout set is exactly such a point set) and keep the top K. No new labels, no retrain of anything upstream. |
| `reanchor` evicting the lowest-softmax-weight planes | Softmax weight averaged over training media *is* a smooth territory size. The literature's version is sharper and gives a principled eviction criterion (empty territory) plus an exact fallback (test of usefulness) for the ones it would evict. |
| "K = 1000 → 2000 changes nothing" | Expected. Cut count is not the lever; territory coverage of the region you evaluate in is. |
| Auditing the seeded planes | The test-of-usefulness LP, run once over the K = 1000 seeded planes per organism, answers "how many of these are provably redundant" exactly. Affordable as a one-off. |

### 3c. Under-prediction — the *other* sign, and why most of the toolkit misses it

**An under-prediction is a certificate that the head has left the family.** A min of
*supporting* hyperplanes of a concave function is an upper bound everywhere, so it
cannot read low. There are exactly two ways to get one, and they are separable in
minutes:

| mechanism | test | verdict here (`p4`, n=21, GCA_000007325.1) |
| --- | --- | --- |
| the softmin's downward gap, `<= T*ln(K_active)` | re-evaluate the trained head at `T -> 1e-6` (`groupmax.with_temp`) | **refuted** — moves `mu_hat` by 0.008. The `T*ln(K)` bound (0.75 in mu units) is loose because ~1 plane is near-active there |
| planes no longer valid tangents | check `mu_hat >= mu` on the head's **own training rows** | **confirmed** — 48.1% of `p4` rows under-predicted before repair, 53-68% in the bottom-5% `mu` band |

Do this before reading anything: it decides which of the four families below applies,
and the second row is checkable with no new solves.

| Reference | Why it matters here |
| --- | --- |
| Pereira & Pinto, Math. Prog. 1991; Philpott & Guan, **On the convergence of stochastic dual dynamic programming**, ORL 2008; Füllner & Rebennack's review (3b) | **Cut validity is a structural invariant, not a fitted property.** SDDP keeps it by never modifying a cut once added — which is exactly what a gradient method does to ours. The technique that follows is a *validity projection*: hold the slopes, and set each intercept to the tightest value that keeps the plane above every training label. Closed form, no refit — `groupmax.repair_intercepts`, `--gm-repair`. |
| Koenker & Bassett, **Regression quantiles**, Econometrica 1978; Newey & Powell, **Asymmetric least squares estimation**, Econometrica 1987 | The asymmetric-loss family `--w-under` belongs to. The hinge is the `tau -> 1` limit taken implicitly; the literature's contribution is that `tau` can be *chosen*, with stated coverage, instead of tuning a weight against the composition. |
| Balázs, **Convex regression: theory, practice and applications** (2016, thesis) — already cited in Part 2 | The hard-constrained version: fit subject to `f(u_r) >= mu_r` for all `r`. With slopes fixed that is a linear program in the intercepts, i.e. the same object as the validity projection above — worth knowing they coincide, because it says the projection is *optimal* for its slopes and not a heuristic. |
| Vovk, Gammerman & Shafer, **Algorithmic Learning in a Random World**; the conformal-MBO paper in 3b | One-sided conformal: inflate by the `(1-alpha)` quantile of the **signed** residual on a calibration set, for a finite-sample guarantee in whichever direction you need. The caveat has bitten this project twice — validity is w.r.t. the calibration distribution, so it must be `community_holdout`, never held-out media from the design being changed. |
| Kumar et al., **Conservative Q-Learning**, NeurIPS 2020; Osband et al., **Bootstrapped DQN**, NeurIPS 2016 | Why C4 had nothing to offer, and why there is no symmetric fix. min-over-ensemble is pessimism, max is optimism — and **a concave family admits only the min**, since a max of concave functions is not concave. Do not reach for "max over seeds". |

**The mismatch worth stating in a writeup.** The offline-MBO and offline-RL literatures
are overwhelmingly about *conservatism*: a learned surrogate that is optimistic
off-distribution gets exploited by whatever optimises against it. This head is
conservative by construction, and the failure that remains is the opposite sign. So
most of that machinery points the wrong way here, and the interesting question — how to
keep a structurally one-sided estimator *tight* rather than how to make an unconstrained
one safe — does not appear to be addressed.

### 3b. The rest of the training loop

| Reference | Why it matters here |
| --- | --- |
| Füllner & Rebennack, **SDDP and its Variants — a review**, [optimization-online](https://optimization-online.org/wp-content/uploads/2021/01/SDDP-Review.pdf); Pereira & Pinto, Math. Prog. 1991 | The umbrella survey for the above, plus cut *dominance*, multicut vs single-cut, and the known slow tail of convergence. |
| Dai et al., **Neural Stochastic Dual Dynamic Programming**, [arXiv:2112.00874](https://arxiv.org/abs/2112.00874); Bae et al., **Deep Value Function Networks**, [AISTATS 2023](https://proceedings.mlr.press/v206/bae23a.html) | Replacing an accumulated cut set with a learned model that outputs a piecewise-linear value function — the same trade, in energy systems. Useful for how they handle generalisation across instance families. |
| **Offline Model-Based Optimization: A Comprehensive Review**, [arXiv:2503.17286](https://arxiv.org/html/2503.17286v2) | The field built around P21 and the n=21 tail: a learned surrogate optimised against will find where it is optimistic. Read the conservatism/pessimism sections. `--w-under` is a structural special case — a max-affine head is one-sided, so conservatism needs no ensemble. |
| **Conformal Candidate Certification for Offline MBO**, [arXiv:2606.15217](https://arxiv.org/pdf/2606.15217) | Uses a signed **one-sided** nonconformity score, large when the surrogate overestimates. Our hinge plus the community-holdout gate, with a finite-sample guarantee attached. |
| Kuleshov, Fenner, Ermon, **Accurate Uncertainties for Deep Learning Using Calibrated Regression**, [ICML 2018](https://arxiv.org/abs/1807.00263) | The family `calibrate.py` belongs to, and explicit that recalibration is **distribution-dependent**: valid only for the distribution it was fit on. The measured result that the same calibration helps on `r1` and is the dominant error on `p4` is that caveat in the wild. |
| Settles, **Active Learning Literature Survey** (TR 1648, 2009); [Active learning for adaptive surrogate model improvement](https://pmc.ncbi.nlm.nih.gov/articles/PMC11236939/) (2024) | Stage 4 of the §8.5 progression. The failure documented across this literature: **acquisition computed on a pool drawn from the training distribution cannot find distribution shift** — the `cfs topup` result. Design the pool, not the acquisition function. |

---

## Part 4 — The use cases, and who else has tried

| Reference | Why it matters here |
| --- | --- |
| Mahadevan, Edwards, Doyle, **Dynamic FBA of Diauxic Growth in *E. coli***, Biophys. J. 2002 | What `compose/dfba.py` is: the SOA variant with the LP replaced. Worth re-reading for the stability discussion — the two-clock problem (members doubling vs pool emptying) is acknowledged there. |
| Chan, Simons, Maranas, **SteadyCom**, [PLoS Comput Biol 2017](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005539); Khandelwal et al., **cFBA**, PLoS ONE 2013 | §8.2. The balanced-growth condition and why it is nonlinear in abundances; SteadyCom's reformulation tells you what the surrogate must supply (per-member growth on a shared medium — we have it) and what it need not. |
| Diener, Gibbons, Resendis-Antonio, **MICOM**, mSystems 2020 | §8.3, and the legacy package. Worth being able to state precisely why per-organism FBA ground truth keeps a Head B error distinguishable from a modelling choice. |
| **Coupling FBA with Reactive Transport Modeling through Machine Learning**, [Sci Rep 2025](https://www.nature.com/articles/s41598-025-89997-9) | **Nearest published prior art**: an ANN replacing the LP as the source/sink term in a simulation, same two motivations (speed, and removing LP failures). Note what it does *not* do: no concavity, no monotonicity, no dual supervision, no per-organism composition. |
| **Minimizing the Number of Optimizations for Efficient Community dFBA**, [bioRxiv 2020](https://www.biorxiv.org/content/10.1101/2020.03.12.988592) | The non-ML alternative: reuse the LP basis across timesteps. A baseline any speed claim should acknowledge — and it sharpens what a surrogate buys that basis reuse cannot (differentiability, and a medium-design program with no solver in the loop). |
| **gsMOBO — Multiobjective Design of Growth Media with GEMs and Bayesian Optimization**, [CSBJ 2025](https://spj.science.org/doi/10.34133/csbj.0072); [ML-led medium optimisation, Commun Biol 2025](https://www.nature.com/articles/s42003-025-08039-2) | The M10/M11 competitor. BO needs a solve per acquisition and scales badly in dimension; the convex program here is one projected-gradient run over 444 metabolites. |
| **SIMBA-GNN**, [npj Syst Biol Appl 2025](https://www.nature.com/articles/s41540-025-00631-w) | The opposite hybrid — simulation as feature extraction rather than the thing amortized. Useful contrast: their model cannot answer a counterfactual medium question. |

---

## Part 5 — Open questions → what to read

| Open question | Read | What it should tell you |
| --- | --- | --- |
| The n=21 over-prediction tail that survived four design changes | Offline MBO review §conservatism; conformal one-sided certification; Balázs 2016 | Whether to bound the head from below, quantile-ise it, or design the composition around a one-sided estimator |
| The n=21 **under**-prediction that remains once the calibration is stripped | **Part 3c above** — SDDP cut validity; asymmetric-loss regression; one-sided conformal | That an under-prediction is a validity failure, not an accuracy one, and that the repair is a projection with a closed form rather than a refit |
| Cut selection: K inert above 1000, planes in the wrong place | **Part 3a above** — territory / Level 1 / LML1, test of usefulness | That 10× fewer well-chosen cuts perform the same, and that the selection criterion should use the point set you actually evaluate at |
| Newton on a Hessian sum of rank ~20 of 365 | Qi & Sun, *A nonsmooth version of Newton's method*, Math. Prog. 1993; active-set / reduced-space methods | That a rank-deficient generalised Hessian is expected for a piecewise-linear value function, and the semismooth machinery built for it |
| Choosing `T`: accuracy vs smoothness for downstream HMC | LogSumExp near-optimality bound; Higham & Mary on LSE numerics | The exact `T·ln(K)` offset being paid, why it shrinks only logarithmically in K, where float precision stops you |
| Stage 4 active learning: which acquisition, which pool | Settles' survey; the adaptive-surrogate literature | That pool construction dominates acquisition choice, and that the bound-looseness score is a legitimate cheap alternative to query-by-committee |
| Non-unique duals under degeneracy (the 68.9%) | Bertsimas & Tsitsiklis ch. 4; Reznik & Segrè | That an "exact tangent" is a selection from a set, which reframes D4's elastic-net label uniqueness as choosing a canonical dual |

---

## What appears not to exist yet

A search across the convex-regression, amortized-optimization and metabolic-modelling
literatures turns up no published surrogate that is simultaneously **concave and monotone by
construction**, **supervised on LP duals**, and **composed per-organism into a community
simulation**. The nearest neighbours each drop one leg: the reactive-transport ANN has no
structure, GroupMax and the max-affine statistics have no biology, and the community-FBA
methods keep the LP. If this is written up, that three-way intersection is the claim — and
the concavity is what makes the §13 medium-design programs convex, which is the part a
reviewer will care about more than the speedup.

---

*Compiled 2026-09-01.*
