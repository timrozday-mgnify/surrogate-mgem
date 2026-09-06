# Cuts, Duals and Surrogates — a reading map for `surrogate-mgem`

Head A is a smoothed max of affine pieces, seeded from LP duals, trained to match both a
value and its gradient, and composed into a dFBA integrator. Every one of those clauses
belongs to a literature with its own name, and three of them predict failures already
measured in this repo.

Ordered by the model's own dataflow — target, class, training, use, open — rather than by
field or date. Where a paper predicts something already measured here, the entry says so;
those are the ones to read first.

The **composition** of head and solver — the LP-in-the-loop hybrids, and whether the
project should be reframed around them — is a separate map:
[`hybrid-framing.md`](hybrid-framing.md).

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
| Pereira & Pinto, Math. Prog. 1991; Philpott & Guan, **On the convergence of stochastic dual dynamic programming**, ORL 2008; Füllner & Rebennack's review (3b) — **and this is now the measured best arm on `n=21`: selected label tangents with NO gradient training plus the validity repair give 0.175 against 0.272-0.36 for every trained head.** Two further things transferred literally: SDDP's **forward pass** as the trial-point set (the dFBA path's own `c_true`, which selects **3-10 cuts per organism** and reproduces the 45-103-cut model), and its **stability centre** (a proximal term on the slopes, `--w-prox`) — which is the one that did *not* transfer: monotonically harmful once it binds. | **Cut validity is a structural invariant, not a fitted property.** SDDP keeps it by never modifying a cut once added — which is exactly what a gradient method does to ours. The technique that follows is a *validity projection*: hold the slopes, and set each intercept to the tightest value that keeps the plane above every training label. Closed form, no refit — `groupmax.repair_intercepts`, `--gm-repair`. |
| Koenker & Bassett, **Regression quantiles**, Econometrica 1978; Newey & Powell, **Asymmetric least squares estimation**, Econometrica 1987 | The asymmetric-loss family. Built as `--w-tau` (expectile, `tau = 0.5` reproducing the MSE exactly) and **measured: it works and it loses to the hinge.** A1's tail falls 3.5x over `tau` 0.5 -> 0.9 then turns over at 0.99, but an expectile tilts *every* row, so it degrades the fit everywhere (community `mu_rel_median` 0.254 vs `--w-under`'s 0.052) where the hinge is exactly zero on compliant rows. **For a provable violation, pay only at the violation** — the asymmetric-loss framing is the right theory and the wrong shape. |
| Balázs, **Convex regression: theory, practice and applications** (2016, thesis) — already cited in Part 2 | The hard-constrained version: fit subject to `f(u_r) >= mu_r` for all `r`. With slopes fixed that is a linear program in the intercepts, i.e. the same object as the validity projection above — worth knowing they coincide, because it says the projection is *optimal* for its slopes and not a heuristic. |
| Vovk, Gammerman & Shafer, **Algorithmic Learning in a Random World**; the conformal-MBO paper in 3b | One-sided conformal: inflate by the `(1-alpha)` quantile of the **signed** residual on a calibration set, for a finite-sample guarantee in whichever direction you need. The caveat has bitten this project twice — validity is w.r.t. the calibration distribution, so it must be `community_holdout`, never held-out media from the design being changed. |
| Lemaréchal, Nemirovskii & Nesterov, **New variants of bundle methods**, Math. Prog. 1995; Kiwiel, *Proximal level bundle methods* | The stability-centre idea behind `--w-prox`: keep an outer approximation while stopping the iterate wandering. Built and **refuted** here — monotonically harmful at every weight once normalised so it binds. The durable lesson is the trap it exposed: the label slopes span **five decades**, so a global `mean(d²)/mean(a0²)` normalisation is set by a handful of huge planes and is silently inert. Normalise per plane. |
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

## Part 3d — Head B: the *argmin* map, and why none of Parts 1–3 applies

Head A is the value function of a parametric LP. Head B is the **argmin** of the same
LP, and almost nothing said about the value carries over: the argmin is not concave,
not monotone, not continuous, and has no dual to supervise it. What it *is* is
piecewise affine over polyhedral critical regions, confined to a low-dimensional flux
subspace, and pinned exactly on the uptake bound wherever the corresponding dual is
non-zero. Measured on the labels (design spec §8.6f): rank **12–39** of 138–259
exchanges at 99.99% of variance, and `dual ⇒ flux on the bound` at **0.996–1.000**.

| Reference | Why it matters here |
| --- | --- |
| Gal & Nemhauser, multiparametric LP; Borrelli, Bemporad & Morari, **Predictive Control for Linear and Hybrid Systems**, ch. on explicit MPC | The structure of the target. The optimiser is affine on each *critical region* — a polyhedron of parameters sharing an optimal active set — so `c -> z` is piecewise affine, and a smooth MLP is approximating a partition by a single smooth sheet. This is the Head B analogue of "concavity in the wrong coordinate". |
| Jones, Kvasnica et al., **Lexicographic perturbation for multiparametric linear programming**, Automatica 2007 | **Degeneracy makes the argmin non-unique**: overlapping regions, discontinuous optimisers. M1 measured 68.9% degenerate exchange-FVA observations, and D4's elastic net is exactly the lexicographic-perturbation trick by another name — a strictly convex tiebreak that selects one canonical optimiser. Read this to know what continuity the labels do and do not have. |
| Bertsimas & Stellato, **The Voice of Optimization**, [arXiv:1812.09991](https://arxiv.org/pdf/1812.09991); **Online Mixed-Integer Optimization in Milliseconds** | Learn the optimal *strategy* (active set), then recover the solution exactly by solving the small system it implies — instead of regressing the solution. For us the classifier is free: complementary slackness says the tight set is where Head A's gradient is non-zero, and Head A is at `mu_rel <= 5e-4`. **Measured and refuted anyway** (§8.6f): `mu_and_z`'s clamp already lands on the bound for every tight entry, and the tight set carries only 1–10% of the squared error. |
| Chen, Fazlyab et al. / Katz, Pappas et al., **Universal approximation of parametric optimization via piecewise-linear policy approximation**, [arXiv:2308.10534](https://arxiv.org/pdf/2308.10534) | If the PWA structure is ever worth building in (B6), this is the approximation theory for it — including how many pieces a target with `N` critical regions needs. |
| Klamt et al., **From elementary flux modes to elementary flux vectors**, [PLoS Comput Biol 2017](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005409); Bhadra, Blomberg, Castillo & Rousu, **Principal metabolic flux mode analysis**, [Bioinformatics 2018](https://academic.oup.com/bioinformatics/article/34/14/2409/4840578) | Why an SVD of the label fluxes has such low rank: the steady-state flux cone has an inner description by generating vectors, and any observed flux is a conical combination of them. Bhadra et al. is the practical version — extract a small basis of flux modes by convex optimisation and work in its coordinates, which is exactly **B1**. |
| Famili & Palsson, **The convex basis of the left null space of the stoichiometric matrix**, [Biophys J 2003](https://www.cell.com/fulltext/S0006-3495(03)74450-6); Haraldsdóttir & Fleming, **Identification of conserved moieties**, [PLoS Comput Biol 2016](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004999) | The exact conservation relations `l·S = 0` that any feasible flux obeys, and how to compute them. Note the caveat the second paper states outright: **adding exchange reactions destroys the left null space**, so the useful relations for `z` are the *elemental/moiety* ones (carbon, nitrogen, charge) balanced against the biomass drain, not conservation of concentration. Worth knowing before building B5 — and B1 gets them for free, since a conservation law is a direction of zero variance the SVD discards. |
| Donti, Rolnick & Kolter, **DC3: a learning method for optimization with hard constraints**, [arXiv:2104.12225](https://arxiv.org/abs/2104.12225) | The general recipe when a prediction must satisfy constraints: **completion** (predict the free coordinates, solve the equalities for the rest) plus **correction** (unrolled gradient steps on the inequality violations). Our MM clamp is a one-shot correction; B1's basis is a completion in disguise; B5 would be completion done literally. Note the measured order of value here: the correction bought the composition (§8.6d) and the *projection* bought 0.001 (§8.6f). |
| Sturm & Wexler, **Conservation laws in a neural network architecture**, [GMD 15, 3417, 2022](https://gmd.copernicus.org/articles/15/3417/2022/) | The same idea in atmospheric chemistry, and the cleanest statement of the design: predict **fluxes** in a penultimate layer and derive the state change from them, so conservation holds by construction rather than by penalty. Head B already predicts fluxes; the missing half is the balance they must satisfy. |
| **Linking intra- and extra-cellular metabolic domains via neural-network surrogates for dynamic metabolic control**, [arXiv:2310.17179](https://arxiv.org/pdf/2310.17179); **Hybrid physics-informed metabolic cybergenetics**, [arXiv:2401.00670](https://arxiv.org/pdf/2401.00670) | The nearest prior art for Head B specifically: an NN mapping manipulable intracellular fluxes to exchange fluxes, used inside a dynamic controller. Same object, no structure imposed — useful mainly to show that the unconstrained version is what everyone builds, and it is what fails here off-distribution. |
| Chen, Roberts et al., **Coupling FBA with reactive transport modeling**, [Sci Rep 2025](https://www.nature.com/articles/s41598-025-89997-9) (also Part 4) | Read here too: their ANN *is* Head B, and the paper's own error analysis is about the same regime — near-depletion, where the LP's basis changes fastest. |
| Cutler & Breiman, **Archetypal analysis**, Technometrics 1994 | The stricter version of B1 if the linear subspace is not enough: represent each prediction as a convex combination of extreme observed behaviours, so the head cannot leave the convex hull of the training fluxes at all. Bounds magnitude, not just direction — the failure in §8.6e was a 3300x magnitude blow-up. |

**The measured verdict, in one line.** The uptake side is already handled by a
projection (§8.6d's clamp) and complementary slackness adds nothing on top of it;
the error lives on the **secretion** side (48–69% of it), which is unbounded above.

**Updated 2026-09-03, after B1–B6 all ran.** The subspace (B1) was built and is
null at both horizons; the active set (B6) does not discriminate the failing states
(Hamming 0.0); coverage (rounds 2–4) improves `dc_rel` exactly as the reach proxy
predicts and does not reach the batch endpoint. So the open question moved from
*which class* to *what to do about extrapolation and about a discrete endpoint*,
and the reading below is the part of the literature that speaks to that.

| Reference | Why it matters here |
| --- | --- |
| Chen, Rubanova, Bettencourt, Duvenaud, **Neural Ordinary Differential Equations**, NeurIPS 2018; the differentiable-simulator literature generally | §8.6g's solution 3, and the only one that optimises what the gate measures. Six times now a better `dc` has not bought the endpoint, because *when* a batch stops turns on which metabolite empties first; a loss on the trajectory sees that and a per-state MSE cannot. `compose.dfba.integrate` is already an explicit Euler map in JAX, so the adjoint is available. |
| **Minimizing the Number of Optimizations for Efficient Community dFBA**, [bioRxiv 2020](https://www.biorxiv.org/content/10.1101/2020.03.12.988592) (also Part 4) | The non-ML baseline *and* the shape of the fallback in §8.6g's solution 4: reuse/short-circuit the LP where it is cheap, solve it where it is not. Our trigger is predicted depletion depth (24% of member-steps carry 72% of the error); theirs is basis validity. Any speed claim has to be stated against this, not against a cold LP. |
| Settles, **Active Learning Literature Survey** (also 3b) | Read again for the *self-labelling* loop: the failure it documents is acquisition over a pool drawn from the training distribution. The fallback avoids it structurally — the pool is the trajectory, which is the shifted distribution — and the remaining risk is the covariate shift the loop induces in itself, which is why every label is kept and the trajectory re-run after each retrain (Guigues's store-and-select, 3a). |
| Vovk, Gammerman & Shafer (also 3c); Angelopoulos & Bates, **A Gentle Introduction to Conformal Prediction** | §13.6's missing error model. The nonconformity score is available and measured: NN distance in `x` predicts `dc_rel` at Spearman +0.673 **across cells**. The caveat is now measured too — the same quantity is *anti*-correlated with error **within** a trajectory (lift 0.9x), so calibrate per run, not per step. |
| Famili & Palsson 2003; Haraldsdóttir & Fleming 2016; Donti et al., **DC3** (all above) | Re-read for §8.6g's solution 2, the secretion-side inequality `E z <= 0`. Note which half of DC3 has paid here: the *correction* (the MM clamp) bought the composition, the *projection* (B1) bought 0.001. **Built 2026-09-04 and kept**: labels violate it at 0.0000, Head B at 34-53%, and the *minimum-norm* projection is 14/30 cells better on the endpoint and never worse. A uniform shrink satisfying the same inequalities made `dc_rel` worse on 10 of 11 cells — DC3's correction step is a projection for a reason. |

**Updated 2026-09-04, after all four of §8.6g's solutions ran.** Only the fallback
moved the gate, and none of the four changed Head B as a model:

* **The differentiable simulator (solution 3) is refuted here, and the reason is
  structural rather than about neural ODEs.** `d(log X)/dt` is Head A's frozen
  `mu`, so the trajectory loss reaches Head B *only* through the pool sum
  `sum_i X_i z_i`: N members' fluxes collapse into one vector per step and the
  individual `z_i` are unidentifiable (held-out R2 0.577 -> -31.98 while the loss
  moved 9%). Monocultures restore identifiability — and there the whole gain is
  reproduced by the label term alone at `--w-traj 0`. Anyone reading the neural-ODE
  literature for this project should start from *what the loss can identify*, not
  from the adjoint.
* **The bioRxiv 2020 dFBA baseline is now the right comparison and the gap is
  quantified.** Our trigger fires on 24.6% of member-steps live — matching the
  offline ROC exactly — and takes the 8-doubling gate from 0.041 to 0.028, n=21 to
  0.015. State speed against that paper's basis-reuse, not against a cold LP.
* **Settles's warning did not bite, and something else did.** The self-labelling
  loop is built (`--fallback-media` -> `cfs generate --media` -> retrain with
  `x_scale` pinned) and its first pass, 71 media, is null against a matched
  control — but the *control itself*, a fresh fit with no new rows, moved the
  8-doubling mean 0.085 -> 0.143. The active-learning risk to manage here is not
  acquisition bias, it is that retraining variance exceeds the signal being
  acquired.

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

## Part 4b — The hybrid: surrogate *and* solver, and the methods for it

Added 2026-09-06, after three applications shipped with an LP in the loop. Full
stock-take, prior-art comparison and the ranked plan: **[`hybrid-framing.md`](hybrid-framing.md)**.
Read that first; this table is the bibliography for it.

| Reference | Why it matters here |
| --- | --- |
| Alexandrov, Dennis, Lewis & Torczon, **trust-region model management** with first-order-consistency corrections; Eason & Biegler, **A trust region filter method for glass box/black box optimization**, [AIChE J 2016](https://aiche.onlinelibrary.wiley.com/doi/abs/10.1002/aic.15325), [2018 sequel](https://aiche.onlinelibrary.wiley.com/doi/abs/10.1002/aic.16364); Hameed et al., [AIChE 2026](https://aiche.onlinelibrary.wiley.com/doi/10.1002/aic.70236) (Hessian information) | **Built for §13.2 and it works** (§13.2b): correct the head to the LP's value *and* gradient at the trust-region centre and the loop converges to a KKT point of the *true* problem with no global accuracy requirement — which is why the unmet M3 gate stops constraining this use case. Read the 2018 sequel for the sampling-region variant that avoids shrinking the radius to zero. |
| Kelley, **cutting-plane method** for convex programs; the bundle-method literature (Lemarechal, Nemirovskii & Nesterov 1995; Kiwiel) — also Part 3a | **Built for §13.3** (§13.3b). Where the surrogate is in the *constraints* and the objective is exact, the trust region does not transfer but the bundle does: a tangent to a concave `mu_true` is an upper bound everywhere, so requiring it to clear the growth floor is necessary for the true constraint. Note the two failure modes both literatures warn about and this project hit: a **single subgradient stalls at a kink** (§13.2b), and a **cutting-plane subproblem is nonsmooth**, so it wants an LP/QP over the epigraph rather than a subgradient step (P28). |
| Dembo, Eisenstat & Steihaug, **Inexact Newton methods**, 1982; Eisenstat & Walker, **Choosing the forcing terms**, [SISC 1996](https://users.wpi.edu/~walker/Papers/forcing_terms,SISC_17,1996,16-32.pdf) | §13.4's `--mix-mu-rel` is a *fixed* forcing term where the theory says it should be a schedule, and "a pure LP residual with a surrogate Jacobian does not converge" is this literature's textbook failure, already observed. Jacobian-free Newton–Krylov needs only Jacobian-*vector* products, which the smoothed head supplies where the LP supplies nothing. **Not built.** |
| Chapman, Kratochvíl, Ebenhöh & Wilken, **Algebraic differentiation for fast sensitivity analysis of optimal flux modes in metabolic models**, [Bioinformatics 2025](https://academic.oup.com/bioinformatics/article/41/6/btaf287/8125804) (`DifferentiableMetabolism.jl`) | **The reason not to call this project a Jacobian estimator.** Implicit differentiation of a pruned GEM's KKT system gives every `d(flux)/d(param)` exactly — but at **7.48 s / 6.24 s** per full Jacobian against this repo's 1.70 s for a whole 21-member community, it is a *sensitivity-analysis* tool and this is a many-query one. Their pruning theorem manufactures the unique optimum implicit differentiation needs; **D4's elastic net already provides one**, which is what made the Head B Sobolev arm testable — and it is refuted (§7d of the framing doc): `dz/dc` is zero in all but 1–3 of ~180 directions and 96–99% of what remains is a proportional rescale Head A already supplies. |
| Bertsimas & Stellato (Part 3d); the GNN warm-start literature, e.g. [arXiv:2511.13174](https://arxiv.org/pdf/2511.13174) | The semi-amortized move the LP fallback makes available and has not taken: hand the solver a warm start or a predicted basis, so the surrogate *pays for* the LP it triggers rather than merely standing aside. B6's negative does not close this — it asked whether the limiting set discriminates failing states, not whether it saves simplex iterations. **Not built**; needs the optimal basis stored in the label shards. |
| Höffner, Harwood & Barton, **DFBAlab** / lexicographic LP, [BMC Bioinformatics 2014](https://link.springer.com/article/10.1186/s12859-014-0409-8); **interior-point** and **NLP/KKT** reformulations of dFBA, Comput. Chem. Eng. [2019](https://www.sciencedirect.com/science/article/abs/pii/S0098135418309190) / [2022](https://www.sciencedirect.com/science/article/abs/pii/S0098135422004343) | The non-learned competitors for the *smoothness* this project gets by fitting. They make the embedded LP unique (lexicographic) or smooth (IPM/KKT) so an ODE/NLP solver can integrate and differentiate it. **Any differentiability or speed claim must be stated against these, not against a cold simplex.** What they do not have is global concavity, which is what makes §13.2/§13.3 convex programs. |

---

## Part 5 — Open questions → what to read

| Open question | Read | What it should tell you |
| --- | --- | --- |
| The n=21 over-prediction tail that survived four design changes | Offline MBO review §conservatism; conformal one-sided certification; Balázs 2016 | Whether to bound the head from below, quantile-ise it, or design the composition around a one-sided estimator |
| The n=21 **under**-prediction that remains once the calibration is stripped | **Part 3c above** — SDDP cut validity; asymmetric-loss regression; one-sided conformal | That an under-prediction is a validity failure, not an accuracy one, and that the repair is a projection with a closed form rather than a refit |
| Cut selection: K inert above 1000, planes in the wrong place | **Part 3a above** — territory / Level 1 / LML1, test of usefulness | That 10× fewer well-chosen cuts perform the same, and that the selection criterion should use the point set you actually evaluate at |
| Head B: 12-26% flux error, concentrated on secretion and on depleted media | **Part 3d above** — mpLP critical regions; flux cone / principal flux modes; DC3; conserved moieties | That the argmin map is piecewise affine and confined to a ~25-dimensional subspace, so the fix is a reparametrisation (B1), not another loss term |
| Newton on a Hessian sum of rank ~20 of 365 | Qi & Sun, *A nonsmooth version of Newton's method*, Math. Prog. 1993; active-set / reduced-space methods | That a rank-deficient generalised Hessian is expected for a piecewise-linear value function, and the semismooth machinery built for it |
| Choosing `T`: accuracy vs smoothness for downstream HMC | LogSumExp near-optimality bound; Higham & Mary on LSE numerics | The exact `T·ln(K)` offset being paid, why it shrinks only logarithmically in K, where float precision stops you |
| Stage 4 active learning: which acquisition, which pool | Settles' survey; the adaptive-surrogate literature | That pool construction dominates acquisition choice, and that the bound-looseness score is a legitimate cheap alternative to query-by-committee |
| Non-unique duals under degeneracy (the 68.9%) | Bertsimas & Tsitsiklis ch. 4; Reznik & Segrè | That an "exact tangent" is a selection from a set, which reframes D4's elastic-net label uniqueness as choosing a canonical dual |

---

## What appears not to exist yet

A search across the convex-regression, amortized-optimization and metabolic-modelling
literatures turns up no published surrogate that is simultaneously **concave and monotone by
construction**, **supervised on LP duals**, and **composed per-organism into a community
simulation**.

**Sharpened 2026-09-06.** State the claim as a *relaxation*, not an estimator: a smooth,
globally concave, everywhere-defined relaxation of a piecewise-linear LP value function. The
estimator framing is contested by Chapman et al. 2025 (Part 4b); the relaxation framing is
contested only by the Barton-group reformulations, which are neither concave nor amortized.
And the sharpest demonstration of what the relaxation buys is §13.2b: the exact LP oracle's
gradient is a **subgradient selection at a kink**, and a trust-region method built on it
*stalls there* while the smoothed head walks through. Not accuracy — a usable direction at a
corner. The nearest neighbours each drop one leg: the reactive-transport ANN has no
structure, GroupMax and the max-affine statistics have no biology, and the community-FBA
methods keep the LP. If this is written up, that three-way intersection is the claim — and
the concavity is what makes the §13 medium-design programs convex, which is the part a
reviewer will care about more than the speedup.

---

*Compiled 2026-09-01; Part 4b and the hybrid stock-take added 2026-09-06.*
