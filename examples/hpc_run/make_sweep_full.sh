#!/usr/bin/env bash
# Regenerate sweep_full.csv — the M3b/v2 sweep, as five named arms.
#
#   ./make_sweep_full.sh > sweep_full.csv
#
# Not one cross product: the axes are arm-specific (w_grad matters for the ICNN
# heads, phi/k_code only for the deepsets, and the rows arm re-runs a *subset* of
# the others against a second label root). Each arm is its own `make_sweep.py`
# invocation so the flag string of every cell stays literal and greppable.
#
# LABELS_20K / LABELS_4K are the two label roots. The 4000-media one is what makes
# the rows axis real: the previous run shipped a `/path/to/...` placeholder there,
# every task silently read the 20000-media root, and `m4k__rf` came back bitwise
# identical to `m20k__rf`. Generate it with
#
#   OUTDIR=labels_out_4k ./run_labels.sh --label_media 4000 -c site.config
#
set -euo pipefail
cd "$(dirname "$0")"

L20="${LABELS_20K:-labels_out/labels}"
L4="${LABELS_4K:-labels_out_4k/labels}"
GEN=./make_sweep.py

# One header, then every arm's rows with theirs stripped. (An `emit` helper that
# tracked "have I printed the header yet" would not work: each pipeline stage is a
# subshell, so the flag never survives the first arm.)
echo "cell_id,arch,labels,args"
emit() { grep -v '^cell_id,'; }

# --- Arm A: the w_grad frontier. THE axis -- it is the only knob measured to move
# the gate. On `icnn-u`, one organism, 1500 epochs: w_grad 0 -> cosine 0.390 / R2
# 0.821, 1 -> 0.710/0.802, 3 -> 0.816/0.787, 10 -> 0.903/0.756, 30 -> 0.753/0.511.
# 10 is the best of six points, 10-30 was never probed, and it is ONE organism --
# the gate is the worst of 21. That is what this arm settles.
$GEN --labels "m20k=$L20" --arch icnn-u --width 128 --depth 3 \
     --w-grad 1,3,10,15,20 --epochs 1500 | emit

# --- Arm B: capacity, at the w_grad that makes the gradient term bind. The 2026-08
# run found width/depth inert, but measured it at w_grad 1 -- deep in the
# value-dominated regime where the Sobolev term barely pulls. This is the honest

# width 128 / depth 3 is Arm A's `g10` cell -- same knobs, so it is left out here
# rather than run twice under a second id (the generator dedupes within one
# invocation, not across them).
$GEN --labels "m20k=$L20" --arch icnn-u --width 512,1024 --depth 3,6 \
     --w-grad 10 --epochs 1500 | emit

# --- Arm C: controls at the same w_grad, so the frontier has a floor and a ceiling
# that were measured the same way. `icnn` is the x-space head the coordinate fix
# replaces; `mlp` drops both constraints; `rf` is the non-parametric floor.
$GEN --labels "m20k=$L20" --arch icnn --width 128 --depth 3 --w-grad 10 --epochs 1500 | emit
$GEN --labels "m20k=$L20" --arch mlp  --width 512 --depth 3 --w-grad 10 --epochs 1500 | emit
$GEN --labels "m20k=$L20" --arch rf   --n-estimators 100 --delta 0.05 | emit

# --- Arm D: CUT. It was `deepset-u-private`, phi {32,64} x k_code {16,64}, and at
# 5-11 h per organism per cell it was 420-920 cpu-h -- more than 80% of the run.
#
# It is the per-metabolite bet, and the full-width x-space rerun (2026-08-27, four
# `deepset-private` cells x 21 organisms) refuted it rather than merely deferring
# it. Best cell ph32/kc64: worst cosine 0.742 / median 0.810 / R2 0.564 against
# `icnn w128/d3`'s 0.678 / 0.755 / 0.541, for ~157 cpu-h a cell against 1.6. Its
# claimed conditioning advantage is gone (median Hessian cond 1e11-1e12, WORSE than
# the ICNN's 1e10 -- the 5.9e5 smoke number was an underfit head), and
# `cfs master-jacobian` shows that per-organism number does not reach §8 anyway.
# Above all it does not fit the per-metabolite cells better, which was the bet: the
# lift is uniform (<25 rows +0.012, 25-100 +0.062, 100-400 +0.020, >=400 +0.010)
# and on the 284 cells where the ICNN scores <0.5 deepset scores 0.312 vs 0.286.
# Both capacity axes are inert (paired median dcos -0.001 for k_code, +0.001 for
# phi).
#
# `deepset-u` would inherit the coordinate fix but NOT the larger lever: layer 1 is
# a per-metabolite scalar phi_m: R -> R^k, so `groupmax.init_from_tangents` (which
# seeds affine planes over the whole metabolite vector, 0.5982 -> 0.9733 cosine) has
# nothing to write into. Its other structural limit is untouched too -- mean pooling
# makes d(mu)/dx_m = <rho'(S), d(phi_m)/dx_m>, so the rest of the medium reaches the
# gradient pattern only through a `k_code`-wide vector, a narrow channel for what is
# an argmin across metabolites.
#
# The arch is built, tested and registered (`--arch deepset-u{,-private}`). To
# reopen the question cheaply, run ONE organism (CR626927.1) at ph32/kc64,
# `--w-grad 10`, ~7.5 h, and compare against `icnn-u` 0.903 and seeded `groupmax-u`
# 0.973 on the same held-out media; below 0.903 the arm is dead. To restore the
# full arm:
#
#   $GEN --labels "m20k=$L20" --arch deepset-u-private --width 512 --depth 3 \
#        --w-grad 10 --epochs 1500 --emb-dim 8 --phi-hidden 32,64 --k-code 16,64

# --- Arm F: does depth/width buy anything ONCE THE PIECES ARE SEEDED? This is the
# only place that asks -- Arm G is width 1 / depth 1 throughout, where the head is
# exactly `min_k(a_k.w + c_k)` and nothing composes.
#
# It was 9 cells of random-init group/temperature grid. That grid is cut because it
# is now known to measure the wrong thing: at width 128 / depth 3 from random init,
# `groupmax-u` scores cosine 0.7117 at T=0.1 and 0.7602 at T=0.03 -- both with a
# median Hessian condition of **exactly 0**, i.e. the head collapsed to a single
# affine piece. The collapse is not a quirk of the narrow width-1 configuration; it
# is what random init does to a group-max head generally, and Arm G's A/B
# establishes it more cheaply.
#
# What is left is the seeded version at the same temperatures as Arm G, so the two
# are comparable at matched init and T and the only difference is the architecture
# above the first layer. The seed is a *warm start* here rather than a reproduction
# -- wider than width 1, the head is a non-negative sum of group-wise minima, not
# one global minimum (see `groupmax.init_from_tangents`). Dropped along with the
# random cells: the `--gm-group` axis at width 128, which is largely redundant with
# Arm G's `K` since both move the total number of affine pieces.
$GEN --labels "m20k=$L20" --arch groupmax-u --width 128 --depth 3 \
     --w-grad 10 --epochs 1500 --gm-group 8 --gm-temp 0.01,0.03,0.1 \
     --gm-init labels | emit

# --- Arm G: the same head SEEDED from the labels' own supporting hyperplanes,
# ranked by active-set frequency, instead of from noise. At width 1 / depth 1 the
# head IS min_k(a_k.w + c_k) and the seed reproduces the pruned tangent model
# outright (unit-tested), so training starts at held-out cosine ~0.95 rather than
# at random. `--gm-group` is the number of affine pieces K: 100 ranked planes
# already match ~2000 random ones.
#
# The arm exists because random init measurably does not get there. Matched A/B on
# CR626927.1 (width 1, depth 1, K=250, T=0.03, same seed, only `--gm-init`
# differing): labels 0.9733 cosine / 0.996 R2, random 0.5982 / 0.4795 with a median
# Hessian condition of exactly 0 -- i.e. the head collapsed to a single affine piece
# and 249 of 250 planes never became active.
#
# **That control is NOT in this sweep.** The `--gm-init random` cells were cut for
# queue budget, so the attribution above rests on one organism measured on a laptop.
# If a reviewer needs it across the roster, add `,random` back to the flag below: it
# doubles the arm to 12 cells and ~35 cpu-h.
#
# **`T` is an accuracy knob and nothing else** -- measured 2026-08-26, and this arm
# used to say the opposite. It was pointed at the *blunt* half of the axis
# (0.03-0.3) on the theory that conditioning binds before accuracy does: the seeded
# head reaches cosine 0.9733 at T=0.03 with a median Hessian condition of 1.9e24,
# and §8.4 Newton-solves under `positive_semidefinite_tag`. That inference was from
# the wrong matrix. `cfs master-jacobian` reports the spectrum of what Newton
# actually inverts, `sum_i X_i H_i` over the 365 shared exchanges, at real held-out
# media: after a diagonal (Jacobi) preconditioner it carries curvature in ~10-25 of
# 365 directions, and that count does not improve with T (22 at 0.01, 20 at 0.03,
# 13 at 0.3). Going 30x blunter costs cosine 0.951 -> 0.833 and buys nothing Newton
# can use -- the Hessian sum is singular either way, and the `inflow(c)` supply term
# is what makes the solve well-posed, with `cond = 1 + top_ev/lam` set by the supply
# model rather than by the head.
#
# So the range moves to the *sharp* half, which is where accuracy lives, and 0.3 is
# dropped for collapsing (cosine 0.83 / R2 0.74 on the untrained tangent model).
# 0.01 and 0.03 score within 0.001 of each other untrained and 0.1 is past the knee,
# so these three points span it. Treat the untrained numbers as a lower bound: the
# smoothing bias is roughly a constant per active set, so fine-tuning can absorb
# much of it into the intercepts, which an untrained tangent model cannot -- the
# trained knee should sit blunter than 0.03, which is the reason 0.1 stays in.
$GEN --labels "m20k=$L20" --arch groupmax-u --width 1 --depth 1 \
     --w-grad 10 --epochs 1500 --gm-group 100,1000 --gm-temp 0.01,0.03,0.1 \
     --gm-init labels | emit

# --- Arm H: the re-anchor pass, and the seed axis it exposed. `--gm-reanchor N`
# re-seeds the least-used planes from the worst-fit rows' tangents mid-training.
# Seeding fixes *initialisation*; planes still drift dead during training and
# cannot revive (their softmax weight and their gradient are both exp(-gap/T), and
# the measured gaps are 0.2-1.5 at T=0.03-0.1). The loss does not object: the
# neighbour that takes over fits the value to 0.1%, so only the slope is wrong --
# predicted d(mu)/dw 1e-9 against a true 1.3e-3.
#
# Measured, AAXE02 x 5 seeds, grp1000 T=0.03, only --gm-reanchor differing:
# worst 0.927 -> 0.962, mean 0.945 -> 0.966, sd 0.0149 -> 0.0029, p05 0.500 ->
# 0.724. The cells it revives are exactly the dead ones (`EX_o2_e` 0.500/0.605/
# 0.676 -> 0.92-0.93; `EX_thr__L_e` 0.613/0.820/0.858 -> 0.964-0.975), healthy
# cells and value R2 are untouched, and it costs under 2% of runtime.
#
# The seed axis is here because that spread is the real finding: at sd 0.015 per
# organism against 0.007 between Arm G's top three cells, a *single-seed* cell
# ranking is noise. Three seeds per cell is the minimum that says so.
$GEN --labels "m20k=$L20" --arch groupmax-u --width 1 --depth 1 \
     --w-grad 10 --epochs 1500 --gm-group 1000 --gm-temp 0.03 \
     --gm-init labels --gm-reanchor 0,3 --seed 0,1,2 | emit

# --- Arm E: the rows axis, for real this time. 4000 vs 20000 media on the two heads
# that bracket the question plus the forest. Prediction on file, so this is a real
# test: rows will NOT help. Train-vs-held-out gap at the w_grad optimum is 0.005
# cosine / 0.013 R2 -- the head underfits, and rows only buy variance.
$GEN --labels "m4k=$L4" --arch icnn-u --width 128 --depth 3 --w-grad 10 --epochs 1500 | emit
$GEN --labels "m4k=$L4" --arch icnn   --width 128 --depth 3 --w-grad 10 --epochs 1500 | emit
$GEN --labels "m4k=$L4" --arch rf     --n-estimators 100 --delta 0.05 | emit
