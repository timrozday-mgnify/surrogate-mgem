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

# --- Arm D: deepset-u, the per-metabolite head with its coordinate fixed. THIS ARM
# IS THE COST -- `phi` is priced per metabolite, measured 5-11 h per organism
# against the ICNN's 5 minutes, so these four cells are most of the run. Drop them
# first if the queue budget is tight; Arms A-C answer the gate question on their
# own. The shared-trunk variant is deliberately absent: it OOM-killed in every cell
# last run, and sharing across organisms was measured to hurt (private beat shared
# on every metric), so it is not worth its cost twice.
$GEN --labels "m20k=$L20" --arch deepset-u-private --width 512 --depth 3 \
     --w-grad 10 --epochs 1500 --emb-dim 8 --phi-hidden 32,64 --k-code 16,64 | emit

# --- Arm F: groupmax-u -- kinks as the primitive rather than as a sum of smooth
# ridges. `min_k(a_k.u + c_k)` is what an LP value function IS, and a softplus ICNN
# spends many units approximating one corner; a max unit is one corner. The head
# nests max-affine exactly at width 1 / depth 1 (asserted in the unit tests).
#
# `--gm-temp` is the axis that matters and it is NOT just accuracy: curvature scales
# as 1/T, so this arm measures the accuracy-vs-conditioning frontier §8's Newton
# actually has to buy from. T -> 0 recovers the exact-but-Newton-hostile hard max
# (zero Hessian inside a piece: P3). Group size trades corners per unit against
# parameters.
#
# The temperature range is measured, not guessed: on the pruned label-tangent model
# (same hypothesis class, no optimiser in the way) T=0.01-0.03 keeps the hard min's
# accuracy -- cosine 0.95-0.96, R2 0.996 -- while T=0.1 is already past the knee
# (0.923/0.929) and T=0.3 collapses (0.83/0.79, R2 0.74). The old 0.03-0.3 range
# spent two of its three points below the knee.
$GEN --labels "m20k=$L20" --arch groupmax-u --width 128 --depth 3 \
     --w-grad 10 --epochs 1500 --gm-group 4,8,16 --gm-temp 0.01,0.03,0.1 | emit

# --- Arm E: the rows axis, for real this time. 4000 vs 20000 media on the two heads
# that bracket the question plus the forest. Prediction on file, so this is a real
# test: rows will NOT help. Train-vs-held-out gap at the w_grad optimum is 0.005
# cosine / 0.013 R2 -- the head underfits, and rows only buy variance.
$GEN --labels "m4k=$L4" --arch icnn-u --width 128 --depth 3 --w-grad 10 --epochs 1500 | emit
$GEN --labels "m4k=$L4" --arch icnn   --width 128 --depth 3 --w-grad 10 --epochs 1500 | emit
$GEN --labels "m4k=$L4" --arch rf     --n-estimators 100 --delta 0.05 | emit
