"""``cfs {qc, freeze-index, degeneracy, active-subspace, generate, train-value, ...}``.

M0-M3 of the v2 plan (``docs/design/community-fba-surrogates-plan-v2.md``). The
M0-M2 subcommands need the ``data`` extra (cobra; ``qc`` also needs the ``memote``
CLI; ``generate`` also needs ``pyarrow``). The Nextflow ``QC_MODELS`` process runs
``qc`` then ``freeze-index``; ``DEGENERACY_SURVEY`` runs ``degeneracy``. §4
sampling is ``active-subspace`` (the sensitivity sweep) then ``generate`` (label
shards -> parquet by organism × eps).

``train-value`` and ``baseline-rf`` are M3 and are the odd ones out: they read
those label shards rather than any model, so they need the ``jax`` extra (the
forest only needs sklearn) and no solver stack, and take ``--labels``/``--index``
instead of ``--roster``. CLI-only for now — the heads train in minutes on a
laptop, so there is no Nextflow stage until M4 wants a cluster.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    """Return the top-level parser with one subparser per M0/M1 command."""
    parser = argparse.ArgumentParser(prog="cfs", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    qc = sub.add_parser("qc", help="M0: EGC pre-flight + MEMOTE over the roster.")
    qc.add_argument("--roster", type=Path, required=True, help="CSV: genome_id, model_path.")
    qc.add_argument("--outdir", type=Path, required=True, help="Output directory.")
    qc.add_argument("--no-memote", action="store_true", help="Skip MEMOTE (EGC gate only).")

    fi = sub.add_parser("freeze-index", help="M0: derive + freeze the metabolite index.")
    fi.add_argument("--roster", type=Path, required=True, help="CSV: genome_id, model_path.")
    fi.add_argument(
        "--out",
        type=Path,
        default=Path("config/metabolite_index.json"),
        help="Index JSON path (default config/metabolite_index.json).",
    )

    dg = sub.add_parser("degeneracy", help="M1: exchange-FVA degeneracy survey (decides D4).")
    dg.add_argument("--roster", type=Path, required=True, help="CSV: genome_id, model_path.")
    dg.add_argument("--outdir", type=Path, required=True, help="Output directory.")
    dg.add_argument("--alpha", default="1.0,0.7", help="Comma-separated growth fractions.")
    dg.add_argument("--n-media", type=int, default=50, help="Media sampled per organism.")
    dg.add_argument("--seed", type=int, default=0)

    asp = sub.add_parser("active-subspace", help="§4.2: per-organism sensitive-metabolite sweep.")
    asp.add_argument("--roster", type=Path, required=True, help="CSV: genome_id, model_path.")
    asp.add_argument("--out", type=Path, required=True, help="Subspaces JSON (fed to generate).")
    asp.add_argument("--tol", type=float, default=1e-3, help="Relative mu_max-drop threshold.")

    gen = sub.add_parser("generate", help="§4.5: solve label shards -> parquet by (organism, eps).")
    gen.add_argument("--roster", type=Path, required=True, help="CSV: genome_id, model_path.")
    gen.add_argument("--index", type=Path, required=True, help="Frozen metabolite_index.json.")
    gen.add_argument("--outdir", type=Path, required=True, help="Parquet shard root.")
    gen.add_argument("--n-media", type=int, help="Override media per organism (default 20000).")
    gen.add_argument(
        "--bg-perturb",
        type=float,
        help="Fraction of media whose background is perturbed off its rich level, "
        "over a random share of it (default 0.10). This is the community regime — "
        "raise it for a round meant to cover §8.1 media.",
    )
    gen.add_argument(
        "--mid-mu",
        type=float,
        help="Share of the bulk budget spent on mid-`mu` media: a community-sized "
        "share of A_i between each metabolite's own onset and its 50%%-recovery "
        "point (default 0.15). B2 -- the band E1 showed the labels had emptied.",
    )
    gen.add_argument(
        "--scales",
        type=Path,
        help="JSON {genome_id: {exchange: scale}} from "
        "design.limiting_scales — fallback bands for the metabolites the LP "
        "demand probe finds no limiting regime for (§4.7).",
    )
    gen.add_argument(
        "--no-probe",
        action="store_true",
        help="Skip the §4.7 demand probe; "
        "band placement then falls back to --scales, roster median, 1.0.",
    )
    gen.add_argument(
        "--focus-weights",
        type=Path,
        help="JSON {genome_id: {exchange: weight}} "
        "from `cfs topup` — skews the focus budget toward the metabolites the "
        "trained head gets measurably wrong (§4.6).",
    )
    gen.add_argument(
        "--round",
        type=int,
        default=0,
        dest="round_idx",
        help="Top-up round index; >0 writes part.round<n>.parquet alongside the "
        "base shards instead of overwriting them.",
    )
    gen.add_argument(
        "--media",
        type=Path,
        help="NPZ with `media` (n x |exchanges| concentrations), `exchanges` and "
        "`index_hash` — label exactly these media instead of sampling a design. "
        "Use with --round N. This is how the media the §8.1 composition actually "
        "visits get labelled (`make_traj_pool.py`).",
    )
    gen.add_argument("--seed", type=int, default=0)

    tu = sub.add_parser(
        "topup", help="§4.6: held-out diagnostics -> focus weights for the next generate round."
    )
    tu.add_argument(
        "--diagnostics", type=Path, required=True, help="diagnostics.json from train-value."
    )
    tu.add_argument(
        "--labels",
        type=Path,
        required=True,
        help="Label root holding the <id>.subspace.json sidecars.",
    )
    tu.add_argument("--out", type=Path, required=True, help="Focus-weights JSON.")
    tu.add_argument(
        "--floor",
        type=float,
        default=0.25,
        help="Share reserved for metabolites already predicted well.",
    )

    tv = sub.add_parser("train-value", help="M3: train Head A (concave mu_max) on label shards.")
    tv.add_argument("--labels", type=Path, required=True, help="Label shard root (§4.5).")
    tv.add_argument("--index", type=Path, required=True, help="Frozen metabolite_index.json.")
    tv.add_argument("--out", type=Path, required=True, help="Checkpoint + diagnostics dir.")
    tv.add_argument("--eps", type=float, default=1e-3, help="Which eps family level to train on.")
    tv.add_argument(
        "--arch",
        default="icnn",
        choices=[
            "icnn",
            "icnn-u",
            "deepset",
            "deepset-private",
            "deepset-u",
            "deepset-u-private",
            "groupmax-u",
            "mlp",
        ],
        help="Head architecture. `mlp` is unconstrained — a ceiling "
        "measurement, not a usable head.",
    )
    tv.add_argument(
        "--emb-dim", type=int, default=8, help="Metabolite embedding width (deepset only)."
    )
    tv.add_argument(
        "--gm-group",
        type=int,
        default=None,
        help="groupmax-u only: units per max group. width=1 depth=1 group=K is "
        "plain max-affine. Default 8.",
    )
    tv.add_argument(
        "--gm-temp",
        type=float,
        default=None,
        help="groupmax-u only: softmax temperature. Sets how much the head "
        "over-predicts a slow-growing medium (the smoothing sits ~T*ln(K) below "
        "the hard min, an absolute offset). Default 0.01, measured; 0.03 and "
        "0.003 are both worse.",
    )
    tv.add_argument(
        "--gm-init",
        choices=["random", "labels"],
        default=None,
        help="groupmax-u only: `labels` seeds the first layer from the label "
        "tangents ranked by active-set frequency, instead of from noise.",
    )
    tv.add_argument(
        "--gm-reanchor",
        type=int,
        default=0,
        help="groupmax-u only: number of re-anchor passes -- re-seed the least-used "
        "planes from the worst-fit rows' tangents, evenly spaced over the run. "
        "Seeding fixes initialisation; this fixes planes that go dead during it.",
    )
    tv.add_argument(
        "--gm-select",
        choices=["active-set", "level1"],
        default="active-set",
        help="Which label tangents fill the plane budget. 'active-set' buckets rows "
        "by dual support pattern; 'level1' is SDDP's cut selection (de Matos "
        "Level 1 / the territory algorithm) -- keep the cuts that are the active "
        "minimum at some trial point. Measured there: 10x fewer, well-chosen cuts "
        "represent the value function as well as the full set.",
    )
    tv.add_argument(
        "--gm-repair",
        action="store_true",
        help="After training, reset every plane's intercept to the tightest value "
        "that keeps it above every training label (`groupmax.repair_intercepts`). "
        "Restores SDDP's cut-validity invariant, which training breaks: an "
        "under-prediction is proof the head has left the outer-approximation "
        "family. Exact only at --width 1 --depth 1; closed form, no refit.",
    )
    tv.add_argument(
        "--gm-eval-temp",
        type=float,
        default=None,
        help="Ship the head at this temperature instead of --gm-temp: the soft "
        "argmax is needed to train (gradient must reach every plane) and not to "
        "predict. The smoothing sits c*T*ln(n_active) below the hard min, and "
        "--gm-repair cancels it with one uniform lift sized by the *max* over "
        "training rows -- so at a starved medium, where a single plane is active, "
        "the lift is uncancelled and the head reads high by a constant. That "
        "constant is the low-`mu` floor, and it is proportional to T: on the n=1 "
        "titration 0.0107 -> 0.0011 -> 0.0001 over T 1e-2/1e-3/1e-4, with worst "
        "held-out grad cosine 0.909 -> 0.952 and §8.1's n=21 log-X 0.175 -> 0.009. "
        "Use with --gm-repair; costs curvature (P3), which does not reach §8.4.",
    )
    tv.add_argument(
        "--gm-trial-media",
        type=Path,
        default=None,
        help="community_holdout.npz whose media are the Level 1 trial points. The "
        "point set is the lever: ranking over community-regime media puts planes "
        "where §8.1 evaluates. Default: the organism's training rows.",
    )
    tv.add_argument(
        "--phi-hidden",
        type=int,
        default=None,
        help="deepset only: `phi` trunk width. Default derives it from "
        "--width (width // 8), which was a laptop-runtime choice.",
    )
    tv.add_argument(
        "--k-code", type=int, default=None, help="deepset only: pooled code width (default 16)."
    )
    tv.add_argument("--width", type=int, default=128)
    tv.add_argument("--depth", type=int, default=3)
    tv.add_argument("--epochs", type=int, default=400)
    tv.add_argument("--batch", type=int, default=512)
    tv.add_argument("--lr", type=float, default=3e-3)
    tv.add_argument("--w-grad", type=float, default=1.0, help="Sobolev term weight (§7.1).")
    tv.add_argument(
        "--w-rel",
        type=float,
        default=0.0,
        help="Weight on the *relative* value error, on top of the absolute MSE. "
        "The MSE alone leaves the head over-predicting every medium below 75%% of "
        "max mu (median +98%% below 5%%), which is what a slow member costs the §8 "
        "composition. 0.3 removes the bias at no cost in grad_cosine or R2.",
    )
    tv.add_argument(
        "--w-under",
        type=float,
        default=0.0,
        help="Weight on a one-sided relative penalty for *under*-prediction. "
        "mu_max is concave and the head is a min of affine pieces, so a row where "
        "mu_hat < mu proves a plane has drifted below the target. The probe_lo "
        "design under-predicts 53-68%% of its bottom-5%%-mu training rows (0.2%% "
        "before it), and a slow member predicted near 0 reads as dead in §8.1.",
    )
    tv.add_argument(
        "--w-tau",
        type=float,
        default=0.5,
        help="Expectile level for the value loss (asymmetric least squares): "
        "residuals on the under-predicting side get weight tau, the rest 1-tau. "
        "0.5 is the plain MSE, bit for bit. Unlike --w-under's hinge, which only "
        "sees rows already in violation, this reweights every row and so moves the "
        "*slopes* -- which is what --gm-repair provably cannot fix. Choose it from "
        "`value_under_rate` in the diagnostics, not from a composition run.",
    )
    tv.add_argument(
        "--w-prox",
        type=float,
        default=0.0,
        help="Proximal weight anchoring the first-layer slopes to the tangents "
        "--gm-init labels seeded them with: a stability centre, in the sense of "
        "level/proximal bundle methods. Needs a seeded groupmax head. Motivated by "
        "measurement: frozen cuts win the n=21 tail and gradient training wins the "
        "bulk, and cut selection is exhausted, so how far the slopes may leave the "
        "duals is the remaining lever.",
    )
    tv.add_argument(
        "--gm-temp-final",
        type=float,
        default=None,
        help="groupmax-u only: anneal the temperature from --gm-temp to this over "
        "the run, in 3 geometric steps. Low T is what stops the head over-predicting "
        "slow media; a low *fixed* T trains worse, so this separates the two.",
    )
    tv.add_argument(
        "--x-scale-from",
        type=Path,
        default=None,
        help="Pin the input coordinate to this checkpoint's `x_scale` instead of "
        "recomputing it from these rows (§8.6f trap 1). Required to extend an "
        "existing head with new label rounds -- otherwise every round moves `x` "
        "under it (P14) and the two are not comparable.",
    )
    tv.add_argument("--seed", type=int, default=0)
    tv.add_argument(
        "--organisms",
        help="Comma-separated genome_ids to stack (default: every shard under "
        "--labels). One organism per job is what the sweep fans out; only the "
        "shared-trunk `deepset` pools anything across the stack.",
    )

    tb = sub.add_parser("train-behaviour", help="M4: train Head B (exchange fluxes) on labels.")
    tb.add_argument("--labels", type=Path, required=True, help="Label shard root (§4.5).")
    tb.add_argument("--index", type=Path, required=True, help="Frozen metabolite_index.json.")
    tb.add_argument("--out", type=Path, required=True, help="Checkpoint + diagnostics dir.")
    tb.add_argument("--eps", type=float, default=1e-3, help="Which eps family level to train on.")
    tb.add_argument("--width", type=int, default=256)
    tb.add_argument("--depth", type=int, default=3)
    tb.add_argument("--epochs", type=int, default=300)
    tb.add_argument("--batch", type=int, default=512)
    tb.add_argument("--lr", type=float, default=3e-3)
    tb.add_argument(
        "--x-scale-from",
        type=Path,
        default=None,
        help="Pin the input coordinate to this checkpoint's `x_scale` instead of "
        "recomputing it from these rows (§8.6f trap 1). Required to extend an "
        "existing head with new label rounds -- otherwise every round moves `x` "
        "under it (P14) and the two are not comparable.",
    )
    tb.add_argument("--seed", type=int, default=0)
    tb.add_argument("--organisms", help="Comma-separated genome_ids (default: every shard).")
    tb.add_argument(
        "--w-mm",
        type=float,
        default=0.0,
        help="Weight on a one-sided hinge against §3.3's uptake bound "
        "`z_m >= -Vmax_m * u_m` (`behaviour.mm_floor`). Every label satisfies it, "
        "so a prediction below it is a *provable* violation -- same shape and same "
        "reason as Head A's --w-under. `compose.dfba` already projects onto the "
        "bound at inference; the projection's bite ranks the 10 §8.1 communities "
        "by trajectory error (0.000 at the best cell, 0.41/0.28 at the two worst), "
        "with individual predictions 13x outside it, so the net is spending "
        "capacity on outputs the LP cannot produce. Default 0 (off).",
    )
    tt = sub.add_parser(
        "train-traj",
        help="§8.6g(3): fine-tune Head B on stored community trajectories (endpoint loss).",
    )
    tt.add_argument("--value", type=Path, required=True, help="Head A checkpoint (frozen).")
    tt.add_argument(
        "--behaviour", type=Path, required=True, help="Head B checkpoint to start from."
    )
    tt.add_argument("--out", type=Path, required=True, help="Fine-tuned Head B checkpoint dir.")
    tt.add_argument(
        "--runs",
        required=True,
        help="Comma-separated `cfs community` output dirs supplying the true "
        "trajectories. **Never the 10 benchmark communities** -- the M5 numbers are "
        "quoted on those; use the 16 n=15 sets, as §8.6d's round 2 did.",
    )
    tt.add_argument("--epochs", type=int, default=20)
    tt.add_argument("--lr", type=float, default=1e-5)
    tt.add_argument(
        "--clip",
        type=float,
        default=1.0,
        help="Global-norm gradient clip. The premise check measured d(logX)/dz "
        "reaching ~300 through 40 Euler steps, so this is load-bearing, not hygiene.",
    )
    tt.add_argument(
        "--w-anchor",
        type=float,
        default=0.0,
        help="Weight on mean squared parameter drift from the starting head, "
        "normalised per leaf. Unconstrained, 60 epochs on 32 trajectories buy 9%% "
        "of the trajectory loss and take held-out label R2 from 0.577 to -31.98, "
        "with both community gates 2.5-7x worse. Default 0 (off) records that.",
    )
    tt.add_argument(
        "--w-label",
        type=float,
        default=0.0,
        help="Weight on the per-state label loss, evaluated on a random minibatch "
        "beside the trajectory term. The two see different things: only the "
        "trajectory term sees the endpoint, and only the label term sees a single "
        "organism's flux vector rather than the pool sum. Needs --labels/--index.",
    )
    tt.add_argument(
        "--w-traj",
        type=float,
        default=1.0,
        help="Weight on the trajectory term. `--w-traj 0 --w-label W` is the "
        "control that attributes a joint run's result to the trajectory half "
        "rather than to the label rounds the starting head had not seen.",
    )
    tt.add_argument("--labels", type=Path, help="Label shard root, for --w-label.")
    tt.add_argument("--index", type=Path, help="Frozen metabolite_index.json, for --w-label.")
    tt.add_argument("--batch", type=int, default=256, help="Label-term minibatch (media/step).")
    tt.add_argument("--seed", type=int, default=0)

    tb.add_argument(
        "--basis-var",
        type=float,
        default=0.0,
        help="B1 (§8.6f): emit coordinates in the label flux subspace instead of "
        "one free flux per exchange. The head's output layer becomes the rank of "
        "the training specific-flux matrix at this explained-variance cutoff -- "
        "12-39 of 138-259 exchanges at the default, a basis that reconstructs "
        "held-out truth to 0.1-0.5%% where the trained head manages 9-26%%. Every "
        "conservation relation is a zero-variance direction it discards for free. "
        "**Measured and refuted**: the composition is unchanged at every size and "
        "held-out worst R2 falls 0.935 -> 0.920, so the default is 0 (off, the "
        "full-width head). 0.9999 is the cutoff that gives rank 12-39.",
    )

    cm = sub.add_parser(
        "community", help="M5/§8.1: compose the frozen heads into communities vs the LP."
    )
    cm.add_argument("--roster", type=Path, required=True, help="CSV: genome_id, model_path.")
    cm.add_argument("--labels", type=Path, required=True, help="Label root (for the subspaces).")
    cm.add_argument(
        "--value",
        type=Path,
        required=True,
        help="Head A checkpoint dir, or a comma-separated list: the composition "
        "then uses the pointwise min over them (C4 — valid for an upper-bound "
        "family; only the first dir's metadata is read).",
    )
    cm.add_argument("--behaviour", type=Path, required=True, help="Head B checkpoint dir.")
    cm.add_argument("--out", type=Path, required=True, help="Report + trajectory dir.")
    cm.add_argument(
        "--communities",
        help="Semicolon-separated member lists, e.g. 'A,B;C,D,E'. Default: sample --sizes.",
    )
    cm.add_argument(
        "--sizes",
        default="2,2,2,5",
        help="Community sizes to sample from the checkpoint's roster when "
        "--communities is not given.",
    )
    cm.add_argument("--steps", type=int, default=100, help="Euler steps (one LP/organism each).")
    cm.add_argument(
        "--doublings",
        type=float,
        default=4.0,
        help="Horizon, in doublings of the fastest member at t=0.",
    )
    cm.add_argument(
        "--biomass",
        type=float,
        default=None,
        help="Total initial biomass (gDW/L). Default: solved for, so the pool "
        "empties exactly at the end of the horizon.",
    )
    cm.add_argument("--eps", type=float, default=1e-3, help="Elastic-net level for the LP truth.")
    cm.add_argument("--scales", type=Path, default=None, help="Band scales JSON for the medium.")
    cm.add_argument(
        "--fallback-depth",
        type=float,
        default=0.0,
        help="§8.6g(4): solve the true LP for any member whose predicted depletion "
        "depth `mu_hat(t)/mu_hat(0)` falls below this, and use it for that step. "
        "Measured offline at 0.9: fires on 24%% of member-steps and captures 72%% "
        "of the accumulated |d log X| (lift 3.0x). 0 disables it (default).",
    )
    cm.add_argument(
        "--fallback-media",
        type=Path,
        default=None,
        help="Write the states the fallback fired at to this .npz, in the layout "
        "`cfs generate --media` reads. That is the self-labelling half: label "
        "them, then retrain with `x_scale` pinned to the current checkpoint.",
    )
    cm.add_argument("--seed", type=int, default=0)

    ch = sub.add_parser(
        "community-holdout",
        help="A1/§8.5: make or score a community-regime held-out label set — the "
        "one ruler a change to the sampling design cannot move (P24).",
    )
    ch.add_argument("action", choices=["make", "score"])
    ch.add_argument("--out", type=Path, required=True, help="Report dir (make: the npz too).")
    ch.add_argument("--roster", type=Path, help="make: CSV genome_id, model_path.")
    ch.add_argument("--labels", type=Path, help="make: label root, for the subspaces.")
    ch.add_argument("--index", type=Path, help="make: frozen metabolite_index.json.")
    ch.add_argument("--communities", help="make: semicolon-separated member lists.")
    ch.add_argument("--n-media", type=int, default=200, help="make: media per community.")
    ch.add_argument("--eps", type=float, default=1e-3)
    ch.add_argument("--holdout", type=Path, help="score: community_holdout.npz.")
    ch.add_argument("--value", type=Path, help="score: Head A checkpoint dir.")
    ch.add_argument("--seed", type=int, default=0)

    gx = sub.add_parser(
        "maximise-growth",
        help="§13.2/M10: convex medium design — maximise one organism's mu, then V5 it.",
    )
    gx.add_argument("--roster", type=Path, required=True, help="Roster YAML (for the V5 LP).")
    gx.add_argument("--labels", type=Path, required=True, help="Label root: start media (§4.3).")
    gx.add_argument("--value", type=Path, required=True, help="Head A checkpoint dir.")
    gx.add_argument("--out", type=Path, required=True, help="Report dir.")
    gx.add_argument("--organisms", required=True, help="Comma-separated genome_ids to design for.")
    gx.add_argument("--cases", type=int, default=20, help="(organism, medium draw) pairs.")
    gx.add_argument(
        "--budget-mult",
        type=float,
        default=1.0,
        help="Budget as a multiple of the start medium's own cost (default: reallocate it).",
    )
    gx.add_argument(
        "--trust-decades",
        type=float,
        default=0.5,
        help="P21 trust region: how far each metabolite may move from the start "
        "medium, in decades of the head's own input coordinate x. Unconstrained, "
        "the designer leaves the design and the true LP stops growing.",
    )
    gx.add_argument("--iters", type=int, default=300)
    gx.add_argument("--scales", type=Path, default=None, help="Band scales JSON for the draw.")
    gx.add_argument("--seed", type=int, default=0)

    ix = sub.add_parser(
        "interactions",
        help="§13.5/M13: explore the metabolic interactions a community can reach "
        "and design the media that facilitate them (exploratory — P22).",
    )
    ix.add_argument("--roster", type=Path, required=True, help="Roster YAML (for the V5 LP).")
    ix.add_argument("--labels", type=Path, required=True, help="Label root: media draws (§4.3).")
    ix.add_argument("--value", type=Path, required=True, help="Head A checkpoint dir.")
    ix.add_argument("--behaviour", type=Path, required=True, help="Head B checkpoint dir.")
    ix.add_argument("--out", type=Path, required=True, help="Report dir.")
    ix.add_argument(
        "--communities", required=True, help="Semicolon-separated member lists, 'A,B;C,D,E'."
    )
    ix.add_argument("--draws", type=int, default=64, help="Media surveyed per community.")
    ix.add_argument(
        "--starts", type=int, default=4,
        help="Multistart count. E is non-concave, so the spread across starts is part "
        "of the answer, not overhead.",
    )
    ix.add_argument(
        "--trust-decades", type=float, default=0.5,
        help="P21 trust region in the head's own input coordinate, intersected over "
        "members. Unconstrained, the designer leaves the design.",
    )
    ix.add_argument(
        "--budget-mult", type=float, default=1.0,
        help="Budget as a multiple of the start medium's own cost (default: reallocate it).",
    )
    ix.add_argument("--iters", type=int, default=120)
    ix.add_argument("--alpha", type=float, default=1.0, help="Growth fraction for Head B.")
    ix.add_argument(
        "--verify-steps", type=int, default=8,
        help="Trust-region iterations with the true LP as the acceptance test; 0 "
        "disables. Without it the ascent optimises a magnitude the head "
        "over-predicts by 1.6x to infinity and the true rate does not follow; with "
        "it the designed medium cannot be worse than its start under the LP. "
        "Measured over five 2-member cells: 5/5 cells improve their true rate at a "
        "median +27%%, against 2/5 and -6%% unverified, and the designed medium's "
        "E_hat/E_true falls 2.40 -> 1.67. Costs one FBA per member per iteration.",
    )
    ix.add_argument(
        "--no-verify", action="store_true",
        help="Skip the true-LP round-trip. Only for a structure-only survey — the "
        "objective is on flux magnitude, which is Head B's weakest axis (P22).",
    )
    ix.add_argument(
        "--buffered", default="EX_h_e,EX_h2o_e",
        help="Species the vessel holds, not the community: pinned at a saturating "
        "concentration and not counted as interactions. A chemostat is "
        "pH-controlled and aqueous, so protons and water are supplied by the "
        "buffer and the solvent — an experimenter cannot dial them, and a proton "
        "one member secretes goes into the buffer rather than into another "
        "member. Not cosmetic: with them counted, E is proton exchange — EX_h_e "
        "alone was 97.6%% of one community's true rate. CO2/O2/NH4/Pi are "
        "deliberately absent: nothing buffers those and they are real "
        "cross-feeding currencies. '' buffers nothing.",
    )
    ix.add_argument(
        "--no-screen", action="store_true",
        help="Seed the multistart by the head's own E instead of by the LP's. "
        "Measured Spearman(E_hat, E_true) over 64 draws on one community: -0.053, "
        "with every E_hat-seeded start at a true rate of zero — so this seeds "
        "where the head is most optimistic, which is what the search exploits.",
    )
    ix.add_argument(
        "--seed-mode", choices=("draws", "candidate"), default="candidate",
        help="'draws': random §4.3 media, and whether one contains a handover is "
        "luck. 'candidate': enumerate the metabolites the labels say some member "
        "secretes and another takes up (no LP), then seed one start per one with "
        "that metabolite's uptake bound opened — so the multistart covers every "
        "reachable link by construction. Measured on the roster: 11-13 candidate "
        "metabolites for a pair, 62 for all 21, against 444 exchanges.",
    )
    ix.add_argument(
        "--inhibition", type=Path, default=None,
        help="§13.11/M16: JSON mapping exchange id -> equilibrium concentration, "
        "in the medium's own units, plus an optional \"default\" key applied to "
        "every other (unbuffered) exchange — per P30 an unparameterised exchange "
        "is modelled as infinitely tolerant of its own product and the LP routes "
        "flux through exactly those, so the layer must be complete. Turns on thermodynamic product inhibition in "
        "the **true LP only** — secretion capacity falls affinely to zero as the "
        "external concentration reaches equilibrium, so a member's waste inhibits "
        "itself and its neighbours, which is the negative interaction §13.5 "
        "otherwise cannot express. The heads are unchanged and nothing is "
        "relabelled, so omitting this is plain FBA bit for bit and the two arms "
        "are directly comparable. Affine on purpose: the hyperbolic Ki form is "
        "convex and would cost §13.2/§13.3 their convexity (P30/P31).",
    )
    ix.add_argument(
        "--box", type=int, default=3,
        help="With --seed-mode candidate: extra starts per candidate drawn inside "
        "the envelope of every labelled medium where the donor secreted that "
        "metabolite. That region is 1e-5 to 1e-19 of the design volume, so a §4.3 "
        "draw never lands in it, and inside it the secretion rate is 1.1-1704x the "
        "base rate -- largest exactly on the rare metabolites sampling misses. "
        "0 disables.",
    )
    ix.add_argument("--scales", type=Path, default=None, help="Band scales JSON for the draws.")
    ix.add_argument("--seed", type=int, default=0)

    mm = sub.add_parser(
        "minimal-medium",
        help="§13.3/M11: the smallest medium every member grows on, then V6 it.",
    )
    mm.add_argument("--roster", type=Path, required=True, help="Roster YAML (for the V6 LPs).")
    mm.add_argument(
        "--labels", type=Path, required=True, help="Label root: rich start media (§4.3)."
    )
    mm.add_argument("--value", type=Path, required=True, help="Head A checkpoint dir.")
    mm.add_argument("--out", type=Path, required=True, help="Report dir.")
    mm.add_argument("--organisms", required=True, help="Comma-separated genome_ids: the community.")
    mm.add_argument("--cases", type=int, default=5, help="Rich medium draws.")
    mm.add_argument(
        "--target-frac",
        type=float,
        default=0.5,
        help="Growth floor, as a fraction of each member's mu on the rich medium.",
    )
    mm.add_argument(
        "--lp-repair",
        action="store_true",
        help="After designing, solve the true LP and raise components back to the "
        "rich level until every member meets its floor. Catches what "
        "--keep-essential cannot: that audit is over *single* knockouts, so it is "
        "blind to an alternative-route set — a design that zeroes both "
        "EX_trp__L_e and EX_indole_e kills a member with neither essential alone.",
    )
    mm.add_argument(
        "--cuts",
        type=int,
        default=0,
        metavar="N",
        help="Kelley cutting planes on the growth constraints: at most N rounds of "
        "design -> true-LP check -> add the members' tangents. Each round costs one "
        "FBA per member and tightens the model toward the true feasible set. 0 "
        "(default) is the single design every earlier number was measured with.",
    )
    mm.add_argument(
        "--no-milp",
        action="store_true",
        help="Skip the per-organism exact MILP reference (cobra minimal_medium).",
    )
    mm.add_argument(
        "--all-metabolites",
        action="store_true",
        help="Design over every exchange, not just the members' active subspaces. "
        "P21: the answer then leaves the design and the true LP stops growing.",
    )
    mm.add_argument(
        "--no-keep-essential",
        action="store_true",
        help="Let the design zero a metabolite the true LP calls essential. Head A "
        "cannot represent essentiality, so this reproduces the V6 failure.",
    )
    mm.add_argument("--scales", type=Path, default=None, help="Band scales JSON for the draw.")
    mm.add_argument("--seed", type=int, default=0)

    sim = sub.add_parser(
        "simulate", help="§13.1: integrate one community forward — batch or chemostat, no LP."
    )
    sim.add_argument("--value", type=Path, required=True, help="Head A checkpoint dir.")
    sim.add_argument("--behaviour", type=Path, required=True, help="Head B checkpoint dir.")
    sim.add_argument("--out", type=Path, required=True, help="Report + trajectory dir.")
    sim.add_argument("--organisms", required=True, help="Comma-separated genome_ids.")
    sim.add_argument("--medium", type=Path, default=None, help="JSON {exchange_id: mM}.")
    sim.add_argument(
        "--labels", type=Path, default=None, help="Label root: draw a §4.3 medium instead."
    )
    sim.add_argument("--scales", type=Path, default=None, help="Band scales JSON for the draw.")
    sim.add_argument(
        "--abundances", default=None, help="Comma-separated initial shares (default: equal)."
    )
    sim.add_argument(
        "--biomass",
        type=float,
        default=None,
        help="Total inoculum, gDW/L. Default: solved for, so the pool empties at the end.",
    )
    sim.add_argument("--steps", type=int, default=200)
    sim.add_argument("--hours", type=float, default=None, help="Horizon (default: --doublings).")
    sim.add_argument("--doublings", type=float, default=4.0)
    sim.add_argument(
        "--dilution", type=float, default=0.0, help="Chemostat D (1/h). 0 = batch culture."
    )
    sim.add_argument("--feed", type=Path, default=None, help="Feed JSON (default: the medium).")
    sim.add_argument(
        "--stiff",
        action="store_true",
        help="Integrate with BDF in log X instead of explicit Euler. A chemostat "
        "transient is stiff — the pool equilibrates fast while biomass grows "
        "slowly — and Euler either ratchets X up or washes out to the spurious "
        "extinction state depending only on the step.",
    )
    sim.add_argument("--seed", type=int, default=0)

    ss = sub.add_parser(
        "steady-state",
        help="M12/§13.4: Newton-solve a chemostat fixed point — coexistence, "
        "stability, invasion and feed sensitivities. No LP.",
    )
    ss.add_argument("--value", type=Path, required=True, help="Head A checkpoint dir.")
    ss.add_argument("--behaviour", type=Path, required=True, help="Head B checkpoint dir.")
    ss.add_argument("--out", type=Path, required=True, help="Report dir.")
    ss.add_argument("--organisms", required=True, help="Comma-separated genome_ids.")
    ss.add_argument("--medium", type=Path, default=None, help="Feed JSON {exchange_id: mM}.")
    ss.add_argument(
        "--labels", type=Path, default=None, help="Label root: draw a §4.3 feed instead."
    )
    ss.add_argument("--scales", type=Path, default=None, help="Band scales JSON for the draw.")
    ss.add_argument(
        "--dilution", type=float, default=None, help="D (1/h). Default: --dilution-frac of max mu."
    )
    ss.add_argument("--dilution-frac", type=float, default=0.2)
    ss.add_argument(
        "--roster",
        type=Path,
        default=None,
        help="Roster TSV. Given: solve the true LP for the residual and keep the "
        "surrogate for the Jacobian (inexact Newton). An equilibrium is one state, "
        "so this costs G solves per iteration, not per Jacobian column.",
    )
    ss.add_argument("--eps", type=float, default=1e-3, help="Elastic-net eps for the LP residual.")
    ss.add_argument(
        "--mix-mu-rel",
        type=float,
        default=None,
        help="With --roster: solve both and keep the surrogate for any member "
        "whose mu agrees with the LP within this relative tolerance, substituting "
        "the LP only where they diverge. Keeps the residual consistent with the "
        "Jacobian where it can be, at the cost of solving everything anyway.",
    )
    ss.add_argument(
        "--mix-z-rel",
        type=float,
        default=None,
        help="Second --mix-mu-rel trigger, relative on z in the 2-norm. Needed in "
        "practice: a mu-only trigger fires on nothing exactly where Head B is "
        "wrong, since Head A is the accurate head.",
    )
    ss.add_argument(
        "--jac-temp",
        type=float,
        default=None,
        help="Evaluate Head A at this temperature in the Jacobian only. Free by "
        "construction: the residual keeps the shipped temperature and decides the "
        "fixed point, the Jacobian only decides the rate.",
    )
    ss.add_argument(
        "--solver",
        default="newton",
        choices=["newton", "krylov", "df-sane", "hybr", "broyden1"],
        help="Inner root finder. The scipy methods are kept but refuted: none of "
        "them knows X > 0, so they converge on the trivial washout root.",
    )
    ss.add_argument(
        "--ptc",
        type=float,
        default=0.0,
        help="Levenberg-Marquardt trust region: initial damping, escalated when "
        "backtracking fails. 0 = plain Newton, bit for bit. The globalisation a "
        "line search cannot supply — backtracking shortens a bad direction, it "
        "does not replace one.",
    )
    ss.add_argument(
        "--d-steps",
        type=int,
        default=0,
        help="Continuation rungs in D, walked down from the transcritical end "
        "(D = mu_max at the feed, where c = c_feed and X = 0 exactly). 0 = the "
        "bisection warm start alone.",
    )
    ss.add_argument(
        "--readmits",
        type=int,
        default=1,
        help="Re-admissions per member before the anti-cycling ban becomes "
        "permanent. 0 = the original permanent ban, which returns states an "
        "excluded member can invade on 4 of 10 roster cells.",
    )
    ss.add_argument(
        "--invade-rel",
        type=float,
        default=1e-2,
        help="An excluded member counts as invading only if mu_j(c*) exceeds D by "
        "this fraction. Default 1e-2 because Head A's own mu error at a fixed "
        "point is 1e-4 to 9e-3: below that, exclusion and coexistence are not "
        "distinguishable and the tie is not a solver failure.",
    )
    ss.add_argument(
        "--seed-mode",
        default="monoculture",
        choices=["monoculture", "bisect"],
        help="Warm start. 'monoculture' also seeds from each member's own "
        "equilibrium and keeps the state with the most negative invasion margin, "
        "stopping at the first strictly valid one; 'bisect' is the single "
        "bisection start alone, which returns a strictly invadable state on 2 of "
        "4 converging roster cells and collapses the pool on the 2 hardest.",
    )
    ss.add_argument(
        "--seed-probes",
        type=int,
        default=4,
        help="With --seed-mode monoculture: how many members to probe, fastest "
        "grower at the feed first. Caps the cost at 2N extra solves regardless of "
        "community size; the 21-member cell does not finish in an hour uncapped.",
    )
    ss.add_argument(
        "--warm-start",
        type=Path,
        default=None,
        help="steady_state.npz from another solve: start from its (c, X) instead "
        "of the bisection. Members it does not name enter dead, so the loop's own "
        "re-admission test decides whether they can invade — which is the direct "
        "test for multiple fixed points.",
    )
    ss.add_argument(
        "--fd-check",
        type=int,
        default=20,
        help="V4: feed components to finite-difference. 0 = off.",
    )
    ss.add_argument("--seed", type=int, default=0)

    rf = sub.add_parser("baseline-rf", help="Random-forest baseline on the same split and gate.")
    rf.add_argument("--labels", type=Path, required=True, help="Label shard root (§4.5).")
    rf.add_argument("--index", type=Path, required=True, help="Frozen metabolite_index.json.")
    rf.add_argument("--out", type=Path, required=True, help="Diagnostics dir.")
    rf.add_argument("--eps", type=float, default=1e-3)
    rf.add_argument("--n-estimators", type=int, default=100)
    rf.add_argument(
        "--delta",
        type=float,
        default=0.05,
        help="Finite-difference step, in units of each metabolite's kink scale.",
    )
    rf.add_argument("--seed", type=int, default=0)
    rf.add_argument(
        "--organisms",
        help="Comma-separated genome_ids to stack (default: every shard under "
        "--labels). One organism per job is what the sweep fans out; only the "
        "shared-trunk `deepset` pools anything across the stack.",
    )
    mj = sub.add_parser(
        "master-jacobian",
        help="V-for-§8: spectrum of the master Jacobian sum_i X_i H_i at real media.",
    )
    mj.add_argument("--labels", type=Path, required=True, help="Label shard root (§4.5).")
    mj.add_argument("--index", type=Path, required=True, help="Frozen metabolite_index.json.")
    mj.add_argument("--out", type=Path, required=True, help="Report dir.")
    mj.add_argument(
        "--checkpoint",
        type=Path,
        help="A `train-value` output dir. Omitted: seed a width-1 `groupmax-u` from "
        "the label tangents at each --gm-temp, which needs no training.",
    )
    mj.add_argument("--eps", type=float, default=1e-3)
    mj.add_argument("--gm-temp", default="0.01,0.03,0.1,0.3,1.0", help="Temperatures to scan.")
    mj.add_argument("--gm-group", type=int, default=250, help="Affine pieces K when seeding.")
    mj.add_argument("--n-media", type=int, default=6, help="Held-out media to evaluate at.")
    mj.add_argument("--seed", type=int, default=0)
    mj.add_argument("--organisms", help="Comma-separated genome_ids (default: every shard).")

    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")
    # cobra logs one INFO line per solve ('Compartment `C_e` sounds like an
    # external compartment'); at 20k media x 21 organisms that is ~60 MB of
    # stderr per task and nothing else.
    logging.getLogger("cobra").setLevel(logging.WARNING)
    args = build_parser().parse_args(argv)
    organisms = (
        [g for g in getattr(args, "organisms", None).split(",") if g]
        if getattr(args, "organisms", None)
        else None
    )

    if args.command == "master-jacobian":
        from cfs.validate.master_jacobian import run

        run(
            args.labels,
            args.index,
            args.out,
            checkpoint=args.checkpoint,
            eps=args.eps,
            gm_temp=[float(v) for v in args.gm_temp.split(",") if v],
            gm_group=args.gm_group,
            n_media=args.n_media,
            organisms=organisms,
            seed=args.seed,
        )
        return 0

    if args.command == "community-holdout":
        from cfs.validate import community_holdout as CH

        if args.action == "make":
            rep = CH.make(
                args.roster,
                args.labels,
                args.index,
                args.out,
                communities=[c.split(",") for c in args.communities.split(";") if c],
                n_media=args.n_media,
                eps=args.eps,
                seed=args.seed,
            )
        else:
            rep = CH.score(args.holdout, args.value, args.out)
        print(json.dumps(rep, indent=2))
        return 0

    if args.command == "train-value":
        # The only subcommand that reads labels rather than models: no roster, no
        # solver stack, and the jax extra instead of the data one.
        from cfs.surrogate.train import run

        diagnostics = run(
            args.labels,
            args.index,
            args.out,
            eps=args.eps,
            arch=args.arch,
            width=args.width,
            depth=args.depth,
            epochs=args.epochs,
            batch=args.batch,
            lr=args.lr,
            w_grad=args.w_grad,
            w_rel=args.w_rel,
            w_under=args.w_under,
            w_tau=args.w_tau,
            w_prox=args.w_prox,
            emb_dim=args.emb_dim,
            phi_hidden=args.phi_hidden,
            gm_group=args.gm_group,
            gm_temp=args.gm_temp,
            gm_init=args.gm_init,
            gm_reanchor=args.gm_reanchor,
            gm_select=args.gm_select,
            gm_repair=args.gm_repair,
            gm_eval_temp=args.gm_eval_temp,
            gm_trial_media=args.gm_trial_media,
            gm_temp_final=args.gm_temp_final,
            k_code=args.k_code,
            seed=args.seed,
            organisms=organisms,
            x_scale_from=args.x_scale_from,
        )
        print(json.dumps(diagnostics, indent=2))
        return 0 if diagnostics["passed"] else 1

    if args.command == "train-behaviour":
        from cfs.surrogate.behaviour import run as run_b

        print(
            json.dumps(
                run_b(
                    args.labels,
                    args.index,
                    args.out,
                    eps=args.eps,
                    width=args.width,
                    depth=args.depth,
                    epochs=args.epochs,
                    batch=args.batch,
                    lr=args.lr,
                    w_mm=args.w_mm,
                    basis_var=args.basis_var,
                    seed=args.seed,
                    organisms=organisms,
                    x_scale_from=args.x_scale_from,
                )["summary"],
                indent=2,
            )
        )
        return 0

    if args.command == "train-traj":
        from cfs.surrogate.traj import run as run_t

        report = run_t(
            args.value,
            args.behaviour,
            [p for p in args.runs.split(",") if p],
            args.out,
            epochs=args.epochs,
            lr=args.lr,
            clip=args.clip,
            w_anchor=args.w_anchor,
            w_label=args.w_label,
            w_traj=args.w_traj,
            labels_dir=args.labels,
            index=args.index,
            batch=args.batch,
            seed=args.seed,
        )
        print(json.dumps({k: v for k, v in report.items() if k != "history"}, indent=2))
        return 0

    if args.command == "community":
        from cfs.compose.dfba import run as run_c

        if args.communities:
            comms = [[g for g in c.split(",") if g] for c in args.communities.split(";") if c]
        else:
            first = Path(str(args.value).split(",")[0])
            gids = json.loads((first / "value_heads.json").read_text())["genome_ids"]
            rng = __import__("numpy").random.default_rng(args.seed)
            comms = [
                sorted(rng.choice(gids, size=int(n), replace=False).tolist())
                for n in args.sizes.split(",")
                if n
            ]
        report = run_c(
            args.roster,
            args.labels,
            args.value,
            args.behaviour,
            args.out,
            communities=comms,
            steps=args.steps,
            doublings=args.doublings,
            biomass=args.biomass,
            eps=args.eps,
            seed=args.seed,
            scales=args.scales,
            fallback_depth=args.fallback_depth,
            fallback_media=args.fallback_media,
        )
        print(json.dumps(report["summary"], indent=2))
        return 0

    if args.command == "simulate":
        import numpy as np

        from cfs.compose.dfba import simulate

        report = simulate(
            args.value,
            args.behaviour,
            args.out,
            organisms=[g for g in args.organisms.split(",") if g],
            labels_dir=args.labels,
            medium=args.medium,
            abundances=(
                np.array([float(a) for a in args.abundances.split(",")])
                if args.abundances
                else None
            ),
            biomass=args.biomass,
            steps=args.steps,
            hours=args.hours,
            doublings=args.doublings,
            dilution=args.dilution,
            feed=args.feed,
            stiff=args.stiff,
            seed=args.seed,
            scales=args.scales,
        )
        print(json.dumps(report, indent=2))
        return 0

    if args.command == "steady-state":
        from cfs.science.steady import run as run_steady

        report = run_steady(
            args.value,
            args.behaviour,
            args.out,
            organisms=[g for g in args.organisms.split(",") if g],
            labels_dir=args.labels,
            medium=args.medium,
            dilution=args.dilution,
            dilution_frac=args.dilution_frac,
            roster_path=args.roster,
            eps=args.eps,
            mix_mu_rel=args.mix_mu_rel,
            mix_z_rel=args.mix_z_rel,
            jac_temp=args.jac_temp,
            solver=args.solver,
            ptc=args.ptc,
            d_steps=args.d_steps,
            readmits=args.readmits,
            invade_rel=args.invade_rel,
            warm_start=args.warm_start,
            seed_mode=args.seed_mode,
            seed_probes=args.seed_probes,
            seed=args.seed,
            scales=args.scales,
            fd_check=args.fd_check,
        )
        print(json.dumps(report, indent=2))
        return 0

    if args.command == "maximise-growth":
        from cfs.science.growth import run as run_growth

        report = run_growth(
            args.roster,
            args.labels,
            args.value,
            args.out,
            organisms=[g for g in args.organisms.split(",") if g],
            cases=args.cases,
            trust_decades=args.trust_decades,
            budget_mult=args.budget_mult,
            iters=args.iters,
            seed=args.seed,
            scales=args.scales,
        )
        print(json.dumps({k: v for k, v in report.items() if k != "cases"}, indent=2))
        return 0 if report["passed"] else 1

    if args.command == "interactions":
        from cfs.science.interaction import run as run_interactions

        report = run_interactions(
            args.roster,
            args.labels,
            args.value,
            args.behaviour,
            args.out,
            communities=[
                [g for g in part.split(",") if g]
                for part in args.communities.split(";")
                if part.strip()
            ],
            draws=args.draws,
            starts=args.starts,
            trust_decades=args.trust_decades,
            budget_mult=args.budget_mult,
            iters=args.iters,
            alpha=args.alpha,
            seed=args.seed,
            scales=args.scales,
            verify=not args.no_verify,
            verify_steps=args.verify_steps,
            buffered=tuple(m for m in args.buffered.split(",") if m),
            screen=not args.no_screen,
            seed_mode=args.seed_mode,
            box=args.box,
            inhibition=args.inhibition,
        )
        print(json.dumps({k: v for k, v in report.items() if k != "cells"}, indent=2))
        return 0 if report.get("passed", True) else 1

    if args.command == "minimal-medium":
        from cfs.science.minimal import run as run_minimal

        report = run_minimal(
            args.roster,
            args.labels,
            args.value,
            args.out,
            organisms=[g for g in args.organisms.split(",") if g],
            cases=args.cases,
            target_frac=args.target_frac,
            lp_repair=args.lp_repair,
            cut_rounds=args.cuts,
            seed=args.seed,
            scales=args.scales,
            milp=not args.no_milp,
            all_metabolites=args.all_metabolites,
            keep_essential=not args.no_keep_essential,
        )
        skip = ("cases", "knockout_audit")
        print(json.dumps({k: v for k, v in report.items() if k not in skip}, indent=2))
        return 0 if report["passed"] else 1

    if args.command == "baseline-rf":
        # A measurement, not a gate: it always exits 0, however it scores.
        from cfs.surrogate.baseline import run as run_rf

        print(
            json.dumps(
                run_rf(
                    args.labels,
                    args.index,
                    args.out,
                    eps=args.eps,
                    n_estimators=args.n_estimators,
                    delta=args.delta,
                    seed=args.seed,
                    organisms=organisms,
                ),
                indent=2,
            )
        )
        return 0

    if args.command == "topup":
        # Also label-side, no roster: last run's held-out error -> next run's focus
        # budget (§4.6). Metabolites that never limited have no error to read, so
        # they get the floor share rather than nothing — the probe (§4.7) has by
        # now given them a band worth sampling.
        from cfs.sampling.active_subspace import load_subspaces
        from cfs.sampling.design import topup_weights
        from cfs.surrogate.ensemble import unmeasured_metabolites

        diagnostics = json.loads(args.diagnostics.read_text())
        subspaces = {}
        for path in sorted(Path(args.labels).glob("*.subspace.json")):
            subspaces.update({gid: s.active for gid, s in load_subspaces(path).items()})
        unmeasured = unmeasured_metabolites(diagnostics, subspaces)

        weights = {}
        for gid, d in diagnostics["per_organism"].items():
            w = topup_weights(d.get("per_limiting_metabolite", {}), floor=args.floor)
            blind = unmeasured.get(gid, [])
            if blind:
                # Give the never-measured metabolites the same per-metabolite share
                # the floor gives a well-predicted one, then renormalise.
                share = args.floor / max(len(w) + len(blind), 1)
                w = {**w, **dict.fromkeys(blind, share)}
                total = sum(w.values())
                w = {ex: v / total for ex, v in w.items()}
            weights[gid] = w
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(weights, indent=2, sort_keys=True))
        print(
            f"wrote {args.out} — {len(weights)} organisms, "
            f"{sum(len(v) for v in unmeasured.values())} never-measured metabolites"
        )
        return 0

    # Imports are deferred so `--help` works without the data extra installed.
    from surrogate_mgem.data import read_roster

    roster = read_roster(args.roster)

    if args.command == "qc":
        from cfs.groundtruth.qc import qc_roster

        summary = qc_roster(roster, args.outdir, memote=not args.no_memote)
        print(json.dumps(summary, indent=2))
        return 0

    if args.command == "freeze-index":
        from cfs.groundtruth.index import derive_index_from_roster, write_index

        result = derive_index_from_roster(roster)
        digest = write_index(result, args.out)
        print(
            f"wrote {args.out} — {len(result.index)} exchanges "
            f"({len(result.shared)} shared, {len(result.private)} private) "
            f"sha256={digest}"
        )
        return 0

    if args.command == "degeneracy":
        from cobra.io import read_sbml_model

        from cfs.validate.degeneracy import sample_media, survey_roster

        alphas = tuple(float(a) for a in args.alpha.split(","))
        media_per_model = {
            gm.genome_id: sample_media(
                read_sbml_model(str(gm.model_path)), args.n_media, seed=args.seed
            )
            for gm in roster
        }
        rec = survey_roster(roster, media_per_model, args.outdir, alphas=alphas)
        print(json.dumps(rec, indent=2))
        return 0

    if args.command == "active-subspace":
        from cobra.io import read_sbml_model

        from cfs.sampling.active_subspace import active_subspace, write_subspaces

        subspaces = [
            active_subspace(read_sbml_model(str(gm.model_path)), gm.genome_id, tol=args.tol)
            for gm in roster
        ]
        write_subspaces(subspaces, args.out)
        print(
            f"wrote {args.out} — {sum(len(s.active) for s in subspaces)} active "
            f"metabolites across {len(subspaces)} organisms"
        )
        return 0

    if args.command == "generate":
        from dataclasses import replace

        from cfs.sampling.design import SamplingConfig
        from cfs.sampling.generate import generate_roster

        cfg = SamplingConfig(seed=args.seed, probe=not args.no_probe)
        if args.bg_perturb is not None:
            cfg = replace(cfg, frac_bg_perturb=args.bg_perturb)
        if args.mid_mu is not None:
            cfg = replace(cfg, frac_mid_mu=args.mid_mu)
        if args.n_media is not None:
            cfg = replace(cfg, n_media=args.n_media)
        media = None
        if args.media is not None:
            import numpy as np

            from cfs.groundtruth.index import index_hash

            npz = np.load(args.media, allow_pickle=True)
            got, want = str(npz["index_hash"]), index_hash(args.index)
            if got != want:  # P13: a silent index change must not fuse two designs
                raise SystemExit(f"--media index_hash {got} != {want} for {args.index}")
            ex = [str(e) for e in npz["exchanges"]]
            media = [dict(zip(ex, (float(v) for v in row), strict=True)) for row in npz["media"]]
        scales = json.loads(args.scales.read_text()) if args.scales else None
        focus = json.loads(args.focus_weights.read_text()) if args.focus_weights else None
        shards = generate_roster(
            roster,
            args.index,
            args.outdir,
            cfg,
            scales=scales,
            focus_weights=focus,
            round_idx=args.round_idx,
            media=media,
        )
        print(
            json.dumps(
                {
                    s.genome_id: {"n_media": s.n_media, "shards": [str(p) for p in s.paths]}
                    for s in shards
                },
                indent=2,
            )
        )
        return 0

    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
