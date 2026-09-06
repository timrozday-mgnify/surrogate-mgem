"""M10 / §13.2 — maximise one member's growth rate over the medium.

    maximise  mu_k(c)   s.t.   cost . c <= B,   c_lo <= c <= c_hi

``mu_k`` is concave and non-decreasing in ``u``, and ``u = c/(Km+c)`` is concave
and increasing in ``c``, so the objective is concave in ``c`` and the feasible set
is a box cut by one hyperplane: projected gradient ascent converges to the global
optimum, and the gradient is the head's own analytic one. This is the use case the
current heads most nearly meet (worst held-out gradient cosine 0.956), because the
argmax depends on the gradient *direction* and not on the value level.

Two things follow from that and are relied on here:

* the output calibration (:mod:`cfs.surrogate.calibrate`) is an increasing scalar
  map, so it cannot move the argmax. It is applied when a ``mu`` is *reported* and
  ignored inside the ascent;
* every reported optimum is round-tripped through the true LP (V5/P4). An
  optimiser's whole job is to find where the surrogate is most optimistic, so the
  number that matters is not ``mu_hat(c*)`` but ``mu_true(c*)`` against
  ``mu_true(c0)``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

LOGGER = logging.getLogger("cfs.science.growth")


def mu_and_grad(sur, c: np.ndarray, k: int) -> tuple[float, np.ndarray]:
    """``(mu_hat, d mu_hat/dc)`` for organism ``k``, *uncalibrated*.

    Chain rule through the two input maps: ``dmu/dc = dmu/dx . dx/du . du/dc``,
    with ``dx/du = (1-x)^2/s`` (:func:`cfs.surrogate.train._du`) and
    ``du/dc = Km/(Km+c)^2``.
    """
    jnp = sur._jnp
    x = sur._x(c)  # (G, 1, M)
    mu, g = sur.mod.batched_value_and_grad(sur._vheads, jnp.asarray(x))
    xk = np.asarray(x[k, 0], dtype=np.float64)
    gk = np.asarray(g[k, 0], dtype=np.float64)
    dxdu = (1.0 - xk) ** 2 / sur.x_scale[k]
    dudc = sur.km / (sur.km + c) ** 2
    return float(mu[k, 0]) * float(sur.mu_scale[k]), gk * dxdu * dudc * float(sur.mu_scale[k])


def mu_reported(sur, c: np.ndarray, k: int) -> float:
    """``mu_hat`` with the checkpoint's output calibration applied."""
    from cfs.surrogate import calibrate

    raw = mu_and_grad(sur, c, k)[0] / float(sur.mu_scale[k])
    cal = calibrate.apply(np.array([[raw]]), sur.value_cal[k : k + 1])[0, 0]
    return float(max(cal * float(sur.mu_scale[k]), 0.0))


def project(
    y: np.ndarray, cost: np.ndarray, budget: float, c_lo: np.ndarray, c_hi: np.ndarray
) -> np.ndarray:
    """Euclidean projection onto ``{c_lo <= c <= c_hi, cost . c <= budget}``.

    ``clip(y - lam*cost, c_lo, c_hi)`` is the KKT form and its spend is monotone
    decreasing in ``lam``, so one bisection on the multiplier is the whole solve.
    ``cost . c_lo <= budget`` is assumed (it holds when the budget is the start
    medium's own cost and the trust region is centred on it).
    """
    c = np.clip(y, c_lo, c_hi)
    if float(cost @ c) <= budget:
        return c
    lam = 1.0
    while float(cost @ np.clip(y - lam * cost, c_lo, c_hi)) > budget:
        lam *= 2.0
        if lam > 1e12:  # `c_lo` alone already overspends
            return c_lo.copy()
    lo, hi = 0.0, lam
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if float(cost @ np.clip(y - mid * cost, c_lo, c_hi)) > budget:
            lo = mid
        else:
            hi = mid
    return np.clip(y - hi * cost, c_lo, c_hi)


def trust_box(sur, c0: np.ndarray, k: int, decades: float) -> tuple[np.ndarray, np.ndarray]:
    """P21: ``x`` within ``decades`` of ``x(c0)``, per metabolite, returned in ``c``.

    The designer's job is to find where the surrogate is most optimistic, and left
    alone it does: unconstrained, it pays for carbon by zeroing ~50 cheap
    metabolites at once, which is nowhere near any training medium and where the
    true LP does not grow at all — while the head still reports its optimum as an
    improvement. The region is in ``x``, the head's own input coordinate and the
    one §6.3's nearest-training-medium distance is measured in, and centred on a
    §4.3 draw, so staying inside it is staying inside the design.

    It is *multiplicative*, because §4.3's bands are, and because that is what
    stops the one failure the LP actually punishes: an additive radius still lets a
    trace metabolite at ``x ~ 0.1`` go to exactly zero, and dropping an essential
    one takes ``mu`` to 0 however good the rest of the medium is (measured: 3 of 20
    V5 cases, every one of them a case that zeroed something).

    The cost is that a metabolite the start medium has *none* of stays at zero, so
    this designs by reallocation and never by addition. Widen it, or seed the start
    medium, if adding one is the question.
    """
    x0 = np.asarray(sur._x(c0)[k, 0], dtype=np.float64)
    # `x` lives in (0, 1), so past a few decades the upper face is saturated at 1
    # and only the lower one moves; 300 is well past that and keeps 10**d finite.
    f = 10.0 ** min(float(decades), 300.0)
    lo, hi = x0 / f, np.clip(x0 * f, 0.0, 1.0 - 1e-9)
    return _c_of_x(sur, k, lo), _c_of_x(sur, k, hi)


def _c_of_x(sur, k: int, x: np.ndarray) -> np.ndarray:
    """Invert ``x = u/(u+s)``, ``u = c/(Km+c)``."""
    u = np.minimum(sur.x_scale[k] * x / (1.0 - x), 1.0 - 1e-12)
    return sur.km * u / (1.0 - u)


def maximise(
    sur,
    c0: np.ndarray,
    k: int,
    *,
    cost: np.ndarray,
    budget: float,
    c_lo: np.ndarray,
    c_hi: np.ndarray,
    iters: int = 300,
    tol: float = 1e-9,
    grad_shift: np.ndarray | None = None,
    c_ref: np.ndarray | None = None,
    cuts: list[tuple[float, np.ndarray, np.ndarray]] | None = None,
    head_lift: float = 0.0,
) -> tuple[np.ndarray, list[float]]:
    """Projected gradient ascent with a backtracking step. Returns ``(c*, mu path)``.

    The step is adapted rather than fixed because ``d mu/dc`` spans five decades
    within one organism (§7): a step that moves the ions at all overshoots every
    carbon source by orders of magnitude.

    ``grad_shift`` adds the affine term ``s . (c - c_ref)`` to the objective, which
    is how :func:`trf` makes the model *first-order consistent* with the true LP at
    the trust-region centre. An affine term cannot break concavity, so the
    subproblem is still a convex program with a unique optimum.

    ``cuts`` is the bundle form of the same idea and the better one:
    ``(mu_j, g_j, c_j)`` triples from true LP solves, entering as
    ``min_j [mu_j + g_j . (c - c_j)]`` alongside the head. Each is a supporting
    hyperplane of a concave function, hence an upper bound on ``mu_true``
    everywhere and tight at ``c_j``, so the min is concave, is exact at every
    visited point, and — unlike a single tangent — does not overstate the
    achievable gain when the centre sits on a kink. ``head_lift`` is a constant
    added to the head so the newest cut binds at the centre even if the head
    happens to read below the LP there; a constant moves no argmax and breaks no
    concavity.
    """
    shift = None if grad_shift is None else np.asarray(grad_shift, dtype=np.float64)
    ref = c0 if c_ref is None else np.asarray(c_ref, dtype=np.float64)

    def value_and_grad(y):
        v, gr = mu_and_grad(sur, y, k)
        if shift is not None:
            v, gr = v + float(shift @ (y - ref)), gr + shift
        v += head_lift
        for mu_j, g_j, c_j in cuts or ():
            vj = mu_j + float(g_j @ (y - c_j))
            if vj <= v:  # ties go to the cut: that is what pins consistency
                v, gr = vj, g_j
        return v, gr

    c = project(c0.astype(np.float64), cost, budget, c_lo, c_hi)
    mu, g = value_and_grad(c)
    path = [mu]
    step = 0.1 * float(np.linalg.norm(c_hi)) / max(float(np.linalg.norm(g)), 1e-30)
    for _ in range(iters):
        cand = project(c + step * g, cost, budget, c_lo, c_hi)
        mu_c, g_c = value_and_grad(cand)
        if mu_c > mu:
            c, mu, g = cand, mu_c, g_c
            step *= 1.5
            path.append(mu)
            if len(path) > 2 and path[-1] - path[-2] < tol * max(abs(mu), 1e-12):
                break
        else:
            step *= 0.5
            if step * float(np.linalg.norm(g)) < 1e-14:
                break
    return c, path


# --------------------------------------------------------------------------- #
# V5 round-trip
# --------------------------------------------------------------------------- #


def mu_true(model, exchanges: list[str], c: np.ndarray) -> float:
    """``mu_max`` from the true LP at medium ``c`` — §3.3 bounds, FBA only."""
    from cfs.groundtruth.solve import apply_mm_bounds, load_km_defaults

    with model:
        apply_mm_bounds(model, dict(zip(exchanges, c.tolist(), strict=True)), load_km_defaults())
        v = model.slim_optimize()
    return 0.0 if v is None or not np.isfinite(v) else float(v)


def lp_value_and_grad(model, exchanges: list[str], c: np.ndarray, km_cfg=None):
    """``(mu_max, d mu_max/dc)`` from **one** FBA — the LP is a first-order oracle.

    The optimal dual *is* the gradient, so the expensive model costs the same
    whether you want its value, its gradient, or both. Same sign convention and
    same clamp as :func:`cfs.surrogate.data._organism_arrays`: the stored dual is
    ``d mu/d(supply)`` negated, and it is the bound's sensitivity only where the
    bound binds -- elsewhere it is the metabolite's value in the network, which for
    a waste product is positive and would claim that more nutrient lowers growth.
    """
    from cfs.groundtruth.solve import apply_mm_bounds, km_for_exchange, load_km_defaults

    km_cfg = km_cfg if km_cfg is not None else load_km_defaults()
    conc = dict(zip(exchanges, c.tolist(), strict=True))
    with model:
        apply_mm_bounds(model, conc, km_cfg)
        sol = model.optimize()
        if sol.status != "optimal":
            return 0.0, np.zeros_like(c)
        mu = float(sol.objective_value or 0.0)
        pi = {
            ex.id: float(sol.shadow_prices[next(iter(ex.metabolites)).id])
            for ex in model.exchanges
        }
    g = np.zeros_like(c)
    for j, ex in enumerate(exchanges):
        cv = float(c[j])
        p = pi.get(ex, 0.0)
        if cv > 0.0 and p < -1e-9:
            km = km_for_exchange(ex, km_cfg)
            g[j] = -p * 1000.0 * km / (km + cv) ** 2
    return mu, g


def trf(
    sur,
    model,
    exchanges: list[str],
    c0: np.ndarray,
    k: int,
    *,
    cost: np.ndarray,
    budget: float,
    decades: float,
    iters: int = 300,
    max_it: int = 12,
    eta1: float = 0.1,
    eta2: float = 0.7,
    gamma_dec: float = 0.5,
    gamma_inc: float = 2.0,
    min_decades: float = 1e-2,
    max_decades: float = 8.0,
    mode: str = "bundle",
    km_cfg=None,
) -> tuple[np.ndarray, dict]:
    """Trust-region model management: converge to the **true** LP's optimum.

    Alexandrov, Dennis, Lewis & Torczon's first-order-consistency framework, in the
    glass-box/black-box form of Eason & Biegler (AIChE J 2016/2018). The surrogate
    is never trusted globally; at each iteration it is corrected to match the LP's
    value *and* gradient at the trust-region centre, and a ratio test decides
    whether the step is kept and whether the radius grows or shrinks. That is what
    buys convergence to a first-order critical point of the model we actually care
    about, and it is why this needs no accuracy gate on the head: M3's worst-case
    gradient cosine constrains a *globally* trusted surrogate, and the corrected
    model is exact at the centre by construction.

    Two properties of this problem make the correction unusually cheap and clean:

    * **the LP is a first-order oracle** -- the dual is the gradient, so one FBA per
      iteration supplies both halves of the consistency condition;
    * the correction is **affine**, so ``mu_hat + s.(c - c_k)`` is still concave in
      ``c`` and the subproblem is still the convex program of §13.0.

    It also turns P21 from a pitfall into the mechanism. A step that zeroes an
    essential metabolite has ``mu_true = 0``, hence ``rho < 0``, hence rejection and
    a smaller radius -- the designer is *supposed* to walk to where the surrogate is
    optimistic, and the ratio test is what makes that safe. The hand-tuned
    ``--trust-decades`` becomes a starting radius rather than a result.

    **Measured 2026-09-06, 20 V5 cases, paired against the single ascent:**

    | mode | median gain | max | better/eq/worse | max optimism | LPs |
    | --- | --- | --- | --- | --- | --- |
    | single ascent | +2.34% | 22.79x | -- | 0.0730 | 0 |
    | ``shift`` | +2.12% | 6.79x | 7/10/3 | 0.00712 | 219 |
    | **``bundle``** | **+2.35%** | **23.15x** | **7/11/2** | **0.00686** | **70** |

    ``bundle`` is the default and it dominates on both axes at once; ``shift`` is
    kept because *why* it fails is the more useful result:

    * **the LP's gradient at a kink is a subgradient *selection*.** ``mu_max`` is
      piecewise linear in ``u``, so the dual is one element of the subdifferential.
      Instrumented on the worst case under ``shift``: the radius shrinks 16x,
      ``predicted`` tracks it exactly, and ``actual`` stays **pinned at 0.0046** --
      ``rho`` never approaches 1, which cannot happen where the function is
      differentiable. TRF correctly refuses and halts *at the kink* (true ``mu``
      9.29 against the ascent's 22.0). The smoothed head walks through because the
      smoothing averages both sides: this is the mollification argument of
      :func:`cfs.groundtruth.solve.mu_curvature`, restated as an optimiser failure.
      **The bundle is the textbook remedy and it works here** -- the same case goes
      6.79 -> 23.15, and ``CP027002.1`` 0.0064 -> 0.0334;
    * **the additive correction is also the wrong form when the gradient spans
      eight decades.** ``d mu/dc`` reaches 1e8 on the ions, so ``shift = gt - g_h``
      oscillates between norm ~1 and ~5e6, and where it is large the linear term
      swamps the concave head: ``predicted`` reaches 129 against an actual 1.2. The
      bundle has no such term, which is why its rejects fall from 6-18 per case to
      0-1 and it needs 3x fewer LP solves.

    **What still limits ``bundle``: the *inner* solver, not the model.** The two
    remaining losses (``GCA_000164675.2`` 0.0171 -> 0.0143) were first blamed on the
    head reading below the truth at a designed medium; ``20hm_bands/bound_gap.py``
    **refutes that** -- the head is a valid upper bound at all 20 bundle optima and
    all 20 baseline optima, including both loss cases (gap +2.2e-05 and +1.9e-07).
    Since a cut is a valid upper bound too, ``min(head, cuts)`` is one everywhere,
    so the model's maximum over the region is at least the true maximum and the
    better point was inside the model's feasible set with a model value above what
    was returned. **The subproblem solver did not find its own model's maximum**:
    :func:`maximise` is projected subgradient ascent with a backtracking line
    search, and with cuts the objective is a nonsmooth ``min`` that stalls at its
    own kinks. The bundle fixes the *outer* kink and introduces an *inner* one,
    which is exactly why bundle methods solve their subproblem as an LP/QP over the
    epigraph rather than by subgradient steps. A softmin over the cuts -- what Head
    A already does internally -- is the cheap fix in this codebase's idiom. Not
    built.

    Expansion is on the ratio test alone, capped at ``max_decades``. The textbook
    rule also requires the step to reach the trust-region *face*, and that is wrong
    here: the binding constraint is usually the **budget**, so steps are interior,
    expansion never fires, and the radius ratchets down until the loop stops on
    ``min_decades`` while gains remain (measured: 9 of 20 cases ended at exactly six
    halvings, one losing 0.095 -> 0.069 of true gain).
    """
    c = project(c0.astype(np.float64), cost, budget, *trust_box(sur, c0, k, decades))
    f, gt = lp_value_and_grad(model, exchanges, c, km_cfg)
    bundle = [(f, gt, c.copy())]
    radius, n_lp, hist, rejects = decades, 1, [f], 0
    for _ in range(max_it):
        if radius < min_decades:
            break
        mu_h, g_h = mu_and_grad(sur, c, k)
        if mode == "bundle":
            # Lift the head so the centre's own cut binds there; below it, the
            # model would be consistent with neither the head nor the LP.
            kw = {"cuts": bundle, "head_lift": max(f - mu_h, 0.0)}
        else:
            kw = {"grad_shift": gt - g_h, "c_ref": c}

        def model_value(y, _kw=kw):
            v, _ = mu_and_grad(sur, y, k)
            if "grad_shift" in _kw:
                return v + float(_kw["grad_shift"] @ (y - _kw["c_ref"]))
            v += _kw["head_lift"]
            return min([v] + [mj + float(gj @ (y - cj)) for mj, gj, cj in _kw["cuts"]])

        lo, hi = trust_box(sur, c, k, radius)
        cand, _ = maximise(
            sur, c, k, cost=cost, budget=budget, c_lo=lo, c_hi=hi, iters=iters, **kw
        )
        predicted = model_value(cand) - model_value(c)
        if predicted <= 1e-12 * max(abs(f), 1e-12):
            break
        f_new, gt_new = lp_value_and_grad(model, exchanges, cand, km_cfg)
        n_lp += 1
        # A rejected step is still a true tangent, and keeping it is the whole
        # point of a bundle: it is what stops the next model overstating in the
        # direction that just failed.
        bundle.append((f_new, gt_new, cand.copy()))
        rho = (f_new - f) / predicted
        if rho >= eta1:
            c, f, gt = cand, f_new, gt_new
            hist.append(f)
            # Expand on the ratio test alone, capped. Gating expansion on the step
            # reaching the trust-region *face* is the textbook rule and it is wrong
            # here: the binding constraint is usually the **budget**, so the step is
            # interior, expansion never fires, and the radius can only ratchet down
            # until the loop stops on `min_decades` while still improving. Measured:
            # 9 of 20 cases ended at exactly 6 halvings, and one lost 0.095 -> 0.069
            # of true gain against the ungated run.
            if rho >= eta2:
                radius = min(radius * gamma_inc, max_decades)
        else:
            rejects += 1
            radius *= gamma_dec
    return c, {"n_lp": n_lp, "trf_iters": len(hist) - 1, "trf_rejects": rejects,
               "trf_radius": radius, "trf_cuts": len(bundle), "trf_path": hist}


def run(
    roster_path: Path,
    labels_dir: Path,
    value_dir: Path,
    out: Path,
    *,
    organisms: list[str],
    cases: int = 20,
    trust_decades: float = 0.5,
    budget_mult: float = 1.0,
    iters: int = 300,
    seed: int = 0,
    scales: Path | None = None,
    trf_iters: int = 0,
    trf_mode: str = "bundle",
) -> dict:
    """One case per (organism, medium draw): optimise, then V5 the answer."""
    import cobra

    from cfs.compose.dfba import Surrogate, community_medium
    from surrogate_mgem.data import read_roster

    roster = {gm.genome_id: gm for gm in read_roster(Path(roster_path))}
    Path(out).mkdir(parents=True, exist_ok=True)
    rows, media = [], []
    for n in range(cases):
        gid = organisms[n % len(organisms)]
        sur = Surrogate(value_dir, organisms=[gid])
        k = sur.genome_ids.index(gid)
        c0 = community_medium(labels_dir, [gid], sur.exchanges, seed + n, scales)
        c_lo, c_hi = trust_box(sur, c0, k, trust_decades)
        cost = np.ones_like(c0)
        budget = budget_mult * float(cost @ c0)

        model = cobra.io.read_sbml_model(str(roster[gid].model_path))
        if trf_iters:
            c_star, extra = trf(
                sur, model, sur.exchanges, c0, k, cost=cost, budget=budget,
                decades=trust_decades, iters=iters, max_it=trf_iters, mode=trf_mode,
            )
            path = extra.pop("trf_path")
        else:
            c_star, path = maximise(
                sur, c0, k, cost=cost, budget=budget, c_lo=c_lo, c_hi=c_hi, iters=iters
            )
            extra = {"n_lp": 0}
        t0, t1 = mu_true(model, sur.exchanges, c0), mu_true(model, sur.exchanges, c_star)
        rows.append(
            {
                "genome_id": gid,
                "seed": seed + n,
                "mu_hat_start": mu_reported(sur, c0, k),
                "mu_hat_opt": mu_reported(sur, c_star, k),
                "mu_true_start": t0,
                "mu_true_opt": t1,
                # V5: did the *true* LP improve, and by how much less than promised?
                "true_gain": t1 - t0,
                "true_gain_rel": (t1 - t0) / max(t0, 1e-12),
                "optimism": float(mu_reported(sur, c_star, k) - t1),
                "iterations": len(path) - 1,
                "spend": float(cost @ c_star) / max(budget, 1e-30),
                "n_changed": int((np.abs(c_star - c0) > 1e-9 * np.maximum(c0, 1.0)).sum()),
                "n_zeroed": int(((c0 > 0) & (c_star <= 0)).sum()),
                **extra,
            }
        )
        media.append((c0, c_star))
        LOGGER.info("case %d %s: true mu %.4f -> %.4f", n, gid, t0, t1)

    gain = np.array([r["true_gain"] for r in rows])
    rel = np.array([r["optimism"] / max(r["mu_true_opt"], 1e-12) for r in rows], dtype=np.float64)
    report = {
        "cases": rows,
        "n_cases": len(rows),
        # The gate: the surrogate's optimum has to be an improvement under the LP.
        "n_true_improved": int((gain > 0).sum()),
        "worst_true_gain": float(gain.min()),
        "worst_true_gain_rel": float(
            min(r["true_gain"] / max(r["mu_true_start"], 1e-12) for r in rows)
        ),
        "median_optimism_rel": float(np.median(rel)),
        # V5: the surrogate's optimum has to be at least as good under the LP as
        # the medium it started from. A flat case passes trivially and says so in
        # `n_true_improved`; a case that *loses* growth is the failure P4 predicts.
        "passed": bool(all(r["true_gain"] >= -1e-6 * max(r["mu_true_start"], 1e-12) for r in rows)),
    }
    np.savez_compressed(
        Path(out) / "media.npz",
        c_start=np.array([m[0] for m in media]),
        c_opt=np.array([m[1] for m in media]),
        exchanges=np.array(sur.exchanges),
        genome_ids=np.array([r["genome_id"] for r in rows]),
    )
    (Path(out) / "growth_max.json").write_text(json.dumps(report, indent=2))
    return report
