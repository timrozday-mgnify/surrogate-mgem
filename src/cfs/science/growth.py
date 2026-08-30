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
    f = 10.0**decades
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
) -> tuple[np.ndarray, list[float]]:
    """Projected gradient ascent with a backtracking step. Returns ``(c*, mu path)``.

    The step is adapted rather than fixed because ``d mu/dc`` spans five decades
    within one organism (§7): a step that moves the ions at all overshoots every
    carbon source by orders of magnitude.
    """
    c = project(c0.astype(np.float64), cost, budget, c_lo, c_hi)
    mu, g = mu_and_grad(sur, c, k)
    path = [mu]
    step = 0.1 * float(np.linalg.norm(c_hi)) / max(float(np.linalg.norm(g)), 1e-30)
    for _ in range(iters):
        cand = project(c + step * g, cost, budget, c_lo, c_hi)
        mu_c, g_c = mu_and_grad(sur, cand, k)
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

        c_star, path = maximise(
            sur, c0, k, cost=cost, budget=budget, c_lo=c_lo, c_hi=c_hi, iters=iters
        )
        model = cobra.io.read_sbml_model(str(roster[gid].model_path))
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
