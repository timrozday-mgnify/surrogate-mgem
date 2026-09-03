"""M5 / §8.1 — dFBA composition, and its LP ground truth.

The composition operator is one line of the plan::

    dc/dt   = sum_i X_i z_i(c, alpha_i) + inflow(c)
    dX_i/dt = X_i mu_i(c)

Every community-level phenomenon in this framing — competition, cross-feeding,
succession, one organism's waste becoming another's substrate — is that sum. There
is no community LP and no joint objective: the coupling is entirely that Head B's
output for organism i lands in Head A's *input* for organism j, through the shared
pool. That is why §8.1 says build this first, and it is what makes the question
"are the heads good enough to be useful" answerable at all.

Two things are measured here, and they fail independently:

* **The right-hand side**, at states drawn from the true trajectory. This is the
  composition error with no integration in it: given exactly the medium the real
  community is in, does the surrogate agree on who grows and what they excrete?
* **The trajectory**, integrated independently by both. M5's gate is 1%.

The ground truth is per-organism FBA — :func:`cfs.groundtruth.solve.solve`, the
same call that made the labels — evaluated at the community's shared medium and
summed with the same weights. So the comparison isolates the surrogates: same
integrator, same step size, same initial state, same MM bounds. A joint community
LP is a *different model* (SteadyCom's equal-growth constraint, MICOM's tradeoff),
and mixing that difference into this measurement would make a Head B error and a
modelling choice indistinguishable. §8.2/§8.3 are where those framings belong.

Explicit Euler on purpose. It is the same map for both right-hand sides, it costs
one LP per organism per step (RK4 costs four), and a concentration that would go
negative is clipped at zero, which is where the LP's regime actually changes.
Halve ``--steps`` and the two trajectories move together; that is a check the CLI
reports rather than a claim made here.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cfs.surrogate import calibrate

LOGGER = logging.getLogger("cfs.compose.dfba")

# A community medium is one draw from the §4.3 design over the *union* of the
# members' active subspaces, so every metabolite sits in the band its own labels
# were generated in. Sampling outside that is a fair test of nothing.
_MEDIUM_SEED_STRIDE = 7919

# `cfs.surrogate.behaviour.VMAX` is the one definition: Head B now trains under the
# same bound this projects onto, and two copies could drift apart silently.


@dataclass
class Trajectory:
    t: np.ndarray  # (T+1,)
    c: np.ndarray  # (T+1, M)
    x: np.ndarray  # (T+1, G)
    mu: np.ndarray  # (T, G) growth rate used for each step
    dc: np.ndarray  # (T, M) pool derivative used for each step


class Surrogate:
    """Both frozen heads, evaluated together on the shared metabolite index."""

    def __init__(
        self,
        value_dir: Path,
        behaviour_dir: Path | None = None,
        organisms: list[str] | None = None,
    ):
        import jax.numpy as jnp

        from cfs.surrogate import behaviour as B
        from cfs.surrogate import train as T

        # C4: `--value a,b,c` is the min over several Head A seeds. A max-affine
        # head is an *upper* bound on `mu` wherever no tangent is nearby, so the
        # pointwise min of independently seeded heads is still a valid member of
        # the family and can only reduce the one-sided error. The first dir is the
        # primary: its metadata, and its heads, are what `growth`/`minimal` use.
        vdirs = [Path(d) for d in str(value_dir).split(",") if d]
        vheads, vmeta = T.load(vdirs[0])
        # §13.2 needs mu alone, so Head B is optional; `mu_and_z` then refuses.
        bheads, bmeta = (None, {}) if behaviour_dir is None else B.load(Path(behaviour_dir))
        for k in ("index_hash", "genome_ids", "exchanges"):
            if bheads is not None and vmeta[k] != bmeta[k]:
                raise ValueError(f"the two heads disagree on {k} (P13)")
        # x_scale is read off the labels, so two checkpoints trained on different
        # label roots compose into a silently wrong medium coordinate.
        if bheads is not None and not np.allclose(vmeta["x_scale"], bmeta["x_scale"]):
            raise ValueError("the two heads were trained on different x_scale (P14)")

        self.genome_ids = list(vmeta["genome_ids"])
        self.exchanges = list(vmeta["exchanges"])
        self.index_hash = vmeta["index_hash"]
        self.mod = T._ARCH[vmeta.get("arch", {}).get("arch", "icnn")]
        self._B = B
        self._vheads, self._bheads = vheads, bheads
        self._ens = [(self.mod, vheads, np.asarray(vmeta["mu_scale"], dtype=np.float32), None)]
        for d in vdirs[1:]:
            h, m = T.load(d)
            if list(m["genome_ids"]) != list(vmeta["genome_ids"]) or not np.allclose(
                m["x_scale"], vmeta["x_scale"]
            ):
                raise ValueError(f"{d} does not match the primary value checkpoint (P13/P14)")
            cal = m.get("value_cal")
            self._ens.append(
                (
                    T._ARCH[m.get("arch", {}).get("arch", "icnn")],
                    h,
                    np.asarray(m["mu_scale"], dtype=np.float32),
                    None if cal is None else np.asarray(cal, dtype=np.float64),
                )
            )
        self.mask = np.array(vmeta["mask"], dtype=bool)
        self.x_scale = np.asarray(vmeta["x_scale"], dtype=np.float32)
        self.mu_scale = np.asarray(vmeta["mu_scale"], dtype=np.float32)
        # Head A over-predicts slow media, and `d(log X)/dt = mu` integrates exactly
        # that relative error. Applied here, beside `mu_scale`, because it is a
        # property of the checkpoint and not of the head (`cfs.surrogate.calibrate`).
        self.value_cal = np.asarray(
            vmeta.get("value_cal") or T._identity_cal(len(self.genome_ids)), dtype=np.float64
        )
        self.z_scale = None if bheads is None else np.asarray(bmeta["z_scale"], dtype=np.float32)
        # Head B emits flux *per unit growth*; the magnitude comes from Head A,
        # which is the accurate half. A pre-2026-08-30 checkpoint has no
        # `mu_floor` and emits flux directly.
        self.mu_floor = (
            np.asarray(bmeta["mu_floor"], dtype=np.float64) if "mu_floor" in bmeta else None
        )
        # §8.6g(1): the reach proxy's reference set, written by `behaviour.save`.
        # Absent from a pre-2026-09-03 checkpoint, in which case `reach` is None.
        ref = None if behaviour_dir is None else Path(behaviour_dir) / "reference_x.npz"
        self.ref_x = (
            np.load(ref)["x"].astype(np.float32) if ref is not None and ref.exists() else None
        )
        self.km = _km_vector(self.exchanges)
        self._E = _element_matrix(self.exchanges)  # (4, M), the §8.6g(2) bound
        self._bears = (self._E > 0).any(0) if self._E is not None else None
        self._jnp = jnp
        self.members = (
            list(range(len(self.genome_ids)))
            if organisms is None
            else [self.genome_ids.index(g) for g in organisms]
        )

    def _x(self, c: np.ndarray):
        """Concentration -> the heads' input, per organism: ``u/(u + x_scale)``."""
        u = c / (self.km + c)  # (M,)
        return (u / (u + self.x_scale))[:, None, :].astype(np.float32)  # (G, 1, M)

    def _mu(self, x) -> np.ndarray:
        """Calibrated ``mu`` per organism at one medium; min over the seed stack."""
        out = None
        for mod, heads, scale, cal in self._ens:
            raw = np.asarray(mod.batched_value(heads, x))[:, 0]
            m = calibrate.apply(raw[:, None], self.value_cal if cal is None else cal)[:, 0] * scale
            out = m if out is None else np.minimum(out, m)
        return out

    def mu_and_z(self, c: np.ndarray, alpha: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(mu, z)`` for every organism in the stack at medium ``c``."""
        if self._bheads is None:
            raise ValueError("this Surrogate was built without a behaviour checkpoint")
        jnp = self._jnp
        x = jnp.asarray(self._x(c))
        a = jnp.asarray(alpha[:, None], dtype=jnp.float32)
        mu = self._mu(x)
        mu = np.maximum(mu, 0.0)  # P2: an infeasible medium has mu_max = 0.
        zmu = (
            None
            if self.mu_floor is None
            else jnp.asarray(np.maximum(mu, self.mu_floor)[:, None], dtype=jnp.float32)
        )
        z = np.asarray(self._B.flux(self._bheads, x, a, jnp.asarray(self.z_scale), zmu))[:, 0]
        # §3.3's uptake bound, which the LP that made the labels could not violate
        # and Head B can: `z_m >= -Vmax_m * u_m`. Vmax is 1000 on every exchange of
        # every roster GEM (`solve.set_medium_bounds`'s fallback is the same 1000),
        # so this is a constant, not a fit. It is a projection onto a convex set the
        # true `z` is already inside, so it cannot increase the error -- and it is
        # exactly where the composition went wrong: at the worst M5 community the
        # head predicted `EX_glyc3p_e` uptake of -329 against a physical floor of
        # -14, on 28 of one member's 213 exchanges at once.
        z = np.maximum(z, -self._B.VMAX * (c / (self.km + c))) * self.mask
        return mu, self._element_balance(z)

    def _raw_z(self, c: np.ndarray) -> np.ndarray:
        """``z`` after §3.3's clamp but *before* the elemental projection.

        Only for measuring how large that projection is (`20hm_bands/proj_size.py`).
        """
        mu, _ = self.mu_and_z(c, np.ones(len(self.genome_ids), dtype=np.float32))
        jnp = self._jnp
        x = jnp.asarray(self._x(c))
        a = jnp.asarray(np.ones((len(self.genome_ids), 1)), dtype=jnp.float32)
        zmu = (
            None
            if self.mu_floor is None
            else jnp.asarray(np.maximum(mu, self.mu_floor)[:, None], dtype=jnp.float32)
        )
        z = np.asarray(self._B.flux(self._bheads, x, a, jnp.asarray(self.z_scale), zmu))[:, 0]
        return np.maximum(z, -self._B.VMAX * (c / (self.km + c))) * self.mask

    def _element_balance(self, z: np.ndarray) -> np.ndarray:
        """§8.6g(2): project onto `E z <= 0` for C, N, P and S.

        You cannot secrete more of an element than you took up. The LP that made
        the labels could not: violation rate **0.0000** on all four elements over
        21k label rows. Head B has no such constraint and breaks it on 34-43% of
        held-out media and on **53% (carbon) of the states in the communities whose
        endpoint fails**, where the median net carbon flux is positive -- more
        carbon out than in -- by 11% of that element's turnover, against the head's
        own 12-26% relative error (`20hm_bands/element_bound.py`).

        It is the **minimum-norm** projection, in the head's own `z_scale` metric,
        onto a convex set the true `z` is already inside -- so like §3.3's uptake
        clamp it cannot increase the distance to the truth. A cheaper uniform
        shrink of the secretions enforces the same four inequalities and is *not* a
        projection: measured, it left the endpoint unchanged (9 cells of 30 better,
        4 worse) while making `dc_rel` worse on 10 of the 11 cells that moved,
        because it also shrinks the fluxes that were not implicated.

        Four constraints, so the dual is a 4-dimensional non-negative least squares
        and its active set is found by enumerating the 15 non-empty subsets --
        exact, and cheaper than an iterative solver at this size.
        """
        if self._E is None:
            return z
        out = z.copy()
        w = self.z_scale.astype(np.float64) ** 2  # (G, M), the metric
        for k in range(z.shape[0]):
            E = self._E * self.mask[k]
            g = E @ z[k]  # net export per element; > 0 is the violation
            if (g <= 1e-12).all():
                continue
            lam = _dual_nnls((E * w[k]) @ E.T, g)
            out[k] = z[k] - w[k] * (E.T @ lam)
        return out


_ELEMENTS = ("C", "N", "P", "S")


def _dual_nnls(Q: np.ndarray, g: np.ndarray) -> np.ndarray:
    """``argmin_{lam >= 0} 0.5 lam' Q lam - lam' g`` by active-set enumeration.

    The dual of the projection above. ``Q`` is 4x4 and PSD (possibly singular when
    an element carries no flux), so ``lstsq`` rather than ``solve``.
    """
    from itertools import combinations

    best, best_val = np.zeros(len(g)), 0.0
    for r in range(1, len(g) + 1):
        for S in combinations(range(len(g)), r):
            idx = list(S)
            lam_s = np.linalg.lstsq(Q[np.ix_(idx, idx)], g[idx], rcond=None)[0]
            if (lam_s < -1e-12).any():
                continue
            lam = np.zeros(len(g))
            lam[idx] = lam_s
            # Feasibility of the primal: `E z' = g - Q lam <= 0`.
            if (Q @ lam - g < -1e-9).any():
                continue
            val = 0.5 * lam @ Q @ lam - lam @ g
            if val < best_val:
                best, best_val = lam, val
    return best


def _element_matrix(exchanges: list[str]) -> np.ndarray | None:
    """(4, M) atoms per mmol from ``config/exchange_elements.csv``. Absent = 0."""
    import csv

    path = Path(__file__).resolve().parents[1] / "config" / "exchange_elements.csv"
    if not path.exists():
        return None
    col = {ex: j for j, ex in enumerate(exchanges)}
    E = np.zeros((len(_ELEMENTS), len(exchanges)))
    seen = 0
    with path.open() as fh:
        for row in csv.DictReader(line for line in fh if not line.startswith("#")):
            j = col.get(row["exchange_id"])
            if j is None:
                continue
            E[:, j] = [float(row[e]) for e in _ELEMENTS]
            seen += 1
    if seen < len(exchanges):
        LOGGER.info(
            "elemental formulas for %d/%d exchanges; the rest are unconstrained (§8.6g)",
            seen,
            len(exchanges),
        )
    return E


def _km_vector(exchanges: list[str]) -> np.ndarray:
    from cfs.groundtruth.solve import km_for_exchange, load_km_defaults

    km_cfg = load_km_defaults()
    return np.array([km_for_exchange(ex, km_cfg) for ex in exchanges], dtype=np.float64)


# --------------------------------------------------------------------------- #
# The two right-hand sides
# --------------------------------------------------------------------------- #


def rhs_surrogate(sur: Surrogate, c: np.ndarray, X: np.ndarray):
    """``(dc/dt, mu)`` from the frozen heads. ``X`` is over ``sur.members``."""
    mu, z = sur.mu_and_z(c, np.ones(len(sur.genome_ids), dtype=np.float32))
    mu, z = mu[sur.members], z[sur.members]
    return (X[:, None] * z).sum(0), mu


def rhs_truth(models: list, exchanges: list[str], c: np.ndarray, X: np.ndarray, eps: float):
    """``(dc/dt, mu)`` from one FBA + elastic-net solve per organism at medium ``c``."""
    from cfs.groundtruth.solve import load_km_defaults, solve

    km_cfg = load_km_defaults()
    conc = dict(zip(exchanges, c.tolist(), strict=True))
    col = {ex: j for j, ex in enumerate(exchanges)}
    mu = np.zeros(len(models))
    dc = np.zeros(len(exchanges))
    for i, model in enumerate(models):
        sol = solve(model, conc, 1.0, eps, km_cfg)
        if sol.status != "optimal":
            LOGGER.debug("organism %d non-optimal (%s) — treated as no growth (P2)", i, sol.status)
            continue
        mu[i] = sol.mu_max
        for ex, v in sol.z.items():
            dc[col[ex]] += X[i] * v
    return dc, mu


def integrate(rhs, c0: np.ndarray, x0: np.ndarray, dt: float, steps: int) -> Trajectory:
    """Explicit Euler with the pool clipped at zero. Same map for both rhs."""
    c, X = c0.astype(np.float64).copy(), x0.astype(np.float64).copy()
    ts, cs, xs, mus, dcs = [0.0], [c.copy()], [X.copy()], [], []
    for k in range(steps):
        dc, mu = rhs(c, X)
        mus.append(mu)
        dcs.append(dc)
        c = np.maximum(c + dt * dc, 0.0)
        X = X * np.exp(dt * mu)  # exact for constant mu over the step; X stays > 0 (P8)
        ts.append((k + 1) * dt)
        cs.append(c.copy())
        xs.append(X.copy())
    return Trajectory(np.array(ts), np.array(cs), np.array(xs), np.array(mus), np.array(dcs))


# --------------------------------------------------------------------------- #
# The medium
# --------------------------------------------------------------------------- #


def community_medium(
    labels_dir: Path, gids: list[str], exchanges: list[str], seed: int, scales_path: Path | None
) -> np.ndarray:
    """One §4.3 draw over the union of the members' active subspaces.

    Concentrations for exchanges no member can take up are irrelevant to every
    head (they are masked) and to every LP (no reaction), so they stay at 0.
    """
    from cfs.groundtruth.solve import load_km_defaults
    from cfs.sampling.active_subspace import ActiveSubspace, load_subspaces
    from cfs.sampling.design import SamplingConfig, sample_media

    subs = {}
    for gid in gids:
        subs |= load_subspaces(Path(labels_dir) / f"{gid}.subspace.json")
    active = sorted({e for gid in gids for e in subs[gid].active})
    background = sorted({e for gid in gids for e in subs[gid].background} - set(active))
    union = ActiveSubspace("+".join(gids), active, background, {}, 0.0)
    scales = json.loads(Path(scales_path).read_text()) if scales_path else None
    # `sample_media` emits one all-but-one-depleted corner per active metabolite
    # before any bulk draw, so ask for a small bulk budget and take a draw from
    # *after* the corners: a corner is a legitimate design point but always the
    # same shape, and every community would get one.
    rng = np.random.default_rng(seed * _MEDIUM_SEED_STRIDE + 1)
    cfg = SamplingConfig(n_media=len(active) + 64, probe=False, seed=int(rng.integers(1 << 31)))
    media = sample_media(union, load_km_defaults(), cfg, scales=scales)
    medium = media[int(rng.integers(len(active), len(media)))]
    col = {ex: j for j, ex in enumerate(exchanges)}
    c = np.zeros(len(exchanges))
    for ex, v in medium.items():
        if ex in col:
            c[col[ex]] = v
    return c


# --------------------------------------------------------------------------- #
# The comparison
# --------------------------------------------------------------------------- #


def _rel(a: np.ndarray, b: np.ndarray) -> float:
    """Relative L2 difference, the units the M5 gate is stated in."""
    d = np.linalg.norm(b)
    return float(np.linalg.norm(a - b) / d) if d > 0 else float(np.linalg.norm(a - b))


def _first_empty(traj: Trajectory, exchanges: list[str]) -> list[str]:
    """Metabolites present at t=0 and gone by the end -- what ends the culture."""
    gone = (traj.c[0] > 0) & (traj.c[-1] <= 0)
    return [exchanges[j] for j in np.flatnonzero(gone)]


def _recall(rows: list[dict], suffix: str) -> float | None:
    """Share of true cross-feeding links the surrogate also produces. None if none."""
    hit = sum(r["cross_feeding"][f"n_links_recovered{suffix}"] for r in rows)
    tot = sum(r["cross_feeding"][f"n_links_true{suffix}"] for r in rows)
    return hit / tot if tot else None


def _exhausted(traj: Trajectory) -> float:
    """Fraction of the horizon at which growth stops, or 1.0 if it never does."""
    dead = np.flatnonzero(traj.mu.max(1) <= 1e-3 * traj.mu[0].max())
    return float(traj.t[dead[0]] / traj.t[-1]) if len(dead) else 1.0


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / n) if n > 0 else float("nan")


def compare(
    sur: Surrogate,
    models: list,
    c0: np.ndarray,
    x0: np.ndarray,
    dt: float,
    steps: int,
    eps: float,
) -> dict:
    """Integrate both, and score the surrogate rhs along the *true* trajectory."""
    true = integrate(lambda c, X: rhs_truth(models, sur.exchanges, c, X, eps), c0, x0, dt, steps)
    surr = integrate(lambda c, X: rhs_surrogate(sur, c, X), c0, x0, dt, steps)

    # Step-matched: the surrogate rhs at the states the real community visited.
    # This separates "the heads are wrong" from "the error compounded".
    #
    # Scored only while the real community is alive. A batch culture ends with the
    # pool empty and every mu at 0, and there a relative error divides by zero and
    # a cosine is 0/0 -- the first version of this reported `nan` and a 237% mu
    # error for a run whose live phase agreed to 2%. Errors are normalised by the
    # community's *initial* scale, which is fixed for the whole trajectory, so two
    # steps are comparable to each other.
    mu0, dc0 = np.linalg.norm(true.mu[0]), np.linalg.norm(true.dc[0])
    alive = [k for k in range(steps) if true.mu[k].max() > 1e-3 * true.mu[0].max()]
    dc_cos, dc_rel, mu_rel = [], [], []
    # Per *member* relative error, not just the pooled norm. The §8.1 failures are
    # a tail -- one member at +147% inside a 21-member pool whose median |rel| is
    # 0.068 -- and reporting medians alone hid that for three label roots (§8.5 A2).
    mu_member = []
    for k in alive:
        dc, mu = rhs_surrogate(sur, true.c[k], true.x[k])
        if np.linalg.norm(dc) > 0 and np.linalg.norm(true.dc[k]) > 0:
            dc_cos.append(_cos(dc, true.dc[k]))
        dc_rel.append(float(np.linalg.norm(dc - true.dc[k]) / dc0) if dc0 > 0 else np.nan)
        mu_rel.append(float(np.linalg.norm(mu - true.mu[k]) / mu0) if mu0 > 0 else np.nan)
        mu_member.append((mu - true.mu[k]) / np.maximum(true.mu[k], 1e-9 + 1e-3 * true.mu[0].max()))
    per_member = (
        np.median(np.array(mu_member), axis=0) if mu_member else np.full(len(sur.members), np.nan)
    )
    worst = int(np.argmax(np.abs(per_member))) if mu_member else 0

    # Trajectory error, at every step and at the end. Biomass is compared in log
    # space: X grows exponentially, so a relative error on X is dominated by the
    # last step, while log X is the error in the *integrated growth rate*, which is
    # the quantity M5's 1% gate is about.
    # V5 / P4: re-solve the true LP at the state the *surrogate* walked itself to.
    # The composition is free to drift somewhere the labels never went, and a
    # community that grows impossibly fast reads as a discovery. This is the check
    # that catches it, and it costs one solve per organism.
    _, mu_lp_end = rhs_truth(models, sur.exchanges, surr.c[-1], surr.x[-1], eps)
    _, mu_s_end = rhs_surrogate(sur, surr.c[-1], surr.x[-1])

    x_log = np.abs(np.log(surr.x) - np.log(true.x))
    # Per metabolite, relative to *its own* initial level. A plain L2 over the pool
    # is dominated by the replete background and cannot see the scarce metabolite
    # that actually ends the culture -- the one whose depletion is the entire
    # dynamics. Measured: an L2 error of 0.7% over a trajectory where the limiting
    # ion was gone in the truth and untouched in the surrogate.
    live = true.c[0] > 0
    c0 = true.c[0][live]
    c_rel = [
        float(np.max(np.abs(surr.c[k][live] - true.c[k][live]) / c0)) for k in range(steps + 1)
    ]
    return (
        {
            "rhs_on_true_path": {
                "n_steps_alive": len(alive),
                "dc_cosine_median": float(np.median(dc_cos)) if dc_cos else float("nan"),
                "dc_cosine_min": float(np.min(dc_cos)) if dc_cos else float("nan"),
                "dc_rel_median": float(np.median(dc_rel)),
                "mu_rel_median": float(np.median(mu_rel)),
                "mu_rel_max": float(np.max(mu_rel)),
                "mu_rel_per_member": per_member.tolist(),
                "mu_rel_worst_member": [
                    sur.genome_ids[sur.members[worst]],
                    float(per_member[worst]),
                ],
            },
            "trajectory": {
                "x_log_err_final": float(x_log[-1].max()),
                "x_log_err_max": float(x_log.max()),
                "c_rel_err_final": c_rel[-1],
                "c_rel_err_max": float(max(c_rel)),
                # When the pool runs out, in units of the horizon. A community that
                # collapses at the right time for the wrong reason still shows here.
                "t_exhausted_true": _exhausted(true),
                "t_exhausted_surrogate": _exhausted(surr),
                "first_empty_true": _first_empty(true, sur.exchanges),
                "first_empty_surrogate": _first_empty(surr, sur.exchanges),
                "mu_true_initial": true.mu[0].tolist(),
                "mu_surrogate_initial": surr.mu[0].tolist(),
                "x_final_true": true.x[-1].tolist(),
                "x_final_surrogate": surr.x[-1].tolist(),
            },
            "round_trip_end": {
                "mu_lp": mu_lp_end.tolist(),
                "mu_surrogate": mu_s_end.tolist(),
                "overgrowth": float((mu_s_end - mu_lp_end).max() / max(true.mu[0].max(), 1e-30)),
            },
            # Cross-feeding is the community behaviour that does not exist for a single
            # organism: a metabolite no member starts with, that one secretes and
            # another consumes. Counted on the true path, then scored on the surrogate.
            "cross_feeding": _cross_feeding(sur, models, true, eps),
        },
        true,
        surr,
    )


def _cross_feeding(sur: Surrogate, models: list, true: Trajectory, eps: float) -> dict:
    """Which metabolites are produced by one member and consumed by another.

    Evaluated at the mid-point of the true trajectory. A community whose members
    merely compete has none of these, and reproducing them is the thing a
    per-organism surrogate has no direct evidence for: each organism's labels were
    generated alone.
    """
    from cfs.groundtruth.solve import load_km_defaults, solve

    k = len(true.mu) // 2
    c = true.c[k]

    km_cfg = load_km_defaults()
    conc = dict(zip(sur.exchanges, c.tolist(), strict=True))
    col = {ex: j for j, ex in enumerate(sur.exchanges)}
    zt = np.zeros((len(models), len(sur.exchanges)))
    for i, model in enumerate(models):
        sol = solve(model, conc, 1.0, eps, km_cfg)
        for ex, v in sol.z.items():
            zt[i, col[ex]] = v
    _, zs = sur.mu_and_z(c, np.ones(len(sur.genome_ids), dtype=np.float32))
    zs = zs[sur.members]

    tol = 1e-6
    produced = (zt > tol).any(0)
    consumed = (zt < -tol).any(0)
    both = produced & consumed
    # Strict: the metabolite is not in the medium at all, so the only source is a
    # community member. Loose: produced and consumed regardless, which includes
    # a member topping up something the medium already supplies.
    links = np.flatnonzero(both & (c <= 0))
    loose = np.flatnonzero(both)

    def recovered(idx):
        return [j for j in idx if (zs[:, j] > tol).any() and (zs[:, j] < -tol).any()]

    hit, hit_loose = recovered(links), recovered(loose)
    return {
        "n_links_true": int(len(links)),
        "n_links_recovered": int(len(hit)),
        "n_links_true_loose": int(len(loose)),
        "n_links_recovered_loose": int(len(hit_loose)),
        "metabolites_true": [sur.exchanges[j] for j in links],
        "metabolites_recovered": [sur.exchanges[j] for j in hit],
        "z_cosine_per_organism": [_cos(zs[i], zt[i]) for i in range(len(models))],
        "unweighted_dc_cosine": _cos(zs.sum(0), zt.sum(0)),
    }


def run(
    roster_path: Path,
    labels_dir: Path,
    value_dir: Path,
    behaviour_dir: Path,
    out: Path,
    *,
    communities: list[list[str]],
    steps: int = 100,
    doublings: float = 4.0,
    biomass: float | None = None,
    eps: float = 1e-3,
    seed: int = 0,
    scales: Path | None = None,
) -> dict:
    """Compose each community, compare against the LP, write ``community.json``."""
    from surrogate_mgem.data import read_roster

    Path(out).mkdir(parents=True, exist_ok=True)
    roster = {gm.genome_id: gm for gm in read_roster(Path(roster_path))}
    report = {"communities": []}
    for n, gids in enumerate(communities):
        sur = Surrogate(value_dir, behaviour_dir, organisms=gids)
        missing = [g for g in gids if g not in roster]
        if missing:
            raise ValueError(f"community members absent from the roster: {missing}")
        import cobra

        models = [cobra.io.read_sbml_model(str(roster[g].model_path)) for g in gids]
        c0 = community_medium(labels_dir, gids, sur.exchanges, seed + n, scales)

        # Two clocks run in a batch culture: the members double, and the pool
        # empties. They are independent, and if they are far apart the run
        # measures nothing -- a horizon of four doublings on a scarce metabolite
        # killed the true community at step 2 of 30, and a horizon set by the pool
        # alone ends before any biomass has moved. So the *inoculum* is solved for
        # instead: total biomass such that the pool runs out after `doublings`
        # doublings of the fastest member. `dc/dt` is linear in X, so one probe
        # solve at unit biomass fixes it.
        dc1, mu0 = rhs_truth(models, sur.exchanges, c0, np.full(len(gids), 1.0), eps)
        if mu0.max() <= 0:
            LOGGER.warning("community %s cannot grow on its medium -- skipped", gids)
            continue
        t_end = doublings * np.log(2.0) / mu0.max()
        drain = (c0 > 0) & (dc1 < 0)
        if biomass is None and drain.any():
            x_total = float((c0[drain] / -dc1[drain]).min()) / t_end
        else:
            x_total = biomass if biomass is not None else 1e-3
        x0 = np.full(len(gids), x_total / len(gids))
        dt = t_end / steps

        LOGGER.info(
            "community %s: mu0 %s, t_end %.4g h, dt %.3g, %d steps",
            gids,
            np.round(mu0, 3).tolist(),
            t_end,
            dt,
            steps,
        )
        LOGGER.info(
            "  inoculum %.3g gDW/L total, %d exchanges present", x_total, int((c0 > 0).sum())
        )
        res, true, surr = compare(sur, models, c0, x0, dt, steps, eps)
        res |= {"genome_ids": gids, "size": len(gids), "t_end": t_end, "steps": steps, "dt": dt}
        report["communities"].append(res)
        LOGGER.info(
            "  rhs dc cosine %.4f | mu rel %.4f | log-X final err %.4f | c rel final %.4f | "
            "cross-feed %d/%d",
            res["rhs_on_true_path"]["dc_cosine_median"],
            res["rhs_on_true_path"]["mu_rel_median"],
            res["trajectory"]["x_log_err_final"],
            res["trajectory"]["c_rel_err_final"],
            res["cross_feeding"]["n_links_recovered"],
            res["cross_feeding"]["n_links_true"],
        )
        np.savez_compressed(
            Path(out) / f"trajectory_{n}_{'_'.join(gids)}.npz",
            t=true.t,
            c_true=true.c,
            x_true=true.x,
            c_surr=surr.c,
            x_surr=surr.x,
            exchanges=np.array(sur.exchanges),
        )

    by_size: dict[int, list] = {}
    for r in report["communities"]:
        by_size.setdefault(r["size"], []).append(r)
    report["summary"] = {
        str(k): {
            "n": len(v),
            "median_dc_cosine": float(
                np.median([r["rhs_on_true_path"]["dc_cosine_median"] for r in v])
            ),
            "median_mu_rel": float(np.median([r["rhs_on_true_path"]["mu_rel_median"] for r in v])),
            "median_x_log_err_final": float(
                np.median([r["trajectory"]["x_log_err_final"] for r in v])
            ),
            "worst_x_log_err_final": float(np.max([r["trajectory"]["x_log_err_final"] for r in v])),
            "worst_overgrowth": float(np.max([r["round_trip_end"]["overgrowth"] for r in v])),
            # Loose is the headline: the strict "absent from the medium" definition
            # is 0/0 on a §4.3 medium, which holds every uptake exchange at some
            # positive level, and a 0/0 recall reported as 0.0 reads as a failure.
            "cross_feeding_recall": _recall(v, "_loose"),
            "cross_feeding_recall_strict": _recall(v, ""),
            "cross_feeding_links": sum(r["cross_feeding"]["n_links_true_loose"] for r in v),
        }
        for k, v in sorted(by_size.items())
    }
    (Path(out) / "community.json").write_text(json.dumps(report, indent=2))
    return report


# --------------------------------------------------------------------------- #
# Phase 7 §13.1 — forward simulation, batch or continuous, with no LP in it
# --------------------------------------------------------------------------- #


def with_chemostat(rhs, dilution: float, feed: np.ndarray):
    """Wrap a batch right-hand side into a continuous culture at rate ``D``.

        dc/dt = ... + D (c_feed - c)      dX/dt = X (mu - D)

    This is §8.1's ``inflow(c)``, plus washout. It needs no change to
    :func:`integrate`: the biomass update is already ``X exp(dt mu)``, so a net
    growth rate of ``mu - D`` is the whole of washout, and ``D = 0`` is exactly
    the batch culture.
    """
    feed = np.asarray(feed, dtype=np.float64)

    def f(c, X):
        dc, mu = rhs(c, X)
        return dc + dilution * (feed - c), mu - dilution

    return f


def simulate(
    value_dir: Path,
    behaviour_dir: Path,
    out: Path,
    *,
    organisms: list[str],
    labels_dir: Path | None = None,
    medium: Path | None = None,
    abundances: np.ndarray | None = None,
    biomass: float | None = None,
    steps: int = 200,
    hours: float | None = None,
    doublings: float = 4.0,
    dilution: float = 0.0,
    feed: Path | None = None,
    seed: int = 0,
    scales: Path | None = None,
) -> dict:
    """Integrate one community forward from a medium. Surrogate only — no LP.

    ``dilution > 0`` makes it a chemostat fed with ``feed`` (default: the initial
    medium). The accuracy of what comes out is M5's business (§8.1); this is the
    same map with the ground truth, and therefore the cost, removed.
    """
    sur = Surrogate(value_dir, behaviour_dir, organisms=organisms)
    c0 = _medium_vector(sur, organisms, labels_dir, medium, seed, scales)
    c_feed = c0 if feed is None else _read_medium(feed, sur.exchanges)
    share = np.full(len(organisms), 1.0 / len(organisms))
    if abundances is not None:
        share = np.asarray(abundances, float) / np.sum(abundances)

    rhs = lambda c, X: rhs_surrogate(sur, c, X)  # noqa: E731
    if dilution > 0:
        rhs = with_chemostat(rhs, dilution, c_feed)

    # §8.1's two clocks: the members double, and the pool empties (or, in a
    # chemostat, the vessel turns over). A horizon set by one alone measures
    # nothing -- an arbitrary inoculum kills a batch culture at 3% of its horizon.
    # `dc/dt` is linear in X, so one probe at unit biomass fixes the inoculum that
    # empties the pool at the end. Same solve as `run`, surrogate instead of LP.
    dc1, mu0 = rhs(c0, share)
    if mu0.max() <= 0:
        raise ValueError("nobody grows on this medium at t=0")
    if hours is None:
        hours = doublings * np.log(2.0) / mu0.max()
        if dilution > 0:
            hours = max(hours, 5.0 / dilution)
    drain = (c0 > 0) & (dc1 < 0)
    if biomass is None:
        biomass = float((c0[drain] / -dc1[drain]).min()) / hours if drain.any() else 1e-3
    x0 = biomass * share

    traj = integrate(rhs, c0, x0, hours / steps, steps)
    reach = sur.reach(c0)
    Path(out).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        Path(out) / "trajectory.npz",
        t=traj.t,
        c=traj.c,
        x=traj.x,
        mu=traj.mu,
        dc=traj.dc,
        exchanges=np.array(sur.exchanges),
        genome_ids=np.array(organisms),
    )
    report = {
        "genome_ids": organisms,
        "mode": "chemostat" if dilution > 0 else "batch",
        "dilution": dilution,
        "t_end": hours,
        "steps": steps,
        "mu_initial": traj.mu[0].tolist(),
        "mu_final": traj.mu[-1].tolist(),
        "x_initial": traj.x[0].tolist(),
        "x_final": traj.x[-1].tolist(),
        "biomass": biomass,
        # Coexistence, and only meaningful in a chemostat: in a batch culture every
        # mu is 0 at the end because the pool is empty, not because anyone lost.
        "washed_out": (
            [g for g, m in zip(organisms, traj.mu[-1], strict=True) if m <= 0]
            if dilution > 0
            else None
        ),
        # §8.6g(1)'s two runtime predictors, both free and both surrogate-only.
        # Depth is `mu_hat(t)/mu_hat(0)` per member: B4 measured the error rising
        # monotonically as it falls, and a fallback at depth < 0.9 captures 72% of
        # the trajectory error on 24% of the member-steps.
        "reach": None if reach is None else reach.tolist(),
        "depth_final": (traj.mu[-1] / np.where(mu0 > 0, mu0, np.nan)).tolist(),
        "frac_steps_below_depth_0.9": float(
            np.mean(traj.mu / np.where(mu0 > 0, mu0, np.nan) < 0.9)
        ),
        "t_exhausted": _exhausted(traj),
        "first_empty": _first_empty(traj, sur.exchanges),
        "cross_feeding": _cross_feeding_surrogate(sur, traj),
    }
    (Path(out) / "simulation.json").write_text(json.dumps(report, indent=2))
    return report


def _cross_feeding_surrogate(sur: Surrogate, traj: Trajectory) -> list[str]:
    """Metabolites one member secretes and another consumes, mid-trajectory.

    The surrogate's own view of :func:`_cross_feeding` — no LP, so no recall, just
    the links. This is the community structure the run is usually for.
    """
    c = traj.c[len(traj.mu) // 2]
    _, z = sur.mu_and_z(c, np.ones(len(sur.genome_ids), dtype=np.float32))
    z = z[sur.members]
    tol = 1e-6
    both = (z > tol).any(0) & (z < -tol).any(0)
    return [sur.exchanges[j] for j in np.flatnonzero(both)]


def _read_medium(path: Path, exchanges: list[str]) -> np.ndarray:
    """``{exchange_id: mM}`` JSON onto the frozen index. Absent = 0."""
    spec = json.loads(Path(path).read_text())
    col = {ex: j for j, ex in enumerate(exchanges)}
    unknown = sorted(set(spec) - set(col))
    if unknown:
        LOGGER.warning(
            "%d medium entries are not in the index, ignored: %s", len(unknown), unknown[:5]
        )
    c = np.zeros(len(exchanges))
    for ex, v in spec.items():
        if ex in col:
            c[col[ex]] = float(v)
    return c


def _medium_vector(sur, gids, labels_dir, medium, seed, scales) -> np.ndarray:
    if medium is not None:
        return _read_medium(medium, sur.exchanges)
    if labels_dir is None:
        raise ValueError("give either --medium or --labels (to draw a §4.3 medium)")
    return community_medium(labels_dir, gids, sur.exchanges, seed, scales)
