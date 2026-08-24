"""M2 ground-truth solve interface (plan §3.2, §3.3, §5.4).

Per-organism CarveMe FBA is the label source. For a medium of metabolite
concentrations ``c`` and a normalised growth rate ``alpha``:

1. Concentrations become uptake bounds by Michaelis-Menten (§3.3, D5):
   ``lb_m = -Vmax_m * c_m / (Km_m + c_m)``.
2. **Stage 1** — plain FBA gives ``mu_max`` (always unique) and the exchange
   shadow prices (reduced costs, ``dmu_max/d(bound)``).
3. **Stage 2** — growth is fixed to ``alpha * mu_max`` and the exchange fluxes
   ``z`` are pinned by the **elastic-net** objective (D4 decision, §5.4):
       min ||v||_1 + (eps/2)||v||^2   s.t.  S v = 0, lb <= v <= ub, v_bio = alpha*mu_max
   The L2 term makes the primal (hence ``z``) unique and continuous in ``c``;
   ``eps`` is the smoothing scale ``tau`` (D7/§5.5). Solved as a convex QP with
   **Clarabel** (sparse interior point) — HiGHS's QP fails on ~30% of real
   CarveMe solves (see :func:`elastic_net_fluxes`), and a first-order method
   (OSQP) does not converge tightly enough for repeatable labels (§3.4).

COBRApy (LP stage) + Clarabel (QP stage), both in the ``data`` extra. No JAX.
Imports are function-local so the module loads without the solver stack.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

LOGGER = logging.getLogger("cfs.solve")

_FLUX_EPS = 1e-6  # below this a flux is treated as zero (sparsity / z reporting)
_QP_TIME_LIMIT = 60.0  # seconds; a solve this slow is a degenerate outlier, not a label
# Package data, not a repo-root path: the container installs cfs into
# site-packages, where a "walk up to the repo root" lookup resolves to
# /usr/local/lib/python3.11/config/ and the labels stage dies on the first solve.
_DEFAULT_KM = Path(__file__).resolve().parents[1] / "config" / "km_defaults.yaml"


@dataclass
class Solution:
    """Ground-truth labels for one (organism, medium, alpha, eps) solve."""

    mu_max: float
    alpha: float
    eps: float
    z: dict[str, float]  # exchange_id -> net flux per unit biomass (+secretion/-uptake)
    shadow_prices: dict[str, float]  # exchange_id -> dmu_max/d(lower bound)
    status: str  # "optimal" or a solver status string
    fluxes: np.ndarray = field(repr=False, default=None)  # full flux vector (model.reactions order)


# --------------------------------------------------------------------------- #
# Michaelis-Menten uptake bounds (§3.3)
# --------------------------------------------------------------------------- #


def mm_lower_bound(vmax: float, c: float, km: float) -> float:
    """Uptake lower bound from concentration (negative = uptake). ``c=0 -> 0``."""
    if c <= 0.0:
        return 0.0
    return -abs(vmax) * c / (km + c)


def _base_id(exchange_id: str) -> str:
    """BiGG base metabolite token: EX_glc__D_e -> glc, EX_na1_e -> na1."""
    core = exchange_id
    if core.startswith("EX_"):
        core = core[3:]
    return core.split("_", 1)[0]


def load_km_defaults(path: Path | None = None) -> dict:
    """Load ``km_defaults.yaml`` (classes, default, keyword->class map)."""
    import yaml

    path = Path(path) if path is not None else _DEFAULT_KM
    return yaml.safe_load(path.read_text())


def km_for_exchange(exchange_id: str, km_cfg: dict) -> float:
    """Km for one exchange via its transporter class (plan §3.3). Rough by design."""
    base = _base_id(exchange_id)
    for cls, bases in km_cfg.get("keywords", {}).items():
        if base in bases:
            return float(km_cfg["classes"][cls])
    return float(km_cfg["default"])


def apply_mm_bounds(model, concentrations: dict[str, float], km_cfg: dict) -> None:
    """Set exchange uptake lower bounds from concentrations, in place (§3.3).

    ``concentrations`` maps exchange id -> concentration. Vmax is the exchange's
    existing default uptake magnitude (``abs(lower_bound)``). Exchanges absent from
    ``concentrations`` are left at their current bound (the caller sets background
    levels). Call inside a ``with model:`` block to keep the change scoped.
    """
    for ex_id, c in concentrations.items():
        if ex_id not in model.reactions:
            continue
        rxn = model.reactions.get_by_id(ex_id)
        vmax = abs(rxn.lower_bound) or 1000.0
        rxn.lower_bound = mm_lower_bound(vmax, c, km_for_exchange(ex_id, km_cfg))


# --------------------------------------------------------------------------- #
# Elastic-net QP for the exchange fluxes (§5.4)
# --------------------------------------------------------------------------- #


def elastic_net_fluxes(model, biomass_flux: float, eps: float) -> tuple[np.ndarray, str]:
    """Min ``||v||_1 + (eps/2)||v||^2`` s.t. ``Sv=0``, bounds, biomass fixed (§5.4).

    Returns the full flux vector (aligned to ``model.reactions``) and a status
    string. ``biomass_flux`` is ``alpha*mu_max``.

    Solved with **Clarabel** (sparse interior point). HiGHS runs the LP stage but
    not this: its QP active-set method returned "solve error"/"not set" on ~30% of
    real CarveMe solves at the primary ``eps=1e-3``, and stalled for minutes as
    ``eps`` shrank. Clarabel solved the same batch 48/48 at every ``eps`` level and
    ~6x faster (measured on CP009913.1).
    """
    import clarabel
    import scipy.sparse as sp
    from cobra.util import create_stoichiometric_matrix

    rxns = model.reactions
    n = len(rxns)
    biomass = _biomass_reaction(model)
    bi = rxns.index(biomass)

    S = sp.csc_matrix(create_stoichiometric_matrix(model))
    nm = S.shape[0]
    lb = np.array([r.lower_bound for r in rxns], dtype=float)
    ub = np.array([r.upper_bound for r in rxns], dtype=float)
    lb[bi] = ub[bi] = biomass_flux  # fix growth

    # No growth (a depleted medium, or alpha=0) and the origin is feasible: the
    # objective is >= 0 and vanishes at v=0, so that IS the optimum -- no solve
    # needed. Corner-depleted media (§4.3) hit this on every alpha.
    if biomass_flux == 0.0 and (lb <= 0.0).all() and (ub >= 0.0).all():
        return np.zeros(n), "Optimal"

    # Variables x = [v (n); t (n)] with t_i >= |v_i| (L1 epigraph), so the
    # objective is (eps/2)||v||^2 + sum(t) -- linear in t, hence P is PSD, not PD.
    ident = sp.eye(n, format="csc")
    zero_n = sp.csc_matrix((n, n))
    a = sp.vstack(
        [
            sp.hstack([S, sp.csc_matrix((nm, n))]),  # S v = 0        (zero cone)
            sp.hstack([ident, -ident]),  # v - t <= 0     |
            sp.hstack([-ident, -ident]),  # -v - t <= 0    | t >= |v|
            sp.hstack([ident, zero_n]),  # v <= ub
            sp.hstack([-ident, zero_n]),  # -v <= -lb
        ],
        format="csc",
    )
    b = np.concatenate([np.zeros(nm + 2 * n), ub, -lb])
    p = sp.diags(np.concatenate([np.full(n, float(eps)), np.zeros(n)])).tocsc()
    q = np.concatenate([np.zeros(n), np.ones(n)])
    cones = [clarabel.ZeroConeT(nm), clarabel.NonnegativeConeT(4 * n)]

    settings = clarabel.DefaultSettings()
    settings.verbose = False
    # The elastic-net optimum is unique (strictly convex in v), so repeat solves
    # agree to solver tolerance (~1e-11 measured); not literally bitwise, which is
    # fine for labels (§3.4). The time limit is a backstop so one pathological
    # medium cannot hold up a whole shard -- a hit lands as a non-"solved" status
    # on the row.
    settings.time_limit = _QP_TIME_LIMIT
    solution = clarabel.DefaultSolver(p, q, a, b, cones, settings).solve()
    # Clarabel says "Solved"; the label schema (and every consumer) says "optimal".
    status = "Optimal" if str(solution.status) == "Solved" else str(solution.status)
    return np.array(solution.x[:n], dtype=float), status


# --------------------------------------------------------------------------- #
# The solve interface (§3.2)
# --------------------------------------------------------------------------- #


def _biomass_reaction(model):
    """The biomass/objective reaction (CarveMe id starts with Growth/BIOMASS)."""
    obj = [r for r in model.reactions if r.objective_coefficient != 0]
    if obj:
        return obj[0]
    for prefix in ("Growth", "BIOMASS"):
        cand = [r for r in model.reactions if r.id.upper().startswith(prefix.upper())]
        if cand:
            return cand[0]
    raise ValueError(f"{model.id}: cannot identify biomass reaction")


def mu_optimize(model, what: str = "") -> float:
    """``model.optimize()`` objective under the same wall-clock guard as :func:`solve`.

    Every LP in the label pipeline must go through here. GLPK's simplex (cobra's
    default solver) can cycle indefinitely on a near-degenerate medium -- one
    organism sat at 100% CPU for 4.5 h inside `glp_simplex` on a single solve. The
    §4.2 subspace sweep and the §4.7 demand probe run ~800 such LPs per organism
    *before* the first labelled solve, on deliberately depleted media, so an
    unguarded call there hangs the whole task with no output to show for it.

    A timeout returns 0.0 -- read as "no growth", which biases the caller toward
    calling the metabolite limiting. Wrong-but-loud beats hanging; it is logged.
    """
    from cobra.exceptions import OptimizationError

    model.solver.configuration.timeout = int(_QP_TIME_LIMIT)
    try:
        return model.optimize().objective_value or 0.0
    except OptimizationError as err:
        LOGGER.warning("LP gave up after %gs (%s): %s -> mu=0", _QP_TIME_LIMIT, what, err)
        return 0.0


def solve(
    model, concentrations: dict[str, float], alpha: float, eps: float, km_cfg: dict | None = None
) -> Solution:
    """Ground-truth solve for one (medium, alpha, eps) (plan §3.2).

    Returns :class:`Solution` with ``mu_max``, exchange fluxes ``z`` (this
    organism's own exchanges only), and exchange shadow prices. The model is not
    mutated (bounds are applied inside a context).
    """
    from cobra.exceptions import OptimizationError

    km_cfg = km_cfg if km_cfg is not None else load_km_defaults()
    # The LP gets the same wall-clock guard as the QP. GLPK's simplex (cobra's
    # default solver here) can cycle indefinitely on a near-degenerate medium:
    # one organism in the banded 4000-media run sat at 100% CPU for 4.5 h inside
    # `glp_simplex` on a single solve while its 20 siblings finished in ~1 h each.
    # A typical solve is ~1 s, so anything past the limit is that pathology, and a
    # non-optimal row is dropped downstream (P2) instead of stalling the shard.
    # int(): optlang's GLPK interface multiplies this into glpk's int tm_lim.
    model.solver.configuration.timeout = int(_QP_TIME_LIMIT)
    with model:
        apply_mm_bounds(model, concentrations, km_cfg)

        try:
            fba = model.optimize()
        except OptimizationError as err:  # timeout leaves no primal to read
            return Solution(0.0, alpha, eps, {}, {}, str(err).lower())
        mu_max = fba.objective_value or 0.0
        if fba.status != "optimal":
            return Solution(0.0, alpha, eps, {}, {}, fba.status)
        # Shadow price = the dual of the exchanged metabolite's mass balance
        # (dmu_max/d supply). This is the clean quantity: cobra's exchange
        # `reduced_costs` carry a convention scaling and do not match finite
        # differences. Each exchange has exactly one metabolite.
        shadow = {
            ex.id: float(fba.shadow_prices[next(iter(ex.metabolites)).id]) for ex in model.exchanges
        }

        v, status = elastic_net_fluxes(model, alpha * mu_max, eps)

    ex_ids = [ex.id for ex in model.exchanges]
    idx = {r.id: i for i, r in enumerate(model.reactions)}
    z = {ex: float(v[idx[ex]]) for ex in ex_ids if abs(v[idx[ex]]) > _FLUX_EPS}
    return Solution(
        mu_max=float(mu_max),
        alpha=alpha,
        eps=eps,
        z=z,
        shadow_prices=shadow,
        status=status.lower(),
        fluxes=v,
    )
