"""V-for-§8: the spectrum of the master Jacobian, at real media.

§8.4 Newton-solves the price form with ``tags=lx.positive_semidefinite_tag`` on a
Jacobian that is "a sum of PSD Hessians",

    J = sum_i X_i * (-d2 mu_i / du2)  +  supply'(u)

over the |shared| exchanges of §2.1. :func:`cfs.surrogate.train._hessian_cond`
reports the conditioning of one organism's factor on its own dims; that is a
*training* diagnostic and it does not predict what Newton meets here.

Measured 2026-08-26 (21 seeded ``groupmax-u`` heads, K=250, uniform abundances,
6 held-out `20hm_bands` media). Eigenvalues of ``sum_i X_i H_i`` above a fraction
of the top, median over media, out of 365:

| T | 0.01 | 0.03 | 0.1 | 0.3 | 1.0 | `icnn` (trained) |
| --- | --- | --- | --- | --- | --- | --- |
| raw, > 1e-12 | 100 | 147 | 191 | 197 | 217 | 319 |
| **Jacobi, > 1e-12** | **15** | **19** | **12** | **12** | **15** | **365** |

Three conclusions, and the CLI exists so they can be re-checked rather than
believed:

1. **The Hessian sum is singular** -- ~10-20 of 365 directions carry curvature.
   The ``inflow(c)`` supply term is not a modelling detail, it is what makes the
   solve well-posed, and once ``lam I`` is present ``cond(J) = 1 + top_ev/lam``
   exactly. The supply model sets the conditioning; the head does not.
2. **Temperature buys no conditioning.** 0.01 -> 0.3 is 30x blunter and costs
   held-out cosine 0.951 -> 0.833, and moves the curvature rank from 15 to 12.
   So ``--gm-temp`` is an accuracy knob only, and a per-organism
   ``hessian_cond_median`` of 1.9e24 is not a Phase-5 predictor.
3. **The smooth ``icnn``'s ill-conditioning was per-metabolite scaling, not
   curvature**: Jacobi-preconditioned it is full rank at cond ~2e4, where the raw
   matrix is not. ``s`` spans 5107x across the index, so **precondition J
   diagonally in §8.4 regardless of the head** -- which also absorbs the (still
   unwritten) chain rule from ``u`` to the price coordinate.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import jax.numpy as jnp
import numpy as np

LOGGER = logging.getLogger(__name__)

# Fractions of the top eigenvalue at which the spectrum is counted. The lower two
# are only meaningful in float64, which `run` turns on.
CUTS = (1e-3, 1e-6, 1e-9, 1e-12)


def _spectrum(a: np.ndarray) -> np.ndarray:
    """Eigenvalues of the symmetric part, clipped at 0 and scaled by the top."""
    a = (a + a.T) / 2
    ev = np.clip(np.linalg.eigvalsh(a), 0.0, None)
    return ev / max(ev.max(), 1e-300)


def _in_u(mod, head, s):
    """The head as a function of ``u`` directly, so autodiff owns the chain rule.

    A head is concave either in ``w = u/s`` (``picnn_u`` and its descendants,
    which expose ``head_in_diag``) or in ``x = u/(u+s)``. Composing the input map
    here rather than differentiating in the head's coordinate and rescaling by
    hand keeps the second-order term of the ``x`` map -- which is not zero, unlike
    the affine ``w`` one -- and never evaluates ``1 - x``, whose cancellation in
    the replete far field is what :func:`cfs.surrogate.picnn_u.head_in_diag`
    exists to avoid.
    """
    import jax.numpy as jnp

    from cfs.surrogate.picnn_u import W_CAP

    if hasattr(mod, "head_in_diag"):
        f = mod.head_in_diag(head)
        return lambda u: f(jnp.minimum(u / s, W_CAP))
    return lambda u: head(u / (u + s))


def hessians(heads, mod, ds, shared: np.ndarray, u: np.ndarray) -> np.ndarray:
    """``(G, B, S, S)`` curvature ``-d2 mu/du2`` on the shared block, PSD."""
    import jax

    s = jnp.asarray(ds.x_scale)
    out = []
    for i in range(len(ds.genome_ids)):
        fn = _in_u(mod, mod.organism(heads, i), s[i])
        h = np.asarray(jax.vmap(jax.hessian(fn))(u[i]), dtype=np.float64)
        out.append(-h[:, shared[:, None], shared])
    return np.stack(out)


def spectrum_report(h: np.ndarray, abundance: np.ndarray) -> dict:
    """Per-medium spectrum of ``sum_i X_i H_i``, raw and Jacobi-preconditioned."""
    bad = ~np.isfinite(h)
    if bad.any():
        LOGGER.warning(
            "non-finite curvature: %d entries in organisms %s -- zeroed",
            int(bad.sum()),
            sorted(set(np.nonzero(bad)[0].tolist())),
        )
        h = np.where(bad, 0.0, h)
    s = np.einsum("g,gbmn->bmn", abundance, h)
    n = s.shape[-1]
    raw, prec, tops = [], [], []
    for b in range(s.shape[0]):
        a = (s[b] + s[b].T) / 2
        ev = _spectrum(a)
        # Jacobi: is the decay curvature, or just the per-metabolite 1/s^2 spread?
        d = np.sqrt(np.maximum(np.diag(a), 1e-300))
        evp = _spectrum(a / np.outer(d, d))
        raw.append([int((ev > c).sum()) for c in CUTS])
        prec.append([int((evp > c).sum()) for c in CUTS])
        tops.append(float(np.max(np.linalg.eigvalsh(a))))
    return {
        "n_shared": n,
        "cuts": list(CUTS),
        "top_eigenvalue": tops,
        "dims_above_cut": {
            "raw": np.median(raw, 0).tolist(),
            "jacobi": np.median(prec, 0).tolist(),
        },
        # `sum_i X_i H_i` is singular, so this is exact rather than empirical.
        "cond_note": "cond(S + lam I) = 1 + max(top_eigenvalue)/lam",
    }


def run(
    labels: Path,
    index: Path,
    out: Path,
    *,
    checkpoint: Path | None = None,
    eps: float = 1e-3,
    gm_temp: list[float] | None = None,
    gm_group: int = 250,
    n_media: int = 6,
    organisms: list[str] | None = None,
    seed: int = 0,
) -> dict:
    """Report the §8.4 Jacobian spectrum for a checkpoint, or for seeded heads.

    Without ``--checkpoint`` this seeds the reference configuration -- width 1 /
    depth 1 ``groupmax-u``, where the head is exactly ``min_k(a_k.w + c_k)`` and
    :func:`cfs.surrogate.groupmax.init_from_tangents` reproduces the pruned tangent
    model -- at each ``gm_temp``. That needs no training, which is what makes the
    temperature axis measurable at all: ``temp`` is a *static* field, so it is a
    post-hoc knob on a fixed set of planes, not something the fit commits to.
    """
    import jax

    # The 1e-9/1e-12 cuts are below float32's resolution on a Hessian entry.
    jax.config.update("jax_enable_x64", True)

    from cfs.groundtruth.index import load_index
    from cfs.surrogate import groupmax, train
    from cfs.surrogate.data import load_value_dataset

    frozen = load_index(index)
    pos = {e: i for i, e in enumerate(frozen.index)}
    shared = np.asarray(sorted(pos[e] for e in frozen.shared))
    ds = load_value_dataset(labels, index, eps=eps, seed=seed, organisms=organisms)
    g, m = ds.mask.shape
    # Uniform: §8's abundance vector is an unknown of the master problem, and a
    # positive diagonal reweighting cannot change which directions carry curvature.
    abundance = np.full(g, 1.0 / g)
    LOGGER.info("%d organisms, %d exchanges, %d shared", g, m, len(shared))

    cases: list[tuple[str, object, object]] = []
    if checkpoint is not None:
        import equinox as eqx

        # Checkpoints are float32 and `tree_deserialise_leaves` refuses a dtype
        # change, so build the template with x64 off and widen afterwards.
        jax.config.update("jax_enable_x64", False)
        heads, meta = train.load(checkpoint)
        jax.config.update("jax_enable_x64", True)
        heads = jax.tree.map(
            lambda a: a.astype(jnp.float64) if eqx.is_inexact_array(a) else a, heads
        )
        arch = meta.get("arch", {}).get("arch", "icnn")
        cases.append((f"{Path(checkpoint).name} ({arch})", heads, train._ARCH[arch]))
    else:
        for t in gm_temp or [0.03]:
            heads = train._build(
                "groupmax-u",
                jax.random.PRNGKey(seed),
                g,
                m,
                ds.mask,
                1,
                1,
                8,
                gm_group=gm_group,
                gm_temp=t,
            )
            cases.append(
                (
                    f"groupmax-u seeded K={gm_group} T={t}",
                    groupmax.init_from_tangents(heads, ds, seed=seed),
                    groupmax,
                )
            )

    # u = s * w, clipped at s * W_CAP: the head's own domain, and the same point
    # for every arch so two rows of the report are comparable.
    from cfs.surrogate.picnn_u import to_diag

    x = jnp.asarray(ds.x_val[:, :n_media], dtype=jnp.float64)
    u = jnp.asarray(ds.x_scale, dtype=jnp.float64)[:, None, :] * to_diag(x)

    report = {}
    for name, heads, mod in cases:
        r = spectrum_report(hessians(heads, mod, ds, shared, u), abundance)
        report[name] = r
        LOGGER.info(
            "%s: dims above %s -- raw %s, Jacobi %s (of %d)",
            name,
            [f"{c:g}" for c in CUTS],
            r["dims_above_cut"]["raw"],
            r["dims_above_cut"]["jacobi"],
            r["n_shared"],
        )

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "master_jacobian.json").write_text(json.dumps(report, indent=2))
    return report
