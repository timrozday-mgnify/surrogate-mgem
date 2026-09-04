"""A 1-D output calibration for Head A -- the low-``mu`` bias, removed for free.

Head A over-predicts every medium below the plateau: median relative error +0.45
below 5% of max ``mu`` at ``T`` = 0.01 and +0.94 at 0.03, on 21 organisms. That is
not the softmin offset the earlier note blamed -- re-evaluating the *same trained
head* at ``T -> 0`` makes it **worse** (+0.86 against +0.45), because the smoothing
gap is a downward offset that partially cancels it. It is plane placement, and it
is what the class predicts: a min of tangents to a concave function is an upper
bound everywhere, so positive bias is the only bias a max-affine head can have
unless a plane sits tangent at that row.

**The bias is a function of the predicted value alone.** An isotonic map fit on the
training rows and applied to held-out media drives every band's median bias to
<= 0.005 and *raises* R2 (0.9898 -> 0.9901), which also re-confirms that more
low-``mu`` labels cannot help: nothing is missing from ``mu_hat``.

So calibrate instead of reweighting. ``g(m) = a*m - d0 * exp(-m / beta)`` is

* **increasing** -- ``g' = a + (d0/beta) e^{-m/beta} > 0`` for ``a, d0 >= 0``, and
* **concave** -- ``g'' = -(d0/beta^2) e^{-m/beta} < 0``,

so ``g(head(u))`` is still exactly concave and non-decreasing in ``u`` (§8.4's PSD
Hessian tag and ``concavity_violation_rate`` both survive), and the gradient is
scaled by a positive per-row scalar, so **``grad_cosine`` is bit-identical**.
Unlike ``--w-rel``, which buys the bottom by selling the plateau, this trades
nothing. Measured on the frozen checkpoints, held-out media, fit on train rows:

**But removing the bias outright is not what §8 wants, measured 2026-08-30.** A fit
on *relative* residuals takes the bias below 5% of max ``mu`` from +0.446 to -0.033
and every other band inside 0.011 -- and makes the §8.1 composition **worse**:
median log-X error at community size 21 goes 0.047 -> 0.098, and 0.048 -> 0.098 at
size 10, while ``median_mu_rel`` on the small slow communities improves (0.050 ->
0.027 at size 3). The map is downward-only (``d0 >= 0``) and the plateau was already
under-predicting (-0.005), so buying the bottom pushes the top from -0.005 to
-0.009 -- and ``d(log X)/dt = mu`` integrates the plateau, not the bottom. It is the
same trade ``--w-rel`` makes, moved after training.

So the fit weight is the knob, and ``_W_FLOOR`` is where it is set: residuals are
divided by ``max(|mu|, _W_FLOOR * max|mu|)``, which stops the bottom of the range
dominating. At 0.3 the plateau bias comes out *better* than the uncalibrated head
(-0.005 -> -0.002) and the low band still improves (+0.446 -> **-0.250**) -- and
that version is a clear win on the §8.1 composition, the same 10 communities and
media as ``community_T01``, per-organism FBA truth:

| median log-X error | n=2 | n=3 | n=5 | n=10 | n=21 |
| --- | --- | --- | --- | --- | --- |
| ``value_T01`` uncalibrated | 0.034 | **0.050** | 0.051 | 0.048 | 0.047 |
| ``--w-rel 0.3`` | 0.058 | 0.082 | 0.060 | 0.065 | 0.064 |
| ``_W_FLOOR`` = 0.3 calibration | **0.024** | 0.072 | **0.044** | **0.014** | **0.016** |
| ``_W_FLOOR`` = 0 (pure relative) | 0.046 | 0.074 | 0.061 | 0.098 | 0.098 |

``median_mu_rel`` at size 21 goes 0.009 -> 0.005, ``grad_cosine`` and R2 are
unchanged (0.9283 / 0.9899), and sizes 10 and 21 are 1.4% / 1.6% against M5's 1%
gate. Size 3 is the one regression. **So do not tune this fit by the low-``mu``
bias**; that is the diagnostic that motivated it, not the objective.

It lives in the checkpoint metadata next to ``mu_scale`` and is applied where
``mu_scale`` is -- ``train.evaluate`` and ``compose.dfba.Surrogate.mu_and_z`` --
not inside the head, so every existing checkpoint deserialises unchanged and an
uncalibrated one reads ``d0 = 0``, the exact identity.
"""

from __future__ import annotations

import numpy as np

# The fit is on binned medians, not raw rows: 74% of rows sit on the plateau and a
# plain least squares over them reproduces the very imbalance the bias comes from.
_N_BINS = 40
# Residual weighting. Pure relative residuals (a floor of 0) fit the bottom of the
# range perfectly and cost the plateau, which is the half §8.1 integrates -- see the
# module docstring. 0.3 trades the other way.
_W_FLOOR = 0.3
# `beta` must not collapse: a few organisms predict slightly negative `mu`, where a
# short decay length turns `exp(-m/beta)` into a huge correction (worst R2 -7).
_BETA_FLOOR = 0.05


def fit(mu_hat: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """``(G, N)`` training predictions and targets -> ``(G, 3)`` of ``(d0, beta, a)``.

    Both in ``mu_scale``d units, which is what the head emits.
    """
    from scipy.optimize import least_squares

    out = np.zeros((mu_hat.shape[0], 3))
    out[:, 2] = 1.0
    for i in range(mu_hat.shape[0]):
        p, y = np.asarray(mu_hat[i]), np.asarray(mu[i])
        edges = np.quantile(p, np.linspace(0.002, 0.998, _N_BINS))
        sel = [(p >= a) & (p < b) for a, b in zip(edges[:-1], edges[1:], strict=True)]
        med = np.array([np.median(y[s]) if s.any() else np.nan for s in sel])
        m, ok = 0.5 * (edges[:-1] + edges[1:]), ~np.isnan(med)
        span = float(p.max() - p.min())
        if ok.sum() < 4 or span <= 0:
            continue
        r = least_squares(
            # Relative residuals: the whole point is the bottom of the range.
            lambda q, m=m[ok], y=med[ok]: (
                (q[2] * m - q[0] * np.exp(-m / q[1]) - y)
                / np.maximum(np.abs(y), _W_FLOOR * np.abs(y).max())
            ),
            [0.1, 0.3 * span, 1.0],
            bounds=([0.0, _BETA_FLOOR * span, 0.5], [np.inf, span, 2.0]),
        )
        out[i] = r.x
    return out


def apply(mu_hat, cal) -> np.ndarray:
    """``g(mu_hat)`` for a ``(G, 3)`` calibration of ``(d0, beta, a)``.

    ``d0 = 0, a = 1`` is the identity, which is what an uncalibrated checkpoint
    reads. A two-column ``cal`` (before ``a`` existed) is read as ``a = 1``.
    """
    cal = np.asarray(cal)
    d0, beta = cal[:, 0], cal[:, 1]
    a = cal[:, 2] if cal.shape[1] > 2 else np.ones_like(d0)
    shape = (-1,) + (1,) * (np.ndim(mu_hat) - 1)
    m = np.asarray(mu_hat)
    # The exponent is clipped, not just the divisor. `beta = 0` is what an
    # uncalibrated checkpoint stores, and `-m/1e-12` overflows to `inf` for any
    # negative raw prediction; `d0 * inf` is then **NaN**, not the 0 the identity
    # is supposed to be. Head A's raw output does go negative at a scarce medium,
    # so this poisoned the whole dFBA trajectory at step 0 the first time an
    # uncalibrated checkpoint was composed. 700 is where `exp` overflows float64;
    # below it the term is constant, so `g` stays increasing (`g' = a > 0`).
    return a.reshape(shape) * m - d0.reshape(shape) * np.exp(
        np.minimum(-m / np.maximum(beta.reshape(shape), 1e-12), 700.0)
    )


def deriv(mu_hat, cal) -> np.ndarray:
    """``g'(mu_hat) = a + (d0/beta) exp(-m/beta)``, matching :func:`apply` exactly.

    Anything that differentiates a *reported* ``mu`` through the calibration needs
    this -- `cfs steady-state`'s analytic Jacobian rows, and §13.3's constraint
    gradients. It is positive by construction (``a, d0 >= 0``), which is what keeps
    a calibrated head monotone in ``u``.

    The same exponent clip as :func:`apply`, and for the same reason: an
    uncalibrated checkpoint stores ``beta = 0``, and Head A's raw output does go
    negative at a scarce medium. Above the clip ``exp`` is constant, so the
    derivative there is ``a`` plus a constant rather than ``inf``; with ``d0 = 0``
    -- the identity -- it is exactly ``a``.
    """
    cal = np.asarray(cal)
    d0, beta = cal[:, 0], cal[:, 1]
    a = cal[:, 2] if cal.shape[1] > 2 else np.ones_like(d0)
    shape = (-1,) + (1,) * (np.ndim(mu_hat) - 1)
    m = np.asarray(mu_hat)
    b = np.maximum(beta.reshape(shape), 1e-12)
    return a.reshape(shape) + (d0.reshape(shape) / b) * np.exp(np.minimum(-m / b, 700.0))
