"""The mass balance of a hidden stock, read with its counting noise (a Kalman filter).

The plain balance H(t) = H(0) + alpha x cumsum(A's loss) - cumsum(B's arrivals) uses A's loss at
its rate, not at what really left A in each step: the difference is counting noise, and summed over
the run it is a random walk as large as H itself (2000 steps of ~20 a step: ~200). So H is followed
step by step instead, as a filter:

  - each step, B's change tells how much arrived from H (y + kB x B: B's change less what other
    found flows explain, plus what B itself lost), which is d x H(t-1) up to counting noise: this
    corrects the estimate of H(t-1);
  - then H moves by the balance: + alpha x A's loss, - what arrived in B; its uncertainty grows by the
    counting noise of both (a count's variance is its size).

alpha, d and kB (B's own drain, refitted with H) are those under which B's changes are best predicted
(the innovations' likelihood, its scale concentrated out). H's start is not a parameter: the filter
starts from a wide prior.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit


@njit(cache=True)
def _filter(F, y, Bp, other, alpha, d, kb, m0, P0, out):
    """Innovation sum of squares (scaled), sum of log variances, and H at every time (out: len(F) + 1;
    H(t) as estimated once B's change in the step after t is seen)."""
    m, P = m0, P0
    ss = 0.0
    ld = 0.0
    n = F.shape[0]
    for t in range(n):
        arr = y[t] + kb * Bp[t]
        R = max(d * max(m, 0.0), 1.0) + kb * Bp[t] + abs(other[t]) + 1.0
        S = d * d * P + R
        e = arr - d * m
        ss += e * e / S
        ld += math.log(S)
        K = P * d / S
        m = m + K * e
        P = (1.0 - K * d) * P
        out[t] = m
        m = m + alpha * F[t] - arr
        P = P + alpha * F[t] + kb * Bp[t] + 1.0
    out[n] = m
    return ss, ld


def run(F: np.ndarray, y: np.ndarray, Bp: np.ndarray, other: np.ndarray, alpha: float, d: float, kb: float, m0: float, P0: float) -> tuple[float, np.ndarray]:
    """(-2 log-likelihood with the noise scale concentrated out, H at every time)."""
    out = np.zeros(len(F) + 1)
    ss, ld = _filter(F, y, Bp, other, alpha, d, kb, m0, P0, out)
    n = len(F)
    return n * math.log(max(ss / n, 1e-300)) + ld, out


def fit(F: np.ndarray, y: np.ndarray, B: np.ndarray, other: np.ndarray, kb0: float, refit_kb: bool = True,
        d_starts=(0.03, 0.1, 0.3)) -> dict | None:
    """alpha, d (and kB) by the filter's likelihood, from a few starting delays; H's level series."""
    from scipy.optimize import minimize

    F, y, Bp, other = (np.ascontiguousarray(x, dtype=float) for x in (F, y, B[:-1], other))
    if len(F) < 20 or not np.isfinite(F).all() or float(F.mean()) <= 0:
        return None
    arr0 = float((y + kb0 * Bp).mean())
    P0 = max(1.0, (arr0 * 30) ** 2)

    def unpack(p):
        alpha, d = p[0], p[1]
        kb = p[2] if refit_kb else kb0
        return alpha, d, kb

    def cost(p):
        alpha, d, kb = unpack(p)
        if alpha <= 0 or d <= 1e-4 or d >= 1 or kb < 0:
            return 1e300
        return run(F, y, Bp, other, alpha, d, kb, arr0 / d, P0)[0]

    best = None
    for d0 in d_starts:
        alpha0 = max(1e-3, arr0 / float(F.mean()))
        x0 = [alpha0, d0] + ([max(kb0, 1e-3)] if refit_kb else [])
        opt = minimize(cost, np.array(x0), method="Nelder-Mead", options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 800})
        if best is None or opt.fun < best.fun:
            best = opt
    alpha, d, kb = unpack(best.x)
    if not np.isfinite(best.fun) or best.fun >= 1e299:
        return None
    _, H = run(F, y, Bp, other, alpha, d, kb, arr0 / d, P0)
    return {"alpha": float(alpha), "d": float(d), "kb": float(kb), "h0": float(H[0]), "H": H, "nll": float(best.fun)}
