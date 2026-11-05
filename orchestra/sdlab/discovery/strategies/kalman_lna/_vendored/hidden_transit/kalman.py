# Vendored for kalman_lna from sdlab/discovery/strategies/hidden_transit/kalman.py at commit ec2f9e8 (imports rewritten to the
# vendored copies; otherwise unchanged). Do not edit: re-vendor instead.
"""A transit stock between two measured stocks, followed by a one-state Kalman filter.

The local model of a pair (sender A, receiver B, unmeasured H between them), one step at a time:

    A(t) - A(t-1) = xA(t) + cA(t) - kA A(t-1)               (xA: what the kept flows do to A)
    H(t)          = (1 - d) H(t-1) + kin A(t-1)            (kin <= kA; the rest leaves to outside)
    B(t) - B(t-1) = xB(t) + cB + d H(t-1) - kb B(t-1)

Every flow is a count around its rate (variance = rate, Poisson), so the noise is shared: what
really left A towards H is seen (with the sign turned) in A's change and is part of H's change; what
really left H is part of H's change and of B's. Because A and B are measured exactly, only H is
uncertain: a scalar filter. Each step the change of A and of B are predicted, their errors weighted
by their variances (the likelihood), and H corrected by both (the covariance of A's change with H's
is minus the A -> H noise: the main reason a filter on B alone, or a regression on a copy of A,
reads the delay too long).

`scale` 1 fixes the noise at the counts' own variance (the data are counts); 0 concentrates an
unknown scale out (noise of unknown size, or none: the fit is then a weighted least squares).
The plain alternative (no H: B gets kd x A directly) is scored by the same likelihood (`plain`).

A's outside inflow cA(t) is one constant or pieces (the main line's cuts): for a given kA each
piece's value is the mean of A's change plus kA x A over it (A's row alone says it; H does not enter
A's mean), so it is never a number the optimiser searches.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit

FLOOR = 0.05       # a count's variance is never taken below this (a rate of ~0 still moves sometimes)


@njit(cache=True)
def _transit(A, B, xA, gA, xB, gB, cAs, kA, kin, d, kb, cB, m0, P0, out):
    n = A.shape[0]
    m, P = m0, P0
    ss = 0.0
    ld = 0.0
    out[0] = m
    for t in range(1, n):
        a = A[t - 1]
        b = B[t - 1]
        cA = cAs[t - 1]
        vo = max(kin * a, FLOOR)
        vA = gA[t - 1] + abs(cA) + max((kA - kin) * a, 0.0) + FLOOR
        vp = max(d * max(m, 0.0), FLOOR)
        vB = gB[t - 1] + max(kb * b, 0.0) + abs(cB) + FLOOR
        muA = xA[t - 1] + cA - kA * a
        muB = xB[t - 1] + cB + d * m - kb * b
        muH = (1.0 - d) * m + kin * a
        sA = vA + vo
        sB = d * d * P + vp + vB
        sH = (1.0 - d) * (1.0 - d) * P + vo + vp
        cAH = -vo
        cBH = d * (1.0 - d) * P - vp
        eA = (A[t] - a) - muA
        eB = (B[t] - b) - muB
        ss += eA * eA / sA + eB * eB / sB
        ld += math.log(sA) + math.log(sB)
        m = muH + cAH / sA * eA + cBH / sB * eB
        P = sH - cAH * cAH / sA - cBH * cBH / sB
        if P < 1e-9:
            P = 1e-9
        out[t] = m
    return ss, ld


@njit(cache=True)
def _plain(A, B, xA, gA, xB, gB, cAs, kA, kd, kb, cB):
    n = A.shape[0]
    ss = 0.0
    ld = 0.0
    for t in range(1, n):
        a = A[t - 1]
        b = B[t - 1]
        cA = cAs[t - 1]
        vd = max(kd * a, FLOOR)
        sA = gA[t - 1] + abs(cA) + max((kA - kd) * a, 0.0) + vd + FLOOR
        sB = gB[t - 1] + max(kb * b, 0.0) + abs(cB) + vd + FLOOR
        eA = (A[t] - a) - (xA[t - 1] + cA - kA * a)
        eB = (B[t] - b) - (xB[t - 1] + cB + kd * a - kb * b)
        ss += eA * eA / sA + eB * eB / sB
        ld += math.log(sA) + math.log(sB)
    return ss, ld


def _nll(ss: float, ld: float, n: int, scale: float) -> float:
    """-2 log-likelihood (up to a constant): fixed scale, or the scale concentrated out."""
    if scale > 0:
        return ss / scale + ld + n * math.log(scale)
    return n * math.log(max(ss / n, 1e-300)) + ld


class Pair:
    """The data of one pair: the two levels, what the kept flows do to each (net and gross, per
    step), and the pieces of A's outside inflow (seg: each step's piece, or None: no inflow of its
    own beyond the kept flows)."""

    def __init__(self, A, B, xA, gA, xB, gB, scale: float, seg: np.ndarray | None = None):
        self.A, self.B = (np.ascontiguousarray(v, dtype=float) for v in (A, B))
        self.xA, self.gA, self.xB, self.gB = (np.ascontiguousarray(v, dtype=float) for v in (xA, gA, xB, gB))
        self.scale = scale
        self.n = 2 * (len(self.A) - 1)
        self.P0 = max(1.0, float(np.mean(np.abs(self.B))) ** 2)
        self.seg = None if seg is None else np.asarray(seg, dtype=int)
        if self.seg is not None:
            self.count = np.maximum(1, np.bincount(self.seg))
            self.dA = np.diff(self.A) - self.xA

    @property
    def pieces(self) -> int:
        return 0 if self.seg is None else len(self.count)

    def values(self, kA: float) -> np.ndarray:
        """A's outside inflow per piece, given kA."""
        if self.seg is None:
            return np.zeros(0)
        return np.bincount(self.seg, weights=self.dA + kA * self.A[:-1], minlength=len(self.count)) / self.count

    def a_in(self, kA: float) -> np.ndarray:
        if self.seg is None:
            return np.zeros(len(self.A) - 1)
        return np.ascontiguousarray(self.values(kA)[self.seg])

    def transit(self, kA, kin, d, kb, cB, m0) -> tuple[float, np.ndarray]:
        out = np.zeros(len(self.A))
        ss, ld = _transit(self.A, self.B, self.xA, self.gA, self.xB, self.gB, self.a_in(kA), kA, kin, d, kb, cB, m0, self.P0, out)
        return _nll(ss, ld, self.n, self.scale), out

    def plain(self, kA, kd, kb, cB) -> float:
        ss, ld = _plain(self.A, self.B, self.xA, self.gA, self.xB, self.gB, self.a_in(kA), kA, kd, kb, cB)
        return _nll(ss, ld, self.n, self.scale)
