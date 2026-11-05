# Vendored for kalman_lna from sdlab/discovery/strategies/lna/small.py at commit ec2f9e8 (imports rewritten to the
# vendored copies; otherwise unchanged). Do not edit: re-vendor instead.
"""Many small covariance matrices at once (one per step): inverse, log-determinant and W r, by a
Cholesky factorisation per step (numba; numpy's batched inverse is slower for these sizes)."""

from __future__ import annotations

import math

import numpy as np
from numba import njit


@njit(cache=True)
def inv_logdet(sig: np.ndarray, r: np.ndarray):
    """sig: T x S x S (symmetric positive definite), r: S x T -> (W = sig^-1, sum of log det,
    a = W r per step: T x S). A matrix that is not positive definite gives logdet = inf."""
    T, S, _ = sig.shape
    W = np.empty((T, S, S))
    a = np.empty((T, S))
    L = np.zeros((S, S))
    Li = np.zeros((S, S))
    total = 0.0
    for t in range(T):
        for i in range(S):
            for j in range(i + 1):
                v = sig[t, i, j]
                for k in range(j):
                    v -= L[i, k] * L[j, k]
                if i == j:
                    if v <= 0.0:
                        return W, math.inf, a
                    L[i, i] = math.sqrt(v)
                else:
                    L[i, j] = v / L[j, j]
        for i in range(S):
            total += 2.0 * math.log(L[i, i])
        # Li = L^-1 (lower triangular)
        for i in range(S):
            Li[i, i] = 1.0 / L[i, i]
            for j in range(i):
                v = 0.0
                for k in range(j, i):
                    v -= L[i, k] * Li[k, j]
                Li[i, j] = v / L[i, i]
        for i in range(S):
            for j in range(i + 1):
                v = 0.0
                for k in range(i, S):
                    v += Li[k, i] * Li[k, j]
                W[t, i, j] = v
                W[t, j, i] = v
        for i in range(S):
            v = 0.0
            for j in range(S):
                v += W[t, i, j] * r[j, t]
            a[t, i] = v
    return W, total, a
