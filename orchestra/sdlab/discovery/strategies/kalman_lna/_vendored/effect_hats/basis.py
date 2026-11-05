# Vendored for kalman_lna from sdlab/discovery/strategies/effect_hats/basis.py at commit ec2f9e8 (imports rewritten to the
# vendored copies; otherwise unchanged). Do not edit: re-vendor instead.
"""Hat functions: a curve of a stock's level as a weighted sum of tents over a few knots.

A piecewise-linear curve through (x_j, w_j) is sum_j w_j h_j(x), with h_j the tent that is 1 at knot
j and 0 at its neighbours (the first and last stay 1 beyond the ends, as an SD Lab table effect stays
flat beyond its points). The tents sum to 1, so a rate base x f(C) is linear in the w_j: the columns
base x h_j(C) go into least squares like levels, and the fitted weights are the table's values.
"""

from __future__ import annotations

import numpy as np


def knots(x: np.ndarray, k: int) -> np.ndarray:
    """k knots spread evenly over where the level spends its time (0.5th to 99.5th percentile)."""
    x = np.asarray(x, dtype=float)
    lo, hi = np.percentile(x, 0.5), np.percentile(x, 99.5)
    if hi - lo <= 1e-9 * max(1.0, abs(hi)):
        lo, hi = float(np.min(x)), float(np.max(x))
    return np.linspace(lo, hi, max(2, k))


def hats(x: np.ndarray, kn: np.ndarray) -> np.ndarray:
    """K x n: tent j of every value of x (flat beyond the end knots)."""
    x = np.clip(np.asarray(x, dtype=float), kn[0], kn[-1])
    K = len(kn)
    out = np.zeros((K, x.shape[0]))
    i = np.clip(np.searchsorted(kn, x, side="right") - 1, 0, K - 2)
    w = (x - kn[i]) / np.maximum(1e-12, kn[i + 1] - kn[i])
    out[i, np.arange(x.shape[0])] = 1 - w
    out[i + 1, np.arange(x.shape[0])] += w
    return out


def curve(x: np.ndarray, kn: np.ndarray, w: np.ndarray) -> np.ndarray:
    return np.asarray(w, dtype=float) @ hats(x, kn)
