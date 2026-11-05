"""Gross errors found before any model: a single reported value far from both its neighbours, in
the same direction, while the neighbours agree with each other (a level that jumps and stays is a
change; a peak of a curve or a pulse that decays is not an error; one value that jumps and comes
straight back is). Far means beyond CUT robust standard deviations of the point-to-point scatter,
beyond RATIO times the neighbours' own difference, and beyond FRAC of the stock's level around it (a
gross error is large against the level itself). Values just after an announced injection are never
errors."""

from __future__ import annotations

import numpy as np

CUT = 6.0          # robust standard deviations of the scatter (the filter's own check, at 5, follows the fit)
RATIO = 3.0        # times the difference between the two neighbours
FRAC = 0.25        # of the stock's level around the point (the median, plus 10)
LOCAL = 10         # points on each side the scatter and the level are read over
AFTER = 12         # steps after an announced injection (a delay of the report included) left alone


def gross_errors(t: np.ndarray, Y: np.ndarray, seen: np.ndarray, injections: list | None = None) -> np.ndarray:
    """points x stocks: True where a value is a gross error."""
    M, S = Y.shape
    t = np.asarray(t)
    out = np.zeros((M, S), dtype=bool)
    calm = np.ones(M, dtype=bool)
    for j in injections or []:
        calm &= ~((t >= int(j["t"])) & (t <= int(j["t"]) + AFTER))
    for s in range(S):
        idx = np.flatnonzero(seen[:, s])
        y = Y[idx, s]
        if len(y) < 7:
            continue
        a = y[1:-1] - y[:-2]                         # from the one before
        b = y[1:-1] - y[2:]                          # from the one after
        same = np.sign(a) == np.sign(b)
        d = np.where(same, np.minimum(np.abs(a), np.abs(b)), 0.0)
        gap = np.abs(y[2:] - y[:-2])
        # the scatter and the level around each point (a stock's noise and size can change a lot over a run)
        dif = np.abs(np.diff(y))
        sd = np.empty(len(y) - 2)
        level = np.empty(len(y) - 2)
        for i in range(1, len(y) - 1):
            lo, hi = max(0, i - LOCAL), min(len(y), i + LOCAL + 1)
            sd[i - 1] = 1.4826 * float(np.median(dif[lo: hi - 1])) / np.sqrt(2.0)
            level[i - 1] = float(np.median(np.abs(y[lo:hi]))) + 10.0
        hit = (d > CUT * sd) & (d > RATIO * gap) & (d > FRAC * level)
        out[idx[1:-1], s] = hit & calm[idx[1:-1]]
    return out
