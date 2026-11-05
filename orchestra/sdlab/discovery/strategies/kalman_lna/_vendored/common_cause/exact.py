# Vendored for kalman_lna from sdlab/discovery/strategies/common_cause/exact.py at commit ec2f9e8 (imports rewritten to the
# vendored copies; otherwise unchanged). Do not edit: re-vendor instead.
"""Noise-free data: least squares that knows where float precision ends.

On exact data a row's true columns fit to rounding error, so its residual sum of squares is
numerical junk (1e-20 of the signal, or 1e-9 from fit_subset's ridge). Two shared pieces then go
wrong:

- BIC compares T x log(rss): between two junk values, any column that shaves the junk "gains"
  hundreds, so stepwise keeps adding columns, and the greedy path that took a wrong column first
  never finds the sparse exact fit (with an exact collinearity among the levels, several sets fit
  exactly and only the sparsest is the truth);
- PELT prices a change by the variance of the residual's successive differences: on junk that is
  ~0, so the ridge's tiny systematic bias is cut into many pieces.

Both are fixed by a floor at the data's precision: a fit's rss is never below REL x T x the mean
square of every stock's change (so all exact fits tie, and backward steps drop what the floor
cannot see), and PELT's noise variance is never below the same scale. With counting noise the
floor is far below the noise and changes nothing.

`floored(dy)` patches the shared functions in place for the duration of one identify call (each
ladder and bench item runs in its own process, one at a time), and restores them after.
"""

from __future__ import annotations

import math
from contextlib import contextmanager

import numpy as np

from orchestra.sdlab.discovery import matrix as _matrix
from orchestra.sdlab.discovery import regression as _regression
from orchestra.sdlab.discovery.changes import _last_cut
from orchestra.sdlab.discovery.regression import Fit
from orchestra.sdlab.discovery.strategies.kalman_lna._vendored.common_cause import joint as _joint
from orchestra.sdlab.discovery.strategies.kalman_lna._vendored.common_cause import rows as _rows

REL = 1e-8           # the floor on a fit's mean square residual, relative to the changes' mean square

_ORIG = _regression.fit_subset
_ORIG_STEPWISE = _regression.stepwise


REL_TIE = 1.0 + 1e-9  # a fit within this factor of the floor is exact


def make_fit_subset(scale: float):
    def fit_subset(sums, cols):
        f = _ORIG(sums, cols)
        floor = REL * scale * sums.T
        if f.rss < floor:
            f = Fit(beta=f.beta, rss=floor, se=f.se)
        return f
    fit_subset.floor = lambda sums: REL * scale * sums.T
    return fit_subset


def make_pelt(scale: float):
    def pelt(r: np.ndarray, min_piece: int, mult: float) -> list[int]:
        r = np.nan_to_num(np.asarray(r, dtype=float))
        n = r.shape[0]
        dv = float(np.sum(np.diff(r) ** 2))
        var = max(dv / (2 * max(1, n - 1)), REL * scale * 1e2, 1e-300)
        last = _last_cut(r, int(min_piece), mult * math.log(n) * var)
        cuts, t = [], n
        while t > 0:
            cuts.append(t)
            t = int(last[t])
        cuts.append(0)
        return cuts[::-1]

    return pelt


def make_scan(scale: float, min_piece: int):
    pelt = make_pelt(scale)
    return lambda r, phase: pelt(r, min_piece, 0.5 if phase == "fine" else 1)


def scale_of(run: dict, ids: list[str]) -> float:
    lv = np.nan_to_num(np.array([np.asarray(run["levels"].get(i, []), dtype=float) for i in ids]))
    d = np.diff(lv, axis=1)
    return float(np.mean(d * d)) if d.size else 1.0


def make_stepwise(fit_subset):
    """regression.stepwise, then (only when the fit is exact: at the floor) exchanges: a chosen
    column swapped for another, then columns dropped while the fit stays exact, kept when the row
    ends with fewer columns. With an exact collinearity among the levels the greedy path can end
    at an exact fit that is not the sparsest (the truth); an exchange walks along the tie."""
    def at_floor(sums, cols):
        return fit_subset(sums, cols).rss <= REL_TIE * floor_of(sums)

    def floor_of(sums):
        return fit_subset.floor(sums)

    def drops(sums, cols):
        cols = list(cols)
        changed = True
        while changed:
            changed = False
            for j in list(cols):
                trial = [q for q in cols if q != j]
                if at_floor(sums, trial):
                    cols, changed = trial, True
                    break
        return cols

    def stepwise(sums, start, allowed, add_gain, max_steps):
        chosen = _ORIG_STEPWISE(sums, start, allowed, add_gain, max_steps)
        if not chosen or not at_floor(sums, chosen):
            return chosen
        chosen = drops(sums, chosen)
        improved = True
        while improved:
            improved = False
            for j in list(chosen):
                for i in allowed:
                    if i in chosen:
                        continue
                    trial = [q for q in chosen if q != j] + [i]
                    if not at_floor(sums, trial):
                        continue
                    trial = drops(sums, trial)
                    if len(trial) < len(chosen):
                        chosen, improved = trial, True
                        break
                if improved:
                    break
        return chosen

    return stepwise


@contextmanager
def floored(scale: float):
    fs, pe = make_fit_subset(scale), make_pelt(scale)
    saved = (_regression.fit_subset, _matrix.fit_subset, _rows.fit_subset, _matrix.pelt, _joint.pelt, _matrix.stepwise)
    _regression.fit_subset = _matrix.fit_subset = _rows.fit_subset = fs
    _matrix.pelt = _joint.pelt = pe
    _matrix.stepwise = make_stepwise(fs)
    try:
        yield
    finally:
        (_regression.fit_subset, _matrix.fit_subset, _rows.fit_subset, _matrix.pelt, _joint.pelt, _matrix.stepwise) = saved


def _key(cols: list[int], col: list[tuple[int, int]], s: int) -> tuple:
    """How plain an exact row is: fewer columns, then fewer columns reading other stocks, then those
    nearer the row's own stock in the page's order (related stocks are listed together), then
    those listed before it (a flow's source is usually listed first)."""
    cross = [col[q][0] - s for q in cols if col[q][0] != s]
    return (len(cols), len(cross), sum(abs(d) for d in cross), sum(d > 0 for d in cross))


def tiebreak(row: dict, s: int, y: np.ndarray) -> dict:
    """A row that fits exactly (at the floor) is moved, by exchanges and drops that keep it exact, to
    the plainest exact set (_key): with collinear levels (identical stocks; stocks whose gaps decay
    alike) every one of them fits, and the row's own level, then no cross term, is the answer
    parsimony gives. Rows that do not fit exactly are returned unchanged. Call inside floored()."""
    fs = _regression.fit_subset
    floor = getattr(fs, "floor", None)
    cols, X = list(row.get("set") or []), row.get("X")
    if floor is None or X is None or not len(X) or not cols:
        return row
    from orchestra.sdlab.discovery.regression import row_sums
    sums = row_sums(y, X, row.get("cuts"))
    exact = lambda c: fs(sums, c).rss <= REL_TIE * floor(sums)  # noqa: E731
    if not exact(cols):
        return row
    col = row["col"]
    best = (cols, _key(cols, col, s))
    improved = True
    while improved:
        improved = False
        cur = best[0]
        trials = [[q for q in cur if q != j] for j in cur]
        trials += [[q for q in cur if q != j] + [i] for j in cur for i in range(len(col)) if i not in cur]
        for t in trials:
            if not exact(t):
                continue
            # drop what the exact fit does not need
            changed = True
            while changed:
                changed = False
                for j in list(t):
                    tt = [q for q in t if q != j]
                    if exact(tt):
                        t, changed = tt, True
                        break
            k = _key(t, col, s)
            if k < best[1]:
                best, improved = (t, k), True
                break
    if best[0] == cols:
        return row
    f = fs(sums, best[0])
    return {**row, "set": best[0], "rss": f.rss, "beta": f.beta, "p": len(best[0]) + (len(row["cuts"]) - 1 if row.get("cuts") else 0)}
