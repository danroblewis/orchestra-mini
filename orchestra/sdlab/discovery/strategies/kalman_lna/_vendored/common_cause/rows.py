# Vendored for kalman_lna from sdlab/discovery/strategies/common_cause/rows.py at commit ec2f9e8 (imports rewritten to the
# vendored copies; otherwise unchanged). Do not edit: re-vendor instead.
"""(earned_pieces, vendored at a3a5e01) Each row's levels and outside-inflow form chosen together, every change paying its full price.

The matrix chooses a row's levels in the presence of fine, cheap pieces (so that the levels are not
biased by unexplained changes) and only at the end asks whether the pieces pay. When they do not,
the row keeps the levels chosen beside them: often too few (a level the pieces stood in for), so
that the pieces then pay after all, for the levels' sake. Here the matrix's row is one candidate;
the others are the levels chosen afresh by backward elimination from every level the row may read
(and, with numbers per configuration, every level's number per configuration), with no outside
inflow, a constant one, or the matrix's pieces. All are compared by BIC with each change in the
outside inflow priced as a value and a place (2 parameters).
"""

from __future__ import annotations

import math

import numpy as np

from scipy.stats import chi2

from orchestra.sdlab.discovery.regression import fit_subset, row_sums

ALPHA = 0.01        # the whiteness test's level


def _bic(T: int, rss: float, p: float) -> float:
    return T * math.log(max(rss, 1e-12) / T) + p * math.log(T)


def inflow_params(cuts: list[int] | None, per_change: float = 2) -> float:
    """A constant is one parameter; every change adds a value and a place."""
    return 0 if not cuts else 1 + per_change * (len(cuts) - 2)


def columns(s: int, ids: list[str], prev: np.ndarray, cfg: np.ndarray, L: int, allowed: list[int]) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """The row's columns: each allowed level, and with configurations its number's departure in
    every configuration but the first (col: (stock index, configuration or -1 for shared))."""
    X, col = [], []
    for j in allowed:
        X.append(prev[j])
        col.append((j, -1))
        for c in range(1, L):
            m = cfg == c
            if m.any() and not m.all():
                X.append(np.where(m, prev[j], 0.0))
                col.append((j, c))
    return np.array(X) if X else np.zeros((0, prev.shape[1])), col


def backward(y: np.ndarray, X: np.ndarray, cuts: list[int] | None, start: list[int], extra_p: float) -> tuple[list[int], float, float]:
    """Backward elimination by BIC from start (then re-adding any column that pays): the chosen
    columns, rss and BIC."""
    T = y.shape[0]
    sums = row_sums(y, X, cuts)
    chosen = list(start)

    def sc(cols):
        f = fit_subset(sums, cols)
        return _bic(T, f.rss, len(cols) + extra_p), f.rss

    cur, rss = sc(chosen)
    while True:
        best = None
        for j in range(X.shape[0]):
            trial = [q for q in chosen if q != j] if j in chosen else chosen + [j]
            s, r = sc(trial)
            if s < cur - 1e-9 and (best is None or s < best[1]):
                best = (trial, s, r)
        if best is None:
            return chosen, rss, cur
        chosen, cur, rss = best


def white(e: np.ndarray, alpha: float = ALPHA) -> bool:
    """Ljung-Box: no autocorrelation in e up to lag ceil(sqrt(T)) at level alpha (a row that leaves
    a slow pattern in its residuals is missing something: here, an outside inflow that changes)."""
    e = np.asarray(e, dtype=float) - float(np.mean(e))
    n = e.shape[0]
    v = float(e @ e)
    if n < 10 or v <= 0:
        return True
    m = min(n - 2, int(math.ceil(math.sqrt(n))))
    r = np.array([float(e[k:] @ e[:-k]) / v for k in range(1, m + 1)])
    q = n * (n + 2) * float(np.sum(r * r / (n - np.arange(1, m + 1))))
    return bool(chi2.sf(q, m) > alpha)


def reselect(row: dict, s: int, ids: list[str], stocks: list[dict], prev: np.ndarray, dy: np.ndarray, cfg: np.ndarray, L: int,
             per_change: float = 2, alpha: float = ALPHA) -> dict:
    """The row as the matrix fitted it, or the levels chosen afresh with no outside inflow or a
    constant one. A row with pieces is replaced only by one whose residuals are white (nothing left
    for pieces to explain) and whose BIC, every change priced as a value and a place, is lower; a
    row without pieces by one with a lower BIC."""
    T = dy.shape[1]
    y = dy[s]
    group = {x["id"]: x.get("group") for x in stocks}
    final = {x["id"] for x in stocks if x.get("final")}
    allowed = [j for j in range(len(ids)) if not (ids[s] in final and j == s)
               and (j == s or not group.get(ids[j]) or not group.get(ids[s]) or group[ids[j]] == group[ids[s]])]
    pieces = bool(row["cuts"]) and len(row["cuts"]) > 2
    best = {"row": row, "bic": _bic(T, row["rss"], len(row["set"]) + inflow_params(row["cuts"], per_change)), "name": "matrix"}
    X, col = columns(s, ids, prev, cfg, L, allowed)
    if not len(col):
        return best
    # with pieces, the levels are chosen beside them (so that changes in the inflow cannot hide in
    # levels that happen to follow them), and then the pieces are dropped
    start = backward(y, X, row["cuts"], list(range(len(col))), inflow_params(row["cuts"], per_change))[0] if pieces else None
    for name, cuts in (("none", None), ("constant", [0, T])):
        if pieces:
            chosen = start
            f = fit_subset(row_sums(y, X, cuts), chosen)
            rss, b = f.rss, _bic(T, f.rss, len(chosen) + inflow_params(cuts, per_change))
        else:
            chosen, rss, b = backward(y, X, cuts, list(range(len(col))), inflow_params(cuts, per_change))
        if b >= best["bic"] - 1e-9:
            continue
        if pieces:
            e = y - (f.beta @ X[chosen] if chosen else 0.0)
            if not white(e - (e.mean() if cuts else 0.0), alpha):
                continue
        best = {"row": {**row, "X": X, "col": col, "set": chosen, "cuts": cuts, "rss": rss,
                        "p": len(chosen) + (len(cuts) - 1 if cuts else 0)}, "bic": b, "name": name}
    return best
