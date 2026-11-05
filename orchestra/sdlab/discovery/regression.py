"""One stock's regression: its change in a step on the levels at the start of the step.

The outside inflow in pieces is removed by taking out each piece's mean (so only the levels' columns
remain), and every candidate set of levels is then fitted from precomputed sums.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class Sums:
    G: np.ndarray
    b: np.ndarray
    yy: float
    T: int
    pieces: int


@dataclass
class Fit:
    beta: np.ndarray
    rss: float
    se: np.ndarray


def demean(a: np.ndarray, cuts: list[int]) -> np.ndarray:
    """a (columns x steps, or steps) with each piece's mean taken out."""
    starts = np.asarray(cuts[:-1])
    lens = np.diff(cuts)
    seg = np.repeat(np.arange(len(lens)), lens)
    means = np.add.reduceat(a, starts, axis=-1) / lens
    return a - means[..., seg]


def row_sums(y: np.ndarray, X: np.ndarray, cuts: list[int] | None) -> Sums:
    yc, Xc = (demean(y, cuts), demean(X, cuts)) if cuts else (y, X)
    return Sums(G=Xc @ Xc.T, b=Xc @ yc, yy=float(yc @ yc), T=y.shape[0], pieces=len(cuts) - 1 if cuts else 0)


def fit_subset(sums: Sums, cols: list[int]) -> Fit:
    if not cols:
        return Fit(beta=np.zeros(0), rss=sums.yy, se=np.zeros(0))
    idx = np.asarray(cols)
    G = sums.G[np.ix_(idx, idx)].copy()
    G[np.diag_indices_from(G)] += 1e-9 * (np.abs(np.diag(G)) + 1)
    b = sums.b[idx]
    try:
        inv = np.linalg.inv(G)
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(G)
    x = inv @ b
    rss = max(sums.yy - float(x @ b), 1e-12)
    s2 = rss / max(1, sums.T - len(cols) - sums.pieces)
    return Fit(beta=x, rss=rss, se=np.sqrt(np.maximum(0, np.diag(inv)) * s2))


def bic(T: int, rss: float, p: int) -> float:
    return T * math.log(max(rss, 1e-12) / T) + p * math.log(T)


def stepwise(sums: Sums, start: list[int], allowed: list[int], add_gain: float, max_steps: int) -> list[int]:
    """Add the level that improves BIC most (by more than add_gain), or drop one whose removal
    improves it by more than 2, until nothing does."""
    chosen = list(start)
    score = bic(sums.T, fit_subset(sums, chosen).rss, len(chosen) + sums.pieces)
    for _ in range(max_steps):
        best = None
        for j in allowed:
            if j in chosen:
                continue
            sc = bic(sums.T, fit_subset(sums, chosen + [j]).rss, len(chosen) + 1 + sums.pieces)
            if sc < score - add_gain and (best is None or sc < best[1]):
                best = (chosen + [j], sc)
        for j in chosen:
            ns = [x for x in chosen if x != j]
            sc = bic(sums.T, fit_subset(sums, ns).rss, len(ns) + sums.pieces)
            if sc < score - 2 and (best is None or sc < best[1]):
                best = (ns, sc)
        if best is None:
            break
        chosen, score = best
    return chosen


def piece_values(y: np.ndarray, X: np.ndarray, cols: list[int], beta: np.ndarray, cuts: list[int]) -> list[dict]:
    """Each piece's level of the outside inflow, with the levels' part taken out."""
    res = y - (beta @ X[cols] if cols else 0)
    return [{"a": a, "b": b, "w": b - a, "v": float(res[a:b].mean())} for a, b in zip(cuts[:-1], cuts[1:])]
