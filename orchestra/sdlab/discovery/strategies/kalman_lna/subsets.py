"""Best-subset least squares in batches (k8s-312 round 2, docs/research/fixes/312-kalman_lna-r2.md).

rows.py ranks every subset of up to three readers by its least-squares fit, for every row, every derived column and
every hidden-stock candidate: 2.6 million `numpy.linalg.lstsq` calls on 312 none-long, most of its rows pass. Here
all subsets of one size are fitted at once from the scaled Gram matrix (a stack of k x k solves) and their sums of
squares are computed on the residuals themselves (exact for the coefficients found, so no cancellation below the
rounding). The few best are then refitted by least squares, so the subset chosen and its numbers are what the
one-at-a-time search gave; only the ranking of the rest comes from the batch.
"""

from __future__ import annotations

import numpy as np

CELLS = 4_000_000      # a batch holds at most this many numbers (steps x subsets x readers)
REFIT = 3              # the best this many by the batch are refitted by least squares


def lstsq(A: np.ndarray, y: np.ndarray):
    b, *_ = np.linalg.lstsq(A, y, rcond=None)
    e = y - A @ b
    return b, float(e @ e)


def normal(M: np.ndarray, y: np.ndarray) -> np.ndarray | None:
    """y's least-squares coefficients on M from the scaled normal equations; None where they are singular."""
    sc = np.sqrt((M * M).sum(axis=0))
    if not np.all(sc > 0):
        return None
    Ms = M / sc
    try:
        b = np.linalg.solve(Ms.T @ Ms, Ms.T @ y) / sc
    except np.linalg.LinAlgError:
        return None
    return b if np.all(np.isfinite(b)) else None


def _solve(G: np.ndarray, g: np.ndarray) -> np.ndarray:
    """A stack of k x k systems; a singular one's numbers are NaN (its subset is then fitted alone)."""
    try:
        b = np.linalg.solve(G, g[..., None])[..., 0]
    except np.linalg.LinAlgError:
        b = np.full(g.shape, np.nan)
        for m in range(len(G)):
            try:
                b[m] = np.linalg.solve(G[m], g[m])
            except np.linalg.LinAlgError:
                pass
    return b


def fits(X: np.ndarray, y: np.ndarray, combos: np.ndarray, rows: np.ndarray | None = None):
    """(coefs (m, k), sse (m,)) of y on X's columns combos[m] for every subset m, fitted on the steps rows (all when
    None) and scored there; the coefficients from the scaled Gram matrix (least squares where it is singular)."""
    Xo, yo = (X, y) if rows is None else (X[rows], y[rows])
    m, k = combos.shape
    sc = np.sqrt((Xo * Xo).sum(axis=0))
    sc[sc == 0] = 1.0
    Xs = Xo / sc
    G, g = Xs.T @ Xs, Xs.T @ yo
    B = _solve(G[combos[:, :, None], combos[:, None, :]], g[combos]) / sc[combos]
    for i in np.flatnonzero(~np.all(np.isfinite(B), axis=1)):
        B[i] = lstsq(Xo[:, combos[i]], yo)[0]
    return B, sse(Xo, yo, combos, B)


def sse(X: np.ndarray, y: np.ndarray, combos: np.ndarray, B: np.ndarray, fn=None) -> np.ndarray:
    """Each subset's sum of squares of y - fn(X_sub b) on every step (fn: e.g. the floor at 0; None: as is)."""
    n = X.shape[0]
    m, k = combos.shape
    out = np.empty(m)
    step = max(1, CELLS // max(1, n * k))
    for s in range(0, m, step):
        c, b = combos[s:s + step], B[s:s + step]
        P = np.einsum("nmk,mk->nm", X[:, c], b)
        if fn is not None:
            P = fn(P)
        out[s:s + step] = ((y[:, None] - P) ** 2).sum(axis=0)
    return out


def reread(X: np.ndarray, y: np.ndarray, combos: np.ndarray, B: np.ndarray, least: int, fn=None):
    """(coefs, sse, enough) of each subset refitted on the steps where its term acts (X_sub b > 0), scored with fn on
    every step; enough: it acts on at least `least` steps (else its row is left as it was)."""
    n = X.shape[0]
    m, k = combos.shape
    B2, s2, ok = B.copy(), np.full(m, np.inf), np.zeros(m, dtype=bool)
    step = max(1, CELLS // max(1, n * k))
    for s in range(0, m, step):
        c = combos[s:s + step]
        Xc = X[:, c]                                          # (n, chunk, k)
        act = np.einsum("nmk,mk->nm", Xc, B[s:s + step]) > 0
        ok[s:s + step] = act.sum(axis=0) >= least
        Xa = Xc * act[:, :, None]
        G = np.einsum("nmi,nmj->mij", Xa, Xa)
        g = np.einsum("nmi,n->mi", Xa, y)
        sc = np.sqrt(np.einsum("mii->mi", G))
        sc[sc == 0] = 1.0
        b = _solve(G / (sc[:, :, None] * sc[:, None, :]), g / sc) / sc
        for i in np.flatnonzero(~np.all(np.isfinite(b), axis=1) & ok[s:s + step]):
            rows = act[:, i]
            b[i] = lstsq(X[rows][:, c[i]], y[rows])[0]
        B2[s:s + step] = np.where(ok[s:s + step, None], b, B[s:s + step])
    s2[ok] = sse(X, y, combos[ok], B2[ok], fn)
    return B2, s2, ok


def best(X: np.ndarray, y: np.ndarray, combos: list[tuple]):
    """(subset, coefs, sse) of the subset among combos (all the same size) with the least least-squares sse: the
    batch ranks them, the REFIT best are refitted by least squares (ties to the first in combos' order)."""
    C = np.asarray(combos, dtype=int).reshape(len(combos), -1)
    _, s = fits(X, y, C)
    top = None
    for i in sorted(np.argsort(s, kind="stable")[:REFIT]):
        b, e = lstsq(X[:, combos[i]], y)
        if top is None or e < top[2]:
            top = (combos[i], b, e)
    return top
