# Vendored for kalman_lna from sdlab/discovery/strategies/effect_hats/screen.py at commit ec2f9e8 (imports rewritten to the
# vendored copies; otherwise unchanged). Do not edit: re-vendor instead.
"""Where an effect might be: a group of tent columns base x h_j(C) that explains what the rows leave.

For every pair (base: a measured stock or 1, C: a measured stock that moves), the group of K tent
columns of C times the base is projected, after the levels (and the pieces of each row's changing
outside inflow, as effects.py does), on what each row leaves unexplained. The group's gain in a row is
the share of that row's noise it explains; a flow shows in up to two rows (where it leaves, where it
arrives), so a pair's score is its two best rows less the group's price (K - 1 numbers per row).
A product base x C is one column whichever stock is called the base, so both readings are kept as
candidates and the joint fit and the judge choose.
"""

from __future__ import annotations

import math

import numpy as np

from orchestra.sdlab.discovery.changes import pelt
from orchestra.sdlab.discovery.regression import demean
from orchestra.sdlab.discovery.strategies.kalman_lna._vendored.effect_hats.basis import hats, knots


def _resid(Z: np.ndarray, Q: np.ndarray) -> np.ndarray:
    return Z - (Z @ Q) @ Q.T


def scan(stocks: list[dict], prev: np.ndarray, dy: np.ndarray, opts: dict, extra: list[np.ndarray] | None = None,
         skip: set | None = None) -> list[dict]:
    """[{"base", "stock", "rows": [(row, gain)], "score"}] best first, at most opts["most"]; `extra`:
    columns (effects already in the model) taken out with the levels; `skip`: (base, stock) pairs
    not to offer."""
    ids = [s["id"] for s in stocks]
    S, n = prev.shape
    group = {s["id"]: s.get("group") for s in stocks}
    final = {s["id"] for s in stocks if s.get("final")}
    K = opts.get("screen_knots", 4)
    D = np.vstack([np.ones(n), prev] + [np.asarray(e, dtype=float)[None, :n] for e in extra or []])
    try:
        U, sv, _ = np.linalg.svd(np.nan_to_num(D.T), full_matrices=False)
    except np.linalg.LinAlgError:
        return []
    Q = U[:, sv > 1e-9 * sv[0]]
    R = _resid(dy, Q)
    cuts = [pelt(R[s], 5, 1) if opts.get("pieces", True) else [0, n] for s in range(S)]
    Rd = np.array([demean(R[s], cuts[s]) for s in range(S)])
    var = np.maximum(1e-12, (Rd ** 2).mean(axis=1))
    log_n = math.log(n)
    out = []
    for c in range(S):
        if np.ptp(prev[c]) <= 1e-9 * max(1.0, float(np.max(np.abs(prev[c])))):
            continue
        H = hats(prev[c], knots(prev[c], K))
        for a in [None] + list(range(S)):
            if a is not None and group[ids[a]] and group[ids[c]] and group[ids[a]] != group[ids[c]]:
                continue
            if skip and (None if a is None else ids[a], ids[c]) in skip:
                continue
            Z = H if a is None else H * prev[a]
            Zr = _resid(Z, Q)
            gains = []
            for s in range(S):
                if ids[s] in final and a == s:
                    gains.append(0.0)
                    continue
                Zd = demean(Zr, cuts[s])
                try:
                    u, sz, _ = np.linalg.svd(np.nan_to_num(Zd.T), full_matrices=False)
                except np.linalg.LinAlgError:
                    sz = np.zeros(0)
                if not len(sz) or sz[0] <= 1e-12:
                    gains.append(0.0)
                    continue
                u = u[:, sz > 1e-6 * sz[0]]
                proj = u.T @ Rd[s]
                gains.append(float(proj @ proj) / var[s] - u.shape[1] * log_n)
            order = sorted(range(S), key=lambda s: -gains[s])[:2]
            rows = [(ids[s], gains[s]) for s in order if gains[s] > opts.get("row_gain", 2 * log_n)]
            score = sum(g for _, g in rows)
            if rows and score > opts.get("screen_gain", 6 * log_n):
                out.append({"base": None if a is None else ids[a], "stock": ids[c], "rows": rows, "score": score})
    out.sort(key=lambda x: -x["score"])
    return out[: opts.get("most", 6)]


def placements(cand: dict) -> list[tuple[str | None, str | None]]:
    """(from, to) the effect's flow may run between, given the rows it shows in."""
    rows = [r for r, _ in cand["rows"]]
    out = []
    if len(rows) == 2:
        p, q = rows
        out += [(p, q), (q, p)]
    for p in rows:
        out += [(p, None), (None, p)]
    # a flow out of its base stock may show most where it arrives and in a stock it uses up (a
    # co-flow), and least in the base's own row
    b = cand.get("base")
    if b is not None and b not in rows:
        out += [(b, p) for p in rows]
    return out


def fit_curve(ids: list[str], prev: np.ndarray, dy: np.ndarray, base: str | None, stock: str, frm: str | None, to: str | None,
              kn: np.ndarray, pieces: bool = True, also: tuple = ()) -> dict | None:
    """The effect's tent weights fitted jointly in the rows of its ends: the shared group with the
    flow's signs, each row with its own levels (less the base's level where it would make the group
    ambiguous) and its own outside inflow (pieces of what is left). {weights, rss, positive}."""
    S, n = prev.shape
    idx = {s: i for i, s in enumerate(ids)}
    H = hats(prev[idx[stock]], kn)
    Z = H if base is None else H * prev[idx[base]]
    K = len(kn)
    rows = [(r, sg) for r, sg in ((frm, -1.0), (to, 1.0)) if r is not None]
    # rows in `also` are fitted alongside without the effect (all their levels), so placements that
    # touch different rows are compared on the same rows
    rows += [(r, 0.0) for r in also if r not in {frm, to}]
    # the base's own column in a row would be the same as the group's floor there; with two rows the
    # group's floor is pinned by where the flow arrives, so only the receiving row loses it
    excl = {}
    for r, sg in rows:
        drop = {"one"} if base is None else {base}
        if sg == 0 or (frm is not None and to is not None and sg < 0):
            drop = set()
        excl[r] = drop
    cols_per_row = {}
    for r, _ in rows:
        cs = [] if "one" in excl[r] else [np.ones(n)]
        cs += [prev[j] for j in range(S) if ids[j] not in excl[r]]
        cols_per_row[r] = np.array(cs)
    # first pass without pieces, then each row's pieces from what it leaves
    cuts = {r: [0, n] for r, _ in rows}
    for p in range(2 if pieces else 1):
        blocks_y, blocks_X = [], []
        P = K + sum(cols_per_row[r].shape[0] + len(cuts[r]) - 2 for r, _ in rows)
        off = K
        for r, sg in rows:
            m = cols_per_row[r].shape[0] + len(cuts[r]) - 2
            Xr = np.zeros((n, P))
            Xr[:, :K] = sg * Z.T
            Xr[:, off:off + cols_per_row[r].shape[0]] = cols_per_row[r].T
            for q, (a, b) in enumerate(zip(cuts[r][1:-1], cuts[r][2:])):
                Xr[a:b, off + cols_per_row[r].shape[0] + q] = 1.0
            off += m
            blocks_X.append(Xr)
            blocks_y.append(dy[idx[r]])
        X, y = np.vstack(blocks_X), np.concatenate(blocks_y)
        try:
            theta = np.linalg.lstsq(np.nan_to_num(X), np.nan_to_num(y), rcond=None)[0]
        except np.linalg.LinAlgError:
            return None
        res = (y - X @ theta).reshape(len(rows), n)
        if p == 0 and pieces:
            cuts = {r: pelt(res[i], 5, 1) for i, (r, _) in enumerate(rows)}
    w = theta[:K]
    f = H.T @ w
    return {"weights": w, "rss": float((res ** 2).sum()), "row_rss": (res ** 2).sum(axis=1), "positive": float(np.mean(f > 0)),
            "rows": [r for r, _ in rows]}
