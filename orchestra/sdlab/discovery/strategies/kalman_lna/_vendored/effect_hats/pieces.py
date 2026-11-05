# Vendored for kalman_lna from sdlab/discovery/strategies/effect_hats/pieces.py at commit ec2f9e8 (imports rewritten to the
# vendored copies; otherwise unchanged). Do not edit: re-vendor instead.
"""Outside inflow pieces as a few pooled values, re-cut where the inflow really changes.

The matrix method cuts a row's outside inflow into pieces (PELT on what its levels leave) and gives
each piece its own value. Many pieces of a stock that relaxes inside each of them take up part of its
drain: with a value per stretch, the drain is read about 8% too fast on counting noise (the short
panel's bias), while pooling the stretches that share a value removes it. Here the pieces' values
are pooled (closest first, while the data cannot tell them apart), and the steps are then assigned
again to the pooled values (Viterbi: each change costs log T in the row's noise), so a short piece
cut by noise does not hold a value of its own.
"""

from __future__ import annotations

import math

import numpy as np

from orchestra.sdlab.discovery.strategies.kalman_lna._vendored.effect_hats.joint import _pools, _stoich, refit


def _cluster(values: list[tuple[float, int, list[int]]], s2: float, T: int) -> list[list]:
    clusters = [list(v) for v in values]
    while len(clusters) > 1:
        clusters.sort(key=lambda c: c[0])
        costs = [(a[0] - b[0]) ** 2 * a[1] * b[1] / (a[1] + b[1]) for a, b in zip(clusters, clusters[1:])]
        j = int(np.argmin(costs))
        if costs[j] > s2 * math.log(T):
            break
        a, b = clusters[j], clusters[j + 1]
        clusters[j:j + 2] = [[(a[0] * a[1] + b[0] * b[1]) / (a[1] + b[1]), a[1] + b[1], a[2] + b[2]]]
    return clusters


def pool_pieces(flows: list[dict], ids: list[str], run: dict, opts: dict | None = None) -> list[dict]:
    """Each outside inflow's pieces pooled by value (agglomerative, closest values first, while a
    merge costs less than log T in the row's noise), the flows refitted."""
    idx = {s: i for i, s in enumerate(ids)}
    fl, info = refit(flows, ids, run, opts)
    T = len(run["t"]) - 1
    sig2 = info["rss"] / max(1, T)
    out = []
    for f in fl:
        row = f.get("to") if f.get("to") in idx else f.get("from")
        if not f.get("inflow") or len(f["inflow"]) < 2 or row not in idx:
            out.append(f)
            continue
        vals = []
        for members in _pools(f):
            n = sum(max(1, f["inflow"][i]["to"] - f["inflow"][i]["from"] + 1) for i in members)
            vals.append((float(f["inflow"][members[0]]["value"]), n, list(members)))
        clusters = _cluster(vals, max(1e-12, float(sig2[idx[row]])), T)
        pool = {i: q for q, c in enumerate(clusters) for i in c[2]}
        out.append({**f, "inflow": [{**g, "pool": pool[i]} for i, g in enumerate(f["inflow"])]})
    return refit(out, ids, run, opts)[0]


def _viterbi(u: np.ndarray, levels: np.ndarray, change: float) -> np.ndarray:
    """The state (index into levels) of each step: least squares to its level plus `change` per switch."""
    n, m = len(u), len(levels)
    cost = (u[:, None] - levels[None, :]) ** 2
    acc = cost[0].copy()
    back = np.zeros((n, m), dtype=np.int32)
    for t in range(1, n):
        best = int(np.argmin(acc))
        stay = acc
        move = acc[best] + change
        back[t] = np.where(stay <= move, np.arange(m), best)
        acc = np.minimum(stay, move) + cost[t]
    path = np.zeros(n, dtype=np.int32)
    path[-1] = int(np.argmin(acc))
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def _segments(path: np.ndarray, tt: np.ndarray, levels: np.ndarray) -> list[dict]:
    segs, start, T = [], 0, len(path)
    for k in range(1, T + 1):
        if k == T or path[k] != path[start]:
            segs.append({"from": int(tt[start]), "to": int(tt[k - 1]), "value": float(levels[path[start]]), "pool": int(path[start])})
            start = k
    return segs


def _now(f: dict, tt: np.ndarray) -> np.ndarray:
    now = np.zeros(len(tt))
    for g in f["inflow"]:
        now[(tt >= g["from"]) & (tt <= g["to"])] = float(g["value"])
    return now


def _kmeans(x: np.ndarray, w: np.ndarray, L: int) -> np.ndarray:
    order = np.argsort(x)
    cw = np.cumsum(w[order]) / w.sum()
    c = np.array([x[order][min(len(x) - 1, int(np.searchsorted(cw, (q + 0.5) / L)))] for q in range(L)], dtype=float)
    for _ in range(20):
        lab = np.argmin(np.abs(x[:, None] - c[None, :]), axis=1)
        new = np.array([np.average(x[lab == q], weights=w[lab == q]) if np.any(lab == q) else c[q] for q in range(L)])
        if np.allclose(new, c):
            break
        c = new
    return np.unique(np.round(c, 12))


def resegment(flows: list[dict], ids: list[str], run: dict, opts: dict | None = None, rounds: int = 4, most: int = 4,
              change: float = 2.0, level_price: float | None = None) -> list[dict]:
    """Each changing outside inflow re-cut as a few pooled values: for 1..most values (started from
    the pieces' values), each step is assigned to a value (Viterbi; a change costs `change` x log T in
    the row's noise) and the model refitted, alternately for a few rounds; the count of values whose
    model prices best (price(): one number per value and per change, and level_price x log T more per
    value beyond the first: freely placed changes let an extra value fit the noise) is kept."""
    from orchestra.sdlab.discovery.strategies.kalman_lna._vendored.effect_hats.price import price

    opts = opts or {}
    lp = float(opts.get("level_price", 2.0) if level_price is None else level_price) * math.log(len(run["t"]) - 1)
    idx = {s: i for i, s in enumerate(ids)}
    tt = np.asarray(run["t"], dtype=float)[1:]
    T = len(tt)
    cur, _ = refit(flows, ids, run, opts)
    for j in range(len(cur)):
        f = cur[j]
        st = _stoich(f, idx, len(ids))
        rows = np.flatnonzero(st)
        if not f.get("inflow") or len(f["inflow"]) < 2 or len(rows) != 1:
            continue
        r, sign = int(rows[0]), float(st[rows[0]])
        vals = np.array([float(g["value"]) for g in f["inflow"]])
        lens = np.array([max(1, g["to"] - g["from"] + 1) for g in f["inflow"]], dtype=float)
        best = (price(cur, ids, run) + lp * (len(_pools(cur[j])) - 1), cur)
        for L in range(1, min(most, len(vals)) + 1):
            levels = _kmeans(vals, lens, L)
            trial = cur
            for _ in range(rounds):
                fl, info = refit(trial, ids, run, opts)
                g = fl[j]
                u = (_now(g, tt) if g.get("inflow") else np.full(T, levels[0])) + info["res"][r] / sign
                s2 = max(1e-12, float(info["rss"][r]) / T)
                path = _viterbi(u, levels, change * s2 * math.log(T))
                used = np.unique(path)
                path = np.searchsorted(used, path)
                trial = fl[:j] + [{**g, "inflow": _segments(path, tt, levels[used])}] + fl[j + 1:]
                fl, _ = refit(trial, ids, run, opts)
                pools = _pools(fl[j])
                new = np.array([float(fl[j]["inflow"][m[0]]["value"]) for m in pools])
                trial = fl
                if len(new) == len(levels) and np.allclose(new, levels, rtol=1e-6, atol=1e-9):
                    break
                levels = new
            p = price(trial, ids, run) + lp * (len(_pools(trial[j])) - 1)
            if p < best[0]:
                best = (p, trial)
        cur = best[1]
    return cur
