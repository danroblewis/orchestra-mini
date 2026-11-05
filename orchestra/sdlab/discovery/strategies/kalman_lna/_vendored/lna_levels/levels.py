# Vendored for kalman_lna from sdlab/discovery/strategies/lna_levels/levels.py at commit ec2f9e8 (imports rewritten to the
# vendored copies; otherwise unchanged). Do not edit: re-vendor instead.
"""Outside inflows as a few recurring levels (a regime model of the inflow), and a stock's growth
with itself offered back as inflow pieces.

lna cuts an outside inflow into pieces, one number each, by PELT and BIC. A square wave (two levels,
switching now and then, unannounced) then costs one number per piece, so on short or noisy runs a
self-growth term (outside -> s reading s) or a level of another stock is cheaper than the pieces,
and on long runs each piece is measured from its own steps only. Here a stock's pieces are grouped
into K levels (1-D k-means on the pieces' numbers, every K below the number of pieces), each level a
column that is 1 on its pieces' steps (a virtual stock read by a link into s), fitted by lna's
likelihood: K numbers instead of one per piece, each measured from all its steps. A grouping is kept
when its BIC beats the pieces'. And where s grows with itself, the model without the growth, with
an outside inflow into s cut afresh into pieces, is offered: kept when that inflow steps (three
pieces or more) and, regrouped, beats the growth's BIC.
"""

from __future__ import annotations


import math

import numpy as np

from orchestra.sdlab.discovery.strategies.kalman_lna._vendored.lna.atoms import Atom, features, stoich, variances
from orchestra.sdlab.discovery.strategies.kalman_lna._vendored.lna.changes import pieces
from orchestra.sdlab.discovery.strategies.kalman_lna._vendored.lna.fit import Data, Fitted, fit
from orchestra.sdlab.discovery.strategies.kalman_lna._vendored.lna.flows import _flow, r4
from orchestra.sdlab.discovery.strategies.kalman_lna._vendored.lna.run import bic, search

MAX_LEVELS = 4
TIE = 2.0            # BIC units: a model with more levels must beat one with fewer by more than this


def _aug(d: Data, rows: list[np.ndarray]) -> Data:
    if not rows:
        return d
    R = np.array(rows, dtype=float)
    free = None if d.free is None else np.concatenate([d.free, np.zeros(len(rows), bool)])
    return Data(np.vstack([d.prev, R]), np.vstack([d.dy, np.zeros_like(R)]), d.cfg, d.delta, free)


def _tau(cur: Fitted, n: int) -> np.ndarray | None:
    if cur.tau is None:
        return None
    return np.concatenate([cur.tau, np.zeros(max(0, n - len(cur.tau)))])


def kmeans_1d(values: np.ndarray, weights: np.ndarray, K: int) -> list[int]:
    """Optimal weighted 1-D k-means (dynamic programming over the sorted values): each value's group."""
    order = np.argsort(values)
    v, w = values[order], weights[order]
    n = len(v)
    cw, cv, cv2 = np.concatenate([[0], np.cumsum(w)]), np.concatenate([[0], np.cumsum(w * v)]), np.concatenate([[0], np.cumsum(w * v * v)])

    def cost(i, j):                          # values i..j-1
        W = cw[j] - cw[i]
        return 0.0 if W <= 0 else (cv2[j] - cv2[i]) - (cv[j] - cv[i]) ** 2 / W

    D = np.full((K + 1, n + 1), np.inf)
    B = np.zeros((K + 1, n + 1), int)
    D[0, 0] = 0.0
    for k in range(1, K + 1):
        for j in range(k, n + 1):
            for i in range(k - 1, j):
                c = D[k - 1, i] + cost(i, j)
                if c < D[k, j]:
                    D[k, j], B[k, j] = c, i
    labels = np.zeros(n, int)
    j = n
    for k in range(K, 0, -1):
        i = B[k, j]
        labels[i:j] = k - 1
        j = i
    out = np.zeros(n, int)
    out[order] = labels
    return list(out)


def _row_constant(cur: Fitted, d: Data, eps: float = 0.05) -> float:
    """What one virtual row adds to the negative log-likelihood: it has no flows and no change, so
    each step adds half the log of its variance (lna's floor eps plus the rounding delta)."""
    return 0.5 * d.dy.shape[1] * math.log(eps + d.delta)


def viterbi(u: np.ndarray, var: np.ndarray, levels: np.ndarray, penalty: float) -> np.ndarray:
    """Each step's level: the path of least Gaussian cost (u_t around the level, variance var_t plus
    the level itself, counting noise) plus `penalty` per switch."""
    T, K = len(u), len(levels)
    v = np.maximum(var[:, None] + np.maximum(levels[None, :], 0.0), 1e-6)
    cost = 0.5 * (u[:, None] - levels[None, :]) ** 2 / v + 0.5 * np.log(v)
    acc = cost[0].copy()
    back = np.zeros((T, K), int)
    for t in range(1, T):
        stay = acc
        move = acc.min() + penalty
        j = int(acc.argmin())
        back[t] = np.where(stay <= move, np.arange(K), j)
        acc = np.minimum(stay, move) + cost[t]
    out = np.zeros(T, int)
    out[-1] = int(acc.argmin())
    for t in range(T - 1, 0, -1):
        out[t - 1] = back[t, out[t]]
    return out


def switches(mask_rows: list[np.ndarray]) -> int:
    lab = np.argmax(np.array(mask_rows), axis=0)
    return int((np.diff(lab) != 0).sum())


def _leftover(cur: Fitted, d: Data, s: int, drop: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Stock s's change less what the atoms other than `drop` give it, and their counting variance."""
    S = d.dy.shape[0]
    keep = [i for i in range(len(cur.atoms)) if i not in drop]
    atoms = [cur.atoms[i] for i in keep]
    V, Z = stoich(atoms, S), features(atoms, d.prev, d.cfg)
    th = cur.theta[keep]
    u = d.dy[s] - (V @ (th[:, None] * Z))[s]
    var = variances(atoms, th, Z, S, eps=0.0)[s] + d.delta + (0.0 if cur.tau is None else float(cur.tau[s]))
    return u, var


def segmented(cur: Fitted, d: Data, s: int, idx: list[int], rows: list[np.ndarray], c1: float, max_levels: int = MAX_LEVELS, rounds: int = 4):
    """Stock s's inflow as K recurring levels whose switches are found afresh (Viterbi on what the
    other flows leave, a switch priced at log T), refitted with lna's likelihood, alternating.
    (score with switches priced, fitted, new rows) of the best K, or None."""
    S, T = d.dy.shape
    P0 = d.prev.shape[0]                     # level rows go after every row of prev (stocks, inputs, other columns)
    vals = np.array([cur.theta[i] for i in idx])
    lens = np.array([b - a for a, b in (cur.atoms[i].span for i in idx)], dtype=float)
    u, var = _leftover(cur, _aug(d, rows), s, idx)
    rest = [i for i in range(len(cur.atoms)) if i not in idx]
    best = None
    for K in range(2, min(max_levels, len(idx)) + 1):
        lab = np.array(kmeans_1d(vals, lens, K))
        L = np.array([float(np.average(vals[lab == k], weights=lens[lab == k])) for k in range(K)])
        g, path = None, None
        for _ in range(rounds):
            new = viterbi(u, var, L, math.log(T))
            if path is not None and np.array_equal(new, path):
                break
            path = new
            used = [k for k in range(K) if (path == k).any()]
            new_rows = [(path == k).astype(float) for k in used]
            atoms = [cur.atoms[i] for i in rest] + [Atom("link", dst=s, driver=P0 + len(rows) + k) for k in range(len(used))]
            dg = _aug(d, rows + new_rows)
            g = fit(atoms, dg, np.concatenate([cur.theta[rest], np.maximum(L[used], 1e-3)]), tau=_tau(cur, S + len(rows) + len(used)))
            L = np.array([g.theta[len(rest) + used.index(k)] if k in used else L[k] for k in range(K)])
            # what the other flows leave, on the refitted model (the level atoms dropped)
            gl = Fitted(g.atoms, g.theta, g.nll, None if g.tau is None else g.tau)
            u, var = _leftover(gl, dg, s, list(range(len(rest), len(g.atoms))))
        if g is None:
            continue
        score = bic(g, dg) - 2 * c1 * (len(rows) + len(new_rows)) + math.log(T) * switches(new_rows)
        if best is None or score < best[0] - TIE:       # K rises: more levels only for a clear gain
            best = (score, g, new_rows)
    return best


def regroup(cur: Fitted, d: Data, max_levels: int = MAX_LEVELS) -> tuple[Fitted, dict, float]:
    """(the model without the level atoms, {stock: [(steps mask, number)]}, its BIC as if on the real
    rows, with every cut or switch priced at log T). Stocks are taken in turn; a grouping (the
    pieces' own cuts, or switches found afresh) stays when it lowers that BIC."""
    S, T = d.dy.shape
    P0 = d.prev.shape[0]                     # level rows go after every row of prev (stocks, inputs, other columns)
    c1 = None
    rows: list[np.ndarray] = []
    owner: list[int] = []                    # rows[k] belongs to stock owner[k]
    cuts = sum(max(0, sum(1 for a in cur.atoms if a.kind == "in" and a.dst == s_ and a.span is not None) - 1) for s_ in range(S))
    best_bic = bic(cur, d) + math.log(T) * cuts
    for s in range(S):
        idx = [i for i, a in enumerate(cur.atoms) if a.kind == "in" and a.dst == s and a.span is not None and a.era is None and cur.theta[i] > 0]
        if len(idx) < 3:
            continue
        if c1 is None:
            c1 = _row_constant(cur, d)
        vals = np.array([cur.theta[i] for i in idx])
        spans = [cur.atoms[i].span for i in idx]
        lens = np.array([b - a for a, b in spans], dtype=float)
        rest = [i for i in range(len(cur.atoms)) if i not in idx]
        options = []                         # (score, fitted, rows) for each grouping tried
        for K in range(1, min(max_levels, len(idx) - 1) + 1):
            lab = kmeans_1d(vals, lens, K)
            new_rows = []
            for k in range(K):
                m = np.zeros(T)
                for (a, b), g in zip(spans, lab):
                    if g == k:
                        m[a:b] = 1.0
                new_rows.append(m)
            atoms = [cur.atoms[i] for i in rest] + [Atom("link", dst=s, driver=P0 + len(rows) + k) for k in range(K)]
            start = np.concatenate([cur.theta[rest], [float(np.average(vals[np.array(lab) == k], weights=lens[np.array(lab) == k])) for k in range(K)]])
            dg = _aug(d, rows + new_rows)
            g = fit(atoms, dg, start, tau=_tau(cur, S + len(rows) + K))
            options.append((bic(g, dg) - 2 * c1 * (len(rows) + K) + math.log(T) * (cuts - (len(idx) - 1) + switches(new_rows)), g, new_rows))
        seg = segmented(cur, d, s, idx, rows, c1, max_levels)
        if seg is not None:
            options.append((seg[0] + math.log(T) * (cuts - (len(idx) - 1)), seg[1], seg[2]))
        # the fewest levels among the groupings within TIE of the best, if that beats the pieces
        options = [o for o in options if o[0] < best_bic]
        choice = None
        if options:
            top = min(o[0] for o in options)
            choice = min((o for o in options if o[0] <= top + TIE), key=lambda o: (len(o[2]), o[0]))
        if choice is not None:
            best_bic, cur, new_rows = choice
            cuts += switches(new_rows) - (len(idx) - 1)
            rows += new_rows
            owner += [s] * len(new_rows)
    if not rows:
        return cur, {}, best_bic
    keep = [i for i, a in enumerate(cur.atoms) if a.driver is None or a.driver < P0]
    levels: dict[int, list] = {}
    for i, a in enumerate(cur.atoms):
        if a.driver is not None and a.driver >= P0 and cur.theta[i] > 0:
            levels.setdefault(a.dst, []).append((rows[a.driver - P0] > 0, float(cur.theta[i])))
    tau = None if cur.tau is None else cur.tau[:S]
    return Fitted([cur.atoms[i] for i in keep], cur.theta[keep], cur.nll, tau), levels, best_bic


def without_growth(cur: Fitted, d: Data) -> Fitted:
    """Where a stock grows with itself (outside -> s reading s), the model without the growth, with
    an outside inflow into s (added if it had none) cut afresh into pieces: kept when that inflow
    steps (3 pieces or more) and, both regrouped into levels, it has the lower BIC by TIE."""
    grow = [i for i, a in enumerate(cur.atoms) if a.kind == "grow" and a.era is None and cur.theta[i] > 0]
    if not grow:
        return cur
    rest = [i for i in range(len(cur.atoms)) if i not in grow]
    atoms, theta = [cur.atoms[i] for i in rest], list(cur.theta[rest])
    for i in grow:
        s = cur.atoms[i].dst
        if not any(a.kind == "in" and a.dst == s for a in atoms):
            atoms.append(Atom("in", dst=s))
            theta.append(max(1e-3, float(cur.theta[i] * np.maximum(d.prev[s], 0).mean())))
    g = fit(atoms, d, np.array(theta), tau=cur.tau)
    g = search(pieces(g, d, mult=0.5), d)
    # only an inflow that steps back and forth (3 pieces or more) stands in for growth: a single
    # change of level is what growth towards an equilibrium looks like (births and deaths)
    stepping = all(sum(1 for a in g.atoms if a.kind == "in" and a.dst == cur.atoms[i].dst and a.span is not None) >= 3 for i in grow)
    if stepping and regroup(g, d)[2] < regroup(cur, d)[2] - TIE:
        return g
    return cur


def level_flows(flows: list[dict], levels: dict, ids: list[str], at) -> list[dict]:
    """Each regrouped stock's levels as one outside inflow in pieces (consecutive steps at one level)."""
    out = list(flows)
    for s, items in levels.items():
        T = len(items[0][0])
        value = np.zeros(T)
        for m, th in items:
            value[m] += th
        cuts = [0] + [k for k in range(1, T) if value[k] != value[k - 1]] + [T]
        pcs = [{"from": at(a), "to": at(b - 1), "value": r4(value[a])} for a, b in zip(cuts[:-1], cuts[1:])]
        mean = float(value.mean())
        out.append(_flow(None, ids[s], {"form": "constant", "value": r4(mean)}, mean, inflow=pcs, changes=len(pcs) - 1, levels=len(items)))
    return out


