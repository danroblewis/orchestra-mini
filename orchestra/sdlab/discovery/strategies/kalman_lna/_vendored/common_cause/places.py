# Vendored for kalman_lna from sdlab/discovery/strategies/common_cause/places.py at commit ec2f9e8 (imports rewritten to the
# vendored copies; otherwise unchanged). Do not edit: re-vendor instead.
"""Shared change places: a common cause steps several outside inflows at the same moments.

A stock whose outside inflow steps when another's does can be fitted, nearly as well and far more
cheaply, by reading the other stock's level (it follows its own inflow within a few steps): one
number in place of a value and a place for every change. Priced row by row, BIC takes the
cross-link. But a change place already earned by another row's pieces is known: this row pays only
for its values there. So each row is offered one more candidate: pieces at the places the other
rows earned (each a value, 1 parameter), with its levels chosen afresh by backward elimination from
every level it may read. A cross-link survives only if it explains something those shared pieces do
not; a row that does not change with the others pays a value per place for nothing and keeps its
own form.
"""

from __future__ import annotations

import numpy as np

from orchestra.sdlab.discovery.regression import fit_subset, row_sums
from orchestra.sdlab.discovery.strategies.kalman_lna._vendored.common_cause.rows import _bic, backward, columns, inflow_params, white


def noise_var(r: np.ndarray) -> float:
    """A series' noise variance from its successive differences (changes do not inflate it)."""
    n = r.shape[0]
    return float(np.sum(np.diff(r) ** 2)) / (2 * max(1, n - 1))


def joint_pelt(R: np.ndarray, min_piece: int, pen: float, var_floor: float = 1e-300) -> list[int]:
    """Cuts 0 < ... < n shared by the series R (rows x n): each segment costs the sum over rows of
    its squared deviations from the row's segment mean, over the row's noise variance; each change
    costs pen. Optimal partitioning (O(n^2), vectorised over the last cut)."""
    R = np.nan_to_num(np.asarray(R, dtype=float))
    k, n = R.shape
    v = np.array([max(noise_var(r), var_floor) for r in R])
    Z = R / np.sqrt(v)[:, None]
    S1 = np.concatenate([np.zeros((k, 1)), np.cumsum(Z, axis=1)], axis=1)
    S2 = np.concatenate([[0.0], np.cumsum((Z * Z).sum(0))])
    f = np.full(n + 1, np.inf)
    f[0] = -pen
    last = np.zeros(n + 1, dtype=int)
    for t in range(min_piece, n + 1):
        hi = t - min_piece if t < n else t - 1
        s = np.arange(0, hi + 1)
        s = s[np.isfinite(f[s])]
        if not len(s):
            s = np.array([0])
        d = (S1[:, t][:, None] - S1[:, s]) ** 2
        cost = (S2[t] - S2[s]) - d.sum(0) / (t - s)
        tot = f[s] + cost + pen
        i = int(np.argmin(tot))
        f[t], last[t] = tot[i], s[i]
    cuts, t = [], n
    while t > 0:
        cuts.append(t)
        t = int(last[t])
    cuts.append(0)
    return cuts[::-1]


def prune(y: np.ndarray, X: np.ndarray, cols: list[int], cuts: list[int], extra) -> tuple[list[int], float]:
    """Places dropped one at a time while BIC improves (a value per place)."""
    T = y.shape[0]

    def score(c):
        return _bic(T, fit_subset(row_sums(y, X, c), cols).rss, len(cols) + extra(c))

    cur = score(cuts)
    while len(cuts) > 2:
        best = None
        for i in range(1, len(cuts) - 1):
            c = cuts[:i] + cuts[i + 1:]
            s = score(c)
            if s < cur - 1e-9 and (best is None or s < best[1]):
                best = (c, s)
        if best is None:
            break
        cuts, cur = best
    return cuts, cur


def _allowed(s: int, ids: list[str], stocks: list[dict]) -> list[int]:
    group = {x["id"]: x.get("group") for x in stocks}
    final = {x["id"] for x in stocks if x.get("final")}
    return [j for j in range(len(ids)) if not (ids[s] in final and j == s)
            and (j == s or not group.get(ids[j]) or not group.get(ids[s]) or group[ids[j]] == group[ids[s]])]


def _shared_price(c):
    return 0 if not c else 1 + (len(c) - 2)


def candidate(row: dict, s: int, cuts: list[int], ids, stocks, prev, dy, cfg, L) -> dict | None:
    """Row s with pieces at the given places (a value each), its levels chosen afresh by backward
    elimination from every level it may read, places it does not need dropped (and, with none
    left, the constant too if it does not pay)."""
    y = dy[s]
    X, col = columns(s, ids, prev, cfg, L, _allowed(s, ids, stocks))
    if not len(col):
        return None
    chosen, _, _ = backward(y, X, cuts, list(range(len(col))), _shared_price(cuts))
    cuts, _ = prune(y, X, chosen, cuts, _shared_price)
    chosen, rss, b = backward(y, X, cuts, chosen, _shared_price(cuts))
    if len(cuts) <= 2:
        c2, r2, b2 = backward(y, X, None, chosen, 0)
        if b2 < b:
            chosen, rss, b, cuts = c2, r2, b2, None
    return {**row, "X": X, "col": col, "set": chosen, "cuts": cuts, "rss": rss,
            "p": len(chosen) + (len(cuts) - 1 if cuts else 0), "shared": True, "bic": b}


def own_changes(s: int, y: np.ndarray, prev: np.ndarray, pelt, min_piece: int) -> tuple[np.ndarray, bool]:
    """What a row's own level leaves beside pieces of an outside inflow (found as the matrix does:
    fine pieces first, so the level, which follows its inflow, does not absorb the changes), and
    whether those pieces pay at the textbook price (a value and a place each)."""
    T = y.shape[0]
    X = prev[s][None, :]
    cuts = [0, T]
    for p in range(4):
        f = fit_subset(row_sums(y, X, cuts), [0])
        left = y - f.beta @ X
        nxt = pelt(left, min_piece, 0.5 if p == 0 else 1)
        if nxt == cuts:
            break
        cuts = nxt
    f = fit_subset(row_sums(y, X, cuts), [0])
    left = y - f.beta @ X
    cuts2 = pelt(left, min_piece, 2)
    if len(cuts2) <= 2:
        return left, False
    pieces = _bic(T, fit_subset(row_sums(y, X, cuts2), [0]).rss, 1 + inflow_params(cuts2, 2))
    const = _bic(T, fit_subset(row_sums(y, X, [0, T]), [0]).rss, 2)
    return left, pieces < const


def honest(row: dict, y: np.ndarray, pelt, min_piece: int, per_change: float) -> dict | None:
    """A changing row with its pieces found again beside its own levels at the textbook price
    (fine pieces first, then a value and a place each): the matrix's cheap pieces over-cut and
    co-adapt with the drain. None for a row without pieces."""
    if not changing(row):
        return None
    X, cols = row["X"], row["set"]
    cuts = [0, y.shape[0]]
    for p in range(4):
        f = fit_subset(row_sums(y, X, cuts), cols)
        nxt = pelt(y - (f.beta @ X[cols] if cols else 0.0), min_piece, 0.5 if p == 0 else 1)
        if nxt == cuts:
            break
        cuts = nxt
    f = fit_subset(row_sums(y, X, cuts), cols)
    cuts = pelt(y - (f.beta @ X[cols] if cols else 0.0), min_piece, 2)
    if len(cuts) <= 2:
        return None
    f = fit_subset(row_sums(y, X, cuts), cols)
    b = _bic(y.shape[0], f.rss, len(cols) + inflow_params(cuts, per_change))
    return {**row, "cuts": cuts, "rss": f.rss, "p": len(cols) + len(cuts) - 1, "bic": b}


def own_bic(row: dict, T: int, per_change: float) -> float:
    return _bic(T, row["rss"], len(row["set"]) + inflow_params(row["cuts"], per_change))


def changing(row: dict) -> bool:
    return bool(row.get("cuts")) and len(row["cuts"]) > 2


def accept(c: dict | None, row: dict, T: int, per_change: float, alpha: float) -> bool:
    """The candidate replaces the row when its BIC is lower; a row with pieces is replaced by one
    without only when that one's residuals are white (as rows.reselect)."""
    if c is None or c["bic"] >= own_bic(row, T, per_change) - 1e-9:
        return False
    if changing(row) and not changing(c):
        e = _resid(c)
        return white(e - (e.mean() if c["cuts"] else 0.0), alpha)
    return True


def _resid(r: dict) -> np.ndarray:
    return r["_y"] - (fit_subset(row_sums(r["_y"], r["X"], r["cuts"]), r["set"]).beta @ r["X"][r["set"]] if r["set"] else 0.0)


def _left(rows: list[dict], W: list[int], dy: np.ndarray) -> dict:
    """What each row in W leaves of its change beside its own levels, with the other rows' of W
    (suspected stand-ins for shared changes) left out."""
    res = {}
    for s in W:
        r = rows[s]
        cols = [q for q in r["set"] if r["col"][q][0] not in W or r["col"][q][0] == s]
        f = fit_subset(row_sums(dy[s], r["X"], r["cuts"]), cols)
        res[s] = dy[s] - (f.beta @ r["X"][cols] if cols else 0.0)
    return res


def offer(rows: list[dict], ids: list[str], stocks: list[dict], prev: np.ndarray, dy: np.ndarray, cfg: np.ndarray, L: int,
          pelt, min_piece: int = 5, per_change: float = 2, trace: bool = False, rounds: int = 3, alpha: float = 0.01) -> list[dict]:
    """The rows that show changes of their own (in what their own level leaves, at the textbook
    price) are the witnesses. Each row is offered pieces at the places the *other* witnesses show
    jointly (a place paid once by them, a value per witness), at a value each: a place this row
    did not pay for. While rows take them, the witnesses are the rows that took shared pieces, their
    residuals with other witnesses' levels (suspected stand-ins for the changes) left out."""
    T = dy.shape[1]
    log_t = float(np.log(T))
    out = list(rows)
    for s, row in enumerate(rows):
        h = honest(row, dy[s], pelt, min_piece, per_change)
        if h is not None and h["bic"] < own_bic(row, dy.shape[1], per_change) - 1e-9:
            out[s] = rows[s] = h
    rows = list(out)
    # the witnesses: rows whose own fit has pieces (what its levels leave, other witnesses' levels
    # left out); with none, every row that shows changes beside its own level alone
    res = _left(out, [s for s, r in enumerate(out) if changing(r)], dy)
    if not res:
        own = [own_changes(s, dy[s], prev, pelt, min_piece) for s in range(len(rows))]
        res = {s: own[s][0] for s in range(len(rows)) if own[s][1]}
    if len(res) < 2 and not any(changing(r) for r in rows):
        return rows
    for rnd in range(rounds):
        K = sorted(res)
        found: dict = {}

        def places_of(others: tuple) -> list[int]:
            if others not in found:
                found[others] = joint_pelt(np.array([res[k] for k in others]), min_piece, (1 + len(others)) * log_t)
            return found[others]

        changed = False
        for s, row in enumerate(rows):
            others = tuple(k for k in K if k != s)
            if not others:
                continue
            cuts = places_of(others)
            if len(cuts) <= 2:
                continue
            c = candidate(row, s, cuts, ids, stocks, prev, dy, cfg, L)
            if c is not None:
                c["_y"] = dy[s]
            ok = accept(c, row, T, per_change, alpha)
            if trace and c:
                print(f"  round {rnd} {ids[s]}: {len(cuts) - 2} places from {[ids[k] for k in others]} -> {len(c['cuts'] or [0, 0]) - 2}, {[ids[c['col'][q][0]] for q in c['set']]}: {c['bic']:.1f} vs own {own_bic(row, T, per_change):.1f} {[ids[row['col'][q][0]] for q in row['set']]}{' taken' if ok else ''}")
            new = c if ok else row
            if new.get("set") != out[s].get("set") or new.get("cuts") != out[s].get("cuts"):
                changed = True
            out[s] = new
        W = [s for s, r in enumerate(out) if changing(r)]
        if not changed or len(W) < 2:
            break
        res = _left(out, W, dy)
    for r in out:
        r.pop("_y", None)
    return out
