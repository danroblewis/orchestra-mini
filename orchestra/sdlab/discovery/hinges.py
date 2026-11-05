"""Floors: flows that are a straight line in a stock's level only above (or below) a threshold.

SD Lab's rates are floored at 0, so a flow `max(0, c + k x)` with c < 0 runs only once x passes
-c/k (escalations once the queue passes 300, sales once the bales pass 200), and a gap `max(0, (goal
- x) / time)` only while x is below its goal. Through the steps where the floor holds the flow is 0,
and a straight line fitted through them bends: a weaker slope, a constant that belongs to the flow
lands in the stock's outside inflow, and the found model runs the flow where the true one is shut.

A hinge column is `max(0, sign x (x - theta))`: one per stock at most, its threshold and side chosen
from the data before the rows are fitted (`choose`), and then offered to every row beside the levels
(matrix.identify_one, opts["extra_columns"]). A row that takes it has a flow with a floor; a move with
a floor shows the same column in the source's row (-k) and the receiver's (+k), so the kink belongs
to that flow and not to the row's other flows. `read_flows_hinged` reads a hinge's numbers as a flow
`linear {c: -k sign theta, terms: [{x, k sign}]}`, which the model floors at 0 exactly as the truth's.
"""

from __future__ import annotations

import math

import numpy as np

from .flows import read_flows
from .regression import demean

QUANTILES = np.arange(0.05, 0.951, 0.05)


def column(x: np.ndarray, theta: float, sign: int) -> np.ndarray:
    return np.maximum(0.0, sign * (x - theta))


def gains(y: np.ndarray, Z: np.ndarray, H: np.ndarray, cuts: list[int]) -> np.ndarray:
    """For each candidate column (rows of H): how much it improves y's fit on the levels Z (and a
    mean per piece of `cuts`), in BIC terms, T log(rss / rss with it)."""
    T = y.shape[0]
    yc, Zc, Hc = demean(y, cuts), demean(Z, cuts), demean(H, cuts)
    G = Zc @ Zc.T
    G[np.diag_indices_from(G)] += 1e-9 * (np.abs(np.diag(G)) + 1)
    P = np.linalg.solve(G, Zc)                     # S x T: Z's projection weights
    ry = yc - (Zc @ yc) @ P
    rH = Hc - (Hc @ Zc.T) @ P
    rss = float(ry @ ry)
    num = (rH @ ry) ** 2
    den = np.einsum("ij,ij->i", rH, rH)
    drop = np.where(den > 1e-9 * (np.abs(Hc).sum(axis=1) + 1), num / np.maximum(den, 1e-300), 0.0)
    return T * np.log(max(rss, 1e-12) / np.maximum(rss - np.minimum(drop, rss * (1 - 1e-12)), 1e-12))


def choose(stocks: list[dict], prev: np.ndarray, dy: np.ndarray, cuts: list[list[int] | None], min_score: float | None = None,
           above: np.ndarray | None = None, greedy: bool = True, same_unit: bool = True) -> list[dict]:
    """At most one hinge per stock: the threshold (a quantile of a stock's level) whose column
    improves the rows that would take it most, each row paying log T for its number; kept while the
    total is worth more than 2 log T (the threshold is a number too). `greedy`: one at a time, each
    scored beside the hinges already taken (so a receiver's row, kinked only because its inflow is a
    hinge on the source, does not earn a hinge of its own); else every stock's best on its own. The
    two sides max(0, x - theta) and max(0, theta - x) differ by x - theta, which the level and the
    pieces' means already span, so they fit alike and only one is offered. `same_unit`: a floor
    belongs to a flow, and a flow moves things of one unit, so a hinge is scored on, and offered to,
    the rows of stocks in its base's unit. With `above` (per stock), only thresholds above it."""
    S, T = prev.shape
    log_t = math.log(T)
    min_score = 2 * log_t if min_score is None else min_score
    sign = 1
    cand = []
    for a in range(S):
        x = prev[a]
        for th in np.unique(np.quantile(x, QUANTILES)):
            if above is not None and th <= above[a]:
                continue
            h = column(x, float(th), sign)
            if 0.05 <= (h > 0).mean() <= 0.95:
                cand.append((a, float(th), h))
    unit = [s.get("unit") or "" for s in stocks]
    out = []

    def scores(cand):
        H = np.array([c[2] for c in cand])
        base_unit = np.array([unit[c[0]] for c in cand])
        Z = np.vstack([prev] + [h["x"][None, :] for h in out])
        score = np.zeros(len(cand))
        for s in range(S):
            g = np.maximum(0.0, gains(dy[s], Z, H, cuts[s] or [0, T]) - log_t)
            score += np.where(base_unit == unit[s], g, 0.0) if same_unit else g
        return score

    def side(a, th):
        # the side whose hinge best stands in for the stock's own level: a floored flow is 0 on one
        # side of its threshold and needs no straight-line part beside it (the sides fit alike only
        # with the level kept, and read as different flows, each floored on its own)
        Z = np.vstack([np.delete(prev, a, axis=0)] + [h["x"][None, :] for h in out])
        H = np.array([column(prev[a], th, 1), column(prev[a], th, -1)])
        g = np.zeros(2)
        for s in range(S):
            if not same_unit or unit[s] == unit[a]:
                g += gains(dy[s], Z, H, cuts[s] or [0, T])
        return 1 if g[0] >= g[1] else -1

    def take(c, score):
        a, th, _ = c
        sg = side(a, th)
        rows = [x["id"] for x in stocks if (x.get("unit") or "") == unit[a]] if same_unit else None
        out.append({"id": f"{stocks[a]['id']}{'>' if sg > 0 else '<'}{th:g}", "base": stocks[a]["id"], "theta": th, "sign": sg,
                    "x": column(prev[a], th, sg), "score": float(score), "rows": rows})

    if not greedy and cand:
        score = scores(cand)
        for a in range(S):
            idx = [i for i, c in enumerate(cand) if c[0] == a]
            i = max(idx, key=lambda i: score[i]) if idx else None
            if i is not None and score[i] > min_score:
                take(cand[i], score[i])
        return out
    while cand:
        score = scores(cand)
        i = int(np.argmax(score))
        if score[i] <= min_score:
            break
        take(cand[i], score[i])
        cand = [c for c in cand if c[0] != cand[i][0]]
    return out


def _rate(rate: dict, hinge: dict[str, dict]) -> dict:
    """A rate on hinge columns -> the same rate on the stocks' levels, with the threshold as its
    constant (floored at 0 by the model, so it is the hinge itself)."""
    form = rate.get("form")
    if form == "proportional" and rate.get("stock") in hinge:
        e = hinge[rate["stock"]]
        k = float(rate["k"])
        return {"form": "linear", "c": -k * e["sign"] * e["theta"], "terms": [{"stock": e["base"], "k": k * e["sign"]}]}
    if form == "linear" and any(t.get("stock") in hinge for t in rate.get("terms") or []):
        c, terms = float(rate.get("c") or 0), []
        for t in rate["terms"]:
            e = hinge.get(t.get("stock"))
            if e is None:
                terms.append(t)
            else:
                c -= float(t["k"]) * e["sign"] * e["theta"]
                terms.append({"stock": e["base"], "k": float(t["k"]) * e["sign"]})
        return {"form": "linear", "c": c, "terms": terms}
    return rate


def read_flows_hinged(stocks: list[dict], rows: list[dict], prev: np.ndarray, tol: float, extras: list[dict]) -> list[dict]:
    """flows.read_flows, with each hinge column read as a stock of its own (its base's unit and
    group: a drain on it in its base's row, gains on it in other rows), then put back on its base."""
    hinge = {e["id"]: e for e in extras}
    by_id = {s["id"]: s for s in stocks}
    pseudo = [{"id": e["id"], "unit": by_id[e["base"]].get("unit"), "group": by_id[e["base"]].get("group")} for e in extras]
    own = {e["id"]: 0.0 for e in extras}
    rows2 = []
    for r in rows:
        coef = dict(r["coef"])
        for e in extras:
            if e["base"] == r["id"] and e["id"] in coef:
                own[e["id"]] = coef.pop(e["id"])
        rows2.append({**r, "coef": coef})
    rows2 += [{"id": e["id"], "coef": {e["id"]: own[e["id"]]} if own[e["id"]] else {}, "inflow": []} for e in extras]
    prev2 = np.vstack([prev] + [np.asarray(e["x"], dtype=float)[None, :] for e in extras])
    out = []
    for f in read_flows(stocks + pseudo, rows2, prev2, tol):
        g = dict(f)
        for key in ("from", "to", "driver"):
            if g.get(key) in hinge:
                g[key] = hinge[g[key]]["base"]
        g["rate"] = _rate(f["rate"], hinge)
        g["rates"] = [g["rate"]]
        on = next((f.get(k) for k in ("from", "to", "driver") if f.get(k) in hinge), None) or _stock_of(f["rate"])
        if on in hinge:
            g["floor"] = on
        out.append(g)
    return out


def _stock_of(rate: dict) -> str | None:
    if rate.get("form") == "proportional":
        return rate.get("stock")
    if rate.get("form") == "linear":
        return next((t.get("stock") for t in rate.get("terms") or []), None)
    return None
