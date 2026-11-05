"""A capacity limit: a flow out of X at MIN(a x X, b x Y) (docs/research/k8s-306-s1.md, 2026-10-21).

A rate that is the lesser of a demand on X and a capacity read off Y (another measured stock that is
not a counter, or a measured input) is not linear in the levels, so a linear row fit replaces it by a
proxy and the rest of the model compensates. Here it is offered to the strategy's own fit as one more
measured series, the regime-clipped source `m = MIN(X, theta x Y)`, so `a x m` is an ordinary linear
term that the strategy fits, prices and keeps or drops jointly with the rest of its model
(theta = b / a, the regime threshold). Afterwards every term reading m is written back as
`MIN(a x X, a theta x Y)` (an expression flow), so the answer reads only measured stocks and inputs.

`offer` finds (X, Y, theta) from the data alone. For each non-final measured stock X, its one-step
changes are regressed on a constant, every measured level and every input, with and without m; theta
is chosen over quantiles of X / Y and refined by golden section. The best pair is then refined on X's
row together with the other row m explains best (where the flow goes), by the profile likelihood of
both rows. m is offered only where it wins decisively on X's row: Gaussian BIC better by more than
PRICE x log T, with each regime holding on at least MIN_SHARE of the steps (a limit that never binds,
or always does, is a linear rate). The same rule for every case; nothing reads the case or its truth.

(k8s-312, kalman_lna) This is kalman_lna's own variant of capmin.py (copied from its capmin.py at 201a23d, which
stays identical to the other strategies' copies, tests/test_capmin.py): a regime must hold on MIN_SHARE of the steps
or on MIN_STEPS of them, whichever is fewer. Everything else is capmin.py's.
"""

from __future__ import annotations

import copy
import math

import numpy as np

PRICE = 6.0            # the extra numbers' price (theta and the arm), in log T, plus a margin
MIN_SHARE = 0.05       # each regime holds on at least this share of the steps ... (before 312)
MIN_STEPS = 5          # (k8s-312, kalman_lna) ... or on at least this many steps, whichever is fewer: a limit that binds on
                       # 7 of 299 steps (312 none-short: queue -> inflight at min(0.6 queue, 40 ready)) is a kink the
                       # data show exactly, and the kink test (beyond every product of two levels) still has to pass
GRID = 24              # theta candidates (quantiles of X / Y)
NAME = "capmin__"      # the offered series' name prefix
RESOLUTION = 1e-8      # a residual variance below this share of the mean squared change is rounding


def level_series(name) -> bool:
    """Is this input the offered series? It is a function of measured levels, not something from
    outside, so a row reads it wherever it may read the levels (a declared boundary's rule for
    measured inputs does not apply to it)."""
    return str(name).startswith(NAME)


def offered(opts: dict | None) -> dict:
    """{series name: (X, its receiver or None)} for the series offer() added to opts."""
    return {k: (v["source"], v.get("to")) for k, v in ((opts or {}).get("capmin") or {}).items()}


def readers(name, opts: dict | None) -> tuple | None:
    """The stocks whose rows may read the offered series `name`: the flow it limits leaves X and enters
    its receiver (only X when it leaves for outside); None when `name` is not an offered series."""
    got = offered(opts).get(name)
    return None if got is None else tuple(x for x in got if x is not None)


def excluded(stock: str, names: list[str], inputs, inputs_ok: bool, opts: dict | None) -> list[int]:
    """The input columns (indices into names) a row of `stock` may not read: a measured input where the
    boundary has no measured side for it (not inputs_ok), and an offered series outside the rows of
    the flow it limits."""
    out = []
    for j in inputs:
        r = readers(names[j], opts)
        if (r is None and not inputs_ok) or (r is not None and stock not in r):
            out.append(j)
    return out


def _steps(stocks: list[dict], run: dict, opts: dict):
    """Each measured stock's one-step change less announced injections, the levels its rates read
    (injections added), and each measured input at the step's end (the engine's convention)."""
    ids = [s["id"] for s in stocks]
    final = {s["id"] for s in stocks if s.get("final")}
    t = np.asarray(run["t"], dtype=float)
    L = np.nan_to_num(np.array([np.asarray(run["levels"].get(i, []), dtype=float) for i in ids]))
    if L.ndim != 2 or L.shape[1] < 20:
        return None
    dy = L[:, 1:] - L[:, :-1]
    read = L[:, :-1].copy()
    at = {float(x): k for k, x in enumerate(t[1:])}
    index = {s: i for i, s in enumerate(ids)}
    for j in opts.get("injections") or []:
        k, i = at.get(float(j["t"])), index.get(j.get("stock"))
        if k is not None and i is not None:
            dy[i, k] -= float(j["amount"])
            read[i, k] = max(0.0, read[i, k] + float(j["amount"]))
    T = dy.shape[1]
    inputs = {}
    for name, v in (opts.get("inputs") or {}).items():
        if name in index:
            continue
        u = np.nan_to_num(np.asarray(v, dtype=float))
        if len(u) >= T + 1:
            inputs[name] = u[1:T + 1]
    return ids, final, dy, read, inputs


def _rss(B: np.ndarray, y: np.ndarray) -> float:
    coef, *_ = np.linalg.lstsq(B, y, rcond=None)
    r = y - B @ coef
    return float(r @ r)


def _golden(f, grid: np.ndarray) -> tuple[float, float]:
    """The minimum of f over grid, refined by golden section between the best point's neighbours."""
    vals = [f(th) for th in grid]
    j = int(np.argmin(vals))
    lo, hi = grid[max(0, j - 1)], grid[min(len(grid) - 1, j + 1)]
    th, best = float(grid[j]), vals[j]
    for _ in range(40):
        if hi - lo <= 1e-9 * max(1.0, abs(hi)):
            break
        a, b = hi - 0.618 * (hi - lo), lo + 0.618 * (hi - lo)
        fa, fb = f(a), f(b)
        if fa < fb:
            hi = b
            if fa < best:
                th, best = float(a), fa
        else:
            lo = a
            if fb < best:
                th, best = float(b), fb
    return th, best


def _design(ids, read, inputs, opts, t, T):
    """The linear rows' columns (a constant, every measured level, every input; per announced era when
    there are eras), and the same with every product of two of the levels and inputs (a smooth
    second-order surface: what a MIN has to beat to be a kink rather than a curve)."""
    cols = [np.ones(T)] + [read[i] for i in range(len(ids))] + list(inputs.values())
    starts = sorted(float(e["start"]) for e in (opts.get("eras") or []) if e and e.get("start") is not None)
    if len(starts) > 1:
        era = np.searchsorted(np.array(starts), t[1:T + 1], side="right") - 1
        cols = [c * (era == e) for e in sorted(set(era.tolist())) for c in cols]
    vals = [read[i] for i in range(len(ids))] + list(inputs.values())
    prods = [vals[i] * vals[j] for i in range(len(vals)) for j in range(i, len(vals))]
    B = np.column_stack(cols)
    Q = np.column_stack(cols + prods)
    return B, Q


def _gain(B: np.ndarray, m: np.ndarray, y: np.ndarray, floor: float) -> tuple[float, float]:
    """(T log of how much m lowers the row's residual on top of B, m's coefficient)."""
    coef, *_ = np.linalg.lstsq(np.column_stack([B, m]), y, rcond=None)
    r1 = y - np.column_stack([B, m]) @ coef
    return len(y) * math.log((_rss(B, y) + floor) / (float(r1 @ r1) + floor)), float(coef[-1])


def offer(stocks: list[dict], run: dict, opts: dict | None) -> tuple[dict, dict | None]:
    """opts with the regime-clipped series added as a measured input where a capacity limit wins, and
    what restore() needs (None where none is offered).

    For each non-final stock X and capacity Y: theta on X's row (a grid of X / Y quantiles, then golden
    section); m must take from X (a negative coefficient in X's row). Its receiver is the row where m
    enters with a positive coefficient and lowers the residual most (none: an outflow to outside), and
    theta is refined on X's row and the receiver's together. The evidence is the kink: what m adds to
    each of those rows on top of every product of two levels or inputs (a smooth curve is not a limit),
    summed over the two rows; it must exceed PRICE x log T, and so must m's gain on the linear rows."""
    opts = dict(opts or {})
    if any(str(k).startswith(NAME) for k in (opts.get("inputs") or {})):
        return opts, None                                  # already offered (a strategy calling itself)
    got = _steps(stocks, run, opts)
    if got is None:
        return opts, None
    ids, final, dy, read, inputs = got
    T = dy.shape[1]
    B, Q = _design(ids, read, inputs, opts, np.asarray(run["t"], dtype=float), T)
    caps = [(ids[i], read[i]) for i in range(len(ids)) if ids[i] not in final] + list(inputs.items())
    floor = [RESOLUTION * float(np.mean(dy[i] ** 2)) * T + 1e-300 for i in range(len(ids))]   # the data's resolution
    price = PRICE * math.log(T)
    best = None
    for xi, x in enumerate(ids):
        X = read[xi]
        if x in final or np.ptp(X) <= 0:
            continue
        for name, Y in caps:
            if name == x or np.ptp(Y) <= 0 or (Y > 0).sum() < 10:
                continue
            ratio = X[Y > 0] / Y[Y > 0]
            edge = min(MIN_SHARE, MIN_STEPS / len(ratio))
            grid = np.unique(np.quantile(ratio, np.linspace(edge, 1 - edge, GRID)))
            th, _ = _golden(lambda th: _rss(np.column_stack([B, np.minimum(X, th * Y)]), dy[xi]), grid)
            m = np.minimum(X, th * Y)
            gx, cx = _gain(B, m, dy[xi], floor[xi])
            if cx >= 0 or gx <= 0:                         # a limited outflow of X takes from X
                continue
            to = None
            for z in range(len(ids)):
                if z != xi:
                    gz, cz = _gain(B, m, dy[z], floor[z])
                    if cz > 0 and gz > price and (to is None or gz > to[0]):
                        to = (gz, z)
            rows = [xi] + ([to[1]] if to else [])
            th, _ = _golden(lambda th: sum(math.log(_rss(np.column_stack([B, np.minimum(X, th * Y)]), dy[r]) + floor[r])
                                           for r in rows), grid)
            share = float(np.mean(X <= th * Y))
            least = min(MIN_SHARE * T, MIN_STEPS)
            if not (least <= int(np.sum(X <= th * Y)) and least <= int(np.sum(X > th * Y))):
                continue
            m = np.minimum(X, th * Y)
            gain = sum(_gain(B, m, dy[r], floor[r])[0] for r in rows)
            kink = sum(_gain(Q, m, dy[r], floor[r])[0] for r in rows)
            if gain > price and kink > price and (best is None or kink > best["kink"]):
                best = {"source": x, "cap": name, "theta": th, "gain": gain, "kink": kink, "share": share}
                if to:
                    best["to"] = ids[to[1]]
    if best is None:
        return opts, None
    X = read[ids.index(best["source"])]
    Y = read[ids.index(best["cap"])] if best["cap"] in ids else inputs[best["cap"]]
    m = np.minimum(X, best["theta"] * Y)
    name = f"{NAME}{best['source']}__{best['cap']}"
    series = np.concatenate([[min(X[0], best["theta"] * Y[0])], m])     # input i is read by step i - 1
    inp = dict(opts.get("inputs") or {})
    inp[name] = [float(v) for v in series]
    opts["inputs"] = inp
    best["name"] = name
    opts["capmin"] = {**(opts.get("capmin") or {}), name: {"source": best["source"], "to": best.get("to")}}
    return opts, best


def _expr(rate: dict, cap: dict) -> dict:
    """A linear rate with a term reading the offered series, as an expression with the MIN written out."""
    name, x, y, th = cap["name"], cap["source"], cap["cap"], cap["theta"]
    form = rate.get("form")
    if form == "proportional":
        terms, c = [{"stock": rate.get("stock"), "k": rate.get("k", 0.0)}], 0.0
    elif form == "linear":
        terms, c = list(rate.get("terms") or []), float(rate.get("c") or 0.0)
    else:
        return rate
    if not any(t.get("stock") == name for t in terms):
        return rate
    parts = [repr(c)] if c else []
    for t in terms:
        k = float(t.get("k") or 0.0)
        if t.get("stock") == name:
            parts.append(f"min({k!r} * {x}, {k * th!r} * {y})" if k >= 0 else f"{k!r} * min({x}, {th!r} * {y})")
        else:
            parts.append(f"{k!r} * {t.get('stock')}")
    return {"form": "expr", "expr": " + ".join(parts) or "0"}


def restore(result: dict, cap: dict | None) -> dict:
    """The strategy's answer with every rate that reads the offered series written as its MIN."""
    if not cap or not isinstance(result, dict):
        return result
    out = copy.deepcopy(result)
    for n, f in enumerate(out.get("flows") or []):
        for g in f.get("byTime") or []:
            if g.get("rate"):
                g["rate"] = _expr(g["rate"], cap)
        if f.get("rate"):
            f["rate"] = _expr(f["rate"], cap)
        if f.get("rates"):
            f["rates"] = [_expr(r, cap) for r in f["rates"]]
        if f.get("input") == cap["name"]:
            f.pop("input", None)
        rates = [g.get("rate") or {} for g in f.get("byTime") or []] + [f.get("rate") or {}]
        if any(r.get("form") == "expr" for r in rates):
            f.setdefault("id", f"capmin{n}")             # an expression flow is named (the engine's compiler wants one)
    out["capmin"] = {k: cap[k] for k in ("source", "cap", "to", "theta", "gain", "kink", "share") if k in cap}
    return out


def wrap(identify, stocks: list[dict], run: dict, opts: dict | None, *args, **kw) -> dict:
    """identify(stocks, run, opts) with a capacity limit offered where the data show one."""
    o, cap = offer(stocks, run, opts)
    return restore(identify(stocks, run, o, *args, **kw), cap)
