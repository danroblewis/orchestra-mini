"""A stock that tracks a target read off another stock (a controller; kubernetes 308, 2026-10-21).

docs/research/fixes/308-kalman_lna.md. A controller (an autoscaler, a thermostat with a dead band, a
reorder rule) moves a counted stock S towards a target set by another measured stock or input X: the
target is a whole number of units, T = ceil(X / a), held between a floor and a cap. On each of S's
outside sides (an inflow from outside rises towards the target, an outflow to outside falls towards
it) the controller acts in one of two ways:
- the gap:  rise max(0, min(H, T) - S), fall max(0, S - max(L, T)) (all at once);
- a step:   rise d x (S < min(H, T)),   fall d x (S > max(L, T))     (d a step while it is off).
A linear row can only draw a proxy for this (a drift of S, or links to whatever X moves with), so the
rest of the model compensates and the flows' amounts are wrong.

`revise` runs after the strategy's answer. For every measured, non-final stock with an outside side
(where a boundary is declared: only its estimated sides), the outside net flow is read off the data
(S's one-step change, less announced injections and the answer's flows between S and other measured
stocks). Each side's form (gap or step), reader X (every other measured non-final stock, every measured
input) and threshold a are searched exactly: the predicted amounts change only where a crosses some
X_t / m (m a whole number near S_t), so every interval between those breakpoints is tried and the
middle of the best one is kept; the cap H and floor L are tried at the extreme levels S reached, and
kept only when they pay for themselves. The tracker replaces S's outside flows only where it explains
EXPLAINED of the outside net's squares (a rule, not a trend) and beats
them by Gaussian BIC with a margin of PRICE x log T. The same rule for every case; nothing reads the
case or its truth.

To adopt it in another strategy: copy this file into the package and call
`result = tracker.revise(result, stocks, run, opts)` on the strategy's final answer (after any MIN or
other expression is written back). It needs nothing else from the package.
"""

from __future__ import annotations

import copy
import math
import time

import numpy as np

PRICE = 6.0            # the margin the tracker must win by, in log T (as capmin's)
RESOLUTION = 1e-8      # a residual variance below this share of the mean squared change is rounding
CHUNK = 256            # thresholds tried at once
MOST = 6000            # at most this many breakpoints per reader (thinned evenly beyond)
MIN_FIRES = 3          # a side must act on at least this many steps to be read as a controller
TIME = 20.0            # seconds for the whole search (the stocks with the worst-fitted outside nets first)
EXPLAINED = 0.9        # a controller acts by a rule: it must explain this share of the outside net's squares


def _rate_value(rate: dict, state: dict) -> float | None:
    form = (rate or {}).get("form")
    if form == "proportional":
        return float(rate.get("k") or 0.0) * state.get(rate.get("stock"), 0.0)
    if form == "linear":
        return float(rate.get("c") or 0.0) + sum(float(t.get("k") or 0.0) * state.get(t.get("stock"), 0.0)
                                                 for t in rate.get("terms") or [])
    if form == "constant":                                 # (regress-kalman_lna) kalman_lna writes "value" too (309)
        return float(rate.get("value", rate.get("c", rate.get("k", 0.0))) or 0.0)
    if form == "expr":
        try:
            return float(eval(rate["expr"], {"__builtins__": {}}, {"min": min, "max": max, "ceil": math.ceil,
                                                                 "abs": abs, **state}))
        except Exception:                                  # noqa: BLE001 (an expression this reader cannot run)
            return None
    return None


def _amounts(f: dict, states: list[dict]) -> np.ndarray | None:
    if f.get("byTime") or f.get("effects") or f.get("uses"):
        return None
    out = np.zeros(len(states))
    for i, s in enumerate(states):
        v = _rate_value(f.get("rate") or {}, s)
        if v is None:
            return None
        out[i] = max(0.0, v)
    return out


def _data(stocks: list[dict], run: dict, opts: dict):
    ids = [s["id"] for s in stocks]
    final = {s["id"] for s in stocks if s.get("final")}
    t = np.asarray(run["t"], dtype=float)
    L = np.array([np.asarray(run["levels"].get(i, []), dtype=float) for i in ids])
    if L.ndim != 2 or L.shape[1] < 20 or not np.isfinite(L).all() or np.any(np.diff(t) != 1):
        return None
    dy = L[:, 1:] - L[:, :-1]
    read = L[:, :-1].copy()
    T = dy.shape[1]
    at = {float(x): k for k, x in enumerate(t[1:])}
    index = {s: i for i, s in enumerate(ids)}
    for j in opts.get("injections") or []:
        k, i = at.get(float(j["t"])), index.get(j.get("stock"))
        if k is not None and i is not None:
            dy[i, k] -= float(j["amount"])
            read[i, k] = read[i, k] + float(j["amount"])
    inputs = {}
    for name, v in (opts.get("inputs") or {}).items():
        if name in index or str(name).startswith("capmin__"):
            continue
        u = np.nan_to_num(np.asarray(v, dtype=float))
        if len(u) >= T + 1:
            inputs[name] = u[1:T + 1]                       # an input is read at the step's end
    states = [{**{ids[i]: float(read[i, k]) for i in range(len(ids))}, **{n: float(u[k]) for n, u in inputs.items()}}
              for k in range(T)]
    return ids, final, dy, read, inputs, states


def _predict(side: str, form: str, S: np.ndarray, target: np.ndarray, d: float) -> np.ndarray:
    """Amounts per step (rows: thresholds) of one side's form, given the clamped target."""
    if side == "in":
        return np.maximum(0.0, target - S) if form == "gap" else d * (S < target)
    return np.maximum(0.0, S - target) if form == "gap" else d * (S > target)


def _breakpoints(X: np.ndarray, S: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Every threshold a at which ceil(X_t / a) crosses a whole number near S_t."""
    top = float(np.max(y)) if len(y) else 0.0
    if not np.isfinite(top) or top > MOST:
        return np.zeros(0)                                   # (regress-kalman_lna, from matrix_v2's copy at 307af7c) no
                                                             # whole-number controller moves this much a step (a runaway net)
    top = int(np.ceil(top)) + 2
    ms = np.arange(-2, top + 1)
    m = np.round(S)[:, None] + ms[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        b = (X[:, None] / m).ravel()
    b = np.unique(b[np.isfinite(b) & (b > 0)])
    if len(b) > MOST:
        b = b[np.linspace(0, len(b) - 1, MOST).astype(int)]
    return b


def _fit_side(side: str, form: str, X: np.ndarray, S: np.ndarray, y: np.ndarray, clamp: float | None):
    """(sse, a, d) of the best threshold for this side, form, reader and clamp; None where none acts."""
    br = _breakpoints(X, S, y)
    if len(br) < 2:
        return None
    mids = np.sqrt(br[:-1] * br[1:])
    fired = y > 0
    d = float(np.median(y[fired])) if fired.any() else 0.0
    if d <= 0:
        return None
    sse = np.empty(len(mids))
    for lo in range(0, len(mids), CHUNK):
        a = mids[lo:lo + CHUNK]
        target = np.ceil(X[None, :] / a[:, None])
        if clamp is not None:
            target = np.minimum(clamp, target) if side == "in" else np.maximum(clamp, target)
        p = _predict(side, form, S[None, :], target, d)
        sse[lo:lo + CHUNK] = ((p - y[None, :]) ** 2).sum(axis=1)
    j = int(np.argmin(sse))
    best = sse[j]
    tie = 1e-9 * max(1.0, float(y @ y))
    i0 = j
    while i0 > 0 and sse[i0 - 1] <= best + tie:
        i0 -= 1
    i1 = j
    while i1 < len(sse) - 1 and sse[i1 + 1] <= best + tie:
        i1 += 1
    a = math.sqrt(br[i0] * br[i1 + 1])                       # the middle of the best interval
    if form == "step":                                       # d refitted on the steps it acts at a
        target = np.ceil(X / a)
        if clamp is not None:
            target = np.minimum(clamp, target) if side == "in" else np.maximum(clamp, target)
        on = _predict(side, "step", S, target, 1.0) > 0
        d = float(np.mean(y[on])) if on.any() else d
    return float(best), a, d


def _side(side: str, S: np.ndarray, y: np.ndarray, readers: list[tuple[str, np.ndarray]], floor: float,
          deadline: float = math.inf):
    """The best controller for one side: {form, reader, a, d, clamp, sse, numbers}; None where the side
    acts on fewer than MIN_FIRES steps."""
    if int((y > 0).sum()) < MIN_FIRES:
        return None
    T = len(y)
    logT = math.log(T)
    clamps = [None, float(np.max(S + y)) if side == "in" else float(np.min(S - y))]
    best = None
    for name, X in readers:
        if np.ptp(X) <= 0 or np.any(X < 0) or time.monotonic() > deadline:
            continue
        for form in ("gap", "step"):
            for clamp in clamps:
                got = _fit_side(side, form, X, S, y, clamp)
                if got is None:
                    continue
                sse, a, d = got
                k = 1 + (form == "step") + (clamp is not None)
                score = T * math.log(sse + floor) + k * logT
                if best is None or score < best["score"] - 1e-9:
                    best = {"form": form, "reader": name, "a": a, "d": d, "clamp": clamp, "sse": sse, "numbers": k,
                            "score": score}
    return best


def _expr(side: str, c: dict, S: str) -> str:
    T = f"ceil({c['reader']} / {c['a']!r})"
    if c["clamp"] is not None:
        T = f"min({c['clamp']!r}, {T})" if side == "in" else f"max({c['clamp']!r}, {T})"
    if side == "in":
        return f"max(0, {T} - {S})" if c["form"] == "gap" else f"{c['d']!r} * ({S} < {T})"
    return f"max(0, {S} - {T})" if c["form"] == "gap" else f"{c['d']!r} * ({S} > {T})"


def _numbers(f: dict) -> int:
    r = f.get("rate") or {}
    if r.get("form") == "linear":
        return len(r.get("terms") or []) + (1 if r.get("c") else 0)
    if r.get("form") == "expr":
        return max(1, r.get("expr", "").count("."))
    return 1


def revise(result: dict, stocks: list[dict], run: dict, opts: dict | None) -> dict:
    """The answer with each counted stock's outside flows replaced by a tracker where one wins."""
    if not isinstance(result, dict) or not result.get("flows"):
        return result
    if result.get("lag") or result.get("scales") or (result.get("kalman") or {}).get("reading"):
        return result                                        # the answer reads the data in another frame
    opts = opts or {}
    got = _data(stocks, run, opts)
    if got is None:
        return result
    ids, final, dy, read, inputs, states = got
    # (regress-kalman_lna, from common_cause's copy at 7778159) the answer's own hidden stocks read at their inferred
    # levels (a step's start), so a hidden stage feeding S (outside -> H -> S, rows.py's delayed inflow) is priced as
    # what it delivers
    for H, lv in (result.get("inferred_levels") or {}).items():
        if H not in ids and len(lv) >= dy.shape[1]:
            for k, st in enumerate(states):
                st[H] = float(lv[k])
    T = dy.shape[1]
    logT = math.log(T)
    flows = list(result["flows"])
    declared = bool(run.get("boundary_declared"))
    bd = run.get("boundary") or {}
    deadline = time.monotonic() + TIME
    todo = []
    for si, S in enumerate(ids):
        got = _stock(si, S, ids, final, dy, read, states, flows, declared, bd)
        if got is not None:
            todo.append(got)
    todo.sort(key=lambda g: -g["misfit"])                   # the worst-fitted outside nets first
    changed, drop, added = [], set(), []
    for g in todo:
        si, S, sides, outside, net, base, k_base, floor = (g[k] for k in ("si", "S", "sides", "outside", "net", "base",
                                                                           "k_base", "floor"))
        readers = [(ids[j], read[j]) for j in range(len(ids)) if j != si and ids[j] not in final] + list(inputs.items())
        Sv = read[si]
        fits = {}
        for s in sides:
            y = np.maximum(net, 0.0) if s == "in" else np.maximum(-net, 0.0)
            fits[s] = _side(s, Sv, y, readers, floor, deadline)
        fits = {s: c for s, c in fits.items() if c is not None}
        if not fits:
            continue
        pred = np.zeros(T)
        for s, c in fits.items():
            target = np.ceil(dict(readers)[c["reader"]] / c["a"])
            if c["clamp"] is not None:
                target = np.minimum(c["clamp"], target) if s == "in" else np.maximum(c["clamp"], target)
            p = _predict(s, c["form"], Sv, target, c["d"])
            c["idle"] = int((p <= 0).sum())
            pred += p if s == "in" else -p
        if any(c["idle"] < MIN_FIRES for c in fits.values()):
            continue                                         # (regress-kalman_lna, as common_cause's and matrix_v2's
                                                             # copies) a rule acting on every step is a rate, not a controller
        k_new = sum(c["numbers"] for c in fits.values())
        for s in sides:                                      # a side the tracker leaves keeps its flows
            if s not in fits:
                for n in outside[s]:
                    k_new += _numbers(flows[n])
                    if S not in (flows[n].get("from"), flows[n].get("to")):
                        continue                             # a hidden stage's own inflow: numbers, no amount on S
                    amt = _amounts(flows[n], states)
                    pred += amt if s == "in" else -amt
        bic = lambda sse, k: T * math.log((sse + floor) / T) + k * logT  # noqa: E731
        sse_new = float(((net - pred) ** 2).sum())
        sse_base = float(((net - base) ** 2).sum())
        if sse_new > (1 - EXPLAINED) * float(net @ net) or not bic(sse_new, k_new) + PRICE * logT < bic(sse_base, k_base):
            continue
        drop |= {n for s in fits for n in outside[s]}
        for s, c in fits.items():
            rate = {"form": "expr", "expr": _expr(s, c, S)}
            added.append({"from": None if s == "in" else S, "to": S if s == "in" else None, "ratio": 1,
                          "rate": rate, "rates": [dict(rate)], "id": f"tracker_{s}_{S}"})
        changed.append({"stock": S, **{s: dict(c) for s, c in fits.items()}, "sse": [sse_base, sse_new]})
    if not changed:
        return result
    out = copy.copy(result)
    out["flows"] = [f for n, f in enumerate(flows) if n not in drop] + added
    gone = {x for n in drop for x in (flows[n].get("from"), flows[n].get("to"))} - {x for f in out["flows"]
                                                                                   for x in (f.get("from"), f.get("to"))}
    if gone and out.get("inferred_levels"):                  # (regress-kalman_lna) a replaced hidden stage's levels go too
        out["inferred_levels"] = {h: v for h, v in out["inferred_levels"].items() if h not in gone}
    out["tracker"] = changed
    return out


def _stock(si, S, ids, final, dy, read, states, flows, declared, bd) -> dict | None:
    """S's outside net flow (its change less the answer's flows between it and other measured stocks), the
    answer's outside flows on the sides a tracker may take, and how badly they fit; None where S is not a
    candidate (final, constant, no estimated side, a flow this reader cannot run, or fitted to resolution)."""
    if S in final or np.ptp(read[si]) <= 0:
        return None
    T = dy.shape[1]
    sides = [side for side in ("in", "out") if not declared or "estimate" in ((bd.get(S) or {}).get(side) or [])]
    if not sides:
        return None
    inside = np.zeros(T)
    outside = {"in": [], "out": []}
    stages = _stages(S, ids, flows)
    for n, f in enumerate(flows):
        a, b = f.get("from"), f.get("to")
        if S not in (a, b):
            continue
        if a in stages:                                      # (regress-kalman_lna) a hidden stage is an outside inflow
            outside["in"].append(n)
            continue
        if a is None or b is None:
            outside["in" if b == S else "out"].append(n)
            continue
        amt = _amounts(f, states)
        if amt is None:
            return None
        inside += amt if b == S else -float(f.get("ratio") or 1) * amt
    if any(outside[s] for s in ("in", "out") if s not in sides):
        return None                                          # an outside flow the tracker may not replace
    net = dy[si] - inside
    base = np.zeros(T)
    k_base = 0
    for s in sides:
        for n in outside[s]:
            amt = _amounts(flows[n], states)
            if amt is None:
                return None
            base += amt if s == "in" else -amt
            k_base += _numbers(flows[n])
    for H, ns in stages.items():                             # the stage's own flows leave with the side it feeds
        outside["in"] += [n for n in ns if n not in outside["in"]]
        k_base += sum(_numbers(flows[n]) for n in ns if flows[n].get("to") != S)
    floor = RESOLUTION * float(np.mean(dy[si] ** 2)) * T + 1e-300
    sse_base = float(((net - base) ** 2).sum())
    if sse_base <= floor:
        return None                                          # the answer already fits S to the data's resolution
    return {"si": si, "S": S, "sides": sides, "outside": outside, "net": net, "base": base, "k_base": k_base,
            "floor": floor, "misfit": sse_base / (float(net @ net) + floor)}


def _stages(S: str, ids: list[str], flows: list[dict]) -> dict[str, list[int]]:
    """(regress-kalman_lna; copied from common_cause/tracker.py at 7778159, port-common_cause) The hidden stages feeding
    S: a stock H nobody measures whose only outflow is H -> S and whose inflows come from outside (rows.py's delayed
    inflow, outside -> H -> S, since febd195). What H delivers is S's outside inflow, so a tracker taking S's inflow side
    replaces the whole stage: {H: indices of every flow touching H}. Before, H -> S read as an inside flow at 0 (H has no
    measured level), so the tracker's inflow was added beside the stage: two inflows on one declared side (308, 309
    none-short, "invented boundary")."""
    out: dict[str, list[int]] = {}
    known = set(ids)
    for H in {f.get("from") for f in flows if f.get("to") == S} - known - {None}:
        ns = [n for n, f in enumerate(flows) if H in (f.get("from"), f.get("to"))]
        if all((f.get("from") == H and f.get("to") == S) or (f.get("to") == H and f.get("from") is None)
               for f in (flows[n] for n in ns)):
            out[H] = ns
    return out
