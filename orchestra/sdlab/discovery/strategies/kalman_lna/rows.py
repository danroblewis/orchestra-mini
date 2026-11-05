"""Each stock's row read off the data, and the flows rebuilt where the answer misses it (kubernetes 309, 2026-10-21).

docs/research/fixes/309-kalman_lna.md. The strategy's answer is a set of flows; the flows touching a stock S
give S's row: its one-step change as a function of the levels. Where the answer's row is far from what the
data show (a near-collinear set of levels let the search pick the wrong readers; or a flow floored at 0,
`max(0, B . x)`, that a linear row can only draw a proxy for), the stock's row is read off the data directly
and the flows touching it are rebuilt from the rows:

1. Rows. For every measured stock, two forms are fitted to its one-step change (less announced injections),
   reading every measured non-final stock, every measured input and a constant:
   - linear: the best sparse subset (every subset up to SUB terms, then forward steps up to MOST_TERMS);
   - linear + one floored term on a side the boundary allows (an outside inflow max(0, B . x), or an
     outflow): the always-on part fitted by trimmed least squares on every support up to two readers
     (the steps where the floored term is off fit it exactly), the floored term's readers searched on the
     steps it acts, then both refitted jointly with the active steps re-read from B . x > 0 until stable.
   The best by Gaussian BIC (the rounding of the data as a floor) is the stock's row.
2. Which stocks. A stock is rebuilt where its row beats the answer's flows on it by BIC + PRICE log T and
   explains EXPLAINED of its change's squares; a stock joined to a rebuilt one by a flow is rebuilt too
   where its own row is not worse than its answer (so a move is never cut on one side only). Flows
   between rebuilt stocks and the others stay, and the rebuilt rows are refitted on the change they leave.
3. Flows from rows. A positive term reading X in T's row, X a rebuilt stock in T's unit with a drain of
   its own (a negative self term) at least that large, is a move X -> T (the largest first); what is left
   of a drain is X's outflow to outside; every other term is an outside link on the side its sign gives
   (one flow per side: the positive terms in, the negative out; the floored term its own side, alone).
   A side the declared boundary does not allow removes the stock from the rebuild.
4. The rebuild is kept only where, over the rebuilt stocks, the new flows beat the answer's by BIC +
   PRICE log T. The same rule for every case; nothing reads the case or its truth.

To adopt it in another strategy: copy this file into the package and call
`result = rows.revise(result, stocks, run, opts)` on the answer after any MIN or expression is written back
and before a controller pass (tracker.py): the controller then reads the rebuilt neighbours. Self-contained.
"""

from __future__ import annotations

import copy
import itertools
import math
import time

import numpy as np

from . import delayed as _delayed
from . import derived as _derived
from . import subsets as _subsets

PRICE = 6.0            # the margin a rebuild must win by, in log T
SIMPLER = 2.0          # (316) ... where the row reads fewer numbers than the answer spends on the stock
RESOLUTION = 1e-8      # a residual variance below this share of the mean squared change is rounding
EXPLAINED = 0.5        # a rebuilt row explains at least this share of the stock's change's squares (counting noise leaves the rest)
SUB = 3                # every subset up to this many readers is tried
MOST_TERMS = 5         # then forward steps up to this many
BASE_SUB = 2           # the always-on part of a floored row: every support up to this many readers
MIN_ON = 5             # a floored term acts on at least this many steps
TIME = 600.0           # seconds for the rows and the derived-column search (310: 90; 311: 200, the capped move one
                       # round more; 312: 600, the delayed rows and a seeded limit: 272 s on the Mac for none-long,
                       # node2 is up to 3 x slower; under the 1800 s case limit; 312 round 2: 176 s, subsets.py)
BUILD_TIME = 40.0      # seconds for the rebuild after them (the whole pass's budget before 310)
TOL_MOVE = 0.02        # a move may take this share more than the source's drain (rounding)
REPAIRS = 3            # rounds of refitting a row that needs a side the boundary lacks
LEFT = 1e-4            # what a drain leaves below this share of itself after its moves is rounding
ACTIVE_ROUNDS = 3      # (311) active-set rereads of a floored term's readers in the search
REREAD_SUPPORTS = 3    # (311) ... for this many of the best always-on parts
INTEGRATOR_ROUNDS = 4  # (311) rounds of a switched integrator's threshold and drain rate refitted in turn
RICH_SHARE = 0.4       # (317) the share of TIME the rows' richer forms (schedule, integrator, delay) may use, after
                       # every stock's plain row is read; each row's delay fit gets an even slice of what is left
RICH_SLICE = 20.0      # (317) ... and at least this many seconds


def _rate_value(rate: dict, state: dict) -> float | None:
    form = (rate or {}).get("form")
    if form == "proportional":
        return float(rate.get("k") or 0.0) * state.get(rate.get("stock"), 0.0)
    if form == "linear":
        return float(rate.get("c") or 0.0) + sum(float(t.get("k") or 0.0) * state.get(t.get("stock"), 0.0)
                                                 for t in rate.get("terms") or [])
    if form == "constant":
        return float(rate.get("value", rate.get("c", 0.0)) or 0.0)
    if form == "expr":
        try:
            return float(eval(rate["expr"], {"__builtins__": {}}, {"min": min, "max": max, "ceil": math.ceil,
                                                                 "abs": abs, **state}))
        except Exception:                                  # noqa: BLE001 (an expression this reader cannot run)
            return None
    return None


def _amounts(f: dict, states: list[dict], t_end: np.ndarray) -> np.ndarray | None:
    """The flow's amount on each step (floored at 0), pieces and eras included; None where it cannot run. (regress-
    kalman_lna) A co-flow runs: what it also takes from its "uses" stocks is per x this amount (answer, _co_group)."""
    if f.get("effects"):
        return None
    T = len(states)
    out = np.zeros(T)
    if f.get("inflow"):
        for k in range(T):
            seg = next((g for g in f["inflow"] if g["from"] <= t_end[k] <= g["to"]), f["inflow"][-1])
            out[k] = max(0.0, float(seg.get("value") or 0.0))
        return out
    segs = f.get("byTime")
    for k, s in enumerate(states):
        rate = f.get("rate") or {}
        if segs:
            rate = next((g for g in segs if g["from"] <= t_end[k] <= g["to"]), segs[-1]).get("rate") or {}
        v = _rate_value(rate, s)
        if v is None:
            return None
        out[k] = max(0.0, v)
    return out


def _co_group(f: dict) -> set:
    """(regress-kalman_lna) The measured stocks a co-flow touches: its ends and every stock it draws on; empty for a
    flow without "uses". 309 none-long since febd195 (capsteps): the base answer's pending -> ready co-flow (uses
    desired) made answer() None on pending and ready, so the rows never rebuilt them and the gap flows were lost."""
    if not f.get("uses"):
        return set()
    return ({f.get("from"), f.get("to")} | {u.get("stock") for u in f["uses"]}) - {None}


def _numbers(f: dict) -> int:
    if f.get("numbers"):
        return int(f["numbers"])
    if f.get("inflow"):
        return len(f["inflow"])
    if f.get("byTime"):
        return sum(_numbers({"rate": g.get("rate")}) for g in f["byTime"])
    r = f.get("rate") or {}
    if r.get("form") == "linear":
        return len(r.get("terms") or []) + (1 if r.get("c") else 0)
    if r.get("form") == "expr":
        return max(1, r.get("expr", "").count("."))
    return 1


def _data(stocks: list[dict], run: dict, opts: dict):
    ids = [s["id"] for s in stocks]
    t = np.asarray(run["t"], dtype=float)
    L = np.array([np.asarray(run["levels"].get(i, []), dtype=float) for i in ids])
    if L.ndim != 2 or L.shape[1] < 30 or not np.isfinite(L).all() or np.any(np.diff(t) != 1):
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
    return ids, dy, read, inputs, states, t[1:]


def _lstsq(A: np.ndarray, y: np.ndarray):
    b, *_ = np.linalg.lstsq(A, y, rcond=None)
    e = y - A @ b
    return b, float(e @ e)


def _sparse(X: np.ndarray, y: np.ndarray, floor: float, cols: list[int] | None = None):
    """(bic, support, coefs, sse) of the best sparse linear fit of y on X's columns (BIC; floor: rounding).
    (312 round 2) Each size's subsets are ranked in one batch (subsets.py) and the best few refitted by least squares."""
    n, p = X.shape
    cols = list(range(p)) if cols is None else cols
    logn = math.log(max(n, 2))
    bic = lambda sse, k: n * math.log((sse + floor) / n) + k * logn  # noqa: E731
    best = (bic(float(y @ y), 0), (), np.zeros(0), float(y @ y))
    by_size = {}
    for k in range(1, min(SUB, len(cols)) + 1):
        sub, b, sse = _subsets.best(X, y, list(itertools.combinations(cols, k)))
        by_size[k] = (bic(sse, k), sub, b, sse)
    for k, got in by_size.items():
        if got[0] < best[0]:
            best = got
    cur = by_size.get(min(SUB, len(cols)))
    while cur is not None and len(cur[1]) < min(MOST_TERMS, len(cols)):
        cands = [tuple(sorted(cur[1] + (j,))) for j in cols if j not in cur[1]]
        if not cands:
            break
        sub, b, sse = _subsets.best(X, y, cands)
        step = (bic(sse, len(sub)), sub, b, sse)
        if step[3] >= cur[3] * (1 - 1e-9):
            break
        cur = step
        if cur[0] < best[0]:
            best = cur
    return best


def _trimmed(A: np.ndarray, y: np.ndarray, keep: float = 0.5, rounds: int = 6) -> np.ndarray:
    """Least trimmed squares: the half of the informative steps (those where a reader is not 0) it fits best."""
    b, _ = _lstsq(A, y)
    info = np.flatnonzero(np.any(A != 0, axis=1))
    if len(info) < A.shape[1] + 2:
        return b
    for _ in range(rounds):
        r = np.abs(y[info] - A[info] @ b)
        idx = info[np.argsort(r)[: max(A.shape[1] + 2, int(keep * len(info)))]]
        b, _ = _lstsq(A[idx], y[idx])
    return b


def _floored(X: np.ndarray, y: np.ndarray, sign: float, floor: float, deadline: float, a_cols: list[int] | None = None,
             b_cols: list[int] | None = None, strict: tuple = (1.0,)):
    """(bic, base support, a, floored support, b, sse) of y = X_A a + sign max(0, X_B b); None where none fits.
    strict: multiples of the noise threshold the floored term's steps are read at (312, delayed.py: (1, 4), the second
    from the nonzero residuals, where the level's rounding put a few steps past the first one and the term acts on
    only 20)."""
    n, p = X.shape
    logn = math.log(max(n, 2))
    best = None
    res = math.sqrt(floor / n)
    a_cols = list(range(p)) if a_cols is None else a_cols
    supports = [()] + [s for k in range(1, BASE_SUB + 1) for s in itertools.combinations(a_cols, k)]
    tried = []
    for A in supports:
        if time.monotonic() > deadline:
            break
        a = _trimmed(X[:, A], y) if A else np.zeros(0)
        r = y - (X[:, A] @ a if A else 0.0)
        small = np.abs(r) <= np.quantile(np.abs(r), 0.5)
        mad = float(np.median(np.abs(r[small]))) if small.any() else 0.0
        tol0 = max(4.0 * 1.4826 * mad, 10 * res, 1e-9)
        tol1 = max(tol0, _delayed.noise(r)) if strict != (1.0,) else tol0   # (312) zeros of a level held still left out
        for mult in strict:
            on = sign * r > mult * (tol0 if mult == 1.0 else tol1)
            if int(on.sum()) < MIN_ON:
                continue
            top_A = None
            for B, b in _hinge_readers(X, r, on, sign, floor, n, logn, b_cols=b_cols):
                got = _joint(X, y, A, a, B, b, sign, floor, n, logn)
                if got is not None and (top_A is None or got[0] < top_A):
                    top_A = got[0]
                if got is not None and (best is None or got[0] < best[0]):
                    best = got
            if top_A is not None:
                tried.append((top_A, A, a, r, on))
    # (311) the few best always-on parts searched again with each floored candidate's steps re-read (ACTIVE_ROUNDS)
    for _, A, a, r, on in sorted(tried, key=lambda c: c[0])[:REREAD_SUPPORTS]:
        if time.monotonic() > deadline:
            break
        for B, b in _hinge_readers(X, r, on, sign, floor, n, logn, b_cols=b_cols, reread=True):
            got = _joint(X, y, A, a, B, b, sign, floor, n, logn)
            if got is not None and (best is None or got[0] < best[0]):
                best = got
    while best is not None:                                  # terms of either part that do not pay are dropped
        _, A, a, B, b, _ = best
        trial = None
        for drop in [("A", j) for j in range(len(A))] + [("B", j) for j in range(len(B))] * (len(B) > 1):
            A2 = tuple(x for i, x in enumerate(A) if not (drop[0] == "A" and i == drop[1]))
            B2 = tuple(x for i, x in enumerate(B) if not (drop[0] == "B" and i == drop[1]))
            a2 = np.array([v for i, v in enumerate(a) if not (drop[0] == "A" and i == drop[1])])
            b2 = np.array([v for i, v in enumerate(b) if not (drop[0] == "B" and i == drop[1])])
            got = _joint(X, y, A2, a2, B2, b2, sign, floor, n, logn)
            if got is not None and got[0] < (trial or best)[0]:
                trial = got
        if trial is None:
            break
        best = trial
    return best


def _joint(X, y, A, a, B, b, sign, floor, n, logn):
    """(bic, A, a, B, b, sse) of both parts refitted jointly, the active steps re-read from X_B b > 0."""
    on1, exact = None, False
    for it in range(12):
        on2 = X[:, B] @ b > 0
        if int(on2.sum()) < MIN_ON:
            return None
        M = np.hstack([X[:, A], sign * X[:, B] * on2[:, None]])
        done = on1 is not None and np.array_equal(on2, on1)
        # (312 round 2) the active set is found by the scaled normal equations (fast), the last fit by least squares,
        # so the numbers are least squares' on the set it settles on (an active set that moves then goes on exactly)
        coef = _subsets.normal(M, y) if not (exact or done or it == 11) else None
        if coef is None:
            coef, _ = _lstsq(M, y)
            if done or it == 11:
                a, b = coef[: len(A)], coef[len(A):]
                if done and not np.array_equal(X[:, B] @ b > 0, on2) and not exact:
                    exact, on1 = True, on2
                    continue
                break
        a, b = coef[: len(A)], coef[len(A):]
        on1 = on2
    pred = (X[:, A] @ a if len(A) else 0.0) + sign * np.maximum(0.0, X[:, B] @ b)
    sse = float(((y - pred) ** 2).sum())
    return n * math.log((sse + floor) / n) + (len(A) + len(B)) * logn, A, a, B, b, sse


def _hinge_readers(X, r, on, sign, floor, n, logn, top: int = 3, b_cols: list[int] | None = None, reread: bool = False):
    """The floored term's best readers for what the always-on part leaves (r): each subset fitted on the steps
    it acts (on) and scored on every step (off steps must read B . x <= 0); the top few (support, coefs).
    (312) Each subset's first fit from the columns' scaled Gram matrix on the acting steps; (312 round 2) all subsets
    of a size at once (subsets.py), their active-set rereads too."""
    cols = list(range(X.shape[1])) if b_cols is None else b_cols
    target = sign * r
    hinge = lambda P: np.maximum(0.0, P)  # noqa: E731
    bics, subs, coefs = [], [], []
    for k in range(1, min(SUB, len(cols)) + 1):
        combos = list(itertools.combinations(cols, k))
        C = np.asarray(combos, dtype=int).reshape(len(combos), k)
        B, _ = _subsets.fits(X, target, C, rows=on)
        sse = _subsets.sse(X, target, C, B, hinge)
        live = np.arange(len(combos)) if reread else np.zeros(0, dtype=int)
        for _ in range(ACTIVE_ROUNDS if reread else 0):
            # (311) the steps re-read from where the term acts (X_B b > 0): a few steps the always-on part misses
            # by a little (an imprecise kill threshold) pulled the true readers' fit toward 0 on the first read
            if not len(live):
                break
            B2, s2, enough = _subsets.reread(X, target, C[live], B[live], MIN_ON, hinge)
            better = enough & (s2 < sse[live] * (1 - 1e-9))
            B[live[better]], sse[live[better]] = B2[better], s2[better]
            live = live[better]
        bics.append(n * np.log((sse + floor) / n) + k * logn)
        subs += combos
        coefs += list(B)
    if not subs:
        return []
    order = np.argsort(np.concatenate(bics), kind="stable")[:top]
    return [(subs[i], coefs[i]) for i in order]


def _sides(S: str, declared: bool, bd: dict) -> set[str]:
    """The sides a rebuilt row may put a flow on. (312) A side declared only "input" is fed by exactly 1 x the input
    (boundary.py enforces it and _build keeps that flow), so no rebuilt flow goes there."""
    if not declared:
        return {"in", "out"}
    d = bd.get(S) or {}
    return {s for s in ("in", "out") if d.get(s) and not _input_only(d.get(s))}


def _input_only(kinds) -> bool:
    kinds = [kinds] if isinstance(kinds, str) else list(kinds or [])
    return bool(kinds) and all(k == "input" for k in kinds)


def _fixed_input(f: dict, declared: bool, bd: dict, inputs) -> bool:
    """(312) The flow that feeds a side declared only "input": from outside, reading measured inputs only."""
    if not declared or f.get("from") is not None or f.get("to") is None:
        return False
    if not _input_only((bd.get(f["to"]) or {}).get("in")):
        return False
    r = f.get("rate") or {}
    reads = {r.get("stock")} if r.get("form") == "proportional" else {t.get("stock") for t in r.get("terms") or []}
    return bool(reads) and reads <= set(inputs) and not f.get("byTime")


def revise(result: dict, stocks: list[dict], run: dict, opts: dict | None, offered: dict | None = None) -> dict:
    """The answer with the flows of the stocks whose rows it misses rebuilt from rows read off the data; offered: the
    limit capmin offered to the fit ({source, cap, theta, ...}), which the rows may read as a capped move (312)."""
    if not isinstance(result, dict) or not result.get("flows"):
        return result
    if result.get("lag") or result.get("scales") or (result.get("kalman") or {}).get("reading"):
        return result                                        # the answer reads the data in another frame
    opts = opts or {}
    got = _data(stocks, run, opts)
    if got is None:
        return result
    ids, dy, read, inputs, states, t_end = got
    deadline = time.monotonic() + TIME
    T = dy.shape[1]
    logT = math.log(T)
    unit = {s["id"]: s.get("unit") or "" for s in stocks}
    final = {s["id"] for s in stocks if s.get("final")}
    declared = bool(run.get("boundary_declared"))
    bd = run.get("boundary") or {}
    alone = {S for S in ids if sum(unit.get(o) == unit.get(S) for o in ids) == 1}

    def movable(Sx, St):
        """May a move X -> T carry X's loss into T? In one unit, always. (316) Across units only where the declared
        boundary and the units leave no other way: X alone in its unit with no outside outflow (what it loses can
        only go into another stock), or T alone in its unit with no outside inflow (what it gains can only come
        from one): pgbouncer's waiting queries become active connections, the move is the conservation claim."""
        if unit.get(Sx) == unit.get(St):
            return True
        if not declared or Sx in final or St in final:
            return False
        return ((Sx in alone and "out" not in _sides(Sx, declared, bd))
                or (St in alone and "in" not in _sides(St, declared, bd)))

    readers = [i for i, s in enumerate(ids) if s not in final]
    names = [ids[i] for i in readers] + list(inputs) + [None]
    X = np.vstack([read[i] for i in readers] + [inputs[k] for k in inputs] + [np.ones(T)]).T
    keep_cols = [j for j in range(X.shape[1]) if np.ptp(X[:, j]) > 0 or names[j] is None]
    X, names = X[:, keep_cols], [names[j] for j in keep_cols]
    flows = list(result["flows"])
    amts = [_amounts(f, states, t_end) for f in flows]
    floor = {S: RESOLUTION * float(np.mean(dy[i] ** 2)) * T + 1e-300 for i, S in enumerate(ids)}
    bic = lambda sse, k, S: T * math.log((sse + floor[S]) / T) + k * logT  # noqa: E731

    def answer(S, skip=frozenset()):
        """(sse, numbers) of the answer's flows on S, flows in skip left out; None where one cannot run."""
        i = ids.index(S)
        pred, k = np.zeros(T), 0
        for n, f in enumerate(flows):
            uses = [u for u in f.get("uses") or [] if u.get("stock") == S]
            if n in skip or (S not in (f.get("from"), f.get("to")) and not uses):
                continue
            if amts[n] is None:
                return None
            if S in (f.get("from"), f.get("to")):
                pred += amts[n] if f.get("to") == S else -float(f.get("ratio") or 1) * amts[n]
                k += _numbers(f)
            for u in uses:                                   # (regress-kalman_lna) a co-flow's draw on S
                pred -= float(u.get("per") or 0.0) * amts[n]
                k += 1
        return float(((dy[i] - pred) ** 2).sum()), k

    rich_ok: set = set()                                 # (317) the stocks whose richer forms have been read
    slice_end = [math.inf]                               # (317) the delay fit's own deadline (its slice of the pass)

    def row(S, y, self_k=None, exclude=frozenset(), rich=None):
        """The best row for S's change y: {bic, sse, lin: {name: k}, hinge: (side, {name: k}) | None}; with
        self_k, S's own term is fixed at self_k (its drain is what its moves take) and the rest is searched;
        the always-on part reads no name in exclude (links on a side the boundary does not allow)."""
        yy, cols = y, [j for j, n in enumerate(names) if n not in exclude]
        if self_k is not None and S in names and S not in exclude:
            j = names.index(S)
            yy = y - self_k * X[:, j]
            cols.remove(j)                                   # S's own term fixed; its floored term may read it
        lin = _sparse(X, yy, floor[S], cols)
        out = {"bic": lin[0], "sse": lin[3], "lin": dict(zip([names[j] for j in lin[1]], map(float, lin[2]))),
               "hinge": None, "res": yy - (X[:, list(lin[1])] @ lin[2] if len(lin[1]) else 0.0)}
        if lin[3] > floor[S] and S not in final:
            # a floored term reads levels only, S's own among them even where its always-on term is fixed (309: the
            # gap desired - pending - ready once pending's drain is fixed at what its move takes; cols lacks it then)
            b_cols = [j for j, n in enumerate(names) if not str(n).startswith("__")]
            for side in sorted(_sides(S, declared, bd)):
                sign = 1.0 if side == "in" else -1.0
                h = _floored(X, yy, sign, floor[S], deadline, cols, b_cols)
                if h is not None and h[0] < out["bic"] - 1e-9:
                    pred = (X[:, list(h[1])] @ h[2] if len(h[1]) else 0.0) + sign * np.maximum(0.0, X[:, list(h[3])] @ h[4])
                    out = {"bic": h[0], "sse": h[5], "lin": dict(zip([names[j] for j in h[1]], map(float, h[2]))),
                           "hinge": (side, dict(zip([names[j] for j in h[3]], map(float, h[4])))), "res": yy - pred}
        rich = S in rich_ok if rich is None else rich
        if (rich and out["sse"] > 100 * floor[S] and S not in final and "in" in _sides(S, declared, bd)
                and f"__s_{S}" not in exclude):
            plain = out
            out = _hidden(S, yy, cols, _schedule(S, yy, cols, plain), plain)
            out = _with_delay(S, yy, cols, plain, out)
        if self_k is not None:
            out["lin"][S] = out["lin"].get(S, 0.0) + self_k
            out["bic"] += logT
        return out

    delay_memo: dict = {}

    def _with_delay(S, yy, cols, plain, best):
        """(312, delayed.py) the row with its inflow through a hidden first-order delay (an order max(0, C . x) placed
        into a stock nobody measures, a H of it arriving a step), beside the always-on part and a floored outflow,
        where it beats the plain row by PRICE log T and the row's other forms (a schedule, an integrator)."""
        if time.monotonic() > deadline or best["sse"] <= 100 * floor[S]:
            return best                                      # (a schedule or an integrator already at the data's rounding)
        # it reads levels, inputs and a constant only (no derived column), so it is fitted once for each row's change
        b_cols = [j for j, n in enumerate(names) if not str(n).startswith("__")]
        a_cols = [j for j in cols if j in b_cols]
        key = (S, hash(yy.tobytes()), tuple(a_cols))
        if key not in delay_memo:
            out_side = "out" in _sides(S, declared, bd)

            def floored(z, a_cols, keep=None):
                keep = np.ones(len(z), dtype=bool) if keep is None else keep
                h = _floored(X[keep], z[keep], -1.0, floor[S], deadline, a_cols, b_cols, (1.0, 4.0))
                return None if h is None else (h[1], h[2], h[3], h[4])
            delay_memo[key] = _delayed.fit(X, yy, floor[S], min(deadline, slice_end[0]), a_cols, b_cols, floored, out_side)
        got = delay_memo[key]
        if got is None:
            return best
        b = T * math.log((got["sse"] + floor[S]) / T) + got["k"] * logT
        if opts.get("rows_trace"):
            print("rows delayed", S, round(b, 1), "plain", round(plain["bic"], 1), "best", round(best["bic"], 1), got["a"],
                  [names[j] for j in got["C"]], got["c"], [names[j] for j in got["B"]], got["b"],
                  [names[j] for j in got["D"]], got["d"])
        if not (b + PRICE * logT < plain["bic"] and b < best["bic"]):
            return best
        o = (got["A"], got["lin"], got["B"], got["b"], got["D"], got["d"])
        pred = _delayed.predict(X, got["a"], got["C"], got["c"], got["h0"], o)
        inflow = _delayed.inflow(X, got["a"], got["C"], got["c"], got["h0"])
        return {"bic": b, "sse": got["sse"], "lin": dict(zip([names[j] for j in got["A"]], map(float, got["lin"]))),
                "hinge": ("out", dict(zip([names[j] for j in got["B"]], map(float, got["b"])))) if got["B"] else None,
                "res": yy - pred,
                "delay": {"a": got["a"], "order": dict(zip([names[j] for j in got["C"]], map(float, got["c"]))),
                          "h0": got["h0"], "inflow": inflow,
                          "second": dict(zip([names[j] for j in got["D"]], map(float, got["d"]))),
                          "levels": _delayed.hidden_levels(X, got["a"], got["C"], got["c"], got["h0"])}}

    qs: dict = {}
    derived_meta: dict[str, dict] = {}                     # (310, derived.py) columns beyond the levels

    def _schedule(S, yy, cols, out):
        """(310, derived.py) the row with one constant per block of P steps as fixed effects, where that wins by BIC."""
        best = out
        cs = [j for j in cols if names[j] is not None]
        for P in _derived.PIECES:
            if T // P < 3 or time.monotonic() > deadline:
                continue
            Q = qs.get(P)
            if Q is None:
                Q = qs[P] = _derived.basis(T, P)
            Xd, yd = _derived.demean(X[:, cs], Q), _derived.demean(yy, Q)
            keep = [k for k in range(len(cs)) if np.ptp(Xd[:, k]) > 1e-9 * (np.ptp(X[:, cs[k]]) + 1e-300)]
            lin = _sparse(Xd, yd, floor[S], keep)
            sup = [cs[k] for k in lin[1]]
            base = X[:, sup] @ lin[2] if sup else np.zeros(T)
            sch = _derived.fitted(yy - base, Q)
            res = yy - base - sch
            sse = float(res @ res)
            b = T * math.log((sse + floor[S]) / T) + (len(sup) + Q.shape[1]) * logT
            if b + PRICE * logT < best["bic"]:
                idx = _derived.blocks(T, P)
                best = {"bic": b, "sse": sse, "lin": {**dict(zip([names[j] for j in sup], map(float, lin[2]))), f"__s_{S}": 1.0},
                        "hinge": None, "res": res, "sched": sch, "sched_idx": idx, "sched_q": Q}
        return best

    def _with(Z, S, yy, cs):
        """(bic before the extra numbers, support, coefs, fitted extra part, residual) of yy on readers cs plus
        the columns Z (fixed effects: the readers chosen on what Z leaves)."""
        U, sv, _ = np.linalg.svd(Z, full_matrices=False)
        Q = U[:, sv > 1e-10 * sv[0]]
        Xd, yd = _derived.demean(X[:, cs], Q), _derived.demean(yy, Q)
        keep = [k for k in range(len(cs)) if np.ptp(Xd[:, k]) > 1e-9 * (np.ptp(X[:, cs[k]]) + 1e-300)]
        lin = _sparse(Xd, yd, floor[S], keep)
        sup = [cs[k] for k in lin[1]]
        base = X[:, sup] @ lin[2] if sup else np.zeros(T)
        extra = _derived.fitted(yy - base, Q)
        res = yy - base - extra
        return sup, lin[2], extra, res, Q

    def _hidden(S, yy, cols, best, plain):
        """(310, derived.py 3) the row reading a hidden integrator (constant inflow, a drain at a constant rate or
        switched by a derived switch), written out as a schedule, where it wins by BIC over the plain row."""
        cs = [j for j in cols if names[j] is not None]
        switches = [np.ones(T)] + [m["switch"] for m in derived_meta.values() if m.get("switch") is not None]
        top = None
        for sw in switches:
            if time.monotonic() > deadline:
                break

            def sse_at(a, sw=sw):
                h0, h1 = _derived.integrator(a, sw)
                Z = np.vstack([h0, h1, np.ones(T)]).T
                sup, coef, extra, res, Q = _with(Z, S, yy, cs)
                return float(res @ res), (sup, coef, extra, res, Q)

            grid = _derived.decays(sw)
            vals = [sse_at(a)[0] for a in grid]
            k = int(np.argmin(vals))
            a = grid[k]
            if 0 < k < len(grid) - 1:
                a, _ = _derived.golden(lambda x: sse_at(x)[0], grid[k - 1], grid[k + 1])
            sse, (sup, coef, extra, res, Q) = sse_at(a)
            b = T * math.log((sse + floor[S]) / T) + (len(sup) + 4) * logT
            if top is None or b < top[0]:
                top = (b, sse, sup, coef, extra, res, Q, a, sw)
        if top is None or not top[0] + PRICE * logT < plain["bic"] or not top[0] < best["bic"]:
            return best
        b, sse, sup, coef, extra, res, Q, a, sw = top
        sw_meta = next((m for m in derived_meta.values() if m.get("switch") is sw), None)
        if sw_meta is not None and sw_meta.get("hinge"):
            # (311) the switch's threshold refitted on this row with the drain rate (the readers held): the integrator
            # accumulates a threshold's error step after step, so its own row reads the threshold far more sharply than
            # the rows of the switched flow (kalman_lna 311 none-short: 479.994 for 480 left 80 x the truth's misfit)
            got = _integrator_threshold(S, yy, cs, sup, a, sw_meta)
            if got is not None and got[0] < sse:
                a, sw_meta = got[1], got[2]
                sw = sw_meta["switch"]
                h0, h1 = _derived.integrator(a, sw)
                sup, coef, extra, res, Q = _with(np.vstack([h0, h1, np.ones(T)]).T, S, yy, cs)
                sse = float(res @ res)
                b = T * math.log((sse + floor[S]) / T) + (len(sup) + 4) * logT
        P = _derived.PIECES[0]
        out = {"bic": b, "sse": sse, "lin": {**dict(zip([names[j] for j in sup], map(float, coef))), f"__s_{S}": 1.0},
               "hinge": None, "res": res, "sched": extra, "sched_idx": _derived.blocks(T, P), "sched_q": Q,
               "hidden": True}
        # (310, round 2) the integrator itself, so it is written as a hidden stock H (h' = 1 - a s H) the row reads,
        # not as a schedule: extra = p0 h0 + p1 h1 + p2 = p1 H + p2 with H's start p0 / p1
        h0, h1 = _derived.integrator(a, sw)
        p, *_ = np.linalg.lstsq(np.vstack([h0, h1, np.ones(T)]).T, extra, rcond=None)
        if abs(p[1]) > 1e-12:
            out["integrator"] = {"a": float(a), "switch": sw_meta, "p": [float(x) for x in p], "h": h1 + (p[0] / p[1]) * h0}
        return out

    def _integrator_threshold(S, yy, cs, sup, a, meta):
        """(sse, a, meta) with the switch's threshold and the drain rate refitted together on S's row, its readers sup
        held (coordinate descent, golden section on each); None where nothing moves."""
        ia, ib, up, theta = meta["hinge"]
        yb = X[:, ib]
        R0 = X[:, sup] if sup else np.zeros((T, 0))

        def sw_of(th):
            return np.maximum(0.0, yb - th) if up else np.maximum(0.0, th - yb)

        def f(a_, th):
            h0, h1 = _derived.integrator(a_, sw_of(th))
            return _lstsq(np.hstack([R0, np.vstack([h0, h1, np.ones(T)]).T]), yy)[1]

        span = 0.01 * float(np.ptp(yb)) or 1.0
        best = (f(a, theta), a, theta)
        for _ in range(INTEGRATOR_ROUNDS):
            th, v = _derived.golden(lambda z: f(best[1], z), best[2] - span, best[2] + span)
            if v < best[0]:
                best = (v, best[1], th)
            if best[1] > 0:
                a2, v = _derived.golden(lambda z: f(z, best[2]), best[1] / 1.5, best[1] * 1.5)
                if v < best[0]:
                    best = (v, a2, best[2])
            span /= 3.0
        if best[2] == theta and best[1] == a:
            return None
        th = best[2]
        expr = _derived.hinge_expr(names[ia], names[ib], th, up)
        return best[0], best[1], {**meta, "expr": expr, "hinge": (ia, ib, up, th), "switch": sw_of(th)}

    def _ratio_hidden(resid, taken, stock_cols):
        """(310) a rate per unit of another stock and a hidden integrator are found together: on every row with an
        inflow side whose row is not at rounding, each ratio column is scored with the row refitted on its readers,
        the column and the best hidden integrator (derived.py 3); (gain, meta, column) of the best, or None."""
        best = None
        switches = [np.ones(T)] + [m["switch"] for m in derived_meta.values() if m.get("switch") is not None]
        grid = [(w, a) for w, sw in enumerate(switches) for a in _derived.decays(sw)]
        Hs = [_derived.integrator(a, switches[w]) for w, a in grid]
        for S in resid:
            if S in final or "in" not in _sides(S, declared, bd) or time.monotonic() > deadline:
                continue
            r = rows[S]
            y = dy[ids.index(S)]
            own = [n for n in r["lin"] if n in names and n is not None]
            sup = [X[:, names.index(n)] for n in own + ([S] if S in names and S not in own else [])]   # and its own level
            scan = []
            for ia in stock_cols:
                for ib in stock_cols:
                    if ia == ib:
                        continue
                    e = _derived.ratio_expr(names[ia], names[ib])
                    if e in taken:
                        continue
                    col = X[:, ia] / np.maximum(1.0, X[:, ib])
                    sses = [_lstsq(np.vstack(sup + [col, h0, h1, np.ones(T)]).T, y)[1] for h0, h1 in Hs]
                    k = int(np.argmin(sses))
                    scan.append((sses[k], k, ia, ib, e, col))
            for sse0, k, ia, ib, e, col in sorted(scan, key=lambda c: c[0])[:3]:     # the drain rate refined
                w, a = grid[k]
                same = [g for g in range(len(grid)) if grid[g][0] == w]
                pos = same.index(k)
                lo, hi = grid[same[max(0, pos - 1)]][1], grid[same[min(len(same) - 1, pos + 1)]][1]

                def f(x, col=col, w=w):
                    h0, h1 = _derived.integrator(x, switches[w])
                    return _lstsq(np.vstack(sup + [col, h0, h1, np.ones(T)]).T, y)[1]

                top = min(sse0, _derived.golden(f, lo, hi)[1]) if hi > lo else sse0
                g = T * math.log((r["sse"] + floor[S]) / (top + floor[S])) - 5 * logT
                if best is None or g > best[0]:
                    best = (g, {"expr": e, "reads": [names[ia], names[ib]], "base": names[ia], "row": S}, col)
        return best

    cap_fits: dict = {}

    def _capped(resid, spans, taken):
        """(311, derived.py 4) a move X -> T capped by the room other levels leave, min(a X, max(0, B . z)): for every
        pair of same-unit stocks whose rows are not at rounding, and every room of up to CAP_READERS levels (with or
        without a constant), the cap is read off max(what T gains, what X loses) as a lower bound of the move, and
        scored on what it adds to both rows beyond their span (the source must lose it, the receiver gain it), net of
        its numbers; the CAP_TRY best as [(gain, meta, column)] (the caller judges them by refitting their rows)."""
        found = []
        one = names.index(None) if None in names else None
        if one is None:
            return []

        def consider(Sx, St, ix, x, zcols, a, b):
            Z = X[:, zcols]
            col = _derived.cap_column(x, Z, a, b)
            g = 0.0
            for S, sgn in ((Sx, -1.0), (St, 1.0)):
                if S not in resid:
                    continue
                Q = spans.get(S)
                c = _derived.demean(col, Q) if Q is not None else col
                rr = _derived.demean(resid[S], Q) if Q is not None else resid[S]
                if float(c @ rr) * sgn <= 0:
                    return                                  # the source must lose it and the receiver gain it
                g += _derived._gain(c, rr, float(rr @ rr), floor[S], T)
            g -= (len(zcols) + 1) * logT
            e = _derived.cap_expr(Sx, a, [(names[j], bb) for j, bb in zip(zcols, b)])
            if e in taken or not math.isfinite(g):
                return
            if opts.get("rows_trace_cap") and g > -1e9:
                print("cap cand", Sx, St, round(g, 1), e)
            found.append((g, {"expr": e, "reads": [Sx] + [names[j] for j in zcols if names[j] is not None], "base": Sx,
                              "row": St, "room": [names[j] for j in zcols],
                              "cap": (ix, zcols, float(a), np.asarray(b, dtype=float))}, col))

        def seeds(Sx, St):
            """(316) a cap read off a row's floored term that reads X with a negative number: the receiver's change
            T' = ... - max(0, room - a X) is T' = ... - room + min(a X, room), so min(a X, max(0, room)) is the move,
            with the room exactly what the floored term reads besides X (no lower bound to fit it to)."""
            out = []
            for S in (St, Sx):
                h = (rows.get(S) or {}).get("hinge")
                if not h or h[1].get(Sx, 0.0) >= 0:
                    continue
                room = [(n, k) for n, k in h[1].items() if n != Sx]
                if not room or any(n not in names or (n is not None and n not in ids) for n, _ in room):
                    continue
                out.append((-float(h[1][Sx]), [names.index(n) for n, _ in room], np.array([k for _, k in room], dtype=float)))
            if unit.get(Sx) != unit.get(St) and St in names:
                got = _pool_seed(Sx, St)
                if got is not None:
                    out.append(got)
            return out

        def _pool_seed(Sx, St):
            """(316) across units, where counting noise hides the floored term: a pool's room, min(a X, max(0, c - b T)),
            a constant less a share of the receiver's own level (K slots, each one held takes b of them; 311's room read
            its receiver too), fitted on both rows at once with each row's other readers profiled out (least squares,
            each row weighted by its misfit) over a grid of a and b, c by golden section, then polished."""
            key = ("pool", Sx, St)
            if key in cap_fits:
                return cap_fits[key]
            cap_fits[key] = None
            from scipy.optimize import least_squares

            ix, it = names.index(Sx), names.index(St)
            x, tl = X[:, ix], X[:, it]
            design = {}
            for S in (Sx, St):
                r = rows.get(S) or {}
                use = {n for n in r.get("lin", {}) if n in names and not str(n).startswith("__")}
                if r.get("hinge"):
                    use |= {n for n in r["hinge"][1] if n in names}
                use |= {S, None}
                cols = [names.index(n) for n in sorted(use, key=lambda n: (n is None, str(n)))]
                design[S] = (X[:, cols], dy[ids.index(S)], 1.0 / math.sqrt(max(r.get("sse", 1.0), floor[S])))

            def resid_of(th):
                col = _derived.cap_column(x, np.column_stack([tl, np.ones(T)]), th[0], np.array([-th[1], th[2]]))
                out = []
                for S, sgn in ((Sx, -1.0), (St, 1.0)):
                    A, y, w = design[S]
                    yy = y - sgn * col
                    k = np.linalg.lstsq(A, yy, rcond=None)[0]
                    out.append(w * (yy - A @ k))
                return np.concatenate(out)

            def sse(th):
                r = resid_of(th)
                return float(r @ r)

            best = None
            pos = x > 1e-9
            if int(pos.sum()) < 2 * MIN_ON or time.monotonic() > deadline:
                return None
            for a in np.linspace(0.1, 1.0, 10):
                for b in (0.0, 0.1, 0.25, 0.5, 1.0):
                    v = a * x + b * tl                       # the room binds where c is below this
                    grid = np.unique(np.quantile(v[pos], np.linspace(0.05, 0.95, 19)))
                    c, s = _golden_min(lambda c: sse((a, b, c)), grid)
                    if best is None or s < best[0]:
                        best = (s, np.array([a, b, c]))
            try:
                got = least_squares(resid_of, best[1], method="lm", x_scale="jac", max_nfev=400)
                th = got.x if float(got.fun @ got.fun) < best[0] else best[1]
            except Exception:                              # noqa: BLE001 (a failed polish keeps the grid's numbers)
                th = best[1]
            a, b, c = (float(v) for v in th)
            room = a * x - (c - b * tl)
            if a <= 0 or int(np.sum(room > 0)) < MIN_ON or int(np.sum(room <= 0)) < MIN_ON:
                return None                                  # both regimes act, or it is a linear rate
            cap_fits[key] = (a, [it, names.index(None)], np.array([-b, c]))
            return cap_fits[key]
        cand = [S for S in resid if S not in final and S in names]
        # (316) a receiver across units (movable) may be a stock whose row is already read (a floored inflow that is
        # the cap's other side): only the source's gain scores it then, and the judge below refits both rows
        recv = cand + [S for S in rows if S not in resid and S not in final and S in names]
        for Sx in cand:
            ix = names.index(Sx)
            x = X[:, ix]
            for St in recv:
                if (St == Sx or not movable(Sx, St) or (St not in resid and unit.get(St) == unit.get(Sx))
                        or time.monotonic() > deadline):
                    continue
                # what T gains or X loses, whichever is more: the move's amount where T has no other outflow or X no
                # other inflow on the step (a lower bound elsewhere, fitted as one)
                targets = [np.maximum(0.0, np.maximum(dy[ids.index(St)], -dy[ids.index(Sx)]))]
                for a, zcols, b in seeds(Sx, St):
                    consider(Sx, St, ix, x, zcols, a, b)
                if unit.get(St) != unit.get(Sx):
                    continue                               # (316) across units: the seeds only (the room search over
                                                           # every pair doubled counts-short's time, 436 -> 1015 s)
                zs = [j for j in stock_cols if j != ix]
                for k in range(1, _derived.CAP_READERS + 1):
                    for sub, const in itertools.product(itertools.combinations(zs, k), (False, True)):
                        zcols = list(sub) + [one] * const          # the room with or without a constant
                        Z = X[:, zcols]
                        for n0, f0 in enumerate(targets):
                            key = (Sx, St, tuple(zcols), n0)
                            if key not in cap_fits:              # the fit reads only the data: once for every round
                                cap_fits[key] = _derived.capfit(f0, x, Z, MIN_ON, lower=True)
                            got = cap_fits[key]
                            if got is None:
                                continue
                            consider(Sx, St, ix, x, zcols, got[1], got[2])
        found.sort(key=lambda c: -c[0])
        return found[:_derived.CAP_TRY]

    if offered and offered.get("source") in names and offered.get("cap") in names:
        # (312) the limit capmin offered the fit, MIN(X, theta Y), is a capped move the rows may read (min(1 X, max(0,
        # theta Y))): the strategy's own search may leave it unread (312 none-short: queue -> inflight read 2.06 nodes),
        # and a limit that binds on a few steps only is not one the capped search below reads off the rows' residuals
        ix, iy = names.index(offered["source"]), names.index(offered["cap"])
        th = float(offered["theta"])
        X = np.hstack([X, _derived.cap_column(X[:, ix], X[:, [iy]], 1.0, np.array([th]))[:, None]])
        derived_meta[f"__d{len(derived_meta)}"] = {
            "expr": _derived.cap_expr(names[ix], 1.0, [(names[iy], th)]), "reads": [names[ix], names[iy]],
            "base": names[ix], "row": offered.get("to"), "room": [names[iy]], "cap": (ix, [iy], 1.0, np.array([th])),
            "offered": True}
        names = names + [f"__d{len(derived_meta) - 1}"]
    rows, old = {}, {}
    for i, S in enumerate(ids):
        if np.ptp(read[i]) <= 0 and not np.any(dy[i]):
            continue
        a = answer(S)
        if a is None:
            continue
        old[S] = a
        rows[S] = row(S, dy[i], rich=False)
    def _span(r):
        """An orthonormal basis of what row r already reads (its readers, its floored term, its schedule): a
        new column is scored on what it adds beyond them."""
        cols = [X[:, names.index(n)] for n in r["lin"] if n in names]
        if r.get("hinge"):
            side, hb = r["hinge"]
            z = sum(k * X[:, names.index(n)] for n, k in hb.items())
            cols += [X[:, names.index(n)] * (z > 0) for n in hb]
        A = np.vstack(cols).T if cols else np.zeros((T, 0))
        if r.get("sched_q") is not None:
            A = np.hstack([A, r["sched_q"]])
        if A.shape[1] == 0:
            return None
        U, sv, _ = np.linalg.svd(A, full_matrices=False)
        return U[:, sv > 1e-10 * sv[0]]

    stock_cols = [j for j, n in enumerate(names) if n in set(ids)]
    t_der = time.monotonic()
    if opts.get("rows_trace"):
        print("rows t base", round(t_der - deadline + TIME, 1))
    if time.monotonic() > deadline:
        return result                                        # the plain rows themselves ran out of time
    # (317) a larger model may need more derived columns: ROUNDS (and one capped move) for every 12 stocks (317's three
    # services each have an OOM switch, a per-pod memory ratio and a pod start capped by the room); the rounds leave
    # RICH_SHARE of the pass to the richer row forms after them
    per = max(1, len(ids) // 12)
    rounds, caps = _derived.ROUNDS * per, per
    full_deadline, deadline = deadline, deadline - RICH_SHARE * TIME
    for _ in range(rounds + caps):                           # (311) a capped move is one round beyond ROUNDS
        if time.monotonic() > deadline:
            break
        before = (X, names, rows, dict(derived_meta))       # a round cut short by the clock is undone whole
        resid = {S: r["res"] for S, r in rows.items() if r["sse"] > 100 * floor[S]}
        if not resid:
            break
        spans = {S: _span(rows[S]) for S in resid}
        got = joint = None
        if sum(not m.get("cap") for m in derived_meta.values()) < rounds:
            got = _derived.search(X, names, stock_cols, resid, floor, spans, T, {m["expr"] for m in derived_meta.values()},
                                  [m["hinge"] for m in derived_meta.values() if m.get("hinge")])
            joint = _ratio_hidden(resid, {m["expr"] for m in derived_meta.values()}, stock_cols)
        capped = None
        if sum(bool(m.get("cap") and not m.get("offered")) for m in derived_meta.values()) < caps:
            # (311) the few best caps by what they add beyond the rows' spans, each then judged by its two rows refitted
            # with it (a cap lets a row's floored term be read right, which no linear score sees), net of its numbers
            for g0, meta_c, col_c in _capped(resid, spans, {m["expr"] for m in derived_meta.values()}):
                if time.monotonic() > deadline:
                    break
                X0, names0 = X, names
                X, names = np.hstack([X0, col_c[:, None]]), names0 + ["__try"]
                pair = (meta_c["base"], meta_c["row"])
                g = sum(rows[S]["bic"] - row(S, dy[ids.index(S)])["bic"] for S in pair) - len(meta_c["cap"][1]) * logT
                X, names = X0, names0
                if opts.get("rows_trace"):
                    print("rows cap tried", round(g0, 1), round(g, 1), meta_c["expr"])
                if capped is None or g > capped[0]:
                    capped = (g, meta_c, col_c)
        if opts.get("rows_trace"):
            print("rows joint", None if joint is None else (round(joint[0], 1), joint[1]["expr"], joint[1]["row"]),
                  None if got is None else round(got[0], 1),
                  "capped", None if capped is None else (round(capped[0], 1), capped[1]["expr"]))
        if joint is not None and (got is None or joint[0] > got[0]):
            got = joint
        if capped is not None and (got is None or capped[0] > got[0]):
            got = capped
        if got is None or got[0] < PRICE * logT:
            break
        name = f"__d{len(derived_meta)}"
        derived_meta[name] = got[1]
        X = np.hstack([X, got[2][:, None]])
        names = names + [name]
        if opts.get("rows_trace"):
            print("rows derived", name, got[1], round(got[0], 1))
        rows = {S: row(S, dy[ids.index(S)]) for S in rows}
        for _ in range(_derived.CAP_REFITS if derived_meta[name].get("cap") else 2):   # a switch's threshold (a cap's
            # numbers) refitted on the rows reading it, the rows refitted after each
            j = names.index(name)
            parts = {S: r["res"] + r["lin"][name] * X[:, j] for S, r in rows.items() if r["lin"].get(name)}
            if derived_meta[name].get("cap"):
                meta, col = _derived.refine_cap(X, derived_meta[name], parts,
                                                {S: r["lin"][name] for S, r in rows.items() if r["lin"].get(name)})
            else:
                # (311) each row weighted by how well it fits: the clean row (a counter that reads only the switch)
                # fixes the threshold, rows whose other flows are not read yet only blur it (480.00 -> 480.27 on 311)
                meta, col = _derived.refine(X, names, derived_meta[name], parts,
                                            {S: 1.0 / max(rows[S]["sse"], floor[S]) for S in parts})
            if col is None:
                break
            keep = (derived_meta[name], X[:, j].copy(), rows)
            derived_meta[name] = meta
            X[:, j] = col
            rows = {S: row(S, dy[ids.index(S)]) for S in rows}
            if meta.get("cap") and sum(r["bic"] for r in rows.values()) > sum(r["bic"] for r in keep[2].values()):
                derived_meta[name], X[:, j], rows = keep[0], keep[1], keep[2]     # (311) a refit the rows lose by: undone
                break
        m = derived_meta[name]
        if m.get("cap") and None in m["room"] and len(m["room"]) > 1 and time.monotonic() < deadline:
            # (311) the room without its constant, kept where the rows lose less than the number it saves (on a short
            # run the nodes' level barely moves where the room binds, so a constant and the nodes' number trade off)
            ix, zcols, a_c, b_c = m["cap"]
            k0 = m["room"].index(None)
            zc = [z for n_, z in enumerate(zcols) if n_ != k0]
            bc = np.array([v for n_, v in enumerate(b_c) if n_ != k0])
            bc = np.linalg.lstsq(X[:, zc], X[:, zcols] @ b_c, rcond=None)[0]   # the same room as near as it goes
            trial = {**m, "room": [r_ for r_ in m["room"] if r_ is not None],
                     "cap": (ix, zc, a_c, bc), "expr": _derived.cap_expr(m["reads"][0], a_c, list(zip([r_ for r_ in m["room"] if r_ is not None], bc)))}
            j = names.index(name)
            parts = {S: r["res"] + r["lin"][name] * X[:, j] for S, r in rows.items() if r["lin"].get(name)}
            got_t = _derived.refine_cap(X, trial, parts, {S: r["lin"][name] for S, r in rows.items() if r["lin"].get(name)})
            trial, col = (got_t[0], got_t[1]) if got_t[1] is not None else (trial, _derived.cap_column(X[:, ix], X[:, zc], a_c, bc))
            keep = (m, X[:, j].copy(), rows)
            derived_meta[name], X[:, j] = trial, col
            rows = {S: row(S, dy[ids.index(S)]) for S in rows}
            if sum(r["bic"] for r in rows.values()) > sum(r["bic"] for r in keep[2].values()) + logT:
                derived_meta[name], X[:, j], rows = keep[0], keep[1], keep[2]
        if opts.get("rows_trace"):
            print("rows derived refined", name, derived_meta[name]["expr"], "t", round(time.monotonic() - t_der, 1))
        if time.monotonic() > deadline:
            X, names, rows, derived_meta = before
            if opts.get("rows_trace"):
                print("rows derived round undone: out of time")
            break
    deadline = full_deadline
    # (317) every stock's plain row first (above), the derived columns on them, then the richer forms (a schedule, a hidden integrator, a delayed inflow) for
    # the rows that miss most, each delay fit on its slice of RICH_SHARE of the pass: on 317 (36 stocks) the delay fits
    # of the first rows took the whole 600 s, so the rows of the rest were never read and nothing was rebuilt
    rich_end = deadline
    todo = sorted((S for S in rows if rows[S]["sse"] > 100 * floor[S] and S not in final and "in" in _sides(S, declared, bd)),
                  key=lambda S: -rows[S]["sse"] / floor[S])
    rich_ok |= set(rows) - set(todo)                     # (nothing richer is tried on them)
    for n_, S in enumerate(todo):
        if time.monotonic() > rich_end:
            break
        slice_end[0] = time.monotonic() + max(RICH_SLICE, (rich_end - time.monotonic()) / (len(todo) - n_))
        rich_ok.add(S)
        rows[S] = row(S, dy[ids.index(S)])
    slice_end[0] = math.inf
    # the rebuild has its own budget after the derived search, so a slower machine loses derived columns, never the
    # rebuild of the plain rows (309 counts-long on node2: the search used the whole pass and the answer was kept)
    deadline = time.monotonic() + BUILD_TIME
    if opts.get("rows_trace"):
        for S, r in rows.items():
            print("rows", S, round(r["bic"], 1), round(bic(*old[S], S), 1), r["sse"], old[S], r["lin"], r["hinge"])
    def margin(S, r):
        """(316) PRICE log T; SIMPLER log T where the row is plain and reads fewer numbers than the answer spends on S
        (a simpler row that also fits better: counts-long's one-step rows beat oefit's move-plus-compensating-inflow
        on frontend_inflight by 43 and its bent link on pricing_queue by 28, under 6 log T = 46)."""
        plain = r.get("sched") is None and r.get("delay") is None and r.get("integrator") is None
        k = len(r["lin"]) + (len(r["hinge"][1]) if r.get("hinge") else 0)
        return (SIMPLER if plain and k < old[S][1] else PRICE) * logT

    wins = {S for S, r in rows.items()
            if r["bic"] + margin(S, r) < bic(*old[S], S) and r["sse"] <= (1 - EXPLAINED) * float(dy[ids.index(S)] @ dy[ids.index(S)])}
    if not wins:
        return result
    B = set(wins)
    fine = {S for S in old if old[S][0] <= (1 - EXPLAINED) * float(dy[ids.index(S)] @ dy[ids.index(S)])}
    while True:                                              # a stock joined to a rebuilt one by a flow joins
        grow = set()
        for n, f in enumerate(flows):
            a, b = f.get("from"), f.get("to")
            if a is None or b is None or (a in B) == (b in B):
                continue
            X_ = b if a in B else a
            if X_ in rows and rows[X_]["bic"] <= bic(*old[X_], X_) + PRICE * logT:
                grow.add(X_)
        for f in flows:                                      # (regress-kalman_lna) a co-flow is rebuilt whole or kept
            group = _co_group(f)                             # whole: its ends and the stocks it draws on join together
            if group & B and not group <= B:
                if not (group - B) <= set(rows):
                    return result                            # one of them has no row to rebuild it from
                grow |= group - B
        if not grow:
            break
        B |= grow
    hidden_lv: dict = {}
    for _ in range(len(ids) + 1):
        hidden_lv.clear()
        built = _build(B, flows, amts, rows, ids, dy, X, names, unit, final, declared, bd, row, T, fine, floor,
                       derived_meta, t_end, hidden_lv, movable)
        if built is None:
            return result
        new_flows, bad = built
        if opts.get("rows_trace"):
            print("rows build", sorted(B), "bad", sorted(bad))
        if not bad:
            break
        B -= bad
        if not B:
            return result
    else:
        return result
    full = {H[len("__full_"):]: hidden_lv.pop(H) for H in [k for k in hidden_lv if k.startswith("__full_")]}
    if hidden_lv:                                            # (310 round 2) the rows' hidden stocks at each step's start
        states = [{**st, **{H: float(h[k]) for H, h in hidden_lv.items()}} for k, st in enumerate(states)]
    new_amts = [_amounts(f, states, t_end) for f in new_flows]
    tot_old = tot_new = 0.0
    for S in B:
        i = ids.index(S)
        pred, k = np.zeros(T), 0
        for n, f in enumerate(new_flows):
            if S not in (f.get("from"), f.get("to")):
                continue
            if new_amts[n] is None:
                return result
            pred += new_amts[n] if f.get("to") == S else -float(f.get("ratio") or 1) * new_amts[n]
            k += _numbers(f)
        tot_new += bic(float(((dy[i] - pred) ** 2).sum()), k, S)
        tot_old += bic(*old[S], S)
    if not tot_new + PRICE * logT < tot_old:
        return result
    out = copy.copy(result)
    out["flows"] = [_finish(f) for f in new_flows]
    if hidden_lv:                                            # their levels on the run (aligned with t), for the judge
        lv = dict(out.get("inferred_levels") or {})
        for H, h in hidden_lv.items():
            if H in full:                                    # (312) a delayed inflow's stock: its levels as fitted
                lv[H] = [float(v) for v in full[H]]
                continue
            f = next((g for g in out["flows"] if g.get("from") == H), None)
            drain = _amounts(f, states, t_end) if f is not None and not f.get("effects") else None
            if f is not None and drain is None:
                rate = float(f["rate"]["k"])
                e = f["effects"][0]
                (x0, y0), (x1, y1) = e["points"]
                xs = np.array([st[e["stock"]] for st in states])
                drain = rate * h * np.clip(y0 + (xs - x0) * (y1 - y0) / (x1 - x0), min(y0, y1), max(y0, y1))
            last = float(h[-1]) + 1.0 - (float(drain[-1]) if drain is not None else 0.0)
            lv[H] = [float(v) for v in h] + [last]
        out["inferred_levels"] = lv
    out["rows"] = {"rebuilt": sorted(B), "bic": [tot_old, tot_new],
                   **({"derived": [m["expr"] for m in derived_meta.values()]} if derived_meta else {})}
    return out


def _rate(terms: dict[str | None, float]) -> dict:
    c = float(terms.get(None, 0.0))
    links = [{"stock": s, "k": float(k)} for s, k in terms.items() if s is not None]
    if not links:
        return {"form": "constant", "value": c}
    if len(links) == 1 and not c:
        return {"form": "proportional", "stock": links[0]["stock"], "k": links[0]["k"]}
    return {"form": "linear", "c": c, "terms": links}


def _flow(a, b, terms, ctx: dict | None = None) -> dict:
    ctx = ctx or {}
    meta, sched = ctx.get("derived") or {}, ctx.get("sched") or {}
    if not any(s is not None and str(s).startswith("__") for s in terms):
        r = _rate(terms)
        return {"from": a, "to": b, "ratio": 1, "rate": r, "rates": [dict(r)], "id": f"rows_{a}_{b}"}
    # (310, derived.py) a rate reading a derived column is an expression; one carrying a schedule, one per block
    c = float(terms.get(None, 0.0))
    only = [s_ for s_ in terms if s_ is not None]
    if not c and len(only) == 1 and (meta.get(only[0]) or {}).get("cap") and terms[only[0]] > 0:
        # (311) k min(a X, max(0, room)) = min(k a X, max(0, k room)): the capped move written as one MIN
        m, k = meta[only[0]], float(terms[only[0]])
        _, _, ca, cb = m["cap"]
        e = _derived.cap_expr(m["reads"][0], k * ca, [(z, k * bb) for z, bb in zip(m["room"], cb)])
        r = {"form": "expr", "expr": e}
        return {"from": a, "to": b, "ratio": 1, "rate": r, "rates": [dict(r)], "id": f"rows_{a}_{b}",
                "numbers": len(cb) + 1}
    parts = []
    for s_, k in terms.items():
        if s_ is None or str(s_).startswith("__s_"):
            continue
        parts.append(f"{float(k)!r} * ({meta[s_]['expr']})" if s_ in meta else f"{float(k)!r} * {s_}")
    sname = next((s_ for s_ in terms if s_ is not None and str(s_).startswith("__s_")), None)
    numbers = len(parts) + (1 if c else 0)
    if sname is not None and sname in (ctx.get("hid") or {}):
        # (310 round 2) the row reads its hidden stock H: m (p1 H + p2), H's own flows written by _build
        p = ctx["hid"][sname][1]["p"]
        m = float(terms[sname])
        r = {"form": "expr", "expr": " + ".join([repr(c + m * p[2]), f"{m * p[1]!r} * {ctx['hidden_names'][sname]}"] + parts)}
        return {"from": a, "to": b, "ratio": 1, "rate": r, "rates": [dict(r)], "id": f"rows_{a}_{b}", "numbers": numbers + 2}
    if sname is None:
        r = {"form": "expr", "expr": " + ".join([repr(c)] + parts)}
        out = {"from": a, "to": b, "ratio": 1, "rate": r, "rates": [dict(r)], "id": f"rows_{a}_{b}", "numbers": numbers}
        only = [s_ for s_ in terms if s_ is not None]
        if not c and len(only) == 1 and (meta.get(only[0]) or {}).get("hinge"):
            out["_effect"] = _effect_form(meta[only[0]], float(terms[only[0]]), ctx.get("names") or [], ctx.get("X"))
        return out
    series, idx = sched[sname]
    t_end = ctx["t_end"]
    segs = []
    for blk in range(int(idx.max()) + 1):
        at = np.flatnonzero(idx == blk)
        v = c + float(terms[sname]) * float(np.mean(series[at]))     # the block at the schedule's mean over it
        segs.append({"from": float(t_end[at[0]]), "to": float(t_end[at[-1]]),
                     "rate": {"form": "expr", "expr": " + ".join([repr(v)] + parts)}})
    segs[0]["from"] = float(t_end[0]) - 1.0
    r = dict(segs[-1]["rate"])
    return {"from": a, "to": b, "ratio": 1, "rate": r, "rates": [dict(r)], "byTime": segs, "id": f"rows_{a}_{b}",
            "numbers": numbers + len(segs)}


def _effect_form(m: dict, k: float, names: list, X) -> dict:
    """k X max(0, Y - theta) (or max(0, theta - Y)) as the engine's own form: a rate k X with an effect of Y, a
    table that is 0 past the threshold and rises by 1 a unit of Y beyond it (over ten times Y's range)."""
    ia, ib, up, theta = m["hinge"]
    span = 10.0 * float(np.ptp(X[:, ib])) + 1.0 if X is not None else 1e4
    pts = [[theta, 0.0], [theta + span, span]] if up else [[theta - span, span], [theta, 0.0]]
    r = {"form": "proportional", "stock": names[ia], "k": k}
    return {"rate": r, "rates": [dict(r)], "effects": [{"stock": names[ib], "shape": "table", "points": pts}]}


def _finish(f: dict) -> dict:
    """A rebuilt flow in its final form (310: a switch column as a rate with an effect)."""
    if "_effect" not in f:
        return f
    g = {k: v for k, v in f.items() if k not in ("_effect", "numbers")}
    g.update(f["_effect"])
    return g


def _pair(R, B, unit, final, meta=None, movable=None, closed=None):
    """Moves, outside links in and out, each stock's drain and what its moves leave of it. (310) A derived column
    with base X (derived.py: X x a switch on another stock) that X's row loses and T's gains, T in X's unit, is a
    move X -> T too; where two stocks gain it, the move goes to the one that is not a counter (final). (316) A capped
    column (derived.py 4) may cross units where movable(X, T) says the boundary leaves it no other way."""
    meta = meta or {}
    drain = {S: max(0.0, -R[S]["lin"].get(S, 0.0)) for S in B}
    moves, link_in, link_out = [], {S: {} for S in B}, {S: {} for S in B}
    pool = {(S, d): -k for S in B for d, k in R[S]["lin"].items() if d in meta and meta[d]["base"] == S and k < 0}
    cands, dcands = [], []
    for Tt in B:
        for x, k in R[Tt]["lin"].items():
            if x == Tt:
                continue
            if (Tt, x) in pool:
                continue                                     # what Tt loses in its own derived column: below
            base = meta[x]["base"] if x in meta else None
            if k > 0 and base is not None and (base, x) in pool and base != Tt and (
                    unit.get(base) == unit.get(Tt)
                    or (movable is not None and meta[x].get("cap") is not None and movable(base, Tt))):
                dcands.append((Tt in final, -k, base, Tt, x))
            elif k > 0 and x in B and unit.get(x) == unit.get(Tt) and x not in final:
                cands.append((k, x, Tt))
            elif k > 0:
                link_in[Tt][x] = k
            else:
                link_out[Tt][x] = -k
    left = dict(drain)
    for k, x, Tt in sorted(cands, key=lambda c: -c[0]):
        if k <= left[x] * (1 + TOL_MOVE) + 1e-12:
            moves.append((x, Tt, {x: k}))
            left[x] = max(0.0, left[x] - k)
        elif closed is not None and closed(Tt) and drain[x] > 0:
            # (316) T may take nothing from outside, so what it gains reading X is a move even where it is more than X's
            # drain (counting noise reads the two rows' ends apart: 0.623 against 0.607); _build refits X's row with
            # its drain fixed at what its moves take
            moves.append((x, Tt, {x: k}))
            left[x] = 0.0
        else:
            link_in[Tt][x] = k
    dleft = dict(pool)
    for _, mk, base, Tt, d in sorted(dcands):
        k = -mk
        if k <= dleft[(base, d)] * (1 + TOL_MOVE) + 1e-12:
            moves.append((base, Tt, {d: k}))
            dleft[(base, d)] = max(0.0, dleft[(base, d)] - k)
        else:
            link_in[Tt][d] = k
    for (S, d), v in dleft.items():
        if v > TOL_MOVE * pool[(S, d)]:                     # less is the rounding of the move's two ends
            link_out[S][d] = v
    return moves, link_in, link_out, drain, left


def _hidden_name(S: str, taken: set) -> str:
    h, n = f"{S}_hidden", 2
    while h in taken:
        h, n = f"{S}_hidden{n}", n + 1
    taken.add(h)
    return h


def _hidden_flows(H: str, integ: dict, names: list, X) -> list[dict]:
    """(310 round 2) a hidden integrator h' = 1 - a s h as a found model's own hidden stock H: an outside inflow of
    1 a step and a drain a H, switched (s = max(0, Y - theta) or max(0, theta - Y)) by an effect table on Y."""
    one = {"form": "constant", "value": 1.0}
    out = [{"from": None, "to": H, "ratio": 1, "rate": one, "rates": [dict(one)], "id": f"rows_in_{H}"}]
    if integ["a"] > 0:
        r = {"form": "proportional", "stock": H, "k": float(integ["a"])}
        f = {"from": H, "to": None, "ratio": 1, "rate": r, "rates": [dict(r)], "id": f"rows_{H}_out"}
        m = integ.get("switch")
        if m is not None:
            ia, ib, up, theta = m["hinge"]
            span = 10.0 * float(np.ptp(X[:, ib])) + 1.0
            pts = [[theta, 0.0], [theta + span, span]] if up else [[theta - span, span], [theta, 0.0]]
            f["effects"] = [{"stock": names[ib], "shape": "table", "points": pts}]
        out.append(f)
    return out


def _lin_expr(terms: dict) -> str:
    return " + ".join(f"{float(k)!r}" if n is None else f"{float(k)!r} * {n}" for n, k in terms.items()) or "0.0"


def _two_floors(f: dict, first: dict, second: dict) -> dict:
    """(312) an outflow floored twice, max(0, min(B . z, D . z)), as an expression (its numbers: B's and D's)."""
    r = {"form": "expr", "expr": f"max(0, min({_lin_expr(first)}, {_lin_expr(second)}))"}
    return {**{k: v for k, v in f.items() if k not in ("rate", "rates")}, "rate": r, "rates": [dict(r)],
            "numbers": len(first) + len(second)}


def _delay_flows(H: str, S: str, d: dict) -> list[dict]:
    """(312, delayed.py) a delayed inflow as the found model's own hidden stock H: outside -> H at the order
    max(0, C . x) (an expression: the floor is the order's own), H -> S at a H."""
    r_in = {"form": "expr", "expr": f"max(0, {_lin_expr(d['order'])})"}
    r_out = {"form": "proportional", "stock": H, "k": float(d["a"])}
    return [{"from": None, "to": H, "ratio": 1, "rate": r_in, "rates": [dict(r_in)], "id": f"rows_in_{H}",
             "numbers": len(d["order"])},
            {"from": H, "to": S, "ratio": 1, "rate": r_out, "rates": [dict(r_out)], "id": f"rows_{H}_{S}"}]


def _build(B, flows, amts, rows, ids, dy, X, names, unit, final, declared, bd, row, T, fine, floor, meta=None, t_end=None,
           hidden_out: dict | None = None, movable=None):
    """(flows, stocks to leave out) of the rebuild over B; None where a flow cannot run."""
    series = {n: X[:, j] for j, n in enumerate(names)}
    sched: dict = {}
    hid: dict = {}

    def note_sched():
        for S in B:
            if R[S].get("sched") is not None:
                series[f"__s_{S}"] = R[S]["sched"]
                sched[f"__s_{S}"] = (R[S]["sched"], R[S]["sched_idx"])
                hid.pop(f"__s_{S}", None)
                if R[S].get("integrator") is not None:
                    hid[f"__s_{S}"] = (S, R[S]["integrator"])
    kept, kept_on = [], {S: np.zeros(T) for S in B}
    for n, f in enumerate(flows):
        a, b = f.get("from"), f.get("to")
        if a not in B and b not in B:
            kept.append(f)
            continue
        if _fixed_input(f, declared, bd, names) and amts[n] is not None:
            kept.append(f)                                   # (312) 1 x the measured input: fixed, never rebuilt
            kept_on[b] += amts[n]
            continue
        if a is None or b is None or (a in B and b in B):
            continue                                         # rebuilt
        if amts[n] is None:
            return None
        other = b if a in B else a
        if other not in fine:
            continue                                         # the other stock's answer misses it too: cut
        kept.append(f)                                       # between a rebuilt stock and a fitted one: stays
        if b in B:
            kept_on[b] += amts[n]
        if a in B:
            kept_on[a] -= float(f.get("ratio") or 1) * amts[n]
    R, y = {}, {}
    for S in B:
        y[S] = dy[ids.index(S)] - kept_on[S]
        R[S] = row(S, y[S]) if np.any(kept_on[S]) else rows[S]
    note_sched()
    fixed, excl = {}, {S: set() for S in B}
    for _ in range(REPAIRS + 1):
        moves, link_in, link_out, drain, left = _pair(R, B, unit, final, meta, movable,
                                                      lambda S: "in" not in _sides(S, declared, bd))
        taken_by = {S: sum(t.get(S, 0.0) for x_, _, t in moves if x_ == S) for S in B}
        for S in B:                                          # a link that moves S by less than the data's
            res = 10.0 * math.sqrt(floor[S] / T)             # rounding is rounding, not a flow
            for links in (link_in[S], link_out[S]):
                for x in [x for x, k in links.items() if np.max(np.abs(k * series[x])) < res]:
                    del links[x]
        redo = []
        for S in B:                                          # a row that needs a side the boundary lacks is
            sides = _sides(S, declared, bd)                  # refitted without those links (and, with no
            off = set()                                      # outflow, with its drain fixed at its moves')
            if "in" not in sides:
                off |= {x for x in link_in[S] if x != S}
                if R[S]["lin"].get(S, 0.0) > 0:
                    off.add(S)                               # growth in proportion to itself is an inflow
            if "out" not in sides:
                off |= {x for x in link_out[S] if x != S}
            self_k = fixed.get(S)
            if "out" not in sides and left[S] > LEFT * drain[S] and drain[S] > left[S]:
                self_k = -(drain[S] - left[S])
            elif "out" not in sides and taken_by[S] > drain[S] * (1 + TOL_MOVE) + 1e-12:
                self_k = -taken_by[S]                        # (316) overdrawn by a move into a closed stock
            if off - excl[S] or self_k != fixed.get(S):
                excl[S] |= off
                if self_k is not None:
                    fixed[S] = self_k
                redo.append(S)
        if not redo:
            break
        for S in redo:
            R[S] = row(S, y[S], fixed.get(S), frozenset(excl[S]))
        note_sched()
    bad = set()
    for S in B:
        self_k = R[S]["lin"].get(S, 0.0)
        if self_k > 0:
            link_in[S][S] = self_k
        if left[S] > LEFT * drain[S]:
            link_out[S][S] = left[S]
        sides = _sides(S, declared, bd)
        h = R[S]["hinge"]
        need = {"in": bool(link_in[S]), "out": bool(link_out[S])}
        if R[S].get("delay") is not None:
            if need["in"]:
                bad.add(S)                                   # (312) one flow per side: the delayed inflow alone
                continue
            need["in"] = True
        if h is not None:
            if need[h[0]]:
                bad.add(S)                                   # one flow per side: the floored term alone
                continue
            need[h[0]] = True
        if any(v and s not in sides for s, v in need.items()):
            bad.add(S)
        if S in bad and __import__("os").environ.get("BADTRACE"):  # TEMPDEBUG
            print("BAD", S, need, sides, "delay", R[S].get("delay") is not None, "hinge", h, "sched", R[S].get("sched") is not None, "integ", R[S].get("integrator") is not None, link_in[S], link_out[S], R[S]["lin"])
    if bad:
        return flows, bad
    new = [(x, Tt, dict(terms), False) for x, Tt, terms in moves]
    for S in sorted(B):
        h = R[S]["hinge"]
        if link_in[S]:
            new.append((None, S, dict(link_in[S]), False))
        if link_out[S]:
            new.append((S, None, dict(link_out[S]), False))
        if h is not None:
            new.append((None, S, dict(h[1]), True) if h[0] == "in" else (S, None, dict(h[1]), True))
    left_y = {S: (y[S] - R[S]["delay"]["inflow"]) if R[S].get("delay") is not None else y[S] for S in B}
    new = _together(new, B, left_y, R, series, floor, T)
    out = list(kept)
    taken = set(ids) | set(names)
    ctx = {"derived": meta or {}, "sched": sched, "t_end": t_end, "names": names, "X": X, "hid": hid,
           "hidden_names": {sn: _hidden_name(S, taken) for sn, (S, _) in hid.items()}}
    for a, b, terms, _ in new:
        out.append(_flow(a, b, terms, ctx))
    for S in sorted(B):                                      # (312) a delayed inflow: outside -> H -> S
        d = R[S].get("delay")
        if d is None:
            continue
        if d.get("second") and R[S]["hinge"] is not None:   # its outflow floored twice: max(0, min(B . z, D . z))
            for n_, f in enumerate(out):
                if f.get("from") == S and f.get("to") is None and f.get("id") == f"rows_{S}_None":
                    out[n_] = _two_floors(f, R[S]["hinge"][1], d["second"])
        H = _hidden_name(S, taken)
        out += _delay_flows(H, S, d)
        if hidden_out is not None:
            hidden_out[H] = d["levels"][:-1]
            hidden_out[f"__full_{H}"] = d["levels"]
    for sn, H in ctx["hidden_names"].items():               # (310 round 2) each integrator a row reads is a hidden stock
        if any(H in str((f.get("rate") or {}).get("expr", "")) for f in out):
            out += _hidden_flows(H, hid[sn][1], names, X)
            if hidden_out is not None:
                hidden_out[H] = hid[sn][1]["h"]
    return out, set()


def _together(new, B, y, R, series, floor, T, rounds: int = 4):
    """Every rebuilt flow's numbers refitted at once over the rebuilt stocks' rows (a move's number shows in
    both its stocks), each row weighted by its own residual scale; a floored flow's active steps re-read. (312) The
    flows of a row with a delayed inflow keep the numbers its own fit gave them (delayed.py fits them on the one-step
    error; the run's error here carries the hidden stock's misfit forward)."""
    order = sorted(B)
    w = {S: 1.0 / math.sqrt(max(R[S]["sse"], floor[S]) / T) for S in order}
    yy = np.concatenate([w[S] * y[S] for S in order])
    held = {S for S in order if R[S].get("delay") is not None}
    params = [(n, x) for n, (a, b, terms, _) in enumerate(new) if not ({a, b} & held) for x in terms]
    if not params:
        return new
    cur = [dict(t) for _, _, t, _ in new]

    def design(cur):
        M = np.zeros((len(order) * T, len(params)))
        for c, (n, x) in enumerate(params):
            a, b, _, hinged = new[n]
            col = series[x].copy()
            if hinged:
                col = col * (sum(k * series[z] for z, k in cur[n].items()) > 0)
            for r, S in enumerate(order):
                if S == b:
                    M[r * T:(r + 1) * T, c] += w[S] * col
                if S == a:
                    M[r * T:(r + 1) * T, c] -= w[S] * col
        return M

    def sse(cur):
        tot = 0.0
        for S in order:
            pred = np.zeros(T)
            for n, (a, b, _, _) in enumerate(new):
                if S in (a, b):
                    amt = np.maximum(0.0, sum(k * series[z] for z, k in cur[n].items()))
                    pred += amt if S == b else -amt
            tot += float((w[S] * (y[S] - pred)) @ (w[S] * (y[S] - pred)))
        return tot

    best, best_sse = cur, sse(cur)
    for _ in range(rounds):
        coef, *_ = np.linalg.lstsq(design(best), yy, rcond=None)
        trial = [dict(t) for t in best]
        for c, (n, x) in enumerate(params):
            trial[n][x] = float(coef[c])
        got = sse(trial)
        if not got < best_sse * (1 - 1e-9):
            break
        best, best_sse = trial, got
    return [(a, b, best[n], h) for n, (a, b, _, h) in enumerate(new)]


def _golden_min(f, grid: np.ndarray) -> tuple[float, float]:
    """(316) The minimum of f over grid, refined by golden section between the best point's neighbours."""
    vals = [f(v) for v in grid]
    j = int(np.argmin(vals))
    lo, hi = float(grid[max(0, j - 1)]), float(grid[min(len(grid) - 1, j + 1)])
    best = (float(grid[j]), vals[j])
    for _ in range(30):
        if hi - lo <= 1e-9 * max(1.0, abs(hi)):
            break
        a, b = hi - 0.618 * (hi - lo), lo + 0.618 * (hi - lo)
        fa, fb = f(a), f(b)
        if fa < fb:
            hi = b
            if fa < best[1]:
                best = (a, fa)
        else:
            lo = a
            if fb < best[1]:
                best = (b, fb)
    return best
