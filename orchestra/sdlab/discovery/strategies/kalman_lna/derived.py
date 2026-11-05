"""Columns a row may read beyond the levels, and a slow outside drive read as a schedule (kubernetes 310, 2026-10-21).

docs/research/fixes/310-kalman_lna.md. rows.py reads each stock's change as a sparse linear function of the levels,
or that plus one term floored at 0. Two kinds of rate are outside that library, and both are general:

1. **Derived columns.** A flow proportional to one stock that switches on past a threshold of another (the first
   segment of an effect table: `X * max(0, Y - theta)`, or below it, `X * max(0, theta - Y)`), and a rate per unit
   of another stock (`X / max(1, Y)`, an amount per pod). `search` scores every such column, for every pair of
   measured stocks and (for the switch) a grid of thresholds refined by golden section, by how much of each row's
   residual it explains; the best one joins the readers when it gains PRICE log T on some row. Up to ROUNDS
   columns are added, each searched on the residuals the earlier ones leave.
2. **A schedule.** A stock fed from outside by something nobody measures that drifts slowly (an unmeasured
   integrator, a trend) leaves a row residual that is smooth in time. rows.py fits the row with a continuous
   piecewise-linear function of time (a knot every P steps) as fixed effects, the readers chosen on what that
   leaves, for a few knot spacings, by BIC (one number a knot); the inflow carries the schedule as time segments,
   each block of P steps at the schedule's mean over it. It is offered only on a declared (or undeclared)
   inflow side, where the row without it leaves 100 times the rounding.

Nothing reads the case or its truth; the same search runs on every row.
"""

from __future__ import annotations

import math

import numpy as np

ROUNDS = 3             # at most this many derived columns
THETAS = 40            # threshold grid of a switch column (quantiles of the switch's reader)
SLOW = 0.9             # a residual this autocorrelated (lag 1) is a slow drive, a schedule may carry it
PIECES = (10, 20, 30, 50, 100)   # block sizes a schedule is tried at (at least 3 blocks)
NEAR = 0.05            # a second switch on the same pair needs its threshold this share of the reader's range away


def hinge_expr(x: str, y: str, theta: float, up: bool) -> str:
    return f"{x} * max(0, {y} - {theta!r})" if up else f"{x} * max(0, {theta!r} - {y})"


def ratio_expr(x: str, y: str) -> str:
    return f"{x} / max(1, {y})"


def blocks(T: int, P: int) -> np.ndarray:
    return np.arange(T) // P


def basis(T: int, P: int) -> np.ndarray:
    """An orthonormal basis of the continuous piecewise-linear functions of time with a knot every P steps."""
    knots = np.arange(0, T - 1 + P, P, dtype=float)
    knots[-1] = max(knots[-1], T - 1)
    knots = np.unique(np.minimum(knots, T - 1))
    t = np.arange(T, dtype=float)
    H = np.zeros((T, len(knots)))
    for k, c in enumerate(knots):
        left = knots[k - 1] if k else None
        right = knots[k + 1] if k + 1 < len(knots) else None
        h = np.zeros(T)
        if left is not None:
            m = (t >= left) & (t <= c)
            h[m] = (t[m] - left) / (c - left)
        if right is not None:
            m = (t >= c) & (t <= right)
            h[m] = (right - t[m]) / (right - c)
        h[t == c] = 1.0
        H[:, k] = h
    Q, _ = np.linalg.qr(H)
    return Q


def demean(A: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """What the schedule (the span of Q) leaves of A: the fixed effects' projection."""
    A = np.asarray(A, dtype=float)
    return A - Q @ (Q.T @ A)


def fitted(r: np.ndarray, Q: np.ndarray) -> np.ndarray:
    return Q @ (Q.T @ r)


def slow(r: np.ndarray) -> bool:
    r = np.asarray(r, dtype=float) - float(np.mean(r))
    d = float(r @ r)
    return d > 0 and float(r[1:] @ r[:-1]) / d > SLOW


def _gain(c: np.ndarray, r: np.ndarray, sse: float, floor: float, T: int) -> float:
    cc = float(c @ c)
    if cc <= 0:
        return -math.inf
    red = float(c @ r) ** 2 / cc
    return T * math.log((sse + floor) / (max(0.0, sse - red) + floor))


def search(X: np.ndarray, names: list, stock_cols: list[int], resid: dict, floors: dict, proj: dict, T: int,
           taken: set[str], hinges: list | tuple = ()):
    """The best derived column for the rows' residuals: (gain, name-free meta, series) or None.
    resid: {row: residual}; proj: {row: block index or None} (a row fitted with a schedule is searched on the
    within-block changes); meta: {"expr", "reads", "base"}."""
    logT = math.log(T)
    best = None
    prepared = {S: (proj[S], (demean(r, proj[S]) if proj[S] is not None else r)) for S, r in resid.items()}
    sses = {S: float(rr @ rr) for S, (_, rr) in prepared.items()}

    def score(col):
        top = (-math.inf, None)
        for S, (idx, rr) in prepared.items():
            c = demean(col, idx) if idx is not None else col
            g = _gain(c, rr, sses[S], floors[S], T)
            if g > top[0]:
                top = (g, S)
        return top

    for ia in stock_cols:
        xa = X[:, ia]
        for ib in stock_cols:
            if ib == ia:
                continue
            xb = X[:, ib]
            e = ratio_expr(names[ia], names[ib])
            if e not in taken:
                col = xa / np.maximum(1.0, xb)
                g, S = score(col)
                if g - logT > (best[0] if best else -math.inf):
                    best = (g - logT, {"expr": e, "reads": [names[ia], names[ib]], "base": names[ia], "row": S}, col)
            lo, hi = float(np.min(xb)), float(np.max(xb))
            if hi <= lo:
                continue
            for up in (True, False):
                qs = np.quantile(xb, np.linspace(0.5, 0.995, THETAS) if up else np.linspace(0.005, 0.5, THETAS))

                def f(theta, up=up):
                    col = xa * (np.maximum(0.0, xb - theta) if up else np.maximum(0.0, theta - xb))
                    return score(col)[0], col

                vals = [f(q)[0] for q in qs]
                k = int(np.argmax(vals))
                if not math.isfinite(vals[k]):
                    continue
                a = float(qs[max(0, k - 1)])
                b = float(qs[min(len(qs) - 1, k + 1)])
                gr = (math.sqrt(5) - 1) / 2                      # golden section on the threshold
                c1, c2 = b - gr * (b - a), a + gr * (b - a)
                f1, f2 = f(c1)[0], f(c2)[0]
                for _ in range(24):
                    if f1 > f2:
                        b, c2, f2 = c2, c1, f1
                        c1 = b - gr * (b - a)
                        f1 = f(c1)[0]
                    else:
                        a, c1, f1 = c1, c2, f2
                        c2 = a + gr * (b - a)
                        f2 = f(c2)[0]
                cands = [(vals[k], float(qs[k])), (f1, c1), (f2, c2)]
                g, theta = max(cands)
                e = hinge_expr(names[ia], names[ib], theta, up)
                if e in taken or any(h[:3] == (ia, ib, up) and abs(h[3] - theta) < NEAR * (hi - lo) for h in hinges):
                    continue                                  # (311) the same switch again at a threshold a hair away
                g -= 2 * logT                                     # its number and its threshold
                if g > (best[0] if best else -math.inf):
                    best = (g, {"expr": e, "reads": [names[ia], names[ib]], "base": names[ia], "row": score(f(theta)[1])[1],
                                "hinge": (ia, ib, up, theta),
                                "switch": np.maximum(0.0, xb - theta) if up else np.maximum(0.0, theta - xb)}, f(theta)[1])
    return best


def refine(X: np.ndarray, names: list, meta: dict, parts: dict, weights: dict | None = None) -> tuple[dict, np.ndarray]:
    """A switch column's threshold refitted on the rows that read it (parts: {row: what the column and the
    residual leave of it}), the other terms of each row held: golden section around the searched threshold."""
    if not meta.get("hinge"):
        return meta, None
    ia, ib, up, theta = meta["hinge"]
    xa, xb = X[:, ia], X[:, ib]
    span = 0.02 * float(np.ptp(xb)) or 1.0

    def col(th):
        return xa * (np.maximum(0.0, xb - th) if up else np.maximum(0.0, th - xb))

    def sse(th):
        c = col(th)
        cc = float(c @ c)
        if cc <= 0:
            return math.inf
        return sum((weights or {}).get(S, 1.0) * (float(z @ z) - float(c @ z) ** 2 / cc) for S, z in parts.items())

    a, b = theta - span, theta + span
    gr = (math.sqrt(5) - 1) / 2
    c1, c2 = b - gr * (b - a), a + gr * (b - a)
    f1, f2 = sse(c1), sse(c2)
    for _ in range(60):
        if f1 < f2:
            b, c2, f2 = c2, c1, f1
            c1 = b - gr * (b - a)
            f1 = sse(c1)
        else:
            a, c1, f1 = c1, c2, f2
            c2 = a + gr * (b - a)
            f2 = sse(c2)
    th = c1 if f1 < f2 else c2
    if not sse(th) < sse(theta):
        return meta, None
    return {**meta, "expr": hinge_expr(names[ia], names[ib], th, up), "hinge": (ia, ib, up, th),
            "switch": np.maximum(0.0, xb - th) if up else np.maximum(0.0, th - xb)}, col(th)


# 3. A hidden integrator (310): a stock nobody measures, fed at a constant rate and drained in proportion to itself,
#    at a constant rate or switched by one of the derived switches (derived column / its base), read by one row.
#    Its level is linear in its initial level and its inflow for a given drain rate a, so the row reads two
#    columns per a; a is searched on a grid and refined by golden section.

DECAYS = 24            # drain rates on the grid (log-spaced, per unit of the switch's mean, and 0)


def integrator(a: float, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(response to a unit initial level, response to a unit inflow) of h' = 1 - a s h, read at each step's start."""
    T = len(s)
    g = np.clip(1.0 - a * s, 0.0, None)
    h0, h1 = np.empty(T), np.empty(T)
    x0, x1 = 1.0, 0.0
    for k in range(T):
        h0[k], h1[k] = x0, x1
        x0, x1 = x0 * g[k], x1 * g[k] + 1.0
    return h0, h1


def decays(s: np.ndarray) -> list[float]:
    m = float(np.mean(np.abs(s))) or 1.0
    return [0.0] + list(np.geomspace(1e-4, 1.0, DECAYS - 1) / m)


def golden(f, a: float, b: float, rounds: int = 30) -> tuple[float, float]:
    """(argmin, min) of f on [a, b]."""
    gr = (math.sqrt(5) - 1) / 2
    c1, c2 = b - gr * (b - a), a + gr * (b - a)
    f1, f2 = f(c1), f(c2)
    for _ in range(rounds):
        if f1 < f2:
            b, c2, f2 = c2, c1, f1
            c1 = b - gr * (b - a)
            f1 = f(c1)
        else:
            a, c1, f1 = c1, c2, f2
            c2 = a + gr * (b - a)
            f2 = f(c2)
    return (c1, f1) if f1 < f2 else (c2, f2)


# 4. A capped move (311): a move X -> T limited by the room another set of levels leaves, min(a X, max(0, c + B . z))
#    (pods start at a rate of the pending ones, but only into the room the nodes leave). One column a row may read;
#    the source's row loses it and a same-unit row gains it, so rows.py pairs it as a move. Its numbers are first read
#    off what T gains (or X loses) by alternating the two regimes, then refitted on the rows that read it.

CAP_READERS = 2        # the room reads at most this many levels (and a constant)
CAP_ITERS = 40
CAP_TRY = 4            # caps judged by refitting their rows
CAP_REFITS = 3         # rounds of the cap's numbers and its rows refitted in turn


def cap_expr(x: str, a: float, zs: list[tuple[str | None, float]]) -> str:
    """min(a x, max(0, room)), the room a sum of numbers times levels (None: the constant)."""
    room = " + ".join(f"{float(b)!r}" if z is None else f"{float(b)!r} * {z}" for z, b in zs) or "0.0"
    return f"min({float(a)!r} * {x}, max(0, {room}))"


def cap_column(x: np.ndarray, Z: np.ndarray, a: float, b: np.ndarray) -> np.ndarray:
    return np.minimum(a * x, np.maximum(0.0, Z @ b))


def capfit(f: np.ndarray, x: np.ndarray, Z: np.ndarray, min_on: int = 5, lower: bool = False):
    """(sse, a, b) of f ~ min(a x, max(0, Z b)) (Z's columns: the room's readers, a constant among them or not), by alternating the regime each step is
    in (a on the steps the rate limits, b on the steps the room limits); None where either regime has under min_on steps."""
    pos = x > 1e-9
    if int(pos.sum()) < 2 * min_on:
        return None
    a = float(np.quantile(f[pos] / x[pos], 0.9))
    bound = f < 0.9 * a * x
    if int(bound.sum()) <= Z.shape[1]:
        return None
    b = np.linalg.lstsq(Z[bound], f[bound], rcond=None)[0]
    best = None
    for _ in range(CAP_ITERS):
        room = Z @ b
        pred = np.minimum(a * x, np.maximum(0.0, room))
        sse = float(((f - pred) ** 2).sum())
        if best is not None and sse >= best[0] * (1 - 1e-12):
            break
        best = (sse, a, b.copy())
        free = a * x <= room                               # where X is empty the rate limits too (0 either way)
        sel = (~free) & ((room > 0) | (f > 1e-12))
        if int((free & pos).sum()) < min_on or int(sel.sum()) < max(min_on, Z.shape[1] + 1):
            return None
        a = float(x[free] @ f[free] / max(float(x[free] @ x[free]), 1e-300))
        b = np.linalg.lstsq(Z[sel], f[sel], rcond=None)[0]
    if best is None:
        return None
    sse, a, b = best
    b = np.asarray(b, dtype=float)
    if lower:
        # f is a lower bound of the move (T's own outflows, X's own inflows hide part of it on some steps): steps
        # well below the fit carry another flow and are left out, the rest refitted, until the steps settle
        use = np.ones(len(f), dtype=bool)
        for _ in range(CAP_ITERS // 4):
            room = Z @ b
            e = f - np.minimum(a * x, np.maximum(0.0, room))
            s = 1.4826 * float(np.median(np.abs(e))) + 1e-9
            use2 = e >= -3.0 * s
            free = a * x <= room
            fr, sel = free & use2 & pos, (~free) & ((room > 0) | (f > 1e-12)) & use2
            if int(fr.sum()) < min_on or int(sel.sum()) < max(min_on, Z.shape[1] + 1) or np.array_equal(use2, use):
                break
            use = use2
            a = float(x[fr] @ f[fr] / max(float(x[fr] @ x[fr]), 1e-300))
            b = np.linalg.lstsq(Z[sel], f[sel], rcond=None)[0]
        sse = float(((f - np.minimum(a * x, np.maximum(0.0, Z @ b))) ** 2).sum())
    room = Z @ b
    free = (a * x <= room) & pos
    if int(free.sum()) < min_on or int((~free & pos).sum()) < min_on or a <= 0:
        return None
    return sse, a, b


def refine_cap(X: np.ndarray, meta: dict, parts: dict, coefs: dict, min_on: int = 5) -> tuple[dict, np.ndarray | None]:
    """A capped column's numbers refitted on the rows that read it (parts: {row: what the column and the residual
    leave of it}; coefs: {row: the row's coefficient on it}), each row's coefficient held and each row weighted by
    how well it fits (the row that reads its other flows right says most about the move). Two starts: the numbers
    as they are, and the regimes refitted (capfit) on the move's amount the weighted rows give."""
    from scipy.optimize import least_squares

    ix, zcols, a, b = meta["cap"]
    x = X[:, ix]
    Z = X[:, zcols]
    col0 = cap_column(x, Z, a, b)
    rows_ = []
    for S in parts:
        k = coefs.get(S, 0.0)
        if abs(k) > 1e-12:
            e = parts[S] - k * col0
            rows_.append((parts[S], k, 1.0 / math.sqrt(max(float(e @ e) / len(e), 1e-300))))
    if not rows_:
        return meta, None

    def res(th):
        col = cap_column(x, Z, th[0], th[1:])
        return np.concatenate([w * (p - k * col) for p, k, w in rows_])

    starts = [np.concatenate([[a], b])]
    amount = sum(w * w * k * p for p, k, w in rows_) / sum(w * w * k * k for p, k, w in rows_)
    got = capfit(amount, x, Z, min_on)
    if got is not None:
        starts.append(np.concatenate([[got[1]], got[2]]))
    r0 = res(starts[0])
    best = (float(r0 @ r0), starts[0])
    for th0 in starts:
        try:
            got = least_squares(res, th0, method="lm", x_scale="jac", max_nfev=200 * len(th0))
        except Exception:                                  # noqa: BLE001 (a failed polish keeps the searched numbers)
            continue
        if got.x[0] > 0 and float(got.fun @ got.fun) < best[0]:
            best = (float(got.fun @ got.fun), got.x)
    if best[1] is starts[0]:
        return meta, None
    a2, b2 = float(best[1][0]), np.asarray(best[1][1:], dtype=float)
    return {**meta, "cap": (ix, zcols, a2, b2), "expr": cap_expr(meta["reads"][0], a2, list(zip(meta["room"], b2)))}, \
        cap_column(x, Z, a2, b2)
