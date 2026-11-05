"""A row whose inflow arrives through a hidden first-order delay (kubernetes 312, 2026-10-22).

docs/research/fixes/312-kalman_lna.md. A measured stock S may be fed from outside through a stock nobody measures,
H (a DELAY1: H' = u - a H, S gains a H a step), where the order u is itself floored, u = max(0, C . x) (an order
placed only while a shortfall is positive). S's row is then

    dS = a H + X_A lin - max(0, X_B . b)        H_{t+1} = (1 - a) H_t + max(0, X_C . c)_t,   H_0 a number

(an always-on linear part and at most one floored outflow beside it). Every number is linear once a and the two
active sets are fixed, so the fit is the floored row's active-set least squares with the delayed columns
G_a(on . x) = a sum_s (1 - a)^(t-1-s) on_s x_s, and a decay column for H_0:

1. the outflow's start: the floored outflow read off S's change alone (the steps it falls);
2. for each a on a grid (delays of 1 to 50 steps), the order u is read off the inflow part by inverting the delay,
   u_t = (v_{t+1} - (1 - a) v_t) / a (v: S's change less the outflow and the linear part), its readers searched
   as a floored term's are (every subset up to SUB levels, on the steps the order is placed, re-read), and the
   best few judged by the joint refit of the whole row;
3. a refined by golden section around the best grid point, the outflow and the linear part refitted on what the
   delayed inflow leaves, and the search repeated (ROUNDS);
4. kept by the caller where its BIC beats the row's other forms (a schedule, an integrator) by PRICE log T.

It is written by rows.py as the found model's own hidden stock: outside -> H at max(0, C . x), H -> S at a H.
Nothing reads the case or its truth; the same rule for every row with an inflow side.

To adopt it in another strategy: copy this file with kalman_lna's rows.py (it calls `fit` from `row`, and `_build`
writes the two flows); numpy and scipy.signal only.
"""

from __future__ import annotations

import itertools
import math
import time

import numpy as np
from scipy.signal import lfilter

GRID = 20             # decay rates a on the grid (log-spaced from 1/50 to 1)
ROUNDS = 2            # rounds of the outflow refitted on what the delayed inflow leaves, then the inflow again
TOP = 3               # order-reader subsets judged by the joint refit at each a
SUB = 3               # the order reads at most this many levels (and no constant beyond them)
MIN_ON = 5            # the order is placed on at least this many steps
DEBUG = False
TRIM = 0.05           # the share of steps the decay's score leaves out (least trimmed squares)
POLISH = 6            # order-reader subsets polished by least trimmed squares at each decay
OUT_ROUNDS = 4        # rounds of the outflow read off what the one-step inflow leaves
POLISHED_DECAYS = 3   # the decays (best by the quick fits) whose order readers are polished


def delay(a: float, Z: np.ndarray) -> np.ndarray:
    """a H for H_{t+1} = (1 - a) H_t + Z_t, H_0 = 0, column by column."""
    return lfilter([0.0, a], [1.0, -(1.0 - a)], Z, axis=0)


def start(a: float, T: int) -> np.ndarray:
    """a H for H_0 = 1 and no order."""
    return a * (1.0 - a) ** np.arange(T)


def _lstsq(M: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(M, y, rcond=None)[0]


NONE = ((), np.zeros(0), (), np.zeros(0), (), np.zeros(0))   # no linear part and no outflow


def outflow(X, B, b, D=(), d=()) -> np.ndarray:
    """The outflow max(0, X_B b), or with a second floor max(0, min(X_B b, X_D d)) (a threshold that is itself
    floored: 312's nodes removed above max(2, pods / 2))."""
    if not len(B):
        return np.zeros(X.shape[0])
    z = X[:, list(B)] @ np.asarray(b, dtype=float)
    if len(D):
        z = np.minimum(z, X[:, list(D)] @ np.asarray(d, dtype=float))
    return np.maximum(0.0, z)


def rest(X, o) -> np.ndarray:
    """What the linear part and the outflow o = (A, lin, B, b, D, d) add to the row."""
    A, lin, B, b, D, d = o
    z = X[:, list(A)] @ np.asarray(lin, dtype=float) if len(A) else np.zeros(X.shape[0])
    return z - outflow(X, B, b, D, d)


def inflow(X, a, C, c, h0) -> np.ndarray:
    """a H on each step: the order max(0, X_C c) through the delay, from H's start h0."""
    u = np.maximum(0.0, X[:, list(C)] @ np.asarray(c, dtype=float))
    return delay(a, u[:, None])[:, 0] + h0 * start(a, X.shape[0])


def predict(X, a, C, c, h0, o) -> np.ndarray:
    return inflow(X, a, C, c, h0) + rest(X, o)


def hidden_levels(X, a, C, c, h0) -> np.ndarray:
    """H at each step's start and one past the last step (T + 1 values)."""
    u = np.maximum(0.0, X[:, list(C)] @ np.asarray(c, dtype=float))
    H = np.zeros(len(u) + 1)
    H[0] = h0
    for t in range(len(u)):
        H[t + 1] = (1.0 - a) * H[t] + u[t]
    return H


def noise(r: np.ndarray) -> float:
    """4 robust standard deviations of what is noise in r: the median of the smaller half of its nonzero sizes (a
    level that holds exactly still, 312's nodes for hundreds of steps, gives exact zeros that are no measure of the
    rounding elsewhere)."""
    m = np.abs(r[r != 0])
    if not len(m):
        return 0.0
    small = m[m <= np.quantile(m, 0.5)]
    return 4.0 * 1.4826 * float(np.median(small))


def _quick(Z: np.ndarray, u: np.ndarray, on: np.ndarray, rounds: int = 4) -> np.ndarray:
    """u ~ max(0, Z c), a start: least squares on the steps the order is placed, then a few Gauss-Newton steps of the
    hinge on all but the TRIM share of steps it misses most (least squares on the kept steps where Z c > 0; the
    kept steps where it is off fit 0 whatever c is)."""
    cc = _lstsq(Z[on], u[on])
    n = len(u) - int(TRIM * len(u))
    for _ in range(rounds):
        z = Z @ cc
        e = np.abs(u - np.maximum(0.0, z))
        keep = np.zeros(len(u), dtype=bool)
        keep[np.argpartition(e, n - 1)[:n]] = True
        act = keep & (z > 0)
        if int(act.sum()) < max(MIN_ON, Z.shape[1] + 1):
            break
        c2 = _lstsq(Z[act], u[act])
        if np.allclose(c2, cc, rtol=1e-10, atol=0.0):
            break
        cc = c2
    return cc


def _trimmed(e: np.ndarray) -> float:
    e2 = np.sort(e * e)
    return float(e2[: len(e2) - int(TRIM * len(e2))].sum())


def _robust(Z: np.ndarray, u: np.ndarray, cc: np.ndarray, rounds: int = 10) -> np.ndarray:
    """u ~ max(0, Z c) on every step by least trimmed squares: the hinge fitted by least squares (scipy) on all but the
    TRIM share of steps it misses most, those steps re-read until they hold. Every step counts, the off steps too (the
    floor is what pins numbers that the placed steps alone leave collinear), and a few steps where the order is not
    exactly floored linear (312: an injection leaves the room negative) do not pull the numbers."""
    from scipy.optimize import least_squares

    keep = np.ones(len(u), dtype=bool)
    n = len(u) - int(TRIM * len(u))
    for _ in range(rounds):
        try:
            cc = least_squares(lambda c: (u - np.maximum(0.0, Z @ c))[keep], cc, method="lm").x
        except Exception:                                   # noqa: BLE001 (a failed polish keeps the start)
            break
        e = np.abs(u - np.maximum(0.0, Z @ cc))
        k2 = np.zeros(len(u), dtype=bool)
        k2[np.argsort(e)[:n]] = True
        if np.array_equal(k2, keep):
            break
        keep = k2
    return cc


def _orders(X: np.ndarray, v: np.ndarray, a: float, b_cols: list[int], res: float, support=None, polish: bool = True):
    """[(score, support, coefs)] of the best few order-reader subsets for the inflow part v at decay a. The order is
    read by inverting the delay (u_t = (v_{t+1} - (1 - a) v_t) / a) and fitted as a floored term: every subset by
    least squares on the steps it is placed (_quick), the POLISH best of those by least trimmed squares on every
    step (_robust). The score is the one-step error of v the order leaves, a (u - max(0, Z c)), as least trimmed
    squares (an equation error: the hidden stock's own misfit is not carried forward, and a few steps the form
    misses do not choose the decay)."""
    u = (v[1:] - (1.0 - a) * v[:-1]) / a
    Xu = X[:-1]
    on = u > max(noise(u), 10.0 * res / a, 1e-12)          # the order is placed beyond the data's resolution
    if int(on.sum()) < MIN_ON:
        return []
    subs = [tuple(support)] if support is not None else [sub for k in range(1, SUB + 1) for sub in itertools.combinations(b_cols, k)]
    quick = []
    for sub in subs:
        Z = Xu[:, list(sub)]
        cc = _quick(Z, u, on)
        quick.append((_trimmed(a * (u - np.maximum(0.0, Z @ cc))), sub, cc))
    quick.sort(key=lambda g: g[0])
    if not polish:
        return quick[:TOP]
    got = []
    for _, sub, cc in quick[:POLISH]:
        Z = Xu[:, list(sub)]
        cc = _robust(Z, u, cc)
        got.append((_trimmed(a * (u - np.maximum(0.0, Z @ cc))), sub, cc))
    got.sort(key=lambda g: g[0])
    return got[:TOP]


def _golden(f, lo: float, hi: float, rounds: int = 30):
    g = (math.sqrt(5) - 1) / 2
    x1, x2 = hi - g * (hi - lo), lo + g * (hi - lo)
    f1, f2 = f(x1), f(x2)
    for _ in range(rounds):
        if f1 < f2:
            hi, x2, f2 = x2, x1, f1
            x1 = hi - g * (hi - lo)
            f1 = f(x1)
        else:
            lo, x1, f1 = x1, x2, f2
            x2 = lo + g * (hi - lo)
            f2 = f(x2)
    return (x1, f1) if f1 < f2 else (x2, f2)


def _polish_out(X, y, a, C, c, o, rounds: int = 10):
    """o with the linear part's and the outflow's numbers refitted on the one-step error with the order held:
    w_t = y_t - (1 - a) y_{t-1} - a u^_{t-1} = D(rest)_t, D(z)_t = z_t - (1 - a) z_{t-1} (every term on observed
    levels, so nothing the hidden stock carries is fitted), by least trimmed squares (TRIM); None where it fails."""
    from scipy.optimize import least_squares

    A, lin, B, b, D, d = o
    if not len(A) and not len(B):
        return None
    uhat = np.maximum(0.0, X[:, list(C)] @ np.asarray(c, dtype=float))
    w = y[1:] - (1.0 - a) * y[:-1] - a * uhat[:-1]
    na, nb = len(A), len(B)

    def model(th):
        z = rest(X, (A, th[:na], B, th[na:na + nb], D, th[na + nb:]))
        return z[1:] - (1.0 - a) * z[:-1]
    th = np.concatenate([np.asarray(lin, dtype=float), np.asarray(b, dtype=float), np.asarray(d, dtype=float)])
    keep = np.ones(len(w), dtype=bool)
    n = len(w) - int(TRIM * len(w))
    for _ in range(rounds):
        try:
            th = least_squares(lambda t: (w - model(t))[keep], th, method="lm").x
        except Exception:                                   # noqa: BLE001 (a failed polish keeps the numbers)
            return None
        e = np.abs(w - model(th))
        k2 = np.zeros(len(w), dtype=bool)
        k2[np.argsort(e)[:n]] = True
        if np.array_equal(k2, keep):
            break
        keep = k2
    return (tuple(A), th[:na], tuple(B), th[na:na + nb], tuple(D), th[na + nb:])


def _eq_score(X, y, a, C, c, o) -> float:
    """The whole row's one-step error, y_t - (1 - a) y_{t-1} - a u^_{t-1} - D(rest)_t, as least trimmed squares: what
    the rounds are compared on (the run's error carries the hidden stock's misfit forward)."""
    uhat = np.maximum(0.0, X[:, list(C)] @ np.asarray(c, dtype=float))
    z = rest(X, o)
    e = (y[1:] - (1.0 - a) * y[:-1] - a * uhat[:-1]) - (z[1:] - (1.0 - a) * z[:-1])
    return _trimmed(e)


def _h0(X, y, a, C, c, o) -> tuple[float, float]:
    """(sse, h0): H's start fitted on the run's (output) error with everything else held."""
    e = y - predict(X, a, C, c, 0.0, o)
    h = start(a, len(y))
    h0 = float(h @ e / max(float(h @ h), 1e-300))
    e = e - h0 * h
    return float(e @ e), h0


def _second(X, t_out, keep, o, b_cols, res):
    """o with a second floor on its outflow, max(0, min(X_B b, X_D d)), where that wins by BIC on the steps in keep:
    the steps the first floor overstates (t_out, the outflow the data show, well below it) start D's fit, every
    subset of up to SUB readers (a constant among them), then the two regimes alternate (each active step to the
    lower of the two), both refitted on their own steps; None where nothing wins."""
    A, lin, B, b, D, d = o
    if not len(B):
        return None
    f1 = X[:, list(B)] @ b
    tol = max(noise(t_out - np.maximum(0.0, f1)), 10.0 * res, 1e-12)
    over = keep & (f1 > tol) & (np.maximum(0.0, f1) - t_out > tol)
    if int(over.sum()) < MIN_ON:
        return None
    n = int(keep.sum())
    logn = math.log(max(n, 2))
    e1 = (t_out - np.maximum(0.0, f1))[keep]
    best = (n * math.log((float(e1 @ e1) + n * res * res) / n), None)
    XB = X[:, list(B)]
    for k in range(1, SUB + 1):
        for sub in itertools.combinations(b_cols, k):
            XD = X[:, list(sub)]
            dd = _lstsq(XD[over], t_out[over])
            bb = np.asarray(b, dtype=float)
            for _ in range(10):
                zb, zd = XB @ bb, XD @ dd
                act = keep & (np.minimum(zb, zd) > 0)
                rb, rd = act & (zb <= zd), act & (zd < zb)
                if int(rd.sum()) < MIN_ON or int(rb.sum()) < max(MIN_ON, len(B) + 1):
                    break
                b2, d2 = _lstsq(XB[rb], t_out[rb]), _lstsq(XD[rd], t_out[rd])
                if np.allclose(b2, bb, rtol=1e-10, atol=0.0) and np.allclose(d2, dd, rtol=1e-10, atol=0.0):
                    break
                bb, dd = b2, d2
            e = (t_out - np.maximum(0.0, np.minimum(XB @ bb, XD @ dd)))[keep]
            score = n * math.log((float(e @ e) + n * res * res) / n) + k * logn
            if score + logn < best[0]:
                best = (score, (tuple(A), lin, tuple(B), bb, tuple(sub), dd))
    return best[1]


def _order(X, v, grid, b_cols, res):
    """(a, C, c) of the order for the inflow part v: the decay on the grid, then golden section around it with the
    order's readers held; None where no order is placed."""
    quick = []
    for a in grid:                                        # every decay with the quick fits, then the best few polished
        got = _orders(X, v, a, b_cols, res, polish=False)
        if got:
            quick.append((got[0][0], a))
    scan = []
    for _, a in sorted(quick)[:POLISHED_DECAYS]:
        got = _orders(X, v, a, b_cols, res)
        if got:
            scan.append((got[0][0], a, got[0][1]))
    if not scan:
        return None
    k = int(np.argmin([g[0] for g in scan]))
    _, a0, C = scan[k]
    pos = int(np.searchsorted(grid, a0))
    lo, hi = grid[max(0, pos - 1)], grid[min(len(grid) - 1, pos + 1)]
    memo = {}

    def score(a):
        if a not in memo:
            memo[a] = _orders(X, v, a, [], res, support=C)
        return memo[a][0][0] if memo[a] else math.inf
    a1 = a0
    if hi > lo:
        a1, s1 = _golden(score, lo, hi)
        if not s1 < score(a0):
            a1 = a0
    score(a1)
    if not memo.get(a1):
        return None
    sc, C, c = memo[a1][0]
    n = len(v) - 1 - int(TRIM * (len(v) - 1))
    while len(C) > 1:                                     # a reader that does not pay its log n is dropped
        trial = None
        for j in range(len(C)):
            got = _orders(X, v, a1, [], res, support=tuple(x for i, x in enumerate(C) if i != j))
            if got and n * math.log((got[0][0] + n * res * res) / (sc + n * res * res)) < math.log(n):
                if trial is None or got[0][0] < trial[0]:
                    trial = got[0]
        if trial is None:
            break
        sc, C, c = trial
    return float(a1), tuple(C), c


def _clear(X, v, a, C, c, res, out_on=None) -> np.ndarray:
    """The steps the order's misfit does not reach: a step whose order the form misses by far more than the rest
    (beyond 10 x the rms one-step error of the steps least trimmed squares keeps, and 10 x the data's resolution)
    reaches the hidden stock
    from the next step on, fading by 1 - a a step; those steps are left out (until it is under 1e-3 of itself)
    where the outflow is read off the run's error. Steps where the current outflow acts (out_on, at t or t + 1) are
    never flagged: there the misfit may be the outflow's, and leaving them out would hide it."""
    T = len(v)
    u = (v[1:] - (1.0 - a) * v[:-1]) / a
    e = np.abs(a * (u - np.maximum(0.0, X[:-1, list(C)] @ c)))
    kept = np.sort(e)[: len(e) - int(TRIM * len(e))]
    scale = math.sqrt(float(np.mean(kept ** 2))) if len(kept) else 0.0      # the rms error of the steps it fits
    flag = e > max(10.0 * scale, 10.0 * res)
    if out_on is not None:                                # a misfit where the outflow acts is the outflow's to explain
        flag &= ~(out_on[:-1] | out_on[1:])
    bad = np.flatnonzero(flag)
    reach = int(math.ceil(math.log(1e-3) / math.log(max(1e-9, 1.0 - a)))) if a < 1 else 1
    keep = np.ones(T, dtype=bool)
    for t in bad:
        keep[t + 1: t + 1 + reach] = False
    return keep


def _outflow(X, y, a, C, c, o, floored, a_cols, b_cols, res):
    """o (the outflow and linear part) with the order held: polished on the one-step error (_polish_out), the support
    read again on what the one-step inflow leaves (v^_t = (1 - a) v_{t-1} + a u^_{t-1}, v the data's inflow part under
    the current outflow; a second floor tried on it, _second), until the support holds (OUT_ROUNDS)."""
    T = len(y)
    for _ in range(OUT_ROUNDS):
        got = _polish_out(X, y, a, C, c, o)
        if got is None:
            break
        o = got
        v = y - rest(X, o)
        uhat = np.maximum(0.0, X[:, list(C)] @ c)
        vhat = np.empty(T)
        vhat[0] = v[0]
        vhat[1:] = (1.0 - a) * v[:-1] + a * uhat[:-1]
        keep = _clear(X, v, a, C, c, res, outflow(X, *o[2:]) > 0 if len(o[2]) else None)
        got = _read_out(X, y - vhat, keep, floored, a_cols, b_cols, res)
        if got is None or tuple(map(tuple, (got[0], got[2], got[4]))) == tuple(map(tuple, (o[0], o[2], o[4]))):
            break
        o = got
    return o


def _read_out(X, z, keep, floored, a_cols, b_cols, res):
    """The linear part and the outflow read off z (= linear part - outflow) on the steps in keep: the caller's floored
    row, then a second floor where it wins (_second); None where no floored row fits."""
    got = floored(z, a_cols, keep)
    if got is None:
        return None
    o = (tuple(got[0]), np.asarray(got[1], dtype=float), tuple(got[2]), np.asarray(got[3], dtype=float), (), np.zeros(0))
    lin_part = X[:, list(o[0])] @ o[1] if len(o[0]) else np.zeros(X.shape[0])
    two = _second(X, lin_part - z, keep if keep is not None else np.ones(len(z), dtype=bool), o, b_cols, res)
    return two if two is not None else o


def _prune(X, y, a, C, c, o, res):
    """o with each term of the linear part and the outflow that does not pay its log n on the one-step error dropped
    (the rest refitted, _polish_out), one at a time, the cheapest first."""
    n = len(y) - 1 - int(TRIM * (len(y) - 1))
    floor = n * res * res
    cur = _eq_score(X, y, a, C, c, o)
    while True:
        trial = None
        for part in (0, 2, 4):
            if part in (2, 4) and len(o[part]) <= (0 if part == 4 else 1):
                continue
            for j in range(len(o[part])):
                p2 = list(o)
                p2[part] = tuple(x for i, x in enumerate(o[part]) if i != j)
                p2[part + 1] = np.array([x for i, x in enumerate(o[part + 1]) if i != j])
                if part == 4 and not p2[4]:
                    p2[5] = np.zeros(0)
                got = _polish_out(X, y, a, C, c, tuple(p2)) if (p2[0] or p2[2]) else tuple(p2)
                if got is None:
                    continue
                sc = _eq_score(X, y, a, C, c, got)
                if n * math.log((sc + floor) / (cur + floor)) < math.log(n) and (trial is None or sc < trial[0]):
                    trial = (sc, got)
        if trial is None:
            return o
        cur, o = trial


def fit(X: np.ndarray, y: np.ndarray, floor: float, deadline: float, a_cols: list[int], b_cols: list[int],
        floored, out_side: bool = True) -> dict | None:
    """The delayed row of y (see the module): {sse, k (its numbers), a, C, c, h0, A, lin, B, b, D, d}, or None.
    floored(y, a_cols, keep=None) -> (A, lin, B, b) of the caller's floored-row search (sign -1; on the steps in keep)
    or None. Each round: the order (decay and readers) read off the inflow part the current outflow leaves (_order),
    then the outflow refitted with it held (_outflow) from two starts, the current outflow and the one read off the
    run's error the delayed order leaves; rounds and starts are compared on the whole row's one-step error
    (_eq_score). H's start is fitted last, and the answer's sse is the run's (output) error."""
    T = len(y)
    res = math.sqrt(floor / T)
    o = NONE
    if out_side:
        got = _read_out(X, y, None, lambda z, cols, keep: floored(z, [], keep), [], b_cols, res)
        if got is not None:
            o = got
    best = None
    grid = np.exp(np.linspace(math.log(1.0 / 50), 0.0, GRID))
    for _ in range(ROUNDS):
        if time.monotonic() > deadline:
            break
        near = grid
        if best is not None:                              # a later round: the decays near the last one
            pos = int(np.argmin(np.abs(grid - best[1])))
            near = grid[max(0, pos - 2): pos + 3]
        got = _order(X, y - rest(X, o), near, b_cols, res)
        if got is None:
            break
        a, C, c = got
        outs = [o]
        if out_side:
            keep = _clear(X, y - rest(X, o), a, C, c, res, outflow(X, *o[2:]) > 0 if len(o[2]) else None)
            _, h0 = _h0(X, y, a, C, c, NONE)
            got = _read_out(X, y - inflow(X, a, C, c, h0), keep, floored, a_cols, b_cols, res)
            if got is not None:
                outs.append(got)
        top = None
        for o2 in outs if out_side else [NONE]:
            o2 = _outflow(X, y, a, C, c, o2, floored, a_cols, b_cols, res) if out_side and (len(o2[0]) or len(o2[2])) else o2
            eq = _eq_score(X, y, a, C, c, o2)
            if DEBUG:
                print("delayed", round(a, 6), C, np.round(c, 5), o2[0], np.round(o2[1], 5), o2[2], np.round(o2[3], 5),
                      o2[4], np.round(o2[5], 5), "eq", eq)
            if top is None or eq < top[0]:
                top = (eq, a, C, c, o2)
        o = top[4]
        if best is None or top[0] < best[0]:
            best = top
        if not out_side:
            break
    if best is None:
        return None
    _, a, C, c, o = best
    if out_side:
        o = _prune(X, y, a, C, c, o, res)
    sse, h0 = _h0(X, y, a, C, c, o)
    A, lin, B, b, D, d = o
    return {"sse": sse, "k": len(C) + 2 + len(A) + len(B) + len(D), "a": float(a), "C": tuple(C),
            "c": np.asarray(c, dtype=float), "h0": float(h0), "A": tuple(A), "lin": np.asarray(lin, dtype=float),
            "B": tuple(B), "b": np.asarray(b, dtype=float), "D": tuple(D), "d": np.asarray(d, dtype=float)}
