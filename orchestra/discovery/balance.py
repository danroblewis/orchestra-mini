"""A level's balance: the counters, each added or subtracted, that explain its changes.

``Δlevel ≈ Σ inflows − Σ outflows`` in every step. The counters are found by a local search on an
objective of squared error plus a penalty per counter used, so a counter must explain a real share
to be kept: at least ``MIN_GAIN`` of the level's variance, and at least one event's worth of
squared change (``EVENT_FLOOR``), because on a level that barely moves one per cent of its variance
is less than the noise of a single step at the edge of the window. A balance that is exact in every
step earns back most of one penalty (``COMPLETION``): the rare counter that makes it exact (a
refund among a hundred orders) is then worth keeping, where alone it could never pay its way. A
counter that counted fewer than ``MIN_EVENTS`` events is not searched: on three events a level
and a counter coincide by chance as readily as a stock and its flow. It may only complete a
balance (``complete``), exactly: wherever each start's search ends is completed before the best
is chosen, so an exact balance that needs one beats a near miss by a stand-in no rare counter
completes (another instrument's count of both ways out, one of them a step late). The bar is on the evidence for the balance, not on each counter:
counters below the bar alone carry a balance only when together they make it exact in every step
and counted at least ``MIN_EVENTS`` between them (a state work is sent back to, three rejections
from review and one from the merge, four returns to work: eight changes, each explained, and no
counter moving when the level did not). Short of exact, they are no balance.

The search is cheap because it runs on the counters' Gram matrix (every pair's dot product): the
error after any one change is a few multiplications, not a pass over the steps. That pays for
starting it from many places and keeping the best, which a search from one start needs: one term
of a balance alone often makes the fit worse (inflow without its outflows), and which local
optimum a single start reaches depends on noise.

A balance with more counters than the level's changes can test sheds the ones that add nothing
it can test (``supported``). The changes can test fewer counters than half the units the level
changed in over the whole window, and always two, an inflow and an outflow (``allowed``). Beyond
that, the squared error no longer tells a counter from a coincidence: among the many subsets the
search tries, one lowers it by chance. What still tells is exactness: a counter that makes no
unit exact that the others leave inexact, and that no unit pins on its own (``alone``), is taken
out. A counter whose removal leaves fewer units exact always stays, so an exact balance is never
reduced, however many counters it has (a quiet shelf baked, sold, discarded and eaten from at
once; a second outflow that only ever comes with the first).

A counter far larger than the level (``resolvable``) is not searched: its squares' rounding is
more than the level's variance.

Copied from the old Orchestra's ``sd/balance.py``; levels slower than the grid (judged there on the
stretches between their writes) are judged on single steps here."""

from __future__ import annotations

import math

from .collinear import cancel, merge

MIN_GAIN = 0.01         # each counter must explain at least this share of the level's variance
EVENT_FLOOR = 1.0       # ... and at least one event's worth of squared change
MIN_EVENTS = 5          # ... and must have counted this many events over the steps: fewer coincide by chance (a level: changed in this many steps)
MAX_TERMS = 8
COMPLETION = 0.5        # an exact balance earns back a penalty less this many events' worth
EXACT_SSE = 0.5         # a balance with less squared error than this is exact in every step (levels are counts)
PAIR_STARTS = 8         # the search also starts from this many best pairs of counters
RESOLVABLE = 1e10       # a counter whose squared changes are more than this many times the level's is not searched


class LeastSquares:
    """Every counter's dot product with every other (the Gram matrix), and ordinary least squares
    of any level's changes on all counters at once: the normal matrix is the same for every level,
    so it is factored once (Cholesky, lightly regularised so collinear counters don't break it)."""

    def __init__(self, xs: list[list[float]]):
        self.xs = xs
        m = len(xs)
        self.g = [[0.0] * m for _ in range(m)]
        for i in range(m):
            for j in range(i + 1):
                self.g[i][j] = self.g[j][i] = sum(a * b for a, b in zip(xs[i], xs[j]))
        g = [row[:] for row in self.g]
        ridge = 1e-6 * (sum(g[i][i] for i in range(m)) / max(1, m)) + 1e-12
        for i in range(m):
            g[i][i] += ridge
        self.L = [[0.0] * m for _ in range(m)]
        for i in range(m):
            for j in range(i + 1):
                v = g[i][j] - sum(self.L[i][k] * self.L[j][k] for k in range(j))
                self.L[i][j] = math.sqrt(max(v, 1e-18)) if i == j else v / self.L[j][j]

    def against(self, y: list[float]) -> list[float]:
        """Every counter's dot product with ``y``."""
        return [sum(a * v for a, v in zip(x, y)) for x in self.xs]

    def solve(self, y: list[float]) -> list[float]:
        m, L = len(self.xs), self.L
        b = self.against(y)
        z = [0.0] * m
        for i in range(m):
            z[i] = (b[i] - sum(L[i][k] * z[k] for k in range(i))) / L[i][i]
        c = [0.0] * m
        for i in reversed(range(m)):
            c[i] = (z[i] - sum(L[k][i] * c[k] for k in range(i + 1, m))) / L[i][i]
        return c


def enough_events(deltas: list[float]) -> bool:
    """Whether a counter counted enough over these steps to carry a balance: a level and a counter
    that coincide on three events are a coincidence as often as a stock and its flow. The bar is
    on events, not steps: the oven's four orders of twenty loaves are eighty events whose size is
    what matches."""
    return sum(deltas) >= MIN_EVENTS


def complete(coef: dict[int, int], resid: list[float], sse: float, xs: list[list[float]], rare: list[int]
             ) -> tuple[dict[int, int], list[float], float]:
    """Finish a balance with the counters that counted too few events to be searched: added one at
    a time, the one that lowers the squared error most first, and kept only when the balance ends
    exact in every step (the oven's four orders of twenty loaves; a refund among a hundred
    orders). Short of exact, a rare counter explains a coincidence as readily as a flow."""
    if sse < EXACT_SSE or not rare:
        return coef, resid, sse
    trial, r, e = dict(coef), list(resid), sse
    pool = [j for j in rare if j not in trial]
    while e >= EXACT_SSE and pool:
        best = None
        for j in pool:
            for sign in (1, -1):
                r2 = [v - sign * x for v, x in zip(r, xs[j])]
                e2 = sum(v * v for v in r2)
                if e2 < e - 1e-9 and (best is None or e2 < best[0]):
                    best = (e2, j, sign, r2)
        if best is None:
            break
        e, j, sign, r = best
        trial[j] = sign
        pool.remove(j)
    return (trial, r, e) if e < EXACT_SSE else (coef, resid, sse)


def completed(g: list[list[float]], b: list[float], coef: dict[int, int], sse: float, rare: list[int]
              ) -> tuple[dict[int, int], float]:
    """``complete`` on the Gram matrix, cheap enough to try on every balance the search ends at
    (``balance_of``): the balance finished exactly by rare counters, or as it was."""
    if sse < EXACT_SSE or not rare:
        return coef, sse
    trial, e = dict(coef), sse
    xr = {j: b[j] - sum(s * g[j][k] for k, s in trial.items()) for j in rare}   # the residual's dot product with each
    pool = [j for j in rare if j not in trial]
    while e >= EXACT_SSE and pool:
        best = None
        for j in pool:
            for sign in (1, -1):
                e2 = e - 2 * sign * xr[j] + g[j][j]
                if e2 < e - 1e-9 and (best is None or e2 < best[0]):
                    best = (e2, j, sign)
        if best is None:
            break
        e, j, sign = best
        trial[j] = sign
        pool.remove(j)
        for k in pool:
            xr[k] -= sign * g[k][j]
    return (trial, max(0.0, e)) if e < EXACT_SSE else (coef, sse)


def allowed(y: list[float]) -> int:
    """How many counters the changes ``y`` of a level can test: fewer than half the units it changed
    in, so that each half of the window holds more changes than counters, and never fewer than two
    (an inflow and an outflow). Ten changes test four."""
    return max(2, (sum(1 for v in y if abs(v) >= 0.5) - 1) // 2)


def alone(j: int, coef: dict[int, int], y: list[float], xs: list[list[float]]) -> bool:
    """Whether counter ``j`` of the balance ``coef`` is pinned by a unit of its own: one the balance
    fits, in which ``j`` moved and no other counter of the balance moved the level the same way
    (``j`` the only inflow that came, or the only outflow). There its coefficient is the unit's
    change less the other way's, not a share of a sum it makes with others: a rejection the only
    outflow beside an inflow, the level holding, pins the rejections as surely as a change would."""
    for u, v in enumerate(y):
        mine = coef[j] * xs[j][u]
        if not mine or any(coef[k] * xs[k][u] * mine > 0 for k in coef if k != j):
            continue
        if abs(v - sum(sign * xs[k][u] for k, sign in coef.items())) <= 0.5:
            return True
    return False


def exact_units(coef: dict[int, int], y: list[float], xs: list[list[float]]) -> int:
    """How many units the balance ``coef`` explains exactly."""
    return sum(1 for u, v in enumerate(y) if abs(v - sum(sign * xs[k][u] for k, sign in coef.items())) <= 0.5)


def supported(coef: dict[int, int], y: list[float], xs: list[list[float]]) -> dict[int, int]:
    """``coef``, while it has more counters than the level's changes can test (``allowed``), less
    one counter that adds nothing they can test: no unit pins it on its own (``alone``), and
    without it as many units are exact (the one that counted least first). Stops at the limit, or
    when every counter left is pinned or makes some unit exact: an exact balance stays whole."""
    coef, limit = dict(coef), allowed(y)
    while len(coef) > limit:
        now = exact_units(coef, y, xs)
        idle = [j for j in coef if not alone(j, coef, y, xs)
                and exact_units({k: s for k, s in coef.items() if k != j}, y, xs) >= now]
        if not idle:
            break
        coef.pop(min(idle, key=lambda j: (sum(abs(v) for v in xs[j]), j)))
    return coef


def parsimonious(level, terms: list[tuple], steps: list[int] | None = None) -> list[tuple]:
    """``supported`` on a balance assembled from two searches (``discover.own_balance``: a
    machine's own counters, then another's on the residual they leave), each held to its own
    limit, so that the whole is held to the level's. Over ``steps``, as the balance was found."""
    idx = list(range(len(level.deltas))) if steps is None else steps
    y = [level.deltas[u] for u in idx]
    xs = [[c.deltas[u] for u in idx] for c, _ in terms]
    kept = supported({j: sign for j, (_, sign) in enumerate(terms)}, y, xs)
    return [t for j, t in enumerate(terms) if j in kept]


def penalty(sst: float) -> float:
    """What one counter in a balance costs, for a level with this variance of its changes."""
    return max(MIN_GAIN * sst, EVENT_FLOOR)


def search(g: list[list[float]], b: list[float], yy: float, cost: list[float], live: list[int],
           start: dict[int, int], objective) -> tuple[dict[int, int], float]:
    """The local search from ``start``: repeatedly the change that lowers the objective most (any
    counter to +1, −1 or 0, or one chosen counter swapped for another of the same sign: near-identical
    counters can't be traded one step at a time), until none does. On the Gram matrix ``g``, with
    ``b`` the counters' dot products with the level's changes and ``yy`` the changes' own: the
    residual's dot product with every counter is kept up to date, so each candidate costs O(1)."""
    coef = dict(start)
    m = len(b)
    xr = [b[j] - sum(s * g[j][k] for k, s in coef.items()) for j in range(m)]
    sse = yy - 2 * sum(s * b[j] for j, s in coef.items()) + sum(s * t * g[j][k] for j, s in coef.items() for k, t in coef.items())
    tc = sum(cost[j] for j in coef)                   # what the chosen counters are charged
    cur = objective(sse, tc)
    for _ in range(4 * MAX_TERMS):
        best = None                                   # (objective, changes {j: new coefficient}, sse after, charge after)
        for j in live:                                # change one counter's coefficient
            old = coef.get(j, 0)
            for new in (-1, 0, 1):
                if new == old or (new and not old and len(coef) >= MAX_TERMS):
                    continue
                d = new - old
                e = sse - 2 * d * xr[j] + d * d * g[j][j]
                t = tc + cost[j] * ((new != 0) - (old != 0))
                o = objective(e, t)
                if o < cur - 1e-9 and (best is None or o < best[0]):
                    best = (o, {j: new}, e, t)
        for j, sign in coef.items():                  # swap a chosen counter for another, same sign
            for k in live:
                if k in coef:
                    continue
                e = sse + 2 * sign * (xr[j] - xr[k]) + g[j][j] + g[k][k] - 2 * g[j][k]
                t = tc - cost[j] + cost[k]
                o = objective(e, t)
                if o < cur - 1e-9 and (best is None or o < best[0]):
                    best = (o, {j: 0, k: sign}, e, t)
        if best is None:
            break
        cur, changes, sse, tc = best
        for j, new in changes.items():
            d = new - coef.get(j, 0)
            for i in range(m):
                xr[i] -= d * g[i][j]
            if new:
                coef[j] = new
            else:
                coef.pop(j, None)
    return coef, max(0.0, sse)


def starts_for(ls: LeastSquares, y: list[float], b: list[float], yy: float, live: list[int]) -> list[dict[int, int]]:
    """Where the search starts: least squares on all counters rounded to +1, −1 or 0; nothing;
    each counter alone, with the sign that fits; and the ``PAIR_STARTS`` pairs with the least
    error. A balance that is really one counter or a pair is reached from its own start whatever
    least squares, pulled about by collinear counters, suggests."""
    out: list[dict[int, int]] = [{}]
    rounded = {j: (1 if v > 0 else -1) for j, v in enumerate(ls.solve(y)) if abs(v) >= 0.5 and j in live}
    if rounded and len(rounded) <= MAX_TERMS:
        out.append(rounded)
    out += [{j: (1 if b[j] >= 0 else -1)} for j in live]
    pairs = []
    g = ls.g
    for a, j in enumerate(live):
        for k in live[a + 1:]:
            for sj in (1, -1):
                for sk in (1, -1):
                    e = yy - 2 * (sj * b[j] + sk * b[k]) + g[j][j] + g[k][k] + 2 * sj * sk * g[j][k]
                    pairs.append((e, {j: sj, k: sk}))
    pairs.sort(key=lambda p: p[0])
    out += [p[1] for p in pairs[:PAIR_STARTS]]
    return out


def balance(level, counters: list, ls: LeastSquares | None = None, steps: list[int] | None = None
            ) -> tuple[list[tuple], float, float]:
    """The counters (each +1 or −1) that best explain the level's changes over ``steps`` (every
    step by default); and the balance's R² and the share of steps it explains exactly.

    The search (``search``) runs from every start in ``starts_for`` and the best objective is kept.
    Last, any small set of chosen counters that cancels to nothing is dropped, and any that equals
    one unchosen counter is replaced by it (``sd.collinear``). ``ls`` is the counters' least squares
    over the same steps, when the caller has it (it is the same for every level)."""
    y = level.deltas
    xs = [c.deltas for c in counters]
    if steps is not None:
        y = [y[u] for u in steps]
        xs = [[x[u] for u in steps] for x in xs]
    return balance_of(y, xs, counters, ls)


def balance_of(y: list[float], xs: list[list[float]], counters: list, ls: LeastSquares | None = None
               ) -> tuple[list[tuple], float, float]:
    """``balance`` on a level's changes ``y`` per unit and each counter's ``xs`` over the same units:
    the search itself, which the shift null (``sd.identify``) also runs on a level moved in time."""
    mean = sum(y) / max(1, len(y))
    sst = sum((v - mean) ** 2 for v in y) or 1e-12
    if not counters or not y:
        return [], 0.0, sum(1 for v in y if abs(v) <= 0.5) / max(1, len(y))
    pen = penalty(sst)
    # a measurement (labelled with the instrument or the machine that wrote it) is preferred over
    # Orchestra's generic event count of the same thing when the two fit about equally: it costs a
    # little less to use
    cost = [pen * (1.0 if (c.labels.get("instrument") or c.labels.get("machine")) else 1.1) for c in counters]
    bonus = max(0.0, pen - COMPLETION * EVENT_FLOOR)

    def objective(e: float, charge: float) -> float:
        """Squared error plus what the chosen counters are charged, less the completion bonus."""
        return e + charge - (bonus if e < EXACT_SSE else 0.0)

    if ls is None:
        ls = LeastSquares(xs)
    yy = sum(v * v for v in y)
    live = [j for j, x in enumerate(xs) if enough_events(x) and resolvable(x, yy)]
    rare = [j for j, x in enumerate(xs) if any(x) and j not in live and resolvable(x, yy)]
    b = ls.against(y)
    best = None
    seen: set[tuple] = set()
    for start in starts_for(ls, y, b, yy, live):
        coef, sse = search(ls.g, b, yy, cost, live, start, objective)
        if (sig := tuple(sorted(coef.items()))) in seen:
            continue
        seen.add(sig)
        # completed before it is compared (``completed``): an exact balance a rare counter
        # completes earns its bonus only completed, and short of it may score worse than nothing,
        # or than a stand-in's near miss, which no rare counter makes exact
        coef, sse = completed(ls.g, b, coef, sse, rare)
        if (kept := supported(coef, y, xs)) != coef:
            coef = kept
            sse = max(0.0, yy - 2 * sum(s * b[j] for j, s in coef.items())
                      + sum(s * t * ls.g[j][k] for j, s in coef.items() for k, t in coef.items()))
        key = (objective(sse, sum(cost[j] for j in coef)), len(coef), sorted(coef.items()))
        if best is None or key < best[0]:
            best = (key, coef, sse)
    coef = best[1]
    resid = [v - sum(sign * xs[j][i] for j, sign in coef.items()) for i, v in enumerate(y)]
    sse = sum(r * r for r in resid)
    while True:
        before = len(coef), sorted(coef.items())
        coef, resid, sse = cancel(coef, resid, sse, xs, pen, sst)
        coef, resid, sse = merge(coef, resid, sse, xs, ls.g, cost, objective)
        if (len(coef), sorted(coef.items())) == before:
            break
    coef, resid, sse = complete(coef, resid, sse, xs, rare)
    if (kept := supported(coef, y, xs)) != coef:
        coef = kept
        resid = [v - sum(sign * xs[j][i] for j, sign in coef.items()) for i, v in enumerate(y)]
        sse = sum(r * r for r in resid)
    if not any(j in live for j in coef) and not (sse < EXACT_SSE and sum(sum(abs(v) for v in xs[j]) for j in coef) >= MIN_EVENTS):
        coef, resid, sse = {}, list(y), yy            # rare counters alone: a balance only when exact on enough events between them
    chosen = [(counters[j], sign) for j, sign in sorted(coef.items())]
    exact = sum(1 for r in resid if abs(r) <= 0.5) / len(resid)
    return chosen, 1 - sse / sst, exact


def resolvable(x: list[float], yy: float) -> bool:
    """Whether a counter can be weighed against a level in floating point: the search works on
    sums of squares, and a counter whose changes are millions of times the level's (the bytes a
    node sent, against a pool that gains two connections now and then) has squares whose rounding
    alone is larger than the level's whole variance. Such a counter cannot balance it (it would
    take another as large to cancel it to the last digit), and searched it only adds rounding:
    every pair with it looks exact, and crowds out the pair that is."""
    return sum(v * v for v in x) <= RESOLVABLE * max(yy, 1.0)


def fit_of(level, terms: list[tuple], steps: list[int] | None = None) -> tuple[float, float]:
    """A given balance's R² and share of exact steps, over ``steps``."""
    idx = list(range(len(level.deltas))) if steps is None else steps
    y = [level.deltas[u] for u in idx]
    resid = [v - sum(sign * c.deltas[u] for c, sign in terms) for v, u in zip(y, idx)]
    mean = sum(y) / max(1, len(y))
    sst = sum((v - mean) ** 2 for v in y) or 1e-12
    return 1 - sum(r * r for r in resid) / sst, sum(1 for r in resid if abs(r) <= 0.5) / max(1, len(resid))
