"""Sets of counters that are one: a small set of a balance's counters whose signed sum is nothing
(``cancel``), or is exactly another counter (``merge``).

The balance search changes one counter at a time, so it cannot leave such a set by itself: each
member alone is a big change for the worse. Here the set is traded as a whole, when the objective
(squared error plus a penalty per counter) falls. A set counts as equal to its stand-in when the
two differ in at most ``CANCEL_STEPS`` of the steps either moved in: counters stamped a moment
apart put the odd event in the next step. Copied from the old Orchestra's ``sd/collinear.py``."""

from __future__ import annotations

from itertools import combinations

CANCEL_SIZE = 4         # a cancelling or collinear set is at most this many counters
CANCEL_STEPS = 0.1      # ... whose signed sum differs from its stand-in in at most this share of the steps it moved in
CANCEL_SHARE = 0.05     # ... and (for cancel) whose removal costs less than this share of the level's variance (or its penalty)


def cancel(coef: dict[int, int], resid: list[float], sse: float, xs: list[list[float]], pen: float, sst: float
           ) -> tuple[dict[int, int], list[float], float]:
    """Drop any small set of the chosen counters whose signed sum is about zero in every step: two
    near-identical counters with opposite signs, or a triple like ``+made −baked −late`` that is
    identically zero. Such a set explains nothing and only soaks up noise, and the search cannot
    remove it one counter at a time (each alone is a big change). It goes when removing it costs
    less than it is charged (its size × the penalty) or less than ``CANCEL_SHARE`` of the variance."""
    changed = True
    while changed and len(coef) >= 2:
        changed = False
        n = len(resid)
        for size in range(2, min(CANCEL_SIZE, len(coef)) + 1):
            for sub in combinations(sorted(coef), size):
                s = [sum(coef[j] * xs[j][i] for j in sub) for i in range(n)]
                moved = sum(1 for i in range(n) if any(xs[j][i] for j in sub))
                if not moved or sum(1 for v in s if abs(v) > 0.5) > CANCEL_STEPS * moved:
                    continue
                r2 = [r + v for r, v in zip(resid, s)]
                e = sum(r * r for r in r2)
                if e - sse < max(CANCEL_SHARE * sst, size * pen):
                    resid, sse = r2, e
                    for j in sub:
                        coef.pop(j)
                    changed = True
                    break
            if changed:
                break
    return coef, resid, sse


def merge(coef: dict[int, int], resid: list[float], sse: float, xs: list[list[float]], g: list[list[float]],
          cost: list[float], objective) -> tuple[dict[int, int], list[float], float]:
    """Replace a small set of the chosen counters by the one unchosen counter that equals their
    signed sum (``order.queued − app_orders − returns − failed`` is ``orders_taken``), when the
    objective falls: the one counter costs one penalty where the set cost several. The mirror of
    ``cancel``. ``g`` is the counters' Gram matrix, which says cheaply whether a counter is near a
    sum before the steps are compared; ``objective(sse, charge)`` is the search's, ``charge`` the
    chosen counters' ``cost`` summed."""
    n, m = len(resid), len(xs)
    changed = True
    while changed and len(coef) >= 2:
        changed = False
        free = [k for k in range(m) if k not in coef and any(xs[k])]
        for size in range(2, min(CANCEL_SIZE, len(coef)) + 1):
            for sub in combinations(sorted(coef), size):
                norm = sum(coef[j] * coef[l] * g[j][l] for j in sub for l in sub)
                if norm <= 0:
                    continue
                s = None
                for k in free:
                    for t in (1, -1):
                        if norm - 2 * t * sum(coef[j] * g[j][k] for j in sub) + g[k][k] > 0.5 * norm:
                            continue
                        if s is None:
                            s = [sum(coef[j] * xs[j][i] for j in sub) for i in range(n)]
                        moved = sum(1 for i in range(n) if s[i] or xs[k][i])
                        if not moved or sum(1 for i in range(n) if abs(s[i] - t * xs[k][i]) > 0.5) > CANCEL_STEPS * moved:
                            continue
                        r2 = [r + a - t * b for r, a, b in zip(resid, s, xs[k])]
                        e = sum(r * r for r in r2)
                        after = {j: v for j, v in coef.items() if j not in sub} | {k: t}
                        if objective(e, sum(cost[j] for j in after)) < objective(sse, sum(cost[j] for j in coef)) - 1e-9:
                            coef, resid, sse, changed = after, r2, e, True
                            break
                    if changed:
                        break
                if changed:
                    break
            if changed:
                break
    return coef, resid, sse
