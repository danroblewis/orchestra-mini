# Vendored for kalman_lna from sdlab/discovery/strategies/lna/coflows.py at commit ec2f9e8 (imports rewritten to the
# vendored copies; otherwise unchanged). Do not edit: re-vendor instead.
"""Co-flows: a move a -> b that uses up `per` of c as it goes.

In the averages a co-flow is a move plus a drain of c driven by a (c loses per x the move's rate),
so the two readings fit the means equally well. The covariance tells them apart: a co-flow's draw is
shared by a, b and c (c mirrors it at the ratio per), an independent drain is drawn on its own. So a
drain of c driven by a beside a move a -> b offers a co-flow atom with per = drain / move, the
pruning keeps whichever reading the likelihood prefers, and per is then refined.
"""

from __future__ import annotations

import numpy as np

from orchestra.sdlab.discovery.strategies.kalman_lna._vendored.lna.atoms import Atom
from orchestra.sdlab.discovery.strategies.kalman_lna._vendored.lna.fit import TRIAL, Data, Fitted, fit, polish


PER = (0.01, 100.0)          # the co-flow ratios offered (beyond them, a near-zero move or drain)


def offer(cur: Fitted) -> list[tuple[Atom, float]]:
    """(co-flow atom, starting number) for every drain beside a move from its driver."""
    out = []
    for i, dr in enumerate(cur.atoms):
        if dr.kind != "drain" or cur.theta[i] <= 0 or dr.era is not None:
            continue
        for j, mv in enumerate(cur.atoms):
            if mv.kind != "move" or mv.driver != dr.driver or cur.theta[j] <= 0 or dr.src in (mv.src, mv.dst) or mv.uses or mv.era is not None:
                continue
            per = float(cur.theta[i] / cur.theta[j])
            if not PER[0] <= per <= PER[1]:
                continue
            out.append((mv.with_(uses=((dr.src, per),)), float(cur.theta[j])))
    return out


def with_coflows(cur: Fitted, d: Data) -> Fitted:
    extra = [a for a in offer(cur) if a[0] not in cur.atoms]
    if not extra:
        return cur
    atoms = cur.atoms + [a for a, _ in extra]
    return fit(atoms, d, np.append(cur.theta, [th for _, th in extra]), tau=cur.tau)


def refine_per(cur: Fitted, d: Data, steps: int = 8) -> Fitted:
    """Each co-flow's per, by golden-section search on its log (the other numbers refitted)."""
    for i in range(len(cur.atoms)):
        a = cur.atoms[i]
        if not a.uses or cur.theta[i] <= 0:
            continue
        c, per = a.uses[0]

        def at(p, cur=cur, i=i, a=a, c=c):
            atoms = list(cur.atoms)
            atoms[i] = a.with_(uses=((c, float(p)),))
            return fit(atoms, d, cur.theta, maxiter=TRIAL, tau=cur.tau)

        lo, hi = np.log(per / 1.6), np.log(per * 1.6)
        g = (np.sqrt(5) - 1) / 2
        x1, x2 = hi - g * (hi - lo), lo + g * (hi - lo)
        f1, f2 = at(np.exp(x1)), at(np.exp(x2))
        for _ in range(steps):
            if f1.nll < f2.nll:
                hi, x2, f2 = x2, x1, f1
                x1 = hi - g * (hi - lo)
                f1 = at(np.exp(x1))
            else:
                lo, x1, f1 = x1, x2, f2
                x2 = lo + g * (hi - lo)
                f2 = at(np.exp(x2))
        best = min((f1, f2), key=lambda h: h.nll)
        if best.nll < cur.nll:
            cur = polish(best, d)
    return cur
