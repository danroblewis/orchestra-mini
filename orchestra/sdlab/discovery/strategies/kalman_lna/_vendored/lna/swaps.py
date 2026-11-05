# Vendored for kalman_lna from sdlab/discovery/strategies/lna/swaps.py at commit ec2f9e8 (imports rewritten to the
# vendored copies; otherwise unchanged). Do not edit: re-vendor instead.
# (kalman_lna, 2026-10-20) Edited after vendoring for the declared boundary: atoms outside it are not offered
# (kalman_lna/inside.py); see docs/research/kalman_lna.md.
"""Readings with the same averages and the same count of numbers, told apart by the noise alone.

A drain of b driven by a (b loses k x a, drawn on its own) and a goal gap in b's outside inflow
(the inflow is c - k x a, one draw) give b the same mean change; their variances differ by 2 k x a.
Each drain of a stock with an outside inflow is tried as a gap (and each gap back as a drain), and
the swap is kept when the likelihood prefers it.
"""

from __future__ import annotations

from orchestra.sdlab.discovery.strategies.kalman_lna._vendored.lna.fit import TRIAL, Data, Fitted, fit, polish


def _alternatives(a, atoms):
    if a.era is not None or a.span is not None:
        return []
    if a.kind in ("drain", "out") and any(b.kind == "in" and b.dst == a.src for b in atoms):
        return [a.with_(kind="gap", dst=a.src, src=None)]
    if a.kind == "gap":
        return [a.with_(kind="out" if a.driver == a.dst else "drain", src=a.dst, dst=None)]
    return []


def swaps(cur: Fitted, d: Data, rounds: int = 6, margin: float = 2.0) -> Fitted:
    """A gap must beat the drain by `margin` (the drain is the usual reading); back the other way, any gain."""
    for _ in range(rounds):
        best = None
        for i, a in enumerate(cur.atoms):
            if cur.theta[i] <= 0:
                continue
            for b in _alternatives(a, cur.atoms):
                if b in cur.atoms or not getattr(d, "allow", lambda _: True)(b):     # (kalman_lna) the declared boundary
                    continue
                atoms = list(cur.atoms)
                atoms[i] = b
                g = fit(atoms, d, cur.theta, maxiter=TRIAL, tau=cur.tau)
                need = margin if b.kind == "gap" else 1e-6
                if g.nll < cur.nll - need and (best is None or g.nll < best.nll):
                    best = g
        if best is None:
            return cur
        cur = polish(best, d)
    return cur

