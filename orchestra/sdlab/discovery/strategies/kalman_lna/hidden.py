"""An unmeasured stock between two measured ones (A -> H -> B), as one more state of the filter.

Where the chosen model has B gaining in proportion to A (a move A -> B or a link A => B), the
alternative routes A's loss through H: a move A -> H at k and a move H -> B at d (a delay of 1 / d),
one number more than the move. H is never seen; the filter follows it from A's and B's reports (its
counting noise is shared with both, as a move's is). Kept when it beats the model as it stands by
the BIC price of its extra number plus a margin, with a delay H turns over many times in the run.
"""

from __future__ import annotations

import numpy as np

from ._vendored.lna.atoms import Atom
from . import search as S
from .model import Fitted, Problem

DELAYS = (2.0, 5.0, 10.0, 20.0)   # starting delays tried
MARGIN = 2.0                      # BIC margin, in units of the price of one number
SHORTEST = 1.5                    # a shorter delay is a move
TURNS = 15                        # H must turn over at least this many times in the run


def _path(a_ref: np.ndarray, k: float, d: float) -> np.ndarray:
    h = np.zeros_like(a_ref)
    x = k * float(a_ref[1]) / max(d, 1e-3)
    for i in range(len(a_ref)):
        h[i] = x
        x = (1 - d) * x + k * a_ref[i]
    return h


def candidates(cur: Fitted, nm: int) -> list[tuple[int, int, str]]:
    """(a, b, what H replaces): B gaining in proportion to A ("gain": the move or link A -> B), or A
    losing to outside while B has an outside inflow ("exit": A's outflow, B's inflow left to the search)."""
    out = []
    live = [(a, th) for a, th in zip(cur.atoms, cur.theta) if th > 0]
    for a, _ in live:
        if a.kind in ("move", "link") and a.driver is not None and a.dst is not None and a.driver < nm and a.dst < nm:
            if a.driver != a.dst and (a.driver, a.dst, "gain") not in out:
                out.append((a.driver, a.dst, "gain"))
    for a, _ in live:
        if a.kind == "out" and a.era is None and a.src < nm:
            for b, _ in live:
                if b.kind == "in" and b.dst != a.src and b.dst < nm and (a.src, b.dst, "gain") not in out and (a.src, b.dst, "exit") not in out:
                    out.append((a.src, b.dst, "exit"))
    return out


def hidden(pr: Problem, cur: Fitted, search, most: int = 1) -> tuple[Problem, Fitted]:
    """The model with a hidden stock where it pays; (pr, cur) unchanged otherwise. Every candidate is
    fitted from a few starting delays (its other flows as found); the best few, by BIC, are then
    searched (flows dropped or swapped) and the best kept if it beats the model as it stands."""
    if pr.hidden:
        return pr, cur
    base_bic = S.bic(pr, cur)
    p = S.price(pr)
    span_steps = pr.lay.K
    quick = []
    for a, b, what in candidates(cur, pr.nm):
        name = f"{pr.names[a]}~{pr.names[b]}"
        h = pr.n
        if what == "gain":
            drop = [i for i, x in enumerate(cur.atoms) if x.kind in ("move", "link") and x.driver == a and x.dst == b and x.era is None]
        else:
            drop = [i for i, x in enumerate(cur.atoms) if x.kind == "out" and x.src == a and x.era is None]
        k_in = float(sum(cur.theta[i] for i in drop))
        if k_in <= 0:
            continue
        keep = [x for i, x in enumerate(cur.atoms) if i not in drop]
        a_in, h_out = Atom("move", src=a, dst=h, driver=a), Atom("move", src=h, dst=b, driver=h)
        best_here = None
        for delay in DELAYS:
            d = 1.0 / delay
            ref = _path(pr.zref[a], k_in, d)
            p2 = Problem(pr.lay, pr.cfg, pr.deadline, hidden=[name], hidden_ref=ref[None, :])
            p2.free_tau[: pr.nm] = pr.free_tau[: pr.nm]
            p2.noise, p2.scaled = pr.noise, pr.scaled
            p2.set_reference(np.vstack([pr.ref[: pr.nm], ref[None, :]]), pr.yref2 ** 0.5)
            start = Fitted(keep, np.array([cur.theta[cur.atoms.index(x)] for x in keep]), cur.nll, np.append(cur.tau, 0.0),
                           cur.alpha, cur.rho2, cur.scale)
            g = S.refit(p2, keep + [a_in, h_out], start, extra={a_in: k_in, h_out: d}, trial=True)
            if not all(x in g.atoms for x in (a_in, h_out)) or g.theta[g.atoms.index(h_out)] <= 0 or g.theta[g.atoms.index(a_in)] <= 0:
                continue
            if best_here is None or g.nll < best_here[2].nll:
                best_here = (name, p2, g, what)
        if best_here is not None:
            quick.append(best_here)
    quick.sort(key=lambda q: S.bic(q[1], q[2]))
    best = None
    for name, p2, g, what in quick[:2]:
        if S.bic(p2, g) > base_bic + 4 * p:          # far behind before the search: not worth one
            break
        p2.refresh(g)
        g = S.refit(p2, g.atoms, g)
        a_in, h_out = [x for x in g.atoms if x.kind == "move" and p2.n - 1 in (x.src, x.dst)][:2] if len(g.atoms) else (None, None)
        g = search(p2, g)
        hs = [x for x, th in zip(g.atoms, g.theta) if th > 0 and x.kind == "move" and p2.n - 1 in (x.src, x.dst)]
        if len(hs) < 2:
            continue
        dd = float(next(th for x, th in zip(g.atoms, g.theta) if x.kind == "move" and x.src == p2.n - 1))
        b2 = S.bic(p2, g)
        pr.notes.append(f"hidden {name} ({what}, delay {1 / dd:.1f}): BIC gain {base_bic - b2:.1f}")
        if not (SHORTEST <= 1.0 / dd <= span_steps / TURNS):
            continue
        if b2 < base_bic - MARGIN * p and (best is None or b2 < best[0]):
            best = (b2, p2, g)
    if best is None:
        return pr, cur
    return best[1], best[2]
