"""Choosing the atoms by the Kalman likelihood: the superset fitted, the variance beyond counting
noise freed only where it pays, backward elimination by BIC (Wald-screened), drain <-> gap swaps,
outside inflows cut into pieces where the smoothed process noise changes level, and a reporting
scale for a stock that conservation pins."""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import lsq_linear

from ...changes import pelt
from ._vendored.lna.atoms import Atom, stoich
from .model import Fitted, Problem

TRIAL_ITER = 80


def start_theta(pr: Problem, atoms: list[Atom]) -> np.ndarray:
    """Non-negative least squares of each step's change on the atoms' features, on the reference
    path (a smoothed path, so the errors in the levels bias it little), each row weighted by its spread."""
    if not atoms:
        return np.zeros(0)
    V, drv, pool, sign, act = pr.arrays(atoms)
    ref = pr.zref                                       # level at each step's start (+ injection)
    K = ref.shape[1]
    # each step's change on the reference path: the next step's start level minus this one's
    dy = np.zeros((pr.n, K))
    dy[:, 1:-1] = ref[:, 2:] - ref[:, 1:-1]
    steps = np.arange(1, K - 1)
    Z = np.where(drv[:, None] >= 0, ref[np.maximum(drv, 0)], 1.0) * act         # F x K
    w = 1.0 / np.sqrt(np.maximum(dy[:, steps].var(axis=1), 1e-6))
    Amat = np.concatenate([(V[:, s][:, None] * Z[:, steps]).T * w[s] for s in range(pr.n)], axis=0)
    b = np.concatenate([dy[s, steps] * w[s] for s in range(pr.n)])
    sc = np.maximum(np.abs(Amat).max(axis=0), 1e-12)
    r = lsq_linear(Amat / sc, b, bounds=(0, np.inf), method="bvls" if len(atoms) < 60 else "trf")
    return r.x / sc


def warm(cur: Fitted | None, atoms: list[Atom], extra: dict | None = None) -> np.ndarray:
    """Starting numbers for `atoms` from a fitted model (by atom), `extra` (atom -> number) for new ones."""
    have = {} if cur is None else {a: th for a, th in zip(cur.atoms, cur.theta)}
    extra = extra or {}
    return np.array([extra.get(a, have.get(a, 1e-3)) for a in atoms], dtype=float)


def refit(pr: Problem, atoms: list[Atom], cur: Fitted | None, extra: dict | None = None, trial: bool = False) -> Fitted:
    th0 = warm(cur, atoms, extra) if cur is not None else start_theta(pr, atoms)
    return pr.fit(atoms, start=cur, theta0=np.maximum(th0, 1e-6), maxiter=TRIAL_ITER if trial else 400, ftol=1e-7 if trial else 1e-9)


def numbers(atoms: list[Atom], idx) -> int:
    return sum(1 + len(atoms[i].uses) for i in idx)


def price(pr: Problem) -> float:
    """The BIC price of one number in the negative log-likelihood: log(points) / 2."""
    return 0.5 * math.log(max(10, int(pr.lay.mask.any(axis=1).sum())))


def bic(pr: Problem, cur: Fitted) -> float:
    return cur.nll + price(pr) * (numbers(cur.atoms, range(len(cur.atoms))) + int(pr.free_tau.sum()) + len(pr.noise) + len(pr.scaled))


def free_variance(pr: Problem, cur: Fitted) -> Fitted:
    """Each stock's variance beyond counting noise is kept free only where fixing it at 0 costs
    more than the BIC price."""
    p = price(pr)
    for s in np.argsort(cur.tau):
        if not pr.free_tau[s]:
            continue
        pr.free_tau[s] = False
        g = refit(pr, cur.atoms, cur, trial=True)
        if g.nll - cur.nll < p:
            cur = g
        else:
            pr.free_tau[s] = True
    return refit(pr, cur.atoms, cur)


def noise_free(pr: Problem, cur: Fitted) -> Fitted:
    """The measurement-error numbers each kept only where fixing it at 0 costs more than the BIC price."""
    p = price(pr)
    for name in ("alpha", "rho2"):
        if name not in pr.noise:
            continue
        keep = pr.noise
        pr.noise = tuple(x for x in keep if x != name)
        g = refit(pr, cur.atoms, Fitted(cur.atoms, cur.theta, cur.nll, cur.tau, 0.0 if name == "alpha" else cur.alpha,
                                        0.0 if name == "rho2" else cur.rho2, cur.scale), trial=True)
        if g.nll - cur.nll < p:
            cur = g
        else:
            pr.noise = keep
    return refit(pr, cur.atoms, cur)


def wald(pr: Problem, cur: Fitted) -> dict:
    """Each flow's (key's) approximate cost of removal per number: half its numbers' squared
    distance from 0 in their standard errors (Fisher information of the filter). Inflow pieces are
    not weighed here."""
    pt, pi, fim = pr.information(cur)
    pos = {int(pi[k]): k for k in range(len(pt)) if pt[k] == 0}
    try:
        C = np.linalg.pinv(fim + 1e-9 * np.eye(len(fim)) * max(1e-12, float(np.abs(np.diag(fim)).max())))
    except np.linalg.LinAlgError:
        return {}
    out = {}
    for key in dict.fromkeys(a.key() for a in cur.atoms):
        idx = [i for i, a in enumerate(cur.atoms) if a.key() == key]
        if any(cur.atoms[i].span is not None for i in idx):
            continue
        th = cur.theta[idx]
        c = C[np.ix_([pos[i] for i in idx], [pos[i] for i in idx])]
        try:
            w = 0.5 * float(th @ np.linalg.solve(c, th))
        except np.linalg.LinAlgError:
            w = 0.0
        out[key] = (w / numbers(cur.atoms, idx), idx)
    return out


def drop_zeros(pr: Problem, cur: Fitted) -> Fitted:
    live = [i for i, th in enumerate(cur.theta) if th > 0]
    if len(live) == len(cur.atoms):
        return cur
    return refit(pr, [cur.atoms[i] for i in live], cur)


def prune(pr: Problem, cur: Fitted, tries: int = 3) -> Fitted:
    """Flows whose removal costs less than the BIC price go, the cheapest first. A batch of the
    cheapest (by Wald) is tried first, all at once (a large superset would otherwise take one refit
    per flow); when it costs too much, one flow at a time."""
    p = price(pr)
    while True:
        live = [i for i, th in enumerate(cur.theta) if th > 0]
        if len(live) < len(cur.atoms):
            cur = refit(pr, [cur.atoms[i] for i in live], cur)
            continue
        if not cur.atoms:
            return cur
        order = sorted(wald(pr, cur).items(), key=lambda kv: kv[1][0])
        done = False
        cheap = [idx for _key, (w, idx) in order if w < 0.25 * p]
        if len(cheap) > 1:
            drop = {i for idx in cheap for i in idx}
            rest = [j for j in range(len(cur.atoms)) if j not in drop]
            g = refit(pr, [cur.atoms[j] for j in rest], cur, trial=True)
            if (g.nll - cur.nll) / numbers(cur.atoms, sorted(drop)) < p:
                cur, done = refit(pr, g.atoms, g), True
        if done:
            continue
        for key, (w, idx) in order[:tries]:
            if w > 4 * p:
                break
            rest = [j for j in range(len(cur.atoms)) if j not in idx]
            g = refit(pr, [cur.atoms[j] for j in rest], cur, trial=True)
            if (g.nll - cur.nll) / numbers(cur.atoms, idx) < p:
                cur, done = refit(pr, g.atoms, g), True
                break
        if not done:
            return cur


def _alternatives(a: Atom, atoms: list[Atom]) -> list[Atom]:
    if a.era is not None or a.span is not None:
        return []
    if a.kind in ("drain", "out") and any(b.kind == "in" and b.dst == a.src for b in atoms):
        return [a.with_(kind="gap", dst=a.src, src=None)]
    if a.kind == "gap":
        return [a.with_(kind="out" if a.driver == a.dst else "drain", src=a.dst, dst=None)]
    if a.kind == "link" and any(b.kind == "move" and b.src == a.driver and b.dst == a.dst for b in atoms):
        return []
    return []


def swaps(pr: Problem, cur: Fitted, rounds: int = 4, margin: float = 2.0) -> Fitted:
    """Drain <-> goal gap (same means, different noise): a gap must win by `margin`."""
    for _ in range(rounds):
        best = None
        allow = getattr(pr.lay, "allow", None)          # the declared boundary (inside.py)
        for i, a in enumerate(cur.atoms):
            for b in _alternatives(a, cur.atoms):
                if b in cur.atoms or (allow is not None and not allow(b)):
                    continue
                atoms = list(cur.atoms)
                atoms[i] = b
                g = refit(pr, atoms, cur, extra={b: cur.theta[i]}, trial=True)
                need = margin if b.kind == "gap" else 1e-6
                if g.nll < cur.nll - need and (best is None or g.nll < best.nll):
                    best = g
        if best is None:
            return cur
        cur = refit(pr, best.atoms, best)
    return cur


COUNTS = (1, 2, 3, 4, 6, 8, 11, 16, 23, 32, 45, 64)   # the numbers of pieces tried (then their neighbours)


def merges(y: np.ndarray, cuts: list[int]) -> dict[int, list[int]]:
    """{pieces: cuts}: the finest cuts and every coarser set made by merging, one at a time, the
    two neighbouring pieces whose means differ least (in the squared error they add)."""
    cuts = list(cuts)
    out = {}
    while True:
        out[len(cuts) - 1] = list(cuts)
        if len(cuts) <= 2:
            return out
        best, at = None, None
        for i in range(1, len(cuts) - 1):
            a, b, c = cuts[i - 1], cuts[i], cuts[i + 1]
            n1, n2 = b - a, c - b
            cost = n1 * n2 / (n1 + n2) * (float(y[a:b].mean()) - float(y[b:c].mean())) ** 2
            if best is None or cost < best:
                best, at = cost, i
        del cuts[at]


PIECE = 1.5          # a piece costs this many numbers: its value, and where it starts (chosen among many)


def pieces(pr: Problem, cur: Fitted, min_len: int = 5, mult: float = 1.0) -> Fitted:
    """Each stock's outside inflow cut afresh where the smoothed process noise (what the flows leave
    unexplained, step by step), with the inflow added back, changes level (PELT; the cut sets made
    by merging its pieces, tried at a few counts); kept if the likelihood gains more than the price
    of the extra pieces.

    A stock that grows with itself beside its outflow (births and deaths: one number of variance
    beyond the drain) can stand in for a changing inflow as a random walk, and then the inflow's
    changes show only as bursts in the process noise; so the cuts are also read off the model
    without that growth (and that model, with pieces, is a candidate too)."""
    p = price(pr)
    K = pr.lay.K
    for s in range(pr.n):
        idx = [i for i, a in enumerate(cur.atoms) if a.kind == "in" and a.dst == s and a.era is None]
        if not idx:
            continue
        bases = [(cur, 0)]
        grows = [i for i, a in enumerate(cur.atoms) if a.kind == "grow" and a.dst == s and a.era is None]
        if grows and any(a.kind == "out" and a.src == s for a in cur.atoms):
            rest = [i for i in range(len(cur.atoms)) if i not in grows]
            bases.append((refit(pr, [cur.atoms[i] for i in rest], cur, trial=True), len(grows)))
        best = None
        for base, removed in bases:
            got = _pieces_of(pr, cur, base, s, removed, p, K, min_len, mult)
            if got is not None and (best is None or got[0] > best[0]):
                best = got
        if best is not None:
            cur = refit(pr, best[1].atoms, best[1])
    return cur


def _pieces_of(pr, cur, base, s, removed, p, K, min_len, mult):
    """The best (gain, fitted) over the cut sets read off `base` for stock s (gain against cur)."""
    idx = [i for i, a in enumerate(base.atoms) if a.kind == "in" and a.dst == s and a.era is None]
    n_old = len([i for i, a in enumerate(cur.atoms) if a.kind == "in" and a.dst == s and a.era is None])
    try:
        wm = pr.smoothed(base)[3]
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(wm[s])):
        return None
    own = np.zeros(K)
    for i in idx:
        a = base.atoms[i]
        m = np.zeros(K)
        lo, hi = a.span if a.span else (1, K)
        m[lo:hi] = 1.0
        own += base.theta[i] * m
    y = (wm[s] + own)[1:]
    keep = [i for i in range(len(base.atoms)) if i not in idx]
    levels = merges(y, pelt(y, min_len, mult))
    tried: dict[int, float] = {}
    best = None

    def attempt(k):
        nonlocal best
        if k in tried or k not in levels:
            return
        cuts = levels[k]
        new = [(a + 1, b + 1, max(1e-3, float(y[a:b].mean()))) for a, b in zip(cuts[:-1], cuts[1:])]
        atoms = [base.atoms[i] for i in keep] + [Atom("in", dst=s, span=None if len(new) == 1 else (a, b)) for a, b, _ in new]
        if sorted(atoms, key=repr) == sorted(cur.atoms, key=repr):
            tried[k] = 0.0
            return
        extra = {at: v for at, (_, _, v) in zip(atoms[len(keep):], new)}
        g = refit(pr, atoms, base, extra=extra, trial=True)
        gain = cur.nll - g.nll - PIECE * p * (len(new) - n_old) + p * removed
        tried[k] = gain
        if gain > 0 and (best is None or gain > best[0]):
            best = (gain, g)

    for k in COUNTS:
        attempt(min(k, max(levels)))
    top = max(tried, key=tried.get)
    for k in range(max(1, top - 3), top + 4):          # the best count's neighbours
        attempt(k)
    return best


def merge_eras(pr: Problem, cur: Fitted) -> Fitted:
    """A flow whose numbers per era the data does not tell apart gets one number."""
    if pr.cfg is None:
        return cur
    p = price(pr)
    while True:
        best = None
        for key in dict.fromkeys(a.key() for a in cur.atoms if a.era is not None):
            split = [i for i, a in enumerate(cur.atoms) if a.key() == key and a.era is not None]
            if len(split) < 2:
                continue
            rest = [j for j in range(len(cur.atoms)) if j not in split]
            one = cur.atoms[split[0]].with_(era=None)
            g = refit(pr, [cur.atoms[j] for j in rest] + [one], cur, extra={one: float(np.mean(cur.theta[split]))}, trial=True)
            cost = (g.nll - cur.nll) / (len(split) - 1)
            if cost < p and (best is None or cost < best[0]):
                best = (cost, g)
        if best is None:
            return cur
        cur = refit(pr, best[1].atoms, best[1])


SCALE_MARGIN = 0.0   # a scale must lower the BIC by this many prices of a number


def scales(pr: Problem, cur: Fitted, run_search) -> Fitted:
    """A stock counted with a bias (its reported levels s x the true ones) where conservation pins
    s: a stock a move between two measured stocks joins (through an unmeasured stock the scale is
    pinned only weakly). Each such stock's scale is freed in turn and the search re-run; kept when
    the BIC falls."""
    best_bic = bic(pr, cur)
    for s in range(pr.nm):
        if s in pr.scaled:
            continue
        joined = any(a.kind == "move" and s in (a.src, a.dst) and a.src < pr.nm and a.dst < pr.nm and th > 0
                     for a, th in zip(cur.atoms, cur.theta))
        if not joined:
            continue
        before, taus = pr.scaled, pr.free_tau.copy()
        pr.scaled = before + (s,)
        try:
            # the variance beyond counting noise was freed for the model without the scale (a
            # stock's reports at 1.2x its counts spread 1.44x as much): weighed again with it
            g = free_variance(pr, run_search(refit(pr, cur.atoms, cur)))
        except Exception:
            pr.scaled, pr.free_tau = before, taus
            raise
        b = bic(pr, g)
        pr.notes.append(f"scale {pr.names[s]}: {float(g.scale[s]):.3f}, BIC gain {best_bic - b:.1f}")
        if b < best_bic - SCALE_MARGIN * price(pr) and abs(float(g.scale[s]) - 1.0) > 1e-3:
            cur, best_bic = g, b
        else:
            pr.scaled, pr.free_tau = before, taus
    return cur


def superset_atoms(S: int, unit: list[str], final: set[int], group: list) -> list[Atom]:
    from ._vendored.lna.atoms import superset
    return superset(S, unit, final, group, gaps=False)


def stoich_ok(atoms: list[Atom], S: int) -> bool:
    return stoich(atoms, S).shape[0] == S
