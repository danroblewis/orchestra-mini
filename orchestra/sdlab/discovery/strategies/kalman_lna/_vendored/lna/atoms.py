# Vendored for kalman_lna from sdlab/discovery/strategies/lna/atoms.py at commit ec2f9e8 (imports rewritten to the
# vendored copies; otherwise unchanged). Do not edit: re-vendor instead.
"""Flow atoms: the smallest flows a model is made of, each with one number.

    in      outside -> s              theta (a constant, over a piece of steps)
    out     s -> outside              theta x s
    move    a -> b                    theta x a
    link    outside -> b              theta x a   (nothing leaves a)
    grow    outside -> a              theta x a   (a stock that grows with itself: births)
    drain   c -> outside              theta x a   (c drains faster when a is high)
    gap     outside -> b              -theta x a  (b's outside inflow falls as a rises: a goal gap;
                                                   one draw with b's inflow, so its noise is theirs)
    a move may carry `uses` [(c, per)]: each unit moved also uses up per of c (a co-flow)

An atom's number may hold in one era only (`era`), and an inflow over a span of steps (`span`).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class Atom:
    kind: str
    src: int | None = None
    dst: int | None = None
    driver: int | None = None
    uses: tuple = ()                 # ((stock index, per), ...)
    era: int | None = None           # index into the configurations, or None: every step
    span: tuple | None = None        # (a, b): steps a..b-1 (an inflow piece)

    def key(self) -> tuple:
        """The atom without its era or span (one flow of the model)."""
        return (self.kind, self.src, self.dst, self.driver, self.uses)

    def with_(self, **kw) -> "Atom":
        return replace(self, **kw)


def stoich(atoms: list[Atom], S: int) -> np.ndarray:
    V = np.zeros((S, len(atoms)))
    for f, a in enumerate(atoms):
        if a.src is not None:
            V[a.src, f] -= 1
        if a.dst is not None:
            V[a.dst, f] += -1 if a.kind == "gap" else 1
        for c, per in a.uses:
            V[c, f] -= per
    return V


def features(atoms: list[Atom], prev: np.ndarray, cfg: np.ndarray | None) -> np.ndarray:
    S, T = prev.shape
    Z = np.empty((len(atoms), T))
    for f, a in enumerate(atoms):
        z = np.ones(T) if a.driver is None else np.maximum(prev[a.driver], 0.0)
        if a.era is not None and cfg is not None:
            z = np.where(cfg == a.era, z, 0.0)
        if a.span is not None:
            m = np.zeros(T)
            m[a.span[0]:a.span[1]] = 1.0
            z = z * m
        Z[f] = z
    return Z


def pooling(atoms: list[Atom]) -> tuple[np.ndarray, np.ndarray]:
    """Per atom: the stock whose outside inflow it is part of (-1: none), and its sign there."""
    pool = np.array([a.dst if a.kind in ("in", "gap") else -1 for a in atoms], dtype=int)
    sign = np.array([-1.0 if a.kind == "gap" else 1.0 for a in atoms])
    return pool, sign


def variances(atoms: list[Atom], theta: np.ndarray, Z: np.ndarray, S: int, eps: float = 0.05) -> np.ndarray:
    """Each stock's counting-noise variance, step by step (stocks x steps)."""
    V = stoich(atoms, S)
    q = theta[:, None] * Z
    pool, sign = pooling(atoms)
    rest = pool < 0
    out = (V[:, rest] ** 2) @ q[rest]
    net = np.zeros_like(out)
    if (~rest).any():
        np.add.at(net, pool[~rest], sign[~rest, None] * q[~rest])
    return out + np.maximum(np.abs(net), eps)


def superset(S: int, unit: list[str], final: set[int], group: list, allow_links: bool = True, gaps: bool = True) -> list[Atom]:
    """Every atom between the stocks the page allows (same unit for moves; groups as the matrix
    reads them: a stock affects another only in its own group, or when either has none)."""
    def may(j, s):
        return j == s or not group[j] or not group[s] or group[j] == group[s]

    out = []
    for s in range(S):
        out.append(Atom("in", dst=s))
        if s not in final:
            out.append(Atom("out", src=s, driver=s))
        out.append(Atom("grow", dst=s, driver=s))
    for a in range(S):
        for b in range(S):
            if a == b or not may(a, b):
                continue
            if unit[a] == unit[b] and a not in final:
                out.append(Atom("move", src=a, dst=b, driver=a))
            if allow_links:
                out.append(Atom("link", dst=b, driver=a))
            if b not in final:
                out.append(Atom("drain", src=b, driver=a))
    if gaps:
        out += [Atom("gap", dst=b, driver=a) for b in range(S) for a in range(S) if may(a, b)]
    return out


def restricted(S: int, coef: list[dict[int, float]], unit: list[str], final: set[int]) -> list[Atom]:
    """The atoms that can make the numbers a fitted matrix has (coef[s]: {j: B[s][j]} for its nonzero
    entries), each entry in every reading the averages allow: a stock's own number as an outflow or
    growth; a gain as a move (same unit) or a link; a loss as a drain."""
    out = [Atom("in", dst=s) for s in range(S)]
    for s in range(S):
        for j, k in coef[s].items():
            if j == s:
                if s not in final:
                    out.append(Atom("out", src=s, driver=s))
                out.append(Atom("grow", dst=s, driver=s))
            elif k > 0:
                if unit[j] == unit[s] and j not in final:
                    out.append(Atom("move", src=j, dst=s, driver=j))
                out.append(Atom("link", dst=s, driver=j))
            elif s not in final:
                out.append(Atom("drain", src=s, driver=j))
    return list(dict.fromkeys(out))
