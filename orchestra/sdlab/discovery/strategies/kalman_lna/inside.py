"""lna's atoms kept inside the announced boundary (boundary.py, docs/research/declared-boundary.md).

An atom meets the outside on one side of one stock: `in`, `grow`, `link` and `gap` are inflows from
outside into dst; `out` and `drain` are outflows from src to outside; a `move` stays inside. Its kind
is "measured" when it reads a measured input (a driver among the input rows), else "estimate". An
atom is offered only where the boundary allows that stock, side and kind; a stock nobody measures (a
hidden stock, index beyond the measured ones) is inside the boundary and is never restricted here.
With no boundary declared every atom is allowed.
"""

from __future__ import annotations

from .boundary import Boundary

INFLOWS = ("in", "grow", "link", "gap")
OUTFLOWS = ("out", "drain")


def side(a) -> tuple[int, str] | None:
    """(stock index, side) where the atom meets the outside; None for a move."""
    if a.kind in INFLOWS:
        return a.dst, "in"
    if a.kind in OUTFLOWS:
        return a.src, "out"
    return None


class Allow:
    """allow(atom): whether the boundary b lets the atom be offered. ids: the measured stocks (atom
    indices 0..len-1); inputs: the indices that are measured inputs' rows."""

    def __init__(self, b: Boundary, ids: list[str], inputs: set[int] = frozenset()):
        self.b, self.ids, self.inputs = b, list(ids), set(inputs)

    def __call__(self, a) -> bool:
        where = side(a)
        if where is None or not self.b.declared:
            return True
        s, sd = where
        if s is None or s >= len(self.ids):
            return True
        kind = "measured" if a.driver is not None and a.driver in self.inputs else "estimate"
        return self.b.may(self.ids[s], sd, kind)

    def keep(self, atoms: list) -> list:
        return [a for a in atoms if self(a)]
