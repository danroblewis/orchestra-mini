"""What discovery reads: every series of the window on a grid of steps (``gather``), from the
project's metrics store, every store's series within its match (``orchestra.stores``), and
Orchestra's count of the bus, ``orchestra_events_total{type, source}``.

A series is sampled at each step as its last value at or before it. A **counter** (a store's flow;
an own series named ``_total`` or ``_count`` that never falls but by a reset) is read by its
increments, a drop being a reset; before its first point it is 0 when it is Orchestra's count of the
bus (which counts from the first event) and its first value otherwise (it was counting already).
A **level** is read by its changes, holding its first value before its first point. Orchestra's
other series (``orchestra_*``) and its own events (``orchestra.*`` types) are no measures of the
system and are left out, with the reason."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field
from itertools import pairwise

from .. import hooks, metrics
from ..stores.kinds import kind_of

EVENTS = "orchestra_events_total"
HIDDEN = ("project", "command")       # labels never part of a series' identity in the model


@dataclass
class Series:
    name: str
    labels: dict[str, str]
    points: list[tuple[float, float]]
    values: list[float]                     # at each grid step
    counter: bool
    source: str = "own"                     # the metrics store, or a store's name
    query: str | None = None
    live: list[int] | None = None           # the steps whose change is measured; None: every step
    extra: dict = field(default_factory=dict)

    @property
    def key(self) -> str:
        inner = ",".join(f'{k}="{v}"' for k, v in sorted(self.labels.items()) if k not in HIDDEN)
        return f"{self.name}{{{inner}}}" if inner else self.name

    @property
    def deltas(self) -> list[float]:
        """Each step's change: a counter's increment (a drop is a reset), a level's difference."""
        cached = self.__dict__.get("_deltas")
        if cached is None or cached[0] is not self.values:
            if self.counter:
                d = [b - a if b >= a else b for a, b in pairwise(self.values)]
            else:
                d = [b - a for a, b in pairwise(self.values)]
            cached = self.__dict__["_deltas"] = (self.values, d)
        return cached[1]


def writer_of(s: Series) -> str | None:
    """For Orchestra's count of the bus, the ``source`` that published the events."""
    if s.name != EVENTS:
        return None
    who = (s.labels.get("source") or "").partition("/")[0]
    return who if who and who != "unknown" else None


def sample(points: list[tuple[float, float]], start: float, step: float, n: int, before: float | None) -> list[float]:
    """The value at each of ``n + 1`` grid steps from ``start``: the last at or before it, ``before``
    (or the first value, when None) before the first point."""
    ts = [t for t, _ in points]
    first = points[0][1] if before is None else before
    out = []
    for i in range(n + 1):
        j = bisect_right(ts, start + i * step) - 1
        out.append(points[j][1] if j >= 0 else first)
    return out


def _counter(name: str, pts: list[tuple[float, float]]) -> bool:
    if kind_of(name, {})[0] != "flow":
        return False
    drops = sum(1 for (_, a), (_, b) in pairwise(pts) if b < a)
    return drops <= max(2, len(pts) // 100)


def gather(project, start: float, end: float, step: float) -> tuple[list[Series], dict[str, str], dict[str, str]]:
    """(the series, what was left out and why by key, each store's error by name)."""
    n = max(2, int((end - start) // step))
    out: list[Series] = []
    left: dict[str, str] = {}
    errors: dict[str, str] = {}
    store = metrics.for_project(project)
    for name, labels, pts in store.dump("", {"project": project.name}):
        pts = [(t, v) for t, v in pts if t <= end]
        if not pts:
            continue
        counter = _counter(name, pts)
        ser = Series(name, dict(labels), pts, [], counter)
        if name.startswith("orchestra_") and name != EVENTS:
            left[ser.key] = "Orchestra's own series, not a measure of the system"
            continue
        if name == EVENTS and (labels.get("type") or "").startswith("orchestra."):
            left[ser.key] = "Orchestra's own events: what Orchestra did, not the system's work"
            continue
        if pts[-1][0] < start and counter:
            left[ser.key] = "not written in the window"
            continue
        ser.values = sample(pts, start, step, n, 0.0 if name == EVENTS else None)
        out.append(ser)
    if project.config.stores and (stores := hooks.hook("stores", "read")) is not None:
        for s in project.config.stores:
            try:
                got, why = stores.read(project, s, start, end, step)
            except Exception as e:                  # a store that cannot be read is said, and the rest stand
                errors[s.name] = f"{type(e).__name__}: {e}"
                continue
            for line in why:
                left[line] = "left out by the store's kinds"
            for g in got:
                if g.points:
                    out.append(Series(g.name, dict(g.labels), g.points, sample(g.points, start, step, n, None), g.counter,
                                      source=s.name, query=g.query))
    return out, left, errors
