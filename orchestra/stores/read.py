"""The series discovery reads from a store (SPEC §6): every metric within its ``match``, as its
``kinds`` say (a flow read as a counter, a level as a gauge, ``ignore`` never), one series per value
of its ``split`` labels (summed over the rest), named by its ``names``.

One listing of the match (at most ``MAX_SERIES`` series), then one range query per metric:
``sum by (split) (increase(m[step]))`` for a flow, its increments added up into a running count (so
a counter's resets are the store's to handle), and ``sum by (split) (m)`` for a level."""

from __future__ import annotations

from dataclasses import dataclass, field

from . import guard
from .kinds import kind_of, name_of
from .reader import Reader


@dataclass
class StoreSeries:
    name: str                       # what discovery calls it: the names template, else the metric
    labels: dict[str, str]          # the split labels, and ``store``
    points: list[tuple[float, float]]
    counter: bool                   # a flow, read as a running count
    metric: str = ""
    query: str = ""
    why: str = ""                   # where its kind came from
    extra: dict = field(default_factory=dict)


def _query(metric: str, kind: str, split: list[str], step: float) -> str:
    sel = metric + "{}"
    inner = f"increase({sel}[{max(1, int(step))}s])" if kind == "flow" else sel
    return f"sum by ({', '.join(split)}) ({inner})" if split else f"sum({inner})"


def running(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    total, out = 0.0, []
    for t, v in points:
        total += max(0.0, v)
        out.append((t, total))
    return out


def read(project, store, start: float, end: float, step: float) -> tuple[list[StoreSeries], list[str]]:
    """The store's series over [start, end] at ``step``, and what was left out and why."""
    reader = Reader(project, store)
    listed = reader.series(store.match, start, end, guard.MAX_SERIES)
    metrics = sorted({ls.get("__name__", "") for ls in listed} - {""})
    out, left = [], []
    for metric in metrics:
        kind, why = kind_of(metric, store.kinds)
        if kind == "ignore":
            left.append(f"{store.name}: {metric}: ignored ({why})")
            continue
        q = _query(metric, kind, store.split, step)
        for labels, pts in reader.query_range(q, start, end, step, counted=True):
            labels = {k: v for k, v in labels.items() if k in store.split}
            name = name_of(store.names, metric, labels) or metric
            out.append(StoreSeries(name, {**labels, "store": store.name}, running(pts) if kind == "flow" else pts,
                                   kind == "flow", metric, q, why))
    return out, left
