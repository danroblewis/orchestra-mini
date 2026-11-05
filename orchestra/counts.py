"""Orchestra's count of the bus: every event counted by its type and source.

- ``orchestra_events_total{project, type, source}``: a cumulative counter per event type and
  source. Events without a source are counted under ``source="unknown"``.

Whether a counter is a flow, and of what, is discovery's business. Events can share an instant;
stores don't agree on the order of points that share one, so only the last value of each series
at each instant is written (``collapse``)."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from .events import event_time
from .metrics import MetricsStore, Point

EVENTS = "orchestra_events_total"


def collapse(points: list[Point]) -> list[Point]:
    """Keep only the last point of each series at each timestamp, in time order."""
    last: dict[tuple, Point] = {}
    for p in points:
        last[(p.key(), p.ts)] = p                 # later points overwrite; dicts keep first-insert order
    return sorted(last.values(), key=lambda p: p.ts)


@dataclass
class EventCollector:
    """Counts events by (type, source). Feed it events in log order with ``observe``; it keeps
    the running totals, so it can follow a live bus after a replay."""

    project: str
    counts: Counter = field(default_factory=Counter)

    def observe(self, ev: dict) -> list[Point]:
        ts = event_time(ev)
        if ts is None or not isinstance(ev.get("type"), str):
            return []
        key = (ev["type"], str(ev.get("source") or "unknown"))
        self.counts[key] += 1
        return [Point(EVENTS, self.counts[key], ts, {"project": self.project, "type": key[0], "source": key[1]})]


@dataclass
class BackfillReport:
    events: int
    points: int
    first: float | None
    last: float | None
    types: dict[str, int]
    counts: Counter = field(default_factory=Counter)    # the running totals, to go on counting from


def backfill(events: list[dict], *, project: str, store: MetricsStore) -> BackfillReport:
    """Replay a whole event log into the metrics store. The points are computed before anything
    is deleted; when the write fails after the old counters were deleted it raises, never returns,
    and the caller replays again (the host retries until it succeeds), so the counters are never
    left deleted by a replay that said it was done."""
    col = EventCollector(project)
    points: list[Point] = []
    for ev in events:
        points += col.observe(ev)
    points = collapse(points)
    store.delete(EVENTS, {"project": project})
    store.write(points)
    stamps = [t for t in (event_time(e) for e in events) if t is not None]
    types = Counter()
    for (ty, _), n in col.counts.items():
        types[ty] += n
    return BackfillReport(len(events), len(points), min(stamps, default=None), max(stamps, default=None), dict(types),
                          col.counts)
