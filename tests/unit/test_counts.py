"""Orchestra's count of the bus: every event counted by type and source, whatever it means."""

from __future__ import annotations

from orchestra.counts import EVENTS, EventCollector, backfill, collapse
from orchestra.events import envelope
from orchestra.metrics.file import FileMetrics


def ev(t: float, type: str, source: str | None = "oven") -> dict:
    e = envelope(type, source or "x")
    e["time"] = f"2026-10-20T10:00:{t:06.3f}+00:00"
    if source is None:                        # an event with no source
        e = {"time": e["time"], "type": type}
    return e


def test_counts_by_type_and_source():
    col = EventCollector("p")
    pts = [p for e in [ev(0, "bread.baked"), ev(1, "bread.baked"), ev(2, "bread.sold", "shop"), ev(3, "old", None)]
           for p in col.observe(e)]
    assert col.counts == {("bread.baked", "oven"): 2, ("bread.sold", "shop"): 1, ("old", "unknown"): 1}
    assert [p.value for p in pts if p.labels["type"] == "bread.baked"] == [1, 2], "a cumulative counter"
    assert all(p.series == EVENTS and p.labels["project"] == "p" for p in pts)


def test_one_value_per_series_per_instant():
    col = EventCollector("p")
    pts = collapse([p for e in [ev(5, "a"), ev(5, "a"), ev(6, "a")] for p in col.observe(e)])
    assert [(p.ts % 60, p.value) for p in pts] == [(5.0, 2), (6.0, 3)]


def test_backfill_replaces_on_rerun(tmp_path):
    store = FileMetrics(tmp_path / "m.jsonl")
    log = [ev(0, "bread.baked"), ev(1, "bread.baked"), ev(2, "bread.sold", "shop")]
    for _ in range(2):
        rep = backfill(log, project="p", store=store)
    assert rep.events == 3 and rep.types == {"bread.baked": 2, "bread.sold": 1}
    assert [v for _, v in store.raw(EVENTS, {"project": "p", "type": "bread.baked", "source": "oven"})] == [1, 2]
