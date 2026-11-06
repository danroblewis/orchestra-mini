"""The one read function (orchestra/facts.py): repeats skipped, ``seq`` stamped in log order, read
incrementally, and held facts told apart."""

from __future__ import annotations

from orchestra.events import Events
from orchestra.facts import FactLog, held, observed, read


def put(events, type, data=None, subject="A-1", source="poller", **kw):
    events.bus.append({"specversion": "1.0", "id": kw.pop("id", None) or f"{type}:{len(events.read(0)[0])}", "type": type,
                       "source": source, "subject": subject, "time": "2026-11-05T10:00:00.000Z", "data": data, **kw})


def test_repeats_are_skipped_wherever_they_came_from(tmp_path):
    events = Events(tmp_path / "e.jsonl")
    put(events, "jira.issue.observed", {"status": "Open", "x": {"b": 1, "a": 2}})
    put(events, "jira.issue.observed", {"x": {"a": 2, "b": 1}, "status": "Open"})       # canonical JSON: the same
    put(events, "jira.issue.observed", {"status": "Open"}, source="another")           # another source: not a repeat
    put(events, "jira.issue.observed", {"status": "Open"}, subject="A-2")              # another subject: not one
    put(events, "jira.issue.observed", {"status": "Done", "x": {"a": 2, "b": 1}})
    put(events, "jira.issue.observed", {"status": "Open", "x": {"b": 1, "a": 2}})      # back to an earlier state: a change
    put(events, "ci.check.failed", {"n": 1})
    put(events, "ci.check.failed", {"n": 1})                                           # an event, not an observation
    facts = read(events)
    assert [f["seq"] for f in facts] == [1, 3, 4, 5, 6, 7, 8], "seq is the position in the log, repeats left out"
    assert observed({"type": "a.b.observed"}) and not observed({"type": "a.b.observed_by"})


def test_seq_keeps_log_order_and_survives_a_malformed_line(tmp_path):
    events = Events(tmp_path / "e.jsonl")
    put(events, "a.b.made")
    with (tmp_path / "e.jsonl").open("a") as f:
        f.write("{not json\n")
    put(events, "a.b.done")
    assert [(f["type"], f["seq"]) for f in read(events)] == [("a.b.made", 1), ("a.b.done", 3)]


def test_a_fact_log_reads_what_the_bus_gained(tmp_path):
    events = Events(tmp_path / "e.jsonl")
    log = FactLog(events)
    put(events, "a.b.made")
    assert [f["seq"] for f in log.update()] == [1]
    put(events, "a.b.made", subject="A-2")
    put(events, "a.b.done")
    assert [f["seq"] for f in log.update()] == [2, 3] and log.update() == []
    assert [f["seq"] for f in log.after(1)] == [2, 3] and [f["seq"] for f in log.of("A-1")] == [1, 3]


def test_held_is_past_the_depth_limit():
    assert held({"depth": 9}) and not held({"depth": 8}) and not held({}) and not held({"depth": True})
