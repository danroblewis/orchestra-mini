"""Fact lines (orchestra/lines.py): defaults, what is not a fact, dedupe by id, repeats skipped."""

from __future__ import annotations

import json

from orchestra.events import Events
from orchestra.lines import Publisher, parse


def test_a_line_takes_the_defaults():
    ev, why = parse('{"type": "tracker.issue.created", "subject": "ABC-1"}\n', "tracker")
    assert why is None and ev["source"] == "tracker" and ev["subject"] == "ABC-1" and ev["specversion"] == "1.0"
    assert ev["time"].endswith("Z") and len(ev["id"]) == 32
    again, _ = parse('{"type": "tracker.issue.created", "subject": "ABC-1"}', "tracker")
    assert again["id"] == ev["id"], "the default id is a hash of the line"
    salted, _ = parse('{"type": "tracker.issue.created", "subject": "ABC-1"}', "tracker", salt="run-2")
    assert salted["id"] != ev["id"]


def test_given_fields_are_kept():
    line = json.dumps({"type": "a.b.c", "source": "person.tracker.alice", "id": "x:1", "time": "2026-11-05T10:00:00+02:00",
                       "data": {"k": 1}})
    ev, _ = parse(line, "cmd")
    assert (ev["source"], ev["id"], ev["time"], ev["data"]) == ("person.tracker.alice", "x:1", "2026-11-05T08:00:00.000Z", {"k": 1})


def test_what_is_not_a_fact():
    assert parse("", "c") == (None, None)
    assert parse("hello", "c") == (None, "not JSON")
    assert parse("{bad", "c") == (None, "not JSON")
    assert parse('{"subject": "x"}', "c") == (None, "no type")
    assert parse('{"type": "a", "time": "yesterday"}', "c")[0] is None
    assert parse('{"type": "a", "subject": 3}', "c") == (None, "subject is not a string")
    assert parse('{"type": "a", "causedby": 3}', "c") == (None, "causedby is not a string")


def test_a_line_may_name_its_cause():
    ev, _ = parse('{"type": "a.b.c", "causedby": "f1"}', "c")
    assert ev["causedby"] == "f1" and "depth" not in ev


def publish(pub, line, source="c", salt=""):
    ev, _ = parse(line, source, salt)
    return pub.publish(ev) == "published"


def test_the_publisher_drops_an_id_the_bus_has(tmp_path):
    events = Events(tmp_path / "e.jsonl")
    a, b = Publisher(events), Publisher(Events(tmp_path / "e.jsonl"))
    assert publish(a, '{"type": "x.y.z", "id": "1"}') and not publish(a, '{"type": "x.y.z", "id": "1"}')
    assert not publish(b, '{"type": "x.y.z", "id": "1"}'), "another publisher sees what the first wrote"
    assert publish(b, '{"type": "x.y.z", "id": "2"}')
    assert [e["id"] for e in events.read(0)[0]] == ["1", "2"]


def test_actor_lines_are_published(tmp_path):
    events = Events(tmp_path / "e.jsonl")
    publish(Publisher(events), '{"type": "orchestra.actor", "source": "person.tracker.alice", "data": {"name": "Alice A"}}', "tracker")
    (ev,), _ = events.read(0)
    assert ev["source"] == "person.tracker.alice" and ev["data"]["name"] == "Alice A"


def test_the_publisher_skips_a_repeated_observation_and_stamps_recordedtime(tmp_path):
    events = Events(tmp_path / "e.jsonl")
    pub = Publisher(events)
    seen = '{"type": "jira.issue.observed", "subject": "A-1", "data": {"status": "Open", "n": [1, 2]}}'
    assert publish(pub, seen, salt="1") and not publish(pub, seen, salt="2"), "same data in another run: a repeat"
    ev, _ = parse(seen, "c", salt="3")
    assert Publisher(Events(tmp_path / "e.jsonl")).publish(ev) == "repeat", "another publisher can tell from the bus"
    assert publish(pub, seen.replace("Open", "Done"), salt="4") and publish(pub, seen, salt="5"), "a return to an earlier state is a change"
    assert publish(pub, '{"type": "jira.issue.observed", "subject": "A-2", "data": {"status": "Open", "n": [1, 2]}}')
    told = '{"type": "ci.check.failed", "subject": "A-1", "data": {"k": 1}}'
    assert publish(pub, told, salt="6") and publish(pub, told, salt="7"), "only observations repeat"
    assert all(e["recordedtime"].endswith("Z") for e in events.read(0)[0])
