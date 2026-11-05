"""Event lines (orchestra/lines.py): defaults, what is not an event, dedupe by id."""

from __future__ import annotations

import json

from orchestra.events import Events
from orchestra.lines import Publisher, parse


def test_a_line_takes_the_defaults():
    ev, why = parse('{"type": "jira.issue.created", "subject": "ABC-1"}\n', "jira")
    assert why is None and ev["source"] == "jira" and ev["subject"] == "ABC-1" and ev["specversion"] == "1.0"
    assert ev["time"].endswith("Z") and len(ev["id"]) == 32
    again, _ = parse('{"type": "jira.issue.created", "subject": "ABC-1"}', "jira")
    assert again["id"] == ev["id"], "the default id is a hash of the line"
    salted, _ = parse('{"type": "jira.issue.created", "subject": "ABC-1"}', "jira", salt="run-2")
    assert salted["id"] != ev["id"]


def test_given_fields_are_kept():
    line = json.dumps({"type": "a.b.c", "source": "person.jira.alice", "id": "x:1", "time": "2026-11-05T10:00:00+02:00",
                       "data": {"k": 1}})
    ev, _ = parse(line, "cmd")
    assert (ev["source"], ev["id"], ev["time"], ev["data"]) == ("person.jira.alice", "x:1", "2026-11-05T08:00:00.000Z", {"k": 1})


def test_what_is_not_an_event():
    assert parse("", "c") == (None, None)
    assert parse("hello", "c") == (None, "not JSON")
    assert parse("{bad", "c") == (None, "not JSON")
    assert parse('{"subject": "x"}', "c") == (None, "no type")
    assert parse('{"type": "a", "time": "yesterday"}', "c")[0] is None
    assert parse('{"type": "a", "subject": 3}', "c") == (None, "subject is not a string")


def test_the_publisher_drops_an_id_the_bus_has(tmp_path):
    events = Events(tmp_path / "e.jsonl")
    a, b = Publisher(events), Publisher(Events(tmp_path / "e.jsonl"))
    kept_out = []
    got = a.lines(['{"type": "x.y.z", "id": "1"}\n', "noise\n", '{"type": "x.y.z", "id": "1"}\n'], "c", log=kept_out.append)
    assert len(got) == 1 and kept_out == ["noise"]
    assert b.lines(['{"type": "x.y.z", "id": "1"}', '{"type": "x.y.z", "id": "2"}'], "c")[0]["id"] == "2", \
        "another publisher sees what the first wrote"
    assert [e["id"] for e in events.read(0)[0]] == ["1", "2"]


def test_actor_lines_are_published(tmp_path):
    events = Events(tmp_path / "e.jsonl")
    Publisher(events).lines(['{"type": "orchestra.actor", "source": "person.jira.alice", "data": {"name": "Alice A"}}'], "jira")
    (ev,), _ = events.read(0)
    assert ev["source"] == "person.jira.alice" and ev["data"]["name"] == "Alice A"
