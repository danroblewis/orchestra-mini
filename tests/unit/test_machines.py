"""Machines (orchestra/machines.py): every process and every source heard on the bus."""

from __future__ import annotations

from orchestra import runs
from orchestra.events import Events
from orchestra.machines import machines


def test_processes_and_sources(make_project):
    project = make_project({"processes": [{"name": "worker", "kind": "program", "command": "x", "on": ["a.b.c"]},
                                          {"name": "idle", "kind": "program", "command": "y", "every": "1h"}]})
    events = Events.for_project(project)
    events.publish("orchestra.actor", {"name": "Alice A"}, source="person.tracker.alice")
    events.publish("tracker.issue.moved", source="person.tracker.alice", time="2026-11-05T10:00:00Z")
    events.publish("tracker.issue.moved", source="person.tracker.alice", time="2026-11-05T11:00:00Z")
    events.publish("work.done", source="worker", time="2026-11-05T09:00:00Z")
    runs.record(project, {"id": "r1", "process": "worker", "start": "2026-11-05T09:00:00.000Z"})
    found = {m["name"]: m for m in machines(project)}
    assert list(found) == ["idle", "worker", "person.tracker.alice"]
    w, a = found["worker"], found["person.tracker.alice"]
    assert (w["process"], w["listens_for"], w["publishes"], w["runs"]) == ("program", ["a.b.c"], {"work.done": 1}, 1)
    assert (a["process"], a["label"], a["publishes"], a["last_seen"]) == (None, "Alice A", {"tracker.issue.moved": 2},
                                                                          "2026-11-05T11:00:00.000Z")
    assert found["idle"]["every"] == 3600 and found["idle"]["last_seen"] is None
