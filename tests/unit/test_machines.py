"""Machines (orchestra/machines.py): every configured machine and every source heard on the bus."""

from __future__ import annotations

from orchestra import runs
from orchestra.events import Events
from orchestra.machines import machines


def test_configured_machines_and_sources(make_project):
    project = make_project({"machines": [{"name": "worker", "command": "x", "on": ["a.b.c"]},
                                         {"name": "idle", "command": "y", "every": "1h"},
                                         {"name": "tail", "command": "z", "continuous": True}]})
    events = Events.for_project(project)
    events.publish("orchestra.actor", {"name": "Alice A"}, source="person.tracker.alice")
    events.publish("tracker.issue.moved", source="person.tracker.alice", time="2026-11-05T10:00:00Z")
    events.publish("tracker.issue.moved", source="person.tracker.alice", time="2026-11-05T11:00:00Z")
    events.publish("work.done", source="worker", time="2026-11-05T09:00:00Z")
    runs.record(project, {"id": "r1", "machine": "worker", "start": "2026-11-05T09:00:00.000Z"})
    found = {m["name"]: m for m in machines(project)}
    assert list(found) == ["idle", "tail", "worker", "person.tracker.alice"]
    w, a = found["worker"], found["person.tracker.alice"]
    assert (w["command"], w["continuous"], w["listens_for"], w["publishes"], w["runs"]) == ("x", False, ["a.b.c"], {"work.done": 1}, 1)
    assert found["tail"]["continuous"] is True
    assert (a["command"], a["label"], a["publishes"], a["last_seen"]) == (None, "Alice A", {"tracker.issue.moved": 2},
                                                                          "2026-11-05T11:00:00.000Z")
    assert found["idle"]["every"] == 3600 and found["idle"]["last_seen"] is None
