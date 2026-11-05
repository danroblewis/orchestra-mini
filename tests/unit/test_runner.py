"""The scheduler (orchestra/runner.py): commands on their intervals, events handed to processes in
order, a continuous program fed on stdin, the count of the bus."""

from __future__ import annotations

import threading
import time

from orchestra import metrics, runs
from orchestra.events import Events
from orchestra.runner import Runner, listens


def wait_for(cond, timeout=10.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.05)
    return False


def running(project):
    r = Runner(project, threading.Event(), quiet=True)
    r.start()
    return r


def test_listens_by_glob_and_never_to_itself():
    from orchestra.config.schema import Process
    p = Process(name="w", kind="program", command="x", on=["tracker.*", "pr.commented"])
    assert listens(p, {"type": "tracker.issue.created", "source": "tracker"})
    assert not listens(p, {"type": "tracker.issue.created", "source": "w"})
    assert not listens(p, {"type": "pr.opened", "source": "gh"})


def test_events_queue_in_order_one_run_at_a_time(make_project):
    project = make_project({"processes": [{"name": "slow", "kind": "program", "command": "./s.py", "on": "t.*"}]}, **{"s.py": """
        import json, sys, time
        ev = json.loads(sys.stdin.readline())
        time.sleep(0.3)
        print(json.dumps({"type": "done.it", "data": {"n": ev["data"]["n"]}}))
    """})
    r = running(project)
    try:
        events = Events.for_project(project)
        for n in range(3):
            events.publish("t.thing", {"n": n}, source="me")
        assert wait_for(lambda: len([e for e in events.read(0)[0] if e["type"] == "done.it"]) == 3)
        assert [e["data"]["n"] for e in events.read(0)[0] if e["type"] == "done.it"] == [0, 1, 2]
        recs = sorted(runs.read(project, "slow"), key=lambda x: x["start"])
        assert all(a["end"] <= b["start"] for a, b in zip(recs, recs[1:], strict=False)), "one run at a time"
    finally:
        r.stop.set()


def test_a_continuous_program_gets_later_events_on_stdin(make_project):
    project = make_project({"processes": [{"name": "echo", "kind": "program", "command": "./e.py", "continuous": True,
                                           "on": "ping.sent"}]}, **{"e.py": """
        import json, sys
        for line in sys.stdin:
            ev = json.loads(line)
            print(json.dumps({"type": "pong.sent", "subject": ev["subject"]}), flush=True)
    """})
    r = running(project)
    try:
        events = Events.for_project(project)
        assert wait_for(lambda: "echo" in r.live)
        events.publish("ping.sent", subject="a", source="me")
        events.publish("ping.sent", subject="b", source="me")
        assert wait_for(lambda: [e.get("subject") for e in events.read(0)[0] if e["type"] == "pong.sent"] == ["a", "b"])
    finally:
        r.stop.set()


def test_commands_and_the_count_of_the_bus(make_project):
    project = make_project({"events": [{"name": "feed", "command": "./f.py", "every": "1s"}],
                            "metrics": [{"name": "gauge", "command": "./g.py", "every": "1s"}],
                            "processes": [{"name": "tick", "kind": "program", "command": "./t.py", "every": "1s"}]}, **{
        "f.py": 'print(\'{"type": "feed.item.seen", "id": "same"}\')\nprint("chatter")',
        "g.py": 'print("depth 3")',
        "t.py": 'import os\nprint(\'{"type": "tick.ran", "data": {"event": "\' + os.environ["ORCHESTRA_EVENT"] + \'"}}\')',
    })
    r = running(project)
    store = metrics.for_project(project)
    events = Events.for_project(project)
    try:
        assert wait_for(lambda: len(runs.read(project, "tick")) >= 2, 6)
        assert wait_for(lambda: store.latest("orchestra_events_total", {"type": "tick.ran", "source": "tick"}) is not None)
    finally:
        r.stop.set()
    assert [e["type"] for e in events.read(0)[0]].count("feed.item.seen") == 1, "dropped by id"
    assert store.latest("depth", {"project": "t", "command": "gauge"})[1] == 3.0
    assert "chatter" in (project.state_dir / "logs" / "feed.log").read_text()
    assert next(e for e in events.read(0)[0] if e["type"] == "tick.ran")["data"] == {"event": ""}
    assert r.health["feed"]["ok"] and r.health["gauge"]["points"] == 1
