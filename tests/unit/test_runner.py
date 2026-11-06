"""The scheduler (orchestra/runner.py): once machines on their events and ticks, in order; continuous
machines fed on an open stdin, their run lines making turns; the count of the bus."""

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
    from orchestra.config.schema import Machine
    m = Machine(name="w", command="x", on=["tracker.*", "pr.commented"])
    assert listens(m, {"type": "tracker.issue.created", "source": "tracker"})
    assert not listens(m, {"type": "tracker.issue.created", "source": "w"})
    assert not listens(m, {"type": "pr.opened", "source": "gh"})


def test_events_queue_in_order_one_run_at_a_time(make_project):
    project = make_project({"machines": [{"name": "slow", "command": "./s.py", "on": "t.*"}]}, **{"s.py": """
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


def test_a_continuous_program_gets_three_events_over_time_on_one_open_stdin(make_project):
    project = make_project({"machines": [{"name": "echo", "command": "./e.py", "continuous": True, "on": "ping.sent"}]},
                           **{"e.py": """
        import json, os, sys
        for n, line in enumerate(sys.stdin, 1):
            ev = json.loads(line)
            print(json.dumps({"type": "pong.sent", "subject": ev["subject"], "data": {"pid": os.getpid()}}), flush=True)
            print(json.dumps({"metric": "pongs_total", "value": n}), flush=True)
    """})
    r = running(project)
    try:
        events = Events.for_project(project)
        assert wait_for(lambda: r.live["echo"].proc is not None)
        for subject in "abc":
            events.publish("ping.sent", subject=subject, source="me")
            assert wait_for(lambda s=subject: s in [e.get("subject") for e in events.read(0)[0] if e["type"] == "pong.sent"])
            time.sleep(0.3)
        pongs = [e for e in events.read(0)[0] if e["type"] == "pong.sent"]
        assert [e["subject"] for e in pongs] == ["a", "b", "c"] and len({e["data"]["pid"] for e in pongs}) == 1, "one process"
        assert metrics.for_project(project).latest("pongs_total", {"project": "t", "machine": "echo"})[1] == 3
    finally:
        r.stop.set()


def test_collectors_ticks_and_the_count_of_the_bus(make_project):
    project = make_project({"machines": [{"name": "feed", "command": "./f.py", "every": "1s"},
                                         {"name": "gauge", "command": "./g.py", "every": "1s"},
                                         {"name": "tick", "command": "./t.py", "every": "1s"}]}, **{
        "f.py": 'print(\'{"type": "feed.item.seen", "id": "same"}\')\nprint("chatter")',
        "g.py": 'print("depth 3")\nprint(\'{"metric": "load", "value": 0.5}\')',
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
    assert store.latest("depth", {"project": "t", "machine": "gauge"})[1] == 3.0
    assert store.latest("load", {"project": "t", "machine": "gauge"})[1] == 0.5
    assert "chatter" in (project.state_dir / "logs" / "feed.log").read_text()
    assert next(e for e in events.read(0)[0] if e["type"] == "tick.ran")["data"] == {"event": ""}
    assert runs.read(project, "gauge")[0]["points"] == 2
