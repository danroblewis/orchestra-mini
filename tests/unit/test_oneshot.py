"""A once machine's run (orchestra/oneshot.py): the event on stdin, both line shapes, Prometheus
text, the machine's env and a run line's cost."""

from __future__ import annotations

from orchestra import metrics, runs
from orchestra.events import Events
from orchestra.lines import Publisher
from orchestra.oneshot import start_run

TRIGGER = {"specversion": "1.0", "id": "e1", "type": "t.task.created", "source": "tracker", "subject": "T-1",
           "time": "2026-11-05T10:00:00.000Z", "data": {"title": "x"}}
ITEM = {"subject": "T-1", "fields": {"title": "x", "size": 3}, "events": [TRIGGER]}


def setup(make_project, monkeypatch, machine: dict, **scripts):
    project = make_project({"machines": [machine]}, **scripts)
    events = Events.for_project(project)
    return project, events, Publisher(events)


def once(project, pub, event=TRIGGER, item=ITEM):
    return start_run(project, project.config.machines[0], event, item, pub, metrics.for_project(project))


def test_a_program_gets_the_event_and_prints_both_line_shapes(make_project, monkeypatch):
    project, events, pub = setup(make_project, monkeypatch, {"name": "p", "command": "./p.py", "on": "t.task.created"},
                                 **{"p.py": """
        import json, os, sys
        ev = json.loads(sys.stdin.readline())
        item = json.loads(os.environ["ORCHESTRA_ITEM"])
        assert sys.stdin.read() == "", "one event, then stdin closed"
        print("thinking")
        print(json.dumps({"type": "t.task.sized", "subject": ev["subject"],
                          "data": {"size": item["fields"]["size"], "same": json.loads(os.environ["ORCHESTRA_EVENT"]) == ev,
                                   "project": os.environ["ORCHESTRA_PROJECT"], "as": os.environ["ORCHESTRA_SOURCE"]}}))
        print(json.dumps({"metric": "task_size", "value": item["fields"]["size"], "labels": {"task": ev["subject"]}}))
    """})
    rec = once(project, pub)
    assert (rec["exit"], rec["events"], rec["types"], rec["event"], rec["points"]) == (0, 1, ["t.task.sized"], "e1", 1)
    (ev,), _ = events.read(0)
    assert ev["source"] == "p" and ev["data"] == {"size": 3, "same": True, "project": "t", "as": "p"}
    assert metrics.for_project(project).latest("task_size", {"project": "t", "machine": "p", "task": "T-1"})[1] == 3.0
    assert "thinking" in open(rec["log"]).read()
    assert runs.read(project, "p")[0]["id"] == rec["id"]
    again = once(project, pub)
    assert again["events"] == 1, "each run's lines are new"


def test_a_scheduled_collector_with_metric_lines(make_project, monkeypatch):
    project, events, pub = setup(make_project, monkeypatch, {"name": "queues", "command": "./q.py", "every": "1m"}, **{"q.py": """
        import json, sys
        assert sys.stdin.read() == "", "nothing on stdin"
        for q, n in (("triage", 5), ("review", 2)):
            print(json.dumps({"metric": "queue_depth", "value": n, "labels": {"queue": q}, "kind": "gauge"}))
    """})
    rec = once(project, pub, None, None)
    assert (rec["exit"], rec["events"], rec["points"], rec["event"]) == (0, 0, 2, None)
    store = metrics.for_project(project)
    assert store.latest("queue_depth", {"project": "t", "machine": "queues", "queue": "review"})[1] == 2.0


def test_prometheus_text_still_works(make_project, monkeypatch):
    project, _, pub = setup(make_project, monkeypatch, {"name": "prom", "command": "./m.py", "every": "1m"},
                            **{"m.py": 'print("# TYPE queue_depth gauge")\nprint(\'queue_depth{queue="a"} 7\')'})
    assert once(project, pub, None, None)["points"] == 1
    assert metrics.for_project(project).latest("queue_depth", {"project": "t", "machine": "prom", "queue": "a"})[1] == 7.0


def test_a_failing_prometheus_round_writes_nothing(make_project, monkeypatch):
    project, _, pub = setup(make_project, monkeypatch, {"name": "q", "command": "./m.py", "every": "1m"},
                            **{"m.py": 'import sys\nprint("queue_depth 7")\nsys.exit(3)'})
    rec = once(project, pub, None, None)
    assert (rec["exit"], rec["points"]) == (3, 0) and metrics.for_project(project).series("queue") == []


def test_the_machines_env_and_a_run_line(make_project, monkeypatch):
    project, _, pub = setup(make_project, monkeypatch, {"name": "m", "command": "./m.py", "every": "1m", "env": {"GREETING": "hi"}},
                            **{"m.py": """
        import json, os
        print(json.dumps({"type": "m.said", "data": {"greeting": os.environ["GREETING"], "continuous": os.environ["ORCHESTRA_CONTINUOUS"]}}))
        print(json.dumps({"cost_usd": 0.1}))
        print(json.dumps({"cost_usd": 0.15, "note": "a second call"}))
    """})
    rec = once(project, pub, None, None)
    assert (rec["cost_usd"], rec["note"], rec["events"]) == (0.25, "a second call", 1)
    (ev,), _ = Events.for_project(project).read(0)
    assert ev["data"] == {"greeting": "hi", "continuous": ""}
    assert runs.spent_today(project, "m") == 0.25
