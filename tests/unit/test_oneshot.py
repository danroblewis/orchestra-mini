"""A once machine's run (orchestra/oneshot.py): program, agent and decision kinds, and the spend cap."""

from __future__ import annotations

import json

from conftest import script
from fakes import CLAUDE

from orchestra import metrics, runs
from orchestra.events import Events
from orchestra.lines import Publisher
from orchestra.oneshot import start_run

TRIGGER = {"specversion": "1.0", "id": "e1", "type": "t.task.created", "source": "tracker", "subject": "T-1",
           "time": "2026-11-05T10:00:00.000Z", "data": {"title": "x"}}
ITEM = {"subject": "T-1", "fields": {"title": "x", "size": 3}, "events": [TRIGGER]}


def setup(make_project, monkeypatch, machine: dict, **scripts):
    script(make_project.repo / "claude", CLAUDE)
    monkeypatch.setenv("FAKE_DIR", str(make_project.repo))
    project = make_project({"agent": {"command": str(make_project.repo / "claude")}, "machines": [machine],
                            "tools": [{"name": "say", "command": "echo {text}", "help": "Say it."}]}, **scripts)
    events = Events.for_project(project)
    return project, events, Publisher(events)


def once(project, pub, event=TRIGGER, item=ITEM):
    return start_run(project, project.config.machines[0], event, item, pub, metrics.for_project(project))


def calls(repo):
    return [json.loads(p.read_text()) for p in sorted(repo.glob("call-*.json"))]


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


def test_an_agent_turn(make_project, monkeypatch):
    (make_project.repo / "w.md").write_text("You are the worker.")
    project, events, pub = setup(make_project, monkeypatch, {"name": "w", "kind": "agent", "prompt": "w.md", "on": "t.task.created",
                                                             "tools": ["say"], "model": "sonnet"})
    rec = once(project, pub)
    assert (rec["exit"], rec["cost_usd"], rec["note"]) == (0, 0.25, None)
    assert sorted(e["type"] for e in events.read(0)[0]) == ["agent.note.left", "agent.work.done"], \
        "the event said twice (text and result) is published once"
    (call,) = calls(project.repo)
    assert call["argv"] == ["-p", "--output-format", "stream-json", "--verbose", "--dangerously-skip-permissions", "--model", "sonnet"]
    assert call["prompt"].startswith("You are the worker.") and '"id": "e1"' in call["prompt"]
    assert "orchestra tool say text=..." in call["prompt"] and '"size": 3' in call["prompt"]
    assert call["machine"] == call["source"] == "w" and json.loads(call["item"])["events"] == 1


def test_the_spend_cap_holds_a_process_until_tomorrow(make_project, monkeypatch):
    (make_project.repo / "w.md").write_text("Work.")
    monkeypatch.setenv("FAKE_COST", "6")
    project, events, pub = setup(make_project, monkeypatch, {"name": "w", "kind": "agent", "prompt": "w.md", "on": "x", "per_day_usd": 10})
    first, second, third = (once(project, pub, item=None) for _ in range(3))
    assert first["cost_usd"] == second["cost_usd"] == 6.0
    assert third["skipped"] and "per_day_usd 10" in third["note"] and "tomorrow" in third["note"]
    assert len(calls(project.repo)) == 2


def test_a_decision(make_project, monkeypatch):
    proc = {"name": "triage", "kind": "decision", "question": "Is this real?", "answers": ["real", "noise"],
            "inputs": ["title"], "on": "t.task.created"}
    project, events, pub = setup(make_project, monkeypatch, proc)
    rec = once(project, pub)
    assert rec["exit"] == 0 and rec["cost_usd"] == 0.25 and rec["types"] == ["decision.made", "triage.real"]
    made, named = events.read(0)[0]
    assert made["data"] == {"machine": "triage", "answer": "real", "confidence": 0.9, "reason": "it looks real"}
    assert made["subject"] == named["subject"] == "T-1" and made["source"] == "triage"
    (call,) = calls(project.repo)
    assert '"title": "x"' in call["prompt"] and '"size"' not in call["prompt"] and "--json-schema" in call["argv"]
    monkeypatch.setenv("FAKE_CONFIDENCE", "0.3")
    once(project, pub)
    assert [e["type"] for e in events.read(0)[0]][2:] == ["decision.made", "triage.unsure"]
