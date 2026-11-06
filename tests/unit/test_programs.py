"""The built-in programs (orchestra/agent.py, orchestra/decide.py) as a machine's command, once and
continuous, against the fake ``claude`` (tests/fakes.py): their lines, cost per turn, the per-day
cap and the run records."""

from __future__ import annotations

import json
import threading
import time

from orchestra import metrics, runs
from orchestra.events import Events
from orchestra.lines import Publisher
from orchestra.oneshot import start_run
from orchestra.runner import Runner

TRIGGER = {"specversion": "1.0", "id": "e1", "type": "t.task.created", "source": "tracker", "subject": "T-1",
           "time": "2026-11-05T10:00:00.000Z", "data": {"title": "x"}}
ITEM = {"subject": "T-1", "fields": {"title": "x", "size": 3}, "events": [TRIGGER]}


def wait_for(cond, timeout=15.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.05)
    return False


def project_with(make_project, machine: dict):
    (make_project.repo / "w.md").write_text("You are the worker.")
    return make_project({"machines": [machine]})


def once(project, event=TRIGGER, item=ITEM):
    events = Events.for_project(project)
    return start_run(project, project.config.machines[0], event, item, Publisher(events), metrics.for_project(project))


def calls(seen):
    return [json.loads(p.read_text()) for p in sorted(seen.glob("call-*.json"))]


def turns(project, machine):
    return sorted((r for r in runs.read(project, machine, 100) if "event" in r), key=lambda r: r["start"])


def test_an_agent_turn(make_project, fake_claude):
    project = project_with(make_project, {"name": "w", "command": "orchestra agent w.md --model sonnet", "on": "t.task.created"})
    rec = once(project)
    assert (rec["exit"], rec["cost_usd"], rec["note"]) == (0, 0.25, None)
    assert sorted(e["type"] for e in Events.for_project(project).read(0)[0]) == ["agent.note.left", "agent.work.done"], \
        "the event said twice (text and result) is published once"
    (call,) = calls(fake_claude)
    assert call["argv"] == ["-p", "--output-format", "stream-json", "--verbose", "--dangerously-skip-permissions", "--model", "sonnet"]
    assert call["prompt"].startswith("You are the worker.") and '"id": "e1"' in call["prompt"] and "Saying what happened" in call["prompt"]
    assert '"size": 3' in call["prompt"] and call["machine"] == call["source"] == "w"
    assert "Working on it." in open(rec["log"]).read(), "what it says that is no line goes to its log"


def test_a_scheduled_agent_turn_has_no_event(make_project, fake_claude):
    project = project_with(make_project, {"name": "w", "command": "orchestra agent w.md", "every": "1h"})
    assert once(project, None, None)["exit"] == 0
    (call,) = calls(fake_claude)
    assert "(none: a scheduled run)" in call["prompt"] and "--model" not in call["argv"]


def test_the_per_day_cap_holds_turns_until_tomorrow(make_project, fake_claude, monkeypatch):
    monkeypatch.setenv("FAKE_COST", "6")
    project = project_with(make_project, {"name": "w", "command": "orchestra agent w.md --per-day-usd 10", "on": "x"})
    first, second, third = (once(project, item=None) for _ in range(3))
    assert first["cost_usd"] == second["cost_usd"] == 6.0
    assert third["skipped"] and "per_day_usd 10" in third["note"] and "tomorrow" in third["note"] and not third["cost_usd"]
    assert len(calls(fake_claude)) == 2


def test_an_agent_with_no_prompt_file_says_so(make_project, fake_claude):
    project = project_with(make_project, {"name": "w", "command": "orchestra agent nope.md", "on": "x"})
    rec = once(project)
    assert rec["exit"] == 2 and "no prompt file nope.md" in open(rec["log"]).read()


def test_a_decision(make_project, fake_claude, monkeypatch):
    project = project_with(make_project, {"name": "triage", "on": "t.task.created",
                                          "command": 'orchestra decide "Is this real?" --answers real,noise --inputs title'})
    rec = once(project)
    assert rec["exit"] == 0 and rec["cost_usd"] == 0.25 and rec["types"] == ["decision.made", "triage.real"]
    events = Events.for_project(project)
    made, named = events.read(0)[0]
    assert made["data"] == {"machine": "triage", "answer": "real", "confidence": 0.9, "reason": "it looks real"}
    assert made["subject"] == named["subject"] == "T-1" and made["source"] == "triage"
    (call,) = calls(fake_claude)
    assert '"title": "x"' in call["prompt"] and '"size"' not in call["prompt"] and "--json-schema" in call["argv"]
    monkeypatch.setenv("FAKE_CONFIDENCE", "0.3")
    once(project)
    assert [e["type"] for e in events.read(0)[0]][2:] == ["decision.made", "triage.unsure"]


def running(project):
    r = Runner(project, threading.Event(), quiet=True)
    r.start()
    return r


def test_a_continuous_agent_keeps_one_session(make_project, fake_claude, monkeypatch):
    (make_project.repo / "a.md").write_text("You answer pings.")
    monkeypatch.setenv("FAKE_COST", "0.5")
    project = make_project({"machines": [{"name": "pinger", "command": "orchestra agent a.md --per-day-usd 1", "continuous": True,
                                          "on": "ping.sent"}]})
    r = running(project)
    events = Events.for_project(project)
    try:
        events.publish("ping.sent", {"title": "first"}, subject="P-1", source="me")    # before it is up: it waits
        assert wait_for(lambda: len(turns(project, "pinger")) == 1)
        events.publish("ping.sent", subject="P-2", source="me")
        assert wait_for(lambda: len(turns(project, "pinger")) == 2)
        events.publish("ping.sent", subject="P-3", source="me")
        assert wait_for(lambda: len(turns(project, "pinger")) == 3)
    finally:
        r.stop.set()
    answered = [e for e in events.read(0)[0] if e["type"] == "agent.ping.answered"]
    assert [(e["subject"], e["source"]) for e in answered] == [("P-1", "pinger"), ("P-2", "pinger")]
    done = turns(project, "pinger")
    assert [(t["subject"], t.get("cost_usd"), t.get("events"), t.get("points")) for t in done[:2]] == \
        [("P-1", 0.5, 1, 1), ("P-2", 0.5, 1, 1)], "each turn's own cost, from the session's running total"
    assert done[2]["subject"] == "P-3" and done[2]["skipped"] and "per_day_usd" in done[2]["note"], "past its cap: no turn today"
    started = json.loads((fake_claude / "continuous.json").read_text())
    argv = started["argv"]
    assert argv[argv.index("--input-format") + 1] == "stream-json" and "--verbose" in argv and started["machine"] == "pinger"
    assert "You answer pings." in argv[argv.index("--append-system-prompt") + 1]
    first, second = [json.loads(line) for line in (fake_claude / "messages.jsonl").read_text().splitlines()]
    assert first["type"] == "user" and first["message"]["role"] == "user"
    assert '"subject": "P-1"' in first["message"]["content"] and "The item's state" in first["message"]["content"]
    assert metrics.for_project(project).latest("pings_answered_total", {"project": "t", "machine": "pinger"})[1] == 2


def test_a_continuous_decision_answers_each_event(make_project, fake_claude):
    project = make_project({"machines": [{"name": "triage", "command": "orchestra decide 'Real?' --answers real,noise",
                                          "continuous": True, "on": "ping.sent"}]})
    r = running(project)
    events = Events.for_project(project)
    try:
        for s in ("A", "B"):
            events.publish("ping.sent", subject=s, source="me")
        assert wait_for(lambda: len(turns(project, "triage")) == 2)
    finally:
        r.stop.set()
    made = [e for e in events.read(0)[0] if e["type"] == "triage.real"]
    assert sorted(e["subject"] for e in made) == ["A", "B"]
    assert [(t["subject"], t["cost_usd"], t["types"]) for t in turns(project, "triage")] == \
        [("A", 0.25, ["decision.made", "triage.real"]), ("B", 0.25, ["decision.made", "triage.real"])]
