"""``orchestra run`` end to end: a temp project whose machines are a scheduled collector of events
(three lines, one twice), a scheduled collector of metrics (Prometheus text and a metric line), a
program on one of those event types, a continuous program, ``orchestra agent`` and ``orchestra
decide`` run by a fake ``claude`` on PATH, and a script the agent runs from its own shell. Run for a few seconds, then the
bus, the metrics, the items' fold, the runs and the machines are what they should be."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import yaml
from conftest import script
from fakes import CLAUDE

from orchestra import metrics, runs
from orchestra.events import Events
from orchestra.items import Items
from orchestra.machines import machines
from orchestra.project import load

FEED = """
import json
created = json.dumps({"type": "tracker.issue.created", "subject": "ABC-1", "id": "tracker:ABC-1:1", "data": {"title": "Fix it"}})
print(created)
print(created)
print(json.dumps({"type": "tracker.issue.transitioned", "subject": "ABC-1", "id": "tracker:ABC-1:2",
                  "source": "person.tracker.alice", "time": "2026-01-01T00:00:00Z", "data": {"to": "In Progress"}}))
"""
SIZER = """
import json, sys
ev = json.loads(sys.stdin.readline())
print(json.dumps({"type": "sizer.issue.sized", "subject": ev["subject"], "data": {"size": 3}}))
"""
NOTIFY = """
import json, subprocess, sys
print("posted", sys.argv[1])
subprocess.run(["orchestra", "emit", json.dumps({"type": "chat.message.posted", "subject": "ABC-1", "data": {"text": sys.argv[1]}})],
               check=True)
"""
WATCHER = """
import json, sys
for line in sys.stdin:
    ev = json.loads(line)
    print(json.dumps({"type": "watcher.decision.seen", "subject": ev["subject"]}), flush=True)
    print(json.dumps({"metric": "decisions_seen_total", "value": 1, "kind": "counter"}), flush=True)
"""
CONFIG = {
    "project": "e2e",
    "machines": [
        {"name": "tracker", "command": "scripts/feed.py", "every": "1s"},
        {"name": "queues", "command": "scripts/queues.py", "every": "1s"},
        {"name": "sizer", "command": "scripts/sizer.py", "on": "tracker.issue.created"},
        {"name": "worker", "command": "orchestra agent prompts/worker.md --per-day-usd 5", "on": "sizer.issue.sized"},
        {"name": "triage", "command": 'orchestra decide "Is the work real?" --answers real,noise', "on": "agent.work.done"},
        {"name": "watcher", "command": "scripts/watcher.py", "continuous": True, "on": "decision.made"},
    ],
}


def test_orchestra_run(tmp_path):
    repo, home, fakebin = tmp_path / "repo", tmp_path / "home", tmp_path / "bin"
    (repo / "prompts").mkdir(parents=True)
    (repo / "prompts" / "worker.md").write_text("You do the work.")
    (repo / "orchestra.yaml").write_text(yaml.safe_dump(CONFIG))
    script(repo / "scripts" / "feed.py", FEED)
    script(repo / "scripts" / "queues.py", 'print("queue_depth{queue=\\"triage\\"} 5")\nprint(\'{"metric": "load", "value": 2}\')')
    script(repo / "scripts" / "watcher.py", WATCHER)
    script(repo / "scripts" / "sizer.py", SIZER)
    script(repo / "scripts" / "notify.py", NOTIFY)
    script(fakebin / "claude", CLAUDE)
    env = {**os.environ, "ORCHESTRA_HOME": str(home), "FAKE_DIR": str(tmp_path), "FAKE_SHELL": "scripts/notify.py 'hello all'",
           "PATH": os.pathsep.join([str(fakebin), str(Path(sys.executable).parent), os.environ["PATH"]])}
    for k in ("ORCHESTRA_REPO", "ORCHESTRA_SOURCE", "ORCHESTRA_MACHINE", "ORCHESTRA_BUS_URL", "ORCHESTRA_METRICS_URL"):
        env.pop(k, None)
    out = subprocess.run([str(Path(sys.executable).parent / "orchestra"), "run", "--seconds", "6"], cwd=repo, env=env,
                         capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr

    os.environ["ORCHESTRA_HOME"] = str(home)
    try:
        project = load(repo)
        evs = list(Events.for_project(project).all(0))
        types = [e["type"] for e in evs]
        by = {e["type"]: e for e in evs}

        # the bus: the duplicated line once, every round's repeats dropped, and the chain
        assert types.count("tracker.issue.created") == 1 and types.count("tracker.issue.transitioned") == 1
        for t in ("sizer.issue.sized", "agent.work.done", "agent.note.left", "chat.message.posted", "decision.made", "triage.real",
                  "watcher.decision.seen"):
            assert types.count(t) == 1, (t, types)
        assert by["tracker.issue.created"]["source"] == "tracker" and by["tracker.issue.transitioned"]["source"] == "person.tracker.alice"
        assert by["sizer.issue.sized"]["source"] == "sizer" and by["agent.work.done"]["source"] == "worker"
        assert by["chat.message.posted"]["source"] == "worker", "a script run from the agent's shell emits as the agent"
        assert by["decision.made"]["data"] == {"machine": "triage", "answer": "real", "confidence": 0.9, "reason": "it looks real"}
        assert all(e["specversion"] == "1.0" and e["id"] and e["time"].endswith("Z") for e in evs)

        # the metrics
        store = metrics.for_project(project)
        assert store.latest("queue_depth", {"project": "e2e", "queue": "triage", "machine": "queues"})[1] == 5.0
        assert store.latest("load", {"project": "e2e", "machine": "queues"})[1] == 2.0
        assert store.latest("decisions_seen_total", {"project": "e2e", "machine": "watcher"})[1] == 1.0
        assert store.latest("orchestra_events_total", {"project": "e2e", "type": "triage.real", "source": "triage"})[1] == 1

        # the items' fold
        item = Items(project, Events.for_project(project)).get("ABC-1")
        assert {k: item["fields"][k] for k in ("title", "size", "ok", "text", "answer")} == \
            {"title": "Fix it", "size": 3, "ok": True, "text": "hello all", "answer": "real"}
        assert item["done"] is False and len(item["events"]) == 9
        assert item["first"] == "2026-01-01T00:00:00.000Z" and item["fields"]["to"] == "In Progress"

        # the runs
        done = {p: runs.read(project, p) for p in ("sizer", "worker", "triage")}
        assert [(r["exit"], r["events"]) for r in done["sizer"]] == [(0, 1)]
        assert [(r["exit"], r["cost_usd"]) for r in done["worker"]] == [(0, 0.25)]
        assert done["worker"][0]["types"] == ["agent.note.left", "agent.work.done"]
        assert [(r["exit"], r["types"]) for r in done["triage"]] == [(0, ["decision.made", "triage.real"])]
        prompts = [json.loads(p.read_text()) for p in sorted(tmp_path.glob("call-*.json"))]
        assert len(prompts) == 2 and prompts[0]["prompt"].startswith("You do the work.") and "--json-schema" in prompts[1]["argv"]

        # the machines
        found = {m["name"]: m for m in machines(project)}
        assert {"sizer", "worker", "triage", "tracker", "watcher", "person.tracker.alice"} <= set(found)
        assert (found["watcher"]["command"], found["watcher"]["continuous"]) == ("scripts/watcher.py", True)
        assert found["sizer"]["listens_for"] == ["tracker.issue.created"] and found["sizer"]["runs"] == 1
        assert found["worker"]["publishes"] == {"agent.work.done": 1, "agent.note.left": 1, "chat.message.posted": 1}
        assert found["triage"]["publishes"] == {"decision.made": 1, "triage.real": 1}
        assert found["person.tracker.alice"]["command"] is None and found["person.tracker.alice"]["last_seen"]
        assert "events endpoint http://127.0.0.1:" in out.stdout and not (home / "e2e" / "endpoint.json").exists()
    finally:
        os.environ.pop("ORCHESTRA_HOME", None)
