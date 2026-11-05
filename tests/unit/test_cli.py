"""The ``orchestra`` command (orchestra/cli.py), in process."""

from __future__ import annotations

import json

from orchestra.cli import main


def test_emit_events_items_runs_tool(make_project, capsys, monkeypatch):
    make_project({"tools": [{"name": "hi", "command": "echo {who}", "help": "Greets."}]})
    repo = str(make_project.repo)
    def run(*a):
        return main(["--repo", repo, *a])

    assert run("init") == 0
    assert run("emit", '{"type": "t.task.created", "subject": "T-1", "data": {"title": "a"}}') == 0
    monkeypatch.setenv("ORCHESTRA_PROCESS", "worker")
    assert run("emit", '{"type": "t.task.closed", "subject": "T-2", "id": "x"}') == 0
    assert run("emit", '{"type": "t.task.closed", "subject": "T-2", "id": "x"}') == 0
    assert "duplicate" in capsys.readouterr().out
    assert run("emit", "{nope") == 2 and run("emit", '{"subject": "x"}') == 2
    capsys.readouterr()
    assert run("events", "--subject", "T-2") == 0
    (line,) = capsys.readouterr().out.splitlines()
    assert json.loads(line)["source"] == "worker"
    assert run("items") == 0
    out = capsys.readouterr().out
    assert "T-1" in out and "T-2" not in out
    assert run("items", "show", "T-2") == 0 and json.loads(capsys.readouterr().out)["done"] is True
    assert run("items", "show", "T-9") == 1 and run("items", "bogus") == 2
    assert run("tool", "hi", "--help") == 0 and capsys.readouterr().out.strip() == "Greets."
    assert run("tool", "hi", "who=you all") == 0 and "you all" in capsys.readouterr().out
    assert run("tool", "hi") == 2 and run("tool", "nope") == 2
    monkeypatch.setenv("ORCHESTRA_PROCESS", "w")
    (make_project.repo / "orchestra.yaml").write_text(
        "project: t\ntools: [{name: hi, command: 'echo {who}'}]\nprocesses: [{name: w, kind: program, command: x, on: a, tools: []}]\n")
    assert run("tool", "hi", "who=x") == 2 and "w may not call hi" in capsys.readouterr().err
    assert run("runs") == 0
    assert run("endpoint") == 1, "no endpoint until orchestra run"
    assert run("model") == 2, "discovery is part 2"


def test_check_and_metrics_and_query(make_project, capsys):
    make_project({"metrics": [{"name": "m", "command": "./nope.sh"}], "tools": [{"name": "t", "command": "ls"}]})
    repo = str(make_project.repo)
    assert main(["--repo", repo, "check"]) == 1
    out = capsys.readouterr().out
    assert "FAIL  command m: ./nope.sh not found" in out and "ok    tool t: ok" in out and "ok    bus file" in out
    import time

    from orchestra import metrics
    from orchestra.metrics import Point
    from orchestra.project import load
    metrics.for_project(load(make_project.repo)).write([Point("queue_depth", 4, time.time() - 5, {"q": "a"})])
    assert main(["--repo", repo, "metrics", "queue"]) == 0 and 'queue_depth{q="a"} 4' in capsys.readouterr().out
    assert main(["--repo", repo, "query", "sum(queue_depth)"]) == 0 and capsys.readouterr().out.strip().endswith(" 4")
    assert main(["--repo", repo, "query", "x", "--store", "s"]) == 2


def test_a_bad_config_says_where(make_project, capsys):
    (make_project.repo / "orchestra.yaml").write_text("procesess: []\n")
    assert main(["--repo", str(make_project.repo), "check"]) == 1
    assert "procesess: unknown key" in capsys.readouterr().out
