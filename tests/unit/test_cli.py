"""The ``orchestra`` command (orchestra/cli.py), in process: the four commands, the query
subcommands, and the programs for machines kept apart in the help."""

from __future__ import annotations

import json
import time

import pytest

from orchestra import metrics, runs
from orchestra.cli import main, parser
from orchestra.metrics import Point
from orchestra.project import load


def test_the_four_commands_and_the_programs_apart(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])
    out = capsys.readouterr().out
    assert "{run,chat,emit,query}" in out and "programs for machines" in out
    commands, programs = out.split("programs for machines")
    assert "orchestra agent <prompt file>" in programs and "orchestra decide" in programs and "agent" not in commands
    assert set(parser()._subparsers._group_actions[0].choices) == {"run", "chat", "emit", "query"}
    for gone in ("init", "setup", "check", "events", "items", "metrics", "model", "runs", "tool", "endpoint", "reset"):
        with pytest.raises(SystemExit):
            main([gone])


def test_emit_and_query_events_items_runs(make_project, capsys, monkeypatch):
    make_project()
    repo = str(make_project.repo)

    def run(*a):
        return main(["--repo", repo, *a])

    assert run("emit", '{"type": "t.task.created", "subject": "T-1", "data": {"title": "a"}}') == 0
    monkeypatch.setenv("ORCHESTRA_SOURCE", "worker")
    assert run("emit", '{"type": "t.task.closed", "subject": "T-2", "id": "x"}') == 0
    assert run("emit", '{"type": "t.task.closed", "subject": "T-2", "id": "x"}') == 0
    assert "duplicate" in capsys.readouterr().out
    assert run("emit", "{nope") == 2 and run("emit", '{"subject": "x"}') == 2
    capsys.readouterr()
    assert run("query", "events", "--subject", "T-2") == 0
    (line,) = capsys.readouterr().out.splitlines()
    assert json.loads(line)["source"] == "worker", "the source from the environment"
    assert run("emit", '{"type": "t.task.noted", "subject": "T-3", "source": "person.tracker.bo"}') == 0
    capsys.readouterr()
    assert run("query", "events", "--subject", "T-3") == 0 and json.loads(capsys.readouterr().out)["source"] == "person.tracker.bo"
    assert run("query", "events", "-n", "1", "--type", "t.task.created") == 0
    assert json.loads(capsys.readouterr().out)["subject"] == "T-1"
    assert run("query", "items") == 0
    out = capsys.readouterr().out
    assert "T-1" in out and "T-2" not in out
    assert run("query", "items", "--kind", "t.task") == 0 and "T-1" in capsys.readouterr().out
    assert run("query", "items", "--kind", "other") == 0 and capsys.readouterr().out == ""
    assert run("query", "items", "show", "T-2") == 0 and json.loads(capsys.readouterr().out)["done"] is True
    assert run("query", "items", "show", "T-9") == 1
    with pytest.raises(SystemExit):
        run("query", "items", "bogus")
    runs.record(load(make_project.repo), {"id": "r1", "machine": "w", "start": "2026-11-05T10:00:00.000Z",
                                          "exit": 0, "events": 2, "points": 3, "cost_usd": 0.5})
    capsys.readouterr()
    assert run("query", "runs") == 0 and run("query", "runs", "w") == 0
    a, b = capsys.readouterr().out.strip().splitlines()
    assert a == b == "2026-11-05T10:00:00.000Z  w  exit 0  2 events  3 points  $0.5000"
    assert run("query", "model") == 1 and "no model yet" in capsys.readouterr().out


def test_emit_a_metric_line(make_project, capsys, monkeypatch):
    project = make_project()
    repo = str(make_project.repo)
    monkeypatch.setenv("ORCHESTRA_SOURCE", "worker")
    monkeypatch.setenv("ORCHESTRA_MACHINE", "worker")
    assert main(["--repo", repo, "emit", '{"metric": "reviews_done_total", "value": 3, "labels": {"team": "a"}, "kind": "counter"}']) == 0
    from orchestra.output import declared_kinds
    assert metrics.for_project(project).latest("reviews_done_total", {"project": "t", "machine": "worker", "team": "a"})[1] == 3
    assert declared_kinds(project) == {"reviews_done_total": "counter"}
    assert main(["--repo", repo, "emit", '{"metric": "orchestra_x", "value": 1}']) == 2 and "Orchestra's own" in capsys.readouterr().err


def test_query_metrics_and_promql(make_project, capsys):
    make_project()
    repo = str(make_project.repo)
    metrics.for_project(load(make_project.repo)).write([Point("queue_depth", 4, time.time() - 5, {"q": "a"})])
    assert main(["--repo", repo, "query", "metrics", "queue"]) == 0 and 'queue_depth{q="a"} 4' in capsys.readouterr().out
    assert main(["--repo", repo, "query", "sum(queue_depth)"]) == 0 and capsys.readouterr().out.strip().endswith(" 4")
    assert main(["--repo", repo, "query", "x", "--store", "s"]) == 2 and "no store s" in capsys.readouterr().err


def test_run_refuses_a_config_with_errors(make_project, capsys):
    make_project({"machines": [{"name": "m", "command": "./nope.sh", "every": "1m"},
                               {"name": "w", "command": "orchestra agent missing.md", "on": "a"},
                               {"name": "d", "command": "orchestra decide 'Real?'", "on": "a"}]})
    assert main(["--repo", str(make_project.repo), "run", "--seconds", "0"]) == 1
    err = capsys.readouterr().err
    assert "error: machine m: ./nope.sh not found" in err and "orchestra agent's prompt file missing.md not found" in err
    assert "error: machine d: orchestra decide needs --answers" in err and "not started" in err


def test_run_refuses_a_config_that_does_not_load(make_project, capsys):
    (make_project.repo / "orchestra.yaml").write_text("machnes: []\n")
    assert main(["--repo", str(make_project.repo), "run", "--seconds", "0"]) == 1
    assert "machnes: unknown key" in capsys.readouterr().err


def test_run_makes_the_state_directory_and_prints_the_endpoint(make_project, capsys):
    project = make_project({"machines": [{"name": "m", "command": "true", "every": "1h"}]})
    assert not project.state_dir.exists()
    assert main(["--repo", str(make_project.repo), "run", "--seconds", "0.5"]) == 0
    out = capsys.readouterr().out
    assert project.state_dir.is_dir() and "events endpoint http://127.0.0.1:" in out and "Authorization: Bearer " in out
