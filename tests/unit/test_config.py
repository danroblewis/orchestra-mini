"""The config file (orchestra/config): layers, merging, refusals, durations, defaults."""

from __future__ import annotations

import json

import pytest
import yaml

from orchestra.config import ConfigError, build_config, parse_duration
from orchestra.project import load


def test_defaults_with_no_file(make_project):
    repo = make_project.repo
    p = load(repo)
    assert p.name == "repo" and p.files == []
    c = p.config
    assert (c.bus, c.metrics_store, c.agent.command, c.discovery.every, c.discovery.window) == ("file", "file", "claude", 300, 3600)


def test_entries_take_their_defaults():
    c = build_config({"metrics": [{"name": "m", "command": "x"}], "events": [{"name": "e", "command": "y"}],
                      "processes": [{"name": "w", "prompt": "p.md", "on": "a.b.c"}]})
    assert (c.metrics[0].every, c.metrics[0].timeout) == (60, 30)
    assert (c.events[0].every, c.events[0].continuous) == (60, False)
    w = c.processes[0]
    assert (w.kind, w.on, w.tools, w.timeout, w.per_day_usd, w.min_confidence) == ("agent", ["a.b.c"], None, 1200, 10, 0.6)


def test_overlay_merges_maps_and_named_lists(make_project, monkeypatch):
    make_project({"bus": "file", "agent": {"command": "claude"},
                  "tools": [{"name": "a", "command": "echo a"}, {"name": "b", "command": "echo b"},
                            {"name": "c", "command": "echo c"}]})
    overlay = make_project.home / "t" / "orchestra.json"
    overlay.parent.mkdir(parents=True)
    overlay.write_text(json.dumps({"agent": {"harness": "claude_code"},
                                   "tools": [{"name": "a", "command": "echo A"}, {"name": "b", "enabled": False},
                                             {"name": "d", "command": "echo d"}]}))
    p = load(make_project.repo)
    assert [(t.name, t.command) for t in p.config.tools] == [("a", "echo A"), ("c", "echo c"), ("d", "echo d")]
    assert p.config.agent.command == "claude" and len(p.files) == 2


def test_orchestra_home_moves_the_overlay_and_state(make_project):
    p = make_project()
    assert p.state_dir == make_project.home / "t"


@pytest.mark.parametrize(("raw", "message"), [
    ({"buss": "file"}, r"^buss: unknown key"),
    ({"processes": [{"name": "w", "prompt": "p", "on": "x", "promt": "q"}]}, r"^processes\[w\]\.promt: unknown key"),
    ({"agent": {"comand": "claude"}}, r"^agent\.comand: unknown key"),
    ({"stores": [{"name": "s", "url": "https://x", "match": "{a=\"b\"}", "token": "abc"}]}, r"stores\[s\]\.token: secrets are environment"),
    ({"tools": [{"name": "t", "command": "curl -H 'Authorization: Bearer abc123'"}]}, r"literal secret"),
    ({"bus": "redis://user:hunter2@host:6379"}, r"bus: looks like a literal secret"),
    ({"stores": [{"name": "s", "url": "https://x", "match": "{a=\"b\"}", "token_env": "not a name"}]}, r"not an environment"),
    ({"metrics": [{"name": "m"}]}, r"metrics\[m\]\.command: required"),
    ({"metrics": [{"name": "m", "command": "x", "every": "5 minutes"}]}, r"metrics\[m\]\.every: bad duration"),
    ({"processes": [{"name": "w", "kind": "robot", "on": "x"}]}, r"kind: 'robot' is not one of"),
    ({"processes": [{"name": "w", "prompt": "p.md"}]}, r"say when it runs"),
    ({"processes": [{"name": "w", "prompt": "p.md", "on": "x", "tools": ["nope"]}]}, r"no tool named 'nope'"),
    ({"processes": [{"name": "d", "kind": "decision", "on": "x"}]}, r"question and its answers"),
    ({"tools": [{"name": "t", "command": "a"}, {"name": "t", "command": "b"}]}, r"given twice"),
    ({"stores": [{"name": "s", "url": "https://x", "match": "{}"}]}, r"match-everything"),
    ({"events": [{"name": "e", "command": "x", "continuous": "yes"}]}, r"expected true or false"),
])
def test_refusals_name_the_path(raw, message):
    with pytest.raises(ConfigError, match=message):
        build_config(raw)


def test_yaml_and_a_bad_file(make_project):
    (make_project.repo / "orchestra.yaml").write_text("project: [unclosed\n")
    with pytest.raises(ConfigError, match="orchestra.yaml"):
        load(make_project.repo)


def test_two_config_files_are_refused(make_project):
    (make_project.repo / "orchestra.yaml").write_text(yaml.safe_dump({"project": "a"}))
    (make_project.repo / "orchestra.json").write_text("{}")
    with pytest.raises(ConfigError, match="more than one"):
        load(make_project.repo)


@pytest.mark.parametrize(("text", "seconds"), [("500ms", 0.5), ("30", 30), ("30s", 30), ("5m", 300), ("2h", 7200),
                                               ("1d", 86400), (" 3 d ", 3 * 86400), (90, 90)])
def test_durations(text, seconds):
    assert parse_duration(text) == seconds


def test_secret_keys_are_refused_but_env_names_pass():
    c = build_config({"stores": [{"name": "s", "url": "https://x", "match": '{a="b"}', "token_env": "X_TOKEN",
                                  "auth": {"header": "X-Key", "token_env": "K"}, "kinds": {"*_total": "flow"}}]})
    assert c.stores[0].token_env == "X_TOKEN" and c.stores[0].auth.header == "X-Key"


def test_on_is_a_key_not_a_boolean(make_project):
    (make_project.repo / "orchestra.yaml").write_text(
        "processes:\n  - name: w\n    prompt: w.md\n    on: [a.b.c]\n    continuous: false\n    model: yes\n")
    (w,) = load(make_project.repo).config.processes
    assert w.on == ["a.b.c"] and w.continuous is False and w.model == "yes"
