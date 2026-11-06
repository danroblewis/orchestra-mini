"""The config file (orchestra/config): layers, merging, refusals, durations, defaults; a machine's
four fields and a store's adapter, with the old fields refused by what says them now."""

from __future__ import annotations

import json

import pytest
import yaml

from orchestra.config import ConfigError, build_config, parse_duration
from orchestra.project import load


def test_defaults_with_no_file(make_project):
    p = load(make_project.repo)
    assert p.name == "repo" and p.files == []
    c = p.config
    assert (c.bus, c.metrics_store, c.discovery.every, c.discovery.window, c.machines, c.stores) == ("file", "file", 300, 3600, [], [])


def test_a_machine_is_four_fields():
    c = build_config({"machines": [{"name": "m", "command": "x", "every": "1m"},
                                   {"name": "w", "command": "orchestra agent p.md", "on": "a.b.c", "env": {"LANG": "C"}},
                                   {"name": "tail", "command": "y", "continuous": True}]})
    m, w, tail = c.machines
    assert (m.every, m.on, m.continuous, m.env) == (60, [], False, {})
    assert (w.on, w.command, w.env) == (["a.b.c"], "orchestra agent p.md", {"LANG": "C"})
    assert (tail.continuous, tail.on, tail.every) == (True, [], None)
    assert c.machine("w") is w


def test_a_store_is_its_adapter_and_its_bounds():
    c = build_config({"stores": [{"name": "s", "command": "orchestra-prometheus https://x/data/metrics --token-env T",
                                  "match": '{a="b"}', "kinds": {"*_total": "flow"}, "names": {"x_*": "{q} x"},
                                  "split": "q", "rate": "2/s", "per_day": 10, "env": {"HTTPS_PROXY": "http://proxy:3128"}}]})
    s = c.store("s")
    assert (s.split, s.rate, s.per_day, s.env["HTTPS_PROXY"]) == (["q"], "2/s", 10, "http://proxy:3128")


def test_overlay_merges_maps_and_named_lists(make_project):
    make_project({"bus": "file", "machines": [{"name": "m", "command": "x", "every": "1m"}, {"name": "n", "command": "y", "every": "1m"},
                                              {"name": "o", "command": "z", "every": "1m"}]})
    overlay = make_project.home / "t" / "orchestra.json"
    overlay.parent.mkdir(parents=True)
    overlay.write_text(json.dumps({"discovery": {"every": "1m"},
                                   "machines": [{"name": "m", "enabled": False}, {"name": "n", "command": "Y", "on": "a"}]}))
    p = load(make_project.repo)
    assert [(m.name, m.command) for m in p.config.machines] == [("n", "Y"), ("o", "z")]
    assert p.config.discovery.every == 60 and p.config.discovery.window == 3600 and len(p.files) == 2


def test_orchestra_home_moves_the_overlay_and_state(make_project):
    assert make_project().state_dir == make_project.home / "t"


@pytest.mark.parametrize(("raw", "message"), [
    ({"buss": "file"}, r"^buss: unknown key"),
    ({"machines": [{"name": "w", "command": "x", "on": "x", "comand": "q"}]}, r"^machines\[w\]\.comand: unknown key"),
    ({"processes": []}, r"^processes: is now machines:"),
    ({"metrics": [{"name": "m", "command": "x"}]}, r"^metrics: is now machines:"),
    ({"tools": [{"name": "t", "command": "x"}]}, r"^tools: gone: an agent runs commands in its own shell"),
    ({"agent": {"command": "claude"}}, r"^agent: gone: an agent is a program now: command: orchestra agent <prompt file>"),
    ({"machines": [{"name": "w", "kind": "agent", "prompt": "p.md", "on": "x"}]},
     r"^machines\[w\]\.kind: an agent is a program now: command: orchestra agent .*orchestra decide"),
    ({"machines": [{"name": "w", "command": "x", "prompt": "p.md", "on": "x"}]}, r"machines\[w\]\.prompt: an agent is a program now"),
    ({"machines": [{"name": "d", "command": "x", "question": "q?", "on": "x"}]}, r"machines\[d\]\.question: a decision is a program now"),
    ({"machines": [{"name": "w", "command": "x", "mode": "continuous"}]}, r"machines\[w\]\.mode: gone: continuous: true keeps it"),
    ({"machines": [{"name": "w", "command": "x", "on": "a", "tools": ["t"]}]}, r"machines\[w\]\.tools: gone"),
    ({"machines": [{"name": "w", "command": "x", "on": "a", "timeout": "5m"}]}, r"machines\[w\]\.timeout: gone"),
    ({"machines": [{"name": "w", "command": "x", "on": "a", "per_day_usd": 3}]}, r"per_day_usd: an agent is a program now: .*--per-day-usd"),
    ({"stores": [{"name": "s", "url": "https://x", "match": '{a="b"}'}]}, r"stores\[s\]\.url: gone: the adapter's now: command: orchestra-prometheus"),
    ({"stores": [{"name": "s", "command": "a", "match": '{a="b"}', "token_env": "X"}]}, r"stores\[s\]\.token_env: gone"),
    ({"stores": [{"name": "s", "command": "a"}]}, r"stores\[s\]\.match: required"),
    ({"stores": [{"name": "s", "command": "a", "match": "{}"}]}, r"match-everything"),
    ({"stores": [{"name": "s", "command": "a", "match": '{__name__=~".+"}'}]}, r"match-everything"),
    ({"stores": [{"name": "s", "command": "a", "match": '{a="b"}', "kinds": {"x": "stock"}}]}, r"kinds\.x: 'stock' is not one of flow"),
    ({"stores": [{"name": "s", "command": "a", "match": '{a="b"}', "token": "abc"}]}, r"stores\[s\]\.token: secrets are environment"),
    ({"machines": [{"name": "w", "command": "curl -H 'Authorization: Bearer abc123'", "on": "a"}]}, r"literal secret"),
    ({"machines": [{"name": "w", "command": "x", "on": "a", "env": {"GH_TOKEN": "ghp_abcdef"}}]}, r"env\.GH_TOKEN: looks like a literal secret"),
    ({"machines": [{"name": "w", "command": "x", "on": "a", "env": {"not a name": "v"}}]}, r"env\.not a name: not an environment"),
    ({"bus": "redis://user:hunter2@host:6379"}, r"bus: looks like a literal secret"),
    ({"machines": [{"name": "m", "every": "1m"}]}, r"machines\[m\]\.command: required"),
    ({"machines": [{"command": "x"}]}, r"machines\[0\]\.name: required"),
    ({"machines": [{"name": "m", "command": "x", "every": "5 minutes"}]}, r"machines\[m\]\.every: bad duration"),
    ({"machines": [{"name": "w", "command": "x", "continuous": "yes"}]}, r"continuous: expected true or false"),
    ({"machines": [{"name": "w", "command": "x"}]}, r"say when it runs: on, every, or continuous: true"),
    ({"machines": [{"name": "w", "command": "x", "continuous": True, "every": "1m"}]}, r"never scheduled"),
    ({"machines": [{"name": "t", "command": "a", "on": "x"}, {"name": "t", "command": "b", "on": "x"}]}, r"given twice"),
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


def test_on_is_a_key_not_a_boolean(make_project):
    (make_project.repo / "orchestra.yaml").write_text(
        "machines:\n  - name: w\n    command: orchestra agent w.md --model yes\n    on: [a.b.c]\n    continuous: true\n")
    (w,) = load(make_project.repo).config.machines
    assert w.on == ["a.b.c"] and w.continuous is True and w.command.endswith("--model yes")
