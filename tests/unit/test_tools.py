"""Tools (orchestra/tools.py): placeholders as single arguments, help, events published."""

from __future__ import annotations

import pytest

from orchestra.config.schema import Tool
from orchestra.events import Events
from orchestra.lines import Publisher
from orchestra.tools import ToolError, call, describe, fill, help_text, parse_args


def test_placeholders_fill_one_argument_each():
    assert fill("post {channel} --text {text} x={channel}", {"channel": "#a b", "text": "hi; rm -rf /"}) == \
        ["post", "#a b", "--text", "hi; rm -rf /", "x=#a b"]
    with pytest.raises(ToolError, match="missing text="):
        fill("post {channel} {text}", {"channel": "c"})
    with pytest.raises(ToolError, match="unknown argument z"):
        fill("post {channel}", {"channel": "c", "z": "1"})
    assert parse_args(["a=1", "b=x=y", "c="]) == {"a": "1", "b": "x=y", "c": ""}
    with pytest.raises(ToolError):
        parse_args(["nokey"])


def test_help_from_the_config_or_the_command(make_project):
    project = make_project(**{"t.py": 'import sys\nprint("usage: t CHANNEL TEXT" if "--help" in sys.argv else "no")'})
    assert help_text(Tool("a", "./t.py {x}", "Posts."), project.repo) == "Posts."
    assert help_text(Tool("a", "./t.py {x}"), project.repo) == "usage: t CHANNEL TEXT"


def test_a_call_prints_and_publishes(make_project, capsys):
    project = make_project({"tools": [{"name": "say", "command": "./s.py {text}", "help": "Say it."}]}, **{"s.py": """
        import json, sys
        print("said", sys.argv[1])
        print(json.dumps({"type": "chat.message.posted", "subject": "chan", "data": {"text": sys.argv[1]}}))
    """})
    events = Events.for_project(project)
    assert call(project, project.config.tools[0], {"text": "a b"}, Publisher(events), "worker") == 0
    assert "said a b" in capsys.readouterr().out
    (ev,), _ = events.read(0)
    assert ev["source"] == "worker" and ev["data"] == {"text": "a b"}
    assert "orchestra tool say text=..." in describe(project, None) and "Say it." in describe(project, ["say"])
    assert describe(project, []) == "You have no tools."
