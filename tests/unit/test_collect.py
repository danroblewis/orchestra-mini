"""A metrics command's round (orchestra/collect.py)."""

from __future__ import annotations

from orchestra import metrics
from orchestra.collect import collect, points_of


def test_points_are_labelled_and_orchestra_series_refused():
    r = points_of("tracker", 'open_issues{team="a",command="x"} 4\norchestra_fake 1\nnot a line\nclosed_total 9 1000\n', 50.0)
    assert [(p.series, p.value, p.ts, p.labels) for p in r.points] == [
        ("closed_total", 9.0, 1.0, {"command": "tracker"}),            # in time order
        ("open_issues", 4.0, 50.0, {"team": "a", "exported_command": "x", "command": "tracker"})]
    assert r.refused == ["orchestra_fake"] and len(r.bad) == 1


def test_a_round_writes_its_points(make_project):
    project = make_project({"metrics": [{"name": "q", "command": "./m.py"}]}, **{"m.py": 'print("queue_depth 7")'})
    store = metrics.for_project(project)
    r = collect(project.config.metrics[0], project, store)
    assert r.error is None
    assert store.latest("queue_depth", {"project": "t", "command": "q"})[1] == 7.0


def test_a_failing_round_writes_nothing(make_project):
    project = make_project({"metrics": [{"name": "q", "command": "./m.py"}]},
                           **{"m.py": 'import sys\nprint("queue_depth 7")\nsys.exit(3)'})
    store = metrics.for_project(project)
    assert collect(project.config.metrics[0], project, store).error == "exit code 3"
    assert store.series("queue") == []
