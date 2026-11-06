"""What a machine says on stdout (orchestra/output.py): events, metric lines, Prometheus text, its log."""

from __future__ import annotations

from orchestra import metrics
from orchestra.discovery.series import gather
from orchestra.events import Events
from orchestra.lines import Publisher
from orchestra.output import Output, declared_kinds, metric, prometheus


def test_a_metric_line():
    p, why = metric({"metric": "open_issues", "value": 4, "labels": {"team": "a", "machine": "x"}, "time": 100}, "tracker", 50.0)
    assert why is None and (p.series, p.value, p.ts, p.labels) == (
        "open_issues", 4.0, 100.0, {"team": "a", "exported_machine": "x", "machine": "tracker"})
    p, _ = metric({"metric": "depth", "value": 1.5, "time": "2026-11-05T10:00:00Z"}, "m", 50.0)
    assert p.ts == 1793872800.0 and p.labels == {"machine": "m"}
    assert metric({"metric": "depth", "value": 2}, "m", 50.0)[0].ts == 50.0, "no time: when it was read"
    for bad, says in [({"metric": "orchestra_x", "value": 1}, "Orchestra's own"), ({"metric": "a b", "value": 1}, "not a metric name"),
                      ({"metric": "d", "value": "3"}, "not a number"), ({"metric": "d", "value": True}, "not a number"),
                      ({"metric": "d", "value": 1, "labels": {"bad-name": "x"}}, "labels"),
                      ({"metric": "d", "value": 1, "kind": "histogram"}, "counter or gauge"),
                      ({"metric": "d", "value": 1, "time": "yesterday"}, "bad time")]:
        assert metric(bad, "m", 0.0)[0] is None and says in metric(bad, "m", 0.0)[1], bad


def test_prometheus_text_is_labelled_and_orchestra_series_refused():
    points, skipped = prometheus('open_issues{team="a",machine="x"} 4\norchestra_fake 1\nnot a line\nclosed_total 9 1000\n',
                                 "tracker", 50.0)
    assert [(p.series, p.value, p.ts, p.labels) for p in points] == [
        ("closed_total", 9.0, 1.0, {"machine": "tracker"}),            # in time order
        ("open_issues", 4.0, 50.0, {"team": "a", "exported_machine": "x", "machine": "tracker"})]
    assert len(skipped) == 2 and any("orchestra_fake refused" in s for s in skipped)


def reader(project, **kw):
    logged: list[str] = []
    store = metrics.for_project(project)
    out = Output(project, Publisher(Events.for_project(project)), store, "m", log=logged.append, **kw)
    return out, store, logged


def test_json_lines_events_metrics_and_the_log(make_project):
    project = make_project()
    out, store, logged = reader(project)
    for line in ["thinking\n", '{"type": "t.task.sized", "subject": "T-1"}\n', '{"metric": "queue_depth", "value": 7}\n',
                 "depth 3\n", '{"neither": 1}\n', "\n", '{"metric": "jobs_done", "value": 2, "kind": "counter"}\n']:
        out.line(line)
    out.finish(True)
    assert [e["type"] for e in out.events] == ["t.task.sized"] and out.events[0]["source"] == "m"
    assert out.points == 2 and store.latest("queue_depth", {"project": "t", "machine": "m"})[1] == 7.0
    assert logged == ["thinking", "depth 3", '{"neither": 1}'], "Prometheus only when it is the first line"
    assert declared_kinds(project) == {"jobs_done": "counter"}


def test_prometheus_text_when_the_first_line_is(make_project):
    project = make_project()
    out, store, _ = reader(project)
    for line in ["# TYPE depth gauge\n", "depth 3\n", '{"type": "t.ran"}\n', '{"metric": "load", "value": 1}\n']:
        out.line(line)
    assert out.points == 1 and store.latest("depth") is None, "the text is read when the run ends"
    out.finish(True)
    assert out.points == 2 and store.latest("depth", {"project": "t", "machine": "m"})[1] == 3.0
    assert [e["type"] for e in out.events] == ["t.ran"]
    failed, store, _ = reader(project)
    failed.line("other 1\n")
    failed.finish(False)
    assert store.latest("other") is None, "a failed run's Prometheus text is not written"
    agent, _, logged = reader(project, prometheus=False)
    agent.line("depth 3\n")
    assert logged == ["depth 3"] and agent.points == 0


def test_a_declared_counter_is_read_as_one(make_project):
    project = make_project()
    out, store, _ = reader(project)
    for i, v in enumerate([1, 2, 4, 7]):
        out.line(f'{{"metric": "served", "value": {v}, "time": {1000 + 60 * i}, "kind": "counter"}}\n')
        out.line(f'{{"metric": "waiting", "value": {v}, "time": {1000 + 60 * i}}}\n')
    found = {s.name: s.counter for s in gather(project, 1000, 1180, 60)[0]}
    assert found == {"served": True, "waiting": False}
