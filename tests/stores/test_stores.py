"""Reading another store through its adapter: the engine (orchestra/stores/engine.py) against a fake
adapter (tests/fake_adapter.py) — the guard, the caps and the request log, the renaming rules,
kinds and split, the error line, a dead adapter started again, one that hangs — and the shipped
``orchestra-prometheus`` against a fake Prometheus (tests/fake_prometheus.py)."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest
from fake_prometheus import FakePrometheus

from orchestra import stores
from orchestra.stores import adapter, guard, reqlog
from orchestra.stores.engine import Engine
from orchestra.stores.guard import Refused, StoreError
from orchestra.stores.limits import DailyCap

NOW = time.time() // 60 * 60
FAKE = Path(__file__).resolve().parents[1] / "fake_adapter.py"
MATCH = '{__name__=~"queue_.*", env="prod"}'


def history():
    ts = [NOW - 3600 + 60 * i for i in range(61)]
    depth = [5 + (i % 7) for i in range(61)]
    return [
        ("queue_depth", {"env": "prod", "queue": "a", "pod": "p1"}, list(zip(ts, depth))),
        ("queue_depth", {"env": "prod", "queue": "b", "pod": "p1"}, [(t, 2.0) for t in ts]),
        ("queue_jobs_total", {"env": "prod", "queue": "a", "service": "pay"}, [(t, 3.0 * i) for i, t in enumerate(ts)]),
        ("queue_wait_bucket", {"env": "prod", "le": "1"}, [(t, 1.0) for t in ts]),
        ("queue_depth", {"env": "dev", "queue": "a"}, [(t, 99.0) for t in ts]),
    ]


@pytest.fixture(autouse=True)
def stopped():
    yield
    adapter.stop_all()


@pytest.fixture
def fake(tmp_path, monkeypatch):
    data = tmp_path / "data.json"
    data.write_text(json.dumps(history()))
    monkeypatch.setenv("FAKE_ADAPTER_DATA", str(data))
    monkeypatch.setenv("FAKE_ADAPTER_LOG", str(tmp_path / "asked.jsonl"))
    monkeypatch.setenv("FAKE_ADAPTER_DIE", str(tmp_path / "die"))
    return tmp_path


def asked(fake) -> list[dict]:
    p = fake / "asked.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []


def store_project(make_project, **over):
    return make_project({"stores": [{"name": "s", "command": f"{sys.executable} {FAKE}", "match": MATCH,
                                     "kinds": {"*_depth": "level"}, "names": {"queue_jobs_*": "{queue} jobs done"},
                                     "split": ["queue"], "rate": "100/s", "per_day": 1000} | over]})


def test_discovery_reads_each_metric_by_its_kind_name_and_split(make_project, fake):
    project = store_project(make_project)
    got, left = stores.read(project, project.config.stores[0], NOW - 1800, NOW, 60)
    by = {(s.name, s.labels.get("queue")): s for s in got}
    assert set(by) == {("queue_depth", "a"), ("queue_depth", "b"), ("a jobs done", "a")}
    assert by[("queue_depth", "a")].counter is False and by[("queue_depth", "b")].points[-1][1] == 2.0
    jobs = by[("a jobs done", "a")]
    assert jobs.counter and jobs.labels == {"queue": "a", "store": "s"}
    steps = [b - a for (_, a), (_, b) in zip(jobs.points, jobs.points[1:])]
    assert steps and all(abs(d - 3.0) < 1e-6 for d in steps), "a flow is its increments, added up"
    assert any("queue_wait_bucket" in x and "ignore" in x for x in left)
    assert all(v != 99.0 for s in got for _, v in s.points), "the dev series lie outside the match"
    for req in asked(fake):
        assert 'env="prod"' in (req.get("query") or req.get("match")), req
    log = reqlog.last(project, "s", 50)
    assert log and {x["op"] for x in log} == {"series", "range"} and all(x.get("sent", True) for x in log)
    assert "chatter" in (project.state_dir / "logs" / "stores" / "s.log").read_text(), "a line that is not JSON goes to its log"


def test_a_refused_query_is_never_asked_and_is_logged(make_project, fake):
    project = store_project(make_project)
    e = Engine(project, project.config.stores[0])
    for q in ("{}", '{__name__=~".+"}', 'rate(queue_depth[3d])', '{env="prod"}'):
        with pytest.raises(Refused):
            e.query_range(q, NOW - 600, NOW, 60)
    with pytest.raises(Refused, match="24 h at most"):
        e.query_range("queue_depth", NOW - 2 * 86400, NOW, 3600)
    assert asked(fake) == []
    log = reqlog.last(project, "s", 10)
    assert len(log) == 5 and all(x["sent"] is False and x["error"] for x in log)


def test_every_query_is_conjoined_with_the_match_and_its_series_counted(make_project, fake, monkeypatch):
    project = store_project(make_project)
    rows = Engine(project, project.config.stores[0]).query_range("sum by (queue) (queue_depth)", NOW - 600, NOW, 60)
    assert {r[0]["queue"] for r in rows} == {"a", "b"}
    series, rng = asked(fake)
    assert series == {"op": "series", "match": '{__name__="queue_depth",__name__=~"queue_.*",env="prod"}'}
    assert rng["op"] == "range" and rng["query"] == 'sum by (queue) ({__name__="queue_depth",__name__=~"queue_.*",env="prod"})'
    monkeypatch.setattr(guard, "MAX_SERIES", 1)
    with pytest.raises(Refused, match="more than 1 series"):
        Engine(project, project.config.stores[0]).query_range("queue_depth", NOW - 600, NOW, 60)
    assert [r["op"] for r in asked(fake)] == ["series", "range", "series"], "the range was never asked"


def test_a_long_window_is_asked_in_chunks(make_project, fake, monkeypatch):
    from orchestra.stores import engine
    monkeypatch.setattr(engine, "MAX_POINTS", 11)
    project = store_project(make_project)
    rows = Engine(project, project.config.stores[0]).query_range("queue_depth", NOW - 3600, NOW, 60, counted=True)
    assert len([r for r in asked(fake) if r["op"] == "range"]) == 6
    assert all(len(pts) == 61 for _, pts in rows), "the chunks merged into one series each"


def test_the_daily_cap(make_project, fake):
    project = store_project(make_project, per_day=2)
    e = Engine(project, project.config.stores[0])
    assert e.health() is None and e.health() is None
    with pytest.raises(DailyCap):
        e.query_range("queue_depth", NOW - 600, NOW, 60, counted=True)
    assert reqlog.last(project, "s", 1)[0]["sent"] is False and len(asked(fake)) == 2
    assert (project.state_dir / "stores" / "s.rate").exists()


def test_the_error_line(make_project, fake):
    project = store_project(make_project)
    e = Engine(project, project.config.stores[0])
    with pytest.raises(StoreError, match="the store said boom"):
        e.query_range("queue_boom", NOW - 600, NOW, 60, counted=True)
    assert reqlog.last(project, "s", 1)[0]["error"] == "the store said boom"
    assert e.health() is None, "the adapter answers the next request"


def test_a_dead_adapter_is_started_again(make_project, fake):
    project = store_project(make_project)
    e = Engine(project, project.config.stores[0])
    assert e.health() is None
    (fake / "die").write_text("")
    assert "exited before answering" in (e.health() or "")
    assert e.health() is None and e.adapter.starts == 2


def test_an_adapter_that_hangs_is_stopped(make_project, fake, monkeypatch):
    monkeypatch.setattr(adapter, "TIMEOUT", 1.0)
    project = store_project(make_project)
    e = Engine(project, project.config.stores[0])
    with pytest.raises(StoreError, match="no answer in 1s"):
        e.adapter.ask({"op": "range", "query": "queue_hang", "start": 0, "end": 1, "step": 1}, timeout=1.0)
    assert e.health() is None and e.adapter.starts == 2


def test_an_adapter_that_cannot_start(make_project):
    p = make_project({"stores": [{"name": "s", "command": "./no-such-adapter", "match": MATCH}]})
    (ok, line), = stores.check(p, p.config.stores[0])
    assert not ok and "cannot start the adapter" in line


def test_the_seams(make_project, fake, capsys):
    project = store_project(make_project)
    assert stores.query(project, "sum by (queue) (queue_depth)", "s") == 0
    rows = [json.loads(line.rsplit(" ", 1)[0]) for line in capsys.readouterr().out.splitlines()]
    assert {r["queue"] for r in rows} == {"a", "b"}
    assert stores.query(project, "{}", "s") == 2
    assert stores.query(project, "x", "nope") == 2
    (ok, line), = stores.check(project, project.config.stores[0])
    assert ok and "answers" in line and "of 1000 requests today" in line


# orchestra-prometheus, the shipped adapter

@pytest.fixture
def prom():
    f = FakePrometheus(history(), token="s3cret", prefix="/data/metrics")
    yield f
    f.close()


def prom_project(make_project, monkeypatch, url, extra=""):
    monkeypatch.setenv("PROM_TOKEN", "s3cret")
    return make_project({"stores": [{"name": "p", "command": f"orchestra-prometheus {url} --token-env PROM_TOKEN{extra}",
                                     "match": MATCH, "split": ["queue"], "rate": "100/s"}]})


def test_orchestra_prometheus_answers_the_three_requests(make_project, monkeypatch, prom):
    project = prom_project(make_project, monkeypatch, prom.url)
    e = Engine(project, project.config.stores[0])
    assert e.health() is None
    assert prom.seen[-1][0] == "/data/metrics/api/v1/status/buildinfo", "the path prefix is kept"
    listed = e.series(MATCH)
    assert {(x["__name__"], x.get("queue")) for x in listed} == {("queue_depth", "a"), ("queue_depth", "b"),
                                                                 ("queue_jobs_total", "a"), ("queue_wait_bucket", None)}
    rows = e.query_range("queue_depth", NOW - 600, NOW, 60)
    assert {(r[0]["__name__"], r[0]["queue"]) for r in rows} == {("queue_depth", "a"), ("queue_depth", "b")}
    assert all(len(pts) == 11 for _, pts in rows)
    assert all(h.get("authorization") == "Bearer s3cret" for _, h in prom.seen)
    got, _ = stores.read(project, project.config.stores[0], NOW - 1800, NOW, 60)
    assert {s.name for s in got} == {"queue_depth", "queue_jobs_total"}
    assert any(q.startswith("sum by (queue) (increase(") for q in prom.queries)


def test_orchestra_prometheus_auth_and_refusals(make_project, monkeypatch, prom):
    project = prom_project(make_project, monkeypatch, prom.url)
    e = Engine(project, project.config.stores[0])
    prom.refuse = [(429, {"Retry-After": "0"}, b"slow down"), (503, {"Retry-After": "0"}, b"<html>busy</html>")]
    assert e.query_range("queue_depth", NOW - 600, NOW, 60, counted=True), "a busy store is waited out"
    prom.refuse = [(401, {}, b"no")]
    with pytest.raises(StoreError, match="refused the credentials") as err:
        e.query_range("queue_depth", NOW - 600, NOW, 60, counted=True)
    assert "s3cret" not in str(err.value)
    monkeypatch.delenv("PROM_TOKEN")
    adapter.stop_all()
    assert "PROM_TOKEN is not set" in (Engine(project, project.config.stores[0]).health() or "")


def test_orchestra_prometheus_with_another_header(make_project, monkeypatch):
    f = FakePrometheus(history(), token="k", header="X-Key")
    try:
        project = prom_project(make_project, monkeypatch, f.url, " --header X-Key")
        monkeypatch.setenv("PROM_TOKEN", "k")
        assert Engine(project, project.config.stores[0]).health() is None
        assert f.seen[-1][1]["x-key"] == "k" and "authorization" not in f.seen[-1][1]
    finally:
        f.close()


def test_orchestra_prometheus_health_falls_back_to_a_cheap_query(make_project, monkeypatch, prom):
    from orchestra.stores import prometheus
    from orchestra.stores.http import Transport

    calls = []

    def call(self, path, form=None, *, post=True):
        calls.append(path)
        if path.endswith("buildinfo"):
            raise StoreError("http://x/api/v1/status/buildinfo: HTTP 404")
        return {"resultType": "vector", "result": []}
    monkeypatch.setattr(Transport, "call", call)
    assert prometheus.answer(Transport("http://x"), {"op": "health"}) == [{"ok": True}]
    assert calls == ["/api/v1/status/buildinfo", "/api/v1/query"]
