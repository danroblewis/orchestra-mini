"""Reading a store against a fake Prometheus (tests/fake_prometheus.py): the guard, the caps and the
request log, auth from the environment, and the series discovery reads (kinds, names, split)."""

from __future__ import annotations

import json
import time

import pytest
from fake_prometheus import FakePrometheus

from orchestra import stores
from orchestra.stores import guard, reqlog
from orchestra.stores.guard import Refused, StoreError
from orchestra.stores.limits import DailyCap
from orchestra.stores.reader import Reader

NOW = time.time() // 60 * 60


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


@pytest.fixture
def fake():
    f = FakePrometheus(history(), token="s3cret")
    yield f
    f.close()


@pytest.fixture
def project(make_project, fake, monkeypatch):
    monkeypatch.setenv("FAKE_TOKEN", "s3cret")
    return make_project({"stores": [{
        "name": "prom", "url": fake.url, "token_env": "FAKE_TOKEN", "match": '{__name__=~"queue_.*", env="prod"}',
        "kinds": {"*_depth": "level"}, "names": {"queue_jobs_*": "{queue} jobs done"}, "split": ["queue"],
        "rate": "100/s", "per_day": 1000}]})


def test_discovery_reads_each_metric_by_its_kind_name_and_split(project, fake):
    got, left = stores.read(project, project.config.stores[0], NOW - 1800, NOW, 60)
    by = {(s.name, s.labels.get("queue")): s for s in got}
    assert set(by) == {("queue_depth", "a"), ("queue_depth", "b"), ("a jobs done", "a")}
    assert by[("queue_depth", "a")].counter is False and by[("queue_depth", "b")].points[-1][1] == 2.0
    jobs = by[("a jobs done", "a")]
    assert jobs.counter and jobs.labels == {"queue": "a", "store": "prom"}
    steps = [b - a for (_, a), (_, b) in zip(jobs.points, jobs.points[1:])]
    assert steps and all(abs(d - 3.0) < 1e-6 for d in steps), "a flow is its increments, added up"
    assert any("queue_wait_bucket" in x and "ignore" in x for x in left)
    assert all(v != 99.0 for s in got for _, v in s.points), "the dev series lie outside the match"
    for q in fake.queries:
        assert 'env="prod"' in q, q
    assert all(h.get("authorization") == "Bearer s3cret" for _, h in fake.seen)


def test_a_header_token(make_project, monkeypatch):
    f = FakePrometheus(history(), token="k", header="X-Key")
    try:
        monkeypatch.setenv("KEY", "k")
        p = make_project({"stores": [{"name": "s", "url": f.url, "match": '{env="prod"}', "auth": {"header": "X-Key", "token_env": "KEY"}}]})
        assert Reader(p, p.config.stores[0]).health() is None
        assert f.seen[-1][1]["x-key"] == "k" and "authorization" not in f.seen[-1][1]
    finally:
        f.close()


def test_a_missing_token_says_which_variable(make_project, fake, monkeypatch):
    monkeypatch.delenv("NOPE", raising=False)
    p = make_project({"stores": [{"name": "s", "url": fake.url, "match": '{env="prod"}', "token_env": "NOPE"}]})
    assert "NOPE is not set" in (Reader(p, p.config.stores[0]).health() or "")


def test_a_refused_query_is_never_sent_and_is_logged(project, fake):
    r = Reader(project, project.config.stores[0])
    before = fake.requests
    for q in ("{}", '{__name__=~".+"}', 'rate(queue_depth[3d])'):
        with pytest.raises(Refused):
            r.query_range(q, NOW - 600, NOW, 60)
    with pytest.raises(Refused, match="24 h at most"):
        r.query_range("queue_depth", NOW - 2 * 86400, NOW, 3600)
    assert fake.requests == before
    log = reqlog.last(project, "prom", 10)
    assert len(log) == 4 and all(x["sent"] is False and x["error"] for x in log)


def test_a_query_selecting_too_many_series_is_refused_before_it_runs(project, fake, monkeypatch):
    monkeypatch.setattr(guard, "MAX_SERIES", 1)
    with pytest.raises(Refused, match="more than 1 series"):
        Reader(project, project.config.stores[0]).query_range("queue_depth", NOW - 600, NOW, 60)
    assert not any(e == "query_range" for e, _ in fake.asks)


def test_the_daily_cap(make_project, fake, monkeypatch):
    monkeypatch.setenv("FAKE_TOKEN", "s3cret")
    p = make_project({"stores": [{"name": "s", "url": fake.url, "match": '{env="prod"}', "token_env": "FAKE_TOKEN",
                                  "rate": "100/s", "per_day": 2}]})
    r = Reader(p, p.config.stores[0])
    assert r.health() is None and r.health() is None
    with pytest.raises(DailyCap):
        r.query_range("queue_depth", NOW - 600, NOW, 60, counted=True)
    assert reqlog.last(p, "s", 1)[0]["sent"] is False
    assert (p.state_dir / "stores" / "s.rate").exists()


def test_a_busy_store_is_waited_out(project, fake):
    fake.refuse = [(429, {"Retry-After": "0"}, b"slow down"), (503, {"Retry-After": "0"}, b"<html>busy</html>")]
    r = Reader(project, project.config.stores[0])
    r.http._sleep = lambda s: None
    assert r.query_range("queue_depth", NOW - 600, NOW, 60, counted=True)
    fake.refuse = [(401, {}, b"no")]
    with pytest.raises(StoreError, match="refused the credentials") as e:
        r.query_range("queue_depth", NOW - 600, NOW, 60, counted=True)
    assert "s3cret" not in str(e.value)


def test_the_seams(project, fake, capsys):
    assert stores.query(project, 'sum by (queue) (queue_depth)', "prom") == 0
    rows = [json.loads(line.rsplit(" ", 1)[0]) for line in capsys.readouterr().out.splitlines()]
    assert {r["queue"] for r in rows} == {"a", "b"}
    assert stores.query(project, "{}", "prom") == 2
    assert stores.query(project, "x", "nope") == 2
    (ok, line), = stores.check(project, project.config.stores[0])
    assert ok and "answers" in line and "of 1000 requests today" in line
