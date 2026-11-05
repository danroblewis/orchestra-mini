"""One conformance suite for every event-bus backend.

The file backend always runs. The server backends run when their URL is set (see README.md):
  ORCHESTRA_TEST_REDIS_URL   e.g. redis://localhost:6379/15
  ORCHESTRA_TEST_AMQP_URL    e.g. amqp://orchestra:orchestra@localhost:5672/
  ORCHESTRA_TEST_KAFKA_URL   e.g. kafka://localhost:9092
Each test uses its own stream, removed afterwards."""

from __future__ import annotations

import multiprocessing as mp
import os
import threading
import uuid
from pathlib import Path

import pytest

from orchestra.bus import open_bus, stream_name
from orchestra.config import Config
from orchestra.events import Events

BACKENDS = {
    "file": "file",
    "redis": os.environ.get("ORCHESTRA_TEST_REDIS_URL"),
    "rabbitmq": os.environ.get("ORCHESTRA_TEST_AMQP_URL"),
    "kafka": os.environ.get("ORCHESTRA_TEST_KAFKA_URL"),
}


def _cleanup(url: str, key: str) -> None:
    name = stream_name(key)
    if url.startswith("redis"):
        import redis
        r = redis.Redis.from_url(url)
        r.delete(name, name + ":n")
        r.close()
    elif url.startswith("amqp"):
        import pika
        c = pika.BlockingConnection(pika.URLParameters(url))
        c.channel().queue_delete(name)
        c.close()
    elif url.startswith("kafka"):
        from confluent_kafka.admin import AdminClient
        admin = AdminClient({"bootstrap.servers": url.split("://", 1)[1].split("?")[0]})   # keep it alive until the future resolves
        admin.delete_topics([name])[name].result(15)


@pytest.fixture(params=list(BACKENDS))
def make_bus(request, tmp_path):
    url = BACKENDS[request.param]
    if not url:
        pytest.skip(f"set the URL for the {request.param} backend to run it")
    opened, keys = [], []

    def make(key: str | None = None):
        """A bus on ``key``'s stream; with no key, the test's first stream (a second instance of it)."""
        if key is None:
            key = keys[0] if keys else f"test-{uuid.uuid4().hex[:12]}"
        if key not in keys:
            keys.append(key)
        bus = open_bus(url, project_key=key, file_path=tmp_path / key / "events.jsonl")
        opened.append(bus)
        return bus

    make.url = url
    make.tmp = tmp_path
    yield make
    for b in opened:
        b.close()
    if url != "file":
        for k in keys:
            _cleanup(url, k)


def _events(n: int, tag: str = "t") -> list[dict]:
    return [{"type": "probe", "tag": tag, "i": i} for i in range(n)]


def test_empty_stream(make_bus):
    bus = make_bus()
    assert bus.read(0) == ([], 0)
    assert bus.end() == 0


def test_appends_come_back_in_order(make_bus):
    bus = make_bus()
    for e in _events(10):
        bus.append(e)
    got, nxt = bus.read(0)
    assert [e["i"] for e in got] == list(range(10))
    assert nxt == 10 and bus.end() == 10


def test_cursor_reads_only_what_is_new(make_bus):
    bus = make_bus()
    for e in _events(10):
        bus.append(e)
    got, nxt = bus.read(7)
    assert [e["i"] for e in got] == [7, 8, 9] and nxt == 10
    assert bus.read(10) == ([], 10)
    assert bus.read(100) == ([], 10), "a cursor past the end reads nothing and returns the end"
    bus.append({"type": "probe", "i": 10})
    got, nxt = bus.read(10)
    assert [e["i"] for e in got] == [10] and nxt == 11


def test_limit_keeps_the_latest_and_startup_idiom(make_bus):
    bus = make_bus()
    for e in _events(20):
        bus.append(e)
    got, nxt = bus.read(0, limit=5)
    assert [e["i"] for e in got] == [15, 16, 17, 18, 19] and nxt == 20
    assert bus.read(0, limit=1)[1] == 20, "the loop finds the end with read(0, limit=1)"


def test_page_reads_forward_in_bounded_pages_and_every_event_once(make_bus):
    bus = make_bus()
    for e in _events(23):
        bus.append(e)
    got, nxt = bus.page(0, 5)
    assert [e["i"] for e in got] == [0, 1, 2, 3, 4] and nxt == 5
    seen, at = [], 0
    while at < bus.end():
        page, at = bus.page(at, 7)
        assert len(page) <= 7
        seen += [e["i"] for e in page]
    assert seen == list(range(23)) and at == 23
    assert bus.page(23, 5) == ([], 23)
    assert bus.page(100, 5) == ([], 23), "a cursor past the end reads nothing and returns the end"
    events = Events(bus=bus)
    assert [e["i"] for e in events.page(20, 100)[0]] == [20, 21, 22]


def test_file_pages_step_over_a_malformed_line_without_repeating_an_event(make_bus):
    if make_bus.url != "file":
        pytest.skip("a malformed line is written into the file backend only")
    bus = make_bus()
    for e in _events(3, "a"):
        bus.append(e)
    with bus.path.open("a") as f:
        f.write("{torn\n")
    for e in _events(5, "b"):
        bus.append(e)
    seen, at, pages = [], 0, 0
    while at < bus.end():
        page, at = bus.page(at, 4)
        pages += 1
        seen += [(e["tag"], e["i"]) for e in page]
    assert seen == [("a", 0), ("a", 1), ("a", 2), ("b", 0), ("b", 1), ("b", 2), ("b", 3), ("b", 4)]
    assert at == 9 and pages == 3


def test_a_second_instance_sees_the_same_stream(make_bus):
    a = make_bus()
    b = make_bus()          # same key: another process, another host
    a.append({"type": "probe", "i": 1})
    b.append({"type": "probe", "i": 2})
    assert [e["i"] for e in a.read(0)[0]] == [1, 2]
    assert [e["i"] for e in b.read(0)[0]] == [1, 2]


def test_concurrent_publishers_lose_nothing(make_bus):
    buses = [make_bus() for _ in range(4)]

    def publish(bus, tag):
        for e in _events(25, tag):
            bus.append(e)

    threads = [threading.Thread(target=publish, args=(b, f"w{k}")) for k, b in enumerate(buses)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    got, nxt = buses[0].read(0, limit=1000)
    assert nxt == 100 and len(got) == 100
    for k in range(4):
        mine = [e["i"] for e in got if e["tag"] == f"w{k}"]
        assert mine == list(range(25)), "each publisher's events stay in its own order"


def _publish_from_process(url: str, key: str, path: str, tag: str, n: int) -> None:
    bus = open_bus(url, project_key=key, file_path=Path(path))
    for i in range(n):
        bus.append({"type": "probe", "tag": tag, "i": i})
    bus.close()


def test_separate_processes_share_one_stream(make_bus):
    pkey = f"proc-{uuid.uuid4().hex[:10]}"
    first = make_bus(pkey)
    path = str(make_bus.tmp / pkey / "events.jsonl")
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=_publish_from_process, args=(make_bus.url, pkey, path, f"p{k}", 20)) for k in range(3)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(60)
        assert p.exitcode == 0
    got, nxt = first.read(0, limit=1000)
    assert nxt == 60
    for k in range(3):
        assert [e["i"] for e in got if e["tag"] == f"p{k}"] == list(range(20))


def test_history_survives_reopening(make_bus):
    bus = make_bus()
    for e in _events(5):
        bus.append(e)
    bus.close()
    again = make_bus()
    assert [e["i"] for e in again.read(0)[0]] == list(range(5))


def test_payloads_round_trip(make_bus):
    bus = make_bus()
    big = "x" * 200_000
    ev = {"type": "probe", "nested": {"a": [1, 2.5, None, True], "ü": "naïve ☃"}, "big": big}
    bus.append(ev)
    (back,), _ = bus.read(0)
    assert back == ev


def test_projects_do_not_see_each_other(make_bus):
    a = make_bus(f"iso-a-{uuid.uuid4().hex[:8]}")
    b = make_bus(f"iso-b-{uuid.uuid4().hex[:8]}")
    a.append({"type": "probe", "who": "a"})
    assert b.read(0) == ([], 0)
    assert [e["who"] for e in a.read(0)[0]] == ["a"]


def test_events_facade_over_any_bus(make_bus):
    bus = make_bus()
    events = Events(bus=bus)
    ev = events.publish("bread.baked", {"loaves": 2}, source="oven", subject="batch-1")
    assert ev["type"] == "bread.baked" and ev["specversion"] == "1.0"
    assert ev["source"] == "oven" and ev["subject"] == "batch-1" and ev["data"] == {"loaves": 2} and ev["id"] and ev["time"]
    got, nxt = events.read(0)
    assert got == [ev] and nxt == 1 == events.end(), "the envelope survives the round trip on every backend"


def test_project_config_and_environment_choose_the_backend(tmp_path, monkeypatch):
    from orchestra.project import Project

    project = Project(name="cfg-test", repo=tmp_path, config=Config(bus="file"), files=[])
    monkeypatch.setenv("ORCHESTRA_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("ORCHESTRA_BUS_URL", raising=False)
    assert Events.for_project(project).bus.name == "file"
    if BACKENDS["redis"]:
        monkeypatch.setenv("ORCHESTRA_BUS_URL", BACKENDS["redis"])
        ev = Events.for_project(project)
        assert ev.bus.name == "redis"
        ev.close()
