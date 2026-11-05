"""Bus calls are bounded and say when they fail: Redis opens with socket timeouts unless its URL
sets them, and a refused connection is a BusError; a Kafka append not acknowledged in time raises."""

import threading

import pytest

from orchestra.bus import BusError


def test_redis_opens_with_timeouts_unless_the_url_sets_them(monkeypatch):
    redis = pytest.importorskip("redis")
    from orchestra.bus import redis_streams

    seen = []
    real = redis.Redis.from_url
    monkeypatch.setattr(redis.Redis, "from_url", staticmethod(lambda url, **kw: seen.append(kw) or real(url, **kw)))
    redis_streams.RedisStreamsBus("redis://127.0.0.1:1/0", "s")
    redis_streams.RedisStreamsBus("redis://127.0.0.1:1/0?socket_timeout=2", "s")
    assert seen[0] == {"socket_timeout": 10.0, "socket_connect_timeout": 10.0}
    assert seen[1] == {"socket_connect_timeout": 10.0}


def test_a_redis_that_cannot_be_reached_is_a_bus_error():
    pytest.importorskip("redis")
    from orchestra.bus.redis_streams import RedisStreamsBus

    bus = RedisStreamsBus("redis://127.0.0.1:1/0", "s")
    with pytest.raises(BusError, match="Redis end"):
        bus.end()
    with pytest.raises(BusError, match="Redis append"):
        bus.append({"type": "x"})


def test_a_kafka_append_not_acknowledged_in_time_raises():
    from orchestra.bus.kafka import KafkaBus

    class Unacked:
        def produce(self, *a, **kw):
            pass

        def flush(self, timeout):
            return 1                          # one message still in the queue

    bus = KafkaBus.__new__(KafkaBus)
    bus._producer, bus._lock, bus._topic = Unacked(), threading.Lock(), "t"
    with pytest.raises(BusError, match="not acknowledged within 30 s"):
        bus.append({"type": "x"})
