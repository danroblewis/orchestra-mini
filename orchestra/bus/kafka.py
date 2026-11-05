"""Kafka backend: one topic with one partition per project, so the whole stream is ordered and a
Kafka offset is exactly an Orchestra offset. Needs ``confluent-kafka`` (``orchestra[bus-kafka]``).

URL: ``kafka://host:port[,host:port...]``. The topic is created on first use (replication factor 1
unless ``?replication=N``); retention is the broker's, so set ``retention.ms=-1`` on the topic or
broker to keep the whole history. Each append waits for the broker's acknowledgement."""

from __future__ import annotations

import json
import threading
import uuid
from urllib.parse import parse_qs, urlparse

from . import BusError


class KafkaBus:
    name = "kafka"

    def __init__(self, url: str, topic: str):
        try:
            import confluent_kafka
            from confluent_kafka.admin import AdminClient, NewTopic
        except ImportError as e:  # pragma: no cover - depends on the environment
            raise BusError("the Kafka event bus needs the 'confluent-kafka' package: install orchestra[bus-kafka]") from e
        u = urlparse(url)
        self._servers = u.netloc
        replication = int(parse_qs(u.query).get("replication", ["1"])[0])
        self._ck = confluent_kafka
        self._topic = topic
        self._lock = threading.Lock()
        admin = AdminClient({"bootstrap.servers": self._servers})
        if topic not in admin.list_topics(timeout=10).topics:
            fut = admin.create_topics([NewTopic(topic, num_partitions=1, replication_factor=replication,
                                                config={"retention.ms": "-1"})])[topic]
            try:
                fut.result(timeout=15)
            except confluent_kafka.KafkaException as e:
                if e.args[0].code() != confluent_kafka.KafkaError.TOPIC_ALREADY_EXISTS:
                    raise
        self._producer = confluent_kafka.Producer({"bootstrap.servers": self._servers, "acks": "all",
                                                   "enable.idempotence": True})
        self._consumer = confluent_kafka.Consumer({"bootstrap.servers": self._servers,
                                                   "group.id": f"orchestra-reader-{uuid.uuid4()}",
                                                   "enable.auto.commit": False, "auto.offset.reset": "earliest"})
        self._tp = confluent_kafka.TopicPartition(topic, 0)
        self._closed = False

    def append(self, event: dict) -> None:
        errors: list = []
        with self._lock:
            self._producer.produce(self._topic, json.dumps(event).encode(), partition=0,
                                   on_delivery=lambda err, _msg: err and errors.append(err))
            unsent = self._producer.flush(30)
        if unsent:
            raise BusError(f"Kafka append to {self._topic}: not acknowledged within 30 s")
        if errors:
            raise BusError(f"Kafka append to {self._topic} failed: {errors[0]}")

    def end(self) -> int:
        with self._lock:
            _low, high = self._consumer.get_watermark_offsets(self._tp, timeout=10, cached=False)
        return high

    def _range(self, start: int, bound: int) -> list[dict]:
        """The messages at offsets start..bound-1 (under the lock)."""
        self._consumer.assign([self._ck.TopicPartition(self._topic, 0, start)])
        out: list[dict] = []
        try:
            while True:
                msg = self._consumer.poll(10)
                if msg is None:
                    raise BusError(f"Kafka topic {self._topic}: no message for 10s before offset {bound - 1}")
                if msg.error():
                    raise BusError(f"Kafka read from {self._topic}: {msg.error()}")
                if msg.offset() < bound:
                    out.append(json.loads(msg.value()))
                if msg.offset() >= bound - 1:
                    break
        finally:
            self._consumer.unassign()
        return out

    def read(self, since: int = 0, limit: int = 500) -> tuple[list[dict], int]:
        with self._lock:
            low, end = self._consumer.get_watermark_offsets(self._tp, timeout=10, cached=False)
            start = max(since, end - limit, low) if limit else end
            if start >= end:
                return [], end
            return self._range(start, end), end

    def page(self, since: int = 0, limit: int = 500) -> tuple[list[dict], int]:
        with self._lock:
            low, end = self._consumer.get_watermark_offsets(self._tp, timeout=10, cached=False)
            if since >= end:
                return [], end
            start = max(since, low)                 # what retention removed is skipped, never waited for
            bound = min(start + max(1, limit), end)
            return (self._range(start, bound) if start < bound else []), bound

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._producer.flush(10)
            self._consumer.close()
