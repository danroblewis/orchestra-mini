"""RabbitMQ backend: one stream queue per project, over AMQP 0.9.1. Needs ``pika``
(``orchestra[bus-rabbitmq]``). RabbitMQ 3.9 or later.

Stream queues keep every message and number them from 0, so a stream offset is exactly an
Orchestra offset. Publishing uses publisher confirms, so an append is durable when it returns.

The end of the stream: a passive declare reports no message count for stream queues, so ``end``
consumes from ``"last"`` (the start of the stream's last chunk) and takes the highest offset seen
plus one. That costs one short idle wait per call, which is fine for this backend's purpose (a
bus shared by several hosts) but makes it the slowest of the backends to poll."""

from __future__ import annotations

import json
import threading

from . import BusError

_IDLE = 0.15       # seconds without a delivery that mean "no more in this chunk"
_STALL = 10.0      # seconds without a delivery while reading up to a known end: give up


class RabbitStreamBus:
    name = "rabbitmq"

    def __init__(self, url: str, stream: str):
        try:
            import pika
        except ImportError as e:  # pragma: no cover - depends on the environment
            raise BusError("the RabbitMQ event bus needs the 'pika' package: install orchestra[bus-rabbitmq]") from e
        self._pika = pika
        self._params = pika.URLParameters(url)
        self._queue = stream
        self._lock = threading.Lock()          # pika connections are not thread-safe
        self._conn = None
        self._ch = None

    def _channel(self):
        if self._conn is None or self._conn.is_closed:
            self._conn = self._pika.BlockingConnection(self._params)
            self._ch = self._conn.channel()
            self._ch.queue_declare(self._queue, durable=True, arguments={"x-queue-type": "stream"})
            self._ch.confirm_delivery()
            self._ch.basic_qos(prefetch_count=500)
        return self._ch

    def append(self, event: dict) -> None:
        body = json.dumps(event).encode()
        with self._lock:
            self._channel().basic_publish("", self._queue, body, self._pika.BasicProperties(delivery_mode=2))

    def _consume(self, offset, stop_at: int | None) -> list[tuple[int, bytes]]:
        """Deliveries from ``offset`` (an int or "last") until ``stop_at`` is reached, or until the
        stream goes quiet when ``stop_at`` is None."""
        ch = self._channel()
        out: list[tuple[int, bytes]] = []
        timeout = _IDLE if stop_at is None else _STALL
        try:
            for method, props, body in ch.consume(self._queue, inactivity_timeout=timeout,
                                                  arguments={"x-stream-offset": offset}):
                if method is None:
                    if stop_at is not None:
                        raise BusError(f"RabbitMQ stream {self._queue}: no delivery for {_STALL}s before offset {stop_at}")
                    break
                ch.basic_ack(method.delivery_tag)
                n = int(props.headers["x-stream-offset"])
                out.append((n, body))
                if stop_at is not None and n >= stop_at:
                    break
        finally:
            ch.cancel()
        return out

    def end(self) -> int:
        with self._lock:
            last = self._consume("last", None)
        return max(n for n, _ in last) + 1 if last else 0

    def read(self, since: int = 0, limit: int = 500) -> tuple[list[dict], int]:
        end = self.end()
        start = max(since, end - limit) if limit else end
        if start >= end:
            return [], end
        with self._lock:
            got = self._consume(start, end - 1)
        return [json.loads(b) for n, b in got if start <= n < end], end

    def page(self, since: int = 0, limit: int = 500) -> tuple[list[dict], int]:
        end = self.end()
        if since >= end:
            return [], end
        bound = min(since + max(1, limit), end)
        with self._lock:
            got = self._consume(since, bound - 1)
        return [json.loads(b) for n, b in got if since <= n < bound], bound

    def close(self) -> None:
        with self._lock:
            if self._conn is not None and self._conn.is_open:
                self._conn.close()
