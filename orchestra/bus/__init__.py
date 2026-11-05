"""The event bus: Orchestra's append-only, replayable log of facts.

Every backend keeps one ordered stream per project, under Orchestra's own namespace
(``orchestra.<project key>.events``), never mixed with anyone else's topics. Offsets are integers:
an offset is the number of events before a point in the stream, so ``read(0)`` starts at the
beginning and the offset ``read`` returns is where the next call should start. Callers treat
offsets as cursors and never do arithmetic on them.

Backends are chosen by URL (``open_bus``): ``file`` (the default, a JSONL file in the project's
state directory), ``redis://...`` (Redis Streams), ``amqp://...`` (a RabbitMQ stream queue) and
``kafka://host:port`` (one Kafka topic with one partition). Each server backend needs its client
library, installed with the matching extra (``orchestra[bus-redis]`` and so on).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Protocol


class BusError(Exception):
    pass


class EventBus(Protocol):
    name: str

    def append(self, event: dict) -> None:
        """Add one event at the end of the stream. Durable when it returns."""

    def read(self, since: int = 0, limit: int = 500) -> tuple[list[dict], int]:
        """Events after offset ``since``: at most the last ``limit`` of them, in order, and the
        offset of the end of the stream (where the next read should start)."""

    def page(self, since: int = 0, limit: int = 500) -> tuple[list[dict], int]:
        """A bounded read forward: the events at the offsets after ``since`` up to
        ``min(since + limit, end)``, in order, and that bound as the offset to read from next,
        whatever was skipped there (a malformed line, events a retention removed). Paging from 0
        reads every event once, each page costing at most ``limit`` events."""

    def end(self) -> int:
        """The offset of the end of the stream: how many events it holds."""

    def close(self) -> None: ...


def stream_name(project_key: str) -> str:
    """Orchestra's own namespace for a project's events, safe for every backend's naming rules."""
    return "orchestra." + re.sub(r"[^A-Za-z0-9_-]", "_", project_key) + ".events"


def open_bus(url: str | None, *, project_key: str, file_path: Path) -> EventBus:
    """The bus a URL names. None, '' or 'file' is the JSONL file at ``file_path``."""
    if not url or url == "file":
        from .file import FileBus
        return FileBus(file_path)
    scheme = url.split(":", 1)[0]
    name = stream_name(project_key)
    if scheme in ("redis", "rediss"):
        from .redis_streams import RedisStreamsBus
        return RedisStreamsBus(url, name)
    if scheme in ("amqp", "amqps"):
        from .rabbitmq_streams import RabbitStreamBus
        return RabbitStreamBus(url, name)
    if scheme == "kafka":
        from .kafka import KafkaBus
        return KafkaBus(url, name)
    raise BusError(f"unknown event bus URL {url!r}: use file, redis://, amqp:// or kafka://")
