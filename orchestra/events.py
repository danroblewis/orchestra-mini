"""The facade over the bus that keeps the facts (named ``Events`` for the bus it wraps).

``Events`` is what the rest of Orchestra uses; the log itself is an ``EventBus``
(``orchestra.bus``): a JSONL file by default, or Redis Streams, a RabbitMQ stream or Kafka, chosen
by ``bus`` in the config or the ``ORCHESTRA_BUS_URL`` environment variable.

Every fact carries the **CloudEvents** envelope (``specversion`` 1.0): ``id``, ``type``,
``source`` (who asserts it), ``time`` (RFC 3339, when the source says it was true), optional
``subject`` (what it is about) and ``data``, which Orchestra reads only for the built-in
projection ``latest``; and the extensions ``causedby`` (the id of the fact that caused it),
``depth`` (its cause's plus one) and ``recordedtime`` (when the bus recorded it, stamped on
publishing)."""

from __future__ import annotations

import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .bus import EventBus, open_bus
from .bus.file import FileBus

if TYPE_CHECKING:
    from .project import Project

HOST = "orchestra"


def utc_time(t: datetime | str | float) -> str:
    """An envelope's ``time`` in its one format: UTC, to the millisecond, ending in ``Z``
    (``2026-10-27T18:58:31.043Z``), whatever it was given: a datetime, an ISO 8601 string with an
    offset, ``Z`` or ``+00:00``, or Unix seconds. A datetime or string with no zone is taken as UTC."""
    if isinstance(t, bool):
        raise TypeError("time: a datetime, an ISO 8601 string or Unix seconds")
    if isinstance(t, int | float):
        d = datetime.fromtimestamp(t, UTC)
    elif isinstance(t, datetime):
        d = t
    elif isinstance(t, str):
        try:
            d = datetime.fromisoformat(t.strip())
        except ValueError:
            raise ValueError(f"time {t!r} is not ISO 8601 (2026-10-27T18:58:31.043Z)") from None
    else:
        raise TypeError("time: a datetime, an ISO 8601 string or Unix seconds")
    d = d.replace(tzinfo=UTC) if d.tzinfo is None else d.astimezone(UTC)
    return d.strftime("%Y-%m-%dT%H:%M:%S.") + f"{d.microsecond // 1000:03d}Z"


def now() -> str:
    return utc_time(datetime.now(UTC))


def envelope(type: str, source: str, data: Any = None, *, subject: str | None = None,
             time: datetime | str | float | None = None, id: str | None = None, causedby: str | None = None,
             depth: int | None = None) -> dict:
    """A fact in the CloudEvents 1.0 envelope, as a dict: now and a fresh id unless given."""
    ev: dict[str, Any] = {"specversion": "1.0", "id": id or uuid.uuid4().hex, "type": type, "source": source,
                          "time": now() if time is None else utc_time(time)}
    if subject is not None:
        ev["subject"] = subject
    if causedby is not None:
        ev["causedby"] = causedby
    if depth is not None:
        ev["depth"] = depth
    if data is not None:
        ev["datacontenttype"] = "application/json"
        ev["data"] = data
    return ev


def event_time(ev: dict) -> float | None:
    """Unix seconds of a fact's ``time``, or None."""
    raw = ev.get("time")
    try:
        return datetime.fromisoformat(raw).timestamp() if raw else None
    except (TypeError, ValueError):
        return None


class Events:
    def __init__(self, path: Path | None = None, *, bus: EventBus | None = None, source: str = HOST):
        if bus is None and path is None:
            raise ValueError("Events needs a file path or a bus")
        self.bus: EventBus = bus or FileBus(path)
        self.path = path
        self.source = source

    @classmethod
    def for_project(cls, project: Project, source: str = HOST) -> Events:
        url = os.environ.get("ORCHESTRA_BUS_URL") or project.config.bus
        path = project.state_dir / "events.jsonl"
        return cls(path, bus=open_bus(url, project_key=project.name, file_path=path), source=source)

    def publish(self, type: str, data: Any = None, *, source: str | None = None, **ext: Any) -> dict:
        """Publish one fact with the CloudEvents envelope; ``ext`` takes subject, time, id,
        causedby and depth. The bus stamps ``recordedtime``."""
        ev = envelope(type, source or self.source, data, **ext)
        ev["recordedtime"] = now()
        self.bus.append(ev)
        return ev

    def emit(self, type: str, **data: Any) -> dict:
        """Publish a fact from this process's own source, its keyword arguments as the data."""
        return self.publish(type, data or None)

    def read(self, since: int = 0, limit: int = 500) -> tuple[list[dict], int]:
        """Return (facts, next_offset): the last ``limit`` facts after ``since``, as they are on
        the bus (repeats included, no ``seq``: ``orchestra.facts`` reads them for anything
        downstream), and the offset to read from next."""
        return self.bus.read(since, limit)

    def page(self, since: int = 0, limit: int = 500) -> tuple[list[dict], int]:
        """Return (facts, next_offset): the facts at the offsets after ``since`` up to
        ``min(since + limit, end)``, in order, and that bound to read from next (``EventBus.page``)."""
        return self.bus.page(since, limit)

    def after(self, offset: int) -> tuple[list[dict], int]:
        """Return (facts, next_offset): every fact after ``offset``, in order, and the offset
        to read from next. Unlike ``read`` nothing is cut off, even when facts arrive while it
        reads; nothing new is an empty list and the same offset."""
        room = max(self.end() - offset, 1)
        while True:
            evs, nxt = self.read(offset, room)
            if nxt - offset <= room:
                return evs, nxt
            room = nxt - offset                  # more arrived between end() and the read

    def all(self, since: int = 0, page: int = 5000):
        """Every fact after ``since`` as it is on the bus, in order, read in bounded pages."""
        at = since
        while at < self.end():
            evs, nxt = self.page(at, page)
            yield from evs
            if nxt <= at:
                break
            at = nxt

    def end(self) -> int:
        return self.bus.end()

    def close(self) -> None:
        self.bus.close()
