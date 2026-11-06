"""Fact lines (SPEC §2): what a machine or the endpoint says, one JSON object per line, made into a
fact with the envelope, and published once.

A line's ``type`` is required; ``source`` defaults to the command's name; ``id`` to a hash of the
line (with a salt: a process run's id, so each run's lines are new); ``time`` to now; ``causedby``
is kept when the line names it. A line that is not a JSON object with a ``type`` is not a fact:
the caller keeps it in the command's log. ``orchestra.actor`` lines (a source's name for the
pages) are published like any other.

The ``Publisher`` drops a fact whose id the bus has, and skips a repeat (``orchestra.facts``) when
it can tell, so the log grows only on change; the one read function skips any repeat that gets
past it. It also knows each fact's ``depth`` by id, and whether a fact restates one already on the
bus (same type, subject and data)."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path

from .events import Events, envelope, now
from .facts import Repeats, digest

ACTOR = "orchestra.actor"
PUBLISHED, DUPLICATE, REPEAT = "published", "duplicate", "repeat"


def parse(line: str, source: str, salt: str = "") -> tuple[dict | None, str | None]:
    """(the fact, None), or (None, why it is not one); a blank line is (None, None)."""
    text = line.strip()
    if not text:
        return None, None
    if not text.startswith("{"):
        return None, "not JSON"
    try:
        raw = json.loads(text)
    except ValueError:
        return None, "not JSON"
    if not isinstance(raw, dict):
        return None, "not a JSON object"
    return fact(raw, source, hashlib.sha256((salt + text).encode()).hexdigest()[:32])


def fact(raw: dict, source: str, default_id: str) -> tuple[dict | None, str | None]:
    """The envelope of a fact given as a map (a line, a POST body)."""
    t = raw.get("type")
    if not isinstance(t, str) or not t:
        return None, "no type"
    for k in ("subject", "source", "id", "time", "kind", "causedby"):
        if raw.get(k) is not None and not isinstance(raw[k], str):
            return None, f"{k} is not a string"
    try:
        ev = envelope(t, raw.get("source") or source, raw.get("data"), subject=raw.get("subject"),
                      time=raw.get("time"), id=raw.get("id") or default_id, causedby=raw.get("causedby") or None)
    except (TypeError, ValueError) as e:
        return None, str(e)
    if raw.get("kind"):
        ev["kind"] = raw["kind"]             # what the subject is (github.pr, jira.issue): latest says it
    return ev, None


def content(ev: dict) -> str:
    return digest([ev.get("type"), ev.get("subject"), ev.get("data")])


class Publisher:
    """Publishes facts onto the bus, dropping one whose id the bus already has and skipping a
    repeat. What is on the bus is read once, then what was appended since (by anyone) before each
    publish."""

    def __init__(self, events: Events, state: Path | None = None):
        self.events = events
        self.state = state                            # where the repeats' last-seen times are kept, if anywhere
        self._saved = 0.0
        self._lock = threading.Lock()
        self._depth: dict[str, int] = {}             # each id on the bus, and its depth
        self._said: set[str] = set()                 # each fact's type, subject and data
        self._repeats = Repeats()
        self._load()
        self._at = 0
        self._read = False

    def _take(self, ev: dict) -> None:
        if isinstance(ev.get("id"), str):
            d = ev.get("depth")
            self._depth[ev["id"]] = d if isinstance(d, int) and not isinstance(d, bool) else 0
        self._said.add(content(ev))

    @classmethod
    def for_project(cls, project, events: Events | None = None) -> Publisher:
        return cls(events or Events.for_project(project), project.path("repeats.json"))

    def _load(self) -> None:
        """The last-seen times of repeats skipped before, by any process: a repeat is never on the log,
        so a change after a restart could not say ``prevseen`` without them."""
        try:
            rows = json.loads(self.state.read_text()) if self.state else []
        except (OSError, ValueError):
            rows = []
        for row in rows if isinstance(rows, list) else []:
            if isinstance(row, list) and len(row) == 4:
                self._repeats.still[tuple(row[:3])] = row[3]

    def flush(self, force: bool = True) -> None:
        """Keep the last-seen times (at most once a second unless ``force``)."""
        if self.state is None or (not force and time.monotonic() - self._saved < 1.0):
            return
        self._saved = time.monotonic()
        tmp = self.state.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
        try:
            tmp.write_text(json.dumps([[*k, t] for k, t in self._repeats.still.items()]))
            tmp.replace(self.state)
        except OSError:
            pass

    def _refresh(self) -> None:
        self._read = True
        end = self.events.end()
        if end > self._at:
            for ev in self.events.all(self._at):
                self._take(ev)
                self._repeats.seen(ev)
            self._at = end

    def depth_of(self, fact_id: str | None) -> int | None:
        """The depth of a fact on the bus, or None when it is not known."""
        if not fact_id:
            return None
        with self._lock:
            if fact_id not in self._depth:
                self._refresh()
            return self._depth.get(fact_id)

    def said(self, ev: dict) -> bool:
        """Whether a fact with this one's type, subject and data is already on the bus."""
        with self._lock:
            if not self._read:
                self._refresh()
            return content(ev) in self._said

    def publish(self, ev: dict) -> str:
        """Append ``ev`` unless its id is on the bus (``duplicate``) or it repeats the last
        observation of its kind (``repeat``); ``published`` when it was appended."""
        with self._lock:
            self._refresh()
            if ev["id"] in self._depth:
                return DUPLICATE
            if self._repeats.seen(ev):
                self.flush(force=False)
                return REPEAT
            stamped = self._repeats.stamp(ev)
            if "prevseen" in stamped and "prevseen" not in ev:
                ev.update(stamped)
                self.flush()
            ev["recordedtime"] = now()
            self.events.bus.append(ev)
            self._take(ev)
            self._at += 1
            return PUBLISHED
