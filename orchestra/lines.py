"""Event lines (SPEC §2): what a machine or the endpoint says, one JSON object
per line, made into an event with the envelope, and published once.

A line's ``type`` is required; ``source`` defaults to the command's name; ``id`` to a hash of the
line (with a salt: a process run's id, so each run's lines are new); ``time`` to now. A line that
is not a JSON object with a ``type`` is not an event: the caller keeps it in the command's log.
``orchestra.actor`` lines (a source's name for the pages) are published like any other."""

from __future__ import annotations

import hashlib
import json
import threading

from .events import Events, envelope

ACTOR = "orchestra.actor"


def parse(line: str, source: str, salt: str = "") -> tuple[dict | None, str | None]:
    """(the event, None), or (None, why it is not one); a blank line is (None, None)."""
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
    return event(raw, source, hashlib.sha256((salt + text).encode()).hexdigest()[:32])


def event(raw: dict, source: str, default_id: str) -> tuple[dict | None, str | None]:
    """The envelope of an event given as a map (a line, a POST body)."""
    t = raw.get("type")
    if not isinstance(t, str) or not t:
        return None, "no type"
    for k in ("subject", "source", "id", "time", "kind"):
        if raw.get(k) is not None and not isinstance(raw[k], str):
            return None, f"{k} is not a string"
    try:
        ev = envelope(t, raw.get("source") or source, raw.get("data"), subject=raw.get("subject"),
                      time=raw.get("time"), id=raw.get("id") or default_id)
    except (TypeError, ValueError) as e:
        return None, str(e)
    if raw.get("kind"):
        ev["kind"] = raw["kind"]             # what the subject is (github.pr, jira.issue): items say it
    return ev, None


class Publisher:
    """Publishes events onto the bus, dropping one whose id the bus already has. The ids are read
    from the bus once, then from what was appended since (by anyone) before each publish."""

    def __init__(self, events: Events):
        self.events = events
        self._lock = threading.Lock()
        self._seen: set[str] | None = None
        self._at = 0

    def _refresh(self) -> None:
        if self._seen is None:
            self._seen = set()
        end = self.events.end()
        if end > self._at:
            for ev in self.events.all(self._at):
                if isinstance(ev.get("id"), str):
                    self._seen.add(ev["id"])
            self._at = end

    def publish(self, ev: dict) -> bool:
        """Append ``ev`` unless its id is on the bus; whether it was appended."""
        with self._lock:
            self._refresh()
            if ev["id"] in self._seen:
                return False
            self.events.bus.append(ev)
            self._seen.add(ev["id"])
            self._at += 1
            return True
