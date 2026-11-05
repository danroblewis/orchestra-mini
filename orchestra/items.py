"""Items (SPEC §5): everything the bus says about one ``subject``, folded into its current state.

The fold: ``fields``, every key of every event's ``data``, the latest event (by ``time``, ties by
bus order) winning, a key set to null removed; ``type``, the latest event's; ``first`` and
``last``, the times of the first and latest events; ``events``, the history in time order;
``done``, whether the latest event's type ends ``.completed``, ``.closed`` or ``.cancelled`` (so
an event after one reopens the item). A reducer (``items: [{match, command}]``) replaces the fold
for the subjects its glob matches: it gets the item's events as JSON lines on stdin and prints its
state as one JSON object (``subject``, ``events``, ``first`` and ``last`` are added when it leaves
them out).

The folds are kept in ``<state>/items.json`` with the bus offset they were folded to, brought up
to date from there on every read, and rebuilt from the bus when the file is missing."""

from __future__ import annotations

import fnmatch
import json
import os
from pathlib import Path

from .command import log_to, run
from .events import Events, event_time

DONE = (".completed", ".closed", ".cancelled")


def fold(subject: str, events: list[dict]) -> dict:
    evs = sorted(events, key=lambda e: event_time(e) or 0.0)       # stable: ties keep bus order
    fields: dict = {}
    for e in evs:
        if isinstance(e.get("data"), dict):
            for k, v in e["data"].items():
                if v is None:
                    fields.pop(k, None)
                else:
                    fields[k] = v
    last = evs[-1] if evs else {}
    return {"subject": subject, "type": last.get("type"), "first": evs[0].get("time") if evs else None,
            "last": last.get("time"), "done": str(last.get("type", "")).endswith(DONE),
            "fields": fields, "events": evs}


def reduce(project, reducer, subject: str, events: list[dict]) -> dict:
    """The reducer's state for the item, or the fold when the reducer fails (said in its log)."""
    base = fold(subject, events)
    log = project.path("logs", "items-reducer.log")
    stdin = "".join(json.dumps(e) + "\n" for e in base["events"])
    res = run(reducer.command, project.repo, log, timeout=30, stdin=stdin)
    for line in reversed(res.lines if res.ok else []):
        try:
            state = json.loads(line)
        except ValueError:
            continue
        if isinstance(state, dict):
            return {k: base[k] for k in ("subject", "events", "first", "last")} | state
    log_to(log, f"{subject}: the reducer gave no state ({res.error or res.rc}); the fold is used")
    return base


class Items:
    def __init__(self, project, events: Events):
        self.project, self.events = project, events
        self.path: Path = project.path("items.json")
        self.at, self.items = 0, {}
        try:
            raw = json.loads(self.path.read_text())
            self.at, self.items = int(raw["at"]), dict(raw["items"])
        except (FileNotFoundError, ValueError, KeyError, TypeError):
            pass

    def _state(self, subject: str, events: list[dict]) -> dict:
        for r in self.project.config.items:
            if fnmatch.fnmatchcase(subject, r.match):
                return reduce(self.project, r, subject, events)
        return fold(subject, events)

    def update(self) -> list[str]:
        """Fold what the bus gained since; the subjects that changed."""
        end = self.events.end()
        if end < self.at:                       # a shorter bus: another bus, fold it again
            self.at, self.items = 0, {}
        if end == self.at:
            return []
        new: dict[str, list[dict]] = {}
        for ev in self.events.all(self.at):
            if isinstance(ev.get("subject"), str) and ev["subject"]:
                new.setdefault(ev["subject"], []).append(ev)
        for subject, evs in new.items():
            old = self.items.get(subject, {}).get("events", [])
            self.items[subject] = self._state(subject, old + evs)
        self.at = end
        tmp = self.path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps({"at": self.at, "items": self.items}))
        tmp.replace(self.path)
        return list(new)

    def get(self, subject: str) -> dict | None:
        self.update()
        return self.items.get(subject)

    def list(self, closed: bool = False) -> list[dict]:
        """The items, newest first; the closed ones only when asked."""
        self.update()
        out = [i for i in self.items.values() if closed or not i.get("done")]
        return sorted(out, key=lambda i: event_time({"time": i.get("last")}) or 0.0, reverse=True)
