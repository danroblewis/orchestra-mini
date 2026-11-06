"""Facts as they are read (SPEC §2, §6): the one read function every path uses, by which facts reach
a projection, a machine or a query.

It reads the bus in log order and gives each fact its ``seq``, its position in the log (the first
fact is 1, so a projection built to the end has ``built_to`` equal to the log's length). It skips
**repeats**: an observation (a type whose verb is ``observed``) whose ``data``, as canonical JSON,
equals the previous fact's with the same subject, type and source. A repeat says nothing new, so
nothing downstream ever sees one, whether or not the publisher caught it first.

``depth`` past ``MAX_DEPTH`` is **held**: such a fact is recorded and read like any other, but no
machine is given it (``held``)."""

from __future__ import annotations

import hashlib
import json
import threading

MAX_DEPTH = 8
PAGE = 5000


def observed(fact: dict) -> bool:
    """Whether the fact is an observation: a type whose verb is ``observed``."""
    return str(fact.get("type") or "").rsplit(".", 1)[-1] == "observed"


def canonical(data) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)


def digest(data) -> str:
    return hashlib.sha256(canonical(data).encode()).hexdigest()[:24]


def held(fact: dict) -> bool:
    d = fact.get("depth")
    return isinstance(d, int) and not isinstance(d, bool) and d > MAX_DEPTH


class Repeats:
    """The last ``data`` of each (subject, type, source) of observations, to tell a repeat."""

    def __init__(self):
        self.last: dict[tuple, str] = {}

    @staticmethod
    def key(fact: dict) -> tuple:
        return fact.get("subject"), fact.get("type"), fact.get("source")

    def seen(self, fact: dict) -> bool:
        """Whether ``fact`` repeats the last one of its kind; it becomes the last one either way."""
        if not observed(fact):
            return False
        k, d = self.key(fact), digest(fact.get("data"))
        same = self.last.get(k) == d
        self.last[k] = d
        return same


class FactLog:
    """The facts read so far, in log order, each with its ``seq``; ``update`` reads what the bus
    gained since. Safe to share between threads."""

    def __init__(self, events):
        self.events = events
        self.facts: list[dict] = []
        self.at = 0                                  # the bus offset read to
        self.repeats = Repeats()
        self.skipped = 0                             # repeats read and skipped
        self._lock = threading.Lock()

    def _page(self, at: int) -> tuple[list[tuple[int, dict]], int]:
        evs, nxt = self.events.page(at, PAGE)
        if len(evs) == nxt - at:
            return [(at + i + 1, ev) for i, ev in enumerate(evs)], nxt
        out = []                                     # a malformed line in this page: read one at a time
        for off in range(at, nxt):
            one, _ = self.events.page(off, 1)
            out += [(off + 1, ev) for ev in one]
        return out, nxt

    def update(self) -> list[dict]:
        """The facts the bus gained since the last update, repeats left out."""
        with self._lock:
            end = self.events.end()
            if end < self.at:                        # a shorter bus: another bus, read it again
                self.facts, self.at, self.repeats, self.skipped = [], 0, Repeats(), 0
            new: list[dict] = []
            while self.at < end:
                page, nxt = self._page(self.at)
                if nxt <= self.at:
                    break
                for seq, ev in page:
                    if self.repeats.seen(ev):
                        self.skipped += 1
                        continue
                    new.append(ev | {"seq": seq})
                self.at = nxt
            self.facts += new
            return new

    def all(self) -> list[dict]:
        self.update()
        return list(self.facts)

    def after(self, seq: int) -> list[dict]:
        """The facts read so far with a ``seq`` past ``seq``."""
        lo, hi = 0, len(self.facts)
        while lo < hi:                               # seqs only grow: find the first past it
            mid = (lo + hi) // 2
            if self.facts[mid]["seq"] <= seq:
                lo = mid + 1
            else:
                hi = mid
        return self.facts[lo:]

    def of(self, subject: str) -> list[dict]:
        return [f for f in self.facts if f.get("subject") == subject]


def read(events) -> list[dict]:
    """Every fact on the bus, in log order, with its ``seq``, repeats left out."""
    return FactLog(events).all()
