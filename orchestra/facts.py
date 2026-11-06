"""Facts as they are read (SPEC §2, §6): the one read function every path uses, by which facts reach
a projection, a machine or a query.

It reads the bus in log order and gives each fact its ``seq``, its position in the log (the first
fact is 1, so a projection built to the end has ``built_to`` equal to the log's length). It skips
**repeats**: an observation (a type whose verb is ``observed``) whose ``data``, as canonical JSON,
equals the previous fact's with the same subject, type and source. A repeat is not passed on, but
it does say when the state was last seen unchanged: the next observation that differs gets
``prevseen``, the time of the last repeat skipped before it, so a projection knows a change happened
between ``prevseen`` and its ``time`` (the publisher stamps it when it skips the repeats itself, and
this function when the repeats are on the log).

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
        self.still: dict[tuple, str] = {}            # the time of the last repeat skipped, by kind

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
        if same and fact.get("time"):
            self.still[k] = fact["time"]
        return same

    def stamp(self, fact: dict) -> dict:
        """An observation that is not a repeat, with ``prevseen`` when repeats were skipped before it."""
        t = self.still.pop(self.key(fact), None) if observed(fact) else None
        return fact | {"prevseen": t} if t and "prevseen" not in fact else fact


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
            at = self.at
            while at < end:
                page, nxt = self._page(at)
                if nxt <= at:
                    break
                for seq, ev in page:
                    if self.repeats.seen(ev):
                        self.skipped += 1
                        continue
                    new.append(self.repeats.stamp(ev) | {"seq": seq})
                at = nxt
            self.facts += new                        # the facts first, then the offset they are read to
            self.at = at
            return new

    def since(self, seq: int) -> tuple[list[dict], int]:
        """The facts past ``seq`` and the offset they are read to, as one snapshot: what another
        thread reads in between is in neither, so a reader that moves its cursor to the offset
        never skips a fact."""
        with self._lock:
            return self.after(seq), self.at

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
