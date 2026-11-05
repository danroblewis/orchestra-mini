"""How hard Orchestra may lean on another store: ``rate`` (``1/s``, ``30/m``) and ``per_day``.
Copied from the old Orchestra's ``sources/limits.py``, without the per-round budget and shares.

A token bucket per store, shared by every thread and, through a small file in the state directory
(``<state>/stores/<name>.rate``), by every process of the project that reads the store (the runner,
``orchestra query``, the pages). A 429's ``Retry-After`` holds the bucket for that long (``hold``).
Past ``per_day`` requests in one UTC day, ``DailyCap`` says so and when reads resume."""

from __future__ import annotations

import fcntl
import json
import os
import re
import threading
import time
from pathlib import Path

from .guard import StoreError

PER = {"s": 1, "m": 60, "h": 3600, "d": 86400}


class DailyCap(StoreError):
    """The store's daily request cap is reached: no request until the next UTC day."""


def per_second(rate: str) -> float:
    """``1/s``, ``30/m``, ``2/h`` or a plain number (per second) as requests per second."""
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*(?:/\s*([smhd]))?\s*", str(rate))
    if not m:
        raise StoreError(f"rate {rate!r}: give requests per unit, like 1/s or 30/m")
    return max(0.001, float(m[1]) / PER[m[2] or "s"])


def _day(t: float) -> str:
    return time.strftime("%Y-%m-%d", time.gmtime(t))


class RateLimit:
    def __init__(self, rate: float, burst: float, path: Path | None = None, *, per_day: int | None = None,
                 what: str = "the store", sleep=time.sleep, clock=time.time):
        self.rate, self.burst = max(0.001, float(rate)), max(1.0, float(burst))
        self.path = Path(path) if path else None
        self.per_day = int(per_day) if per_day else None
        self.what = what
        self._mem = {"tokens": self.burst, "t": clock(), "hold": 0.0, "day": _day(clock()), "count": 0}
        self._lock = threading.Lock()
        self._sleep, self._clock = sleep, clock

    def _state(self, fh) -> dict:
        try:
            fh.seek(0)
            d = json.loads(fh.read() or "{}")
            return {"tokens": float(d["tokens"]), "t": float(d["t"]), "hold": float(d.get("hold", 0.0)),
                    "day": str(d.get("day") or _day(self._clock())), "count": int(d.get("count") or 0)}
        except (ValueError, KeyError, TypeError):
            return {"tokens": self.burst, "t": self._clock(), "hold": 0.0, "day": _day(self._clock()), "count": 0}

    def _with_state(self, fn):
        """``fn(state) -> result`` under the bucket's locks; the state is written back."""
        with self._lock:
            if self.path is None:
                return fn(self._mem)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
            with os.fdopen(fd, "r+") as fh:
                fcntl.flock(fh, fcntl.LOCK_EX)
                try:
                    st = self._state(fh)
                    out = fn(st)
                    fh.seek(0)
                    fh.truncate()
                    fh.write(json.dumps(st))
                    fh.flush()
                    return out
                finally:
                    fcntl.flock(fh, fcntl.LOCK_UN)

    def _try(self, st: dict) -> float:
        """Take a token: 0 when taken, else how long to wait for one; -1 when the day's cap is spent."""
        now = self._clock()
        if st.get("day") != _day(now):
            st["day"], st["count"] = _day(now), 0
        if self.per_day is not None and st.get("count", 0) >= self.per_day:
            return -1.0
        st["tokens"] = min(self.burst, st["tokens"] + max(0.0, now - st["t"]) * self.rate)
        st["t"] = now
        if now < st["hold"]:
            return st["hold"] - now
        if st["tokens"] >= 1.0 - 1e-6:
            st["tokens"] = max(0.0, st["tokens"] - 1.0)
            st["count"] = st.get("count", 0) + 1
            return 0.0
        return max(0.001, (1.0 - st["tokens"]) / self.rate)

    def take(self, where: str = "") -> None:
        """Wait for a token; ``DailyCap`` when the day's cap is spent."""
        while True:
            wait = self._with_state(self._try)
            if wait < 0:
                raise DailyCap(f"{where}: {self.what} made its {self.per_day} requests today (per_day); "
                               "no more are made until 00:00 UTC")
            if wait == 0:
                return
            self._sleep(min(wait, 5.0))

    def hold(self, seconds: float) -> None:
        """No request from any reader for ``seconds``: the server asked for that (Retry-After)."""
        def fn(st):
            st["hold"] = max(st["hold"], self._clock() + seconds)
        self._with_state(fn)

    def today(self) -> int:
        """Requests given out today (UTC), every process counted."""
        return self._with_state(lambda st: st.get("count", 0) if st.get("day") == _day(self._clock()) else 0)


def for_store(project, store) -> RateLimit:
    rate = per_second(store.rate)
    return RateLimit(rate, max(1.0, rate), project.path("stores", f"{store.name}.rate"), per_day=store.per_day,
                     what=f"the store {store.name}")
