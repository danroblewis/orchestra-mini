"""Reading a store that answers the Prometheus HTTP query API (Prometheus, VictoriaMetrics, Thanos,
Mimir, a managed tenant), through the guard. Adapted from the old Orchestra's
``sources/prometheus.py``: ``query_range`` in chunks of fewer than ``MAX_POINTS`` points per
series, ``series`` for listings, and a ``health`` probe that touches no series. Every query is
conjoined with the store's ``match`` and bounded in window and series before it is sent
(``guard``); a refusal is logged as a request never sent."""

from __future__ import annotations

import math
import time

from . import guard
from .guard import Refused, StoreError
from .http import Transport
from .limits import for_store

MAX_POINTS = 11_000           # Prometheus answers at most 11,000 points per series per query
PROBE = "orchestra_probe_no_such_metric"


def _points(values: list) -> list[tuple[float, float]]:
    out = []
    for t, v in values:
        try:
            x = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(x):                    # staleness NaNs and ±Inf are dropped
            out.append((float(t), x))
    return out


class Reader:
    def __init__(self, project, store):
        self.store = store
        self.http = Transport(project, store, for_store(project, store))

    def _refuse(self, path: str, form: dict, why: str):
        self.http.log(path, form, time.monotonic(), error=why, sent=False)
        raise Refused(why)

    def series(self, selector: str, start: float, end: float, limit: int = guard.MAX_SERIES) -> list[dict]:
        """The label sets (with ``__name__``) of the series ``selector`` selects within the match, at
        most ``limit``."""
        form = {"match[]": [selector], "start": f"{start:.3f}", "end": f"{end:.3f}", "limit": str(limit)}
        try:
            guard.window(start, end)
            if selector.strip() == self.store.match.strip():
                guard.match_matchers(selector)          # the match itself: refused only when it selects everything
            else:
                form["match[]"] = [guard.conjoined(selector, self.store.match)]
        except Refused as e:
            self._refuse("/api/v1/series", form, str(e))
        data = self.http.call("/api/v1/series", form)
        if not isinstance(data, list):
            raise StoreError(f"{self.http.where()}/api/v1/series: not a Prometheus answer")
        return [dict(x) for x in data if isinstance(x, dict)][:limit]

    def bounded(self, q: str, start: float, end: float) -> str:
        """``q`` as it may be sent: conjoined with the match, its window and each selector's series
        within bounds (each counted with a listing of at most ``MAX_SERIES`` + 1)."""
        form = {"query": q, "start": f"{start:.3f}", "end": f"{end:.3f}"}
        try:
            guard.window(start, end)
            sent = guard.conjoined(q, self.store.match)
        except Refused as e:
            self._refuse("/api/v1/query_range", form, str(e))
        for sel in guard.selectors(sent):
            n = len(self.series(sel, start, end, guard.MAX_SERIES + 1))
            if n > guard.MAX_SERIES:
                self._refuse("/api/v1/query_range", form, f"{sel} selects more than {guard.MAX_SERIES} series: narrow it")
        return sent

    def query_range(self, q: str, start: float, end: float, step: float, counted: bool = False
                    ) -> list[tuple[dict, list[tuple[float, float]]]]:
        """``q`` evaluated every ``step`` from ``start`` to ``end``: labels (with ``__name__`` when the
        store keeps it) and points per result series. ``counted``: the caller listed its series
        already (``read``), so they are not counted again; the window and the match still hold."""
        if counted:
            try:
                guard.window(start, end)
                sent = guard.conjoined(q, self.store.match)
            except Refused as e:
                self._refuse("/api/v1/query_range", {"query": q}, str(e))
        else:
            sent = self.bounded(q, start, end)
        span = (MAX_POINTS - 1) * step
        merged: dict[tuple, tuple[dict, dict[float, float]]] = {}
        t = start
        while t <= end:
            hi = min(end, t + span)
            data = self.http.call("/api/v1/query_range", {"query": sent, "start": f"{t:.3f}", "end": f"{hi:.3f}", "step": f"{step:g}"})
            res = data.get("result") if isinstance(data, dict) else None
            if not isinstance(res, list):
                raise StoreError(f"{self.http.where()}/api/v1/query_range: not a Prometheus answer")
            for r in res:
                labels = dict(r.get("metric") or {})
                g = merged.setdefault(tuple(sorted(labels.items())), (labels, {}))
                g[1].update(_points(r.get("values") or []))
            t = hi + step
        if len(merged) > guard.MAX_SERIES:
            raise Refused(f"the answer holds {len(merged)} series, more than {guard.MAX_SERIES}")
        return [(labels, sorted(pts.items())) for labels, pts in merged.values()]

    def health(self) -> str | None:
        """Why the store does not answer, or None: a query about a name no store has, within the match."""
        try:
            sent = guard.conjoined(PROBE, self.store.match)
            self.http.call("/api/v1/query", {"query": sent, "time": f"{time.time():.3f}"})
        except StoreError as e:
            return str(e)
        return None
