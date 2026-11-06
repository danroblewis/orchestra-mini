"""What Orchestra does around every store adapter (SPEC §4): every request within the store's
``match`` and bounded in window and series (``guard``), under its ``rate`` and ``per_day``
(``limits``), and logged (``reqlog``); a refusal is logged as a request never sent.

- ``series(selector)``: the label sets (with ``__name__``) the selector selects within the match,
  at most ``limit``;
- ``query_range(q, start, end, step)``: ``q`` conjoined with the match, each of its selectors
  first counted with ``series`` (at most ``MAX_SERIES``), asked in chunks of fewer than
  ``MAX_POINTS`` points a series;
- ``health()``: why the store does not answer, or None."""

from __future__ import annotations

import math
import time

from . import guard, reqlog
from .adapter import adapter
from .guard import Refused, StoreError
from .limits import DailyCap, for_store

MAX_POINTS = 11_000           # Prometheus answers at most 11,000 points per series per query


class Engine:
    def __init__(self, project, store):
        self.project, self.store = project, store
        self.limiter = for_store(project, store)
        self.adapter = adapter(project, store)

    def _log(self, req: dict, began: float, *, answer: list | None = None, error: str | None = None, sent: bool = True) -> None:
        reqlog.append(self.project, self.store.name, {
            "op": req.get("op"), "query": req.get("query") or req.get("match"),
            "window": [req["start"], req["end"]] if "start" in req else None, **({"step": req["step"]} if "step" in req else {}),
            "series": len(answer) if answer is not None else None,
            "points": sum(len(x.get("points") or []) for x in answer) if answer is not None else None,
            "ms": round(1000 * (time.monotonic() - began), 1), **({"sent": False} if not sent else {}), **({"error": error} if error else {})})

    def _refuse(self, req: dict, why: str):
        self._log(req, time.monotonic(), error=why, sent=False)
        raise Refused(why)

    def ask(self, req: dict) -> list[dict]:
        began = time.monotonic()
        try:
            self.limiter.take(f"store {self.store.name}")
            out = self.adapter.ask(req)
        except StoreError as e:
            self._log(req, began, error=str(e), sent=not isinstance(e, DailyCap))
            raise
        self._log(req, began, answer=out)
        return out

    def series(self, selector: str, limit: int = guard.MAX_SERIES) -> list[dict]:
        req = {"op": "series", "match": selector}
        try:
            if selector.strip() == self.store.match.strip():
                guard.match_matchers(selector)          # the match itself: refused only when it selects everything
            else:
                req["match"] = guard.conjoined(selector, self.store.match)
        except Refused as e:
            self._refuse(req, str(e))
        return [{**(x.get("labels") or {}), "__name__": str(x.get("metric") or "")} for x in self.ask(req)][:limit]

    def query_range(self, q: str, start: float, end: float, step: float, counted: bool = False
                    ) -> list[tuple[dict, list[tuple[float, float]]]]:
        """Labels (with ``__name__`` when the answer names the metric) and points per series.
        ``counted``: the caller listed its series already (``read``); the window and the match still hold."""
        req = {"op": "range", "query": q, "start": start, "end": end, "step": step}
        try:
            guard.window(start, end)
            sent = guard.conjoined(q, self.store.match)
        except Refused as e:
            self._refuse(req, str(e))
        if not counted:
            for sel in guard.selectors(sent):
                if len(self.series(sel, guard.MAX_SERIES + 1)) > guard.MAX_SERIES:
                    self._refuse(req, f"{sel} selects more than {guard.MAX_SERIES} series: narrow it")
        merged: dict[tuple, tuple[dict, dict[float, float]]] = {}
        t, span = start, (MAX_POINTS - 1) * step
        while t <= end:
            hi = min(end, t + span)
            for x in self.ask({"op": "range", "query": sent, "start": t, "end": hi, "step": step}):
                labels = dict(x.get("labels") or {}) | ({"__name__": x["metric"]} if x.get("metric") else {})
                g = merged.setdefault(tuple(sorted(labels.items())), (labels, {}))
                g[1].update((float(a), float(b)) for a, b in x.get("points") or [] if math.isfinite(float(b)))
            t = hi + step
        if len(merged) > guard.MAX_SERIES:
            raise Refused(f"the answer holds {len(merged)} series, more than {guard.MAX_SERIES}")
        return [(labels, sorted(pts.items())) for labels, pts in merged.values()]

    def health(self) -> str | None:
        try:
            got = self.ask({"op": "health"})
        except StoreError as e:
            return str(e)
        ok = next((x for x in got if "ok" in x), {})
        return None if ok.get("ok") is True else str(ok.get("error") or "the adapter did not say ok")
