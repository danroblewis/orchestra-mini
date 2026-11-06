"""``orchestra-prometheus <url> [--token-env NAME] [--header NAME]``: the store adapter for any
server answering the Prometheus HTTP API (Prometheus, VictoriaMetrics, Thanos, Mimir, a managed
tenant; the URL may carry a path prefix, such as ``…/data/metrics``). It speaks the adapter
protocol (SPEC §4), one request a line on stdin, its answer on stdout:

- ``{"op": "series", "match": S}``: ``/api/v1/series`` over the last hour, one
  ``{"metric", "labels"}`` line per series;
- ``{"op": "range", "query": Q, "start", "end", "step"}``: ``/api/v1/query_range``, one
  ``{"metric", "labels", "points"}`` line per series (NaN and ±Inf dropped);
- ``{"op": "health"}``: ``/api/v1/status/buildinfo`` (or ``vector(1)`` where that is not served);
- each answer ends ``{"end": true}``; a failure is one ``{"error": …}`` line before it.

It asks what it is told: the guard, the caps and the log are Orchestra's, around every adapter."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time

from .guard import StoreError
from .http import Transport

SERIES_WINDOW_S = 3600.0


def _split(labels: dict) -> tuple[str, dict]:
    labels = dict(labels or {})
    return str(labels.pop("__name__", "")), labels


def _points(values: list) -> list[list[float]]:
    out = []
    for t, v in values or []:
        try:
            x = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(x):
            out.append([float(t), x])
    return out


def answer(t: Transport, req: dict) -> list[dict]:
    op = req.get("op")
    if op == "series":
        now = time.time()
        data = t.call("/api/v1/series", {"match[]": [str(req["match"])], "start": f"{now - SERIES_WINDOW_S:.3f}", "end": f"{now:.3f}"})
        if not isinstance(data, list):
            raise StoreError("/api/v1/series: not a Prometheus answer")
        return [dict(zip(("metric", "labels"), _split(x), strict=True)) for x in data if isinstance(x, dict)]
    if op == "range":
        form = {"query": str(req["query"]), "start": f"{float(req['start']):.3f}", "end": f"{float(req['end']):.3f}",
                "step": f"{float(req['step']):g}"}
        data = t.call("/api/v1/query_range", form)
        res = data.get("result") if isinstance(data, dict) else None
        if not isinstance(res, list):
            raise StoreError("/api/v1/query_range: not a Prometheus answer")
        out = []
        for r in res:
            name, labels = _split(r.get("metric"))
            out.append({"metric": name, "labels": labels, "points": _points(r.get("values"))})
        return out
    if op == "health":
        try:
            t.call("/api/v1/status/buildinfo", None, post=False)
        except StoreError as e:
            if "HTTP 404" not in str(e):
                return [{"ok": False, "error": str(e)}]
            try:
                t.call("/api/v1/query", {"query": "vector(1)"})
            except StoreError as e2:
                return [{"ok": False, "error": str(e2)}]
        return [{"ok": True}]
    raise StoreError(f"unknown op {op!r}: series, range or health")


def main(argv: list[str] | None = None) -> int:
    a = argparse.ArgumentParser(prog="orchestra-prometheus", description="A store adapter for the Prometheus HTTP API.")
    a.add_argument("url", help="the API's base, with any path prefix: https://tenant.example.io/data/metrics")
    a.add_argument("--token-env", help="the environment variable holding the token (sent as Bearer)")
    a.add_argument("--header", default="Authorization", help="the header the token goes in (default Authorization, as Bearer)")
    args = a.parse_args(argv)
    t = Transport(args.url, args.header, args.token_env)
    for raw in sys.stdin:
        if not raw.strip():
            continue
        try:
            req = json.loads(raw)
            lines = answer(t, req if isinstance(req, dict) else {})
        except (StoreError, ValueError, KeyError, TypeError) as e:
            lines = [{"error": str(e) or type(e).__name__}]
        for x in lines + [{"end": True}]:
            sys.stdout.write(json.dumps(x) + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
