"""A stdlib fake of the Prometheus HTTP query API, evaluating with Orchestra's PromQL subset
(``orchestra/metrics/promql.py``) over points held in memory. Trimmed from the old Orchestra's
``tests/sources/fake_prometheus.py``: a required token (bearer, or another header), refusals queued
as a managed store sends them (a 429 or 503 with ``Retry-After``), a per-query series limit (422),
and a record of every request."""

from __future__ import annotations

import json
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from orchestra.metrics import promql


class TooMany(Exception):
    pass


class FakePrometheus:
    def __init__(self, dump=None, *, token: str | None = None, header: str = "Authorization"):
        self.dump = list(dump or [])            # (name, labels, [(t, v)])
        self.token, self.header = token, header
        self.seen: list[tuple[str, dict]] = []  # (path, headers by lower-case name)
        self.asks: list[tuple[str, dict]] = []  # (endpoint, parameters) of every request answered
        self.queries: list[str] = []
        self.refuse: list[tuple] = []           # the next answers: (status, headers, body bytes)
        self.max_series = 0
        self.requests = 0
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()

    def _authorized(self, headers) -> bool:
        if self.token is None:
            return True
        want = f"Bearer {self.token}" if self.header == "Authorization" else self.token
        return headers.get(self.header) == want

    def _limit(self, n: int) -> None:
        if self.max_series and n > self.max_series:
            raise TooMany(f"query matched {n} series, over the limit of {self.max_series} series per query")

    def answer(self, endpoint: str, form: dict[str, list[str]]):
        def one(k, d=None):
            return (form.get(k) or [d])[0]
        self.asks.append((endpoint, {k: list(v) for k, v in form.items()}))
        if endpoint == "query_range":
            q = one("query")
            self.queries.append(q)
            res = promql.query_range(self.dump, q, float(one("start")), float(one("end")), float(one("step")))
            self._limit(len(res))
            return {"resultType": "matrix", "result": [{"metric": lab, "values": [[t, repr(float(v))] for t, v in pts]} for lab, pts in res]}
        if endpoint == "query":
            q, at = one("query"), float(one("time", time.time()))
            self.queries.append(q)
            res = promql.query_range(self.dump, q, at, at, 60)
            return {"resultType": "vector", "result": [{"metric": lab, "value": [at, repr(pts[-1][1])]} for lab, pts in res]}
        if endpoint == "series":
            start, end = float(one("start", 0)), float(one("end", time.time()))
            seen = {}
            for m in form.get("match[]") or []:
                for lab, _ in promql.select(self.dump, m, start, end):
                    seen.setdefault(tuple(sorted(lab.items())), lab)
            out = sorted(seen.values(), key=lambda x: sorted(x.items()))
            return out[: int(one("limit"))] if one("limit") else out
        raise KeyError(endpoint)

    def _handler(self):
        fake = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _go(self, form: dict):
                fake.requests += 1
                path = urllib.parse.urlsplit(self.path).path
                fake.seen.append((path, {k.lower(): v for k, v in self.headers.items()}))
                endpoint = {"/api/v1/query_range": "query_range", "/api/v1/query": "query", "/api/v1/series": "series"}.get(path)
                if not fake._authorized(self.headers):
                    return self._send(401, {"status": "error", "error": "unauthorized"})
                if fake.refuse:
                    status, headers, body = fake.refuse.pop(0)
                    return self._raw(status, body, "text/html", headers)
                if endpoint is None:
                    return self._send(404, {"status": "error", "error": "not found"})
                try:
                    data = fake.answer(endpoint, form)
                except promql.PromQLError as e:
                    return self._send(400, {"status": "error", "errorType": "bad_data", "error": str(e)})
                except TooMany as e:
                    return self._send(422, {"status": "error", "errorType": "execution", "error": str(e)})
                self._send(200, {"status": "success", "data": data})

            def _send(self, code: int, body: dict):
                self._raw(code, json.dumps(body).encode(), "application/json", {})

            def _raw(self, code: int, raw: bytes, ctype: str, headers: dict):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(raw)))
                for k, v in headers.items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self):
                self._go(urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query))

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                self._go(urllib.parse.parse_qs(self.rfile.read(n).decode()))

        return H
