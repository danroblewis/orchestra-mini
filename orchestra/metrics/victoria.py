"""VictoriaMetrics backend, over its HTTP API (standard library only).

Writes: ``/api/v1/import`` (JSON lines, millisecond timestamps, backfill and out-of-order
allowed). Raw reads: ``/api/v1/export``. Ranges: raw points stepped locally with the same
carry-forward rule as the file backend, so both backends answer ``range`` identically.
Deletes: ``/api/v1/admin/tsdb/delete_series``. Every write ends with ``/internal/force_flush`` so
what was written can be read at once; at Orchestra's volumes that costs nothing worth saving."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request

from . import MetricsError, Point, matchers_of, refuse_future, step_sample


def _selector(series: str, labels: dict[str, str] | None, prefix: bool = False) -> str:
    parts = [f'__name__=~"{series}.*"'] if prefix else [f'__name__="{series}"']
    parts += [f'{k}="{v}"' for k, v in sorted((labels or {}).items())]
    return "{" + ",".join(parts) + "}"


def _matcher_selector(prefix: str, matchers) -> str:
    """A selector for ``recent``: the name by prefix, each label matcher quoted as a string literal."""
    parts = [f'__name__=~{json.dumps(re.escape(prefix) + ".*")}'] if prefix else ['__name__=~".+"']
    parts += [f"{k}{op}{json.dumps(v)}" for k, op, v in matchers_of(matchers)]
    return "{" + ",".join(parts) + "}"


class VictoriaMetrics:
    name = "victoriametrics"

    def __init__(self, url: str, future_skew: float | None = None):
        self.base = url.rstrip("/")
        self.future_skew = future_skew              # refuse points stamped further ahead (``refuse_future``)

    def _req(self, path: str, params: dict | None = None, body: bytes | None = None) -> bytes:
        url = self.base + path + ("?" + urllib.parse.urlencode(params, doseq=True) if params else "")
        try:
            with urllib.request.urlopen(urllib.request.Request(url, data=body, method="POST" if body is not None else "GET"),
                                        timeout=30) as r:
                return r.read()
        except OSError as e:
            raise MetricsError(f"VictoriaMetrics at {self.base}: {e}") from e

    def query_range(self, q: str, start: float, end: float, step: float, labels: dict[str, str] | None = None
                    ) -> list[tuple[dict, list[tuple[float, float]]]]:
        """Natively, through ``/api/v1/query_range``, every selector held to ``labels``
        (``extra_label``). A query VictoriaMetrics refuses raises ``MetricsError`` with its reason."""
        params = {"query": q, "start": start, "end": end, "step": f"{int(step)}s",
                  "extra_label": [f"{k}={v}" for k, v in sorted((labels or {}).items())]}
        url = f"{self.base}/api/v1/query_range?{urllib.parse.urlencode(params, doseq=True)}"
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                d = json.loads(r.read())
        except urllib.error.HTTPError as e:       # it explains a bad query in the body
            try:
                why = json.loads(e.read()).get("error", str(e))
            except ValueError:
                why = str(e)
            raise MetricsError(why) from None
        except OSError as e:
            raise MetricsError(f"VictoriaMetrics at {self.base}: {e}") from e
        if d.get("status") != "success":
            raise MetricsError(d.get("error", "query failed"))
        if d["data"]["resultType"] != "matrix":
            raise MetricsError(f"the query returned a {d['data']['resultType']}, not series")
        return [(res["metric"], [(float(t), float(v)) for t, v in res["values"]]) for res in d["data"]["result"]]

    def write(self, points: list[Point]) -> None:
        points = refuse_future(points, self.name, self.future_skew)
        if not points:
            return
        groups: dict[tuple, dict] = {}
        for p in points:
            g = groups.setdefault(p.key(), {"metric": {"__name__": p.series, **p.labels}, "values": [], "timestamps": []})
            g["values"].append(p.value)
            g["timestamps"].append(round(p.ts * 1000))
        body = "".join(json.dumps(g) + "\n" for g in groups.values()).encode()
        self._req("/api/v1/import", body=body)
        self._req("/internal/force_flush")      # read-after-write; cheap at Orchestra's volumes

    def _export(self, match: str, start=None, end=None) -> list[dict]:
        params = {"match[]": match}
        if start is not None:
            params["start"] = start
        if end is not None:
            params["end"] = end
        out = []
        for line in self._req("/api/v1/export", params).decode().splitlines():
            if line.strip():
                out.append(json.loads(line))
        return out

    def raw(self, series, labels=None, start=None, end=None):
        rows = self._export(_selector(series, labels), 0)        # all of it; the time window is applied here
        if len(rows) > 1:
            raise MetricsError(f"{_selector(series, labels)} matches {len(rows)} series; give more labels")
        if not rows:
            return []
        r = rows[0]
        pts = sorted(zip([t / 1000 for t in r["timestamps"]], r["values"]), key=lambda p: p[0])
        return [(t, v) for t, v in pts if (start is None or t >= start) and (end is None or t <= end)]

    def range(self, series, labels, start, end, step):
        return step_sample(self.raw(series, labels, None, end), start, end, step)

    def latest(self, series, labels=None):
        pts = self.raw(series, labels)
        return pts[-1] if pts else None

    def series(self, prefix="", labels=None):
        # /api/v1/series truncates when asked for all time (start=0), so list from the export, which
        # is reliable over any range and small at Orchestra's volumes
        out = [(dict(r["metric"]).pop("__name__"), {k: v for k, v in r["metric"].items() if k != "__name__"})
               for r in self._export(_selector(prefix, labels, prefix=True), 0)]
        return sorted(out, key=lambda x: (x[0], sorted(x[1].items())))

    def dump(self, prefix="", labels=None):
        out = []
        for r in self._export(_selector(prefix, labels, prefix=True), 0):
            lab = {k: v for k, v in r["metric"].items() if k != "__name__"}
            pts = sorted(zip([t / 1000 for t in r["timestamps"]], r["values"]), key=lambda p: p[0])
            out.append((r["metric"]["__name__"], lab, pts))
        return sorted(out, key=lambda g: (g[0], sorted(g[1].items())))

    def recent(self, prefix, matchers=None, since=0.0, keep=1):
        keep = max(1, int(keep))
        out = []
        for r in self._export(_matcher_selector(prefix, matchers), since):    # start=since: nothing older is read
            lab = {k: v for k, v in r["metric"].items() if k != "__name__"}
            pts = sorted(((t / 1000, v) for t, v in zip(r["timestamps"], r["values"])), key=lambda p: p[0])
            pts = [p for p in pts if p[0] >= since]
            if pts:
                out.append((r["metric"]["__name__"], lab, pts[-keep:]))
        return sorted(out, key=lambda g: (g[0], sorted(g[1].items())))

    def delete(self, series_prefix, labels=None):
        self._req("/api/v1/admin/tsdb/delete_series", {"match[]": _selector(series_prefix, labels, prefix=True)}, body=b"")

    def close(self) -> None:
        pass
