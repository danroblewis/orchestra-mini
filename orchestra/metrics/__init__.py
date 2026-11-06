"""The metrics store: the history of every measured series.

A series is a name plus labels, like Prometheus. Every series carries a ``project`` label, so
projects never collide in a shared store, and readers select a project's series by it.
Orchestra's own series start with ``orchestra_``. Machines (``orchestra.output``) write
points; discovery and the pages read ranges.

Backends, chosen by ``metrics_store`` in the config or ``ORCHESTRA_METRICS_URL``: ``file`` (the
default: ``<state>/metrics.jsonl``) and ``http://host:port`` (VictoriaMetrics, or anything that
speaks its JSON-lines import and the Prometheus query API).

Reading a range uses the last value at or before each step (a stock holds its level until the
next change), the same rule Prometheus-style stores use within their lookback window. The file
backend carries a value forward without limit; ask VictoriaMetrics for ranges with a step no
longer than its lookback (5 minutes by default, ``-search.maxStalenessInterval``) or query raw."""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from ..project import Project


class MetricsError(Exception):
    pass


@dataclass(frozen=True)
class Point:
    series: str
    value: float
    ts: float                                   # unix seconds
    labels: dict[str, str] = field(default_factory=dict)

    def key(self) -> tuple:
        return (self.series, tuple(sorted(self.labels.items())))


class MetricsStore(Protocol):
    name: str

    def write(self, points: list[Point]) -> None: ...

    def raw(self, series: str, labels: dict[str, str] | None = None, start: float | None = None,
            end: float | None = None) -> list[tuple[float, float]]:
        """Every stored (ts, value) of the one series matching ``series`` and ``labels``, in time order."""

    def range(self, series: str, labels: dict[str, str] | None, start: float, end: float,
              step: float) -> list[tuple[float, float]]:
        """The series sampled every ``step`` seconds from ``start`` to ``end``: the last value at or
        before each step; steps before the first point are left out."""

    def latest(self, series: str, labels: dict[str, str] | None = None) -> tuple[float, float] | None: ...

    def series(self, prefix: str = "", labels: dict[str, str] | None = None) -> list[tuple[str, dict[str, str]]]:
        """Every series whose name starts with ``prefix`` (and matches ``labels``): names and labels."""

    def dump(self, prefix: str = "", labels: dict[str, str] | None = None
             ) -> list[tuple[str, dict[str, str], list[tuple[float, float]]]]:
        """Every series whose name starts with ``prefix`` (and matches ``labels``), with all its
        points in time order, in one pass: for catalogues and exploration."""

    def recent(self, prefix: str, matchers=None, since: float = 0.0, keep: int = 1
               ) -> list[tuple[str, dict[str, str], list[tuple[float, float]]]]:
        """Every series whose name starts with ``prefix``, whose labels match ``matchers`` (a
        list of ``(label, "=" | "=~", value)``, or a dict of equalities; ``matches``), and which has a
        point at or after ``since``: its name, labels and last ``keep`` points at or after
        ``since``, in time order. Nothing older than ``since`` is read on any backend."""

    def delete(self, series_prefix: str, labels: dict[str, str] | None = None) -> None:
        """Remove every series whose name starts with ``series_prefix`` (and matches ``labels``)."""

    def query_range(self, q: str, start: float, end: float, step: float, labels: dict[str, str] | None = None
                    ) -> list[tuple[dict[str, str], list[tuple[float, float]]]]:
        """Evaluate the PromQL ``q`` at every ``step`` from ``start`` to ``end`` over the series
        that carry ``labels`` (a project's, say): (labels, [(t, v)]) per result series.
        VictoriaMetrics evaluates it natively; elsewhere ``metrics.promql``'s subset does. Raises
        ``MetricsError`` with the reason when the query is refused."""

    def close(self) -> None: ...


Matcher = tuple[str, str, str]


def matchers_of(matchers) -> list[Matcher]:
    """``(label, op, value)`` triples from a list of them or a dict of equalities; ``op`` is ``=``
    (equal; a missing label is the empty string, as in Prometheus) or ``=~`` (a regular
    expression that must match the whole value). Raises ``MetricsError`` on any other op or a bad
    expression."""
    if not matchers:
        return []
    items = [(k, "=", v) for k, v in matchers.items()] if isinstance(matchers, dict) else list(matchers)
    out = []
    for m in items:
        if not (isinstance(m, tuple | list) and len(m) == 3) or m[1] not in ("=", "=~"):
            raise MetricsError(f"a matcher is (label, '=' or '=~', value), not {m!r}")
        if m[1] == "=~":
            try:
                re.compile(m[2])
            except re.error as e:
                raise MetricsError(f"{m[0]}=~{m[2]!r}: {e}") from None
        out.append((str(m[0]), m[1], str(m[2])))
    return out


def matches(labels: dict[str, str], matchers: list[Matcher]) -> bool:
    """Whether ``labels`` satisfy every matcher (``matchers_of``'s triples)."""
    for k, op, v in matchers:
        got = labels.get(k, "")
        if (got != v) if op == "=" else (re.fullmatch(v, got) is None):
            return False
    return True


FUTURE_SKEW_S = 60.0       # how far past now a point may be stamped (clocks drift) before it is refused
log = logging.getLogger("orchestra.metrics")


def refuse_future(points: list[Point], store: str, skew: float | None, now: float | None = None) -> list[Point]:
    """The points not stamped more than ``skew`` seconds past ``now``; the rest are refused and
    logged. ``None`` refuses nothing (a store made directly, for tests and replays on a clock of
    their own)."""
    if skew is None or not points:
        return points
    limit = (time.time() if now is None else now) + skew
    kept = [p for p in points if p.ts <= limit]
    if len(kept) < len(points):
        late = [p for p in points if p.ts > limit]
        log.warning("metrics store %s: refused %d point(s) stamped more than %g s in the future; the first: %s%s at %.0f, %.0f s ahead",
                    store, len(late), skew, late[0].series, late[0].labels, late[0].ts, late[0].ts - limit + skew)
    return kept


def open_metrics(url: str | None, *, file_path: Path) -> MetricsStore:
    """The store at ``url``. It refuses points stamped more than ``FUTURE_SKEW_S`` in the future,
    whoever writes them: a
    point is a measurement, and one stamped ahead would make its series, and what it is labelled
    with, look alive until a time not yet come."""
    if not url or url == "file":
        from .file import FileMetrics
        return FileMetrics(file_path, future_skew=FUTURE_SKEW_S)
    if url.startswith(("http://", "https://")):
        from .victoria import VictoriaMetrics
        return VictoriaMetrics(url, future_skew=FUTURE_SKEW_S)
    raise MetricsError(f"unknown metrics URL {url!r}: use file or http://host:port (VictoriaMetrics)")


def for_project(project: Project) -> ProjectStore:
    """The project's store, as its config names it, labelling what is written with the project."""
    url = os.environ.get("ORCHESTRA_METRICS_URL") or project.config.metrics_store
    return ProjectStore(open_metrics(url, file_path=project.state_dir / "metrics.jsonl"), project.name)


class ProjectStore:
    """A project's store that labels every point it writes with the project, unless the point
    names one; everything else is the store's own."""

    def __init__(self, store: MetricsStore, project: str):
        self.store, self.project = store, project

    def write(self, points: list[Point]) -> None:
        self.store.write([p if "project" in p.labels else Point(p.series, p.value, p.ts, {**p.labels, "project": self.project})
                          for p in points])

    def __getattr__(self, name):
        return getattr(self.store, name)


def step_sample(points: list[tuple[float, float]], start: float, end: float, step: float) -> list[tuple[float, float]]:
    """Last value at or before each step from start to end (points sorted by time)."""
    out: list[tuple[float, float]] = []
    i, last = 0, None
    t = start
    while t <= end + 1e-9:
        while i < len(points) and points[i][0] <= t:
            last = points[i][1]
            i += 1
        if last is not None:
            out.append((t, last))
        t += step
    return out
