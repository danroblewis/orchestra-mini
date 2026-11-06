"""What a machine says on stdout (SPEC §2), one line at a time: a JSON object with ``metric`` and
``value`` is a measurement, one with ``type`` an event, anything else goes to its log.

A once run whose first non-empty line is Prometheus text (a ``#`` line or a sample) is read as
Prometheus text too: those lines are parsed together when the run ends and written only if it
succeeded. Every point is labelled ``machine=<name>`` (a label of that name given is kept as
``exported_machine``); the store adds ``project``. A point without a time takes the time it was
read. A series named ``orchestra_*`` is Orchestra's own and refused. A metric line's ``kind``
(``counter`` or ``gauge``) is kept in ``<state>/kinds.json`` for discovery; without one the name
says (``_total`` and ``_count`` are counters, the rest gauges)."""

from __future__ import annotations

import json
import math
import time

from . import exposition, lines
from .counts import collapse
from .events import event_time
from .metrics import Point

PREFIX = "orchestra_"
KINDS = ("counter", "gauge")


def _labels(given: dict, machine: str) -> dict[str, str]:
    return {("exported_machine" if k == "machine" else k): str(v) for k, v in given.items()} | {"machine": machine}


def metric(raw: dict, machine: str, now: float) -> tuple[Point | None, str | None]:
    """The point a metric line says, or (None, why not)."""
    name, value, labels, t = raw.get("metric"), raw.get("value"), raw.get("labels") or {}, raw.get("time")
    if not isinstance(name, str) or not exposition.NAME.fullmatch(name):
        return None, f"{name!r} is not a metric name"
    if name.startswith(PREFIX):
        return None, f"{name} refused: the orchestra_ prefix is Orchestra's own"
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        return None, f"{name}: value {value!r} is not a number"
    if not isinstance(labels, dict) or not all(isinstance(k, str) and exposition.LABEL.fullmatch(k) and
                                               isinstance(v, str | int | float) for k, v in labels.items()):
        return None, f"{name}: labels must map label names to values"
    if raw.get("kind") not in (None, *KINDS):
        return None, f"{name}: kind {raw['kind']!r} is not counter or gauge"
    ts = now if t is None else float(t) if isinstance(t, int | float) and not isinstance(t, bool) else event_time({"time": t})
    if ts is None:
        return None, f"{name}: bad time {t!r}"
    return Point(name, float(value), ts, _labels(labels, machine)), None


def prometheus(text: str, machine: str, now: float) -> tuple[list[Point], list[str]]:
    """The points of Prometheus text, and what was skipped, one line each."""
    parsed = exposition.parse(text)
    points, skipped = [], [f"line {n} skipped: {why}" for n, why in parsed.bad]
    for s in parsed.samples:
        if s.name.startswith(PREFIX):
            skipped.append(f"{s.name} refused: the orchestra_ prefix is Orchestra's own")
            continue
        points.append(Point(s.name, s.value, s.ts_ms / 1000 if s.ts_ms is not None else now, _labels(s.labels, machine)))
    return collapse(points), sorted(set(skipped), key=skipped.index)


def _prometheus_line(text: str) -> bool:
    try:
        return text.startswith("#") or bool(exposition.sample(text))
    except ValueError:
        return False


def remember_kinds(project, kinds: dict[str, str]) -> None:
    path = project.path("kinds.json")
    known = declared_kinds(project)
    if any(known.get(k) != v for k, v in kinds.items()):
        tmp = project.path(f"kinds.json.{time.monotonic_ns()}")
        tmp.write_text(json.dumps(known | kinds))
        tmp.replace(path)


def declared_kinds(project) -> dict[str, str]:
    """The kinds metric lines gave, by series name: ``counter`` or ``gauge``."""
    try:
        return json.loads((project.state_dir / "kinds.json").read_text())
    except (OSError, ValueError):
        return {}


class Output:
    """One run's (or one continuous machine's) stdout, read line by line. ``events`` and ``points``
    say what it published and wrote; ``prometheus=False`` reads JSON lines only (an agent's text)."""

    def __init__(self, project, publisher, store, machine: str, salt: str = "", log=None, prometheus: bool = True):
        self.project, self.publisher, self.store, self.machine = project, publisher, store, machine
        self.salt, self.log = salt, log or (lambda text: None)
        self.prom: list[str] | None = None if prometheus else []     # None: the first line decides
        self.t0 = time.time()
        self.events: list[dict] = []
        self.points = 0
        self.kinds: dict[str, str] = {}

    def line(self, text: str) -> None:
        s = text.strip()
        if not s:
            return
        try:
            raw = json.loads(s) if s.startswith("{") else None
        except ValueError:
            raw = None
        if isinstance(raw, dict) and "metric" in raw and "value" in raw:
            p, why = metric(raw, self.machine, time.time())
            if p is None:
                self.log(f"{s} skipped: {why}")
            else:
                if raw.get("kind") and self.kinds.get(p.series) != raw["kind"]:
                    self.kinds[p.series] = raw["kind"]
                    remember_kinds(self.project, {p.series: raw["kind"]})
                self.write([p])
        elif isinstance(raw, dict) and "type" in raw:
            ev, why = lines.parse(s, self.machine, self.salt)
            if ev is None:
                self.log(f"{s} skipped: {why}")
            else:
                self.publish(ev)
        elif self.prom is None and _prometheus_line(s):
            self.prom = [s]
        elif self.prom:
            self.prom.append(s)
        else:
            self.log(text.rstrip("\n"))
        if self.prom is None:
            self.prom = []                      # the first line was not Prometheus text: JSON lines only

    def publish(self, ev: dict) -> None:
        if self.publisher.publish(ev):
            self.events.append(ev)

    def write(self, points: list[Point]) -> None:
        if points:
            self.store.write(points)
            self.points += len(points)

    def finish(self, ok: bool) -> None:
        """The Prometheus text read, written if the run succeeded."""
        if self.prom:
            points, skipped = prometheus("\n".join(self.prom), self.machine, self.t0)
            for why in skipped:
                self.log(why)
            if ok:
                self.write(points)
            self.prom = []
