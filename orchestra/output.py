"""What a machine says on stdout (SPEC §2), one line at a time: a JSON object with ``metric`` and
``value`` is a measurement, one with ``type`` a fact, anything else goes to its log.

Every fact a run writes is stamped in the envelope with its cause (SPEC §2): ``causedby``, the id
of the fact that started the run (``cause``: for a continuous machine the oldest it was given and
has not answered), unless the line names one itself, and ``depth``, its cause's plus one (0 with
no cause: a scheduled run). A fact deeper than ``facts.MAX_DEPTH`` is recorded but held: no machine
gets it, and ``held`` counts them for the run record.

A once run whose first non-empty line is Prometheus text (a ``#`` line or a sample) is read as
Prometheus text too: those lines are parsed together when the run ends and written only if it
succeeded. Every point is labelled ``machine=<name>`` (a label of that name given is kept as
``exported_machine``); the store adds ``project``. A point without a time takes the time it was
read. A series named ``orchestra_*`` is Orchestra's own and refused. A metric line's ``kind``
(``counter`` or ``gauge``) is kept in ``<state>/kinds.json`` for discovery; without one the name
says (``_total`` and ``_count`` are counters, the rest gauges). A line with ``cost_usd`` is a **run
line**: what the run (or a continuous machine's turn) cost, with a ``note`` when there is one."""

from __future__ import annotations

import json
import math
import time

from . import exposition, lines
from .counts import collapse
from .events import event_time
from .facts import held, observed
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


def stamp(ev: dict, cause: dict | None, publisher) -> dict:
    """``ev`` with ``causedby`` and ``depth``: the line's own ``causedby`` when it names one (its
    depth from the bus), else ``cause``'s; depth 0 with neither."""
    given = ev.get("causedby")
    base = cause.get("depth") if cause else None
    base = base if isinstance(base, int) and not isinstance(base, bool) else 0
    if given and (not cause or given != cause.get("id")):
        known = publisher.depth_of(given)
        ev["depth"] = (known if known is not None else base if cause else -1) + 1
    elif cause and cause.get("id"):
        ev["causedby"], ev["depth"] = cause["id"], base + 1
    else:
        ev["depth"] = 0
    return ev


class Output:
    """One run's (or one continuous machine's) stdout, read line by line. ``facts`` and ``points``
    say what it published and wrote, ``repeats`` the repeats it skipped, ``restated`` the facts
    that said again what the bus had (same type, subject and data), ``held`` the facts too deep to
    deliver; ``prometheus=False`` reads JSON lines only (an agent's text). ``cause()`` gives the
    fact that caused what it writes now, or None."""

    def __init__(self, project, publisher, store, machine: str, salt: str = "", log=None, prometheus: bool = True, on_run=None,
                 cause=None):
        self.project, self.publisher, self.store, self.machine = project, publisher, store, machine
        self.salt, self.log = salt, log or (lambda text: None)
        self.prom: list[str] | None = None if prometheus else []     # None: the first line decides
        self.t0 = time.time()
        self.cause = cause or (lambda: None)
        self.facts: list[dict] = []
        self.repeats = self.restated = self.held = 0
        self.points = 0
        self.kinds: dict[str, str] = {}
        self.cost: float | None = None
        self.notes: list[str] = []
        self.skipped = False
        self.on_run = on_run                    # a continuous machine's turn ends at each run line

    def line(self, text: str) -> None:
        s = text.strip()
        if not s:
            return
        try:
            raw = json.loads(s) if s.startswith("{") else None
        except ValueError:
            raw = None
        if isinstance(raw, dict) and "cost_usd" in raw and "type" not in raw and "metric" not in raw:
            self.run_line(raw)
        elif isinstance(raw, dict) and "metric" in raw and "value" in raw:
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
                self.publish(stamp(ev, self.cause(), self.publisher))
        elif self.prom is None and _prometheus_line(s):
            self.prom = [s]
        elif self.prom:
            self.prom.append(s)
        else:
            self.log(text.rstrip("\n"))
        if self.prom is None:
            self.prom = []                      # the first line was not Prometheus text: JSON lines only

    def run_line(self, raw: dict) -> None:
        cost = raw.get("cost_usd")
        cost = float(cost) if isinstance(cost, int | float) and not isinstance(cost, bool) and math.isfinite(cost) else 0.0
        if self.on_run is not None:
            return self.on_run(cost, raw.get("note"), bool(raw.get("skipped")))
        self.cost = (self.cost or 0.0) + cost
        if raw.get("note"):
            self.notes.append(str(raw["note"]))
        self.skipped = self.skipped or bool(raw.get("skipped"))

    def publish(self, ev: dict) -> None:
        restated = not observed(ev) and self.publisher.said(ev)
        done = self.publisher.publish(ev)
        if done == lines.PUBLISHED:
            self.facts.append(ev)
            self.restated += restated
            if held(ev):
                self.held += 1
                self.log(f"{ev['type']} {ev['id']} held: depth {ev['depth']} is past the limit; no machine gets it")
        elif done == lines.REPEAT:
            self.repeats += 1

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
