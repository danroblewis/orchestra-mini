"""A ``metrics:`` command's round (SPEC §1): run it, read its stdout as Prometheus text
(``exposition``), and write the points, each labelled with the project and ``command=<name>``
(a label of the text with that name is kept as ``exported_command``). A point without a
timestamp takes the time the round began. A series named ``orchestra_*`` is Orchestra's own and
refused. A round that fails writes nothing."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .command import Result, log_to, run
from .counts import collapse
from .exposition import parse
from .metrics import Point

PREFIX = "orchestra_"


@dataclass
class Round:
    points: list[Point] = field(default_factory=list)
    bad: list[tuple[int, str]] = field(default_factory=list)
    refused: list[str] = field(default_factory=list)
    error: str | None = None


def points_of(name: str, text: str, now: float) -> Round:
    parsed = parse(text)
    out = Round(bad=parsed.bad)
    points = []
    for s in parsed.samples:
        if s.name.startswith(PREFIX):
            if s.name not in out.refused:
                out.refused.append(s.name)
            continue
        labels = {("exported_command" if k == "command" else k): v for k, v in s.labels.items()}
        points.append(Point(s.name, s.value, s.ts_ms / 1000 if s.ts_ms is not None else now,
                            {**labels, "command": name}))
    out.points = collapse(points)
    return out


def collect(cmd, project, store, env=None, stop=None) -> Round:
    """One round of the ``metrics:`` entry ``cmd``, its points written to ``store``."""
    log = project.path("logs", f"{cmd.name}.log")
    now = time.time()
    res: Result = run(cmd.command, project.repo, log, timeout=cmd.timeout, env=env, stop=stop)
    if not res.ok:
        return Round(error=res.error or ("timed out" if res.timed_out else f"exit code {res.rc}"))
    r = points_of(cmd.name, "".join(res.lines), now)
    for n, why in r.bad:
        log_to(log, f"line {n} skipped: {why}")
    for name in r.refused:
        log_to(log, f"{name} refused: the orchestra_ prefix is Orchestra's own")
    store.write(r.points)
    return r
