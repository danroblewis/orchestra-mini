"""Machines (SPEC §3): every machine in the config and every ``source`` heard on the bus (one
Orchestra observes but doesn't run, such as a person), with the event types each published (and
how many), what each listens for (a machine's ``on``), how it runs (its kind, mode and
``every``), when it was last heard, its name from ``orchestra.actor`` lines, and its runs."""

from __future__ import annotations

from collections import Counter

from . import runs
from .events import Events
from .lines import ACTOR


def machines(project, events: Events | None = None) -> list[dict]:
    events = events or Events.for_project(project)
    found: dict[str, dict] = {}

    def entry(name: str) -> dict:
        return found.setdefault(name, {"name": name, "kind": None, "mode": None, "label": None, "publishes": Counter(),
                                       "listens_for": [], "every": None, "last_seen": None, "runs": 0, "last_run": None})

    for c in project.config.machines:
        m = entry(c.name)
        m.update(kind=c.kind, mode=c.mode, listens_for=list(c.on), every=c.every)
        done = runs.read(project, c.name, 10_000)
        m.update(runs=len(done), last_run=done[0] if done else None)
    for ev in events.all(0):
        src = ev.get("source")
        if not isinstance(src, str) or not src:
            continue
        if ev.get("type") == ACTOR:
            name = (ev.get("data") or {}).get("name") if isinstance(ev.get("data"), dict) else None
            if isinstance(name, str):
                entry(src)["label"] = name
            continue
        m = entry(src)
        m["publishes"][ev.get("type")] += 1
        if (t := ev.get("time")) and (m["last_seen"] is None or t > m["last_seen"]):
            m["last_seen"] = t
    out = []
    for m in found.values():
        m["publishes"] = dict(m["publishes"].most_common())
        out.append(m)
    return sorted(out, key=lambda m: (m["kind"] is None, m["name"]))
