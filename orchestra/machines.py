"""Machines (SPEC §3): every process in the config and every ``source`` heard on the bus, with the
event types each published (and how many), what each listens for (a process's ``on``), when it
was last heard, its name from ``orchestra.actor`` lines, and a process's runs."""

from __future__ import annotations

from collections import Counter

from . import runs
from .events import Events
from .lines import ACTOR


def machines(project, events: Events | None = None) -> list[dict]:
    events = events or Events.for_project(project)
    found: dict[str, dict] = {}

    def entry(name: str) -> dict:
        return found.setdefault(name, {"name": name, "process": None, "label": None, "publishes": Counter(),
                                       "listens_for": [], "every": None, "continuous": False,
                                       "last_seen": None, "runs": 0, "last_run": None})

    for p in project.config.processes:
        m = entry(p.name)
        m.update(process=p.kind, listens_for=list(p.on), every=p.every, continuous=p.continuous)
        done = runs.read(project, p.name, 10_000)
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
    return sorted(out, key=lambda m: (m["process"] is None, m["name"]))
