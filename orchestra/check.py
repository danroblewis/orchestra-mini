"""``orchestra check`` (SPEC §7): the config validated, then each part's health in one line: the
bus, the metrics store, each store, each tool (can it start) and each machine (its kind, how it
runs, whether its command or prompt is there, its last run)."""

from __future__ import annotations

import shutil

from . import hooks, metrics, runs
from .command import argv
from .events import Events


def _runnable(project, cmd: str) -> str | None:
    """Why the command cannot start, or None."""
    try:
        first = argv(cmd)[0]
    except (ValueError, IndexError):
        return "cannot be split into words"
    if "/" in first:
        return None if (project.repo / first).exists() else f"{first} not found"
    return None if shutil.which(first) else f"{first} not on PATH"


def lines(project) -> list[tuple[bool, str]]:
    out: list[tuple[bool, str]] = []
    try:
        ev = Events.for_project(project)
        out.append((True, f"bus {ev.bus.name}: {ev.end()} events"))
    except Exception as e:
        out.append((False, f"bus: {e}"))
    try:
        store = metrics.for_project(project)
        out.append((True, f"metrics store {store.name}: {len(store.series('', {'project': project.name}))} series"))
    except Exception as e:
        out.append((False, f"metrics store: {e}"))
    stores = hooks.hook("stores", "check")
    for s in project.config.stores:
        out += stores.check(project, s) if stores else [(True, f"store {s.name}: configured (reading stores is not installed)")]
    for t in project.config.tools:
        why = _runnable(project, t.command)
        out.append((why is None, f"tool {t.name}: {why or 'ok'}"))
    for m in project.config.machines:
        if m.kind == "program":
            why = _runnable(project, m.command)
        elif m.kind == "agent" and not (project.repo / m.prompt).is_file():
            why = f"prompt {m.prompt} not found"
        else:
            why = None if shutil.which(project.config.agent.command) else f"{project.config.agent.command} not on PATH"
        r = (runs.read(project, m.name, 1) or [None])[0]
        state = f"last run {r['start']} exit {r.get('exit')}" + (f": {r['note']}" if r.get("note") else "") if r else "no runs yet"
        out.append((why is None and not failed(r), f"machine {m.name} ({how(m)}): {why or state}"))
    return out


def failed(r: dict | None) -> bool:
    """Whether a run record says the run went wrong (a held-back run did not)."""
    return r is not None and not r.get("skipped") and (r.get("exit") not in (0, None) or bool(r.get("note")))


def how(m) -> str:
    """A machine's kind and when it runs, in a few words: ``program, every 60s``."""
    when = "continuous" if m.mode == "continuous" else ""
    return ", ".join(filter(None, [m.kind, when, f"on {', '.join(m.on)}" if m.on else "", f"every {m.every:g}s" if m.every else ""]))
