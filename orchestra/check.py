"""``orchestra check`` (SPEC §7): the config validated, then each part's health in one line: the
bus, the metrics store, each store, each command (can it start, its last round) and each process
(what starts it, its prompt or command, its last run)."""

from __future__ import annotations

import json
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
    try:
        health = json.loads((project.state_dir / "health.json").read_text())
    except (FileNotFoundError, ValueError):
        health = {}
    for c in [*project.config.events, *project.config.metrics]:
        why = _runnable(project, c.command)
        h = health.get(c.name)
        last = (f"last round {h['at']}: " + ("ok" if h.get("ok") else f"failed: {h.get('error')}")) if h else "not run yet"
        out.append((why is None and (not h or h.get("ok")), f"command {c.name}: {why or last}"))
    for t in project.config.tools:
        why = _runnable(project, t.command)
        out.append((why is None, f"tool {t.name}: {why or 'ok'}"))
    for p in project.config.processes:
        when = ", ".join(filter(None, [f"on {', '.join(p.on)}" if p.on else "",
                                       f"every {p.every:g}s" if p.every else "", "continuous" if p.continuous else ""]))
        if p.kind == "program":
            why = _runnable(project, p.command)
        elif p.kind == "agent" and not (project.repo / p.prompt).is_file():
            why = f"prompt {p.prompt} not found"
        else:
            why = None if shutil.which(project.config.agent.command) else f"{project.config.agent.command} not on PATH"
        last = runs.read(project, p.name, 1)
        state = f"last run {last[0]['start']} exit {last[0].get('exit')}" if last else "no runs yet"
        out.append((why is None, f"process {p.name} ({p.kind}, {when}): {why or state}"))
    return out
