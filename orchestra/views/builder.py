"""The one builder of views (SPEC §6). Every view is built here, under one file lock in the state
directory (``views/.lock``), so two processes never build at once.

Each projection's views are kept in ``views/<name>.json``, replaced whole: ``hash`` (``run.hash_of``),
``built_to`` (the last ``seq`` built), ``views`` (by key) and ``build`` (when it last ran, how long
it took, its errors). A changed hash, a shorter log, or ``rebuild`` builds a projection again from
the whole log.

- A **per-subject** projection is run once per build with every fact of the subjects that have
  new facts (all of its subjects the first time, and any whose view has an error), in log order;
  it prints one line per subject. A subject it printed no line for, or a run that failed, keeps
  its last good view with ``error`` and ``error_at``.
- A **whole** projection is run with every matching fact whenever one is new, and its views are
  replaced as a set by the lines it prints (``{"key": …}``); a failed run keeps the last set, each
  with ``error`` and ``error_at``.
- ``latest`` is built in, in this process, for every subject."""

from __future__ import annotations

import fcntl
import fnmatch
import json
import os
import threading
import time
from contextlib import contextmanager

from .. import __version__, latest
from ..config.schema import Projection
from ..events import Events, now
from ..facts import FactLog
from .run import hash_of, run_projection

LATEST = Projection(name=latest.NAME, command="", match=["*"])


def projections(project) -> list[Projection]:
    """``latest`` first, then the config's, in config order."""
    return [LATEST, *project.config.projections]


def matches(p: Projection, fact: dict) -> bool:
    s = fact.get("subject")
    if not isinstance(s, str) or not s:
        return False
    if not any(fnmatch.fnmatchcase(s, g) for g in p.match):
        return False
    return not p.types or any(fnmatch.fnmatchcase(str(fact.get("type") or ""), g) for g in p.types)


def given(p: Projection, facts: list[dict], subject: str | None = None) -> list[dict]:
    """What the projection is given: its matching facts (of one subject, for a per-subject one)."""
    return [f for f in facts if matches(p, f) and (p.whole or subject is None or f.get("subject") == subject)]


def view_file(project, name: str):
    return project.path("views", f"{name}.json")


def load(project, name: str) -> dict:
    try:
        raw = json.loads(view_file(project, name).read_text())
        if isinstance(raw, dict) and isinstance(raw.get("views"), dict):
            return raw
    except (OSError, ValueError):
        pass
    return {"name": name, "hash": None, "built_to": 0, "views": {}, "build": None}


def save(project, state: dict) -> None:
    path = view_file(project, state["name"])
    tmp = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(state))
    tmp.replace(path)


@contextmanager
def locked(project):
    with project.path("views", ".lock").open("a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def kept(old: dict | None, key_name: str, key: str, error: str) -> dict:
    """The last good view, marked with what went wrong."""
    return (old or {key_name: key}) | {"error": error, "error_at": now()}


class Builder:
    """Brings every projection's views up to the log. Safe to share between threads."""

    def __init__(self, project, events: Events | None = None, log: FactLog | None = None, timeout: float | None = None):
        self.project = project
        self.log = log or FactLog(events or Events.for_project(project))
        self.timeout = timeout
        self._lock = threading.Lock()

    def build(self, rebuild: bool | set[str] = False, whole: bool = True) -> dict[str, dict]:
        """Build what is behind; ``rebuild`` (all, or these names) from the whole log;
        ``whole=False`` leaves the whole projections as they are. Each projection's state."""
        with self._lock, locked(self.project):
            self.log.update()
            facts, at = self.log.since(0)             # one snapshot: facts read later wait for the next build
            out = {}
            for p in projections(self.project):
                if p.whole and not whole:
                    continue
                again = rebuild is True or (isinstance(rebuild, set) and p.name in rebuild)
                out[p.name] = self._one(p, again, facts, at)
            return out

    def _one(self, p: Projection, again: bool, facts: list[dict], at: int) -> dict:
        state = load(self.project, p.name)
        h = hash_of(self.project, p) if p.command else f"latest:{__version__}"
        if again or state.get("hash") != h or state.get("built_to", 0) > at:
            state = state | {"hash": h, "built_to": 0}
            again = True
        new = [f for f in facts if f["seq"] > state["built_to"] and matches(p, f)]
        if not new and not again:
            if state["built_to"] != at and state.get("build") is not None:
                state["built_to"] = at
                save(self.project, state)
            return state
        t0 = time.monotonic()
        if p.whole:
            errors = self._whole(p, state, facts)
        else:
            errors = self._subjects(p, state, new, again, facts)
        state["built_to"] = at
        state["build"] = {"at": now(), "seconds": round(time.monotonic() - t0, 3), "errors": errors[:5],
                          "error": errors[0] if p.whole and errors else None}
        save(self.project, state)
        return state

    def _subjects(self, p: Projection, state: dict, new: list[dict], again: bool, facts: list[dict]) -> list[str]:
        views: dict = state["views"]
        mine = [f for f in facts if matches(p, f)]
        if again:
            pending = {f["subject"] for f in mine}
            for gone in set(views) - pending:
                views.pop(gone)
        else:
            pending = {f["subject"] for f in new} | {s for s, v in views.items() if v.get("error")}
        stdin = [f for f in mine if f["subject"] in pending]
        if not pending:
            return []
        if p.name == latest.NAME:
            by: dict[str, list[dict]] = {}
            for f in stdin:
                by.setdefault(f["subject"], []).append(f)
            views.update({s: latest.view(s, fs) for s, fs in by.items()})
            return []
        printed, err = run_projection(self.project, p, stdin, self.timeout)
        lines = {o["subject"]: o for o in printed if isinstance(o.get("subject"), str)}
        errors = []
        for s in sorted(pending):
            if err is None and s in lines:
                views[s] = lines[s]
            else:
                why = err or "the projection printed no line for it"
                views[s] = kept(views.get(s), "subject", s, why)
                errors.append(f"{s}: {why}")
        return errors

    def _whole(self, p: Projection, state: dict, facts: list[dict]) -> list[str]:
        printed, err = run_projection(self.project, p, [f for f in facts if matches(p, f)], self.timeout)
        if err is not None:
            state["views"] = {k: kept(v, "key", k, err) for k, v in state["views"].items()}
            return [err]
        state["views"] = {o["key"]: o for o in printed if isinstance(o.get("key"), str)}
        return []
