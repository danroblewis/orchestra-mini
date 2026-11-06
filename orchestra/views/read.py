"""Reading views (SPEC §6): what the pages, ``orchestra query`` and machines read.

``Views`` brings the views up to the log first only when no ``orchestra run`` is alive (it builds
them itself, after each batch of facts); otherwise it reads the files as they are. Every view it
gives carries ``behind``: how many matching facts have a ``seq`` past its projection's ``built_to``
(for a per-subject view, its subject's facts only)."""

from __future__ import annotations

from .. import latest
from ..reset import running
from .builder import Builder, load, matches, projections


class Views:
    def __init__(self, project, builder: Builder | None = None, build: bool | None = None):
        self.project = project
        self.builder = builder or Builder(project)
        if build if build is not None else running(project.state_dir) is None:
            self.builder.build()
        else:
            self.builder.log.update()
        self.states = {p.name: load(project, p.name) for p in projections(project)}

    def _behind(self, p, key: str | None) -> int:
        built_to = self.states[p.name].get("built_to", 0)
        return sum(1 for f in self.builder.log.after(built_to) if matches(p, f) and (p.whole or f.get("subject") == key))

    def _with_behind(self, p, key: str, view: dict) -> dict:
        return view | {"behind": self._behind(p, key)}

    def of(self, subject: str) -> dict[str, dict]:
        """``{projection: view}`` for one subject: ``latest``, each per-subject projection's, and
        each whole projection's view keyed by the subject, in config order."""
        out = {}
        for p in projections(self.project):
            v = self.states[p.name]["views"].get(subject)
            if v is not None:
                out[p.name] = self._with_behind(p, subject, v)
        return out

    def all(self, name: str = latest.NAME) -> list[dict]:
        """Every view of one projection, by key."""
        p = next((p for p in projections(self.project) if p.name == name), None)
        if p is None:
            return []
        return [self._with_behind(p, k, v) for k, v in sorted(self.states[name]["views"].items())]

    def latest(self, closed: bool = False, kind: str | None = None) -> list[dict]:
        """Each subject's ``latest`` view, newest first; the done ones only when asked."""
        found = [v for v in self.states[latest.NAME]["views"].values()
                 if (closed or not said(self.of_cached(v["subject"]), "done")) and (not kind or v.get("kind") == kind)]
        return sorted(found, key=lambda v: str(v.get("last") or ""), reverse=True)

    def of_cached(self, subject: str) -> dict[str, dict]:
        """``of`` without ``behind``: for lists, which never show it."""
        return {p.name: v for p in projections(self.project) if (v := self.states[p.name]["views"].get(subject)) is not None}

    def kinds(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for v in self.latest():
            out[v.get("kind") or "?"] = out.get(v.get("kind") or "?", 0) + 1
        return dict(sorted(out.items()))


def said(views: dict[str, dict], key: str):
    """``waiting_on`` or ``done`` for a subject: from its first view, in config order, that has the
    key at its top level, else from ``latest`` (``fields.waiting_on``, ``done``)."""
    for name, v in views.items():
        if name != latest.NAME and key in v:
            return v[key]
    base = views.get(latest.NAME) or {}
    return base.get("done", False) if key == "done" else (base.get("fields") or {}).get(key)
