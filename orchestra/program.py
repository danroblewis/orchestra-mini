"""What the built-in programs (``orchestra agent``, ``orchestra decide``) share: their lines out (a
JSON object line to stdout, any other text to stderr, the machine's log), the facts on stdin, and
a fact's subject's views and recent facts."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def say(text: str) -> None:
    """Each JSON object line of ``text`` to stdout, the rest to stderr."""
    for part in str(text).splitlines():
        s = part.strip()
        if s:
            print(s, file=sys.stdout if s.startswith("{") else sys.stderr, flush=True)


def line(obj: dict) -> None:
    print(json.dumps(obj), flush=True)


def project():
    from .project import load
    return load(Path(os.environ["ORCHESTRA_REPO"]) if os.environ.get("ORCHESTRA_REPO") else None)


def _given(subject: str) -> dict | None:
    try:
        views = json.loads(Path(os.environ["ORCHESTRA_VIEWS_FILE"]).read_text())
    except (KeyError, OSError, ValueError):
        return None
    base = views.get("latest") if isinstance(views, dict) else None
    return views if isinstance(base, dict) and base.get("subject") == subject else None


def views_of(fact: dict | None) -> tuple[dict, str | None]:
    """``{projection: view}`` for the fact's subject, and the file it came from: the run's
    ``ORCHESTRA_VIEWS_FILE`` when it is that subject's, else read (a continuous machine's)."""
    subject = (fact or {}).get("subject")
    if not subject:
        return {}, None
    if (given := _given(subject)) is not None:
        return given, os.environ["ORCHESTRA_VIEWS_FILE"]
    try:
        from .views import Views
        return Views(project()).of(subject), None
    except Exception:                          # no views are better than no turn
        return {}, None


def recent(subject: str | None, n: int = 10) -> list[dict]:
    """The subject's last ``n`` facts."""
    if not subject:
        return []
    try:
        from .events import Events
        from .facts import read
        return [f for f in read(Events.for_project(project())) if f.get("subject") == subject][-n:]
    except Exception:
        return []


def facts_in() -> list[dict | None]:
    """The facts on stdin, until it closes; one None (a scheduled run) when there are none."""
    out = []
    for raw in sys.stdin:
        try:
            ev = json.loads(raw)
        except ValueError:
            continue
        out.append(ev if isinstance(ev, dict) else None)
    return out or [None]
