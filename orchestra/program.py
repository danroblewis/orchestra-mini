"""What the built-in programs (``orchestra agent``, ``orchestra decide``) share: their lines out (a
JSON object line to stdout, any other text to stderr, the machine's log), the events on stdin, and
an event's item's state."""

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


def item_of(event: dict | None) -> dict | None:
    """The event's item's state: ``ORCHESTRA_ITEM`` when it is that item's, else read from the bus."""
    subject = (event or {}).get("subject")
    if not subject:
        return None
    try:
        given = json.loads(os.environ.get("ORCHESTRA_ITEM") or "null")
    except ValueError:
        given = None
    if isinstance(given, dict) and given.get("subject") == subject:
        return given
    try:
        from .events import Events
        from .items import Items
        from .oneshot import item_state
        p = project()
        return item_state(Items(p, Events.for_project(p)).get(subject))
    except Exception:                          # no item is better than no turn
        return None


def events_in() -> list[dict | None]:
    """The events on stdin, until it closes; one None (a scheduled run) when there are none."""
    out = []
    for raw in sys.stdin:
        try:
            ev = json.loads(raw)
        except ValueError:
            continue
        out.append(ev if isinstance(ev, dict) else None)
    return out or [None]
