"""Tasks: what the owner asks for from the Dashboard, and the built-in machine that does them
(SPEC §5). A request is a fact, ``owner.task.requested`` on subject ``task/<id>``, from source
``owner``; a reply on the task's page is ``owner.task.replied`` on the same subject. The built-in
``worker`` (``orchestra agent`` with ``worker.md``) listens for both, unless the config names a
machine ``worker`` of its own."""

from __future__ import annotations

import shlex
import uuid
from pathlib import Path

from ..config.schema import Machine

PROMPT = Path(__file__).parent / "worker.md"
NAME = "worker"
REQUESTED, REPLIED = "owner.task.requested", "owner.task.replied"
TEXT_MAX = 8000
TITLE_MAX = 80


def builtin() -> Machine:
    return Machine(name=NAME, command=f"orchestra agent {shlex.quote(str(PROMPT))}", on=[REQUESTED, REPLIED])


def with_builtin(machines: list[Machine]) -> list[Machine]:
    """The config's machines, and the built-in worker unless one is named ``worker``."""
    return machines if any(m.name == NAME for m in machines) else [*machines, builtin()]


def title(text: str) -> str:
    first = text.strip().splitlines()[0] if text.strip() else ""
    return first if len(first) <= TITLE_MAX else first[: TITLE_MAX - 1] + "…"


def request(text: str, subject: str | None = None) -> tuple[dict | None, str | None]:
    """The fact for a new task (no subject) or a reply on one (``task/…``), or why not."""
    text = text.strip()
    if not text:
        return None, "say what you want done"
    if len(text) > TEXT_MAX:
        return None, f"at most {TEXT_MAX} characters"
    if subject is None:
        sid = f"task/{uuid.uuid4().hex[:8]}"
        return {"type": REQUESTED, "subject": sid, "data": {"text": text, "title": title(text), "waiting_on": NAME}}, None
    if not subject.startswith("task/"):
        return None, "replies go to tasks only"
    return {"type": REPLIED, "subject": subject, "data": {"text": text, "waiting_on": NAME}}, None
