"""What ``orchestra run`` checks before it starts (SPEC §5), beyond the config loading: each
machine's and each store's command can start (its program is on ``PATH``, Orchestra's own ``bin``
first, or a file in the repository), each ``orchestra agent``'s prompt file is there and each
``orchestra decide`` names its answers (errors: ``run`` refuses to start); ``claude`` is on
``PATH`` when a built-in program needs it, and each store answers (warnings: said, and ``run``
starts)."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from .command import argv


def _which(project, word: str) -> bool:
    if "/" in word:
        return (project.repo / word).exists()
    path = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
    return shutil.which(word, path=path) is not None


def _command(project, what: str, cmd: str) -> list[tuple[str, str]]:
    try:
        words = argv(cmd)
    except ValueError as e:
        return [("error", f"{what}: the command cannot be split into words ({e})")]
    if not words:
        return [("error", f"{what}: an empty command")]
    if not _which(project, words[0]):
        return [("error", f"{what}: {words[0]} not found" + ("" if "/" in words[0] else " on PATH"))]
    out = []
    if words[0] == "orchestra" and words[1:2] == ["agent"]:
        if len(words) < 3 or not (project.repo / words[2]).is_file():
            out.append(("error", f"{what}: orchestra agent's prompt file {words[2] if len(words) > 2 else '(none)'} not found"))
    if words[0] == "orchestra" and words[1:2] == ["decide"] and not any(w.startswith("--answers") for w in words):
        out.append(("error", f"{what}: orchestra decide needs --answers a,b"))
    if words[0] == "orchestra" and words[1:2] in (["agent"], ["decide"]) and not shutil.which("claude"):
        out.append(("warn", f"{what}: claude (Claude Code) is not on PATH"))
    return out


def problems(project, stores: bool = True) -> list[tuple[str, str]]:
    """(``error`` or ``warn``, what), each machine and store in turn."""
    out = []
    for m in project.config.machines:
        out += _command(project, f"machine {m.name}", m.command)
    for s in project.config.stores:
        found = _command(project, f"store {s.name}", s.command)
        out += found
        if stores and not found:
            from . import stores as reading
            out += [("warn", line) for ok, line in reading.check(project, s) if not ok]
    return out


def failed(r: dict | None) -> bool:
    """Whether a run record says the run went wrong (a held-back run did not)."""
    return r is not None and not r.get("skipped") and (r.get("exit") not in (0, None) or bool(r.get("note")))


def how(m) -> str:
    """When a machine runs, in a few words: ``every 60s``, ``continuous, on a.b``."""
    return ", ".join(filter(None, ["continuous" if m.continuous else "", f"on {', '.join(m.on)}" if m.on else "",
                                   f"every {m.every:g}s" if m.every else ""]))
