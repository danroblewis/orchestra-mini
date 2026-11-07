"""One run of a projection's command (SPEC §6): the facts as JSON lines on stdin, its views as JSON
lines on stdout, under ``TIMEOUT`` (past it its process group is killed). It is given only its
stdin: no network is cut off, but its environment is only ``PATH``, ``HOME`` and the locale, so it
gives the same views when it is run again.

A projection's **hash** says when its views must be rebuilt from the log: its entry (``match``,
``types``, ``whole``, ``command``), the script its command names and, for a Python script, the
modules beside it that it imports, and theirs (the project's own kit: a change to a collector in
the same folder rebuilds nothing), none for an inline command such as ``duckdb -c "…"``, and
Orchestra's version."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path

from .. import __version__
from ..command import argv, log_to, run

TIMEOUT = 60.0
KEPT_ENV = ("HOME", "LANG", "LC_ALL", "LC_CTYPE", "TMPDIR")


IMPORT = re.compile(r"^\s*(?:from|import)\s+([A-Za-z_]\w*)", re.M)


def script_file(project, command: str) -> Path | None:
    """The first word of the command that is a file (``.orchestra/ticket.py``)."""
    try:
        words = argv(command)
    except ValueError:
        return None
    for w in words:
        p = Path(w) if os.path.isabs(w) else project.repo / w
        if ("/" in w or "." in w) and p.is_file():
            return p
    return None


def sources(script: Path) -> list[Path]:
    """The script and, for Python, the modules beside it that it imports, and theirs."""
    found, todo = [], [script]
    while todo:
        f = todo.pop()
        if f in found:
            continue
        found.append(f)
        if f.suffix == ".py" or f == script:
            try:
                text = f.read_text(errors="replace")
            except OSError:
                continue
            todo += [m for name in IMPORT.findall(text) if (m := f.parent / f"{name}.py").is_file()]
    return sorted(found)


def hash_of(project, p) -> str:
    h = hashlib.sha256(json.dumps({"match": p.match, "types": p.types, "whole": p.whole, "command": p.command},
                                  sort_keys=True).encode())
    script = script_file(project, p.command) if p.command else None
    for f in sources(script) if script else []:
        h.update(f.name.encode() + b"\0" + f.read_bytes() + b"\0")
    h.update(__version__.encode())
    return h.hexdigest()[:32]


def environment() -> dict:
    env = {k: os.environ[k] for k in KEPT_ENV if k in os.environ}
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
    return env


def run_projection(project, p, facts: list[dict], timeout: float | None = None) -> tuple[list[dict], str | None]:
    """The JSON objects it printed, and None, or what went wrong (then what it printed is not used)."""
    log = project.path("logs", "projections", f"{p.name}.log")
    stdin = "".join(json.dumps(f) + "\n" for f in facts)
    res = run(p.command, project.repo, log, timeout=TIMEOUT if timeout is None else timeout, env=environment(), stdin=stdin)
    if res.error:
        return [], res.error
    if res.timed_out:
        return [], f"timed out after {TIMEOUT if timeout is None else timeout:g} s"
    if res.rc != 0:
        return [], f"exited {res.rc} (its log: {log})"
    out = []
    for line in res.lines:
        s = line.strip()
        try:
            obj = json.loads(s) if s.startswith("{") else None
        except ValueError:
            obj = None
        if isinstance(obj, dict):
            out.append(obj)
        elif s:
            log_to(log, s)
    return out, None
