"""Every request Orchestra sends to a store, logged: ``<state>/stores/<name>.requests.jsonl``, one
line per request sent (a retried one included) and per read refused before sending (the guard, the
daily cap): ``t``, ``endpoint``, ``query``, ``window``, ``step``, ``series``, ``points``, ``ms``,
``sent`` (false for a refusal), ``error``. Never a header, so never a credential. The log keeps the
newest ``KEEP`` lines. A write that fails is dropped: the log never stops a read. Copied from the
old Orchestra's ``sources/reqlog.py``."""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

KEEP = 10_000
SLACK = KEEP // 10
_LOCK = threading.Lock()
_COUNTS: dict[str, int] = {}


def path(project, name: str) -> Path:
    return project.state_dir / "stores" / f"{name}.requests.jsonl"


def _lines(p: Path) -> int:
    try:
        with p.open("rb") as f:
            return sum(1 for _ in f)
    except OSError:
        return 0


def append(project, name: str, entry: dict) -> None:
    p = path(project, name)
    line = json.dumps({"t": round(time.time(), 3), **entry}, ensure_ascii=False, default=str) + "\n"
    try:
        with _LOCK:
            p.parent.mkdir(parents=True, exist_ok=True)
            key = str(p)
            _COUNTS.setdefault(key, _lines(p))
            with p.open("a", encoding="utf-8") as f:
                f.write(line)
            _COUNTS[key] += 1
            if _COUNTS[key] > KEEP + SLACK:
                keep = p.read_bytes().splitlines(keepends=True)[-KEEP:]
                tmp = p.with_name(p.name + f".{os.getpid()}.tmp")
                tmp.write_bytes(b"".join(keep))
                tmp.replace(p)
                _COUNTS[key] = len(keep)
    except OSError:
        pass


def last(project, name: str, n: int = 20) -> list[dict]:
    """The newest ``n`` lines, oldest first."""
    try:
        raw = path(project, name).read_text(encoding="utf-8", errors="replace").splitlines()[-n:]
    except OSError:
        return []
    out = []
    for line in raw:
        try:
            out.append(json.loads(line))
        except ValueError:
            pass
    return out


def answer_size(data) -> tuple[int | None, int | None]:
    """(series, points) of an answer."""
    if isinstance(data, dict) and isinstance(data.get("result"), list):
        res = data["result"]
        return len(res), sum(len(r.get("values") or []) + ("value" in r) for r in res if isinstance(r, dict))
    if isinstance(data, list):
        return len(data), None
    return None, None


def asked_about(form: dict | None) -> tuple:
    """(query, window, step) of a request's form."""
    if not form:
        return None, None, None
    def num(v):
        return float(v) if v is not None else None
    q = form.get("query") or form.get("match[]")
    win = [num(form["start"]), num(form.get("end"))] if "start" in form else [num(form["time"])] if "time" in form else None
    return q, win, num(form.get("step"))
