"""Run records (SPEC §3): one JSON line per machine run in ``<state>/runs/<machine>.jsonl``: its
id, what started it, start, end, exit code, events published and points written, cost when the
harness reports it, its log, and a note (why it was not started, why it failed). A file past
``KEEP`` bytes keeps its newer half, so a collector run every minute never grows it for good."""

from __future__ import annotations

import json
import threading
import uuid
from datetime import UTC, datetime

from .events import now

_lock = threading.Lock()
KEEP = 2_000_000


def new_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]


def record(project, rec: dict) -> None:
    path = project.path("runs", f"{rec['machine']}.jsonl")
    with _lock:
        if path.exists() and path.stat().st_size > KEEP:
            kept = path.read_text(encoding="utf-8").splitlines(keepends=True)
            path.write_text("".join(kept[len(kept) // 2:]), encoding="utf-8")
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")


def read(project, machine: str | None = None, n: int = 50) -> list[dict]:
    """The latest ``n`` runs, of one machine or all, newest first."""
    folder = project.state_dir / "runs"
    files = [folder / f"{machine}.jsonl"] if machine else sorted(folder.glob("*.jsonl"))
    out = []
    for f in files:
        try:
            for line in f.read_text(encoding="utf-8").splitlines():
                try:
                    out.append(json.loads(line))
                except ValueError:
                    pass
        except FileNotFoundError:
            pass
    out.sort(key=lambda r: r.get("start") or "", reverse=True)
    return out[:n]


def spent_today(project, machine: str) -> float:
    """What the machine's runs cost since midnight UTC."""
    day = now()[:10]
    return sum(float(r.get("cost_usd") or 0) for r in read(project, machine, 10_000)
               if str(r.get("start", "")).startswith(day))

