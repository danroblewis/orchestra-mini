"""A once machine's run (SPEC §3): its command run once, recorded.

A run for an event gets it as one JSON line on stdin, then stdin closed, and in ``ORCHESTRA_EVENT``;
a scheduled run gets nothing on stdin. Every run gets the item's state (without its history) in
``ORCHESTRA_ITEM``, ``ORCHESTRA_PROJECT``, ``ORCHESTRA_REPO``, its name in ``ORCHESTRA_MACHINE``
and ``ORCHESTRA_SOURCE`` (what ``orchestra emit`` publishes as), ``ORCHESTRA_RUN``, and
``ORCHESTRA_CONTINUOUS=1`` when it is kept running, in the environment Orchestra runs in,
unstripped, with the machine's ``env`` and Orchestra's own ``bin`` first on ``PATH`` (so
``orchestra`` and ``orchestra-prometheus`` are there). Its stdout is read by ``output.Output``
(each run's event lines are new: the run's id salts their default ids); its run lines give the
run's cost and note. A run past ``TIMEOUT`` is stopped."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from . import runs
from .command import log_to, run
from .events import now
from .output import Output

TIMEOUT = 3600.0


def item_state(item: dict | None) -> dict | None:
    return {k: v for k, v in item.items() if k != "events"} | {"events": len(item.get("events", []))} if item else None


def environment(project, machine, event: dict | None, item: dict | None, run_id: str) -> dict:
    env = dict(os.environ) | dict(machine.env)
    env.update(ORCHESTRA_EVENT=json.dumps(event) if event else "", ORCHESTRA_ITEM=json.dumps(item_state(item)) if item else "",
               ORCHESTRA_PROJECT=project.name, ORCHESTRA_REPO=str(project.repo), ORCHESTRA_MACHINE=machine.name,
               ORCHESTRA_SOURCE=machine.name, ORCHESTRA_RUN=run_id, ORCHESTRA_CONTINUOUS="1" if machine.continuous else "")
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    return env


def start_run(project, machine, event: dict | None, item: dict | None, publisher, store) -> dict:
    """Run the machine once for ``event`` (None: a scheduled run); its run record."""
    run_id = runs.new_id()
    log = project.path("logs", f"{machine.name}.log")
    rec = {"id": run_id, "machine": machine.name, "event": (event or {}).get("id"),
           "subject": (event or {}).get("subject"), "start": now(), "log": str(log)}
    log_to(log, f"--- run {run_id} {rec['start']}" + (f" for {event.get('type')} {event.get('id')}" if event else ""))
    out = Output(project, publisher, store, machine.name, salt=run_id, log=lambda t: log_to(log, t))
    res = run(machine.command, project.repo, log, timeout=TIMEOUT, env=environment(project, machine, event, item, run_id),
              stdin=json.dumps(event) + "\n" if event else "", on_line=out.line)
    note = res.error or ("timed out" if res.timed_out else None) or "; ".join(out.notes) or None
    out.finish(res.rc == 0 and not res.error and not res.timed_out)
    rec.update(end=now(), exit=res.rc, events=len(out.events), types=sorted({e["type"] for e in out.events}),
               points=out.points, cost_usd=out.cost, note=note, **({"skipped": True} if out.skipped else {}))
    runs.record(project, rec)
    return rec
