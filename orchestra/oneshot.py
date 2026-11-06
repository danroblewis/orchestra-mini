"""A once machine's run (SPEC §3): one run of a program, an agent or a decision, recorded.

A run for an event gets it as one JSON line on stdin, then stdin closed (an agent and a decision
get their prompt there instead), and in ``ORCHESTRA_EVENT``; a scheduled run gets nothing on
stdin. Every run of every machine gets the item's state (without its history) in
``ORCHESTRA_ITEM``, ``ORCHESTRA_PROJECT``, ``ORCHESTRA_REPO``, its name in ``ORCHESTRA_MACHINE``
and ``ORCHESTRA_SOURCE`` (what ``orchestra emit`` and ``orchestra tool`` publish as) and
``ORCHESTRA_RUN``, in the environment Orchestra runs in, unstripped, with Orchestra's own ``bin``
first on ``PATH`` so ``orchestra`` is there. Its stdout is read by ``output.Output`` (each run's
event lines are new: the run's id salts their default ids). An agent or a decision past its
``per_day_usd`` today is not started and its run record says so."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from . import runs
from .agent import run_agent
from .command import log_to, run
from .decision import run_decision
from .events import now
from .output import Output


def item_state(item: dict | None) -> dict | None:
    return {k: v for k, v in item.items() if k != "events"} | {"events": len(item.get("events", []))} if item else None


def environment(project, machine, event: dict | None, item: dict | None, run_id: str) -> dict:
    env = dict(os.environ)
    env.update(ORCHESTRA_EVENT=json.dumps(event) if event else "", ORCHESTRA_ITEM=json.dumps(item_state(item)) if item else "",
               ORCHESTRA_PROJECT=project.name, ORCHESTRA_REPO=str(project.repo), ORCHESTRA_MACHINE=machine.name,
               ORCHESTRA_SOURCE=machine.name, ORCHESTRA_RUN=run_id)
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    return env


def over_budget(project, machine) -> str | None:
    """Why an agent or a decision may not run today, or None."""
    if machine.kind == "program" or (spent := runs.spent_today(project, machine.name)) < machine.per_day_usd:
        return None
    return f"not started: ${spent:.2f} spent today reaches per_day_usd {machine.per_day_usd:g}; until tomorrow"


def start_run(project, machine, event: dict | None, item: dict | None, publisher, store) -> dict:
    """Run the machine once for ``event`` (None: a scheduled run); its run record."""
    run_id = runs.new_id()
    log = project.path("logs", f"{machine.name}.log")
    rec = {"id": run_id, "machine": machine.name, "kind": machine.kind, "event": (event or {}).get("id"),
           "subject": (event or {}).get("subject"), "start": now(), "log": str(log)}
    if why := over_budget(project, machine):
        rec.update(end=now(), exit=None, events=0, skipped=True, note=why)
        runs.record(project, rec)
        return rec
    log_to(log, f"--- run {run_id} {rec['start']}" + (f" for {event.get('type')} {event.get('id')}" if event else ""))
    env = environment(project, machine, event, item, run_id)
    out = Output(project, publisher, store, machine.name, salt=run_id, log=lambda t: log_to(log, t),
                 prometheus=machine.kind == "program")
    if machine.kind == "agent":
        rc, cost, note = run_agent(project, machine, event, item_state(item), env, out, log)
    elif machine.kind == "decision":
        rc, cost, note = run_decision(project, machine, event, item_state(item), env, out, log)
    else:
        res = run(machine.command, project.repo, log, timeout=machine.timeout, env=env,
                  stdin=json.dumps(event) + "\n" if event else "", on_line=out.line)
        rc, cost = res.rc, None
        note = res.error or ("timed out" if res.timed_out else None)
    out.finish(rc == 0 and not note)
    rec.update(end=now(), exit=rc, events=len(out.events), types=sorted({e["type"] for e in out.events}),
               points=out.points, cost_usd=cost, note=note)
    runs.record(project, rec)
    return rec
