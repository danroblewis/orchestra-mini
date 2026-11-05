"""Starting a process (SPEC §3): one run of an agent, a decision or a program, recorded.

A run gets the event as one JSON line on stdin (an agent and a decision get their prompt there
instead) and in ``ORCHESTRA_EVENT``, the item's state (without its history) in
``ORCHESTRA_ITEM``, ``ORCHESTRA_PROJECT``, ``ORCHESTRA_REPO``, its own name in
``ORCHESTRA_PROCESS`` (what ``orchestra emit`` and ``orchestra tool`` publish as) and
``ORCHESTRA_RUN``, in the environment Orchestra runs in, unstripped, with Orchestra's own
``bin`` first on ``PATH`` so ``orchestra`` is there. Its stdout lines are events, published as
the process (each run's lines are new: the run's id salts their default ids). A process past
its ``per_day_usd`` today is not started and its run record says so."""

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


def item_state(item: dict | None) -> dict | None:
    return {k: v for k, v in item.items() if k != "events"} | {"events": len(item.get("events", []))} if item else None


def environment(project, proc, event: dict | None, item: dict | None, run_id: str) -> dict:
    env = dict(os.environ)
    env.update(ORCHESTRA_EVENT=json.dumps(event) if event else "", ORCHESTRA_ITEM=json.dumps(item_state(item)) if item else "",
               ORCHESTRA_PROJECT=project.name, ORCHESTRA_REPO=str(project.repo), ORCHESTRA_PROCESS=proc.name,
               ORCHESTRA_RUN=run_id)
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    return env


def start_run(project, proc, event: dict | None, item: dict | None, publisher) -> dict:
    """Run the process once for ``event`` (None: a scheduled run); its run record."""
    run_id = runs.new_id()
    log = project.path("logs", proc.name, f"{run_id}.log")
    rec = {"id": run_id, "process": proc.name, "kind": proc.kind, "event": (event or {}).get("id"),
           "subject": (event or {}).get("subject"), "start": now(), "log": str(log)}
    if proc.kind in ("agent", "decision") and (spent := runs.spent_today(project, proc.name)) >= proc.per_day_usd:
        rec.update(end=now(), exit=None, events=0, skipped=True,
                   note=f"not started: ${spent:.2f} spent today reaches per_day_usd {proc.per_day_usd:g}; until tomorrow")
        runs.record(project, rec)
        return rec
    env = environment(project, proc, event, item, run_id)
    published: list[dict] = []

    def publish_lines(lines):
        published.extend(publisher.lines(lines, proc.name, salt=run_id, log=lambda t: log_to(log, t)))

    def publish_event(ev):
        if publisher.publish(ev):
            published.append(ev)

    if proc.kind == "agent":
        rc, cost, note = run_agent(project, proc, event, item_state(item), env, publish_lines, log)
    elif proc.kind == "decision":
        rc, cost, note = run_decision(project, proc, event, item_state(item), env, publish_event, log)
    else:
        res = run(proc.command, project.repo, log, timeout=proc.timeout, env=env,
                  stdin=json.dumps(event) + "\n" if event else "", on_line=lambda line: publish_lines([line]))
        rc, cost = res.rc, None
        note = res.error or ("timed out" if res.timed_out else None)
    rec.update(end=now(), exit=rc, events=len(published), types=sorted({e["type"] for e in published}),
               cost_usd=cost, note=note)
    runs.record(project, rec)
    return rec
