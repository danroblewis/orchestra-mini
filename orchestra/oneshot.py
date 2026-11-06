"""A once machine's run (SPEC §3): its command run once, recorded.

A run for a fact gets it as one JSON line on stdin, then stdin closed, and in ``ORCHESTRA_FACT``
(and ``ORCHESTRA_EVENT``, for one release); a scheduled run gets nothing on stdin. A run for a
fact about a subject gets ``ORCHESTRA_VIEWS_FILE``, ``{projection: view}`` for that subject, kept
as ``runs/<id>.views.json`` (and ``ORCHESTRA_ITEM``, ``latest`` as it was, for one release). Its
cause is in ``ORCHESTRA_CAUSEDBY`` and ``ORCHESTRA_DEPTH`` (what ``orchestra emit`` stamps). Every
run gets ``ORCHESTRA_PROJECT``, ``ORCHESTRA_REPO``, its name in ``ORCHESTRA_MACHINE`` and
``ORCHESTRA_SOURCE`` (what ``orchestra emit`` publishes as), ``ORCHESTRA_RUN``, and
``ORCHESTRA_CONTINUOUS=1`` when it is kept running, in the environment Orchestra runs in,
unstripped, with the machine's ``env`` and Orchestra's own ``bin`` first on ``PATH`` (so
``orchestra`` and ``orchestra-prometheus`` are there). Its stdout is read by ``output.Output``
(each run's fact lines are new: the run's id salts their default ids, and each is stamped with
the run's fact as its cause); its run lines give the run's cost and note. A run past ``TIMEOUT``
is stopped."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from . import runs
from .command import log_to, run
from .events import now
from .output import Output
from .views import seen

TIMEOUT = 3600.0


def depth(fact: dict | None) -> int:
    d = (fact or {}).get("depth")
    return d if isinstance(d, int) and not isinstance(d, bool) else 0


def environment(project, machine, fact: dict | None, views: dict | None, run_id: str, views_file: str = "") -> dict:
    env = dict(os.environ) | dict(machine.env)
    told = json.dumps(fact) if fact else ""
    env.update(ORCHESTRA_FACT=told, ORCHESTRA_EVENT=told, ORCHESTRA_ITEM=seen.legacy_item(views), ORCHESTRA_VIEWS_FILE=views_file,
               ORCHESTRA_CAUSEDBY=str((fact or {}).get("id") or ""), ORCHESTRA_DEPTH=str(depth(fact) + 1) if fact else "0",
               ORCHESTRA_PROJECT=project.name, ORCHESTRA_REPO=str(project.repo), ORCHESTRA_MACHINE=machine.name,
               ORCHESTRA_SOURCE=machine.name, ORCHESTRA_RUN=run_id, ORCHESTRA_CONTINUOUS="1" if machine.continuous else "")
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    return env


def start_run(project, machine, fact: dict | None, views: dict | None, publisher, store) -> dict:
    """Run the machine once for ``fact`` (None: a scheduled run) with its subject's ``views``; its
    run record."""
    run_id = runs.new_id()
    log = project.path("logs", f"{machine.name}.log")
    rec = {"id": run_id, "machine": machine.name, "fact": (fact or {}).get("id"),
           "subject": (fact or {}).get("subject"), "start": now(), "log": str(log)}
    views_file = seen.write(project, run_id, views) if views else ""
    if views_file:
        rec["views"] = views_file
    log_to(log, f"--- run {run_id} {rec['start']}" + (f" for {fact.get('type')} {fact.get('id')}" if fact else ""))
    out = Output(project, publisher, store, machine.name, salt=run_id, log=lambda t: log_to(log, t), cause=lambda: fact)
    res = run(machine.command, project.repo, log, timeout=TIMEOUT, env=environment(project, machine, fact, views, run_id, views_file),
              stdin=json.dumps(fact) + "\n" if fact else "", on_line=out.line)
    note = res.error or ("timed out" if res.timed_out else None) or "; ".join(out.notes) or None
    out.finish(res.rc == 0 and not res.error and not res.timed_out)
    rec.update(end=now(), exit=res.rc, **counts(out), cost_usd=out.cost, note=note, **({"skipped": True} if out.skipped else {}))
    runs.record(project, rec)
    return rec


def counts(out: Output, since: tuple[int, ...] = (0, 0, 0, 0, 0)) -> dict:
    """What a run (or a turn, from ``since``) published: facts, their types, points, and the
    repeats, restated and held facts when there were any."""
    facts = out.facts[since[0]:]
    rec = {"facts": len(facts), "types": sorted({f["type"] for f in facts}), "points": out.points - since[1]}
    for k, n in (("repeats", out.repeats - since[2]), ("restated", out.restated - since[3]), ("held", out.held - since[4])):
        if n:
            rec[k] = n
    return rec


def mark(out: Output) -> tuple[int, ...]:
    return len(out.facts), out.points, out.repeats, out.restated, out.held
