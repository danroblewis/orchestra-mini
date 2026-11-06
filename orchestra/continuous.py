"""A continuous machine (SPEC §3): started when ``orchestra run`` starts, restarted when it exits,
its stdin kept open. Each event it listens for is written to its stdin as one JSON line as it
comes; events that come while it is restarting wait for it. Its stdout is read like any machine's
(``output``).

Its process's life is one run record. Each run line it prints ends a **turn**: one run record of
its own, for the oldest event written and not yet answered, with the cost and note the line gives
and the events and points since the last turn (``orchestra agent`` prints one per event)."""

from __future__ import annotations

import json
import threading
from collections import deque

from . import command, runs
from .events import now
from .oneshot import environment
from .output import Output

RESTART_S = 5.0


class Continuous:
    def __init__(self, project, machine, publisher, store, stop: threading.Event, say=print):
        self.project, self.machine, self.publisher, self.store, self.stop, self.say = \
            project, machine, publisher, store, stop, say
        self.log = project.path("logs", f"{machine.name}.log")
        self.lock = threading.Lock()
        self.proc = None
        self.out: Output | None = None
        self.waiting: list[dict] = []                 # events that came while it was not running
        self.asked: deque[tuple[dict, str]] = deque()   # events written, waiting for their turn's run line
        self.mark = (0, 0)

    def deliver(self, ev: dict) -> None:
        with self.lock:
            if self.proc is None:
                self.waiting.append(ev)
            else:
                self._write(ev)

    def _write(self, ev: dict) -> None:
        try:
            self.proc.stdin.write(json.dumps(ev) + "\n")
            self.proc.stdin.flush()
            self.asked.append((ev, now()))
        except (OSError, ValueError):
            self.waiting.append(ev)                   # delivered when it is running again

    def _turn(self, cost: float, note, skipped: bool) -> None:
        with self.lock:
            ev, start = self.asked.popleft() if self.asked else ({}, now())
            events, points = self.out.events[self.mark[0]:], self.out.points - self.mark[1]
            self.mark = (len(self.out.events), self.out.points)
            run_id = runs.new_id()
            self.out.salt = run_id
        runs.record(self.project, {
            "id": run_id, "machine": self.machine.name, "continuous": True, "event": ev.get("id"),
            "subject": ev.get("subject"), "start": start, "end": now(), "exit": 0, "events": len(events),
            "types": sorted({e["type"] for e in events}), "points": points, "cost_usd": round(cost, 6),
            "note": str(note) if note else None, "log": str(self.log), **({"skipped": True} if skipped else {})})

    def keep_running(self) -> None:
        m = self.machine
        while not self.stop.is_set():
            run_id = runs.new_id()
            rec = {"id": run_id, "machine": m.name, "continuous": True, "start": now(), "log": str(self.log)}
            try:
                p = command.start(m.command, self.project.repo, self.log, environment(self.project, m, None, None, run_id))
            except (OSError, ValueError) as e:
                runs.record(self.project, rec | {"end": now(), "exit": None, "events": 0, "note": str(e)})
                self.stop.wait(RESTART_S)
                continue
            out = Output(self.project, self.publisher, self.store, m.name, salt=run_id,
                         log=lambda t: command.log_to(self.log, t), prometheus=False, on_run=self._turn)
            with self.lock:
                self.proc, self.out, self.mark = p, out, (0, 0)
                waiting, self.waiting = self.waiting, []
                for ev in waiting:
                    self._write(ev)
            threading.Thread(target=lambda p=p: (self.stop.wait(), command.kill(p)), daemon=True).start()
            for line in p.stdout:
                out.line(line)
            code = p.wait()
            with self.lock:
                self.proc = None
                self.asked.clear()
            runs.record(self.project, rec | {"end": now(), "exit": code, "events": len(out.events), "points": out.points})
            if not self.stop.is_set():
                self.say(f"{m.name}: exited {code}; restarting in {RESTART_S:g}s")
            self.stop.wait(RESTART_S)

    def kill(self) -> None:
        if self.proc is not None:
            command.kill(self.proc)
