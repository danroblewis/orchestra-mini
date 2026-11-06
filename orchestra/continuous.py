"""A continuous machine (SPEC §3): started when ``orchestra run`` starts, restarted when it exits,
its stdin kept open. Each event it listens for is written to its stdin as it comes: a program gets
the event as one JSON line; an agent (``claude -p --input-format stream-json``) one user message
with the event and the item's state. Events that come while it is restarting wait for it. Its
stdout is read like any machine's (``output``).

A program's run record covers its process's life. An agent's covers one turn, from the event
written to its result, with the turn's cost (the change in the session's running total). An agent
past its ``per_day_usd`` today is given no events until tomorrow; each one held back is recorded."""

from __future__ import annotations

import json
import threading
from collections import deque

from . import agent, command, runs
from .events import now
from .oneshot import environment, item_state, over_budget
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
        self.waiting: list[tuple[dict, dict | None]] = []      # events that came while it was not running
        self.turns: deque[dict] = deque()                      # an agent's turns waiting for their result
        self.total, self.mark = 0.0, (0, 0)

    def deliver(self, ev: dict, item: dict | None) -> None:
        with self.lock:
            if self.proc is None:
                self.waiting.append((ev, item))
            else:
                self._write(ev, item)

    def _record(self, ev: dict, **rec) -> dict:
        return {"id": runs.new_id(), "machine": self.machine.name, "kind": self.machine.kind, "mode": "continuous",
                "event": ev.get("id"), "subject": ev.get("subject"), "start": now(), "log": str(self.log), **rec}

    def _write(self, ev: dict, item: dict | None) -> None:
        m = self.machine
        if m.kind == "agent":
            if why := over_budget(self.project, m):
                runs.record(self.project, self._record(ev, end=now(), exit=None, events=0, skipped=True, note=why))
                return
            self.turns.append(self._record(ev))
            self.out.salt = self.turns[0]["id"]
            line = {"type": "user", "message": {"role": "user", "content": agent.about(ev, item_state(item))}}
        else:
            line = ev
        try:
            self.proc.stdin.write(json.dumps(line) + "\n")
            self.proc.stdin.flush()
        except (OSError, ValueError):
            if m.kind == "agent":
                self.turns.pop()
            self.waiting.append((ev, item))               # delivered when it is running again

    def _turn_done(self, result: dict) -> None:
        with self.lock:
            if not self.turns:
                return
            turn = self.turns.popleft()
            total = agent.cost_of(result) or self.total
            events, points = self.out.events[self.mark[0]:], self.out.points - self.mark[1]
            turn.update(end=now(), exit=1 if result.get("is_error") else 0, events=len(events),
                        types=sorted({e["type"] for e in events}), points=points,
                        cost_usd=round(max(0.0, total - self.total), 6), note=agent.error_of(result))
            self.total, self.mark = total, (len(self.out.events), self.out.points)
            if self.turns:
                self.out.salt = self.turns[0]["id"]
        runs.record(self.project, turn)

    def _start(self, run_id: str):
        m = self.machine
        argv = agent.args(self.project, m, continuous=True) if m.kind == "agent" else m.command
        return command.start(argv, self.project.repo, self.log, environment(self.project, m, None, None, run_id))

    def keep_running(self) -> None:
        m = self.machine
        while not self.stop.is_set():
            run_id = runs.new_id()
            rec = {"id": run_id, "machine": m.name, "kind": m.kind, "mode": "continuous", "start": now(), "log": str(self.log)}
            try:
                p = self._start(run_id)
            except OSError as e:
                runs.record(self.project, rec | {"end": now(), "exit": None, "events": 0, "note": str(e)})
                self.stop.wait(RESTART_S)
                continue
            out = Output(self.project, self.publisher, self.store, m.name, salt=run_id,
                         log=lambda t: command.log_to(self.log, t), prometheus=False)
            with self.lock:
                self.proc, self.out, self.total, self.mark = p, out, 0.0, (0, 0)
                waiting, self.waiting = self.waiting, []
                for ev, item in waiting:
                    self._write(ev, item)
            threading.Thread(target=lambda p=p: (self.stop.wait(), command.kill(p)), daemon=True).start()
            turn: dict = {}
            for line in p.stdout:
                if m.kind != "agent":
                    out.line(line)
                elif (result := agent.read(line, out, turn)) is not None:
                    self._turn_done(result)
            code = p.wait()
            with self.lock:
                self.proc = None
                lost, self.turns = list(self.turns), deque()
            for turn in lost:
                runs.record(self.project, turn | {"end": now(), "exit": code, "events": 0,
                                                  "note": "the agent exited before answering"})
            if m.kind != "agent":
                runs.record(self.project, rec | {"end": now(), "exit": code, "events": len(out.events), "points": out.points})
            if not self.stop.is_set():
                self.say(f"{m.name}: exited {code}; restarting in {RESTART_S:g}s")
            self.stop.wait(RESTART_S)

    def kill(self) -> None:
        if self.proc is not None:
            command.kill(self.proc)
