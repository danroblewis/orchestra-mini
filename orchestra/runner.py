"""``orchestra run`` (SPEC §1, §3): one scheduler for every machine the config names.

- The bus followed from where the last run stopped (``<state>/cursor.json``; the end, the first
  time): each new event counted into ``orchestra_events_total``, folded into its item, and handed
  to every machine whose ``on`` names its type (a glob may), except the machine that published it.
- A once machine has one queue and one worker: one run at a time, events waiting in order. With
  ``every`` it also gets a run with no event on each tick, unless its last tick's run is still
  waiting or running (then it skips that turn).
- A continuous machine is started at once, restarted when it exits, and gets its events on stdin
  (``continuous.py``).
- The events endpoint, and the discovery hook when it is there."""

from __future__ import annotations

import fnmatch
import json
import queue
import threading
import time
from datetime import datetime

from . import command, hooks, metrics
from .continuous import Continuous
from .counts import EventCollector, backfill, collapse
from .endpoint import serve
from .events import Events
from .items import Items
from .lines import Publisher
from .oneshot import start_run

TICK = 0.2


def listens(machine, ev: dict) -> bool:
    return ev.get("source") != machine.name and any(fnmatch.fnmatchcase(str(ev.get("type", "")), p) for p in machine.on)


class Runner:
    def __init__(self, project, stop: threading.Event | None = None, quiet: bool = False):
        self.project, self.cfg = project, project.config
        self.stop = stop or threading.Event()
        self.quiet = quiet
        self.events = Events.for_project(project)
        self.publisher = Publisher(self.events)
        self.store = metrics.for_project(project)
        self.items = Items(project, self.events)
        once = [m for m in self.cfg.machines if m.mode == "once"]
        self.queues = {m.name: queue.Queue() for m in once}
        self.ticking: set[str] = set()                     # once machines whose tick's run waits or runs
        self.live = {m.name: Continuous(project, m, self.publisher, self.store, self.stop, self.say)
                     for m in self.cfg.machines if m.mode == "continuous"}
        self.threads: list[threading.Thread] = []
        self.endpoint: str | None = None

    def say(self, text: str) -> None:
        line = f"{datetime.now():%H:%M:%S} {text}"
        command.log_to(self.project.path("logs", "orchestra.log"), line)
        if not self.quiet:
            print(line, flush=True)

    def _thread(self, target, *args) -> None:
        t = threading.Thread(target=self._guard, args=(target, *args), daemon=True)
        t.start()
        self.threads.append(t)

    def _guard(self, target, *args) -> None:
        try:
            target(*args)
        except Exception as e:                              # one part failing never stops the rest
            self.say(f"{getattr(target, '__name__', target)}: {type(e).__name__}: {e}")

    def _schedule(self) -> None:
        due: dict[str, float] = {}
        timed = [m for m in self.cfg.machines if m.mode == "once" and m.every]
        while not self.stop.is_set():
            t = time.monotonic()
            for m in timed:
                if t >= due.get(m.name, 0):
                    due[m.name] = t + m.every
                    if m.name not in self.ticking:
                        self.ticking.add(m.name)
                        self.queues[m.name].put(None)       # a scheduled run, no event
            self.stop.wait(TICK)

    def _follow(self) -> None:
        cursor_file = self.project.path("cursor.json")
        end = self.events.end()
        try:
            cursor = min(int(json.loads(cursor_file.read_text())["at"]), end)
        except (FileNotFoundError, ValueError, KeyError, TypeError):
            cursor = end
        counted = backfill(list(self.events.all(0)) if end else [], project=self.project.name, store=self.store)
        counter = EventCollector(self.project.name, counted.counts)
        if cursor < end:
            evs, _ = self.events.page(cursor, end - cursor)
            self._handle(evs, None)
            cursor = end
        while not self.stop.is_set():
            evs, nxt = self.events.after(cursor)
            if evs:
                self._handle(evs, counter)
                cursor = nxt
                cursor_file.write_text(json.dumps({"at": cursor}))
            self.stop.wait(TICK)

    def _handle(self, evs: list[dict], counter) -> None:
        if counter is not None:
            self.store.write(collapse([p for ev in evs for p in counter.observe(ev)]))
        self.items.update()
        for ev in evs:
            for m in self.cfg.machines:
                if listens(m, ev):
                    if m.mode == "once":
                        self.queues[m.name].put(ev)
                    else:
                        self.live[m.name].deliver(ev, self._item(ev))

    def _item(self, ev: dict | None) -> dict | None:
        return self.items.items.get(ev.get("subject")) if ev and ev.get("subject") else None

    def _work(self, machine) -> None:
        q = self.queues[machine.name]
        while not self.stop.is_set():
            try:
                ev = q.get(timeout=TICK)
            except queue.Empty:
                continue
            try:
                rec = start_run(self.project, machine, ev, self._item(ev), self.publisher, self.store)
            finally:
                if ev is None:
                    self.ticking.discard(machine.name)
            if ev is None and machine.kind == "program" and rec.get("exit") == 0 and not rec.get("note"):
                continue                                    # a quiet collector: its runs say it
            self.say(f"{machine.name}: run {rec['id']} exit {rec.get('exit')}, {rec.get('events', 0)} events"
                     + (f", {rec['points']} points" if rec.get("points") else "")
                     + (f", ${rec['cost_usd']:.4f}" if rec.get("cost_usd") else "")
                     + (f" ({rec['note']})" if rec.get("note") else ""))

    def start(self) -> None:
        self.endpoint = serve(self.project, self.publisher, self.stop)
        self.say(f"orchestra {self.project.name}: events endpoint {self.endpoint}")
        self._thread(self._follow)
        self._thread(self._schedule)
        for m in self.cfg.machines:
            if m.mode == "continuous":
                self._thread(self.live[m.name].keep_running)
            else:
                self._thread(self._work, m)
        if (d := hooks.hook("discovery", "start")) is not None:
            d.start(self.project, self.stop)

    def run(self, seconds: float | None = None) -> None:
        self.start()
        try:
            self.stop.wait(seconds)
        except KeyboardInterrupt:
            pass
        self.stop.set()
        for c in self.live.values():
            c.kill()
        for t in self.threads:
            t.join(2)
        (self.project.state_dir / "endpoint.json").unlink(missing_ok=True)   # not left to the endpoint's daemon thread
