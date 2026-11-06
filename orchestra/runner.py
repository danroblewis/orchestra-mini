"""``orchestra run`` (SPEC §1, §3): one scheduler for every machine the config names.

- The facts followed from where the last run stopped (``<state>/cursor.json``; the end, the first
  time), read through the one read function (``facts.FactLog``: repeats skipped, ``seq``
  stamped): each new fact counted into ``orchestra_events_total`` and handed to every machine
  whose ``on`` names its type (a glob may), except the machine that published it, unless it is
  held (``depth`` past ``facts.MAX_DEPTH``: recorded, said in the log, given to no machine).
- The views: after each batch of facts the builder thread brings them up to the log; routing never
  waits for it. Before a once run for a fact, its subject's per-subject views are brought to at
  least that fact.
- A once machine has one queue and one worker: one run at a time, facts waiting in order. With
  ``every`` it also gets a run with no fact on each tick, unless its last tick's run is still
  waiting or running (then it skips that turn).
- A continuous machine is started at once, restarted when it exits, and gets its facts on stdin
  (``continuous.py``).
- The facts endpoint, and the discovery hook when it is there."""

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
from .endpoint import serve, token
from .events import Events
from .facts import FactLog, held
from .lines import Publisher
from .oneshot import start_run
from .views import Builder, Views

TICK = 0.2


def listens(machine, ev: dict) -> bool:
    return ev.get("source") != machine.name and any(fnmatch.fnmatchcase(str(ev.get("type", "")), p) for p in machine.on)


class Runner:
    def __init__(self, project, stop: threading.Event | None = None, quiet: bool = False):
        self.project, self.cfg = project, project.config
        self.stop = stop or threading.Event()
        self.quiet = quiet
        self.events = Events.for_project(project)
        self.publisher = Publisher.for_project(project, self.events)
        self.store = metrics.for_project(project)
        self.log = FactLog(self.events)
        self.builder = Builder(project, log=self.log)
        self.behind = threading.Event()                    # facts came: the builder thread builds
        once = [m for m in self.cfg.machines if not m.continuous]
        self.queues = {m.name: queue.Queue() for m in once}
        self.ticking: set[str] = set()                     # once machines whose tick's run waits or runs
        self.live = {m.name: Continuous(project, m, self.publisher, self.store, self.stop, self.say)
                     for m in self.cfg.machines if m.continuous}
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
        timed = [m for m in self.cfg.machines if not m.continuous and m.every]
        while not self.stop.is_set():
            t = time.monotonic()
            for m in timed:
                if t >= due.get(m.name, 0):
                    due[m.name] = t + m.every
                    if m.name not in self.ticking:
                        self.ticking.add(m.name)
                        self.queues[m.name].put(None)       # a scheduled run, no fact
            self.stop.wait(TICK)

    def _follow(self) -> None:
        cursor_file = self.project.path("cursor.json")
        end = self.events.end()
        try:
            cursor = min(int(json.loads(cursor_file.read_text())["at"]), end)
        except (FileNotFoundError, ValueError, KeyError, TypeError):
            cursor = end
        self.log.update()
        counted = backfill(self.log.after(0), project=self.project.name, store=self.store)
        counter = EventCollector(self.project.name, counted.counts)
        if cursor < end:
            self._handle([f for f in self.log.after(cursor) if f["seq"] <= end], None)
        cursor = self.log.at
        self.behind.set()                                   # the views brought up to the log at start
        while not self.stop.is_set():
            self.log.update()
            facts = self.log.after(cursor)
            if facts or self.log.at != cursor:
                self._handle(facts, counter)
                cursor = self.log.at
                cursor_file.write_text(json.dumps({"at": cursor}))
            self.stop.wait(TICK)

    def _handle(self, facts: list[dict], counter) -> None:
        if counter is not None:
            self.store.write(collapse([p for f in facts for p in counter.observe(f)]))
        if facts:
            self.behind.set()
        for f in facts:
            routed = [m for m in self.cfg.machines if listens(m, f)]
            if routed and held(f):
                self.say(f"held {f['type']} {f['id']} (depth {f['depth']}): given to no machine")
                continue
            for m in routed:
                if m.continuous:
                    self.live[m.name].deliver(f)
                else:
                    self.queues[m.name].put(f)

    def _build(self) -> None:
        while not self.stop.is_set():
            if self.behind.wait(TICK):
                self.behind.clear()
                try:
                    self.builder.build()
                except Exception as e:                      # said, and tried again on the next batch
                    self.say(f"views: {type(e).__name__}: {e}")

    def _views(self, fact: dict | None) -> dict | None:
        """The fact's subject's views, its per-subject ones built to at least the fact."""
        if not fact or not fact.get("subject"):
            return None
        try:
            self.builder.build(whole=False)
            return Views(self.project, self.builder, build=False).of(fact["subject"]) or None
        except Exception as e:                              # a run with no views is better than none
            self.say(f"views for {fact['subject']}: {type(e).__name__}: {e}")
            return None

    def _work(self, machine) -> None:
        q = self.queues[machine.name]
        while not self.stop.is_set():
            try:
                ev = q.get(timeout=TICK)
            except queue.Empty:
                continue
            try:
                rec = start_run(self.project, machine, ev, self._views(ev), self.publisher, self.store)
            finally:
                if ev is None:
                    self.ticking.discard(machine.name)
            if ev is None and rec.get("exit") == 0 and not rec.get("note") and not rec.get("cost_usd") and not rec.get("held"):
                continue                                    # a quiet collector: its runs say it
            self.say(f"{machine.name}: run {rec['id']} exit {rec.get('exit')}, {rec.get('facts', 0)} facts"
                     + (f", {rec['points']} points" if rec.get("points") else "")
                     + (f", {rec['held']} held" if rec.get("held") else "")
                     + (f", ${rec['cost_usd']:.4f}" if rec.get("cost_usd") else "")
                     + (f" ({rec['note']})" if rec.get("note") else ""))

    def start(self) -> None:
        self.endpoint = serve(self.project, self.publisher, self.stop)
        self.say(f"orchestra {self.project.name}: facts endpoint {self.endpoint}/events")
        if not self.quiet:                                  # the token on the terminal, never in the log
            print(f"  POST one fact as JSON with: Authorization: Bearer {token(self.project)}", flush=True)
        self._thread(self._follow)
        self._thread(self._build)
        self._thread(self._schedule)
        for m in self.cfg.machines:
            if m.continuous:
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
