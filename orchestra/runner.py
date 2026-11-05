"""``orchestra run`` (SPEC §1, §3): one scheduler for everything the config names.

- ``events:`` commands on their ``every`` (or kept running: ``continuous``), their event lines
  published; ``metrics:`` commands on their ``every``, their points written. A command whose last
  round is still running skips its turn. Each one's last round is kept in ``<state>/health.json``.
- The bus followed from where the last run stopped (``<state>/cursor.json``; the end, the first
  time): each new event counted into ``orchestra_events_total``, folded into its item, and handed
  to every process whose ``on`` names its type (a glob may), except the process that published it.
- Each process has one queue and one worker: one run at a time, events waiting in order. A
  scheduled process gets a run with no event every ``every``. A continuous program is started at
  once, restarted when it exits, and gets its events on stdin, one JSON line each.
- The events endpoint, and the discovery hook when it is there."""

from __future__ import annotations

import fnmatch
import json
import queue
import threading
import time
from datetime import datetime

from . import command, hooks, metrics
from .collect import collect
from .counts import EventCollector, backfill, collapse
from .endpoint import serve
from .events import Events, now
from .items import Items
from .lines import Publisher
from .processes import environment, start_run

TICK = 0.2
RESTART_S = 5.0


def listens(proc, ev: dict) -> bool:
    return ev.get("source") != proc.name and any(fnmatch.fnmatchcase(str(ev.get("type", "")), p) for p in proc.on)


class Runner:
    def __init__(self, project, stop: threading.Event | None = None, quiet: bool = False):
        self.project, self.cfg = project, project.config
        self.stop = stop or threading.Event()
        self.quiet = quiet
        self.events = Events.for_project(project)
        self.publisher = Publisher(self.events)
        self.store = metrics.for_project(project)
        self.items = Items(project, self.events)
        self.queues = {p.name: queue.Queue() for p in self.cfg.processes if not p.continuous}
        self.live: dict[str, object] = {}                  # continuous process name -> Popen
        self.health: dict[str, dict] = {}
        self._busy: set[str] = set()
        self._lock = threading.Lock()
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

    def _healthy(self, name: str, **state) -> None:
        with self._lock:
            self.health[name] = {"at": now(), **state}
            tmp = self.project.path("health.json.tmp")
            tmp.write_text(json.dumps(self.health))
            tmp.replace(self.project.path("health.json"))

    def _events_round(self, c) -> None:
        log = self.project.path("logs", f"{c.name}.log")
        got: list[dict] = []
        res = command.run(c.command, self.project.repo, log, timeout=None if c.continuous else c.timeout,
                          stop=self.stop, on_line=lambda line: got.extend(
                              self.publisher.lines([line], c.name, log=lambda t: command.log_to(log, t))))
        error = res.error or ("timed out" if res.timed_out else None if res.rc == 0 else f"exit code {res.rc}")
        self._healthy(c.name, kind="events", ok=error is None, error=error, events=len(got))

    def _metrics_round(self, c) -> None:
        r = collect(c, self.project, self.store, stop=self.stop)
        self._healthy(c.name, kind="metrics", ok=r.error is None, error=r.error, points=len(r.points))

    def _once(self, name: str, fn, c) -> None:
        try:
            fn(c)
        finally:
            self._busy.discard(name)

    def _continuous(self, c) -> None:
        while not self.stop.is_set():
            self._events_round(c)
            self.stop.wait(RESTART_S)

    def _schedule(self) -> None:
        due: dict[str, float] = {}
        jobs = [(c.name, c.every, self._events_round, c) for c in self.cfg.events if not c.continuous]
        jobs += [(c.name, c.every, self._metrics_round, c) for c in self.cfg.metrics]
        jobs += [(p.name, p.every, None, p) for p in self.cfg.processes if p.every and not p.continuous]
        while not self.stop.is_set():
            t = time.monotonic()
            for name, every, fn, c in jobs:
                if t < due.get(name, 0):
                    continue
                due[name] = t + every
                if fn is None:
                    self.queues[name].put(None)             # a scheduled run, no event
                elif name not in self._busy:
                    self._busy.add(name)
                    self._thread(self._once, name, fn, c)
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
            for p in self.cfg.processes:
                if listens(p, ev):
                    self._dispatch(p, ev)

    def _dispatch(self, proc, ev: dict) -> None:
        if not proc.continuous:
            self.queues[proc.name].put(ev)
            return
        p = self.live.get(proc.name)
        try:
            p.stdin.write(json.dumps(ev) + "\n")
            p.stdin.flush()
        except (AttributeError, OSError, ValueError):
            self.say(f"{proc.name}: not running; {ev.get('type')} not delivered")

    def _work(self, proc) -> None:
        q = self.queues[proc.name]
        while not self.stop.is_set():
            try:
                ev = q.get(timeout=TICK)
            except queue.Empty:
                continue
            item = self.items.items.get(ev.get("subject")) if ev and ev.get("subject") else None
            rec = start_run(self.project, proc, ev, item, self.publisher)
            self.say(f"{proc.name}: run {rec['id']} exit {rec.get('exit')}, {rec.get('events', 0)} events"
                     + (f", ${rec['cost_usd']:.4f}" if rec.get("cost_usd") else "")
                     + (f" ({rec['note']})" if rec.get("note") else ""))

    def _keep_running(self, proc) -> None:
        from . import runs
        while not self.stop.is_set():
            run_id = runs.new_id()
            log = self.project.path("logs", proc.name, f"{run_id}.log")
            rec = {"id": run_id, "process": proc.name, "kind": proc.kind, "start": now(), "log": str(log)}
            try:
                p = command.start(proc.command, self.project.repo, log,
                                  environment(self.project, proc, None, None, run_id))
            except OSError as e:
                rec.update(end=now(), exit=None, events=0, note=str(e))
                runs.record(self.project, rec)
                self.stop.wait(RESTART_S)
                continue
            self.live[proc.name] = p
            threading.Thread(target=lambda p=p: (self.stop.wait(), command.kill(p)), daemon=True).start()
            n = 0
            for line in p.stdout:
                n += len(self.publisher.lines([line], proc.name, salt=run_id, log=lambda t, log=log: command.log_to(log, t)))
            rec.update(end=now(), exit=p.wait(), events=n)
            runs.record(self.project, rec)
            self.stop.wait(RESTART_S)

    def start(self) -> None:
        self.endpoint = serve(self.project, self.publisher, self.stop)
        self.say(f"orchestra {self.project.name}: events endpoint {self.endpoint}")
        self._thread(self._follow)
        self._thread(self._schedule)
        for c in self.cfg.events:
            if c.continuous:
                self._thread(self._continuous, c)
        for p in self.cfg.processes:
            self._thread(self._keep_running if p.continuous else self._work, p)
        if (d := hooks.hook("discovery", "start")) is not None:
            d.start(self.project, self.stop)

    def run(self, seconds: float | None = None) -> None:
        self.start()
        try:
            self.stop.wait(seconds)
        except KeyboardInterrupt:
            pass
        self.stop.set()
        for p in self.live.values():
            command.kill(p)
        for t in self.threads:
            t.join(2)
        (self.project.state_dir / "endpoint.json").unlink(missing_ok=True)   # not left to the endpoint's daemon thread
