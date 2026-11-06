"""A store's adapter process (SPEC §4): its ``command``, started on the first request and kept
running, asked one request a line on stdin, its answer read from stdout up to ``{"end": true}``.
An ``{"error": …}`` line makes the answer a ``StoreError``. An adapter that gives no end within
``TIMEOUT`` is killed; one that has exited is started again on the next request. Its stderr goes
to ``<state>/logs/stores/<name>.log``. One process per store per Orchestra process, shared by
its threads; every one is stopped when the process exits."""

from __future__ import annotations

import atexit
import json
import os
import queue
import sys
import threading
import time
from pathlib import Path

from .. import command
from .guard import StoreError

TIMEOUT = 60.0
_lock = threading.Lock()
_running: dict[tuple, Adapter] = {}


class Adapter:
    def __init__(self, project, store):
        self.project, self.store = project, store
        self.log = project.path("logs", "stores", f"{store.name}.log")
        self.lock = threading.Lock()
        self.proc = None
        self.lines: queue.Queue = queue.Queue()
        self.starts = 0

    def _start(self) -> None:
        env = dict(os.environ) | dict(self.store.env)
        env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
        try:
            self.proc = command.start(self.store.command, self.project.repo, self.log, env)
        except (OSError, ValueError) as e:
            raise StoreError(f"cannot start the adapter {self.store.command!r}: {e}") from None
        self.starts += 1
        self.lines = q = queue.Queue()
        out = self.proc.stdout

        def pump():
            for raw in out:
                q.put(raw)
            q.put(None)
        threading.Thread(target=pump, daemon=True).start()

    def stop(self) -> None:
        if self.proc is not None:
            command.kill(self.proc)
            self.proc = None

    def ask(self, req: dict, timeout: float = TIMEOUT) -> list[dict]:
        """The answer's lines (before ``end``); ``StoreError`` for an error line, a dead adapter or no answer."""
        with self.lock:
            if self.proc is None or self.proc.poll() is not None:
                if self.proc is not None:
                    command.log_to(self.log, f"the adapter exited {self.proc.poll()}; starting it again")
                self._start()
            try:
                self.proc.stdin.write(json.dumps(req) + "\n")
                self.proc.stdin.flush()
            except (OSError, ValueError):
                self.stop()
                raise StoreError("the adapter exited; it is started again on the next request") from None
            out, error, deadline = [], None, time.monotonic() + timeout
            while True:
                try:
                    raw = self.lines.get(timeout=max(0.0, deadline - time.monotonic()))
                except queue.Empty:
                    self.stop()
                    raise StoreError(f"the adapter gave no answer in {timeout:g}s; it is started again") from None
                if raw is None:
                    self.stop()
                    raise StoreError("the adapter exited before answering; it is started again on the next request")
                try:
                    got = json.loads(raw)
                except ValueError:
                    got = None
                if not isinstance(got, dict):
                    command.log_to(self.log, raw.rstrip("\n"))
                elif got.get("end") is True:
                    break
                elif "error" in got:
                    error = str(got["error"])
                else:
                    out.append(got)
            if error is not None:
                raise StoreError(error)
            return out


def adapter(project, store) -> Adapter:
    key = (str(project.state_dir), store.name, store.command, tuple(sorted(store.env.items())))
    with _lock:
        if key not in _running:
            _running[key] = Adapter(project, store)
        return _running[key]


@atexit.register
def stop_all() -> None:
    with _lock:
        for a in _running.values():
            a.stop()
        _running.clear()
