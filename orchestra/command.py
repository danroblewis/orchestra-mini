"""Running a command (SPEC §1): its words split as a shell would but run without one, in the
repository, its stdout read line by line, its stderr appended to its log, killed (its whole
process group) past its timeout."""

from __future__ import annotations

import os
import shlex
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

LOG_MAX = 1_000_000          # a log past this many bytes keeps its last half


@dataclass
class Result:
    rc: int | None = None
    lines: list[str] = field(default_factory=list)
    timed_out: bool = False
    error: str | None = None
    seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return self.rc == 0 and not self.timed_out and not self.error


def log_to(path: Path, text: str) -> None:
    """Append to a log, keeping it under ``LOG_MAX``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if path.stat().st_size > LOG_MAX:
            path.write_bytes(path.read_bytes()[-LOG_MAX // 2:])
    except FileNotFoundError:
        pass
    with path.open("a", encoding="utf-8") as f:
        f.write(text if text.endswith("\n") else text + "\n")


def argv(command: str | list[str]) -> list[str]:
    return list(command) if isinstance(command, list) else shlex.split(command)


def kill(p: subprocess.Popen) -> None:
    try:
        os.killpg(p.pid, signal.SIGTERM)
        p.wait(3)
    except (ProcessLookupError, PermissionError, subprocess.TimeoutExpired):
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


def start(command, cwd: Path, log: Path, env: dict | None = None) -> subprocess.Popen:
    """The command started, stdin and stdout piped, stderr to its log."""
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as err:
        return subprocess.Popen(argv(command), cwd=cwd, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=err, text=True, bufsize=1, start_new_session=True)


def run(command, cwd: Path, log: Path, *, timeout: float | None = None, env: dict | None = None,
        stdin: str | None = None, on_line=None, stop: threading.Event | None = None) -> Result:
    """Run to the end (or the timeout, or ``stop``): each stdout line to ``on_line`` as it comes."""
    r, t0 = Result(), time.monotonic()
    try:
        p = start(command, cwd, log, env)
    except (OSError, ValueError) as e:
        r.error = f"cannot start {command!r}: {e}"
        log_to(log, r.error)
        return r

    def feed():
        try:
            if stdin:
                p.stdin.write(stdin)
            p.stdin.close()
        except (BrokenPipeError, OSError):
            pass
    threading.Thread(target=feed, daemon=True).start()

    def watch():
        while p.poll() is None:
            if timeout is not None and time.monotonic() - t0 > timeout:
                r.timed_out = True
                kill(p)
                return
            if stop is not None and stop.is_set():
                kill(p)
                return
            time.sleep(0.05)
    threading.Thread(target=watch, daemon=True).start()
    for line in p.stdout:
        r.lines.append(line)
        if on_line:
            on_line(line)
    r.rc = p.wait()
    r.seconds = time.monotonic() - t0
    if r.timed_out:
        log_to(log, f"timed out after {timeout:g}s")
    return r
