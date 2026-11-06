"""``orchestra agent <prompt file> [--model M] [--per-day-usd N]`` (SPEC §3): a built-in program a
machine names in its ``command``. Like any machine it reads events on stdin and writes event,
metric and run lines on stdout.

It runs Claude Code, ``claude -p --output-format stream-json``, with the owner's environment
unstripped and ``--dangerously-skip-permissions``: it acts with its own shell and the owner's
credentials. Its instructions are the prompt file and how to say what happened; each event comes
with its item's state.

- **once**: one turn per event on stdin (one turn with no event for a scheduled run), the
  instructions and the event given to ``claude`` on its stdin.
- **continuous** (``ORCHESTRA_CONTINUOUS=1``): one session, ``--input-format stream-json`` with the
  instructions appended to the system prompt; each event one user message, one turn at a time.

The JSON lines in what Claude says (its assistant text; the result's, which repeats it, only when a
turn had none) or prints outside the stream go to stdout; any other text to stderr (the log). Each
turn ends with a run line, ``{"cost_usd": …}``: the turn's cost from the stream's
``total_cost_usd`` (a session's running total), and a ``note`` when it failed. Past
``--per-day-usd`` spent today (its machine's run records, and this session's turns) a turn is not
started: its run line says so, with ``"skipped": true``."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

from .program import events_in, item_of, line, project, say

STREAM = {"system", "assistant", "user", "result", "stream_event", "rate_limit_event"}
EMITTING = """## Saying what happened

Say what you did as events: print one JSON object per line in your answer, like
{"type": "<system>.<noun>.<verb>", "subject": "<the item>", "data": {...}}
and measurements as {"metric": "<what it measures>", "value": 3, "labels": {...}},
or run `orchestra emit '<json>'` in your shell."""


class Budget:
    """What its machine spent today, from its run records at start, plus this process's turns."""

    def __init__(self, cap: float | None):
        self.cap, self.extra, self.day, self.base = cap, 0.0, None, 0.0

    def why_not(self) -> str | None:
        if self.cap is None:
            return None
        from . import runs
        from .events import now
        if self.day != now()[:10]:
            self.day, self.extra = now()[:10], 0.0
            try:
                self.base = runs.spent_today(project(), os.environ.get("ORCHESTRA_MACHINE", ""))
            except Exception:
                self.base = 0.0
        spent = self.base + self.extra
        return None if spent < self.cap else \
            f"not started: ${spent:.2f} spent today reaches per_day_usd {self.cap:g}; until tomorrow"

    def spend(self, cost: float) -> None:
        self.extra += cost


def instructions(prompt: Path) -> str:
    return prompt.read_text(encoding="utf-8").rstrip() + "\n\n" + EMITTING + "\n"


def about(event: dict | None, item: dict | None) -> str:
    parts = ["## The event\n\n" + (json.dumps(event, indent=2) if event else "(none: a scheduled run)")]
    if item:
        parts.append("## The item's state\n\n" + json.dumps(item, indent=2))
    return "\n\n".join(parts) + "\n"


def texts(msg: dict) -> list[str]:
    """The text an assistant message or a result carries."""
    if msg.get("type") == "result":
        return [msg["result"]] if isinstance(msg.get("result"), str) else []
    content = (msg.get("message") or {}).get("content") if msg.get("type") == "assistant" else None
    return [p.get("text", "") for p in content or [] if isinstance(p, dict) and p.get("type") == "text"]


def read(raw: str, turn: dict) -> dict | None:
    """One line of the stream: what it says, relayed; the message when it is a result. ``turn``
    remembers whether this turn said anything yet."""
    try:
        msg = json.loads(raw)
    except ValueError:
        msg = None
    if not isinstance(msg, dict) or msg.get("type") not in STREAM:
        say(raw)
        return None
    result = msg.get("type") == "result"
    if not (result and turn.get("said")):
        for t in texts(msg):
            turn["said"] = True
            say(t)
    if result:
        turn["said"] = False
        return msg
    return None


def cost_of(result: dict) -> float | None:
    c = result.get("total_cost_usd")
    return float(c) if isinstance(c, int | float) and not isinstance(c, bool) else None


def error_of(result: dict) -> str | None:
    return str(result.get("result") or result.get("subtype") or "error") if result.get("is_error") else None


def claude(model: str | None, *extra: str) -> list[str]:
    return ["claude", "-p", "--output-format", "stream-json", "--verbose", "--dangerously-skip-permissions",
            *extra, *(["--model", model] if model else [])]


def once(prompt: Path, model: str | None, budget: Budget) -> int:
    rc = 0
    for ev in events_in():
        if why := budget.why_not():
            line({"cost_usd": 0, "note": why, "skipped": True})
            continue
        try:
            p = subprocess.Popen(claude(model), stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        except OSError as e:
            line({"cost_usd": 0, "note": f"cannot start claude (Claude Code): {e}"})
            return 127
        try:
            p.stdin.write(instructions(prompt) + "\n" + about(ev, item_of(ev)))
            p.stdin.close()
        except OSError:
            pass
        seen, turn = {}, {}
        for raw in p.stdout:
            if (result := read(raw, turn)) is not None:
                seen.update(cost=cost_of(result), error=error_of(result))
        code = p.wait()
        rc = rc or code
        budget.spend(seen.get("cost") or 0.0)
        line({"cost_usd": seen.get("cost") or 0} | ({"note": n} if (n := seen.get("error") or (
            None if code == 0 else f"claude exited {code}" if seen else f"claude exited {code} before answering")) else {}))
    return rc


def session(prompt: Path, model: str | None, budget: Budget) -> int:
    try:
        p = subprocess.Popen(claude(model, "--input-format", "stream-json", "--append-system-prompt", instructions(prompt)),
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    except OSError as e:
        print(f"cannot start claude (Claude Code): {e}", file=sys.stderr)
        return 127
    done, total, pending = threading.Event(), [0.0], [False]

    def listen():
        turn: dict = {}
        for raw in p.stdout:
            if (result := read(raw, turn)) is not None:
                now_total = cost_of(result)
                cost = max(0.0, now_total - total[0]) if now_total is not None else 0.0
                total[0] = now_total if now_total is not None else total[0]
                budget.spend(cost)
                line({"cost_usd": round(cost, 6)} | ({"note": n} if (n := error_of(result)) else {}))
                pending[0] = False
                done.set()
        done.set()
    reader = threading.Thread(target=listen, daemon=True)
    reader.start()
    for raw in sys.stdin:
        try:
            ev = json.loads(raw)
        except ValueError:
            continue
        if why := budget.why_not():
            line({"cost_usd": 0, "note": why, "skipped": True})
            continue
        msg = {"type": "user", "message": {"role": "user", "content": about(ev, item_of(ev))}}
        done.clear()
        pending[0] = True
        try:
            p.stdin.write(json.dumps(msg) + "\n")
            p.stdin.flush()
        except OSError:
            break
        done.wait()
        if p.poll() is not None:
            break
    if pending[0]:
        line({"cost_usd": 0, "note": "the agent exited before answering"})
    try:
        p.stdin.close()
    except OSError:
        pass
    code = p.wait()
    reader.join(5)
    return code


def main(argv: list[str] | None = None) -> int:
    a = argparse.ArgumentParser(prog="orchestra agent", description="One Claude Code turn per event (a program for a machine).")
    a.add_argument("prompt", type=Path, help="the instructions file")
    a.add_argument("--model")
    a.add_argument("--per-day-usd", type=float, help="no turn is started past this much spent today")
    args = a.parse_args(argv)
    if not args.prompt.is_file():
        print(f"orchestra agent: no prompt file {args.prompt}", file=sys.stderr)
        return 2
    budget = Budget(args.per_day_usd)
    run = session if os.environ.get("ORCHESTRA_CONTINUOUS") else once
    return run(args.prompt, args.model, budget)
