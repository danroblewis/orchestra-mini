"""The agent kind (SPEC §3): Claude Code, ``claude -p --output-format stream-json``, with the
owner's environment unstripped and ``--dangerously-skip-permissions``: it acts with its own shell
and the owner's credentials. Its instructions are the prompt file with its tools and how to say
what happened; each event comes with the item's state.

- **once**: one turn, the instructions and the event given on stdin, then stdin closed.
- **continuous**: ``--input-format stream-json``, the instructions appended to the system
  prompt, stdin kept open, each event written as one user message (``continuous.py``).

Its events and metrics are the JSON lines in what it says (its assistant text; the result's, which
repeats it, only when a turn had none) or prints outside the stream, and those it writes itself with ``orchestra emit`` or a tool. Its cost
is the stream's results' ``total_cost_usd`` (a session's running total)."""

from __future__ import annotations

import json

from . import tools
from .command import run

STREAM = {"system", "assistant", "user", "result", "stream_event", "rate_limit_event"}
EMITTING = """## Saying what happened

Say what you did as events: print one JSON object per line in your answer, like
{"type": "<system>.<noun>.<verb>", "subject": "<the item>", "data": {...}}
and measurements as {"metric": "<what it measures>", "value": 3, "labels": {...}},
or run `orchestra emit '<json>'` in your shell."""


def instructions(project, machine) -> str:
    text = (project.repo / machine.prompt).read_text(encoding="utf-8")
    return "\n\n".join([text.rstrip(), "## Your tools\n\n" + tools.describe(project, machine.tools), EMITTING]) + "\n"


def about(event: dict | None, item: dict | None) -> str:
    parts = ["## The event\n\n" + (json.dumps(event, indent=2) if event else "(none: a scheduled run)")]
    if item:
        parts.append("## The item's state\n\n" + json.dumps(item, indent=2))
    return "\n\n".join(parts) + "\n"


def args(project, machine, continuous: bool = False) -> list[str]:
    a = [project.config.agent.command, "-p", "--output-format", "stream-json", "--verbose",
         "--dangerously-skip-permissions"]
    if continuous:
        a += ["--input-format", "stream-json", "--append-system-prompt", instructions(project, machine)]
    return a + (["--model", machine.model] if machine.model else [])


def texts(msg: dict) -> list[str]:
    """The text an assistant message or a result carries."""
    if msg.get("type") == "result":
        return [msg["result"]] if isinstance(msg.get("result"), str) else []
    content = (msg.get("message") or {}).get("content") if msg.get("type") == "assistant" else None
    return [p.get("text", "") for p in content or [] if isinstance(p, dict) and p.get("type") == "text"]


def read(line: str, out, turn: dict) -> dict | None:
    """One line of the stream: our lines in its text (or the line itself, printed outside the
    stream) to ``out``; the message when it is a result. ``turn`` remembers whether this turn
    said anything yet."""
    try:
        msg = json.loads(line)
    except ValueError:
        msg = None
    if not isinstance(msg, dict) or msg.get("type") not in STREAM:
        out.line(line)
        return None
    result = msg.get("type") == "result"
    if not (result and turn.get("said")):
        for t in texts(msg):
            turn["said"] = True
            for part in t.splitlines():
                out.line(part)
    if result:
        turn["said"] = False
        return msg
    return None


def cost_of(result: dict) -> float | None:
    c = result.get("total_cost_usd")
    return float(c) if isinstance(c, int | float) and not isinstance(c, bool) else None


def error_of(result: dict) -> str | None:
    return str(result.get("result") or result.get("subtype") or "error") if result.get("is_error") else None


def run_agent(project, machine, event, item, env, out, log) -> tuple[int | None, float | None, str | None]:
    """One turn: (exit code, cost, note)."""
    seen: dict = {}
    turn: dict = {}

    def on_line(line: str) -> None:
        if (result := read(line, out, turn)) is not None:
            seen.update(cost=cost_of(result), error=error_of(result))

    res = run(args(project, machine), project.repo, log, timeout=machine.timeout, env=env,
              stdin=instructions(project, machine) + "\n" + about(event, item), on_line=on_line)
    note = res.error or seen.get("error") or ("timed out" if res.timed_out else None)
    return res.rc, seen.get("cost"), note
