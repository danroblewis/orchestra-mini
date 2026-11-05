"""The agent kind (SPEC §3): one Claude Code turn, ``claude -p --output-format stream-json``, on
the prompt file with the triggering event, the item's state and the tools appended, given on
stdin. It runs with the owner's environment unstripped and ``--dangerously-skip-permissions``:
it acts with its own shell and the owner's credentials. Its events are the JSON event lines in
what it says (any assistant text, the result) or prints outside the stream, and those it
publishes itself with ``orchestra emit`` or a tool. Its cost is the stream's result's
``total_cost_usd``."""

from __future__ import annotations

import json

from . import tools
from .command import log_to, run

STREAM = {"system", "assistant", "user", "result", "stream_event", "rate_limit_event"}
EMITTING = """## Saying what happened

Say what you did as events: print one JSON object per line in your answer, like
{"type": "<system>.<noun>.<verb>", "subject": "<the item>", "data": {...}}
or run `orchestra emit '<json>'` in your shell."""


def prompt(project, proc, event: dict | None, item: dict | None) -> str:
    text = (project.repo / proc.prompt).read_text(encoding="utf-8")
    parts = [text.rstrip(), "## The event\n\n" + (json.dumps(event, indent=2) if event else "(none: a scheduled run)")]
    if item:
        parts.append("## The item's state\n\n" + json.dumps(item, indent=2))
    parts += ["## Your tools\n\n" + tools.describe(project, proc.tools), EMITTING]
    return "\n\n".join(parts) + "\n"


def args(project, proc) -> list[str]:
    a = [project.config.agent.command, "-p", "--output-format", "stream-json", "--verbose",
         "--dangerously-skip-permissions"]
    return a + (["--model", proc.model] if proc.model else [])


def texts(msg: dict) -> list[str]:
    """The text an assistant message or a result carries."""
    if msg.get("type") == "result":
        return [msg["result"]] if isinstance(msg.get("result"), str) else []
    content = (msg.get("message") or {}).get("content") if msg.get("type") == "assistant" else None
    return [p.get("text", "") for p in content or [] if isinstance(p, dict) and p.get("type") == "text"]


def run_agent(project, proc, event, item, env, publish, log) -> tuple[int | None, float | None, str | None]:
    """(exit code, cost, note)."""
    seen = {"cost": None, "error": None}

    def on_line(line: str) -> None:
        try:
            msg = json.loads(line)
        except ValueError:
            log_to(log, line.rstrip("\n"))
            return
        if not isinstance(msg, dict):
            return
        if msg.get("type") not in STREAM:
            publish([line])                     # an event line printed outside the stream
            return
        if msg.get("type") == "result":
            if isinstance(msg.get("total_cost_usd"), int | float):
                seen["cost"] = float(msg["total_cost_usd"])
            if msg.get("is_error"):
                seen["error"] = str(msg.get("result") or msg.get("subtype") or "error")
        for t in texts(msg):
            publish(t.splitlines())

    res = run(args(project, proc), project.repo, log, timeout=proc.timeout, env=env,
              stdin=prompt(project, proc, event, item), on_line=on_line)
    note = res.error or seen["error"] or ("timed out" if res.timed_out else None)
    return res.rc, seen["cost"], note
