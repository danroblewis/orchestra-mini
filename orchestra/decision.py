"""The decision kind (SPEC §3): one model call through the ``claude`` CLI, ``claude -p
--output-format json --json-schema``, so it uses the owner's Claude Code sign-in. It asks the
``question`` with the event and the item's ``inputs`` fields (all of them when none are named),
and publishes ``decision.made`` (the answer, its confidence, a one-line reason) and
``<process>.<answer>``. Below ``min_confidence`` the answer is ``unsure``."""

from __future__ import annotations

import json

from .command import log_to, run
from .lines import event as make_event


def schema(answers: list[str]) -> str:
    return json.dumps({"type": "object", "required": ["answer", "confidence", "reason"], "properties": {
        "answer": {"type": "string", "enum": answers},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reason": {"type": "string", "description": "one line"}}})


def prompt(proc, event, item) -> str:
    fields = (item or {}).get("fields") or {}
    if proc.inputs:
        fields = {k: fields.get(k) for k in proc.inputs}
    return (f"{proc.question}\n\nAnswer one of: {', '.join(proc.answers)}. Give your confidence from 0 to 1 "
            f"and a one-line reason.\n\nThe event:\n{json.dumps(event, indent=2)}\n\n"
            f"The item's fields:\n{json.dumps(fields, indent=2)}\n")


def answer_of(lines: list[str]) -> tuple[dict | None, float | None]:
    """The structured answer and the cost from the CLI's output."""
    for line in reversed(lines):
        try:
            out = json.loads(line)
        except ValueError:
            continue
        if not isinstance(out, dict) or out.get("type") != "result":
            continue
        cost = out.get("total_cost_usd") if isinstance(out.get("total_cost_usd"), int | float) else None
        got = out.get("structured_output")
        if not isinstance(got, dict):
            try:
                got = json.loads(out.get("result") or "")
            except (TypeError, ValueError):
                got = None
        return (got if isinstance(got, dict) else None), cost
    return None, None


def run_decision(project, proc, event, item, env, publish_event, log) -> tuple[int | None, float | None, str | None]:
    argv = [project.config.agent.command, "-p", "--output-format", "json", "--json-schema", schema(proc.answers)]
    argv += ["--model", proc.model] if proc.model else []
    res = run(argv, project.repo, log, timeout=proc.timeout, env=env, stdin=prompt(proc, event, item))
    got, cost = answer_of(res.lines)
    if not res.ok or got is None:
        log_to(log, "".join(res.lines)[-4000:])
        return res.rc, cost, res.error or ("timed out" if res.timed_out else "no answer in the output")
    answer = got.get("answer") if got.get("answer") in proc.answers else "unsure"
    try:
        confidence = float(got.get("confidence"))
    except (TypeError, ValueError):
        confidence = 0.0
    if confidence < proc.min_confidence:
        answer = "unsure"
    data = {"process": proc.name, "answer": answer, "confidence": confidence, "reason": str(got.get("reason") or "")}
    subject = (event or {}).get("subject")
    for t, d in (("decision.made", data), (f"{proc.name}.{answer}", {k: data[k] for k in ("confidence", "reason")})):
        ev, _ = make_event({"type": t, "subject": subject, "data": d}, proc.name,
                           f"{t}:{(event or {}).get('id', '')}:{env.get('ORCHESTRA_RUN', '')}")
        publish_event(ev)
    return res.rc, cost, None
