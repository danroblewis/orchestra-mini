"""``orchestra decide "<question>" --answers a,b,c [--inputs f,g] [--min-confidence 0.6] [--model M]``
(SPEC §3): a built-in program a machine names in its ``command``, reading events on stdin and
writing event and run lines on stdout like any machine.

For each event (once: the one; continuous: each as it comes) it makes one model call through the
``claude`` CLI, ``claude -p --output-format json --json-schema``, so it uses the owner's Claude Code
sign-in. It asks the question with the event and the item's ``inputs`` fields (all of them when
none are named), and prints ``decision.made`` (the answer, its confidence, a one-line reason) and
``<machine>.<answer>`` (so ``on: ci_triage.real`` reads plainly), then the run line with the
call's cost. Below ``--min-confidence`` the answer is ``unsure``."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

from .program import events_in, item_of, line


def schema(answers: list[str]) -> str:
    return json.dumps({"type": "object", "required": ["answer", "confidence", "reason"], "properties": {
        "answer": {"type": "string", "enum": answers},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reason": {"type": "string", "description": "one line"}}})


def prompt(question: str, answers: list[str], inputs: list[str], event, item) -> str:
    fields = (item or {}).get("fields") or {}
    if inputs:
        fields = {k: fields.get(k) for k in inputs}
    return (f"{question}\n\nAnswer one of: {', '.join(answers)}. Give your confidence from 0 to 1 "
            f"and a one-line reason.\n\nThe event:\n{json.dumps(event, indent=2)}\n\n"
            f"The item's fields:\n{json.dumps(fields, indent=2)}\n")


def answer_of(lines: list[str]) -> tuple[dict | None, float | None]:
    """The structured answer and the cost from the CLI's output."""
    for raw in reversed(lines):
        try:
            out = json.loads(raw)
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


def words(text: str) -> list[str]:
    return [w.strip() for w in (text or "").split(",") if w.strip()]


def decide(args, event: dict | None) -> int:
    answers, machine = words(args.answers), os.environ.get("ORCHESTRA_MACHINE") or "decide"
    argv = ["claude", "-p", "--output-format", "json", "--json-schema", schema(answers), *(["--model", args.model] if args.model else [])]
    try:
        r = subprocess.run(argv, input=prompt(args.question, answers, words(args.inputs), event, item_of(event)),
                           capture_output=True, text=True, timeout=600, check=False)
    except (OSError, subprocess.TimeoutExpired) as e:
        line({"cost_usd": 0, "note": f"claude: {e}"})
        return 1
    sys.stderr.write(r.stderr)
    got, cost = answer_of(r.stdout.splitlines())
    if r.returncode != 0 or got is None:
        sys.stderr.write(r.stdout[-4000:])
        line({"cost_usd": cost or 0, "note": "no answer in the output" if r.returncode == 0 else f"claude exited {r.returncode}"})
        return r.returncode or 1
    answer = got.get("answer") if got.get("answer") in answers else "unsure"
    try:
        confidence = float(got.get("confidence"))
    except (TypeError, ValueError):
        confidence = 0.0
    if confidence < args.min_confidence:
        answer = "unsure"
    data = {"machine": machine, "answer": answer, "confidence": confidence, "reason": str(got.get("reason") or "")}
    subject, run = (event or {}).get("subject"), f"{(event or {}).get('id', '')}:{os.environ.get('ORCHESTRA_RUN', '')}"
    line({"type": "decision.made", "subject": subject, "id": f"decision.made:{run}", "data": data})
    line({"type": f"{machine}.{answer}", "subject": subject, "id": f"{machine}.{answer}:{run}",
          "data": {k: data[k] for k in ("confidence", "reason")}})
    line({"cost_usd": cost or 0})
    return 0


def main(argv: list[str] | None = None) -> int:
    a = argparse.ArgumentParser(prog="orchestra decide", description="One model call per event: an answer of a few (a program for a machine).")
    a.add_argument("question")
    a.add_argument("--answers", required=True, help="the answers, comma-separated: real,noise")
    a.add_argument("--inputs", default="", help="the item's fields to include, comma-separated (default: all)")
    a.add_argument("--min-confidence", type=float, default=0.6)
    a.add_argument("--model")
    args = a.parse_args(argv)
    if len(words(args.answers)) < 2:
        print("orchestra decide: --answers names at least two, comma-separated", file=sys.stderr)
        return 2
    if os.environ.get("ORCHESTRA_CONTINUOUS"):
        rc = 0
        for raw in sys.stdin:
            try:
                ev = json.loads(raw)
            except ValueError:
                continue
            rc = decide(args, ev if isinstance(ev, dict) else None) or rc
        return rc
    rc = 0
    for ev in events_in():
        rc = decide(args, ev) or rc
    return rc
