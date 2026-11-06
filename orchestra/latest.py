"""``latest``, the built-in projection (SPEC §6), run in Orchestra's own process for every subject.

A subject's ``latest`` view: ``fields``, every key of every fact's ``data``, the latest fact (by
``time``, ties by ``seq``) winning, a key set to null removed; ``by_type``, the latest fact's
``data`` of each type, so what one fact said stays together; ``type``, the latest fact's;
``first`` and ``last``, the times of the first and latest facts; ``kind``, what the subject is;
``facts``, how many there are; and ``done``, whether the latest type's verb is ``completed``,
``closed`` or ``cancelled`` (so a later fact reopens it). Ordering by ``time`` keeps a month of
history backfilled after today's snapshot from becoming the current state."""

from __future__ import annotations

from .events import event_time

NAME = "latest"
DONE = (".completed", ".closed", ".cancelled")


def kind_of(facts: list[dict]) -> str | None:
    """What the subject is: the first fact's explicit ``kind``, else its type less the verb
    (``github.pr.opened`` is a ``github.pr``); Orchestra's own types (``orchestra.*``) say nothing."""
    for f in facts:
        if isinstance(f.get("kind"), str) and f["kind"]:
            return f["kind"]
    for f in facts:
        parts = str(f.get("type") or "").split(".")
        if parts[0] != "orchestra" and len(parts) >= 3:
            return ".".join(parts[:-1])
    return None


def ordered(facts: list[dict]) -> list[dict]:
    """By ``time``, ties by ``seq`` (log order)."""
    return sorted(facts, key=lambda f: (event_time(f) or 0.0, f.get("seq") or 0))


def view(subject: str, facts: list[dict]) -> dict:
    fs = ordered(facts)
    fields: dict = {}
    by_type: dict = {}
    for f in fs:
        if isinstance(f.get("data"), dict):
            for k, v in f["data"].items():
                if v is None:
                    fields.pop(k, None)
                else:
                    fields[k] = v
        by_type[f.get("type")] = f.get("data")
    last = fs[-1] if fs else {}
    return {"subject": subject, "kind": kind_of(fs), "type": last.get("type"), "first": fs[0].get("time") if fs else None,
            "last": last.get("time"), "done": str(last.get("type", "")).endswith(DONE), "facts": len(fs),
            "fields": fields, "by_type": by_type}
