"""Predictions (SPEC §6): an event ``orchestra.prediction.made`` says what a stock or a flow will do
by when; Orchestra judges it when its time comes and publishes ``orchestra.prediction.judged``.

``data`` of a prediction (Orchestra's own type, so it reads it):

- ``stock`` or ``flow``: the id in the model (``orchestra model``);
- ``expect``: ``up``, ``down`` or ``same``, or ``value``: a number (the level, or the rate per the
  model's time unit), within ``tolerance`` (default 0.1, a share of it);
- ``by``: when to judge it (RFC 3339), or ``within``: a duration from when it was made (``30m``).

Judging compares the stock's level at ``by`` with its level when the prediction was made, or the
flow's rate over (made, by] with its rate over as long before; ``same`` means within the
tolerance. The judgement's ``data``: ``prediction`` (the event's id), ``verdict`` (``met``,
``missed`` or ``unjudged`` with ``why``), ``before``, ``observed`` and what was expected; its
``subject`` is the prediction's. Each prediction is recorded once by its id in
``<state>/model/predictions.json`` with its status."""

from __future__ import annotations

import time

from .. import metrics
from ..config import parse_duration
from ..events import envelope, event_time
from .store import ModelStore

MADE = "orchestra.prediction.made"
JUDGED = "orchestra.prediction.judged"
RECORD = "predictions"


def check(data) -> tuple[dict | None, str | None]:
    """The prediction as recorded, or why it cannot be judged."""
    if not isinstance(data, dict):
        return None, "data is not an object"
    part = "stock" if data.get("stock") else "flow" if data.get("flow") else None
    if part is None:
        return None, "name a stock or a flow"
    out = {"part": part, "id": str(data[part]), "tolerance": float(data.get("tolerance", 0.1))}
    if data.get("expect") in ("up", "down", "same"):
        out["expect"] = data["expect"]
    elif isinstance(data.get("value"), int | float) and not isinstance(data.get("value"), bool):
        out["value"] = float(data["value"])
    else:
        return None, "expect up, down or same, or give a value"
    return out, None


def due(ev: dict, data: dict) -> float | None:
    made = event_time(ev)
    if data.get("by"):
        return event_time({"time": data["by"]})
    if data.get("within") and made is not None:
        try:
            return made + parse_duration(data["within"])
        except Exception:
            return None
    return None


def _level(points: list[tuple[float, float]], t: float) -> float | None:
    before = [v for ts, v in points if ts <= t]
    return before[-1] if before else None


def _rate(points: list[tuple[float, float]], a: float, b: float, unit_s: float) -> float | None:
    lo, hi = _level(points, a), _level(points, b)
    if hi is None:
        return None
    return max(0.0, hi - (lo or 0.0)) / ((b - a) / unit_s)


def _points(project, ref) -> list[tuple[float, float]] | None:
    if ref is None or ref.source != "own":
        return None
    return metrics.for_project(project).raw(ref.name, {**ref.labels, "project": project.name})


def judge(project, model, rec: dict) -> dict:
    """The judgement of one recorded prediction, now that its time has come."""
    p, made, by = rec["check"], rec["made"], rec["due"]
    try:
        part = model.stock(p["id"]) if p["part"] == "stock" else model.flow(p["id"])
    except (StopIteration, AttributeError):
        return {"verdict": "unjudged", "why": f"no {p['part']} {p['id']} in the model"}
    ref = part.series if p["part"] == "stock" else part.counter
    pts = _points(project, ref)
    if pts is None:
        return {"verdict": "unjudged", "why": f"its series is read from another store ({ref.source if ref else 'none'})"}
    unit_s = {"second": 1.0, "minute": 60.0, "hour": 3600.0}.get(model.time_unit, 60.0)
    if p["part"] == "stock":
        before, observed = _level(pts, made), _level(pts, by)
    else:
        before, observed = _rate(pts, 2 * made - by, made, unit_s), _rate(pts, made, by, unit_s)
    if before is None or observed is None:
        return {"verdict": "unjudged", "why": "no data at the time it was made or due"}
    if "value" in p:
        ok = abs(observed - p["value"]) <= p["tolerance"] * max(abs(p["value"]), 1e-9)
    else:
        moved = observed - before
        same = abs(moved) <= p["tolerance"] * max(abs(before), 1e-9)
        ok = same if p["expect"] == "same" else (not same and (moved > 0) == (p["expect"] == "up"))
    return {"verdict": "met" if ok else "missed", "before": before, "observed": observed}


def sync(project, events, publisher, model, now: float | None = None) -> list[dict]:
    """Record the predictions the bus gained, judge those now due, publish the judgements."""
    store = ModelStore.for_project(project)
    now = time.time() if now is None else now
    with store.locked(RECORD):
        rec = store.read_record(RECORD) or {"at": 0, "predictions": {}}
        end = events.end()
        if end < rec["at"]:
            rec = {"at": 0, "predictions": {}}
        for ev in events.all(rec["at"]):
            if ev.get("type") != MADE or ev.get("id") in rec["predictions"]:
                continue
            data = ev.get("data") if isinstance(ev.get("data"), dict) else {}
            checked, why = check(data)
            when = due(ev, data) if checked else None
            if checked and when is None:
                why = "say when: by (a time) or within (a duration)"
            rec["predictions"][ev["id"]] = {"source": ev.get("source"), "subject": ev.get("subject"), "made": event_time(ev),
                                            "due": when, "check": checked, "data": data,
                                            "status": "waiting" if why is None else "invalid", "why": why}
        rec["at"] = end
        judged = []
        for pid, p in rec["predictions"].items():
            if p["status"] != "waiting" or p["due"] > now:
                continue
            verdict = judge(project, model, p) if model is not None else {"verdict": "unjudged", "why": "no model yet"}
            p.update(status="judged", verdict=verdict)
            ev = envelope(JUDGED, "orchestra", {"prediction": pid, **verdict, "expected": p["data"]},
                          subject=p.get("subject") or pid, id=f"judged:{pid}")
            publisher.publish(ev)
            judged.append(ev)
        store.write_record(RECORD, rec)
    return judged
