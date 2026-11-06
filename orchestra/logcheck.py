"""``orchestra query check`` (SPEC §5): what can be checked from the log alone, one line per finding.

- a source whose facts' ``time`` always equals ``recordedtime`` (it stamps now, not the source's
  time); observations, Orchestra's own types and the machines that react to facts (``on``) are
  left out, since their time is now;
- ``data`` over 4 KB, and facts whose ``data`` says ``truncated`` (a list not read to its end);
- a type whose keys stopped appearing or newly appeared (the source changed its shape);
- a collector (a machine with ``every``) silent for more than three of its ``every``, or whose
  last runs each added only facts the log already had;
- each projection run twice over its facts: a missing or ``error`` view, or different bytes the
  second time; and its views as built step by step against the same views built from the whole
  log (a projection that is right only when it sees everything at once)."""

from __future__ import annotations

import json
import time

from . import runs
from .events import Events, event_time
from .facts import canonical, observed, read
from .views.builder import given
from .views.run import run_projection

DATA_MAX = 4096
SAME_S = 1.0             # a time this close to recordedtime is the same: stamped now
RESTATING = 3            # this many runs in a row adding only what the log had


def stamps_now(project, facts: list[dict]) -> list[str]:
    reacting = {m.name for m in project.config.machines if m.on}
    by: dict[str, list[bool]] = {}
    for f in facts:
        src, rec = f.get("source"), event_time({"time": f.get("recordedtime")})
        if rec is None or observed(f) or str(f.get("type", "")).startswith("orchestra.") or src in reacting:
            continue
        t = event_time(f)
        by.setdefault(str(src), []).append(t is not None and abs(t - rec) < SAME_S)
    return [f"source {s}: the time of each of its {len(same)} facts is when it was recorded: it stamps now, "
            "not the source's own time for the change" for s, same in sorted(by.items()) if len(same) >= 2 and all(same)]


def large(facts: list[dict]) -> list[str]:
    by: dict[str, list[int]] = {}
    for f in facts:
        n = len(canonical(f.get("data")).encode())
        if n > DATA_MAX:
            by.setdefault(str(f.get("type")), []).append(n)
    return [f"type {t}: {len(ns)} facts with data over 4 KB (the largest {max(ns) // 1024} KB): leave out bodies and HTML, "
            "and say so in truncated" for t, ns in sorted(by.items())]


def truncated(facts: list[dict]) -> list[str]:
    by: dict[str, int] = {}
    for f in facts:
        if isinstance(f.get("data"), dict) and f["data"].get("truncated"):
            by[str(f.get("type"))] = by.get(str(f.get("type")), 0) + 1
    return [f"type {t}: {n} facts say truncated: read the list to its end (page it), or say what was left out"
            for t, n in sorted(by.items())]


META = ("behind", "error", "error_at")


def stepwise(project, p, first: list[dict]) -> list[str]:
    """The stored views (built step by step) against ``first`` (built from the whole log), when the
    stored ones are up to date."""
    from .views.builder import load
    state = load(project, p.name)
    if not state.get("build") or state.get("built_to") != Events.for_project(project).end():
        return []
    key = "key" if p.whole else "subject"
    fresh = {str(o.get(key)): {k: v for k, v in o.items() if k not in META} for o in first}
    kept = {k: {x: y for x, y in v.items() if x not in META} for k, v in state["views"].items()}
    differ = sorted(k for k in fresh.keys() | kept.keys() if fresh.get(k) != kept.get(k))
    return [f"projection {p.name}: {len(differ)} views built step by step differ from the same views built from the "
            f"whole log (such as {differ[0]}): it must give each subject's view from that subject's facts alone"] if differ else []


def shapes(facts: list[dict]) -> list[str]:
    """Keys of a type's ``data`` present in every fact up to a point and in none after (stopped),
    or in none before a point and in every one after (newly appeared), two or more each side."""
    by: dict[str, list[set]] = {}
    for f in facts:
        if isinstance(f.get("data"), dict):
            by.setdefault(str(f.get("type")), []).append(set(f["data"]))
    out = []
    for t, keys in sorted(by.items()):
        for k in sorted(set().union(*keys)):
            has = [k in ks for ks in keys]
            cut = has.index(not has[0]) if (not has[0]) in has else None
            if cut is None or cut < 2 or len(has) - cut < 2 or any(x == has[0] for x in has[cut:]):
                continue
            out.append(f"type {t}: the key {k} " + ("stopped appearing" if has[0] else "newly appeared")
                       + f" after {cut} facts: the source changed its shape")
    return out


def collectors(project, now: float | None = None) -> list[str]:
    now = time.time() if now is None else now
    out = []
    for m in project.config.machines:
        if not m.every:
            continue
        recs = runs.read(project, m.name, 50)
        if not recs:
            out.append(f"collector {m.name}: no runs yet")
            continue
        last = event_time({"time": recs[0].get("start")}) or 0
        if now - last > 3 * m.every:
            out.append(f"collector {m.name}: silent since {recs[0].get('start')} (it runs every {m.every:g} s)")
        wrote = [r for r in recs if not r.get("continuous")][:RESTATING]
        if len(wrote) == RESTATING and all(r.get("facts", r.get("events", 0)) and r.get("restated") == r.get("facts") for r in wrote):
            out.append(f"collector {m.name}: each of its last {RESTATING} runs added only facts the log already had (same type, "
                       "subject and data): give each change the source's own id, or print the state as <system>.<noun>.observed")
    return out


def projections(project, facts: list[dict]) -> list[str]:
    out = []
    for p in project.config.projections:
        mine = given(p, facts)
        if not mine:
            out.append(f"projection {p.name}: no fact matches it yet")
            continue
        first, err = run_projection(project, p, mine)
        second, err2 = (None, None) if err else run_projection(project, p, mine)
        if err or err2:
            out.append(f"projection {p.name}: {err or err2}")
            continue
        key = "key" if p.whole else "subject"
        if not p.whole:
            printed = {o.get("subject") for o in first}
            missing = sorted({f["subject"] for f in mine} - printed)
            if missing:
                out.append(f"projection {p.name}: no view for {len(missing)} subjects (such as {missing[0]})")
        bad = sorted(str(o.get(key)) for o in first if "error" in o)
        if bad:
            out.append(f"projection {p.name}: an error view for {len(bad)} keys (such as {bad[0]})")
        if json.dumps(first) != json.dumps(second):
            out.append(f"projection {p.name}: different views the second time over the same facts: it must read only its stdin "
                       "and print times, not ages")
        else:
            out += stepwise(project, p, first)
    return out


def report(project) -> list[str]:
    facts = read(Events.for_project(project))
    return stamps_now(project, facts) + large(facts) + truncated(facts) + shapes(facts) + collectors(project) + projections(project, facts)
