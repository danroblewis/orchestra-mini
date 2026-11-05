"""Which machines may be associated with each flow: hypotheses with their evidence, never a label
and never a declaration. Copied from the old Orchestra's ``sd/attribute.py``, without the grouping
of people relayed from one system.

System dynamics has no machines. A machine does actual work (a program, an AI agent, a person) and
is known by its events on the bus; how its work relates to the model's flows is unknown, and
inferring it is what Orchestra is for. A machine may be associated with no flow, one or several,
and several machines with one flow. A flow's counter is a measurement of the system: who wrote it
(an instrument, or the ``machine`` label of a series) says nothing about which machine's work it
reflects, so it is never read for this; nor does any configuration or plugin say it.

Each hypothesis is recorded in ``flow.evidence["machines"]`` (``hypotheses``, best first, each with
its evidence and a ``confidence`` between 0 and 1), from one of:

- **its own events**: the flow's counter is itself one machine's event count
  (``orchestra_events_total`` of one type and source);
- **events on the bus**: a machine whose own events, counted per (type, source) in
  ``orchestra_events_total``, match the counter's increments over the window. The events of one
  type from one machine (its instances summed) are compared with the counter in three ways, and
  the first that holds is the evidence:

  1. *step by step*: at the same step or one step either way for the whole window (two clocks,
     two writes), their increments are equal in at least ``EQUAL_STEPS`` of the steps either moved
     in (confidence: that share);
  2. *event by event*: each increment of the counter is paired with an event no more than one
     step away (``TOLERANCE``), so events that straddle a step boundary, some early and some on
     time, still pair. Two-sided when at least ``EQUAL_STEPS`` of the counter's increments and of
     the events pair; one-sided when every increment of the counter pairs, with the events' total
     within ``RATIO`` of the counter's (the rest of the events may feed another flow; confidence:
     the share of the events that pair). Either is discounted for chance: the same events moved
     far away in time (``CHANCE_SHIFTS``) pair too, by coincidence, with a share that is subtracted
     (confidence: ``paired - chance``, so a match chance could half make is never near certain;
     two-sided only when ``(paired - chance) / (1 - chance)`` is at least ``EQUAL_STEPS``), and a counter so busy that chance alone
     pairs more than ``MAX_CHANCE`` of it is never matched this way;
  3. *by correlation*: they correlate at least ``MIN_CORR`` at one of those lags with totals
     within ``RATIO`` of each other (confidence: the correlation times the smaller ratio of the
     totals).

  Step by step and by correlation are discounted for chance as event by event is: the events
  moved ``CHANCE_SHIFTS`` steps around the live steps are measured the same way, and that share is
  subtracted (a busy event type matches an unrelated counter step by step mostly by coincidence);
  a hypothesis chance makes as well is dropped. An event type that feeds two ways out of one stock
  (a subject ended one way or another) is hypothesised on each, in proportion to the events that
  pair with its counter, when the events its counter did not take pair with the other exits'.

  Every machine that matches is a hypothesis. Machines that tie on every test form one group
  (``tied``) that shares the confidence: timing cannot tell them apart. While the counter has fewer
  than discovery's ``MIN_EVENTS`` events, each confidence is scaled by its share of them
  (``capped``): a few events coincide by chance.

A flow with no hypothesis says so: no machine associated yet. That is a question about the system,
not an answer. None of this changes a flow's equation or the model's stocks and flows. Its limits:
a match is a correlation in time, not a cause; two machines reacting to the same events both match;
and discovery itself may not tell apart two similar flows between the same stocks, so a
hypothesis may belong to a flow's twin. Orchestra's own event types (``orchestra.*``) are never
candidates."""

from __future__ import annotations

import math

from .balance import MIN_EVENTS
from .series import EVENTS as EVENT_COUNTER
from .series import Series, writer_of

EQUAL_STEPS = 0.9        # increments equal in this share of the steps both were live
MIN_CORR = 0.9           # ... or correlated at least this well
RATIO = 2.0              # ... with totals within this factor of each other
MIN_MOVES = 3            # a counter that moved in fewer steps than this cannot be matched to anyone's events
LAGS = (0, -1, 1)        # the event count compared at the same step, a step before and a step after
TOLERANCE = 1            # event by event: an event pairs with an increment at most this many steps away
CHANCE_SHIFTS = (7, 13, 29, 53, 97)   # the events moved this many steps (around the window) pair only by chance
MAX_CHANCE = 0.5         # a counter that chance alone pairs this well cannot be matched event by event
UNSET = "no machine associated yet"
MAX_HYPOTHESES = 5


def _nonzero(s: Series) -> dict[int, float]:
    """The steps the series moved in, and by how much (worked out once per series)."""
    cached = s.__dict__.get("_moved")
    if cached is None or cached[0] is not s.values:
        cached = (s.values, {i: d for i, d in enumerate(s.deltas) if d})
        s.__dict__["_moved"] = cached
    return cached[1]


def match(counter: Series, events: Series, siblings: list[Series] | tuple = ()) -> dict | None:
    """How well ``events`` (one machine's events of one type) explain the counter's increments, or
    None when they do not. Worked on the steps either moved in: a long window is mostly quiet.
    ``siblings``: the counters of the other flows out of the same stock, which the same events may
    feed (one event type ending a subject one way or another): its events are then shared between
    them, each hypothesis in proportion to the events that pair with its own counter."""
    n = min(len(counter.deltas), len(events.deltas))
    xc, xe = _nonzero(counter), _nonzero(events)
    tc, te = sum(xc.values()), sum(xe.values())
    if len(xc) < MIN_MOVES or tc <= 0 or te <= 0 or not (1 / (2 * RATIO) <= te / tc <= 2 * RATIO):
        return None                                   # nowhere near: none of the tests could pass
    live = set(range(n) if counter.live is None else counter.live)
    if events.live is not None:
        live &= set(events.live)
    found = _by_step(xc, xe, live, n)
    if found is not None and found["test"] == "step by step":
        return found
    others: dict[int, float] = {}
    for sib in siblings:
        for i, v in _nonzero(sib).items():
            others[i] = others.get(i, 0.0) + v
    return _by_event(xc, xe, live, n, others) or found


def _pair(x: dict[int, float], y: dict[int, float], left: dict[int, float] | None = None) -> tuple[float, dict[int, float]]:
    """Pair the counter's increments ``x`` with events ``y`` no more than ``TOLERANCE`` steps away,
    each event used once: the amount paired, and how much of it at each offset. Taken in time
    order, each increment from the earliest unused events in reach, which pairs as much as any
    pairing can when every increment reaches equally far. ``left``, when given, is filled with
    the events no increment took."""
    left = left if left is not None else {}
    left.update(y)
    paired, by_offset = 0.0, {}
    for i in sorted(x):
        need = x[i]
        for d in range(-TOLERANCE, TOLERANCE + 1):
            if need <= 1e-9:
                break
            if (have := left.get(i + d, 0.0)) > 0:
                take = min(need, have)
                left[i + d], need, paired = have - take, need - take, paired + take
                by_offset[d] = by_offset.get(d, 0.0) + take
    return paired, by_offset


def _by_event(xc: dict[int, float], xe: dict[int, float], live: set[int], n: int, others: dict[int, float] | None = None) -> dict | None:
    """Event by event, within a step either way, discounted for chance (see the module's text).
    ``others``: the increments of the other exits of the same stock; events this counter did not
    take that pair with them are this event type's other way out, not a miss (``shares its events``)."""
    x = {i: v for i, v in xc.items() if i in live}
    y = {j: v for j, v in xe.items() if j in live}
    tx, ty = sum(x.values()), sum(y.values())
    if len(x) < MIN_MOVES or tx <= 0 or ty <= 0:
        return None
    left: dict[int, float] = {}
    paired, by_offset = _pair(x, y, left)
    chance = sum(_pair(x, {(j + s) % n: v for j, v in y.items() if (j + s) % n in live})[0] for s in CHANCE_SHIFTS) / len(CHANCE_SHIFTS)
    if chance / tx > MAX_CHANCE:
        return None
    covered, explained = paired / tx, paired / ty
    if min(covered, explained) >= EQUAL_STEPS:
        test, share, base = "event by event", min(covered, explained), chance / max(tx, ty)
    elif covered >= 1 - 1e-9 and 1 <= ty / tx <= RATIO:
        test, share, base = "covers every increment", explained, chance / ty
    elif others and covered >= EQUAL_STEPS and (paired + _pair({i: v for i, v in others.items() if i in live}, left)[0]) / ty >= EQUAL_STEPS:
        # the events this counter did not take pair with the stock's other exits: one event type
        # feeding two ways out, hypothesised on each in proportion to the events that pair with it
        test, share, base = "shares its events with another exit", explained, chance / ty
    else:
        return None
    if test == "event by event" and (share - base) / (1 - base) < EQUAL_STEPS:
        return None
    confidence = discounted(share, base)
    return {"confidence": round(confidence, 4), "test": test, "paired": {"flow": round(covered, 4), "events": round(explained, 4)},
            "chance": round(chance / tx, 4), "lag_steps": max(by_offset, key=lambda d: (by_offset[d], -abs(d))),
            "counts": {"flow": round(tx, 3), "events": round(ty, 3)}, "steps": len(live)}


def discounted(share: float, base: float) -> float:
    """A share that paired, less what chance alone pairs: ``share - base``. Never certainty while
    chance pairs anything (the owner's reviewers, 2026-11-02, saw a share of 1 with 42% by chance
    shown as 100%: ``(share - base) / (1 - base)`` is 1 at a share of 1 whatever the base)."""
    return max(0.0, share - base)


def _step_measures(x: dict[int, float], y: dict[int, float], k: int) -> tuple[float, float, int, int]:
    """Step by step, of the counter's increments ``x`` and the events ``y`` over ``k`` steps: the share
    of the steps either moved in where they are equal, their correlation, and how many moved and were equal."""
    tx, ty = sum(x.values()), sum(y.values())
    moved = x.keys() | y.keys()
    same = sum(1 for i in moved if abs(x.get(i, 0.0) - y.get(i, 0.0)) < 1e-9)
    sxy = sum(v * y[i] for i, v in x.items() if i in y) - tx * ty / k
    sxx = sum(v * v for v in x.values()) - tx * tx / k
    syy = sum(v * v for v in y.values()) - ty * ty / k
    corr = sxy / math.sqrt(sxx * syy) if sxx > 0 and syy > 0 else 0.0
    return (same / len(moved) if moved else 0.0), corr, len(moved), same


def _by_step(xc: dict[int, float], xe: dict[int, float], live: set[int], n: int) -> dict | None:
    """Step by step at one lag for the window, or by correlation (see the module's text), each
    discounted for chance as event by event is: the same events moved ``CHANCE_SHIFTS`` steps
    around the live steps match by coincidence too, and that share is subtracted (``chance``). A
    hypothesis chance would make as well is dropped."""
    best = None
    for lag in LAGS:
        edge = 0 if lag < 0 else n - 1 if lag > 0 else None      # the step the lag takes out of the window
        valid = live - {edge} if edge in live else live         # set operations: hundreds of candidates per flow
        if not valid:
            continue
        x = {i: v for i, v in xc.items() if i in valid}
        y = {j - lag: v for j, v in xe.items() if j - lag in valid}
        tx, ty, k = sum(x.values()), sum(y.values()), len(valid)
        if len(x) < MIN_MOVES or tx <= 0 or ty <= 0:
            continue
        equal_moving, corr, moved, same = _step_measures(x, y, k)
        ratio = ty / tx
        if equal_moving >= EQUAL_STEPS:
            share, test = equal_moving, "step by step"
        elif corr >= MIN_CORR and 1 / RATIO <= ratio <= RATIO:
            share, test = corr * min(ratio, 1 / ratio), "by correlation"
        else:
            continue
        # chance: the events moved around the live steps, measured the same way
        order = sorted(valid)
        at = dict(zip(order, range(len(order))))
        shifted = [_step_measures(x, {order[(at[j] + sh) % len(order)]: v for j, v in y.items()}, k) for sh in CHANCE_SHIFTS]
        base = sum((m[0] if test == "step by step" else max(0.0, m[1]) * min(ratio, 1 / ratio)) for m in shifted) / len(shifted)
        confidence = discounted(share, base)
        if confidence <= 0:
            continue
        if best is None or (test, confidence) > (best["test"], best["confidence"]):   # "step by step" first
            best = {"confidence": round(confidence, 4), "test": test, "equal_steps": round((k - moved + same) / k, 4),
                    "equal_steps_moving": round(equal_moving, 4), "corr": round(corr, 4), "lag_steps": lag, "chance": round(base, 4),
                    "counts": {"flow": round(tx, 3), "events": round(ty, 3)}, "steps": k}
    return best


def _why(m: dict) -> str:
    """The evidence of a match, in words."""
    c = m["counts"]
    by_chance = f", {round(100 * m['chance'])}% by chance" if m.get("chance") is not None else ""
    if m["test"] == "step by step":
        return f"match its increments step by step: equal in {round(100 * m['equal_steps_moving'])}% of the steps either moved{by_chance}"
    if m["test"] == "by correlation":
        return f"correlate with its increments ({m['corr']:.2f}), {c['events']:g} events to {c['flow']:g}{by_chance}"
    pairs = (f"{round(100 * m['paired']['flow'])}% of its {c['flow']:g} increments pair with one of the {c['events']:g} events "
             f"within a step, {round(100 * m['chance'])}% by chance")
    if m["test"] == "covers every increment":
        return f"cover every increment: {pairs}; the other events may feed another flow"
    if m["test"] == "shares its events with another exit":
        return f"share their events with another way out of the same stock: {pairs}; the rest pair with the other exits' increments"
    return f"match its increments event by event: {pairs}"


def candidates(series: list[Series]) -> list[Series]:
    """Each machine's event counts, one per (type, machine): what a flow may be explained by."""
    return [s for s in series if s.name == EVENT_COUNTER and s.counter and writer_of(s) is not None
            and not (s.labels.get("type") or "").startswith("orchestra.")]


def tied(found: list[dict]) -> list[dict]:
    """The machines that tie on every test (the same confidence, by the same comparison, on the same
    counts): one group, sharing that confidence, since timing cannot tell them apart (five machines
    reacting to the same five events). Each is marked with its group's index (``tie``)."""
    def key(h):
        return (round(h["confidence"], 6), h.get("by"), h.get("method"), repr(h.get("counts")), repr(h.get("paired")))
    groups: dict = {}
    for h in found:
        groups.setdefault(key(h), []).append(h)
    out = []
    for hs in groups.values():
        if len(hs) > 1:
            for h in hs:
                h["tie"] = len(out)
            out.append({"machines": [h["machine"] for h in hs], "confidence": hs[0]["confidence"],
                        "why": f"{len(hs)} machines: timing cannot tell them apart"})
    return out


def associate(counter: Series | None, events: list[Series], siblings: list[Series] | tuple = ()) -> dict:
    """The machines that may be associated with the flow ``counter`` measures, best first, and
    why: ``flow.evidence["machines"]``. ``siblings``: the counters of the other flows out of the
    same stock (``match``)."""
    if counter is None:
        return {"hypotheses": [], "why": f"{UNSET}: no counter measures it, so no machine's events can be compared with it"}
    if counter.name == EVENT_COUNTER and (who := writer_of(counter)):
        return {"hypotheses": [{"machine": who, "origin": "inferred", "by": "its own events", "confidence": 1.0,
                                "events": {"type": counter.labels.get("type"), "source": who},
                                "why": f"its counter counts {who}'s own events of type {counter.labels.get('type')}"}]}
    best: dict[str, dict] = {}
    for e in events:
        if e is counter or (m := match(counter, e, siblings)) is None:
            continue
        who = writer_of(e)
        h = {"machine": who, "origin": "inferred", "by": "events on the bus",
             "events": {"type": e.labels.get("type"), "source": who}, **m,
             "why": f"{who}'s {e.labels.get('type')} events {_why(m)}"}
        if who not in best or h["confidence"] > best[who]["confidence"]:
            best[who] = h
    events_n = sum(counter.deltas)
    if events_n < MIN_EVENTS:
        for h in best.values():           # too few events to tell a machine from chance: never near certain
            h["confidence"] = round(h["confidence"] * events_n / MIN_EVENTS, 3)
            h["capped"] = (f"only {events_n:g} event{'' if events_n == 1 else 's'} in the window, fewer than the {MIN_EVENTS} discovery "
                           f"needs: its confidence scaled by {events_n:g}/{MIN_EVENTS}")
    found = sorted(best.values(), key=lambda h: (-h["confidence"], h["machine"]))[:MAX_HYPOTHESES]
    if found:
        return {"hypotheses": found, **({"tied": t} if (t := tied(found)) else {})}
    return {"hypotheses": [], "why": f"{UNSET}: no machine's events on the bus match its counter"}
