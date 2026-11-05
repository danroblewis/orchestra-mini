"""Discovery by conservation (SPEC §6): the stocks are the levels that counters balance, the flows
are those counters, and each flow's rate is fitted. A compact version of the old Orchestra's
``sd/discover.py``, which did this with sixty modules more (eras, hops, twins, restating, trust).

1. **Sample** every series on a grid of ``step`` over the window (``series.gather``).
2. **One count per flow.** Two counters whose increments are equal in nearly every step they
   moved (a program's own count and Orchestra's count of the event it publishes for it) are one:
   the program's is kept for the balances; both stay candidates for attribution.
3. **Find stocks.** A level that changed in at least ``MIN_EVENTS`` steps is balanced by the
   counters (``balance``: ``Δlevel ≈ Σ inflows − Σ outflows``); it is a stock when the balance
   fits with R² ≥ ``MIN_R2``. Two levels balanced by the same counters are one stock measured
   twice: the better fit keeps it.
4. **Connect.** A counter that is an outflow of one stock and an inflow of another is a flow
   between them; otherwise it comes from or goes to outside.
5. **Fit** each flow's equation (``fit``) on the window, reading the stocks' paths.
6. **Guess the machines** behind each flow from timing against ``orchestra_events_total``
   (``attribute``): hypotheses with their evidence, never an equation.

Every level and counter left out is recorded in ``model.discovery`` with why."""

from __future__ import annotations

import re

from .attribute import associate, candidates
from .balance import MIN_EVENTS, LeastSquares, balance
from .fit import events_of, fit_flow
from .model import Equation, Flow, Machine, Model, SeriesRef, Stock
from .path import Path
from .series import EVENTS, Series, writer_of

MIN_R2 = 0.9
SAME_STEPS = 0.9
UNITS = {"second": 1.0, "minute": 60.0, "hour": 3600.0}


def slug(s: Series) -> str:
    parts = [s.name] + [v for k, v in sorted(s.labels.items()) if k not in ("project", "command", "store")]
    return re.sub(r"[^a-z0-9]+", "_", " ".join(parts).lower()).strip("_") or "series"


def title(s: Series) -> str:
    said = [f"{k}={v}" for k, v in sorted(s.labels.items()) if k not in ("project", "command", "store")]
    return s.name + (f" ({', '.join(said)})" if said else "")


def ref(s: Series) -> SeriesRef:
    return SeriesRef(s.name, {k: v for k, v in s.labels.items() if k != "project"}, s.source, s.query)


def same_flow(a: Series, b: Series) -> bool:
    moved = [i for i, (x, y) in enumerate(zip(a.deltas, b.deltas)) if x or y]
    return bool(moved) and sum(1 for i in moved if abs(a.deltas[i] - b.deltas[i]) < 1e-9) >= SAME_STEPS * len(moved)


def one_count_each(counters: list[Series], left: dict[str, str]) -> list[Series]:
    kept: list[Series] = []
    for c in sorted(counters, key=lambda c: (c.name == EVENTS, len(c.key), c.key)):
        twin = next((k for k in kept if same_flow(c, k)), None)
        if twin is None:
            kept.append(c)
        else:
            left[c.key] = f"counts what {twin.key} counts, step by step: one flow"
    return kept


def unique(base: str, taken: set[str]) -> str:
    out, i = base, 2
    while out in taken:
        out, i = f"{base}_{i}", i + 1
    taken.add(out)
    return out


def discover(series: list[Series], *, start: float, end: float, step: float, name: str = "model",
             time_unit: str = "minute", left: dict[str, str] | None = None) -> Model:
    left = dict(left or {})
    unit_s = UNITS[time_unit]
    counters = [s for s in series if s.counter and sum(s.deltas) > 0]
    for s in series:
        if s.counter and s not in counters:
            left.setdefault(s.key, "a counter that counted nothing in the window")
    measured = one_count_each(counters, left)
    ls = LeastSquares([c.deltas for c in measured]) if measured else None
    found: list[tuple[Series, list, float, float]] = []
    for lv in (s for s in series if not s.counter):
        moves = sum(1 for d in lv.deltas if abs(d) > 1e-9)
        if moves < MIN_EVENTS:
            left[lv.key] = f"a level that changed in {moves} steps, fewer than {MIN_EVENTS}: nothing to balance"
            continue
        terms, r2, exact = balance(lv, measured, ls) if measured else ([], 0.0, 0.0)
        if not terms or r2 < MIN_R2:
            left[lv.key] = f"no counters balance it (best R² {r2:.2f}, under {MIN_R2}): an input, not a stock"
            continue
        sig = sorted((c.key, s) for c, s in terms)
        twin = next((f for f in found if sorted((c.key, s) for c, s in f[1]) == sig), None)
        if twin is not None:
            if twin[2] >= r2:
                left[lv.key] = f"the same counters balance {twin[0].key}: one stock measured twice"
                continue
            found.remove(twin)
            left[twin[0].key] = f"the same counters balance {lv.key}: one stock measured twice"
        found.append((lv, terms, r2, exact))

    taken: set[str] = set()
    stocks, by_level, claims = [], {}, {}
    for lv, terms, r2, exact in found:
        sid = unique(slug(lv), taken)
        by_level[sid] = lv
        stocks.append(Stock(sid, title(lv), ref(lv), {
            "r2": round(r2, 4), "exact": round(exact, 4), "steps": len(lv.deltas), "key": lv.key,
            "balance": [{"counter": c.key, "sign": s} for c, s in terms]}))
        for c, s in terms:
            claims.setdefault(c.key, (c, {}))[1].setdefault("dst" if s > 0 else "src", []).append((sid, r2))

    window = (end - start) / unit_s
    paths = {sid: Path.of(lv.points, [(start, end)], unit_s) for sid, lv in by_level.items()}
    on_bus = candidates(series)
    flows: list[Flow] = []
    counted: dict[str, Series] = {}
    for key, (c, sides) in claims.items():
        src = max(sides.get("src", []), key=lambda x: x[1], default=(None,))[0]
        dst = max(sides.get("dst", []), key=lambda x: x[1], default=(None,))[0]
        fid = unique(slug(c), taken)
        counted[fid] = c
        evs = events_of(c.points, start, end, 0.0 if c.name == EVENTS else None)
        best, *others = fit_flow(evs, window, paths, src)
        eq = Equation(best["form"], {k: round(v, 6) for k, v in best["params"].items()}, window=(start, end),
                      quality={"loglik": round(best["loglik"], 3), "aic": round(best["aic"], 3), "events": sum(d for _, d in evs)},
                      inputs=list(best["inputs"]))
        flows.append(Flow(fid, title(c), src, dst, ref(c), [eq], {
            "claims": {k: [s for s, _ in v] for k, v in sides.items()},
            "rejected": [{"form": o["form"], "inputs": o["inputs"], "why": o["rejected"]} for o in others if "rejected" in o]}))
    for f in flows:                         # the other ways out of the same stock may share one event type
        siblings = [counted[g.id] for g in flows if g is not f and f.src and g.src == f.src]
        f.evidence["machines"] = associate(counted[f.id], on_bus, siblings)
    machines: dict[str, Machine] = {}
    for f in flows:
        for h in f.evidence["machines"].get("hypotheses", []):
            machines.setdefault(h["machine"], Machine(h["machine"])).attributed.append(f.id)
    return Model(name, name, time_unit, stocks, flows, sorted(machines.values(), key=lambda m: m.id), {
        "strategy": "conservation", "window": [start, end], "step": step, "series": len(series),
        "counters": len(measured), "left_out": left,
        "machines_seen": sorted({w for s in on_bus if (w := writer_of(s))})})
