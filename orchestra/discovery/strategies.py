"""The strategies. ``conservation`` (``discover``) always runs; with the ``sd`` extra installed
(numpy, scipy: ``pip install 'orchestra[sd]'``), SD Lab's ``kalman_lna`` (``orchestra/sdlab``) then
explains what the counters leave unexplained. Adapted from the old Orchestra's ``sd/strategies.py``
and ``sd/rates.py``, without net-flow stocks, held-out rivals, inputs and hidden stocks.

``refine`` hands kalman_lna the stocks the counters do not balance exactly (in ``FLAG_EXACT`` of
the steps), grouped by the measured flows that join them, with their levels and what the counters
already say of each step (``known``). Its answer is read by its size (``explains_residual``): its flows describe either what the known changes leave, and each is kept
as a flow no counter measures, or the whole change, and then a flow between the same ends as a
counter's is that counter's (whose fit stays) and the others are kept only where the counters
balance their stock with R² under ``BALANCED_R2``. Its rates are per step, and are scaled to the
model's time unit. A flow to or from a stock it invents (a hidden stock) is left out, with why."""

from __future__ import annotations

import math
import time

from .model import Equation, Flow, Model

STRATEGY = "kalman_lna"
FLAG_EXACT = 0.97
BALANCED_R2 = 0.98
DEADLINE_S = 60.0
UNITS = {"second": 1.0, "minute": 60.0, "hour": 3600.0}


def installed() -> bool:
    try:
        import numpy  # noqa: F401
        import scipy  # noqa: F401
        return True
    except ImportError:
        return False


def scale(rate: dict, f: float) -> dict:
    """An SD Lab rate per step, as a rate per time unit (``f`` steps per unit)."""
    r = dict(rate)
    form = r.get("form")
    if form == "constant" and "value" in r:
        r["value"] = r["value"] * f
    elif form == "proportional" and "k" in r:
        r["k"] = r["k"] * f
    elif form == "linear":
        r["c"] = r.get("c", 0.0) * f
        r["terms"] = [{**t, "k": t.get("k", 0.0) * f} for t in r.get("terms") or []]
    elif form == "sum":
        r["parts"] = [scale(p, f) for p in r.get("parts") or []]
    return r


def reads(rate: dict) -> list[str]:
    out = [rate["stock"]] if rate.get("stock") else []
    out += [t["stock"] for t in rate.get("terms") or [] if t.get("stock")]
    out += [e["stock"] for e in rate.get("effects") or [] if e.get("stock")]
    for p in rate.get("parts") or []:
        out += reads(p)
    return list(dict.fromkeys(out))


def numbers(rate: dict) -> dict[str, float]:
    out = {("c" if k == "value" else k): float(v) for k, v in rate.items() if isinstance(v, int | float) and not isinstance(v, bool)}
    for i, t in enumerate(rate.get("terms") or []):
        if isinstance(t.get("k"), int | float):
            out[f"k{i + 1}"] = float(t["k"])
    return out


def combine(rates: list[dict]) -> dict:
    """Several rates between the same two ends (one per term SD Lab found) as one."""
    if len(rates) == 1:
        return rates[0]
    if all(r.get("form") in ("constant", "proportional", "linear") for r in rates):
        c, terms = 0.0, []
        for r in rates:
            if r["form"] == "constant":
                c += float(r.get("value", 0.0))
            elif r["form"] == "proportional":
                terms.append({"stock": r.get("stock"), "k": float(r.get("k", 0.0))})
            else:
                c += float(r.get("c", 0.0))
                terms += [dict(t) for t in r.get("terms") or []]
        return {"form": "linear", "c": c, "terms": terms, "combined": len(rates)}
    return {"form": "sum", "parts": [dict(r) for r in rates], "combined": len(rates)}


def explains_residual(flows: list[dict], levels: dict[str, list[float]], known: dict[str, list[float]]) -> bool:
    """Whether the flows describe what the known changes leave, or the whole change: by size, the
    flows' movement per step against the level's moves the counters leave unexplained and against
    all its moves (whichever it is nearer, by ratio). The old Orchestra asked SD Lab's own residual
    (``api.unexplained``), which on a queue whose counters miss one outflow took the whole change
    for the residual."""
    if not flows:
        return True
    moved = sum(abs(x.get("perStep") or 0.0) for x in flows)
    n = max(1, sum(len(v) - 1 for v in levels.values()))
    total = sum(abs(b - a) + abs(k) for sid, v in levels.items() for a, b, k in zip(v, v[1:], known.get(sid, [0.0])[1:])) / n
    left = sum(abs((b - a) - k) for sid, v in levels.items() for a, b, k in zip(v, v[1:], known.get(sid, [0.0])[1:])) / n
    near = lambda x, y: abs(math.log(max(x, 1e-9) / max(y, 1e-9)))  # noqa: E731
    return near(moved, left) < near(moved, total)


def _groups(model: Model, ids: list[str]) -> list[list[str]]:
    root = {s.id: s.id for s in model.stocks}

    def find(x):
        while root[x] != x:
            x = root[x]
        return x
    for f in model.flows:
        if f.src in root and f.dst in root:
            root[find(f.src)] = find(f.dst)
    out: dict[str, list[str]] = {}
    for sid in ids:
        out.setdefault(find(sid), []).append(sid)
    return list(out.values())


def refine(model: Model, series: list, *, start: float, end: float, step: float) -> Model:
    if not installed():
        model.discovery["sdlab"] = f"{STRATEGY} not installed (pip install 'orchestra[sd]'): conservation only"
        return model
    given = [s for s in model.stocks if (s.evidence.get("exact") or 0.0) < FLAG_EXACT]
    if not given:
        model.discovery["sdlab"] = f"{STRATEGY} not run: the counters balance every stock exactly"
        return model
    from ..sdlab.discovery.api import jsonable
    from ..sdlab.discovery.strategies import load

    by_key = {s.key: s for s in series}
    began, f = time.perf_counter(), UNITS[model.time_unit] / step
    has = {(fl.src, fl.dst) for fl in model.flows}
    added, dropped, taken = [], [], {fl.id for fl in model.flows}
    for ids in _groups(model, [s.id for s in given]):
        stocks = [model.stock(i) for i in ids]
        levels, known = {}, {}
        for s in stocks:
            lv = by_key.get(s.evidence.get("key"))
            terms = [(by_key.get(b["counter"]), b["sign"]) for b in s.evidence.get("balance") or []]
            if lv is None or any(c is None for c, _ in terms):
                continue
            levels[s.id] = lv.values
            known[s.id] = [0.0] + [float(sum(sign * c.deltas[i] for c, sign in terms)) for i in range(len(lv.values) - 1)]
        if not levels:
            continue
        n = len(next(iter(levels.values())))
        req = [{"id": i, "unit": "items"} for i in levels]
        run = {"t": list(range(n)), "levels": levels}
        opts = {"eras": None, "known": known, "kalman_budget": DEADLINE_S / 2, "budget": DEADLINE_S / 2, "cap": DEADLINE_S / 4}
        answer = jsonable(load(STRATEGY)(req, run, opts))
        mine = [x for x in answer.get("flows") or [] if all(e is None or e in levels for e in (x.get("from"), x.get("to")))]
        dropped += [{"from": x.get("from"), "to": x.get("to"), "why": "it moves a stock no series measures"}
                    for x in answer.get("flows") or [] if x not in mine]
        residual = explains_residual(mine, levels, known)
        ends: dict[tuple, list[dict]] = {}
        for x in mine:
            ends.setdefault((x.get("from"), x.get("to")), []).append(x)
        for (src, dst), xs in ends.items():
            if not residual and (src, dst) in has:
                continue                                        # a counter measures it: its own fit stays
            r2 = max((model.stock(e).evidence.get("r2") or 0.0) for e in (src, dst) if e)
            if not residual and r2 >= BALANCED_R2:
                dropped.append({"from": src, "to": dst, "why": f"its stock's counters balance it (R² {r2:.2f}): they are its change"})
                continue
            rate = scale(combine([x.get("rate") or {} for x in xs]), f)
            fid = f"unmeasured:{src or 'outside'}>{dst or 'outside'}"
            if fid in taken:
                continue
            taken.add(fid)
            eq = Equation(rate.get("form", "unknown"), numbers(rate), "discovered", f"sdlab:{STRATEGY}", (start, end),
                          {"per_step": float(sum(x.get("perStep") or 0.0 for x in xs))}, reads(rate), rate)
            added.append(Flow(fid, f"{src or 'outside'} → {dst or 'outside'} (no counter measures it)", src, dst, None, [eq],
                              {"machines": {"hypotheses": [], "why": "no counter measures it, so no machine's events can be compared with it"},
                               "found_by": STRATEGY, "describes": "the residual" if residual else "the whole change"}))
    model.flows += added
    model.discovery.update(strategy=f"conservation+{STRATEGY}", sdlab={
        "handed": [s.id for s in given], "added": [fl.id for fl in added], "dropped": dropped,
        "seconds": round(time.perf_counter() - began, 2)})
    return model
