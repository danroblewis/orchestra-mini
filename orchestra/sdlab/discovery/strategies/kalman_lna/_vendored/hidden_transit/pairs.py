# Vendored for kalman_lna from sdlab/discovery/strategies/hidden_transit/pairs.py at commit ec2f9e8 (imports rewritten to the
# vendored copies; otherwise unchanged). Do not edit: re-vendor instead.
"""A pair's part of a found model: which flows a transit stock A -> H -> B would replace, what the
others do to A and B at the measured levels, and the model with the transit stock in their place.

Replaced (the pair's rows are refitted as a whole, so nothing biased by the missing stock stays):
A's losses in proportion to A (to outside, or a move into B), B's arrivals from outside (a constant
or pieces, either sign) and links into B from A, B's drains to outside in proportion to B, and (when
the variant refits it) A's outside constant. Every other flow is kept as found.
"""

from __future__ import annotations

import numpy as np

from orchestra.sdlab.discovery.balance import amounts


def _prop_on(f: dict, s: str) -> bool:
    """A rate in proportion to s alone (proportional, or linear with no constant and s alone)."""
    if f.get("inflow") or f.get("byTime") or f.get("effects") or f.get("uses") or f.get("driver"):
        return False
    r = f.get("rate") or {}
    if r.get("form") == "proportional":
        return r.get("stock") == s
    if r.get("form") == "linear":
        return not r.get("c") and [x.get("stock") for x in r.get("terms") or []] == [s]
    return False


def _outside_const(f: dict, s: str) -> bool:
    """An outside constant of s (into it, or out of it as a negative inflow): a constant or pieces."""
    if f.get("driver") or f.get("effects") or f.get("uses") or f.get("byTime"):
        return False
    if not ((f.get("to") == s and f.get("from") is None) or (f.get("from") == s and f.get("to") is None)):
        return False
    return bool(f.get("inflow")) or (f.get("rate") or {}).get("form") == "constant"


def numbers(f: dict) -> int:
    """How many numbers a flow was fitted with (pieces: a value and a place each)."""
    return 2 * len(f["inflow"]) - 1 if f.get("inflow") else 1


def flow_numbers(f: dict) -> int:
    """How many numbers any found flow carries."""
    if f.get("inflow"):
        return numbers(f)
    rates = [g["rate"] for g in f["byTime"]] if f.get("byTime") else [f.get("rate") or {}]
    n = 0
    for r in rates:
        form = r.get("form")
        n += 2 if form == "gap" else (bool(r.get("c")) + len(r.get("terms") or [])) if form == "linear" else 1
    return n + 2 * len(f.get("effects") or []) + len(f.get("uses") or [])


def at_means(flows: list[dict], stocks: set) -> list[dict]:
    """The flows with the outside constants (pieces) of the stocks held at their mean."""
    out = []
    for f in flows:
        if f.get("inflow") and any(_outside_const(f, s) for s in stocks):
            w = [g["to"] - g["from"] + 1 for g in f["inflow"]]
            m = sum(g["value"] * n for g, n in zip(f["inflow"], w)) / max(1, sum(w))
            f = {k: v for k, v in f.items() if k not in ("inflow", "changes")}
            f["rate"] = {"form": "constant", "value": m}
        out.append(f)
    return out


def split(flows: list[dict], a: str, b: str) -> tuple[list[int], list[int]]:
    """(indexes of the flows the transit replaces, indexes kept)."""
    out = []
    for i, f in enumerate(flows):
        if f.get("from") == a and f.get("to") in (None, b) and _prop_on(f, a):
            out.append(i)
        elif _outside_const(f, b) or _outside_const(f, a):
            out.append(i)
        elif f.get("to") == b and f.get("from") is None and f.get("driver") == a:
            out.append(i)
        elif f.get("from") == b and f.get("to") is None and _prop_on(f, b):
            out.append(i)
    return out, [i for i in range(len(flows)) if i not in set(out)]


def pieces(flows: list[dict], a: str, t: list) -> tuple[np.ndarray | None, list[tuple[int, int]]]:
    """A's outside inflow as the main line cut it: each step's piece (step k moves the levels from
    t[k] to t[k + 1] and counts at t[k + 1]) and the pieces' spans; (None, []) when A has no outside
    constant, one piece when it is not cut."""
    own = [f for f in flows if _outside_const(f, a)]
    if not own:
        return None, []
    tt = np.asarray(t, dtype=float)[1:]
    starts = sorted({int(g["from"]) for f in own for g in f.get("inflow") or []} | {int(t[1]) if len(t) > 1 else int(t[0])})
    starts = [x for x in starts if x <= tt[-1]] or [int(tt[0])]
    seg = np.searchsorted(np.array(starts, dtype=float), tt, side="right") - 1
    seg = np.maximum(seg, 0)
    used = sorted(set(seg.tolist()))
    remap = {j: i for i, j in enumerate(used)}
    seg = np.array([remap[j] for j in seg])
    spans = [(int(starts[j]) if i else int(min(starts[j], tt[0])), int(starts[used[i + 1]] - 1) if i + 1 < len(used) else int(t[-1]))
             for i, j in enumerate(used)]
    return seg, spans


def one_piece(seg: np.ndarray | None, spans: list) -> tuple[np.ndarray | None, list]:
    if seg is None:
        return None, []
    return np.zeros_like(seg), [(spans[0][0], spans[-1][1])]


def candidate(flows: list[dict], a: str, b: str) -> bool:
    """A loses something in proportion to its level (to outside or to B) and B gets something it
    does not lose from a measured stock other than A (from outside, or from A)."""
    loses = any(f.get("from") == a and f.get("to") in (None, b) and _prop_on(f, a) for f in flows)
    gets = any(_outside_const(f, b) or (f.get("to") == b and f.get("from") in (None, a) and (f.get("from") == a or f.get("driver") == a))
               for f in flows)
    return loses and gets


def exogenous(flows: list[dict], keep: list[int], ids: list[str], t: list, L: np.ndarray, stock: str) -> tuple[np.ndarray, np.ndarray]:
    """What the kept flows add to `stock` each step (net) and how much they move through it (gross),
    at the measured levels."""
    kept = [flows[i] for i in keep if stock in (flows[i].get("from"), flows[i].get("to"))
            or any(u.get("stock") == stock for u in flows[i].get("uses") or [])]
    n = L.shape[1] - 1
    if not kept:
        return np.zeros(n), np.zeros(n)
    am = amounts(kept, ids, t, L)
    net, gross = np.zeros(n), np.zeros(n)
    for f, x in zip(kept, am):
        w = (1.0 if f.get("to") == stock else 0.0) - (float(f.get("ratio") or 1) if f.get("from") == stock else 0.0)
        w -= sum(float(u.get("per") or 0) for u in f.get("uses") or [] if u.get("stock") == stock)
        net += w * x
        gross += abs(w) * np.abs(x)
    return net, gross


def build(flows: list[dict], keep: list[int], a: str, b: str, fit: dict, spans: list) -> tuple[list[dict], str]:
    """The model with A -> H -> B in place of the replaced flows, and H's id."""
    h = f"{a}~{b}"
    out = [flows[i] for i in keep]
    prop = lambda s, k: {"form": "proportional", "stock": s, "k": float(k)}  # noqa: E731
    vals = [max(0.0, float(v)) for v in fit["values"]]
    if len(vals) == 1:
        out.append({"from": None, "to": a, "ratio": 1, "rate": {"form": "constant", "value": vals[0]}})
    elif vals:
        w = [hi - lo + 1 for lo, hi in spans]
        out.append({"from": None, "to": a, "ratio": 1, "rate": {"form": "constant", "value": float(np.dot(vals, w) / sum(w))},
                    "inflow": [{"from": lo, "to": hi, "value": v} for (lo, hi), v in zip(spans, vals)], "changes": len(vals) - 1})
    out.append({"from": a, "to": h, "ratio": 1, "rate": prop(a, fit["kin"]), "hidden": h})
    if fit["kA"] - fit["kin"] > 1e-3 * fit["kA"]:
        out.append({"from": a, "to": None, "ratio": 1, "rate": prop(a, fit["kA"] - fit["kin"])})
    out.append({"from": h, "to": b, "ratio": 1, "rate": prop(h, fit["d"]), "hidden": h})
    if fit["kb"] > 1e-6:
        out.append({"from": b, "to": None, "ratio": 1, "rate": prop(b, fit["kb"])})
    if abs(fit["cB"]) > 1e-9:
        out.append({"from": None, "to": b, "ratio": 1, "rate": {"form": "constant", "value": fit["cB"]}} if fit["cB"] > 0
                   else {"from": b, "to": None, "ratio": 1, "rate": {"form": "constant", "value": -fit["cB"]}})
    return out, h
