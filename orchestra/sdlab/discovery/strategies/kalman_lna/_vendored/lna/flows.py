# Vendored for kalman_lna from sdlab/discovery/strategies/lna/flows.py at commit ec2f9e8 (imports rewritten to the
# vendored copies; otherwise unchanged). Do not edit: re-vendor instead.
"""Atoms with their numbers -> the model's flows (the shape read_flows gives)."""

from __future__ import annotations

import numpy as np

from orchestra.sdlab.discovery.strategies.kalman_lna._vendored.lna.atoms import Atom


def r4(v: float) -> float:
    a = abs(v)
    return round(float(v), 2 if a >= 10 else 4 if a >= 1 else 6)


def _flow(src, dst, rate, per_step, **extra) -> dict:
    return {"from": src, "to": dst, "ratio": 1, "rate": rate, "rates": [rate], "perStep": per_step, "nearly": [], "bothWays": False, **extra}


def to_flows(atoms: list[Atom], theta: np.ndarray, ids: list[str], prev: np.ndarray, at, labels: list | None = None, spans: list | None = None, bucket: list | None = None) -> list[dict]:
    """at(k): the time of step k's end. Inflow atoms over spans of steps become one flow with pieces;
    atoms per era become one flow whose rate follows the era (byTime)."""
    means = prev.mean(axis=1)
    groups: dict[tuple, list[tuple[Atom, float]]] = {}
    for a, th in zip(atoms, theta):
        if th > 0:
            groups.setdefault(a.key(), []).append((a, float(th)))
    out = []
    T = prev.shape[1]
    # a goal gap (b's outside inflow c - k x a) is one flow with b's inflow when that is one constant;
    # otherwise it is written as the drain it equals on average (the simulator floors each flow at 0)
    gaps: dict[int, list[tuple[int, float]]] = {}
    for key, items in list(groups.items()):
        if key[0] != "gap":
            continue
        inflow = groups.get(("in", None, key[2], None, ()), [])
        if len(inflow) == 1 and inflow[0][0].span is None and inflow[0][0].era is None and all(a.era is None for a, _ in items):
            gaps.setdefault(key[2], []).append((key[3], sum(th for _, th in items)))
        else:
            kind = "out" if key[3] == key[2] else "drain"
            groups.setdefault((kind, key[2], None, key[3], ()), []).extend((a.with_(kind=kind, src=key[2], dst=None), th) for a, th in items)
        del groups[key]
    for key, items in groups.items():
        kind, src, dst, driver, uses = key
        name = lambda i: None if i is None else ids[i]  # noqa: E731
        extra = {"uses": [{"stock": ids[c], "per": round(per, 4)} for c, per in uses]} if uses else {}
        if kind == "in" and dst in gaps:
            c = items[0][1]
            terms = [{"stock": ids[j], "k": r4(-k)} for j, k in gaps[dst]]
            per_step = c - sum(k * means[j] for j, k in gaps[dst])
            out.append(_flow(None, name(dst), {"form": "linear", "c": r4(c), "terms": terms}, per_step, note="goal gap"))
            continue
        if kind == "in" and all(a.era is None for a, _ in items):
            pieces = sorted(((a.span or (0, T)), th) for a, th in items)
            if len(pieces) > 1 or pieces[0][0] != (0, T):
                # pieces cover every step (a step no piece covers has none)
                inflow = [{"from": at(p[0]), "to": at(p[1] - 1), "value": r4(v)} for p, v in pieces]
                W = sum(p[1] - p[0] for p, _ in pieces)
                m = sum(v * (p[1] - p[0]) for p, v in pieces) / max(1, W)
                out.append(_flow(None, name(dst), {"form": "constant", "value": r4(m)}, m, inflow=inflow, changes=len(pieces) - 1))
            else:
                v = pieces[0][1]
                out.append(_flow(None, name(dst), {"form": "constant", "value": r4(v)}, v))
            continue

        def rate(k):
            if kind == "in":
                return {"form": "constant", "value": r4(k)}
            if kind in ("out", "move"):
                return {"form": "proportional", "stock": ids[driver], "k": r4(k)}
            return {"form": "linear", "c": 0, "terms": [{"stock": ids[driver], "k": r4(k)}]}

        ends = {"in": (None, dst), "out": (src, None), "move": (src, dst), "link": (None, dst), "grow": (None, dst), "drain": (src, None)}[kind]
        if kind in ("link", "grow", "drain"):
            extra["driver"] = ids[driver]
        eras = [a.era for a, _ in items]
        scale = 1.0 if driver is None else means[driver]
        if labels and len(labels) > 1 and any(e is not None for e in eras):
            shared = sum(th for a, th in items if a.era is None)
            bucket = bucket or list(range(len(labels)))
            by_label = [shared + sum(th for a, th in items if a.era == bucket[c]) for c in range(len(labels))]
            rates = [rate(k) for k in by_label]
            by_time = [{"from": sp["from"], "to": sp["to"], "rate": rates[labels.index(sp["label"])]} for sp in spans]
            k = sum(by_label) / len(by_label)
            out.append({**_flow(name(ends[0]), name(ends[1]), rate(k), k * scale, **extra), "byTime": by_time, "rates": rates})
        else:
            k = sum(th for _, th in items)
            out.append(_flow(name(ends[0]), name(ends[1]), rate(k), k * scale, **extra))
    return out
