# Vendored for kalman_lna from sdlab/discovery/strategies/effect_hats/translate.py at commit ec2f9e8 (imports rewritten to the
# vendored copies; otherwise unchanged). Do not edit: re-vendor instead.
"""Flows read off a matrix that had virtual stocks (level: base x f(C)) -> flows with table effects.

A rate's term k x V on a virtual stock V is a flow base x k f(C) between the same ends; a drain of one
stock and a link into another (of the same unit) on the same V are one move; a plain flow k1 x base
between the same ends as an effect flow is folded into its curve (the tents sum to 1). A virtual stock
in an era segment or an outside inflow piece cannot be read back: the candidate is dropped (None).
"""

from __future__ import annotations

import numpy as np

from orchestra.sdlab.discovery.strategies.kalman_lna._vendored.effect_hats.joint import _base, _write


def hat_flow(frm, to, base, stock, kn, weights, prev, idx, extra=None) -> dict:
    rate = {"form": "constant", "value": 1.0} if base is None else {"form": "proportional", "stock": base, "k": 1.0}
    # driver: what the flow follows (a constant-rate effect flow is not an outside inflow to be replaced)
    f = {**(extra or {}), "from": frm, "to": to, "ratio": 1, "rate": rate, "driver": base or stock,
         "hat": {"stock": stock, "knots": [float(x) for x in kn], "weights": [float(w) for w in weights]}}
    f.pop("inflow", None)
    f.pop("byTime", None)
    return _write(f, "hat", np.asarray(weights, dtype=float), prev, idx)


def _split(rate: dict, meta: dict) -> tuple[dict | None, dict[str, float]] | None:
    """(the rate without its virtual terms, or None if nothing is left; {virtual: k})."""
    form = rate.get("form")
    if form == "proportional":
        if rate.get("stock") in meta:
            return None, {rate["stock"]: float(rate.get("k") or 0)}
        return rate, {}
    if form == "linear":
        terms = rate.get("terms") or []
        v = {}
        for x in terms:
            if x.get("stock") in meta:
                v[x["stock"]] = v.get(x["stock"], 0.0) + float(x.get("k") or 0)
        rest = [x for x in terms if x.get("stock") not in meta]
        if not rest and not rate.get("c"):
            return None, v
        return {**rate, "terms": rest}, v
    if form == "gap" and rate.get("stock") in meta:
        return None
    return rate, {}


def translate(flows: list[dict], meta: dict, stocks: list[dict], prev: np.ndarray, ids: list[str], coflows: bool = True,
              pair: bool = True) -> list[dict] | None:
    unit = {s["id"]: s.get("unit") or "" for s in stocks}
    idx = {s: i for i, s in enumerate(ids)}
    plain, eff = [], {}                   # eff[v] = [[from, to, k]]
    for f in flows:
        if f.get("from") in meta or f.get("to") in meta:
            continue
        segs = f.get("byTime") or []
        if any(_split(g.get("rate") or {}, meta) is None or _split(g.get("rate") or {}, meta)[1] for g in segs):
            return None
        sp = _split(f.get("rate") or {}, meta)
        if sp is None:
            return None
        rest, vs = sp
        if vs and (f.get("inflow") or segs or f.get("effects") or f.get("uses")):
            return None
        for v, k in vs.items():
            eff.setdefault(v, []).append([f.get("from"), f.get("to"), k])
        if rest is not None:
            plain.append({**f, "rate": rest, "rates": [rest]} if vs else f)
    out = list(plain)
    for v, items in eff.items():
        m = meta[v]
        outs = [x for x in items if x[0] is not None and x[1] is None and x[2] > 0]
        ins = [x for x in items if x[0] is None and x[1] is not None and x[2] > 0]
        moves = [x for x in items if x not in outs and x not in ins]
        for o in outs if pair else []:
            for i in sorted(ins, key=lambda i: -i[2]):
                if unit.get(i[1]) == unit.get(o[0]) and o[2] > 0 and i[2] > 0:
                    k = min(o[2], i[2])
                    moves.append([o[0], i[1], k])
                    o[2] -= k
                    i[2] -= k
        # what is left draining a third stock on the same curve as a move is that move using it up
        # as it goes (a co-flow: per unit moved, the drain's number over the move's)
        uses: dict[int, list] = {}
        if coflows and moves:
            lead = max(range(len(moves)), key=lambda q: moves[q][2])
            for o in outs:
                if o[2] > 1e-12 and o[0] not in (moves[lead][0], moves[lead][1]) and moves[lead][2] > 0:
                    uses.setdefault(lead, []).append({"stock": o[0], "per": float(o[2] / moves[lead][2])})
                    o[2] = 0.0
        for q, (frm, to, k) in enumerate(moves + outs + ins):
            if abs(k) <= 1e-12:
                continue
            extra = {"uses": uses[q]} if q in uses else None
            out.append(hat_flow(frm, to, m["base"], m["stock"], m["knots"], k * np.asarray(m["weights"]), prev, idx, extra))
    return fold(out)


def fold(flows: list[dict]) -> list[dict]:
    """A plain flow k x base between the same ends as an effect flow on that base goes into its curve;
    two effect flows on the same base and curve stock between the same ends are one."""
    out = list(flows)
    changed = True
    while changed:
        changed = False
        for e in [f for f in out if f.get("hat")]:
            be = _base(e["rate"])
            for p in out:
                if p is e or p.get("from") != e.get("from") or p.get("to") != e.get("to") or p.get("inflow") or p.get("byTime") or p.get("uses"):
                    continue
                if p.get("hat"):
                    same = p["hat"]["stock"] == e["hat"]["stock"] and p["hat"]["knots"] == e["hat"]["knots"] and _base(p["rate"]) == be
                    if not same:
                        continue
                    w = np.asarray(e["hat"]["weights"]) + np.asarray(p["hat"]["weights"])
                elif not p.get("effects") and _base(p.get("rate") or {}) == be:
                    r = p["rate"]
                    k = float(r.get("value") if r.get("form") == "constant" else r.get("k") if r.get("form") == "proportional" else r["terms"][0]["k"])
                    w = np.asarray(e["hat"]["weights"]) + k
                else:
                    continue
                new = {**e, "hat": {**e["hat"], "weights": [float(x) for x in w]}}
                out = [new if f is e else f for f in out if f is not p]
                changed = True
                break
            if changed:
                break
    return out
