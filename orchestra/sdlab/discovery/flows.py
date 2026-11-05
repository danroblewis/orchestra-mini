"""Reading stock-and-flow structure off the matrix.

  - a move A -> B at rate k is -k in A's own entry and +k in B's entry for A;
  - a link (B's arrivals follow A's level, nothing leaves A) is +k for B with no matching -k;
  - what A loses beyond what other stocks are seen to receive goes to outside;
  - a gain in a stock of another unit is a link (things become other things only with a ratio).
"""

from __future__ import annotations

import numpy as np


def r4(v: float) -> float:
    a = abs(v)
    return round(float(v), 1 if a >= 10 else 3 if a >= 1 else 4)


def read_flows(stocks: list[dict], rows: list[dict], prev: np.ndarray, tol: float = 0.003, kinds: dict | None = None,
               extra: list[dict] | None = None, prevx: np.ndarray | None = None) -> list[dict]:
    """rows: one per stock, {"id", "coef": {stock: number}, "inflow": [{from, to, value}]}. kinds:
    (source, receiver) -> "move" | "link", where the shared noise has decided (a link takes nothing
    from the source's drain).

    extra (optional): delayed copies of a stock's level standing for unmeasured stocks ({"id",
    "source", "delay"}; their levels in prevx, after the stocks'). A row's positive number k on a
    copy of A delayed by d is an unmeasured stock H fed from A at k x A and drained into the row's
    stock at d x H (a move out of A as far as A's drain allows, a link beyond it); the flow out
    of H carries "inferred": {"id": H, "copy": copy id, "scale": k / d} (H's level = scale x copy)."""
    kinds = kinds or {}
    ids = [s["id"] for s in stocks]
    B = {r["id"]: r["coef"] for r in rows}
    unit = {s["id"]: s.get("unit") or "" for s in stocks}
    final = {s["id"] for s in stocks if s.get("final")}
    means = {i: float(prev[k].mean()) if prev.shape[1] else 0.0 for k, i in enumerate(ids)}
    flows: list[dict] = []

    def flow(src, dst, rate, **extra):
        form = rate["form"]
        k = rate["k"] if form == "proportional" else rate["terms"][0]["k"] if form == "linear" else 0
        stock = rate.get("stock") if form == "proportional" else rate["terms"][0]["stock"] if form == "linear" else None
        per_step = rate["value"] if form == "constant" else k * means[stock]
        flows.append({"from": src, "to": dst, "ratio": 1, "rate": rate, "rates": [rate], "perStep": per_step, "nearly": [], "bothWays": False, **extra})

    def linear(stock, k):
        return {"form": "linear", "c": 0, "terms": [{"stock": stock, "k": r4(k)}]}

    extra = extra or []
    for i, e in enumerate(extra):
        means[e["id"]] = float(prevx[len(ids) + i].mean()) if prevx is not None and prevx.shape[1] else 0.0
    for a in ids:
        drain = 0.0 if a in final else -(B[a].get(a) or 0.0)
        gains = sorted(((s, B[s][a]) for s in ids if s != a and (B[s].get(a) or 0) > tol), key=lambda x: -x[1])
        left = max(0.0, drain)
        hidden = sorted(((s, B[s][e["id"]], e) for e in extra if e["source"] == a for s in ids if s != a and (B[s].get(e["id"]) or 0) > tol), key=lambda x: -x[1])
        for s, k, e in hidden:
            h, d = f"{e['id']}>{s}", e["delay"]
            means[h] = k / d * means[e["id"]]
            move = max(0.0, min(k, left)) if unit[s] == unit[a] else 0.0
            if move > tol:
                flow(a, h, {"form": "proportional", "stock": a, "k": r4(move)}, hidden=h)
                left -= move
            if k - move > tol:
                flow(None, h, linear(a, k - move), driver=a, hidden=h)
            flow(h, s, {"form": "proportional", "stock": h, "k": r4(d)}, hidden=h, inferred={"id": h, "copy": e["id"], "scale": k / d})
        for s, k in gains:
            move = max(0.0, min(k, left)) if unit[s] == unit[a] and kinds.get((a, s)) != "link" else 0.0
            if move > tol:
                flow(a, s, {"form": "proportional", "stock": a, "k": r4(move)})
                left -= move
            if k - move > tol:
                flow(None, s, linear(a, k - move), driver=a)
        if left > tol:
            flow(a, None, {"form": "proportional", "stock": a, "k": r4(left)})
        if (B[a].get(a) or 0) > tol:
            flow(None, a, linear(a, B[a][a]), driver=a, note="grows with itself")
        for s in ids:
            if s != a and (B[s].get(a) or 0) < -tol:
                flow(s, None, linear(a, -B[s][a]), driver=a, note=f"{s} drains faster when {a} is high")
    for r in rows:
        pieces = r["inflow"]
        if not pieces:
            continue
        W = sum(g["to"] - g["from"] + 1 for g in pieces)
        m = sum(g["value"] * (g["to"] - g["from"] + 1) for g in pieces) / max(1, W)
        if abs(m) <= tol:
            continue
        rate = {"form": "constant", "value": r4(abs(m))}
        if m > 0:
            flow(None, r["id"], rate, inflow=pieces, changes=len(pieces) - 1)
        else:
            flow(r["id"], None, rate, inflow=[{**g, "value": -g["value"]} for g in pieces], changes=len(pieces) - 1)
    return flows


def _key(f: dict) -> str:
    return f"{f['from']}>{f['to']}>{f.get('driver') or ''}>{f['rate']['form']}" + (f">{f['floor']}" if f.get("floor") else "")


def _zero(f: dict) -> dict:
    r = f["rate"]
    if r["form"] == "proportional":
        return {**r, "k": 0}
    if r["form"] == "linear":
        return {**r, "c": 0, "terms": [{**x, "k": 0} for x in r["terms"]]}
    return {"form": "constant", "value": 0}


def merge_by_label(per_label: list[list[dict]], labels: list[str], spans: list[dict]) -> list[dict]:
    """One configuration's flows per label -> flows whose rate follows the configuration in force
    (byTime); the outside inflows are shared, taken from the first."""
    merged: dict[str, dict] = {}
    for flows in per_label:
        for f in flows:
            if not f.get("inflow") and _key(f) not in merged:
                merged[_key(f)] = f
    out = []
    for key, f in merged.items():
        by_label = [next((g["rate"] for g in flows if not g.get("inflow") and _key(g) == key), None) or _zero(f) for flows in per_label]
        by_time = [{"from": sp["from"], "to": sp["to"], "rate": by_label[labels.index(sp["label"])]} for sp in spans]
        out.append({**f, "byTime": by_time, "rates": by_label})
    out.extend(f for f in per_label[0] if f.get("inflow"))
    return out
