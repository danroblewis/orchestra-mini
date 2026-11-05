"""Discovery as the page asks for it: one request in, one JSON-safe result out.

    discover({"stocks": [...], "run": {"t": [...], "levels": {...}}, "eras": [{start, label}],
              "known": {stock: [change per step]}, "inputs"?: {name: [value per step]}, "strategy": "matrix"})

`known` is what the page already knows about each step (its measured flows, and what the model's own
equations predict): the strategy explains the rest. The result is the strategy's, plus `unexplained`
(per stock, the steady drift and the scatter the found flows, and the known changes, leave), with
every number a plain float and no inf or nan (JSON has neither).
"""

from __future__ import annotations

import math
import time

import numpy as np

from .matrix import known_flows
from .simulate import Model, levels_array
from .strategies import load

DEFAULT_STRATEGY = "matrix"


def jsonable(x):
    """x with numpy numbers and arrays made plain, and inf and nan made None."""
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        v = float(x)
        return v if math.isfinite(v) else None
    return x


def unexplained(flows: list[dict], stocks: list[dict], run: dict, inputs: dict | None = None) -> dict:
    """Per stock, what one step of the flows leaves unexplained: its mean (drift) and spread
    (wobble), and whether the drift is clearly more than the scatter (persistent)."""
    ids = [s["id"] for s in stocks]
    A = levels_array(run["levels"], ids)
    n = A.shape[1] if A.ndim == 2 else 0
    if n < 3 or not ids:
        return {}
    t = np.asarray(run["t"], dtype=float)
    pred = Model(flows, ids, inputs={"t": run["t"], "values": inputs} if inputs else None).step(A[:, :-1], t[1:])
    res = A[:, 1:] - pred
    out = {}
    for k, i in enumerate(ids):
        m, sd = float(res[k].mean()), float(res[k].std())
        out[i] = {"drift": m, "wobble": sd, "persistent": abs(m) > 2 * sd / math.sqrt(n - 1) and abs(m) > 0.02}
    return out


def discover(request: dict) -> dict:
    """Run a discovery strategy on what the page sends; the result is JSON-safe."""
    began = time.perf_counter()
    stocks = [{k: s[k] for k in ("id", "unit", "final", "group") if k in s} for s in request["stocks"]]
    run = {"t": list(request["run"]["t"]), "levels": dict(request["run"]["levels"])}
    eras = [{"start": e["start"], "label": e.get("label")} for e in request.get("eras") or [] if e and e.get("start") is not None]
    opts: dict = {"eras": eras if len(eras) > 1 else None}
    known = {k: v for k, v in (request.get("known") or {}).items() if v is not None}
    if known:
        opts["known"] = known
    # measured inputs (outside drivers given as data, e.g. a pump setting): {name: [value per step]}
    # aligned with run.t; a strategy may read them like levels, never as the truth
    inputs = {k: list(v) for k, v in (request.get("inputs") or {}).items() if v is not None}
    if inputs:
        opts["inputs"] = inputs
    strategy = request.get("strategy") or DEFAULT_STRATEGY
    r = load(strategy)(stocks, run, opts)
    flows = r.get("flows") or []
    extra = known_flows(known, [s["id"] for s in stocks], run["t"]) if known else []
    out = {k: v for k, v in r.items() if k not in ("matrix", "eras")}
    if r.get("eras"):
        out["labels"] = r["eras"]                    # the configurations (era labels) the rates follow
    out.update(flows=flows, strategy=strategy, unexplained=unexplained(flows + extra, stocks, run, inputs or None),
               hidden=r.get("hidden") or [], computeMs=round(1000 * (time.perf_counter() - began)))
    return jsonable(out)
