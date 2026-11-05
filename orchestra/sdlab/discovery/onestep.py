"""A whole model's one-step fit, priced by its size: an information criterion over every stock.

Judging by runs on the data feeds a model its own estimated outside inflows, so a model whose
inflow pieces mimic dynamics it lacks can look as good as one that has them. Here every number
costs log T, the pieces included: the model with effects that needs fewer pieces wins only if
it fits the steps as well.
"""

from __future__ import annotations

import math

import numpy as np

from .simulate import Model, levels_array


def size(flows: list[dict]) -> int:
    p = 0
    for f in flows:
        if f.get("inflow"):
            p += len(f["inflow"])
            continue
        rates = [g["rate"] for g in f["byTime"]] if f.get("byTime") else [f.get("rate") or {}]
        p += len({repr(r) for r in rates})
        p += 2 * len(f.get("effects") or [])
    return p


def bic(flows: list[dict], ids: list[str], t: list, levels: dict, inputs: dict | None = None) -> float:
    """inputs (optional): measured inputs, {name: value per step aligned with t}, that the flows may read."""
    A = levels_array(levels, ids)
    T = A.shape[1] - 1
    if T < 2:
        return float("inf")
    nxt = Model(flows, ids, inputs={"t": t, "values": inputs} if inputs else None).step(A[:, :-1], np.asarray(t, dtype=float)[1:])
    rss = ((nxt - A[:, 1:]) ** 2).sum(axis=1)
    return float(sum(T * math.log(max(r, 1e-9) / T) for r in rss) + size(flows) * math.log(T))
