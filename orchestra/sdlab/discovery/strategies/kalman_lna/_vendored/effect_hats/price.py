# Vendored for kalman_lna from sdlab/discovery/strategies/effect_hats/price.py at commit ec2f9e8 (imports rewritten to the
# vendored copies; otherwise unchanged). Do not edit: re-vendor instead.
"""A model's price (one-step BIC with pooled inflow values counted once), and pruning by it.

price() is onestep.bic with size() below: every row's one-step fit, every number costs log T; an
outside inflow costs one number per distinct value and one per change (pooled pieces share a value),
an effect flow's table one number per knot.
"""

from __future__ import annotations

import math

import numpy as np

from orchestra.sdlab.discovery.simulate import Model, levels_array
from orchestra.sdlab.discovery.strategies.kalman_lna._vendored.effect_hats.joint import _pools, refit


def amounts(flows: list[dict], ids: list[str], run: dict) -> np.ndarray:
    """Each flow's mean amount per step on the run's states (what it moves into or out of a stock)."""
    A = levels_array(run["levels"], ids)
    tt = np.asarray(run["t"], dtype=float)[1:]
    out = []
    for f in flows:
        with np.errstate(all="ignore"):
            d = Model([f], ids).step(A[:, :-1], tt) - A[:, :-1]
        out.append(float(np.nan_to_num(np.abs(d).max(axis=0)).mean()))
    return np.array(out)


def prune(flows: list[dict], ids: list[str], run: dict, opts: dict | None = None, score=None, small: float = 0.25) -> list[dict]:
    """Small flows dropped one at a time, refitting the rest jointly, while the model's price falls
    (score(flows), lower is better: by default price()). Only a flow moving less than `small` x the
    model's median flow amount may go: a large term the price would drop (a gap's feedback, say) is
    left, since inflow values would take over its work and the model would not behave."""
    score = score or (lambda fl: price(fl, ids, run))
    cur = flows
    best = score(cur)
    while len(cur) > 1:
        size = amounts(cur, ids, run)
        cutoff = small * float(np.median(size)) if len(size) else 0.0
        trial = None
        for i in range(len(cur)):
            if size[i] >= cutoff:
                continue
            fl, _ = refit(cur[:i] + cur[i + 1:], ids, run, opts)
            s = score(fl)
            if s < best and (trial is None or s < trial[0]):
                trial = (s, fl)
        if trial is None:
            break
        best, cur = trial
    return cur

def size(flows: list[dict]) -> int:
    """onestep.size, with an outside inflow priced by its values (pooled pieces share one) and its changes."""
    p = 0
    for f in flows:
        if f.get("inflow"):
            p += len(_pools(f)) + len(f["inflow"]) - 1      # a number per value, and per change
            continue
        rates = [g["rate"] for g in f["byTime"]] if f.get("byTime") else [f.get("rate") or {}]
        p += len({repr(r) for r in rates})
        p += 2 * len(f.get("effects") or []) if not f.get("hat") else len(f["hat"]["knots"]) - 1
    return p


def price(flows: list[dict], ids: list[str], run: dict) -> float:
    """onestep.bic with size() above: every row's one-step fit, every number costs log T."""
    A = levels_array(run["levels"], ids)
    T = A.shape[1] - 1
    if T < 2:
        return float("inf")
    nxt = Model(flows, ids).step(A[:, :-1], np.asarray(run["t"], dtype=float)[1:])
    rss = ((nxt - A[:, 1:]) ** 2).sum(axis=1)
    return float(sum(T * math.log(max(r, 1e-9) / T) for r in rss) + size(flows) * math.log(T))

