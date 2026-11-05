# Vendored for kalman_lna from sdlab/discovery/strategies/lna_stack/__init__.py at commit ec2f9e8 (imports rewritten to the
# vendored copies; otherwise unchanged). Do not edit: re-vendor instead.
"""lna_stack: lna with the stages built when past ideas were revisited on the ladder
(docs/research/revisit.md) that are recommended, in one model: pulses found first and taken out
(lna_pulses), moves scaled by a third stock (lna_products), inflow pieces as recurring levels
(lna_levels), and, on noise-free runs, ties written the way SD models write them (gap_prior: goal
gaps and co-flows, kept when they behave as the original) and pulses re-measured on the matrix's
model. Each stage is its own package's hooks on lna (lna/run.py HOOKS), unchanged, so this strategy shows
whether they compose; lna's data, stride, measured inputs and budget are lna's own (one budget for
all of its searches). lna_loops is left out: its passes are on cells the data cannot decide, and it can make a
noisy chain behave much worse (revisit.md).
"""

from __future__ import annotations

import math

import numpy as np

from orchestra.sdlab.discovery.identify import identify as _matrix
from orchestra.sdlab.discovery.strategies.kalman_lna._vendored.gap_prior import conventions
from orchestra.sdlab.discovery.strategies.kalman_lna._vendored.impulses import _flows as pulse_flows
from orchestra.sdlab.discovery.strategies.kalman_lna._vendored.impulses import find_pulses
from orchestra.sdlab.discovery.strategies.kalman_lna._vendored.lna import GUARD, judge
from orchestra.sdlab.discovery.strategies.kalman_lna._vendored.lna_pulses import _not_injected, remeasure
from orchestra.sdlab.discovery.strategies.kalman_lna._vendored.lna_pulses.run import Budget
from orchestra.sdlab.discovery.strategies.kalman_lna._vendored.lna_stack.run import hooks


def identify(stocks: list[dict], run: dict, opts: dict | None = None) -> dict:
    opts = dict(opts or {})
    ids = [s["id"] for s in stocks]
    T = len(run["t"]) - 1
    inj = opts.get("injections")
    base = _matrix(stocks, run, dict(opts))
    rest = {k: v for k, v in opts.items() if k != "candidates"}
    budget = Budget(opts)
    found = _not_injected(find_pulses(stocks, run, opts), run, inj)
    r, state, pulses = None, {}, {}
    for _ in range(3):                 # lna with the pulses taken out; the pulses measured again on its model
        known = None
        if found:
            known = np.zeros((len(ids), T))
            for i, a in found.items():
                known[ids.index(i)] = a
        st: dict = {}
        got = budget.discover(stocks, run, rest, base, hooks(st, known))
        if got is None:                # no counting noise (the first search), or out of time on a later one: the last kept
            break
        r, state = got, st
        if not found:
            break
        r["flows"], pulses = remeasure(r["flows"], ids, run, found, inj)
        same = pulses.keys() == found.keys() and all(np.array_equal(pulses[i] != 0, found[i] != 0) for i in pulses)
        found = pulses
        if same:
            break
    if r is not None and found and not pulses:
        st = {}
        got = budget.discover(stocks, run, rest, base, hooks(st, None))     # the pulses did not hold up: without them
        if got is not None:
            r, state = got, st
    if r is not None:
        extra = [g for i, a in pulses.items() for g in pulse_flows(i, a, run["t"])]
        flows = r["flows"] + extra
        theirs = judge(base["flows"], stocks, run, opts)
        # lna's guard on lna's own reading (pulses out) first, then on the result: where lna would
        # keep the matrix's model, so does this
        for fl in (state["lna_flows"] + extra, flows):
            mine = judge(fl, stocks, run, opts)
            if not math.isfinite(mine) or (math.isfinite(theirs) and mine > GUARD * theirs):
                return {**base, "lna": {"judged": {"lna": mine, "matrix": theirs}}}
        r["flows"] = flows
        r["pulses"] = {i: int((a != 0).sum()) for i, a in pulses.items()}
        r["levels"] = {ids[s]: len(g) for s, g in (state.get("levels") or {}).items()}
        r["products"] = state.get("products") or []
        return r
    # no counting noise: the matrix's reading, pulses re-measured on it, ties written as SD writes them
    out = dict(base)
    if found:
        flows, pulses = remeasure(base["flows"], ids, run, found, inj)
        if pulses:
            out["flows"] = flows + [g for i, a in pulses.items() for g in pulse_flows(i, a, run["t"])]
            out["pulses"] = {i: int((a != 0).sum()) for i, a in pulses.items()}
    return conventions(out, stocks, run, opts)
