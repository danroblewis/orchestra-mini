# Vendored for kalman_lna from sdlab/discovery/strategies/common_cause/refit.py at commit ec2f9e8 (imports rewritten to the
# vendored copies; otherwise unchanged). Do not edit: re-vendor instead.
"""earned_pieces' refit (rows reselected, conservation decided jointly), with shared change places
offered to every row after the reselection (places.py)."""

from __future__ import annotations

import numpy as np

from orchestra.sdlab.discovery.matrix import finish
from orchestra.sdlab.discovery.strategies.kalman_lna._vendored.common_cause.exact import tiebreak
from orchestra.sdlab.discovery.strategies.kalman_lna._vendored.common_cause.joint import System, decide
from orchestra.sdlab.discovery.strategies.kalman_lna._vendored.common_cause.places import offer
from orchestra.sdlab.discovery.strategies.kalman_lna._vendored.common_cause.rows import reselect

ALPHA = 0.001       # the whiteness test's level (one test per row and candidate: 1% rejects true rows too often)


def refit(stocks: list[dict], r: dict, opts: dict, pelt) -> dict:
    fit, ctx = r.get("fit"), r.get("ctx")
    if not fit or not ctx:
        return r
    ids, rows = fit["ids"], fit["rows"]
    unit = {s["id"]: s.get("unit") or "" for s in stocks}
    final = {s["id"] for s in stocks if s.get("final")}
    L = len(fit["labels"])
    mp = opts.get("min_piece", 5)
    if opts.get("reselect", True):
        rows = [reselect(row, s, ids, stocks, fit["prev"], fit["dy"], fit["cfg"], L, opts.get("change_params", 2), opts.get("alpha", ALPHA))["row"] for s, row in enumerate(rows)]
    if opts.get("shared_places", True):
        rows = offer(rows, ids, stocks, fit["prev"], fit["dy"], fit["cfg"], L, pelt, mp, opts.get("change_params", 2), opts.get("trace_places", False), alpha=opts.get("alpha", ALPHA))
    if opts.get("tiebreak", True):
        rows = [tiebreak(row, s, fit["dy"][s]) for s, row in enumerate(rows)]
    system = System(rows, ids, unit, final, fit["dy"], fit["cfg"], L, opts.get("counting", True), mp, opts.get("scale", "unit"))
    best = decide(system, ctx["log_t"], trace=opts.get("trace_joint", False))
    at = fit["at"]
    new = []
    for s, row in enumerate(rows):
        coef_l = [{} for _ in range(system.L)]
        for q, b in zip(row["set"], best["beta"][s]):
            j, lab = row["col"][q]
            for c in range(system.L):
                if lab < 0 or lab == c:
                    coef_l[c][ids[j]] = coef_l[c].get(ids[j], 0.0) + float(b)
        u = best["inflow"][s]
        cuts = best["cuts"][s]
        inflow = [{"from": at[a], "to": at[b - 1], "value": float(u[a])} for a, b in zip(cuts[:-1], cuts[1:])] if cuts else []
        new.append({"id": row["id"], "coefL": coef_l, "inflow": inflow, "cuts": cuts, "set": [], "col": [],
                    "X": np.zeros((0, system.T)), "beta": np.zeros(0), "rss": 0.0, "p": 0})
    out = finish(new, {**ctx, "final": set(ids)}, opts)
    out["tied"] = sorted(ids[a] for a in best["tied"])
    return out
