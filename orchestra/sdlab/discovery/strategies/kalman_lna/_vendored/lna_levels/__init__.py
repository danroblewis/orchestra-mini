# Vendored for kalman_lna from sdlab/discovery/strategies/lna_levels/__init__.py at commit ec2f9e8 (imports rewritten to the
# vendored copies; otherwise unchanged). Do not edit: re-vendor instead.
"""lna_levels: lna, with each outside inflow's pieces offered as a few recurring levels, and a
stock's growth with itself beside an inflow offered back as inflow pieces (levels.py).

From the regime-models and inflow-innovations ideas (docs/research/regime-models.md,
inflow-innovations.md; never built on the old bench): an unannounced square wave is a two-state
recurring regime, two numbers rather than one per piece, so the pieces both cost less (BIC) and are
each measured from every step at their level. Measured on the ladder in docs/research/revisit.md.

A stage on lna's hooks (run.py): lna's data, search, budget and fallbacks are lna's own. Noise-free
runs are lna's (the matrix's reading). lna's guard is applied to the result (lna.identify) and to
lna's own reading: where lna would return the matrix's model, so does this.
"""

from __future__ import annotations

import math

from orchestra.sdlab.discovery.strategies.kalman_lna._vendored import lna
from orchestra.sdlab.discovery.strategies.kalman_lna._vendored.lna_levels.run import stage


def own_reading_guard(r: dict, state: dict, stocks: list[dict], run: dict, opts: dict) -> dict:
    """lna's guard on lna's own reading too (levels and products refine a reading lna trusts; they
    do not rescue one it rejects): the matrix's model where that reading is judged GUARD x worse."""
    if r.get("method") != "lna" or "lna_flows" not in state or "base" not in state or r["flows"] == state["lna_flows"]:
        return r
    theirs = r["judged"]["matrix"]
    mine = lna.judge(state["lna_flows"], stocks, run, opts)
    if not math.isfinite(mine) or (math.isfinite(theirs) and mine > lna.GUARD * theirs):
        return {**state["base"], "lna": {"judged": {"lna": mine, "matrix": theirs}, "nll": r["nll"], "dispersion": r["dispersion"]}}
    return r


def identify(stocks: list[dict], run: dict, opts: dict | None = None) -> dict:
    opts = dict(opts or {})
    state: dict = {}
    r = lna.identify(stocks, run, opts, hooks=stage(state))
    r = own_reading_guard(r, state, stocks, run, opts)
    if r.get("method") == "lna":
        r["levels"] = {stocks[s]["id"]: len(g) for s, g in (state.get("levels") or {}).items()}
    return r
