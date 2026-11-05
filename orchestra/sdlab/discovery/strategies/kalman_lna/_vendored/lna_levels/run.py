# Vendored for kalman_lna from sdlab/discovery/strategies/lna_levels/run.py at commit ec2f9e8 (imports rewritten to the
# vendored copies; otherwise unchanged). Do not edit: re-vendor instead.
"""The levels stage as hooks on lna's discover (lna/run.py HOOKS): after lna's search, a self-growth
term offered back as inflow pieces and each stock's inflow pieces regrouped into recurring levels
(levels.py); the levels written as one inflow in pieces per stock. No search code is copied: lna's
data, budget, stride and measured inputs are lna's own."""

from __future__ import annotations

from orchestra.sdlab.discovery.strategies.kalman_lna._vendored.lna.fit import OutOfTime
from orchestra.sdlab.discovery.strategies.kalman_lna._vendored.lna.flows import to_flows
from orchestra.sdlab.discovery.strategies.kalman_lna._vendored.lna_levels.levels import level_flows, regroup, without_growth


def written(cur, d, ctx) -> list[dict]:
    """Atoms as lna writes them (lna/run.py's last steps): its own reading, for the guard."""
    conf, bucket = ctx["conf"], ctx["bucket"]
    flows = to_flows(cur.atoms, cur.theta, ctx.get("names") or ctx["ids"], d.prev, ctx["at"], *((conf[0], conf[2], bucket) if conf else ()))
    for f in flows:
        if f.get("driver") in ctx["inputs"]:
            f["input"] = f["driver"]
    return flows


def stage(state: dict) -> dict:
    """{after_search, flows} for the levels stage; `state` collects what the strategy reports
    (lna_cur: lna's own atoms, levels, lna_flows)."""

    def after_search(cur, d, ctx):
        state.setdefault("lna_cur", cur)
        state.setdefault("base", ctx["base"])
        try:
            g = without_growth(cur, d)
            g, levels, _ = regroup(g, d)
        except OutOfTime:                # past lna's budget: lna's reading stands
            state["levels_out_of_time"] = True
            return cur
        state["levels"] = levels
        return g

    def flows(flows, cur, d, ctx):
        state.setdefault("lna_flows", written(state["lna_cur"], d, ctx))
        return level_flows(flows, state.get("levels") or {}, ctx["ids"], ctx["at"])

    return {"after_search": after_search, "flows": flows}
