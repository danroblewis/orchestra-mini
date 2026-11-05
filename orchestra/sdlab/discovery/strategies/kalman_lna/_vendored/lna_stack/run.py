# Vendored for kalman_lna from sdlab/discovery/strategies/lna_stack/run.py at commit ec2f9e8 (imports rewritten to the
# vendored copies; otherwise unchanged). Do not edit: re-vendor instead.
"""The revisit stages as hooks on lna's discover (lna/run.py HOOKS), in order: known pulses taken out
of the changes (lna_pulses), product columns (lna_products), lna's search, products offered on the
kept moves and outflows (lna_products), self-growth offered back as pieces and the pieces regrouped
into recurring levels (lna_levels); then the flows, with effects and levels written out. Each stage
is its package's own hooks, unchanged; no search code is copied."""

from __future__ import annotations

import numpy as np

from orchestra.sdlab.discovery.strategies.kalman_lna._vendored.lna_levels.run import stage as levels
from orchestra.sdlab.discovery.strategies.kalman_lna._vendored.lna_products.run import stage as products
from orchestra.sdlab.discovery.strategies.kalman_lna._vendored.lna_pulses.run import stage as pulses


def chain(*stages: dict) -> dict:
    """The hooks of several stages, each kind run in the order given."""
    out: dict = {}
    for st in stages:
        for name, fn in st.items():
            first = out.get(name)
            if first is None:
                out[name] = fn
            elif name == "data":
                out[name] = (lambda f, g: lambda p, y, ctx: g(*f(p, y, ctx), ctx))(first, fn)
            elif name == "flows":
                out[name] = (lambda f, g: lambda fl, cur, d, ctx: g(f(fl, cur, d, ctx), cur, d, ctx))(first, fn)
            else:
                out[name] = (lambda f, g: lambda x, d, ctx: g(f(x, d, ctx), d, ctx))(first, fn)
    return out


def hooks(state: dict, known: np.ndarray | None) -> dict:
    lv = levels(state)
    # levels run inside the products stage's after_search, on the same data (the product columns as
    # rows of the changes too, as when the stack was first built)
    return chain(pulses(known), products(state, then=lv["after_search"], rows=True), {"flows": lv["flows"]})
