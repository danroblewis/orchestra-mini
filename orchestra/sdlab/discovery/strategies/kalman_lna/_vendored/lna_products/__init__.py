# Vendored for kalman_lna from sdlab/discovery/strategies/lna_products/__init__.py at commit ec2f9e8 (imports rewritten to the
# vendored copies; otherwise unchanged). Do not edit: re-vendor instead.
"""lna_products: lna with product columns, so a move can be scaled by a linear effect of a third
stock (rate k x a x c / mean(c)), chosen by the same counting-noise likelihood and BIC as every
other flow.

PySINDy's verdict (docs/research/pysindy.md) was "reject as a strategy; carry the product terms
forward", and effect_hats / symbolic fit effect curves by least squares, which on the ladder's noisy
runs of the linear effect (017) read the effect as links and drains. Here the product terms sit in
lna's likelihood, where the move's draw leaving a and arriving in b is what tells a move scaled by c
from a drain of a and a link into b. A stage on lna's hooks (run.py): lna's data, search, budget
and fallbacks are lna's own. Noise-free runs, and runs with fewer than three stocks or more than
lna's scratch size, are lna's. lna's guard (the matrix's model is kept when lna's is judged much
worse) is applied to the result and to lna's own reading.
"""

from __future__ import annotations

from orchestra.sdlab.discovery.strategies.kalman_lna._vendored import lna
from orchestra.sdlab.discovery.strategies.kalman_lna._vendored.lna_levels import own_reading_guard
from orchestra.sdlab.discovery.strategies.kalman_lna._vendored.lna_products.run import stage


def identify(stocks: list[dict], run: dict, opts: dict | None = None) -> dict:
    opts = dict(opts or {})
    state: dict = {}
    r = lna.identify(stocks, run, opts, hooks=stage(state))
    if not state.get("products"):
        return r                          # no product kept: lna's own reading, under lna's guard
    r = own_reading_guard(r, state, stocks, run, opts)
    if r.get("method") == "lna":
        r["products"], r["product_gain"] = state["products"], state.get("product_gain")
    return r

