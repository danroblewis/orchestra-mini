# Vendored for kalman_lna from sdlab/discovery/strategies/lna_pulses/run.py at commit ec2f9e8 (imports rewritten to the
# vendored copies; otherwise unchanged). Do not edit: re-vendor instead.
"""The pulses stage as a hook on lna's discover (lna/run.py HOOKS): known changes, an array (stocks x
steps) of pulses found beforehand, taken out of each step's change before anything is fitted, so no
flow has to absorb them. No search code is copied: lna's data, budget, stride and measured inputs
are lna's own."""

from __future__ import annotations

import time

import numpy as np

from orchestra.sdlab.discovery.strategies.kalman_lna._vendored.lna.run import BUDGET
from orchestra.sdlab.discovery.strategies.kalman_lna._vendored.lna.run import discover as lna_discover


def stage(known: np.ndarray | None) -> dict:
    """{data}: the known changes out of dy (before lna's stride, so every step's pulse is taken out)."""
    if known is None:
        return {}

    def data(prev, dy, ctx):
        return prev, dy - known

    return {"data": data}


class Budget:
    """lna's search budget (opts "lna_budget", default lna's BUDGET) shared by every lna search a
    strategy runs, so that repeated searches stay within one lna's time."""

    def __init__(self, opts: dict):
        self.end = time.monotonic() + float(opts.get("lna_budget", BUDGET))

    def left(self) -> float:
        return self.end - time.monotonic()

    def discover(self, stocks, run, opts, base, hooks=None):
        """lna's discover with what is left of the budget; None past it (as lna's own)."""
        if self.left() <= 0:
            return None
        return lna_discover(stocks, run, {**opts, "lna_budget": self.left()}, base, hooks)
