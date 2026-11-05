# Vendored for kalman_lna from sdlab/discovery/strategies/censored/onestep.py at commit ec2f9e8 (imports rewritten to the
# vendored copies; otherwise unchanged). Do not edit: re-vendor instead.
"""A whole model judged on single steps, in the form the engine steps it.

`Model.change` (sdlab/discovery/simulate.py) applies every rate's floor at 0 and shares an emptied
stock among its outflows, so the one-step residual of a model with a floor or a cap is measured in
the model's real (kinked) form. The BIC counts each flow's numbers, and an outside inflow in pieces
as one value per piece and one change per cut.
"""

from __future__ import annotations

import math

import numpy as np

from orchestra.sdlab.discovery.simulate import Model


def levels(stocks: list[dict], run: dict) -> np.ndarray:
    t = run["t"]
    return np.nan_to_num(np.array([np.asarray(run["levels"].get(s["id"], [0] * len(t)), dtype=float) for s in stocks]))


def changes(flows: list[dict], ids: list[str], t: list, prev: np.ndarray) -> np.ndarray:
    """What the flows add to each stock at every step (stocks x steps), from the levels at its start."""
    if not flows:
        return np.zeros_like(prev)
    times = np.asarray(t, dtype=float)[1 : prev.shape[1] + 1]
    return Model(flows, ids).change(prev, times)


def params(flows: list[dict]) -> int:
    """The numbers a model's flows carry (a piece's value and its cut each count one)."""
    p = 0
    for f in flows:
        if f.get("inflow"):
            p += 2 * len(f["inflow"]) - 1
            continue
        r = f.get("rate") or {}
        p += {"constant": 1, "proportional": 1, "gap": 2}.get(r.get("form"), 1 + len(r.get("terms") or []))
        p += int(f.get("threshold_params") or 0)
    return p


def row_rss(flows: list[dict], ids: list[str], t: list, prev: np.ndarray, dy: np.ndarray) -> np.ndarray:
    return ((dy - changes(flows, ids, t, prev)) ** 2).sum(axis=1)


def outliers(flows: list[dict], ids: list[str], t: list, prev: np.ndarray, dy: np.ndarray, share: float = 0.01) -> np.ndarray:
    """Per row, the steps (stocks x steps, True = left out) where a model's residual is among the
    largest `share`: one-step bursts (inflow spikes) that a Gaussian BIC would let decide alone."""
    r = np.abs(dy - changes(flows, ids, t, prev))
    k = int(share * r.shape[1])
    out = np.zeros_like(r, dtype=bool)
    if k < 1:
        return out
    idx = np.argpartition(-r, k - 1, axis=1)[:, :k]
    np.put_along_axis(out, idx, True, axis=1)
    return out


def bic(flows: list[dict], ids: list[str], t: list, prev: np.ndarray, dy: np.ndarray, floor: np.ndarray,
        skip: np.ndarray | None = None) -> float:
    """Sum over the rows of T log(rss / T), plus log T per number, the rss over the steps not in
    `skip` (stocks x steps). `floor` (one per row) keeps an exact fit from counting as infinitely good."""
    T = prev.shape[1]
    sq = (dy - changes(flows, ids, t, prev)) ** 2
    if skip is not None:
        sq = np.where(skip, 0.0, sq)
    rss = np.maximum(sq.sum(axis=1), floor)
    return float(sum(T * math.log(r / T) for r in rss) + params(flows) * math.log(T))
