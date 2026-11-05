"""Judging a structural addition: does a model with it predict better than the model without it, when
both are fed outside inflows the addition did not tailor?

    test = addition_test(base_flows, with_flows, stocks, run, inflow="anchor")
    test["accept"], test["gain"], test["noise"]

The main line's judge runs every candidate with its OWN outside-inflow pieces, so a candidate whose
added structure moves what the pieces carried (a curve imitating an inflow change) can score as well
as one whose addition is real. Here the addition is judged against the base it was added to, and
both models are fed the BASE model's outside inflow (judges.py, "anchor"): the addition must explain
something the base's inflow does not already carry. With inflow="remeasure" both are instead fed
inflows of the base's form (the same stocks, the same piece times) whose values are re-measured with
each model's own dynamics frozen: the same inflow budget for both, values that fit each.

The margin is the judge's own noise: the gain (the base's error minus the model's, relative to the
base's, over 5, 20 and 100 steps) is computed per start, and the starts are block-bootstrapped (runs
from nearby starts overlap) to give its spread; an addition is accepted when the gain is more than
`z` spreads and more than `floor`. With `pieces` it must also leave the inflow it needs no larger
than the base's: its own pieces' variance (each stock in its own scale of change) no larger.
"""

from __future__ import annotations

import numpy as np

from .judges import dynamics, net_inflow, own_stocks, refit_inflow
from .simulate import Model, levels_array, starts_for

HORIZONS = (5, 20, 100)
STARTS = 80


def start_errors(flows: list[dict], ids: list[str], t: list, levels: dict, U: np.ndarray, horizon: int,
                 starts: int = STARTS) -> tuple[np.ndarray, float]:
    """judges.exo_error per start: each start's squared miss (stocks in their own spread), and the
    data's movement over all starts (exo_error = sum of the first / the second)."""
    A = levels_array(levels, ids)
    n = A.shape[1]
    s0 = starts_for(n, horizon, starts)
    if not len(s0):
        return np.zeros(0), 1.0
    sd = A.std(axis=1)
    sd[sd == 0] = 1
    tt = np.asarray(t, dtype=float)
    idx = s0[None, :] + np.arange(horizon)[:, None]
    model = Model(dynamics(flows), ids)
    L = A[:, s0 - 1]
    with np.errstate(all="ignore"):
        for k in range(horizon):
            L = model.step(L, tt[idx[k]]) + U[:, idx[k]]
            L = np.where(L >= 0, L, 0.0)
        obs = A[:, s0 + horizon - 1]
        e = ((L - obs) / sd[:, None]) ** 2
    e[~np.isfinite(e)] = 1e6
    moved = float((((obs - A[:, s0 - 1]) / sd[:, None]) ** 2).sum())
    return np.minimum(e.sum(axis=0), 1e6), max(1e-9, moved)


def remeasure(U: np.ndarray, flows: list[dict], ids: list[str], t: list, levels: dict) -> np.ndarray:
    """An inflow of U's form (per stock, its runs of one value) with each run's value re-measured as
    the mean of what `flows`' dynamics leave unexplained one step at a time over that run."""
    A = levels_array(levels, ids)
    S, n = A.shape
    out = np.zeros_like(U)
    if n < 3:
        return out
    tt = np.asarray(t, dtype=float)
    with np.errstate(all="ignore"):
        res = np.zeros((S, n))
        res[:, 1:] = np.diff(A, axis=1) - (Model(dynamics(flows), ids).step(A[:, :-1], tt[1:]) - A[:, :-1])
    res = np.nan_to_num(res)
    for i in range(S):
        if not U[i, 1:].any():
            continue
        k = 1
        while k < n:
            e = k
            while e + 1 < n and U[i, e + 1] == U[i, k]:
                e += 1
            out[i, k:e + 1] = res[i, k:e + 1].mean()
            k = e + 1
    return out


def piece_load(flows: list[dict], ids: list[str], t: list, levels: dict) -> float:
    """How much the model's outside-inflow pieces vary: per stock, their variance over the run relative
    to the variance of the stock's step changes, summed."""
    A = levels_array(levels, ids)
    U = net_inflow(flows, ids, t)
    dv = np.diff(A, axis=1).var(axis=1)
    dv[dv == 0] = 1
    return float((U[:, 1:].var(axis=1) / dv).sum())


def _block_boot(d: np.ndarray, b: np.ndarray, moved: float, rng: np.random.Generator, boot: int, block: int):
    """Bootstrap draws of (sum of d, sum of b) / moved, resampling blocks of `block` neighbouring starts."""
    m = len(d)
    nb = max(1, -(-m // block))
    first = rng.integers(0, max(1, m - block + 1), size=(boot, nb))
    idx = (first[..., None] + np.arange(block)).reshape(boot, -1)[:, :m]
    idx = np.minimum(idx, m - 1)
    return d[idx].sum(axis=1) / moved, b[idx].sum(axis=1) / moved


def addition_test(base: list[dict], cand: list[dict], stocks: list[dict], run: dict, inflow: str = "anchor",
                  horizons=HORIZONS, z: float = 2.0, floor: float = 0.02, pieces: bool = False, boot: int = 300,
                  block: int = 8, extra: list[dict] | None = None, seed: int = 0, refit_len: int | None = None) -> dict:
    """Is `cand` (base plus an addition) better than `base`, both fed an inflow the addition did not
    tailor? {"accept", "gain" (relative to the base's error), "noise" (the gain's bootstrap spread),
    "base", "cand" (the two errors), "load" (pieces' variance, base and cand)}."""
    ids = [s["id"] for s in stocks]
    t, levels = run["t"], run["levels"]
    hs = [h for h in horizons if h < len(t) / 4]
    extra = list(extra or [])
    out = {"accept": False, "gain": 0.0, "noise": float("inf"), "base": float("inf"), "cand": float("inf")}
    if not cand or not base or not hs:
        return out
    Ub = net_inflow(base, ids, t)
    if inflow == "anchor":
        Ub_, Uc_ = Ub, Ub
    elif inflow == "remeasure":
        Ub_, Uc_ = remeasure(Ub, base + extra, ids, t, levels), remeasure(Ub, cand + extra, ids, t, levels)
    elif inflow == "refit":
        feed = own_stocks(base, ids)
        Ub_, Uc_ = (refit_inflow(base + extra, ids, t, levels, feed, refit_len),
                    refit_inflow(cand + extra, ids, t, levels, feed, refit_len))
    elif inflow == "own":
        Ub_, Uc_ = Ub, net_inflow(cand, ids, t)
    else:
        raise ValueError(f"unknown inflow {inflow!r}")
    rng = np.random.default_rng(seed)
    gain_parts, boot_d, boot_b, eb, ec = [], [], [], [], []
    for h in hs:
        b, moved = start_errors(base + extra, ids, t, levels, Ub_, h)
        c, _ = start_errors(cand + extra, ids, t, levels, Uc_, h)
        if not len(b):
            continue
        eb.append(b.sum() / moved)
        ec.append(c.sum() / moved)
        bd, bb = _block_boot(b - c, b, moved, rng, boot, block)
        boot_d.append(bd)
        boot_b.append(bb)
        gain_parts.append((b - c).sum() / moved)
    if not eb:
        return out
    E = float(np.mean(eb))
    gain = float(np.mean(gain_parts)) / max(1e-12, E)
    draws = np.mean(boot_d, axis=0) / np.maximum(1e-12, np.mean(boot_b, axis=0))
    noise = float(draws.std())
    out.update(gain=gain, noise=noise, base=E, cand=float(np.mean(ec)))
    accept = gain > max(floor, z * noise)
    if pieces:
        lb, lc = piece_load(base, ids, t, levels), piece_load(cand, ids, t, levels)
        out["load"] = (lb, lc)
        accept = accept and lc <= lb
    out["accept"] = bool(accept)
    return out

