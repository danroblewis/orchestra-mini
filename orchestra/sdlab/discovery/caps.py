"""What an emptied stock hides: outflows cut short because a stock cannot give more than it holds.

SD Lab's engine shares a stock's level among its outflows when they want more than it holds, so on
those steps the measured change is not the flows' equations: the source loses only what it had and
its receivers get less. A straight line fitted through such steps is bent toward them (sd-orders: a
constant sale of 15 a day read as 21.8 a day plus "stock grows with itself").

Given a found model, `shortfall` estimates, for every stock and step, how much more the stock would
have changed had its outflows not been cut short (the model's wanted outflow beyond the level it
had, shared as the engine shares it, taken from the source and added to the receivers). Adding it to
the measured changes and fitting again is one step of EM for censored data; repeating it converges
toward the flows' own equations at a rate given by the share of steps that are cut short.
"""

from __future__ import annotations

import numpy as np

from .simulate import Model


def flow_amounts(flows: list[dict], ids: list[str], t: list, prev: np.ndarray) -> tuple[list[tuple], np.ndarray]:
    """Each flow's wanted amount at every step (floored at 0, as the model steps it), from the
    levels at the start of the step: (source, destination, ratio) per flow, and flows x steps."""
    model = Model(flows, ids)
    T = prev.shape[1]
    times = np.asarray(t, dtype=float)[1 : T + 1]
    zero = np.zeros(T, int)
    ends, amounts = [], []
    for src, dst, ratio, segs, rate, vals in model.parts:
        seg = segs(times) if segs is not None else zero
        a = vals[seg] if vals is not None else rate(prev, seg)
        ends.append((src, dst, ratio))
        amounts.append(np.nan_to_num(np.asarray(a, dtype=float)))
    return ends, np.array(amounts) if amounts else np.zeros((0, T))


def shortfall(flows: list[dict], ids: list[str], t: list, prev: np.ndarray, after: np.ndarray | None = None,
              most: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """(adjustment to the measured changes, stocks x steps; which stock was emptied at which step).

    With `after` (the levels at the end of each step), a stock counts as emptied only where the data
    agrees: it ended the step holding no more than the model says arrived in it (plus counting
    noise), as an emptied stock does. A wrong model then cannot invent cuts the data does not show.
    A stock emptied on more than `most` of the steps is left alone: what its outflows want is seen on
    too few steps to be learned (EM for censored data needs most of them uncut), and its estimated
    outside outflow would only take whatever is added back, round after round."""
    S, T = prev.shape
    ends, amt = flow_amounts(flows, ids, t, prev)
    want, came = np.zeros((S, T)), np.zeros((S, T))
    for (src, dst, ratio), a in zip(ends, amt):
        if src is not None:
            want[src] += a * ratio
        if dst is not None:
            came[dst] += a
    have = np.maximum(0.0, prev)
    capped = want > have
    if after is not None:
        capped &= after <= came + 2 * np.sqrt(came + 1)
    capped &= (capped.mean(axis=1) <= most)[:, None]
    keep = np.where(capped, have / np.where(want > 0, want, 1.0), 1.0)    # the share each outflow gets
    adj = np.zeros((S, T))
    for (src, dst, ratio), a in zip(ends, amt):
        if src is None:
            continue
        cut = a * (1.0 - keep[src])
        adj[src] -= cut * ratio
        if dst is not None:
            adj[dst] += cut
    return adj, capped
