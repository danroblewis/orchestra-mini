"""Pulses: one-step arrivals and short bursts, told apart from noise and from level shifts.

A delivery of 450 doses once a week, a sprint's 25 cards on its first day, a bad release's tickets
for three to six days: inputs that last one to a few steps. The change scan (changes.pelt) has a
5-step minimum piece, so it cannot hold them, and a row fitted through them lets its drain absorb
them. Here they are found in a row's residual in the style of Chen and Liu (1993): the largest
standardised window (1 to `max_len` steps) is taken while it passes a threshold of about
sqrt(2 log T) robust standard deviations; a window that keeps growing when lengthened is a level
shift (left to the change scan), not a pulse. Each pulse step is then an indicator of its own, which
is the same as taking the step out of the row's fit.
"""

from __future__ import annotations

import math

import numpy as np

from .changes import pelt


def robust_scale(r: np.ndarray) -> float:
    """The noise's standard deviation from successive differences (MAD), so that pulses and level
    shifts do not inflate it; falls back to coarser measures where counts make the MAD 0."""
    r = np.asarray(r, dtype=float)
    if r.shape[0] < 3:
        return float(np.std(r)) or 1.0
    d = np.diff(r)
    s = 1.4826 * float(np.median(np.abs(d - np.median(d)))) / math.sqrt(2)
    if s <= 0:
        s = 1.2533 * float(np.mean(np.abs(d - np.median(d)))) / math.sqrt(2)
    return s or 1.0


def running_median(y: np.ndarray, width: int) -> np.ndarray:
    """A centred running median (shorter at the ends): the series' level without its pulses."""
    y = np.asarray(y, dtype=float)
    n, h = y.shape[0], width // 2
    if n == 0:
        return y
    pad = np.concatenate([np.full(h, y[0]), y, np.full(h, y[-1])])
    win = np.lib.stride_tricks.sliding_window_view(pad, 2 * h + 1)
    return np.median(win, axis=1)[:n]


def _window_z(r: np.ndarray, free: np.ndarray, m: int, sigma: float, sign: int = 0) -> np.ndarray:
    """|sum of r over each window of m free steps| / (sigma sqrt m); windows touching a step that is
    not free score 0. Entry i is the window starting at i."""
    n = r.shape[0]
    if m > n:
        return np.zeros(0)
    c = np.concatenate([[0.0], np.cumsum(np.where(free, r, 0.0))])
    taken = np.concatenate([[0], np.cumsum(~free)])
    s = c[m:] - c[:-m]
    ok = (taken[m:] - taken[:-m]) == 0
    return np.where(ok, (np.abs(s) if not sign else np.maximum(0.0, sign * s)) / (sigma * math.sqrt(m)), 0.0)


def detect(r: np.ndarray, max_len: int = 6, shift_len: int = 24, thresh: float | None = None,
           max_share: float = 0.3, sigma: float | None = None, shift_margin: float = 1.0, sign: int = 0) -> list[dict]:
    """Pulses in a residual series r (mean 0 where nothing happens): [{"a", "b", "kind"}] with steps
    a..b-1, kind "pulse" (one step) or "burst" (2 to max_len steps). Iterates: the window (1 to
    max_len steps) with the largest |sum| / (sigma sqrt m) is taken while that passes `thresh`
    (default sqrt(2 log(T x max_len))); one that grows further when lengthened up to shift_len steps
    (by shift_margin) is a level shift, blocked from later windows and not returned. At most max_share of the steps
    become pulses."""
    r = np.nan_to_num(np.asarray(r, dtype=float))
    n = r.shape[0]
    if n < 10:
        return []
    sigma = sigma or robust_scale(r)
    thresh = thresh if thresh is not None else math.sqrt(2 * math.log(n * max_len))
    free = np.ones(n, bool)
    out: list[dict] = []
    used = 0
    while used < max_share * n:
        best = (0.0, 0, 0)
        for m in range(1, max_len + 1):
            z = _window_z(r, free, m, sigma, sign)
            if z.size:
                i = int(np.argmax(z))
                if z[i] > best[0] * (1 + 1e-9):
                    best = (float(z[i]), i, m)
        z0, a, m = best
        if z0 < thresh:
            break
        # a level shift: a longer window over the same steps (and the same sign) scores higher
        sign = np.sign(r[a:a + m].sum())
        shift = None
        for L in range(max_len + 1, shift_len + 1):
            for start in range(max(0, a + m - L), min(a, n - L) + 1):
                seg = slice(start, start + L)
                if not free[seg].all():
                    continue
                s = r[seg].sum()
                if np.sign(s) == sign and abs(s) / (sigma * math.sqrt(L)) > z0 + shift_margin:
                    shift = (start, start + L)
                    break
            if shift:
                break
        if shift:
            free[shift[0]:shift[1]] = False
            continue
        free[a:a + m] = False
        used += m
        out.append({"a": a, "b": a + m, "kind": "pulse" if m == 1 else "burst"})
    return sorted(out, key=lambda p: p["a"])


def series(pulses: list[dict], r: np.ndarray, n: int) -> np.ndarray:
    """The pulses' amounts as a series of length n: r at the pulse steps, 0 elsewhere."""
    out = np.zeros(n)
    for p in pulses:
        out[p["a"]:p["b"]] = r[p["a"]:p["b"]]
    return out


def scan(u: np.ndarray, min_piece: int = 5, rounds: int = 3, floor: float = 0.0, **kw) -> tuple[np.ndarray, list[int]]:
    """Pulses in what a row's levels leave unexplained (u: its outside inflow plus noise and
    pulses): which steps are pulses, and the pieces of the outside inflow. The inflow's pieces are
    found first (changes.pelt, pieces longer than max_len, so a burst cannot be a piece), so a
    change in the inflow is a level shift of u, not a burst. Pulses smeared into the pieces are kept
    out by measuring each piece by its median, and by finding the pieces again with the pulses
    taken out (rounds). A
    pulse next to a change that the neighbouring piece's level explains is the change found a few
    steps late, not a pulse. `floor`: the least noise scale (counting noise around the row's flows)."""
    n = u.shape[0]
    max_len = kw.get("max_len", 6)
    thresh = kw.get("thresh") or math.sqrt(2 * math.log(max(2, n) * max_len))
    hit = np.zeros(n, bool)
    clean = u.copy()
    for _ in range(rounds):
        cuts = pelt(clean, max(min_piece, max_len + 1), 1)
        seg = np.repeat(np.arange(len(cuts) - 1), np.diff(cuts))
        level = np.array([np.median(u[a:b][~hit[a:b]]) if (~hit[a:b]).any() else np.median(u[a:b]) for a, b in zip(cuts[:-1], cuts[1:])])
        r = u - level[seg]
        sigma = max(robust_scale(r), floor)
        found = detect(r, **{**kw, "sigma": sigma, "thresh": thresh})
        new = np.zeros(n, bool)
        for p in found:
            a, b = p["a"], p["b"]
            near = [i for i in range(1, len(cuts) - 1) if a - max_len <= cuts[i] <= b + max_len]
            if any(abs((u[a:b] - level[q]).sum()) / (sigma * math.sqrt(b - a)) < thresh for i in near for q in (i - 1, i)):
                continue
            new[a:b] = True
        if np.array_equal(new, hit):
            break
        hit = new
        clean = np.where(hit, level[seg], u)
    return hit, cuts


def amounts(u: np.ndarray, hit: np.ndarray, cuts: list[int]) -> np.ndarray:
    """Each pulse step's amount: u less its piece's mean over the steps that are not pulses."""
    seg = np.repeat(np.arange(len(cuts) - 1), np.diff(cuts))
    means = np.array([u[a:b][~hit[a:b]].mean() if (~hit[a:b]).any() else np.median(u[a:b]) for a, b in zip(cuts[:-1], cuts[1:])])
    return np.where(hit, u - means[seg], 0.0)


def masked_fit(y: np.ndarray, X: np.ndarray, hit: np.ndarray, cuts: list[int]) -> np.ndarray:
    """The row's numbers (X: columns x steps) with the pulse steps left out (an indicator per pulse
    step) and a mean per piece of the outside inflow."""
    keep = ~hit
    if X.shape[0] == 0 or keep.sum() <= X.shape[0] + len(cuts):
        return np.zeros(X.shape[0])
    yc, Xc = y.astype(float).copy(), X.astype(float).copy()
    for a, b in zip(cuts[:-1], cuts[1:]):
        k = keep[a:b]
        if k.any():
            yc[a:b] -= yc[a:b][k].mean()
            Xc[:, a:b] -= Xc[:, a:b][:, k].mean(axis=1, keepdims=True)
    beta, *_ = np.linalg.lstsq(Xc[:, keep].T, yc[keep], rcond=None)
    return beta


def joint(y: np.ndarray, X: np.ndarray, beta: np.ndarray, min_piece: int = 5, rounds: int = 6, **kw) -> np.ndarray:
    """Pulses found jointly with the row's numbers (Chen and Liu): find the pulses in what the levels
    leave, refit the numbers without the pulse steps, and again until the pulses stay the same.
    Returns the pulses' amounts per step."""
    y = np.nan_to_num(np.asarray(y, dtype=float))
    hit = None
    for _ in range(rounds):
        u = y - (beta @ X if X.shape[0] else 0.0)
        new, cuts = scan(u, min_piece, **kw)
        if not new.any():
            return np.zeros(y.shape[0])
        if hit is not None and np.array_equal(new, hit):
            break
        hit = new
        beta = masked_fit(y, X, hit, cuts)
    u = y - (beta @ X if X.shape[0] else 0.0)
    return amounts(u, hit, cuts)


def in_row(u: np.ndarray, min_piece: int = 5, **kw) -> np.ndarray:
    """Pulses in u with the row's numbers held fixed, as amounts per step."""
    u = np.nan_to_num(np.asarray(u, dtype=float))
    hit, cuts = scan(u, min_piece, **kw)
    return amounts(u, hit, cuts) if hit.any() else np.zeros(u.shape[0])
