# Vendored for kalman_lna from sdlab/discovery/strategies/censored/cap.py at commit ec2f9e8 (imports rewritten to the
# vendored copies; otherwise unchanged). Do not edit: re-vendor instead.
"""An outflow capped by what its stock holds, found by profiling the outflow's constant.

When a stock's outflows want more than it holds, the engine gives them what it has: on those steps
the stock ends holding only what arrived, and the change is "inflow - level", not "inflow - demand".
A straight one-step fit cannot tell a constant demand D from the outside inflow u (only u - D is
seen while the stock has stock), but the emptied steps can: there, everything leaves. So for each D
(and own drain d) on a grid, the outside inflow is what the row's change says arrived,

    u_t = change_t - (moves in)_t + min(level_t, D + d x level_t + (other outflows wanted)_t),

cut into pieces of constant mean (PELT; each change priced at `mult` x log n x the noise variance,
2 by default: the main line's 1 cuts pure counting noise into dozens of pieces), and the (D, d)
whose pieces fit best by BIC is kept, refined by a local search with the cuts held. D = 0 is the
plain fit (returned as "plain_fit"), so a cap is offered only where the emptied steps say it binds.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize, minimize_scalar

from orchestra.sdlab.discovery.changes import pelt


def outside_inflow(D: float, d: float, x: np.ndarray, y: np.ndarray, came: np.ndarray, wanted: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """u_t for a demand D and own drain d, and where the outflows were capped."""
    want = D + d * x + wanted
    have = np.maximum(0.0, x)
    capped = want > have
    out = np.where(capped, have, want)
    return y - came + out, capped


def piece_fit(u: np.ndarray, cuts: list[int]) -> tuple[np.ndarray, float]:
    """Each piece's mean (floored at 0: an outside inflow) and the rss around them."""
    fitted = np.empty_like(u)
    for a, b in zip(cuts[:-1], cuts[1:]):
        fitted[a:b] = max(0.0, float(u[a:b].mean()))
    r = u - fitted
    return fitted, float(r @ r)


def cost(rss: float, T: int, pieces: int, k: int, floor: float) -> float:
    return T * math.log(max(rss, floor) / T) + (2 * pieces - 1 + k) * math.log(T)


def search(x: np.ndarray, y: np.ndarray, came: np.ndarray, wanted: np.ndarray, d0: float, floor: float,
           min_piece: int = 5, grid: int = 17, changes: bool = True, mult: float = 2.0, low_share: float = 0.05) -> dict | None:
    """The best (D, d) with its pieces, or None; with "plain", the best score with D = 0 (no cap).
    d is tried on a grid over [0, 1] and at the main line's d0."""
    T = len(y)
    span = float(np.percentile(np.abs(y), 99)) + float(np.percentile(np.maximum(0.0, x), 50))
    if span <= 0:
        return None
    Ds = np.linspace(0.0, 2.0 * span, grid)
    last = [0.0]

    def cuts_of(u):
        return pelt(u, min_piece, mult) if changes else [0, T]

    best, plain = None, None

    def consider(D, d):
        nonlocal best, plain
        u, capped = outside_inflow(D, d, x, y, came, wanted)
        c = cuts_of(u)
        _, rss = piece_fit(u, c)
        sc = cost(rss, T, len(c) - 1, (1 if D > 0 else 0) + (1 if d > 0 else 0), floor)
        last[0] = sc
        if best is None or sc < best["score"]:
            best = {"D": float(D), "d": float(d), "cuts": c, "rss": rss, "score": sc, "capped": capped}
        if D == 0 and (plain is None or sc < plain[0]):
            plain = (sc, d, {"D": 0.0, "d": float(d), "cuts": c, "rss": rss, "score": sc, "capped": capped})

    # the plain fit first (no demand: the own drain on a grid), then the demand with the drains
    # worth trying (none, the main line's, the plain fit's); each grid coarse, then finer around
    # its best point
    def sweep(values, at):
        scores = []
        for v in values:
            consider(*at(v))
            scores.append(last[0])
        k = int(np.argmin(scores))
        lo, hi = values[max(0, k - 1)], values[min(len(values) - 1, k + 1)]
        for v in np.linspace(lo, hi, 9)[1:-1]:
            consider(*at(v))

    sweep(np.array(sorted({0.0, max(0.0, d0), *np.linspace(0.0, 1.0, 11)[1:]})), lambda d: (0.0, d))
    # a demand can be capped only in a stock that sits near the bottom of its range for a while
    # (an emptied stock holds what arrived): otherwise the demand sweep is skipped
    low = float((x <= x.min() + 0.05 * np.ptp(x)).mean()) if np.ptp(x) > 0 else 0.0
    if low >= low_share:
        for d in sorted({0.0, plain[1]}):
            sweep(Ds[1:], lambda D: (D, d))
    plain_fit = _refine_drain(plain[2], x, y, came, wanted, cuts_of, T, floor)
    if plain_fit["score"] < plain[0]:
        plain = (plain_fit["score"], plain_fit["d"], plain_fit)
        if plain_fit["score"] < best["score"]:
            best = plain_fit
    if best["D"] <= 0:
        return {**best, "plain": plain[0], "plain_fit": plain_fit}
    plain = plain[0]
    best["plain"] = plain
    # refine (D, d) with the cuts held, then cut again
    for _ in range(3):
        c = best["cuts"]
        free_d = best["d"] > 0

        def f(p):
            D, d = p[0], (p[1] if free_d else 0.0)
            if D < 0 or d < 0:
                return 1e300
            return piece_fit(outside_inflow(D, d, x, y, came, wanted)[0], c)[1]

        p0 = [best["D"], best["d"]] if free_d else [best["D"]]
        res = minimize(f, p0, method="Nelder-Mead", options={"xatol": 1e-7, "fatol": 1e-10, "maxiter": 2000})
        D, d = float(res.x[0]), float(res.x[1]) if free_d else 0.0
        u, capped = outside_inflow(D, d, x, y, came, wanted)
        c2 = cuts_of(u)
        _, rss = piece_fit(u, c2)
        sc = cost(rss, T, len(c2) - 1, 1 + free_d, floor)
        if sc >= best["score"] - 1e-9:
            break
        best = {"D": D, "d": d, "cuts": c2, "rss": rss, "score": sc, "capped": capped, "plain": plain}
    best["plain_fit"] = plain_fit
    return best


def _refine_drain(fit: dict, x, y, came, wanted, cuts_of, T: int, floor: float) -> dict:
    """The plain fit's own drain refined continuously with its cuts held, then cut again."""
    if fit["d"] <= 0:
        return fit
    c = fit["cuts"]
    lo, hi = max(0.0, fit["d"] - 0.05), fit["d"] + 0.05
    res = minimize_scalar(lambda d: piece_fit(outside_inflow(0.0, d, x, y, came, wanted)[0], c)[1], bounds=(lo, hi), method="bounded",
                          options={"xatol": 1e-7})
    d = float(res.x)
    u, capped = outside_inflow(0.0, d, x, y, came, wanted)
    c2 = cuts_of(u)
    _, rss = piece_fit(u, c2)
    sc = cost(rss, T, len(c2) - 1, 1, floor)
    return {"D": 0.0, "d": d, "cuts": c2, "rss": rss, "score": sc, "capped": capped} if sc < fit["score"] else fit


def as_flows(fit: dict, row: str, x: np.ndarray, y: np.ndarray, came: np.ndarray, wanted: np.ndarray, at) -> list[dict]:
    """The demand, the own drain and the outside inflow's pieces as flows of the row's stock."""
    u, _ = outside_inflow(fit["D"], fit["d"], x, y, came, wanted)
    out = []
    pieces = []
    for a, b in zip(fit["cuts"][:-1], fit["cuts"][1:]):
        pieces.append({"from": at(a), "to": at(b - 1), "value": max(0.0, float(u[a:b].mean()))})
    W = sum(p["to"] - p["from"] + 1 for p in pieces)
    mean = sum(p["value"] * (p["to"] - p["from"] + 1) for p in pieces) / max(1, W)
    if mean > 0:
        rate = {"form": "constant", "value": round(mean, 4)}
        out.append({"from": None, "to": row, "ratio": 1, "rate": rate, "rates": [rate], "nearly": [], "bothWays": False,
                    "inflow": pieces, "changes": len(pieces) - 1})
    if fit["D"] > 0:
        rate = {"form": "constant", "value": round(fit["D"], 6)}
        out.append({"from": row, "to": None, "ratio": 1, "rate": rate, "rates": [rate], "nearly": [], "bothWays": False, "capped": True})
    if fit["d"] > 0:
        rate = {"form": "proportional", "stock": row, "k": round(fit["d"], 6)}
        out.append({"from": row, "to": None, "ratio": 1, "rate": rate, "rates": [rate], "nearly": [], "bothWays": False})
    return out
