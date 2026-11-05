# Vendored for kalman_lna from sdlab/discovery/strategies/hidden_transit/fit.py at commit ec2f9e8 (imports rewritten to the
# vendored copies; otherwise unchanged). Do not edit: re-vendor instead.
"""Fitting a pair's transit model (and its plain alternative) by the filter's likelihood.

Starting points come from a least-squares profile over the delay: with d given, H is a filter of A
(G(t) = (1 - d) G(t-1) + A(t-1), H = kin G + H(0) (1 - d)^t), so B's row is linear in kin d, H(0) d,
kb (and cB). Its minima over d are the starts. B's response to A is symmetric in d and kb
(kin d / ((z - 1 + d)(z - 1 + kb))): the mirror (d and kb swapped, kin scaled by kb / d) fits B as
well, and only A's balance (kin <= kA; "all of A's loss goes to H" is the simpler model) tells the two
apart; so both are tried.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize
from scipy.signal import lfilter

from orchestra.sdlab.discovery.strategies.kalman_lna._vendored.hidden_transit.kalman import Pair

D_MAX = 0.9           # a delay of at least ~1.1 steps (shorter is a move)


def _sig(x):
    return 1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, x))))


def _exp(x):
    return math.exp(max(-30.0, min(5.0, x)))


def _logit(p):
    p = min(1 - 1e-9, max(1e-9, p))
    return math.log(p / (1 - p))


def _log(x):
    return math.log(max(x, 1e-6))


def a_loss(pair: Pair) -> float:
    """A's loss rate by least squares, beside its pieces."""
    y = np.diff(pair.A) - pair.xA
    cols = [-pair.A[:-1]] + ([np.eye(pair.pieces)[pair.seg].T[j] for j in range(pair.pieces)] if pair.pieces else [])
    beta = np.linalg.lstsq(np.column_stack(cols), y, rcond=None)[0]
    return max(1e-4, float(beta[0]))


def _ls_b(pair: Pair, d: float, cB_on: bool):
    A = pair.A
    G = np.zeros_like(A)
    G[1:] = lfilter([1.0], [1.0, -(1.0 - d)], A[:-1])
    y = np.diff(pair.B) - pair.xB
    t = np.arange(len(y))
    cols = [d * G[:-1], d * (1.0 - d) ** t, -pair.B[:-1]] + ([np.ones(len(y))] if cB_on else [])
    X = np.column_stack(cols)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    r = y - X @ beta
    return float(r @ r), beta


def profile(pair: Pair, cB_on: bool, grid: int = 60) -> list[dict]:
    """Local minima of B's least-squares error over the delay (best first), each with kin, H(0),
    kb, cB."""
    lds = np.linspace(math.log(0.004), math.log(D_MAX), grid)
    vals = [_ls_b(pair, math.exp(x), cB_on)[0] for x in lds]
    out = []
    for i in range(grid):
        if (i == 0 or vals[i] <= vals[i - 1]) and (i == grid - 1 or vals[i] <= vals[i + 1]):
            d = math.exp(lds[i])
            rss, beta = _ls_b(pair, d, cB_on)
            out.append({"d": d, "kin": float(beta[0]), "m0": float(beta[1]), "kb": float(beta[2]),
                        "cB": float(beta[3]) if cB_on else 0.0, "rss": rss})
    out.sort(key=lambda x: x["rss"])
    return out[:3]


def screen(pair: Pair, T: int) -> bool:
    """A cheap first look: does B's row fit better with a delayed copy of A (the profile's best) than
    with A's level itself, both beside B's level and a constant, by more than log T (BIC)?"""
    y = np.diff(pair.B) - pair.xB
    X = np.column_stack([pair.A[:-1], -pair.B[:-1], np.ones(len(y))])
    r = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    plain = float(r @ r)
    prof = profile(pair, True, grid=30)
    if not prof:
        return False
    return T * math.log(max(plain, 1e-300) / max(prof[0]["rss"], 1e-300)) > 2 * math.log(T)


def _minimize(cost, x0):
    """A quasi-Newton descent (numerical gradients), then a short simplex polish (the likelihood has
    kinks where a variance meets its floor)."""
    o = minimize(cost, x0, method="L-BFGS-B", options={"maxiter": 400})
    p = minimize(cost, o.x, method="Nelder-Mead", options={"xatol": 1e-7, "fatol": 1e-7, "maxfev": 600, "adaptive": True})
    return p if p.fun <= o.fun else o


def fit_transit(pair: Pair, cB_on: bool, all_to_h: bool = True, starts: list[dict] | None = None) -> dict | None:
    """The transit model's numbers by maximum likelihood: {kA, kin, d, kb, cB, m0, values (A's inflow
    pieces), nll, k (numbers searched), H (the filtered level)}. all_to_h: kin = kA (A loses only what
    H takes)."""
    kA0 = a_loss(pair)
    starts = starts if starts is not None else profile(pair, cB_on)
    if not starts:
        return None
    tried = []
    for s in starts:
        tried.append(s)
        if s["d"] < D_MAX and s["kb"] > 0 and s["kin"] > 0:                    # the mirror
            tried.append({**s, "d": min(D_MAX * 0.99, s["kb"]), "kb": s["d"], "kin": s["kin"] * s["d"] / s["kb"],
                          "m0": s["m0"] * s["d"] / s["kb"]})

    def unpack(p):
        kA = _exp(p[0])
        i = 1
        if all_to_h:
            kin = kA
        else:
            kin = kA * _sig(p[i])
            i += 1
        d = D_MAX * _sig(p[i])
        kb = _exp(p[i + 1])
        i += 2
        cB = p[i] if cB_on else 0.0
        i += cB_on
        return kA, kin, d, kb, cB, p[i]

    def cost(p):
        v = pair.transit(*unpack(p))[0]
        return v if math.isfinite(v) else 1e300

    best = None
    for s in tried:
        kA = max(kA0, 1e-4)
        x0 = [_log(kA)] + ([] if all_to_h else [_logit(min(0.999, max(0.01, s["kin"] / kA)))])
        x0 += [_logit(min(0.999, s["d"] / D_MAX)), _log(max(s["kb"], 1e-4))] + ([s["cB"]] if cB_on else []) + [s["m0"]]
        o = _minimize(cost, np.array(x0))
        if best is None or o.fun < best.fun:
            best = o
    if best is None or not math.isfinite(best.fun) or best.fun >= 1e299:
        return None
    kA, kin, d, kb, cB, m0 = unpack(best.x)
    nll, H = pair.transit(kA, kin, d, kb, cB, m0)
    return {"kA": kA, "kin": kin, "d": d, "kb": kb, "cB": cB, "m0": m0, "values": pair.values(kA).tolist(), "nll": nll,
            "k": len(best.x), "H": H}


def fit_plain(pair: Pair, cB_on: bool) -> dict:
    """The alternative without H: B gets kd x A straight from A (a move; the rest of A's loss to
    outside), by the same likelihood."""
    kA0 = a_loss(pair)
    y = np.diff(pair.B) - pair.xB
    cols = [pair.A[:-1], -pair.B[:-1]] + ([np.ones(len(y))] if cB_on else [])
    beta = np.linalg.lstsq(np.column_stack(cols), y, rcond=None)[0]

    def unpack(p):
        kA = _exp(p[0])
        kd = kA * _sig(p[1])
        kb = _exp(p[2])
        cB = p[3] if cB_on else 0.0
        return kA, kd, kb, cB

    def cost(p):
        v = pair.plain(*unpack(p))
        return v if math.isfinite(v) else 1e300

    kA = max(kA0, 1e-4)
    x0 = [_log(kA), _logit(min(0.999, max(0.001, float(beta[0]) / kA))), _log(max(float(beta[1]), 1e-4))]
    x0 += [float(beta[2])] if cB_on else []
    o = _minimize(cost, np.array(x0))
    kA, kd, kb, cB = unpack(o.x)
    return {"kA": kA, "kd": kd, "kb": kb, "cB": cB, "nll": float(o.fun), "k": len(o.x)}
