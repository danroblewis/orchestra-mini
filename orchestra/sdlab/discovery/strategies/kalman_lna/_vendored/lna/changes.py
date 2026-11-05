# Vendored for kalman_lna from sdlab/discovery/strategies/lna/changes.py at commit ec2f9e8 (imports rewritten to the
# vendored copies; otherwise unchanged). Do not edit: re-vendor instead.
"""What changes over time: outside inflows in pieces (unannounced changes), and numbers per announced
era. Both are kept only where the likelihood pays for them (BIC)."""

from __future__ import annotations

import math

import numpy as np

from orchestra.sdlab.discovery.changes import pelt
from orchestra.sdlab.discovery.strategies.kalman_lna._vendored.lna.atoms import Atom, features, stoich, variances
from orchestra.sdlab.discovery.strategies.kalman_lna._vendored.lna.fit import TRIAL, Data, Fitted, fit, polish


SCREEN = 0.25        # the share of the BIC price the pieces' gain before refitting must reach


def _moments(cur: Fitted, d: Data) -> tuple[np.ndarray, np.ndarray]:
    """Each stock's residual and its counting-noise standard deviation, step by step."""
    S = d.dy.shape[0]
    V, Z = stoich(cur.atoms, S), features(cur.atoms, d.prev, d.cfg)
    q = cur.theta[:, None] * Z
    return d.dy - V @ q, np.sqrt(variances(cur.atoms, cur.theta, Z, S) + d.delta)


def pieces(cur: Fitted, d: Data, min_len: int = 5, mult: float = 1.0, price: float | None = None) -> Fitted:
    """Each stock's outside inflow cut afresh where what the other flows leave unexplained changes
    level (PELT on the residual standardised by its counting noise), kept if the likelihood gains
    more than the BIC price of the extra pieces."""
    S, T = d.dy.shape
    price = 0.5 * math.log(T) if price is None else price
    for s in range(S):
        idx = [i for i, a in enumerate(cur.atoms) if a.kind == "in" and a.dst == s]
        if not idx:
            continue
        r, sd = _moments(cur, d)
        Z = features([cur.atoms[i] for i in idx], d.prev, d.cfg)
        own = cur.theta[idx] @ Z                         # the inflow as it stands, added back
        y = r[s] + own
        keep = [i for i in range(len(cur.atoms)) if i not in idx]
        n_old = sum(1 for i in idx if cur.theta[i] > 0)
        best = None
        for cuts in (pelt(y / sd[s], min_len, mult), [0, T]):
            new = [(a, b, max(1e-3, float(y[a:b].mean()))) for a, b in zip(cuts[:-1], cuts[1:])]
            atoms = [cur.atoms[i] for i in keep] + [Atom("in", dst=s, span=None if len(new) == 1 else (a, b)) for a, b, _ in new]
            if sorted(atoms, key=repr) == sorted(cur.atoms, key=repr):
                continue
            if len(new) > n_old:
                # what the pieces would gain at most, in the standardised residual (the inflow as
                # it stands taken out; a refit moves the other flows too, so the bar is lower than
                # the price): fit only cuts that could pay for themselves
                z = (y - own) / sd[s]
                gain = 0.5 * sum((b - a) * float(z[a:b].mean()) ** 2 for a, b, _ in new)
                if gain < SCREEN * price * (len(new) - n_old):
                    continue
            g = fit(atoms, d, np.append(cur.theta[keep], [v for _, _, v in new]), tau=cur.tau, loose=True)
            gain = cur.nll - g.nll - price * (len(new) - n_old)
            if gain > 0 and (best is None or gain > best[0]):
                best = (gain, g)
        if best is not None:
            cur = polish(best[1], d)
    return cur


def split_eras(atoms: list[Atom], labels: int) -> list[Atom]:
    """Every atom with a number per era."""
    return [a.with_(era=c) for a in atoms for c in range(labels)]


def merge_eras(cur: Fitted, d: Data) -> Fitted:
    """A flow whose numbers per era the data does not tell apart gets one number (greedy, the
    cheapest merge first, while it costs less than the BIC price of the numbers it saves)."""
    if d.cfg is None:
        return cur
    T = d.dy.shape[1]
    price = 0.5 * math.log(T)
    while True:
        best = None
        for key in dict.fromkeys(a.key() for a in cur.atoms if a.era is not None):
            split = [i for i, a in enumerate(cur.atoms) if a.key() == key and a.era is not None]
            if len(split) < 2:
                continue
            rest = [j for j in range(len(cur.atoms)) if j not in split]
            th = float(np.mean([cur.theta[i] for i in split]))
            g = fit([cur.atoms[j] for j in rest] + [cur.atoms[split[0]].with_(era=None)], d, np.append(cur.theta[rest], th), maxiter=TRIAL, tau=cur.tau)
            cost = (g.nll - cur.nll) / (len(split) - 1)
            if cost < price and (best is None or cost < best[0]):
                best = (cost, g)
        if best is None:
            return cur
        cur = polish(best[1], d)
