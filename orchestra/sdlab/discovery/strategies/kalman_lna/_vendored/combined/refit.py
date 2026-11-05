# Vendored for kalman_lna from sdlab/discovery/strategies/combined/refit.py at commit ec2f9e8 (imports rewritten to the
# vendored copies; otherwise unchanged). Do not edit: re-vendor instead.
"""Every candidate's numbers refitted on the same terms before it is judged.

The specialists hand back numbers fitted their own way, some rounded to four significant figures.
On noise-free data that rounding alone costs a candidate thousands of nats against one fitted to
the data's precision, so the judge would be comparing how numbers were written, not structures.
Here each candidate's structure is kept and its numbers are fitted again by (weighted) least
squares of the one-step changes, all stocks together:

    dy_s(t) = sum over flows f of V[f, s] x m_f(t) x (theta_f . psi_f(t))

where V is where flow f takes from and gives to (co-flows included), m_f its effects at x_t (held
as found), and psi_f its rate's columns: 1 for a constant, the level it reads for a term, a gap
(goal - x) / time as a constant and a term; one set of columns per announced era segment, and one
value per distinct found outside-inflow value (a repeating profile keeps its values tied). Flows
into or out of unmeasured stocks and `available` rates keep their numbers. With counts each row is
weighted by the candidate's own one-step variance. Floors and caps are not in the fit; the judge
keeps the refitted numbers only if they make the candidate's likelihood better.
"""

from __future__ import annotations

import copy

import numpy as np

from orchestra.sdlab.discovery.simulate import Model, _Effects
from orchestra.sdlab.discovery.strategies.kalman_lna._vendored.combined.judge import Data, _refs, expand_via, unmeasured


def _rate_cols(rate: dict, X: np.ndarray, index: dict) -> list[tuple]:
    """The rate's linear columns: (setter, column); None if it cannot be refitted linearly."""
    form = rate.get("form")
    T = X.shape[1]
    one = np.ones(T)

    def lv(s):
        return X[index[s]] if s in index else None
    if form == "constant":
        return [(("value",), one)]
    if form == "proportional":
        x = lv(rate.get("stock"))
        return None if x is None else [(("k",), x)]
    if form == "linear":
        cols = [(("c",), one)] if rate.get("c") else []
        for j, t in enumerate(rate.get("terms") or []):
            x = lv(t.get("stock"))
            if x is None:
                return None
            cols.append((("terms", j), x))
        return cols
    if form == "gap":
        x = lv(rate.get("stock"))
        return None if x is None else [(("gap_c",), one), (("gap_k",), -x)]
    return None


def _set(rate: dict, key: tuple, v: float, others: dict) -> None:
    if key[0] == "terms":
        rate["terms"][key[1]]["k"] = v
    elif key[0] in ("gap_c", "gap_k"):
        others[key[0]] = v
    else:
        rate[key[0]] = v


def refit(flows: list[dict], data: Data, weights: np.ndarray | None = None) -> list[dict] | None:
    """The flows with their numbers refitted (a deep copy), or None when nothing could be."""
    if not flows:
        return None
    flows = expand_via(copy.deepcopy(flows))
    ids = data.ids
    hidden = unmeasured(flows, data)
    S, T = len(ids), data.T
    names = list(data.inputs)
    # the levels a rate may read: the stocks at the start of each step, then the inputs at its time
    X = np.vstack([data.prev] + [np.asarray(data.inputs[k], dtype=float)[1:][None, :] for k in names]) if names else data.prev
    index = {s: i for i, s in enumerate(ids + names)}
    t = data.tt
    model = Model(flows, ids + hidden, inputs=data.model_inputs)
    V = np.zeros((len(flows), S))
    for k, (src, dst, ratio, *_) in enumerate(model.parts):
        if src is not None and src < S:
            V[k, src] -= ratio
        if dst is not None and dst < S:
            V[k, dst] += 1.0
    for i, c, per in model.uses:
        if c < S:
            V[i, c] -= per
    fixed = np.zeros((S, T))
    cols, setters = [], []
    for k, f in enumerate(flows):
        mult = np.ones(T)
        if f.get("effects"):
            mult = _Effects(f["effects"], index)(X)
        refs = _refs(f)
        own = None
        if not (refs & set(hidden)) and not np.allclose(V[k], 0):
            if f.get("inflow"):
                own = []
                segs = f["inflow"]
                vals = {}
                for g in segs:
                    vals.setdefault(round(float(g["value"]), 9), []).append(g)
                from orchestra.sdlab.discovery.simulate import _Segments
                which = _Segments(segs)(t)
                for v, gs in vals.items():
                    ids_g = [segs.index(g) for g in gs]
                    own.append((("inflow", tuple(ids_g)), np.isin(which, ids_g).astype(float)))
            elif f.get("byTime"):
                from orchestra.sdlab.discovery.simulate import _Segments
                segs = f["byTime"]
                which = _Segments(segs)(t)
                own = []
                for j, g in enumerate(segs):
                    rc = _rate_cols(g.get("rate") or {}, X, index)
                    if rc is None:
                        own = None
                        break
                    own += [(("byTime", j) + key, col * (which == j)) for key, col in rc]
            else:
                own = _rate_cols(f.get("rate") or {}, X, index)
        if own is None:
            amt = Model([f], ids, inputs=data.model_inputs)._amounts(data.prev, t)[0] if not refs & set(hidden) else None
            if amt is None:
                return None                                  # a flow through an unmeasured stock: not refitted here
            fixed += V[k][:, None] * amt[None, :]
            continue
        for key, col in own:
            cols.append((V[k][:, None] * (col * mult)[None, :]).ravel())
            setters.append((k, key))
    if not cols:
        return None
    A = np.array(cols).T
    y = (data.dy - fixed).ravel()
    if weights is not None:
        w = weights.ravel()
        A, y = A * w[:, None], y * w
    theta, *_ = np.linalg.lstsq(A, y, rcond=None)
    gaps: dict = {}
    for (k, key), v in zip(setters, theta):
        f = flows[k]
        if key[0] == "inflow":
            for j in key[1]:
                f["inflow"][j]["value"] = float(v)
        elif key[0] == "byTime":
            r = f["byTime"][key[1]]["rate"]
            other = gaps.setdefault((k, key[1]), {})
            _set(r, key[2:], float(v), other)
        else:
            _set(f.setdefault("rate", {}), key, float(v), gaps.setdefault((k, None), {}))
    for (k, j), g in gaps.items():
        if "gap_c" in g and "gap_k" in g:
            r = flows[k]["rate"] if j is None else flows[k]["byTime"][j]["rate"]
            kk = g["gap_k"]
            if kk <= 0:
                return None
            r["time"] = 1.0 / kk
            r["goal"] = g["gap_c"] / kk
    return flows
