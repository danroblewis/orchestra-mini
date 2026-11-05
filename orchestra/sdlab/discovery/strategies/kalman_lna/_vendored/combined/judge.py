# Vendored for kalman_lna from sdlab/discovery/strategies/combined/judge.py at commit ec2f9e8 (imports rewritten to the
# vendored copies; otherwise unchanged). Do not edit: re-vendor instead.
"""One judge for every candidate: the one-step likelihood of the measured changes, with each
candidate's equations applied exactly as the simulator (and the engine) applies them (floors, caps,
effects, co-flows, unmeasured stocks), plus a BIC price on what the candidate had to write down.

    score = NLL + 0.5 x price x log(T) x numbers(candidate)

The likelihood depends on what the data are:

- counts (whole numbers that move): the linear noise approximation. Each flow moves a Poisson
  number of things, so the change from x_t is Gaussian with mean S a(x_t) and covariance
  S diag(a(x_t)) S^T (a: the flows' amounts, S: where each takes from and gives to, co-flows
  included), plus the rounding quantum's variance. A model with the wrong flows gets the wrong
  covariance even where its means are right (a move against a link, births against deaths);
- anything else (noise-free runs, rounded to a quantum): each stock's changes Gaussian around the
  model's one-step prediction with its own variance concentrated out, floored at the quantum the
  levels are written in (a model exact to the data's precision cannot be told from another one by
  its fit, only by its price).

Unmeasured stocks a candidate adds (a hidden stock, a `via` delay) are run open loop from the
measured levels, starting at their own steady state at the first measured state.
"""

from __future__ import annotations

import math

import numpy as np

from orchestra.sdlab.discovery.simulate import Model, behaviour_error, levels_array
from orchestra.sdlab.discovery.strategies.kalman_lna._vendored.combined.params import numbers

QUANTA = (1.0, 0.1, 0.01, 1e-3, 1e-4, 1e-5, 1e-6)


def is_counts(levels: np.ndarray) -> bool:
    x = levels[np.isfinite(levels)]
    return bool(x.size) and np.abs(x - np.round(x)).max() < 1e-9 and np.abs(np.diff(levels, axis=1)).max() > 0


def quantum(levels: np.ndarray) -> float:
    """The coarsest power of ten every level is written in (1 for counts, 1e-4 for the engine's
    noise-free levels); 1e-7 when none fits."""
    x = levels[np.isfinite(levels)]
    for q in QUANTA:
        if np.abs(x / q - np.round(x / q)).max() < 1e-6:
            return q
    return 1e-7


def expand_via(flows: list[dict]) -> list[dict]:
    """A flow with `via: {delay}` written as a hidden stock: into it at the flow's rate, out of it
    at 1 / delay."""
    out = []
    for f in flows:
        via = f.get("via")
        if not via:
            out.append(f)
            continue
        h = f"{f.get('from')}~via~{f.get('to')}"
        d = max(1e-6, float(via.get("delay") or 1.0))
        g = {k: v for k, v in f.items() if k != "via"}
        out.append({**g, "to": h})
        out.append({"from": h, "to": f.get("to"), "ratio": 1, "rate": {"form": "proportional", "stock": h, "k": 1.0 / d}})
    return out


def _refs(f: dict) -> set:
    r = f.get("rate") or {}
    out = {f.get("from"), f.get("to"), r.get("stock")} | {x.get("stock") for x in r.get("terms") or []}
    for g in f.get("byTime") or []:
        gr = g.get("rate") or {}
        out |= {gr.get("stock")} | {x.get("stock") for x in gr.get("terms") or []}
    out |= {e.get("stock") for e in (f.get("effects") or []) + (f.get("uses") or [])}
    return {s for s in out if s}


def unmeasured(flows: list[dict], data: "Data") -> list[str]:
    """The stocks the flows name that are neither measured nor a measured input."""
    return sorted({s for f in flows for s in _refs(f)} - set(data.ids) - set(data.inputs))


class Data:
    """What the judge reads from a run: levels, the change of every step (injections taken out),
    the times each step uses, whether the levels are counts, and their quantum."""

    def __init__(self, stocks: list[dict], run: dict, injections: list[dict] | None = None, inputs: dict | None = None):
        self.ids = [s["id"] for s in stocks]
        # measured inputs ({name: value per step, aligned with t}) that rates and effects may read
        self.inputs = {k: list(v) for k, v in (inputs or {}).items() if k not in self.ids}
        self.model_inputs = {"t": list(run["t"]), "values": self.inputs} if self.inputs else None
        self.L = levels_array(run["levels"], self.ids)
        self.t = np.asarray(run["t"], dtype=float)
        prev = self.L[:, :-1].copy()
        if injections:
            index = {s: i for i, s in enumerate(self.ids)}
            for j in injections:
                i = index.get(j.get("stock"))
                if i is None:
                    continue
                hit = self.t[1:] == float(j["t"])
                prev[i, hit] = np.maximum(0.0, prev[i, hit] + float(j["amount"]))
        self.prev = prev
        self.dy = self.L[:, 1:] - prev
        self.tt = self.t[1:]
        self.T = self.dy.shape[1]
        self.counts = is_counts(self.L)
        self.q = quantum(self.L)
        self.injections = injections


STARTS = np.concatenate([np.linspace(0.0, 2.0, 21), [3.0, 5.0]])


def hidden_paths(model: Model, data: Data, hidden: list[str], starts: dict | None = None) -> np.ndarray:
    """The unmeasured stocks' levels at every step before the last (K x H x T), run open loop from
    K starting levels at once: the measured stocks set to their data each step, the hidden ones
    stepped by the model. The starts are multiples of the hidden stocks' steady state at the first
    measured state (0 to 5 times it), and the candidate's own estimate when it gives one."""
    S, H = len(data.ids), len(hidden)
    x = np.zeros((S + H, 1))
    x[:S, 0] = data.prev[:, 0]
    for _ in range(5000):                                  # the steady state at the first state
        nxt = model.step(x, data.tt[:1])
        if np.allclose(nxt[S:], x[S:], rtol=1e-9, atol=1e-9):
            break
        x[S:] = nxt[S:]
    ss = x[S:, 0]
    cols = [ss * m for m in STARTS]
    if starts:
        cols.append(np.array([float(starts.get(h, ss[i])) for i, h in enumerate(hidden)]))
    K = len(cols)
    X = np.zeros((S + H, K))
    X[S:] = np.array(cols).T
    out = np.zeros((K, H, data.T))
    for i in range(data.T):
        X[:S] = data.prev[:, i:i + 1]
        out[:, :, i] = X[S:].T
        X[S:] = model.step(X, np.full(K, data.tt[i]))[S:]
    return out


def one_step(flows: list[dict], data: Data, starts: dict | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    """The model's expected change of each measured stock at every step (S x T), and with counts
    the covariance of the change (T x S x S). Unmeasured stocks start where the likelihood is best
    among the starts hidden_paths tries."""
    flows = expand_via(flows)
    hidden = unmeasured(flows, data)
    model = Model(flows, data.ids + hidden, inputs=data.model_inputs)
    if not hidden:
        return _moments(model, data.prev, data)
    return _best_path(model, data, hidden, starts)[1]


def _best_path(model: Model, data: Data, hidden: list[str], starts: dict | None) -> tuple:
    """(nll, moments, path) for the start of the unmeasured stocks the likelihood likes best."""
    best = None
    for path in hidden_paths(model, data, hidden, starts):
        m = _moments(model, np.vstack([data.prev, path]), data)
        v = _nll(m, data)
        if best is None or v < best[0]:
            best = (v, m, path)
    return best


def hidden_estimate(flows: list[dict], data: Data, starts: dict | None = None) -> tuple[list[dict], dict]:
    """The flows (via delays written as stocks) and their unmeasured stocks' levels at every time of
    the run, run open loop from the best start ({} when there are none)."""
    flows = expand_via(flows)
    hidden = unmeasured(flows, data)
    if not hidden:
        return flows, {}
    model = Model(flows, data.ids + hidden, inputs=data.model_inputs)
    path = _best_path(model, data, hidden, starts)[2]
    last = model.step(np.vstack([data.prev[:, -1:], path[:, -1:]]), data.tt[-1:])[len(data.ids):, 0]
    return flows, {h: list(path[i]) + [float(last[i])] for i, h in enumerate(hidden)}


HORIZONS = (5, 20, 100)


def behaviour(flows: list[dict], data: Data, run: dict, starts: dict | None = None) -> float:
    """The main line's behaviour judge (runs from many moments, 5, 20 and 100 steps, against the
    data; 1 = no better than nothing changing), with unmeasured stocks started from their open-loop
    estimate."""
    hs = [h for h in HORIZONS if h < len(data.t) / 4]
    if not flows or not hs:
        return float("inf")
    flows, inferred = hidden_estimate(flows, data, starts)
    return sum(behaviour_error(flows, data.ids, run["t"], run["levels"], horizon=h, injections=data.injections,
                               inferred=inferred or None, inputs=data.inputs or None) for h in hs) / len(hs)


def _moments(model: Model, X: np.ndarray, data: Data) -> tuple[np.ndarray, np.ndarray | None]:
    S = len(data.ids)
    ids = model.ids
    amounts = model._amounts(X, data.tt)
    nxt = X.copy()
    V = np.zeros((len(amounts), len(ids)))
    for k, ((src, dst, ratio, *_), a) in enumerate(zip(model.parts, amounts)):
        if src is not None:
            nxt[src] -= a * ratio
            V[k, src] -= ratio
        if dst is not None:
            nxt[dst] += a
            V[k, dst] += 1.0
    for i, c, per in model.uses:
        nxt[c] -= amounts[i] * per
        V[i, c] -= per
    nxt = np.where(nxt >= 0, nxt, 0.0)
    mu = (nxt - X)[:S]
    if not data.counts:
        return mu, None
    A = np.maximum(0.0, np.array(amounts)) if amounts else np.zeros((0, data.T))   # F x T
    Vm = V[:, :S]
    cov = np.einsum("ft,fi,fj->tij", A, Vm, Vm)
    return mu, cov


def nll(flows: list[dict], data: Data, starts: dict | None = None) -> float:
    """-log L of the measured changes under the candidate (see the module's docstring)."""
    if not flows:
        m = (np.zeros_like(data.dy), (np.zeros((data.T, len(data.ids), len(data.ids))) if data.counts else None))
    else:
        m = one_step(flows, data, starts)
    return _nll(m, data)


def _nll(m: tuple, data: Data) -> float:
    mu, cov = m
    r = data.dy - mu
    if not np.all(np.isfinite(r)):
        return float("inf")
    eps = data.q ** 2 / 6.0
    if cov is not None:
        S = len(data.ids)
        C = cov + eps * np.eye(S)[None]
        sign, logdet = np.linalg.slogdet(C)
        z = np.linalg.solve(C, r.T[:, :, None])[:, :, 0]
        return float(0.5 * (logdet.sum() + (r.T * z).sum() + data.T * S * math.log(2 * math.pi)))
    v = np.maximum((r ** 2).mean(axis=1), eps)
    return float(0.5 * data.T * (np.log(2 * math.pi * v) + 1).sum())


def score(flows: list[dict], data: Data, price: float = 1.0, per_flow: float = 1.0, starts: dict | None = None) -> dict:
    """{nll, k, flows, bic}: bic = nll + 0.5 x price x log(T) x (k - (1 - per_flow) x flows),
    k the candidate's numbers with 1 per flow (params.py)."""
    k = numbers(flows, set(data.ids))
    n = nll(flows, data, starts)
    kk = k - (1.0 - per_flow) * len(flows or [])
    return {"nll": n, "k": k, "flows": len(flows or []), "bic": n + 0.5 * price * math.log(max(2, data.T)) * kk}


def misfit(flows: list[dict], data: Data) -> float:
    """How far the candidate is from fitting the data to its noise, in standard errors: with counts,
    the mean squared standardized change (1 when the model is right) against its sampling spread;
    otherwise the residuals' rms against the data's quantum (about 1 when the model is exact to the
    data's precision), on a log scale."""
    m = one_step(flows, data) if flows else (np.zeros_like(data.dy), np.zeros((data.T, len(data.ids), len(data.ids))) if data.counts else None)
    mu, cov = m
    r = data.dy - mu
    eps = data.q ** 2 / 6.0
    S = len(data.ids)
    if cov is not None:
        C = cov + eps * np.eye(S)[None]
        z = np.linalg.solve(C, r.T[:, :, None])[:, :, 0]
        phi = float((r.T * z).sum() / (data.T * S))
        return (phi - 1.0) / math.sqrt(2.0 / (data.T * S))
    return float(np.log10(max(1.0, math.sqrt((r ** 2).mean() / eps))))


def weights(flows: list[dict], data: Data) -> np.ndarray | None:
    """Per stock and step, 1 / the candidate's one-step standard deviation (counts only)."""
    if not data.counts or not flows:
        return None
    _, cov = one_step(flows, data)
    var = np.einsum("tii->it", cov) + data.q ** 2 / 6.0
    return 1.0 / np.sqrt(var)


def averaged(flows: list[dict]) -> list[dict]:
    """The flows with every found outside-inflow series (pieces) replaced by its mean over the run:
    what the model does when it cannot replay the run's own inflow."""
    out = []
    for f in flows or []:
        segs = f.get("inflow")
        if not segs:
            out.append(f)
            continue
        w = np.array([max(1.0, float(g["to"]) - float(g["from"]) + 1) for g in segs])
        v = np.array([float(g["value"]) for g in segs])
        g = {k: x for k, x in f.items() if k != "inflow"}
        g["rate"] = {"form": "constant", "value": float((w * v).sum() / w.sum())}
        out.append(g)
    return out
