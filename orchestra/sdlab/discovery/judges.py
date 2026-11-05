"""Judges: scoring candidate models on the data, for any strategy's candidate loop.

    from orchestra.sdlab.discovery.judges import score, choose
    errors = score([flows_1, flows_2, ...], stocks, run, judge="common")
    best, errors = choose([...], stocks, run, judge="window")   # index of the lowest error

A candidate is a list of flows. Its outside inflows (flows with `inflow` pieces, as `read_flows`
makes them) are not dynamics: the bench feeds every model the truth's outside inflows, and asks
only how its own dynamics behave given them. A judge that runs each candidate with its OWN pieces
(the main line's, "own") lets pieces carry what the dynamics miss: pieces fitted beside a missing
effect, hidden stock or misread flow imitate it on this run and score well. The other judges take
the pieces out of the candidate's hands:

  own       the candidate's own pieces (the main line's judge, simulate.behaviour_error)
  mean      each piece flow replaced by its mean over the run (models that only work with their own
            pieces fail)
  common    every candidate fed the same outside inflow per stock: the per-step median, over the
            candidates, of each one's net outside inflow (one candidate cannot tailor it to itself)
  anchor    every candidate fed the pieces of the candidate "own" ranks first
  split     the candidate's pieces (same times) with values re-measured on every other step
  refit     the dynamics frozen, the outside inflow re-measured coarsely over the whole run (pieces
            of at least a fiftieth of the run) into the stocks the candidate feeds
  window    on each judged window the dynamics are frozen and only a coarse outside inflow is
            refitted (one number per stock per 25 steps of the window, the same budget for every
            candidate, into the stocks at least half the candidates feed): dynamics given inflows
  own+mean, common+window, ...  the mean of the named judges' errors

On the main line's candidates (benchmark v2.1) every judge but "anchor" picks worse than "own";
see docs/research/inflow-conditioned-judge.md.

Each error is behaviour_error's: runs from ~80 moments, 5, 20 and 100 steps (those shorter than a
quarter of the run), the end compared with what was measured, relative to how far the data moved.
`known` flows (what the page already knows) are dynamics here: they are never replaced.
"""

from __future__ import annotations

import numpy as np

from .changes import pelt
from .simulate import Model, behaviour_error, levels_array, starts_for

HORIZONS = (5, 20, 100)
STARTS = 80


def is_piece(f: dict) -> bool:
    return bool(f.get("inflow")) and not f.get("known")


def dynamics(flows: list[dict]) -> list[dict]:
    return [f for f in flows if not is_piece(f)]


def net_inflow(flows: list[dict], ids: list[str], t: list) -> np.ndarray:
    """Each stock's net outside inflow from the candidate's pieces (stocks x steps): column k is what
    arrives with the step to t[k] (column 0 is unused), as simulate.Model runs the pieces."""
    index = {s: i for i, s in enumerate(ids)}
    tt = np.asarray(t, dtype=float)
    U = np.zeros((len(ids), len(tt)))
    for f in flows:
        if not is_piece(f):
            continue
        segs = sorted(f["inflow"], key=lambda g: g["from"])
        froms = np.array([g["from"] for g in segs], dtype=float)
        tos = np.array([g["to"] for g in segs], dtype=float)
        vals = np.maximum(0.0, np.array([float(g["value"]) for g in segs]))
        pos = np.searchsorted(froms, tt, side="right") - 1
        ok = (pos >= 0) & (tt <= tos[np.clip(pos, 0, None)])
        # a time no segment covers takes the last segment in the list (simulate._Segments)
        last = int(np.argmax([g is f["inflow"][-1] for g in segs]))
        v = vals[np.where(ok, np.clip(pos, 0, None), last)]
        if f.get("to") in index:
            U[index[f["to"]]] += v
        if f.get("from") in index:
            U[index[f["from"]]] -= v * float(f.get("ratio") or 1)
    return U


def mean_inflow(flows: list[dict], ids: list[str], t: list) -> np.ndarray:
    """The candidate's pieces at their mean over the run, per stock."""
    U = net_inflow(flows, ids, t)
    out = np.zeros_like(U)
    out[:, 1:] = U[:, 1:].mean(axis=1, keepdims=True)
    return out


def common_inflow(cands: list[list[dict]], ids: list[str], t: list) -> np.ndarray:
    """Per stock and step, the median over the candidates of their net outside inflow."""
    if not cands:
        return np.zeros((len(ids), len(t)))
    return np.median(np.array([net_inflow(c, ids, t) for c in cands]), axis=0)


def fed(cands: list[list[dict]], ids: list[str]) -> list[int]:
    """The stocks at least half the candidates feed from outside (or drain to outside) in pieces."""
    count = np.zeros(len(ids))
    index = {s: i for i, s in enumerate(ids)}
    for c in cands:
        hit = {index[s] for f in c if is_piece(f) for s in (f.get("to"), f.get("from")) if s in index}
        for i in hit:
            count[i] += 1
    return [i for i in range(len(ids)) if count[i] >= max(1, len(cands)) / 2]


def own_stocks(flows: list[dict], ids: list[str]) -> list[int]:
    """The stocks the candidate feeds from outside (or drains to outside) in pieces."""
    index = {s: i for i, s in enumerate(ids)}
    return sorted({index[s] for f in flows if is_piece(f) for s in (f.get("to"), f.get("from")) if s in index})


def refit_inflow(flows: list[dict], ids: list[str], t: list, levels: dict, feed: list[int], min_len: int | None = None,
                 mult: float = 1.0) -> np.ndarray:
    """The outside inflow re-measured with the candidate's dynamics frozen, coarsely: into each stock
    in `feed`, pieces of at least `min_len` steps (default: a fiftieth of the run, at least 20) found
    in what the dynamics leave unexplained one step at a time, each piece's mean; elsewhere none."""
    A = levels_array(levels, ids)
    S, n = A.shape
    U = np.zeros((S, n))
    if n < 3 or not feed:
        return U
    tt = np.asarray(t, dtype=float)
    model = Model(dynamics(flows), ids)
    res = np.diff(A, axis=1) - (model.step(A[:, :-1], tt[1:]) - A[:, :-1])
    ml = min_len or max(20, n // 50)
    for i in feed:
        cuts = pelt(res[i], ml, mult)
        for a, b in zip(cuts[:-1], cuts[1:]):
            U[i, a + 1:b + 1] = res[i, a:b].mean()
    return U


def half_inflow(flows: list[dict], ids: list[str], t: list, levels: dict, parity: int = 0) -> np.ndarray:
    """The candidate's pieces (same times) with each piece's value re-measured on half the steps (every
    other step, `parity`): what the dynamics leave unexplained one step at a time, averaged. A piece
    that is a real inflow measures the same on either half; one that fits its own steps' draws does not."""
    A = levels_array(levels, ids)
    S, n = A.shape
    U = net_inflow(flows, ids, t)
    if n < 3:
        return U
    tt = np.asarray(t, dtype=float)
    model = Model(dynamics(flows), ids)
    res = np.zeros((S, n))
    res[:, 1:] = np.diff(A, axis=1) - (model.step(A[:, :-1], tt[1:]) - A[:, :-1])
    use = (np.arange(n) % 2 == parity) & (np.arange(n) > 0)
    out = np.zeros_like(U)
    for i in range(S):
        if not U[i, 1:].any():
            continue
        # the candidate's pieces: runs of one value
        k = 1
        while k < n:
            e = k
            while e + 1 < n and U[i, e + 1] == U[i, k]:
                e += 1
            m = use[k:e + 1]
            out[i, k:e + 1] = res[i, k:e + 1][m].mean() if m.any() else U[i, k]
            k = e + 1
    return out


def _run(model: Model, L: np.ndarray, times: np.ndarray, U: np.ndarray, every: bool = False):
    """Steps from L (stocks x states) at times (steps x states), U (steps x stocks x states) arriving
    from outside with each step; levels after every step (every) or at the end."""
    out = []
    for k in range(times.shape[0]):
        L = model.step(L, times[k]) + U[k]
        L = np.where(L >= 0, L, 0.0)
        if every:
            out.append(L)
    return np.array(out) if every else L


def _error(end: np.ndarray, obs: np.ndarray, before: np.ndarray, sd: np.ndarray) -> float:
    e = ((end - obs) / sd[:, None]) ** 2
    e[~np.isfinite(e)] = 1e6
    moved = (((obs - before) / sd[:, None]) ** 2).sum()
    return float(e.sum() / max(1e-9, moved))


def exo_error(flows: list[dict], ids: list[str], t: list, levels: dict, U: np.ndarray, horizon: int, starts: int = STARTS) -> float:
    """behaviour_error with the outside inflow U (stocks x steps, as net_inflow) in place of the
    candidate's pieces."""
    A = levels_array(levels, ids)
    n = A.shape[1]
    s0 = starts_for(n, horizon, starts)
    if not len(s0):
        return float("inf")
    sd = A.std(axis=1)
    sd[sd == 0] = 1
    tt = np.asarray(t, dtype=float)
    idx = s0[None, :] + np.arange(horizon)[:, None]               # steps x starts: the step's arrival column
    end = _run(Model(dynamics(flows), ids), A[:, s0 - 1], tt[idx], np.transpose(U[:, idx], (1, 0, 2)))
    return _error(end, A[:, s0 + horizon - 1], A[:, s0 - 1], sd)


def coarse_error(flows: list[dict], ids: list[str], t: list, levels: dict, feed: list[int], horizon: int,
                 pieces: int | None = None, starts: int = STARTS) -> float:
    """Dynamics given inflows: from each start, the candidate's dynamics run `horizon` steps with an
    outside inflow into each stock in `feed` that is constant over each of `pieces` equal blocks of the
    window, chosen (least squares on the whole window's levels, linearised by one extra run per
    number) to follow the data best; the end is then compared as behaviour_error compares it."""
    A = levels_array(levels, ids)
    S, n = A.shape
    s0 = starts_for(n, horizon, starts)
    if not len(s0):
        return float("inf")
    m = len(s0)
    sd = A.std(axis=1)
    sd[sd == 0] = 1
    K = pieces or max(1, horizon // 25)
    block = np.minimum(np.arange(horizon) * K // horizon, K - 1)   # which block each step is in
    P = [(i, b) for i in feed for b in range(K)]
    tt = np.asarray(t, dtype=float)
    idx = s0[None, :] + np.arange(horizon)[:, None]
    obs = np.transpose(A[:, idx], (1, 0, 2))                       # steps x stocks x starts
    model = Model(dynamics(flows), ids)
    # a unit of inflow per step, in each stock's own scale of change
    du = np.maximum(1.0, np.abs(np.diff(A, axis=1)).mean(axis=1))
    cols = 1 + len(P)
    L0 = np.tile(A[:, s0 - 1], cols)
    U = np.zeros((horizon, S, m * cols))
    for q, (i, b) in enumerate(P):
        U[block == b, i, (q + 1) * m:(q + 2) * m] = du[i]
    traj = _run(model, L0, np.tile(tt[idx], cols), U, every=True)  # steps x stocks x (starts x cols)
    base = traj[..., :m]
    if P:
        R = np.stack([(traj[..., (q + 1) * m:(q + 2) * m] - base) / du[P[q][0]] for q in range(len(P))], -1)
        Rz = R / sd[None, :, None, None]
        rz = (obs - base) / sd[None, :, None]
        G = np.einsum("hsmp,hsmq->mpq", Rz, Rz)
        g = np.einsum("hsmp,hsm->mp", Rz, rz)
        ridge = 1e-6 * np.trace(G, axis1=1, axis2=2)[:, None, None] / len(P) + 1e-12
        u = np.linalg.solve(G + ridge * np.eye(len(P))[None], g[..., None])[..., 0]   # starts x pieces
        # where the linearisation fails (a stock at its floor answers nothing), keep the numbers to
        # what an inflow into the stock could plausibly be: twice its largest step changes
        bound = np.array([2 * np.percentile(np.abs(np.diff(A[i])), 99) + du[i] for i, _ in P])
        u = np.clip(u, -bound, bound)
        Uf = np.zeros((horizon, S, m))
        for q, (i, b) in enumerate(P):
            Uf[block == b, i, :] = u[:, q][None, :]
        end = _run(model, A[:, s0 - 1], tt[idx], Uf)
    else:
        end = base[-1]
    return _error(end, A[:, s0 + horizon - 1], A[:, s0 - 1], sd)


def _horizons(run: dict, horizons) -> list[int]:
    return [h for h in horizons if h < len(run["t"]) / 4]


def score(cands: list[list[dict]], stocks: list[dict], run: dict, judge: str = "common", horizons=HORIZONS,
          extra: list[dict] | None = None) -> list[float]:
    """Each candidate's error under `judge` (lower is better; inf for an empty candidate). `extra`
    flows (known changes) are added to every candidate as dynamics."""
    ids = [s["id"] for s in stocks]
    t, levels = run["t"], run["levels"]
    hs = _horizons(run, horizons)
    extra = list(extra or [])
    parts = judge.split("+")
    common = common_inflow(cands, ids, t) if "common" in parts else None
    anchor = None
    if "anchor" in parts:
        own = score(cands, stocks, run, "own", horizons, extra)
        anchor = net_inflow(cands[int(np.argmin(own))], ids, t)
    feed = fed(cands, ids) if "window" in parts else None

    def inflow(p: str, c: list[dict]) -> np.ndarray | None:
        """The outside inflow judge part p feeds candidate c (None: its own pieces, or refitted per window)."""
        if p == "mean":
            return mean_inflow(c, ids, t)
        if p == "common":
            return common
        if p == "anchor":
            return anchor
        if p == "split":
            return half_inflow(c + extra, ids, t, levels)
        if p == "refit":
            return refit_inflow(c + extra, ids, t, levels, own_stocks(c, ids))
        if p in ("own", "window"):
            return None
        raise ValueError(f"unknown judge {p!r}")

    out = []
    for c in cands:
        if not c or not hs:
            out.append(float("inf"))
            continue
        flows = c + extra
        errs = []
        for p in parts:
            U = inflow(p, c)
            for h in hs:
                if p == "own":
                    errs.append(behaviour_error(flows, ids, t, levels, horizon=h))
                elif p == "window":
                    errs.append(coarse_error(flows, ids, t, levels, feed, h))
                else:
                    errs.append(exo_error(flows, ids, t, levels, U, h))
        out.append(float(np.mean(errs)))
    return out


def choose(cands: list[list[dict]], stocks: list[dict], run: dict, judge: str = "common", **kw) -> tuple[int, list[float]]:
    """The index of the candidate with the lowest error, and every candidate's error."""
    errs = score(cands, stocks, run, judge, **kw)
    return int(np.argmin(errs)), errs
