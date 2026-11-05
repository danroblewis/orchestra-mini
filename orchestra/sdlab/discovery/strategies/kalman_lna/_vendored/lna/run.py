# Vendored for kalman_lna from sdlab/discovery/strategies/lna/run.py at commit ec2f9e8 (imports rewritten to the
# vendored copies; otherwise unchanged). Do not edit: re-vendor instead.
# (kalman_lna, 2026-10-20) Edited after vendoring for the declared boundary: atoms outside it are not offered
# (kalman_lna/inside.py); see docs/research/kalman_lna.md.
"""The strategy's steps: the data, the noise check, the atoms chosen, co-flows, the flows."""

from __future__ import annotations

import math
import time

import numpy as np

from orchestra.sdlab.discovery.matrix import configurations, injected
from orchestra.sdlab.discovery.strategies.kalman_lna._vendored.lna.atoms import Atom, features, restricted, stoich, superset, variances
from ...capmin import readers
from orchestra.sdlab.discovery.strategies.kalman_lna._vendored.lna.fit import Data, OutOfTime, extra_variance, fit, mean_start, numbers, prune
from orchestra.sdlab.discovery.strategies.kalman_lna._vendored.lna.changes import merge_eras, pieces, split_eras
from orchestra.sdlab.discovery.strategies.kalman_lna._vendored.lna.coflows import refine_per, with_coflows
from orchestra.sdlab.discovery.strategies.kalman_lna._vendored.lna.flows import to_flows
from orchestra.sdlab.discovery.strategies.kalman_lna._vendored.lna.swaps import swaps
from orchestra.sdlab.discovery.strategies.kalman_lna.boundary import Boundary
from orchestra.sdlab.discovery.strategies.kalman_lna.inside import Allow


def dispersion(atoms: list[Atom], theta: np.ndarray, d: Data) -> float:
    """What the changes' spread is of what counting noise would give the flows (each stock's squared
    residuals over the variance its flows would draw): about 1 under counting noise, 0 without."""
    S, T = d.dy.shape
    V, Z = stoich(atoms, S), features(atoms, d.prev, d.cfg)
    r = d.dy - V @ (theta[:, None] * Z)
    var = variances(atoms, theta, Z, S)
    # the residuals' step-to-step part (half the variance of their differences): counting noise is
    # white, while a misfit (a floor, an effect the atoms lack) is smooth and barely counts here
    # and a few steps where it jumps (a stock that empties) do not count either: the median step,
    # scaled so that counting noise gives about 1 (chi-square with S degrees of freedom, roughly)
    white = 0.5 * (np.diff(r, axis=1) ** 2).sum(axis=0) / np.maximum(1e-12, var[:, 1:].sum(axis=0))
    return float(np.median(white) / (1 - 2 / (9 * S)) ** 3)


def screen(prev: np.ndarray, dy: np.ndarray, z: float = 2.5) -> list[dict[int, float]]:
    """Each row regressed on every level and a constant: the levels with |t| > z (robust standard
    errors), and the row's own level always."""
    S, T = prev.shape
    X = np.vstack([prev, np.ones(T)]).T
    XtX = np.linalg.pinv(X.T @ X)
    out = []
    for s in range(S):
        b = XtX @ X.T @ dy[s]
        e = dy[s] - X @ b
        cov = XtX @ (X.T * e**2) @ X @ XtX
        se = np.sqrt(np.maximum(np.diag(cov), 1e-30))
        out.append({j: float(b[j]) for j in range(S) if j == s or abs(b[j] / se[j]) > z})
    return out


def misfit_of(cur, d: Data) -> float:
    """The largest share, over the stocks, of variance beyond counting noise in the counting variance
    the flows give (0 where the atoms describe the system)."""
    Z = features(cur.atoms, d.prev, d.cfg)
    counting = variances(cur.atoms, cur.theta, Z, d.dy.shape[0]).mean(axis=1) + d.delta
    return float(np.max(np.where(d.free, cur.tau, 0.0) / counting))


def buckets(cfg: np.ndarray, L: int, least: int = 50, share: float = 0.05, most: int = 4) -> list[int]:
    """Each era label's own numbers only if it holds long enough to measure them; the short ones
    share one set (a label held for a few steps would let any reading fit those steps). At most
    `most` sets in all (the longest-held labels keep their own; every atom is split per set, so
    many sets make the search slow)."""
    n = np.bincount(cfg, minlength=L)
    long = [c for c in range(L) if n[c] >= max(least, share * len(cfg))]
    if len(long) > most:
        long = sorted(sorted(long, key=lambda c: -n[c])[: most - 1])
    out = [long.index(c) if c in long else len(long) for c in range(L)]
    return out


# The stages a variant may add (discover(..., hooks={name: callable})), in the order they run:
#   data(prev, dy, ctx) -> (prev, dy)    the data before anything is fitted: e.g. known changes taken
#                                        out of dy, or extra rows appended to prev (columns that atoms
#                                        may use as drivers: index >= the stocks'; name them in
#                                        ctx["names"], the stocks' ids first)
#   atoms(atoms, d, ctx) -> atoms        the candidate atoms, before the noise check and the first fit
#   before_search(full, d, ctx) -> Fitted   the fitted superset, before pieces and pruning
#   after_search(cur, d, ctx) -> Fitted     the chosen atoms (e.g. close loops, offer more and search)
#   flows(flows, cur, d, ctx) -> flows   the flows written from the atoms (a post-fit rewrite)
HOOKS = ("data", "atoms", "before_search", "after_search", "flows")

MAX_MISFIT = 2.0     # past it (with every atom fitted), the matrix's answer is returned
MAX_STEPS = 3000     # longer runs are read at a stride (the ladder's runs are at most 2000 steps)
BUDGET = 40.0        # seconds for the search; past it, the matrix's answer is returned (opts "lna_budget")

SCRATCH = 4          # up to this many stocks, every atom between them is tried


def search(cur, d: Data):
    """From a fitted superset to the chosen atoms: co-flows offered, pruning, pieces re-cut and gap
    swaps until nothing changes, co-flow ratios refined, era numbers merged."""
    cur = prune(with_coflows(cur, d), d)
    for _ in range(3):
        before = cur.atoms
        cur = prune(pieces(prune(swaps(cur, d), d), d), d)
        if cur.atoms == before:
            break
    cur = prune(refine_per(cur, d), d)
    return merge_eras(cur, d)


def bic(cur, d: Data) -> float:
    return 2 * cur.nll + numbers(cur.atoms, range(len(cur.atoms))) * math.log(d.dy.shape[1])


def discover(stocks: list[dict], run: dict, opts: dict, base: dict, hooks: dict | None = None) -> dict | None:
    try:
        return _discover(stocks, run, opts, base, hooks)
    except OutOfTime:
        return None                    # past the budget: the matrix's answer stands


def _discover(stocks: list[dict], run: dict, opts: dict, base: dict, hooks: dict | None = None) -> dict | None:
    """The found model, or None (too short, or no counting noise to read).

    hooks (optional; see HOOKS): stages a variant adds without copying this function. Each gets
    `ctx`, a dict with stocks, ids, t, opts, base, unit, final, group, conf, bucket and (once made)
    d and at, which it may also write to (e.g. ctx["names"]). With no hooks, lna's own behaviour."""
    hooks = hooks or {}
    ids = [s["id"] for s in stocks]
    S = len(ids)
    t = run["t"]
    if len(t) < 30:
        return None
    lv = np.nan_to_num(np.array([np.asarray(run["levels"].get(i, [0] * len(t)), dtype=float) for i in ids]))
    prev, dy = lv[:, :-1], np.diff(lv, axis=1)
    if opts.get("injections"):
        inj = injected(opts["injections"], ids, t)[:, 1:]
        prev, dy = prev + inj, dy - inj
    conf = configurations(opts.get("eras"), t)
    bucket = buckets(conf[1], len(conf[0])) if conf else None
    if bucket is not None and max(bucket) < 1:
        conf, bucket = None, None
    unit = [s.get("unit") or "" for s in stocks]
    final = {k for k, s in enumerate(stocks) if s.get("final")}
    group = [s.get("group") for s in stocks]
    ctx = {"stocks": stocks, "ids": ids, "t": t, "opts": opts, "base": base, "unit": unit, "final": final,
           "group": group, "conf": conf, "bucket": bucket}
    # measured inputs (opts "inputs": {name: value per step aligned with t}): rows after the stocks'
    # levels, which atoms read as drivers (an inflow k x input, or a drain k x input)
    inputs = [(k, np.nan_to_num(np.asarray(v, dtype=float))[1: len(t)]) for k, v in (opts.get("inputs") or {}).items() if k not in ids]
    inputs = [(k, v) for k, v in inputs if len(v) == prev.shape[1]]
    if inputs:
        prev = np.vstack([prev] + [v[None, :] for _, v in inputs])
        ctx["names"] = ids + [k for k, _ in inputs]
    ctx["inputs"] = [k for k, _ in inputs]
    if "data" in hooks:
        prev, dy = hooks["data"](prev, dy, ctx)
    cfg = np.asarray(bucket)[conf[1]] if conf else None
    # a long run: every k-th step only (each step's change given the levels at its start is its own
    # observation, so a stride keeps the whole span and every era), to stay within the budget
    keep = np.arange(prev.shape[1])
    if prev.shape[1] > MAX_STEPS:
        keep = keep[:: math.ceil(prev.shape[1] / MAX_STEPS)]
        prev, dy, cfg = prev[:, keep], dy[:, keep], (cfg[keep] if cfg is not None else None)
    ctx["keep"] = keep
    d = Data(prev, dy, cfg)
    d.deadline = time.monotonic() + float(opts.get("lna_budget", BUDGET))
    ctx["d"] = d
    if S <= SCRATCH:
        atoms = superset(S, unit, final, group, gaps=False)
    else:
        # many stocks: only the readings of numbers the data shows (a row's level clearly nonzero in
        # a regression on every level) or the matrix found (its chosen model's rows)
        index = {v: i for i, v in enumerate(ids)}
        coef = [{index[j]: k for j, k in (row.get("coef") or {}).items() if j in index and abs(k) > 1e-9} for row in base.get("matrix") or []]
        if len(coef) != S:
            return None
        for s_, row in enumerate(screen(prev[:S], dy)):
            for j, k in row.items():
                coef[s_].setdefault(j, k)
        atoms = restricted(S, coef, unit, final)
    for j in range(S, S + len(inputs)):
        r = readers(inputs[j - S][0], opts)
        if r is None:
            atoms += [Atom("link", dst=s_, driver=j) for s_ in range(S)] + [Atom("drain", src=s_, driver=j) for s_ in range(S) if s_ not in final]
        elif ids.index(r[0]) not in final:             # (capmin.py) the offered MIN(X, theta Y): only the flow it limits
            x = ids.index(r[0])
            y = ids.index(r[1]) if len(r) > 1 else None
            atoms.append(Atom("move", src=x, dst=y, driver=j) if y is not None and unit[y] == unit[x] else Atom("drain", src=x, driver=j))
    if "atoms" in hooks:
        atoms = hooks["atoms"](atoms, d, ctx)
    # (kalman_lna) the declared boundary: an outside atom only where the case declares that side
    # (input-driven links and drains are "measured"); swaps keep to it too (d.allow)
    d.allow = Allow(Boundary(run), ids, set(range(S, S + len(inputs))))
    atoms = d.allow.keep(atoms)
    phi = dispersion(atoms, mean_start(atoms, d), d)
    if phi < 0.2:
        return None                    # no counting noise to read: the matrix's reading stands
    if conf:
        atoms = split_eras(atoms, max(bucket) + 1)
    cur = fit(atoms, d)
    d.free = extra_variance(cur, d)
    full = fit(atoms, d, cur.theta, tau=cur.tau)
    # even every atom leaves the changes spreading far beyond counting noise: the system has what
    # the atoms cannot express (effects, caps, hidden stocks, bursts), and the likelihood would buy
    # it with pieces and spurious flows; the matrix's answer stands
    if misfit_of(full, d) > MAX_MISFIT:
        return None
    if "before_search" in hooks:
        full = hooks["before_search"](full, d, ctx)
    # changes in the outside inflows first, beside every flow (so that levels do not stand in for
    # them); and, when that found any, the search again without them first (the fine pass can also
    # lock in pieces that a level or a gap would explain): the lower BIC wins
    cur = search(pieces(full, d, mult=0.5), d)
    if any(a.span for a in cur.atoms):
        other = search(full, d)
        if bic(other, d) < bic(cur, d):
            cur = other
    if "after_search" in hooks:
        cur = hooks["after_search"](cur, d, ctx)
    # how far the changes' spread exceeds what counting noise gives the chosen flows (each stock's
    # extra variance over its counting variance): near 0 where the atoms describe the system
    misfit = misfit_of(cur, d)
    at = lambda k: t[keep[k] + 1]  # noqa: E731
    ctx["at"] = at
    flows = to_flows(cur.atoms, cur.theta, ctx.get("names") or ids, d.prev, at, *((conf[0], conf[2], bucket) if conf else ()))
    for f in flows:
        if f.get("driver") in ctx["inputs"]:
            f["input"] = f["driver"]
    if "flows" in hooks:
        flows = hooks["flows"](flows, cur, d, ctx)
    return {**{k: base[k] for k in ("candidates", "tried", "chosen") if k in base},
            "flows": flows, "method": "lna", "dispersion": phi, "misfit": misfit, "extra_variance": {ids[k]: float(cur.tau[k]) for k in range(S) if d.free[k]}, "nll": cur.nll, "windows": len(t) - 1, "stride": int(keep[1] - keep[0]) if len(keep) > 1 else 1, "window": 1,
            "atoms": [{"kind": a.kind, "src": a.src, "dst": a.dst, "driver": a.driver, "uses": list(a.uses), "theta": float(th)} for a, th in zip(cur.atoms, cur.theta)]}

