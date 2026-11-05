# Vendored for kalman_lna from sdlab/discovery/strategies/lna_pulses/__init__.py at commit ec2f9e8 (imports rewritten to the
# vendored copies; otherwise unchanged). Do not edit: re-vendor instead.
"""lna_pulses: pulses (one-step arrivals, short bursts) found first, then lna on what they leave.

The impulse-inputs experiment (docs/research/impulse-inputs.md, strategy `impulses`) found pulses
jointly with each row and handed them to the matrix as known changes. On the ladder it passes the
pulse case only where the matrix itself does; on noisy runs the matrix's reading of the chain is
what fails. Here the same detection (`impulses.find_pulses`) feeds lna: the pulses are taken out of
each step's change before the counting-noise likelihood reads the moves, so a burst of 300 is not
read as a move, a link or a drain. Without noise (where lna is the matrix's reading) the chosen
model's outside inflows are re-measured with the pulse steps left out. Either way the pulses are
returned as outside flows that arrive in those steps only (as `impulses` writes them). With no
pulse found, the result is lna's. The pulses go out through lna's `data` hook (run.py); lna's
searches here share one lna budget, so lna's data handling, stride and time limit are lna's own.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import chi2

from orchestra.sdlab.discovery.identify import identify as _matrix
from orchestra.sdlab.discovery.matrix import injected
from orchestra.sdlab.discovery.simulate import Model
from orchestra.sdlab.discovery.strategies.kalman_lna._vendored.impulses import _flows as pulse_flows
from orchestra.sdlab.discovery.strategies.kalman_lna._vendored.impulses import find_pulses
from orchestra.sdlab.discovery.strategies.kalman_lna._vendored.lna import GUARD, judge
from orchestra.sdlab.discovery.strategies.kalman_lna._vendored.lna import identify as _lna
from orchestra.sdlab.discovery.strategies.kalman_lna._vendored.lna_pulses.run import Budget, stage


Z = 6.0          # a pulse is kept when it stands this many robust SDs above what the model leaves
SIZE = 0.25      # ... and is at least this share of what the stock moves in a typical step


def _inflow_of(f: dict, stock: str) -> bool:
    r = f.get("rate") or {}
    return f.get("from") is None and f.get("to") == stock and r.get("form") == "constant" and not f.get("byTime") and not f.get("effects")


def pooled(r: np.ndarray, hit: np.ndarray, base: np.ndarray, sd: float, counts: bool) -> np.ndarray:
    """Pulses into one stock whose sizes are one number within their noise (a chi-square test of
    homogeneity at 1%, each pulse's variance the residual spread plus, for counted stocks, its own
    count) are given that one number: n pulses measure it sqrt(n) times better than one does."""
    idx = np.nonzero(hit)[0]
    if len(idx) < 2:
        return r
    a = r[idx]
    if np.any(np.sign(a) != np.sign(a[0])):
        return r
    var = sd**2 + (np.abs(a + base[idx]) if counts else 0.0) + 1e-12 * a**2
    m = float(np.sum(a / var) / np.sum(1 / var))
    if float(np.sum((a - m) ** 2 / var)) > chi2.ppf(0.99, len(idx) - 1):
        return r
    out = r.copy()
    out[idx] = m
    return out


def remeasure(flows: list[dict], ids: list[str], run: dict, found: dict[str, np.ndarray], injections=None, z: float = Z) -> tuple[list[dict], dict[str, np.ndarray]]:
    """Each pulsed stock's outside inflow measured again from what the other flows leave, the pulse
    steps left out; the pulses are what the inflow leaves on those steps, kept only where that
    stands z robust SDs (of what is left on the other steps) clear of 0 and is a burst (SIZE of a
    typical step's movement), and pooled when they are one size."""
    t = np.asarray(run["t"], dtype=float)
    lv = np.nan_to_num(np.array([np.asarray(run["levels"].get(i, [0] * len(t)), dtype=float) for i in ids]))
    prev, dy = lv[:, :-1], np.diff(lv, axis=1)
    if injections:
        inj = injected(injections, ids, list(t))[:, 1:]
        prev, dy = prev + inj, dy - inj
    out = [dict(f) for f in flows]
    pulses = {}
    for stock, amounts in found.items():
        s = ids.index(stock)
        ins = [f for f in out if _inflow_of(f, stock)]
        f = ins[0] if len(ins) == 1 else None
        saved = dict(f) if f is not None else None
        others = [g for g in out if g is not f]
        u = dy[s] - Model(others, ids).change(prev, t[1:])[s]
        hit = amounts != 0
        base = np.zeros_like(u)
        if f is not None and f.get("inflow"):
            segs = []
            for g in f["inflow"]:
                m = (t[1:] >= g["from"]) & (t[1:] <= g["to"])
                keep = m & ~hit
                v = float(u[keep].mean()) if keep.any() else float(g["value"])
                base[m] = v
                segs.append({**g, "value": v})
            w = np.array([max(1, g["to"] - g["from"] + 1) for g in segs], dtype=float)
            mean = float(np.dot([g["value"] for g in segs], w) / w.sum())
            f.update(inflow=segs, rate={"form": "constant", "value": mean}, rates=[{"form": "constant", "value": mean}], perStep=mean)
        elif f is not None:
            v = float(u[~hit].mean()) if (~hit).any() else float(f["rate"]["value"])
            base[:] = v
            f.update(rate={"form": "constant", "value": v}, rates=[{"form": "constant", "value": v}], perStep=v)
        r = u - base
        rest = r[~hit] if (~hit).any() else r
        sd = 1.4826 * float(np.median(np.abs(rest - np.median(rest))))
        # and a pulse is a burst: at least a quarter of what the stock moves in a typical step (its
        # inflow plus its mean change), so a model's small misfit or the data's rounding is not one
        floor = SIZE * (float(np.mean(np.abs(dy[s]))) + float(np.abs(base).mean())) + 1e-3
        hit = hit & (np.abs(r) > z * max(sd, floor))
        counts = bool(np.all(lv[s] == np.round(lv[s])))
        p = np.where(hit, pooled(r, hit, base, sd, counts), 0.0)
        if p.any():
            pulses[stock] = p
        elif f is not None:
            f.clear()
            f.update(saved)                  # no pulse held up: the inflow stays as it was
    return out, pulses


def _not_injected(found: dict[str, np.ndarray], run: dict, injections: list[dict] | None) -> dict[str, np.ndarray]:
    """Announced injections are not pulses (the strategy is told of them), nor is what any stock does
    in the two steps after one (the injected amount moving on)."""
    t = np.asarray(run["t"], dtype=float)
    out = {}
    for stock, a in found.items():
        a = a.copy()
        for j in injections or []:
            a[(t[1:] >= float(j["t"]) - 1) & (t[1:] <= float(j["t"]) + 2)] = 0.0
        if a.any():
            out[stock] = a
    return out


def identify(stocks: list[dict], run: dict, opts: dict | None = None) -> dict:
    opts = dict(opts or {})
    found = _not_injected(find_pulses(stocks, run, opts), run, opts.get("injections"))
    if not found:
        return _lna(stocks, run, opts)
    ids = [s["id"] for s in stocks]
    T = len(run["t"]) - 1
    base = _matrix(stocks, run, dict(opts))
    rest = {k: v for k, v in opts.items() if k != "candidates"}
    inj = opts.get("injections")
    budget = Budget(opts)
    r, flows, pulses = None, None, {}
    for _ in range(3):                 # lna with the pulses taken out; the pulses measured again on its model
        if not found:
            return _lna(stocks, run, opts)
        known = np.zeros((len(ids), T))
        for k, a in found.items():
            known[ids.index(k)] = a
        got = budget.discover(stocks, run, rest, base, stage(known))
        if got is None:                # no counting noise (the first search), or out of time on a later one: the last kept
            break
        r = got
        flows, pulses = remeasure(r["flows"], ids, run, found, inj)
        same = pulses.keys() == found.keys() and all(np.array_equal(pulses[k] != 0, found[k] != 0) for k in pulses)
        found = pulses
        if same:
            break
    if r is not None and not pulses:
        return _lna(stocks, run, opts)
    if r is not None:
        mine = judge(flows + [g for k, a in pulses.items() for g in pulse_flows(k, a, run["t"])], stocks, run, opts)
        theirs = judge(base["flows"], stocks, run, opts)
        if not math.isfinite(mine) or (math.isfinite(theirs) and mine > GUARD * theirs):
            r = None
    if r is None:                       # no counting noise (or the guard): the matrix's reading, re-measured
        flows, pulses = remeasure(base["flows"], ids, run, found, inj)
        if not pulses:
            return _lna(stocks, run, opts)
        r = dict(base)
    r["flows"] = flows + [g for k, a in pulses.items() for g in pulse_flows(k, a, run["t"])]
    r["pulses"] = {k: int((a != 0).sum()) for k, a in pulses.items()}
    return r
