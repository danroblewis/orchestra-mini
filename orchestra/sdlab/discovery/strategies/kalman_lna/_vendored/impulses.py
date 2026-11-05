# Vendored for kalman_lna from sdlab/discovery/strategies/impulses.py at commit ec2f9e8 (imports rewritten to the
# vendored copies; otherwise unchanged). Do not edit: re-vendor instead.
"""Pulses as their own kind of input (docs/research/impulse-inputs.md).

One-step deliveries and short bursts are found in each stock's change jointly with the row's numbers
(pulses.joint: find the pulses in what the levels leave, refit the row without the pulse steps,
repeat), first with the stock's own level as the only column, then with the columns a fitted
matrix chose (rounds). A sudden loss is kept only where another stock gains in the same step (a
batch moved, or a co-flow used up); deliveries from outside are kept on their own. The pulses are
handed to the main line as known changes (opts["known"]: taken out of the rows before they are
fitted, and fed to every candidate when it is judged), so no drain has to absorb them, and are
returned as outside flows that arrive in those steps only.

Options (opts["impulses"], a dict): rounds (matrix fits for the joint detection, default 2);
max_len (longest burst, default 6); thresh (default sqrt(2 log(T x max_len)) robust SDs);
max_share (default 0.3 of the steps); sign (0 both, 1 gains only); paired_outflows ("any": a loss
needs a gain in any other stock, "unit": in a stock of its unit, "" keeps every loss); counts
(default True: whole-number stocks get a counting-noise floor on the scale); start ("own", or
"median": around a running median of the change, mistakes fast stocks' inflow changes for bursts);
fit (the matrix variant for the rounds); trace (a stock id: print each round's numbers).
"""

from __future__ import annotations

import numpy as np

from orchestra.sdlab.discovery import pulses as P
from orchestra.sdlab.discovery.identify import identify as identify_main
from orchestra.sdlab.discovery.matrix import configurations, identify_one


def _residuals(rows: list[dict], ids: list[str], t: list, prev: np.ndarray, dy: np.ndarray, eras, inputs: dict | None = None) -> np.ndarray:
    """dy less what each fitted row's levels (and measured inputs, aligned as matrix aligns them)
    explain (its numbers per configuration): its outside inflow, noise and pulses."""
    T = dy.shape[1]
    cols = {k: np.nan_to_num(np.asarray(v, dtype=float))[1: len(t)] for k, v in (inputs or {}).items() if k not in ids}
    conf = configurations(eras, t)
    out = dy.copy()
    for s, row in enumerate(rows):
        coefL = row.get("coefL") or [row.get("coef") or {}]
        cfg = conf[1] if conf and len(coefL) > 1 else np.zeros(T, dtype=int)
        for c, coef in enumerate(coefL):
            m = cfg == c
            for j, k in coef.items():
                if j in ids:
                    out[s, m] -= k * prev[ids.index(j)][m]
                elif j in cols and len(cols[j]) == T:            # a measured input's column
                    out[s, m] -= k * cols[j][m]
    return out


def _flows(stock: str, amounts: np.ndarray, t: list) -> list[dict]:
    """Pulse amounts (one per step from t[1]) as outside flows into (or out of) the stock: segments
    that hold the pulse on its steps and 0 between them."""
    out = []
    for sign, ends in ((1, {"from": None, "to": stock}), (-1, {"from": stock, "to": None})):
        v = np.maximum(0.0, sign * amounts)
        if not v.any():
            continue
        segs, prev_end = [], t[0]
        for k in np.nonzero(v)[0]:
            at = t[k + 1]
            if at > prev_end + 1 or (not segs and at > prev_end):
                segs.append({"from": prev_end if not segs else prev_end + 1, "to": at - 1, "value": 0.0})
            segs.append({"from": at, "to": at, "value": float(v[k])})
            prev_end = at
        segs.append({"from": prev_end + 1, "to": t[-1] + 1, "value": 0.0})
        mean = float(v.sum() / max(1, len(t) - 1))
        out.append({**ends, "ratio": 1, "rate": {"form": "constant", "value": round(mean, 4)}, "rates": [],
                    "perStep": mean, "inflow": segs, "pulses": int((v > 0).sum()), "nearly": [], "bothWays": False})
    return out


def find_pulses(stocks: list[dict], run: dict, opts: dict) -> dict[str, np.ndarray]:
    """Each stock's pulses as a series of amounts (index i: the change from step i - 1 to i, as
    opts["known"] reads it); stocks without pulses are left out."""
    o = opts.get("impulses") or {}
    ids = [s["id"] for s in stocks]
    t = run["t"]
    lv = np.nan_to_num(np.array([np.asarray(run["levels"].get(i, [0] * len(t)), dtype=float) for i in ids]))
    prev, dy = lv[:, :-1], np.diff(lv, axis=1)
    T = dy.shape[1]
    if T < 20:
        return {}
    kw = {"sign": o.get("sign", 0), **{k: o[k] for k in ("max_len", "thresh", "max_share") if k in o}}
    # stocks measured in whole things have at least counting noise: about the square root of what
    # flows through them in a step (a robust scale alone is near 0 where most steps change nothing)
    counts = [o.get("counts", True) and bool(np.all(lv[s] == np.round(lv[s]))) for s in range(len(ids))]

    def floor(s: int, through: float) -> float:
        return float(np.sqrt(through)) if counts[s] else 0.0

    amounts = np.zeros_like(dy)
    if o.get("start") == "median":
        # first around each change's running median (no model yet); a stock that adjusts within a
        # few steps turns a change in its inflow into a short transient of its change, so this start
        # mistakes such changes for bursts
        for s in range(len(ids)):
            r = dy[s] - P.running_median(dy[s], o.get("width", 15))
            sigma = max(P.robust_scale(r), floor(s, float(np.mean(np.abs(dy[s])))))
            amounts[s] = P.series(P.detect(r, sigma=sigma, **kw), r, T)
    elif o.get("start", "own") == "own":
        # first jointly with the stock's own level only: a fit through the pulses would let the
        # drain absorb them (and turn them into a smooth wave in what is left)
        for s in range(len(ids)):
            amounts[s] = P.joint(dy[s], prev[s:s + 1], np.zeros(1), opts.get("min_piece", 5), floor=floor(s, float(np.mean(np.abs(dy[s])))), **kw)
    # then jointly with the rows: fit with the pulses taken out, find them again in what is left
    fit = {"name": "pulses", **(o.get("fit") or {})}
    for _ in range(o.get("rounds", 2)):
        known = {i: np.concatenate([[0.0], amounts[s]]) for s, i in enumerate(ids) if amounts[s].any()}
        r1 = identify_one(stocks, run, {**{k: v for k, v in opts.items() if k not in ("impulses", "known")}, **fit, **({"known": known} if known else {})})
        if not r1.get("matrix"):
            break
        if o.get("trace"):
            print("  round:", {r["id"]: {j: round(k, 4) for j, k in r["coef"].items()} for r in r1["matrix"] if r["id"] == o["trace"]}, int((amounts != 0).sum()))
        u = _residuals(r1["matrix"], ids, t, prev, dy, opts.get("eras") if fit.get("split_eras") else None, opts.get("inputs"))
        new = np.zeros_like(dy)
        for s, row in enumerate(r1["matrix"]):
            through = float(np.mean(np.abs(dy[s] - u[s])) + abs(np.median(u[s])))
            coef = row.get("coef") or {}
            ins = {k: np.nan_to_num(np.asarray(v, dtype=float))[1: len(t)] for k, v in (opts.get("inputs") or {}).items() if k not in ids}
            coef = {j: k for j, k in coef.items() if j in ids or (j in ins and len(ins[j]) == T)}
            X = np.array([prev[ids.index(j)] if j in ids else ins[j] for j in coef]).reshape(len(coef), T)
            beta = np.array(list(coef.values()), dtype=float)
            new[s] = P.joint(dy[s], X, beta, opts.get("min_piece", 5), floor=floor(s, through), **kw)
        if np.array_equal(new != 0, amounts != 0) and np.allclose(new, amounts):
            break
        amounts = new
    if o.get("paired_outflows", "any"):
        # a sudden loss is kept only where another stock (of the same unit, with "unit") gains in
        # the same step: a batch moved, or a co-flow used up; a lone loss is mostly a flow that
        # stopped (a stock run dry), which pulses would only replay. Deliveries are kept on their own
        unit = [s.get("unit") or "" for s in stocks]
        for s in range(len(ids)):
            gain = np.zeros(T, bool)
            for q in range(len(ids)):
                if q != s and (o.get("paired_outflows", "any") == "any" or unit[q] == unit[s]):
                    gain |= amounts[q] > 0
            amounts[s] = np.where((amounts[s] < 0) & ~gain, 0.0, amounts[s])
    return {i: amounts[s] for s, i in enumerate(ids) if amounts[s].any()}


def identify(stocks: list[dict], run: dict, opts: dict | None = None) -> dict:
    opts = dict(opts or {})
    found = find_pulses(stocks, run, opts)
    if not found:
        return identify_main(stocks, run, {k: v for k, v in opts.items() if k != "impulses"})
    known = {i: np.concatenate([[0.0], a]) for i, a in found.items()}
    r = identify_main(stocks, run, {**{k: v for k, v in opts.items() if k != "impulses"}, "known": known})
    for i, a in found.items():
        r["flows"] = r["flows"] + _flows(i, a, run["t"])
    r["pulses"] = {i: int((a != 0).sum()) for i, a in found.items()}
    return r
