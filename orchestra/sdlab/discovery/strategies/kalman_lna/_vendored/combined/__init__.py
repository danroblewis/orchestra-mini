# Vendored for kalman_lna from sdlab/discovery/strategies/combined/__init__.py at commit ec2f9e8 (imports rewritten to the
# vendored copies; otherwise unchanged). Do not edit: re-vendor instead.
# (kalman_lna, 2026-10-20) Edited after vendoring for the declared boundary: every candidate guarded before
# the judge where one is declared, outside flows priced as boundary claims where none is (boundary.py).
"""combined: the specialist strategies run side by side, and one judge picks among their models
(docs/research/combined.md).

1. Candidates. Each specialist is called as a black box through its own `identify()`, so its
   improvements flow in, in this order: censored (floors that bind, stocks that run empty),
   effect_hats (effects of a third stock's level; also re-cuts the outside inflows), lna (the best
   single strategy on the test cases: flows chosen by the counting likelihood), hidden_transit (an
   unmeasured stock between two measured ones), periodic_inflows (an inflow that repeats),
   gap_prior (lna with a drain read off another stock written as the goal gap it equals),
   lna_pulses (one-step bursts taken out before lna reads the moves), lna_stack (lna with the
   revisit's stages composed; lna_loops is deliberately not a candidate: its wins are on cells the
   data cannot decide) and common_cause (independent and identical stocks, noise-free fits, inflow
   changes the data earns). censored and effect_hats go first because on the realistic models
   they are the best candidates and the lna family the slowest. The specialists that start from
   the matrix method's model, or from lna's, share one computation of it (shared.py). Each runs
   under a cap (deadline.py: `cap` seconds, never past the budget); one that raises or runs out of
   time is left out, and once the budget is spent the rest are skipped. The budget is `budget`
   seconds or 70% of what the caller's timer leaves.
2. One judge (judge.py): the one-step likelihood of the measured changes under each candidate's
   equations, applied exactly as the simulator applies them (measured inputs, floors, caps,
   effects, co-flows, unmeasured stocks run open loop), with counting noise read through the
   linear noise approximation (each flow a Poisson draw) or, for data that are not counts, each
   stock's own variance floored at the data's rounding; plus a BIC price of 0.5 log T for every
   number the candidate wrote down (params.py: rate numbers, inflow values and change places,
   effect points, co-flows, unmeasured stocks). Before it is judged, every candidate's numbers are
   refitted on the same terms (refit.py: least squares of the one-step changes, its structure
   kept), and the refitted numbers are kept when they raise its likelihood.
3. A behaviour guard first sets aside candidates that behave clearly worse over many steps than
   the best one: a candidate whose multi-step error on the data (the main line's judge: runs from
   many moments over 5, 20 and 100 steps, measured inputs fed, unmeasured stocks started from
   their open-loop estimate; 1 = no better than nothing changing) is above both `guard` (4) times
   the best candidate's and `guard_floor` (0.1). The one-step likelihood cannot see a model that
   drifts off over many steps (it picked such models on the realistic examples); the guard only
   removes them and never ranks the rest, because this in-data judge replays each model's own
   inflow pieces and so favours many pieces (a tighter guard, 2x, loses about 20 test cells).
4. The lowest score wins; exact ties go to the model with fewer flows that read a stock other
   than the ones they join, then to fewer numbers, then to the earlier name in PREFERENCE (the
   lna family first).

opts (also SDLAB_COMBINED, JSON): "specialists" (names, in order; default SPECIALISTS), "price"
(the BIC multiplier, default 1), "per_flow" (numbers a flow costs for being there, default 0: a
flow is priced by its numbers), "budget" (seconds, default 240), "cap" (seconds per specialist,
default 90), "refit" (refit every candidate's numbers, default on), "guard" and "guard_floor"
(default 4 and 0.1; guard 0 turns it off), "trace" (keep every candidate's flows in the result).
"""

from __future__ import annotations

import json
import math
import os
import time

from orchestra.sdlab.discovery.strategies.kalman_lna._vendored.censored import identify as censored
from orchestra.sdlab.discovery.strategies.kalman_lna._vendored.common_cause import identify as common_cause
from orchestra.sdlab.discovery.strategies.kalman_lna._vendored.effect_hats import identify as effect_hats
from orchestra.sdlab.discovery.strategies.kalman_lna._vendored.gap_prior import identify as gap_prior
from orchestra.sdlab.discovery.strategies.kalman_lna._vendored.hidden_transit import identify as hidden_transit
from orchestra.sdlab.discovery.strategies.kalman_lna._vendored.lna import identify as lna
from orchestra.sdlab.discovery.strategies.kalman_lna._vendored.lna_pulses import identify as lna_pulses
from orchestra.sdlab.discovery.strategies.kalman_lna._vendored.lna_stack import identify as lna_stack
from orchestra.sdlab.discovery.strategies.kalman_lna._vendored.periodic_inflows import identify as periodic_inflows

from orchestra.sdlab.discovery.strategies.kalman_lna import boundary as B
from orchestra.sdlab.discovery.strategies.kalman_lna._vendored.combined import judge as J
from orchestra.sdlab.discovery.strategies.kalman_lna._vendored.combined.deadline import Overrun, deadline, outer_left
from orchestra.sdlab.discovery.strategies.kalman_lna._vendored.combined.refit import refit
from orchestra.sdlab.discovery.strategies.kalman_lna._vendored.combined.shared import once

STRATEGIES = {"lna": lna, "gap_prior": gap_prior, "lna_pulses": lna_pulses, "lna_stack": lna_stack, "common_cause": common_cause, "censored": censored,
              "hidden_transit": hidden_transit, "periodic_inflows": periodic_inflows, "effect_hats": effect_hats}
SPECIALISTS = ["censored", "effect_hats", "lna", "hidden_transit", "periodic_inflows", "gap_prior", "lna_pulses", "lna_stack", "common_cause"]
# exact ties go to the earlier name here (whatever order they ran in)
PREFERENCE = ["lna", "gap_prior", "lna_pulses", "lna_stack", "censored", "effect_hats", "hidden_transit", "periodic_inflows", "common_cause"]
DEFAULTS = {"price": 1.0, "per_flow": 0.0, "budget": 240.0, "cap": 90.0, "trace": False, "refit": True, "guard": 4.0, "guard_floor": 0.1}


def cross_reads(flows: list[dict]) -> int:
    """Flows whose rate reads a stock other than the ones it joins (a drain that follows another
    level, an inflow that follows another stock): the conventional SD reading of a tie is the one
    where each flow follows its own stocks."""
    n = 0
    for f in flows or []:
        r = f.get("rate") or {}
        reads = {r.get("stock")} | {x.get("stock") for x in r.get("terms") or []}
        reads.discard(None)
        if reads - {f.get("from"), f.get("to")}:
            n += 1
    return n


def candidates(stocks: list[dict], run: dict, opts: dict, names: list[str], cfg: dict) -> list[dict]:
    """Each specialist's result, in order. Each runs under a cap (`cap` seconds, and never past the
    time left in the budget); once the budget is spent the rest are skipped. The budget is `budget`
    seconds, or 70% of what the caller's own timer leaves, whichever is less (the judge needs the
    rest)."""
    out = []
    began = time.perf_counter()
    left = outer_left()
    budget = float(cfg["budget"]) if left is None else min(float(cfg["budget"]), 0.7 * left)
    with once() as cached:
        for name in names:
            spent = time.perf_counter() - began
            if out and spent > budget:
                out.append({"name": name, "skipped": "budget"})
                continue
            t0 = time.perf_counter()
            try:
                with deadline(max(1.0, min(float(cfg["cap"]), budget - spent))):
                    r = (cached.get(name) or STRATEGIES[name])(stocks, run, dict(opts))
            except Overrun as e:
                out.append({"name": name, "error": f"capped: {e}", "seconds": round(time.perf_counter() - t0, 3)})
                continue
            except Exception as e:                       # noqa: BLE001
                out.append({"name": name, "error": f"{type(e).__name__}: {e}"[:300]})
                continue
            out.append({"name": name, "result": r, "seconds": round(time.perf_counter() - t0, 3)})
    return out


def choose(cands: list[dict], data: J.Data, run: dict, cfg: dict) -> dict | None:
    """The candidate with the lowest score among those that do not run away (see the docstring)."""
    price, per_flow = float(cfg["price"]), float(cfg["per_flow"])
    boundary = B.Boundary(run)
    scored = []
    for i, c in enumerate(cands):
        r = c.get("result")
        if not r or r.get("flows") is None:
            continue
        starts = {h: v[0] for h, v in (r.get("inferred_levels") or {}).items() if v}
        try:
            s = J.score(r["flows"], data, price, per_flow, starts)
        except Exception as e:                           # noqa: BLE001
            c["judge_error"] = f"{type(e).__name__}: {e}"[:300]
            continue
        if cfg["refit"]:
            try:
                flows = refit(r["flows"], data, J.weights(r["flows"], data))
                s2 = J.score(flows, data, price, per_flow, starts) if flows else None
            except Exception as e:                       # noqa: BLE001
                c["refit_error"] = f"{type(e).__name__}: {e}"[:300]
                s2 = None
            if s2 is not None and s2["nll"] < s["nll"] - 1e-9:
                c["result"] = r = {**r, "flows": flows, "refitted": True}
                s = {**s2, "refitted": True}
        if not boundary.declared:
            # (kalman_lna) no boundary declared: each outside flow is a boundary claim, CLAIM_PRICE log T
            # on the -2 log L scale (this judge's nll is -log L: half that, times its price multiplier)
            n_claims = len(B.claims(r["flows"], data.ids, boundary, set(data.inputs or {})))
            s = {**s, "claims": n_claims, "bic": s["bic"] + 0.5 * price * B.CLAIM_PRICE * math.log(max(2, data.T)) * n_claims}
        c["score"] = s
        scored.append((i, c, starts))
    if not scored:
        return None
    if cfg["guard"]:
        seen: dict = {}
        for _, c, starts in scored:
            key = json.dumps(c["result"]["flows"], sort_keys=True, default=repr)
            if key not in seen:
                try:
                    seen[key] = J.behaviour(c["result"]["flows"], data, run, starts)
                except Exception:                        # noqa: BLE001
                    seen[key] = float("inf")
            c["score"]["behaviour"] = seen[key]
        good = min(c["score"]["behaviour"] for _, c, _ in scored)
        limit = max(float(cfg["guard"]) * good, float(cfg["guard_floor"]))
        kept = [x for x in scored if x[1]["score"]["behaviour"] <= limit]
        for _, c, _ in scored:
            c["score"]["ran_away"] = c["score"]["behaviour"] > limit
        scored = kept or scored
    def rank(x):
        name = x[1]["name"]
        return (round(x[1]["score"]["bic"], 6), cross_reads(x[1]["result"]["flows"]), x[1]["score"]["k"],
                PREFERENCE.index(name) if name in PREFERENCE else len(PREFERENCE), x[0])
    return min(scored, key=rank)[1]


def identify(stocks: list[dict], run: dict, opts: dict | None = None) -> dict:
    opts = dict(opts or {})
    env = json.loads(os.environ.get("SDLAB_COMBINED") or "{}")
    own = {k: opts.pop(k) for k in list(opts) if k in DEFAULTS or k == "specialists"}
    cfg = {**DEFAULTS, **own, **env}
    names = list(cfg.get("specialists") or SPECIALISTS)
    cands = candidates(stocks, run, opts, names, cfg)
    own: dict = {}
    if B.Boundary(run).declared:
        # (kalman_lna) the declared boundary: the judge sees each specialist's model with every flow
        # outside it replaced by its best-priced alternative (dropped, made a move), so it chooses among
        # models inside the boundary; the chosen specialist's own model is returned, and kalman_lna's
        # measurement-error refit (eiv.py) and final guard (boundary.apply) take it from there: the
        # guard's numbers are one-step least squares, which measurement error biases (303 meas-long)
        for i, c in enumerate(cands):
            r = c.get("result")
            if r and r.get("flows"):
                try:
                    g = B.apply(r, stocks, run, opts, final=False)
                except Exception as e:                   # noqa: BLE001  (the judge sees it as it was)
                    c["boundary_error"] = f"{type(e).__name__}: {e}"[:300]
                    continue
                if g.get("boundary_guard"):
                    own[i], c["result"] = r, g
    data = J.Data(stocks, run, opts.get("injections"), opts.get("inputs"))
    best = choose(cands, data, run, cfg)
    if best is None:
        return {"flows": [], "combined": {"chosen": None, "candidates": _summary(cands, cfg)}}
    i = next(k for k, c in enumerate(cands) if c is best)
    r = dict(own[i]) if i in own else dict(best["result"])
    if i in own:
        r["boundary_judged"] = best["result"].get("boundary_guard")
    r["combined"] = {"chosen": best["name"], "candidates": _summary(cands, cfg)}
    return r


def _summary(cands: list[dict], cfg: dict) -> list[dict]:
    out = []
    for c in cands:
        s = {k: c[k] for k in ("name", "seconds", "error", "skipped", "judge_error") if k in c}
        if "score" in c:
            s.update({k: round(v, 3) if isinstance(v, float) else v for k, v in c["score"].items()})
        if cfg.get("trace") and c.get("result"):
            s["flows"] = c["result"].get("flows")
        out.append(s)
    return out
