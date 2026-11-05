# Vendored for kalman_lna from sdlab/discovery/strategies/effect_hats/__init__.py at commit ec2f9e8 (imports rewritten to the
# vendored copies; otherwise unchanged). Do not edit: re-vendor instead.
"""Effects as tent columns: a flow's rate multiplied by a piecewise-linear curve of a stock's level,
found and fitted by least squares (docs/research/effect-hats.md).

1. The matrix method's model is the base.
2. Screen (screen.py): every pair (base, C) as a group of tent columns base x h_j(C) (basis.py) on
   what the rows leave unexplained; a pair that clearly explains one or two rows is a candidate.
3. Search (search.py): for each candidate and each way its flow can run (between the rows it shows
   in, or out of its base stock), the curve's weights are fitted jointly in those rows (the shared
   group with the flow's signs; knot count by BIC), and the placements compared on the same rows.
   The curve becomes a virtual stock base x f(C) offered to the matrix method in those rows; the
   result is read back into flows with table effects, a drain on the same curve as a move becoming
   its co-flow (translate.py), and every number, the curves' included, refitted jointly across all
   rows (joint.py).
4. The best few are finished: outside inflows pooled into a few values and re-cut where they change
   (pieces.py; per-stretch pieces read a drain up to 20% fast under counting noise), flows the joint
   BIC does not need dropped, and judged.
5. A candidate replaces the base only if its curve explains both halves of the data, it prices better
   (one-step BIC, pooled pieces counted once) than the base with its inflows re-cut the same way, and
   behaves within `margin` of it; among those, the lowest price wins (a flow whose rate is not its
   source's level pays `prior` x log T). With `rounds` > 1 the screen runs again with the accepted
   effects' columns taken out. Without an accepted effect, the base with its inflows re-cut is
   returned when it prices better and behaves within `seg_margin` (seg_base).

opts (also SDLAB_EFFECT_HATS_OPTS, JSON), defaults in DEFAULTS: margin, select ("bic" or "judge"),
prior, knots (knot counts tried), most (candidates per round), per_candidate (placements), finish
(models finished per round), halves, weighted (rows weighted by their noise), prune, pool (re-cut
inflows), guard (steps next to an inflow change left out of the fit), level_price (extra price of
an inflow value, in log T), seg_base, seg_margin, rounds.
"""

from __future__ import annotations

import json
import math
import os

import numpy as np

from orchestra.sdlab.discovery.addition import addition_test
from orchestra.sdlab.discovery.identify import judge
from orchestra.sdlab.discovery.identify import identify as matrix_identify
from orchestra.sdlab.discovery.matrix import injected
from orchestra.sdlab.discovery.strategies.kalman_lna._vendored.effect_hats import search
from orchestra.sdlab.discovery.strategies.kalman_lna._vendored.effect_hats.price import price, prune
from orchestra.sdlab.discovery.strategies.kalman_lna._vendored.effect_hats.pieces import resegment
from orchestra.sdlab.discovery.strategies.kalman_lna._vendored.effect_hats.screen import scan

DEFAULTS = {"margin": 1.1, "select": "bic", "prior": 2.0, "knots": [2, 3, 4, 6, 8], "most": 3, "halves": True,
            "weighted": True, "per_candidate": 2, "prune": True, "pool": True, "finish": 4, "seg_base": True,
            "seg_margin": 3.0, "guard": 2, "level_price": 2.0, "rounds": 1, "harm": None, "no_self_links": True, "pair": True, "coflows": True}


def _arrays(stocks, run, opts):
    ids = [s["id"] for s in stocks]
    lv = np.nan_to_num(np.array([np.asarray(run["levels"][i], dtype=float) for i in ids]))
    prev, dy = lv[:, :-1], np.diff(lv, axis=1)
    if opts.get("injections"):
        inj = injected(opts["injections"], ids, run["t"])[:, 1:]
        prev, dy = prev + inj, dy - inj
    return ids, prev, dy


def identify(stocks: list[dict], run: dict, opts: dict | None = None) -> dict:
    opts = {**json.loads(os.environ.get("SDLAB_EFFECT_HATS_OPTS") or "{}"), **(opts or {})}
    keep = opts.pop("candidates", False)
    o = {**DEFAULTS, **{k: opts.pop(k) for k in list(opts) if k in DEFAULTS}}
    base = matrix_identify(stocks, run, {**opts, "candidates": keep})
    base_e = min((x["error"] for x in base.get("tried") or []), default=float("inf"))
    if len(run["t"]) < 60:
        return base
    ids, prev, dy = _arrays(stocks, run, opts)
    fit_opts = {**opts, "weighted": o["weighted"], "guard": o["guard"], "level_price": o["level_price"], "no_self_links": o["no_self_links"],
                "pair": o["pair"], "coflows": o["coflows"]}
    # the base as the candidates will be: its outside inflows pooled and re-cut (judged that way too,
    # since the judge feeds a model its own pieces, and fewer pieces always look worse)
    seg = resegment(base["flows"], ids, run, fit_opts) if o["pool"] else base["flows"]
    base_b = price(seg, ids, run)
    fair_e = judge(seg, stocks, run, injections=opts.get("injections")) if o["pool"] else base_e
    base.update(bic=base_b, judged={"base": base_e, "recut": fair_e}, effects_tried=[])
    log_t = math.log(max(2, len(run["t"]) - 1))
    accepted, best, chosen, cur_b, cur_e = [], None, None, base_b, fair_e
    tried, shapes, skip = [], set(), set()
    for _ in range(o["rounds"]):
        extra = [search.column(c, run)[0][:-1] for c in accepted]
        cands = scan(stocks, prev, dy, o, extra=extra, skip=skip)
        base["effects_tried"] += [f"{c['base'] or 1} x f({c['stock']}) in {[r for r, _ in c['rows']]} {c['score']:.0f}" for c in cands]
        raw = []
        for cand in cands[: o["most"]]:
            for c in search.placement_fits(cand, ids, prev, dy, o):
                raw += [{**t, "fit": c} for t in search.trials(accepted + [c], stocks, run, prev, ids, opts, fit_opts, shapes)]
        # the most promising few are finished: inflows pooled and re-cut, flows the data does not
        # need dropped, then judged
        raw.sort(key=lambda t: t["bic"])
        done = []
        for t in raw[: o["finish"]]:
            flows = resegment(t["flows"], ids, run, fit_opts) if o["pool"] else t["flows"]
            if o["prune"]:
                flows = prune(flows, ids, run, fit_opts)
            if sum(1 for f in flows if f.get("hat")) < len(accepted) + 1:
                continue
            done.append({**t, "flows": flows, "error": judge(flows, stocks, run, injections=opts.get("injections")),
                         "bic": price(flows, ids, run)})
        tried += done
        def rank(t):
            # SD's usual form: a flow's rate is its source's level (or a constant) times the effect
            f = t["fit"]
            return t["bic"] + (0.0 if f["base"] in (f["frm"], None) else o["prior"] * log_t)
        # the best-priced model is the round's pick, and is then accepted or not: falling back to a
        # worse-priced one that happens to pass the judge picks wrong structures (a link read as a move)
        ok = [t for t in done if t["fit"]["halves_ok"] and t["bic"] < cur_b]
        if o["select"] == "judge":
            ok = [t for t in ok if t["error"] < o["margin"] * cur_e]
            best = min(ok, key=lambda t: t["error"]) if ok else None
        else:
            best = min(ok, key=rank) if ok else None
            if best is not None and not best["error"] < o["margin"] * cur_e:
                best = None
        # (opts["harm"]) and no worse than the base when each is fed a coarse inflow re-measured under
        # its own dynamics (addition.py, "refit")
        if best is not None and o["harm"] is not None and _fair(base["flows"], best["flows"], stocks, run) < -o["harm"]:
            best = None
        if best is None:
            best = chosen
            break
        chosen = best
        accepted.append(best["fit"])
        skip.add((best["fit"]["base"], best["fit"]["stock"]))
        cur_b, cur_e = best["bic"], best["error"]
    base["tried"] = list(base.get("tried") or []) + [{"name": t["name"], "error": t["error"], "bic": t["bic"]} for t in tried]
    if keep:
        base.setdefault("candidates", []).extend({"name": t["name"], "error": t["error"], "flows": _clean(t["flows"])} for t in tried)
    if best is None:
        return _seg_base(base, seg, o, ids, run, base_e, fair_e, stocks)
    out = {**best["r"], "flows": _clean(best["flows"]), "chosen": best["name"], "tried": base["tried"],
           "effects_tried": base["effects_tried"], "bic": best["bic"], "base_bic": base_b, "judged": base["judged"]}
    if keep:
        out["candidates"] = base.get("candidates") or []
    return out


def _fair(base: list[dict], cand: list[dict], stocks: list[dict], run: dict) -> float:
    """The candidate's gain over the base (relative error over 5/20/100 steps), each fed a coarse
    outside inflow re-measured under its own dynamics."""
    return float(addition_test(base, cand, stocks, run, inflow="refit")["gain"])


def _seg_base(base: dict, seg: list[dict], o: dict, ids: list[str], run: dict, base_e: float, fair_e: float,
              stocks: list[dict] | None = None) -> dict:
    """The base, or (opts["seg_base"]) the base with its inflows pooled and re-cut when that prices
    better, behaves within `seg_margin` of it, and is no worse with re-measured coarse inflows."""
    if not o["seg_base"] or seg is base["flows"]:
        return base
    if price(seg, ids, run) < price(base["flows"], ids, run) and fair_e < o["seg_margin"] * base_e \
            and (o["harm"] is None or _fair(base["flows"], seg, stocks, run) >= -o["harm"]):
        return {**base, "flows": _clean(seg), "chosen": f"{base.get('chosen')}, inflows re-cut"}
    return base


def _clean(flows: list[dict]) -> list[dict]:
    """Without the fit's own notes; neighbouring pieces of one pool joined."""
    out = []
    for f in flows:
        f = {k: v for k, v in f.items() if k != "hat"}
        if f.get("inflow"):
            segs = []
            for g in f["inflow"]:
                if segs and g.get("pool") is not None and segs[-1].get("pool") == g.get("pool") and segs[-1]["to"] + 1 >= g["from"]:
                    segs[-1] = {**segs[-1], "to": g["to"]}
                else:
                    segs.append(dict(g))
            f["inflow"] = [{k: v for k, v in g.items() if k != "pool"} for g in segs]
        out.append(f)
    return out
