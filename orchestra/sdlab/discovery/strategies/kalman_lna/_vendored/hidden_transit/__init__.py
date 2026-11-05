# Vendored for kalman_lna from sdlab/discovery/strategies/hidden_transit/__init__.py at commit ec2f9e8 (imports rewritten to the
# vendored copies; otherwise unchanged). Do not edit: re-vendor instead.
"""Experiment hidden_transit (docs/research/hidden_transit.md): an unmeasured transit stock between
two measured ones (A -> H -> B), with the pair's rows refitted as one model by a Kalman filter's
likelihood, on top of the main line.

The main line's model is found as usual. Then, for every pair (A loses in proportion to its level,
to outside or straight to B; B gets things from outside or from A), the pair's rows are refitted
twice by the same likelihood (kalman.py, fit.py): as a transit stock (A's loss goes into H, H drains
into B at d a step: a delay of 1 / d) and as the plain alternative (B gets a share of A's loss
straight away). Each in variants: A's outside inflow refitted as one constant or kept as found; B with
or without an outside constant. The transit stock is accepted when its best variant beats the best
plain one by BIC (numbers fitted, and pieces kept from the main line, cost log T each) by a margin,
and the model with it behaves better on the data than the main line's with B's outside
inflow held at its mean (its pieces, fitted to this run, would make it look better on it). Pairs are taken greedily,
best first, each stock in one pair at most; never against a move B -> A the main line found (a
transit back along a move is a loop's echo, not a stock).

"All of A's loss goes to H" (kin = kA) is the default: B's response to A alone cannot tell a
delay d with B's drain kb from the mirror (d and kb swapped, kin scaled), but A's balance can.
A share (kin < kA, the rest to outside) is accepted only when it is better by BIC.

The counting noise is used as it is: the data are counts (whole numbers), so each flow's variance is
its rate (scale 1), which also pins the rates' sizes where a short run's mean levels cannot. Data
that are not counts get a concentrated scale.

opts: "ht_margin" (BIC margin in units of log n, n the number of changes: default 4),
"ht_found_ratio" (the model with H may behave at most this many times worse than the model as found,
pieces and all; default 1.5), "ht_slack" (the model may behave this much worse than the
reference on the data, default 0: it must behave better), "ht_max" (hidden stocks at most, default 2), "ht_turns" (H must turn over at least this many
times in the run: delay <= T / ht_turns, default 15), "ht_min_delay" (a shorter delay is a move;
default 1.5 steps), "ht_trace".
"""

from __future__ import annotations

import math

import numpy as np

from orchestra.sdlab.discovery.identify import identify as main_line
from orchestra.sdlab.discovery.simulate import behaviour_error, levels_array
from orchestra.sdlab.discovery.strategies.kalman_lna._vendored.hidden_transit.fit import fit_plain, fit_transit, screen
from orchestra.sdlab.discovery.strategies.kalman_lna._vendored.hidden_transit.kalman import Pair
from orchestra.sdlab.discovery.strategies.kalman_lna._vendored.hidden_transit.pairs import at_means, build, candidate, exogenous, flow_numbers, one_piece, pieces, split

HORIZONS = (5, 20, 100)


def judge(flows: list[dict], stocks: list[dict], run: dict, inferred: dict | None = None) -> float:
    hs = [h for h in HORIZONS if h < len(run["t"]) / 4]
    if not flows or not hs:
        return float("inf")
    ids = [s["id"] for s in stocks]
    return sum(behaviour_error(flows, ids, run["t"], run["levels"], horizon=h, inferred=inferred) for h in hs) / len(hs)


def counts(run: dict) -> bool:
    """The data are counts: whole numbers that move (each flow then varies as much as its rate)."""
    moved = False
    for v in run["levels"].values():
        x = np.asarray(v, dtype=float)
        x = x[np.isfinite(x)]
        if x.size and np.abs(x - np.round(x)).max() > 1e-9:
            return False
        moved = moved or (x.size > 1 and np.abs(np.diff(x)).max() > 0)
    return moved


def try_pair(flows: list[dict], ids: list[str], run: dict, levels: dict, a: str, b: str, scale: float) -> dict | None:
    """The pair's best transit and best plain fits over the variants (A's outside inflow as one
    constant or in the main line's pieces, re-measured; B with or without an outside constant), each
    with its BIC, and the found model's own rows by the same likelihood; None when a least-squares
    screen of B's row finds no sign of a delay."""
    t = run["t"]
    L = levels_array(levels, ids)
    T = L.shape[1] - 1
    logn = math.log(2 * T)
    _, keep = split(flows, a, b)
    xA, gA = exogenous(flows, keep, ids, t, L, a)
    xB, gB = exogenous(flows, keep, ids, t, L, b)
    seg, spans = pieces(flows, a, t)
    shapes = [one_piece(seg, spans)] + ([(seg, spans)] if seg is not None and len(spans) > 1 else [])
    pairs = [(Pair(L[ids.index(a)], L[ids.index(b)], xA, gA, xB, gB, scale, sg), sp) for sg, sp in shapes]
    if not screen(pairs[0][0], T):
        return None
    best_t, best_p = None, None
    for pair, sp in pairs:
        extra = 2 * len(sp) - 1 if sp else 0                   # each piece: its value and its place
        for cB_on in (False, True):
            p = fit_plain(pair, cB_on)
            p_bic = p["nll"] + (p["k"] + extra) * logn
            if best_p is None or p_bic < best_p["bic"]:
                best_p = {**p, "bic": p_bic}
            f = fit_transit(pair, cB_on, True)
            if f is None:
                continue
            bic = f["nll"] + (f["k"] + extra) * logn
            if best_t is None or bic < best_t["bic"]:
                best_t = {**f, "bic": bic, "cB_on": cB_on, "spans": sp, "pair": pair, "extra": extra, "all_to_h": True}
    if best_t is None:
        return None
    # a share of A's loss to H (the rest to outside), on the best variant only
    f = fit_transit(best_t["pair"], best_t["cB_on"], False)
    if f is not None and f["nll"] + (f["k"] + best_t["extra"]) * logn < best_t["bic"]:
        best_t = {**best_t, **f, "bic": f["nll"] + (f["k"] + best_t["extra"]) * logn, "all_to_h": False}
    best_t["keep"] = keep
    # the found model's own rows of A and B, by the same likelihood (its pieces cost their numbers)
    allk = list(range(len(flows)))
    xA, gA = exogenous(flows, allk, ids, t, L, a)
    xB, gB = exogenous(flows, allk, ids, t, L, b)
    pair = Pair(L[ids.index(a)], L[ids.index(b)], xA, gA, xB, gB, scale)
    k = sum(flow_numbers(f) for f in flows if {a, b} & {f.get("from"), f.get("to")})
    found = {"nll": pair.plain(0.0, 0.0, 0.0, 0.0), "k": k}
    found["bic"] = found["nll"] + k * logn
    other = min(best_p["bic"], found["bic"])
    return {"transit": best_t, "plain": best_p, "found": found, "gain": other - best_t["bic"]}


def identify(stocks: list[dict], run: dict, opts: dict | None = None) -> dict:
    opts = dict(opts or {})
    margin = opts.pop("ht_margin", 4.0)
    found_ratio = opts.pop("ht_found_ratio", 1.5)
    slack = opts.pop("ht_slack", 0.0)
    shortest = opts.pop("ht_min_delay", 1.5)
    most = opts.pop("ht_max", 2)
    trace = opts.pop("ht_trace", False)
    turns = opts.pop("ht_turns", 15)
    r = main_line(stocks, run, opts)
    flows = r["flows"]
    ids = [s["id"] for s in stocks]
    if len(run["t"]) < 30 or not flows:
        return r
    unit = {s["id"]: (s.get("unit") or "", s.get("group")) for s in stocks}
    final = {s["id"] for s in stocks if s.get("final")}
    scale = 1.0 if counts(run) else 0.0
    inferred: dict[str, list[float]] = {}
    used: set[str] = set()
    base_e = judge(flows, stocks, run)
    tried = []
    for _ in range(most):
        levels = {**run["levels"], **inferred}
        all_ids = ids + list(inferred)
        best = None
        for a in ids:
            for b in ids:
                if a == b or a in final or a in used or b in used or unit[a][0] != unit[b][0]:
                    continue
                if unit[a][1] and unit[b][1] and unit[a][1] != unit[b][1]:
                    continue
                if not candidate(flows, a, b) or any(f.get("from") == b and f.get("to") == a for f in flows):
                    continue
                res = try_pair(flows, all_ids, run, levels, a, b, scale)
                if res is None:
                    continue
                f = res["transit"]
                tried.append({"pair": f"{a}>{b}", "gain": res["gain"], "delay": 1 / f["d"], "kin": f["kin"], "kA": f["kA"], "kb": f["kb"]})
                if trace:
                    print(f"  {a} -> H -> {b}: gain {res['gain']:.1f}; delay {1 / f['d']:.3f}, kin {f['kin']:.4f} of {f['kA']:.4f}, "
                          f"kb {f['kb']:.4f}, A in {np.round(f['values'], 2).tolist()[:6]}, cB {f['cB']:.3f} ({f['cB_on']})")
                # H must turn over many times in the run (a longer delay is a slow drift, which
                # pieces or a schedule explain as well)
                if not shortest <= 1 / f["d"] <= len(run["t"]) / turns:
                    continue
                if res["gain"] > margin * math.log(2 * (len(run["t"]) - 1)) and (best is None or res["gain"] > best[0]["gain"]):
                    best = (res, a, b)
        if best is None:
            break
        res, a, b = best
        f = res["transit"]
        new, h = build(flows, f["keep"], a, b, f, f["spans"])
        inf = {**inferred, h: np.maximum(0.0, f["H"]).tolist()}
        # pieces fitted to this run make the found model look good on it: the reference holds B's
        # outside constants (the arrivals H would explain) at their mean
        e = judge(new, stocks, run, inf)
        # (and A's too when the transit holds A's inflow constant: like against like)
        ref = judge(at_means(flows, {b} | ({a} if len(f["spans"]) <= 1 else set())), stocks, run, inferred)
        if trace:
            print(f"  accept {a} -> H -> {b}? judge {e:.4f} vs {ref:.4f} (as found {base_e:.4f}); plain {res['plain']['bic']:.1f}, found {res['found']['bic']:.1f}, transit {f['bic']:.1f}")
        if not (e <= (1 + slack) * ref + 1e-9 and e <= found_ratio * base_e + 1e-9):
            break
        flows, inferred, base_e = new, inf, e
        used |= {a, b}
    r["flows"] = flows
    r["transit"] = tried
    if inferred:
        r["inferred_levels"] = inferred
        r["chosen"] = f"{r.get('chosen')} + {len(inferred)} transit stock(s)"
    return r
