# Vendored for kalman_lna from sdlab/discovery/strategies/censored/__init__.py at commit ec2f9e8 (imports rewritten to the
# vendored copies; otherwise unchanged). Do not edit: re-vendor instead.
"""Censored: floors and caps fitted in the form the engine steps them, on top of the main line.

Every rate in SD Lab is max(0, ...), and a stock's outflows share what it holds when they want more
(sdlab/sdtruth.py). Where a floor or a cap binds, the measured change is not the flows' equation,
and the main line's straight one-step rows cannot say so: a link c - k x_j that stops at zero comes
out as a drain on x_j plus inflow pieces copying the kink (024-floor-binds), and a constant demand
from a stock that runs empty cannot be told from its inflow at all (only their difference moves the
stock while it holds something; 025-runs-empty gets no flows).

This strategy takes the main line's chosen model and, row by row, offers two rewrites of the row's
flows to and from outside (moves between measured stocks are kept):

- floor.py: a hinge a x max(0, s (x_j - theta)) beside the row's own level (and the other levels it
  read), its threshold chosen by a search over x_j's range; read back as a floored linear rate;
- cap.py: a constant demand D (and own drain d) capped by the level, with the outside inflow in
  pieces measured as what arrived, D chosen by profiling the pieces' fit.

A rewrite is kept only if the whole model's one-step BIC, with floors and caps applied as the engine
applies them (onestep.py), improves by more than `margin` x log T. (Option `judge_slack`: also ask
the result to behave on the data no worse than that many times the main line's model by the main
line's judge; off by default, because that judge replays each model's own inflow pieces and so
prefers the main line's many pieces even where the rewrite is the truth: 013 counts-short, 2.15x.)
"""

from __future__ import annotations

import math

import numpy as np

from orchestra.sdlab.discovery.caps import flow_amounts
from orchestra.sdlab.discovery.identify import identify as main_line
from orchestra.sdlab.discovery.identify import judge
from orchestra.sdlab.discovery.matrix import injected
from orchestra.sdlab.discovery.simulate import Model
from orchestra.sdlab.discovery.strategies.kalman_lna._vendored.censored import cap, floor, onestep

DEFAULTS = {"margin": 2.0, "judge_slack": None, "floors": True, "caps": True, "recut": True, "pelt_mult": 2.0, "passes": 2, "bend": 0.0, "own_floors": False, "trim": 0.01, "floor_margin": None, "links": True}


def _touches_outside(f: dict, row: str) -> bool:
    return (f.get("from") is None and f.get("to") == row) or (f.get("from") == row and f.get("to") is None)


def _reads(f: dict) -> list[str]:
    r = f.get("rate") or {}
    if r.get("form") in ("proportional", "gap"):
        return [r.get("stock")]
    if r.get("form") == "linear":
        return [x.get("stock") for x in r.get("terms") or []]
    return []


def _own_drain(flows: list[dict], row: str) -> float:
    return sum(float(f["rate"].get("k") or 0) for f in flows
               if f.get("from") == row and f.get("to") is None and (f.get("rate") or {}).get("form") == "proportional" and f["rate"].get("stock") == row)


def _grows(flows: list[dict], row: str) -> bool:
    """Whether the main line has the row's stock grow with itself (a link from its own level)."""
    return any(f.get("from") is None and f.get("to") == row and _reads(f) == [row] for f in flows)


def _flows_of(flows, row):
    """The row's flows to and from outside (what a rewrite replaces), and the rest; None if any of
    them follows announced eras (left to the main line)."""
    mine = [f for f in flows if _touches_outside(f, row)]
    if any(f.get("byTime") for f in mine):
        return None
    return mine, [f for f in flows if not _touches_outside(f, row)]


def _came_and_wanted(base: list[dict], ids: list[str], t: list, prev: np.ndarray, i: int) -> tuple[np.ndarray, np.ndarray]:
    """What the kept flows bring into row i (as the model steps them) and what they want out of it."""
    T = prev.shape[1]
    came, wanted = np.zeros(T), np.zeros(T)
    if not base:
        return came, wanted
    model = Model(base, ids)
    times = np.asarray(t, dtype=float)[1 : T + 1]
    amounts = model._amounts(prev, times)
    for (src, dst, *_), a in zip(model.parts, amounts):
        if dst == i:
            came += a
    ends, want = flow_amounts(base, ids, t, prev)
    for (src, _, ratio), a in zip(ends, want):
        if src == i:
            wanted += a * ratio
    for k, c, per in model.uses:
        if c == i:
            wanted += want[k] * per
    return came, wanted


def floor_candidates(flows, ids, t, prev, dy, i, floor_rss, margin, o_bend=0.0, own=True):
    split = _flows_of(flows, ids[i])
    if split is None:
        return []
    mine, rest = split
    pieces = [f for f in mine if f.get("inflow")]
    readers = [j for j in range(len(ids)) if own or j != i]
    out = []
    for keep_pieces in ([False, True] if pieces else [False]):
        base = rest + (pieces if keep_pieces else [])
        removed = [f for f in mine if not (keep_pieces and f.get("inflow"))]
        y = dy[i] - onestep.changes(base, ids, t, prev)[i]
        others = sorted({ids.index(s) for f in removed for s in _reads(f) if s in ids and s != ids[i]})
        for constant in (False, True):
            if keep_pieces and constant:
                continue
            T = len(y)
            for fit in floor.search(y, prev, i, others, constant=constant, readers=readers):
                # a kink only if the hinge beats a straight line through the same columns
                gain = T * math.log(max(fit["line_rss"], floor_rss) / max(fit["rss"], floor_rss))
                bend = T * math.log(max(fit["smooth_rss"], floor_rss) / max(fit["rss"], floor_rss))
                if gain < (1 + margin) * math.log(T) or bend < o_bend * math.log(T):
                    continue
                if fit["beta"][1] > 0 and not _grows(flows, ids[i]):
                    continue                    # self-growth the main line did not find (see links)
                fit["i"] = i
                out.append((base + floor.as_flows(fit, ids), f"floor on {ids[fit['j']]} into {ids[i]}" + (" (inflow pieces kept)" if keep_pieces else "")))
    return out


def link_candidates(flows, ids, t, prev, dy, i):
    """A row the main line reads as "drains faster when x_j is high" beside an outside inflow, read
    instead as one inflow c + k x_j (k < 0), floored at 0 as the engine floors it, instead of an
    outflow that an emptied stock caps (a goal gap whose goal-seeking term the matrix turns into a
    drain). Same numbers on uncapped steps; the one-step BIC in the engine's form decides."""
    split = _flows_of(flows, ids[i])
    if split is None:
        return []
    mine, rest = split
    row = ids[i]
    drains = [f for f in mine if f.get("from") == row and (f.get("rate") or {}).get("form") == "linear"
              and _reads(f) and row not in _reads(f)]
    if not drains:
        return []
    # the link's constant replaces the row's outside inflow: only one that is really constant (a
    # changing one is left to its pieces; a level standing in for them is the main line's old trap)
    for f in mine:
        vals = np.array([g["value"] for g in f.get("inflow") or []], dtype=float)
        if len(vals) > 1 and vals.std() > 0.05 * abs(vals.mean()):
            return []
    y = dy[i] - onestep.changes(rest, ids, t, prev)[i]
    T = len(y)
    others = sorted({ids.index(s) for f in mine for s in _reads(f) if s in ids and s != row})
    out = []
    for j in sorted({ids.index(s) for f in drains for s in _reads(f) if s in ids}):
        cols = [np.ones(T), prev[j], prev[i]] + [prev[m] for m in others if m != j]
        beta = np.linalg.lstsq(np.array(cols).T, y, rcond=None)[0]
        if beta[2] > 0 and not _grows(flows, row):
            # no self-growth the main line did not find: it cancels the drains and runs away
            cols[2] = np.zeros(T)
            beta = np.linalg.lstsq(np.array(cols).T, y, rcond=None)[0]
        c, k, own = float(beta[0]), float(beta[1]), float(beta[2])
        if c <= 0 or k >= 0:
            continue
        fit = {"i": i, "j": j, "s": -1.0, "theta": c / -k, "beta": np.array([-k, own, *beta[3:]]),
               "others": [m for m in others if m != j], "constant": False}
        out.append((rest + floor.as_flows(fit, ids), f"link {c:.3g} - {-k:.3g} x {ids[j]} into {row}"))
    return out


def cap_candidates(flows, ids, t, prev, dy, i, floor_rss, margin, recut=False, mult=2.0, changes=True):
    split = _flows_of(flows, ids[i])
    if split is None:
        return []
    mine, rest = split
    row = ids[i]
    # the outside inflow and the row's own constant or proportional drain are refitted; any other
    # flow to or from outside (a link, a drain read off another stock) is kept and counts as wanted
    refit = [f for f in mine if (f.get("from") is None and not _reads(f)) or
             (f.get("from") == row and (not _reads(f) or _reads(f) == [row]) and (f.get("rate") or {}).get("form") in ("constant", "proportional"))]
    base = [f for f in rest] + [f for f in mine if f not in refit]
    came, wanted = _came_and_wanted(base, ids, t, prev, i)
    fit = cap.search(prev[i], dy[i], came, wanted, _own_drain(refit, row), floor_rss, changes=changes, mult=mult)
    if fit is None:
        return []
    at = lambda k: t[k + 1]  # noqa: E731
    out = []
    # (recut) the plain fit: the outside inflow's pieces at an honest price, with the own drain
    # the re-cut may lower or keep the main line's own drain, never raise it: on a stock whose level
    # barely moves, a drain and a larger inflow trade off almost freely, and BIC alone lets the pair grow
    d0 = _own_drain(refit, row)
    if recut and fit.get("plain_fit") and fit["plain_fit"]["d"] <= 1.1 * d0 + 1e-9:
        pf = fit["plain_fit"]
        out.append((base + cap.as_flows(pf, row, prev[i], dy[i], came, wanted, at), f"inflow into {row} re-cut ({len(pf['cuts']) - 1} pieces, drain {pf['d']:.3g})"))
    if fit["D"] <= 0 or fit["capped"].sum() < max(5, 0.02 * len(dy[i])):
        return out
    if fit["plain"] is not None and fit["score"] > fit["plain"] - margin * math.log(len(dy[i])):
        return out
    return out + [(base + cap.as_flows(fit, row, prev[i], dy[i], came, wanted, at), f"demand {fit['D']:.3g} from {row} capped on {int(fit['capped'].sum())} steps")]


def identify(stocks: list[dict], run: dict, opts: dict | None = None) -> dict:
    opts = dict(opts or {})
    o = {k: opts.pop(k, v) for k, v in DEFAULTS.items()}
    r0 = main_line(stocks, run, opts)
    ids = [s["id"] for s in stocks]
    t = run["t"]
    lv = onestep.levels(stocks, run)
    if lv.shape[1] < 30 or not ids:
        return r0
    prev, dy = lv[:, :-1], np.diff(lv, axis=1)
    if opts.get("injections"):
        inj = injected(opts["injections"], ids, t)[:, 1:]
        prev, dy = prev + inj, dy - inj
    T = prev.shape[1]
    # what an exact fit can reach: levels are recorded to 4 decimals (and numbers are rounded when
    # read back as flows), so a row fitted to within ~1e-4 of the data per step is exact
    floor_rss = T * ((1e-4 * dy.std(axis=1)) ** 2 + 1e-8)
    final = {s["id"] for s in stocks if s.get("final")}
    flows = list(r0.get("flows") or [])
    # the main line's largest 1% residuals per row are left out of every comparison (inflow spikes)
    skip = onestep.outliers(flows, ids, t, prev, dy, o["trim"]) if o["trim"] else None
    score = onestep.bic(flows, ids, t, prev, dy, floor_rss, skip)
    log_t = math.log(T)
    taken = []
    for _ in range(o["passes"]):
        changed = False
        for i, row in enumerate(ids):
            if row in final:
                continue
            cands = []
            if o["floors"]:
                cands += floor_candidates(flows, ids, t, prev, dy, i, float(floor_rss[i]), o["floor_margin"] or o["margin"], o["bend"], o["own_floors"])
            if o["links"]:
                cands += link_candidates(flows, ids, t, prev, dy, i)
            if o["caps"] or o["recut"]:
                cs = cap_candidates(flows, ids, t, prev, dy, i, float(floor_rss[i]), o["margin"], o["recut"], o["pelt_mult"])
                cands += [c for c in cs if o["caps"] or "re-cut" in c[1]]
            best = None
            for fl, name in cands:
                sc = onestep.bic(fl, ids, t, prev, dy, floor_rss, skip)
                if sc < score - o["margin"] * log_t and (best is None or sc < best[1]):
                    best = (fl, sc, name)
            if best:
                flows, score = best[0], best[1]
                taken.append({"row": row, "rewrite": best[2], "bic": round(score, 2)})
                changed = True
        if not changed:
            break
    if not taken:
        return {**r0, "censored": []}
    if not o["judge_slack"]:
        return {**r0, "flows": flows, "censored": taken}
    e0 = judge(r0.get("flows") or [], stocks, run, injections=opts.get("injections"))
    e1 = judge(flows, stocks, run, injections=opts.get("injections"))
    if math.isfinite(e0) and e1 > o["judge_slack"] * e0:
        return {**r0, "censored": taken, "censored_rejected": {"judge": e1, "main_line": e0}}
    return {**r0, "flows": flows, "censored": taken, "judge": {"censored": e1, "main_line": e0}}
