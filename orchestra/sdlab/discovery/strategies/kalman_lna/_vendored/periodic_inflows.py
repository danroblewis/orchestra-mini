# Vendored for kalman_lna from sdlab/discovery/strategies/periodic_inflows.py at commit ec2f9e8 (imports rewritten to the
# vendored copies; otherwise unchanged). Do not edit: re-vendor instead.
"""Outside inflows as a repeating profile (docs/research/periodic-inflows.md).

The main line first. Then each stock whose outside inflow is better told, by BIC, as one period's
profile repeated plus a slow level (discovery/periodic.py) has that inflow taken out as known, and the
main line runs again on the rest: its rows' levels are fitted beside the profile instead of beside
pieces that could follow anything, and whatever the profile leaves is still free to become pieces.
The profile is written into the result as an ordinary outside inflow in segments (with the pieces the
second pass found added in), so the simulator and the page are unchanged. Without a period that pays,
the result is the main line's.
"""

from __future__ import annotations

import numpy as np

from orchestra.sdlab.discovery.identify import identify as matrix_identify
from orchestra.sdlab.discovery.identify import VARIANTS, judge
from orchestra.sdlab.discovery.matrix import configurations, identify_one, known_changes, known_flows
from orchestra.sdlab.discovery.periodic import find, refit


def _series(flow: dict, times: np.ndarray) -> np.ndarray:
    segs = flow["inflow"]
    out = np.full(times.shape[0], float(segs[-1]["value"]))
    for g in reversed(segs):
        m = (times >= g["from"]) & (times <= g["to"])
        out[m] = float(g["value"])
    return out


def _segments(times: np.ndarray, v: np.ndarray) -> list[dict]:
    v = np.round(v, 4)
    out = []
    for k in range(v.shape[0]):
        if out and out[-1]["value"] == v[k]:
            out[-1]["to"] = float(times[k])
        else:
            out.append({"from": float(times[k]), "to": float(times[k]), "value": float(v[k])})
    return out


def with_profiles(flows: list[dict], profiles: dict, t: list) -> list[dict]:
    """flows with each profiled stock's outside inflow = its profile + the pieces found beside it."""
    times = np.asarray(t[1:], dtype=float)
    out = []
    total = {s: fp["series"].copy() for s, fp in profiles.items()}
    for f in flows:
        if f.get("inflow") and not f.get("from") and f.get("to") in total:
            total[f["to"]] += _series(f, times)
            continue
        if f.get("inflow") and f.get("from") in total and not f.get("to"):
            total[f["from"]] -= _series(f, times)
            continue
        out.append(f)
    for s, v in total.items():
        mean = float(v.mean())
        segs = _segments(times, v)
        if mean >= 0:
            out.append({"from": None, "to": s, "ratio": 1, "rate": {"form": "constant", "value": round(mean, 4)}, "rates": [{"form": "constant", "value": round(mean, 4)}],
                        "perStep": mean, "nearly": [], "bothWays": False, "inflow": segs, "changes": len(segs) - 1, "period": profiles[s]["P"]})
        else:
            neg = [{**g, "value": -g["value"]} for g in segs]
            out.append({"from": s, "to": None, "ratio": 1, "rate": {"form": "constant", "value": round(-mean, 4)}, "rates": [{"form": "constant", "value": round(-mean, 4)}],
                        "perStep": -mean, "nearly": [], "bothWays": False, "inflow": neg, "changes": len(neg) - 1, "period": profiles[s]["P"]})
    return out


def remeasure(r: dict, profiles: dict, stocks: list[dict], run: dict, opts: dict) -> None:
    """Each profile measured again beside the numbers its row ends with (conservation may have
    changed its drain after the profile was measured), with the pieces found beside it kept."""
    ids = [s["id"] for s in stocks]
    t = run["t"]
    T = len(t) - 1
    times = np.asarray(t[1:], dtype=float)
    lv = np.nan_to_num(np.array([np.asarray(run["levels"].get(i, [0] * len(t)), dtype=float) for i in ids]))
    prev, dy = lv[:, :-1], np.diff(lv, axis=1)
    if opts.get("known"):
        dy = dy - known_changes(opts["known"], ids, len(t))[:, 1:]
    conf = configurations(opts.get("eras"), t) if r.get("eras") else None
    cfg = conf[1] if conf else np.zeros(T, dtype=int)
    rows = {m["id"]: m for m in r.get("matrix") or []}
    for s, fp in profiles.items():
        row = rows.get(s)
        if row is None:
            continue
        k = ids.index(s)
        res = dy[k].copy()
        for c, coef in enumerate(row["coefL"]):
            m = cfg == c
            for j, b in coef.items():
                res[m] -= b * prev[ids.index(j)][m]
        for f in r["flows"]:
            if f.get("inflow") and not f.get("from") and f.get("to") == s:
                res -= _series(f, times)
            elif f.get("inflow") and f.get("from") == s and not f.get("to"):
                res += _series(f, times)
        fp["series"] = refit(res, fp)


def identify(stocks: list[dict], run: dict, opts: dict | None = None) -> dict:
    opts = dict(opts or {})
    profiles = find(stocks, run, opts)
    if not profiles:
        r = matrix_identify(stocks, run, opts)
        r["periods"] = {}
        return r
    n = len(run["t"])
    ids = [s["id"] for s in stocks]
    known = {k: list(np.nan_to_num(np.asarray(v, dtype=float))[:n]) + [0.0] * max(0, n - len(v)) for k, v in (opts.get("known") or {}).items()}
    for s, fp in profiles.items():
        base = np.asarray(known.get(s) or [0.0] * n, dtype=float)
        base[1:] += fp["series"]
        known[s] = list(base)
    # the main line's candidates, each with the profiles taken out as known; a profiled row's inflow
    # varies, so conservation may set its drain from its receivers, as for a row with pieces. Each
    # candidate's profiles are measured again beside its numbers before it is judged.
    has_eras = len({e.get("label") or e.get("start") for e in opts.get("eras") or [] if e}) >= 2
    extra = known_flows(opts["known"], ids, run["t"]) if opts.get("known") else []
    variants = opts.pop("variants", None) or VARIANTS
    best, tried = None, []
    for v in variants:
        if v.get("split_eras") and not has_eras:
            continue
        r = identify_one(stocks, run, {**opts, **v, "known": known, "varying_inflow": set(profiles)})
        own = {s: dict(fp) for s, fp in profiles.items()}
        remeasure(r, own, stocks, run, opts)
        r["flows"] = with_profiles(r["flows"], own, run["t"])
        e = judge(r["flows"] + extra, stocks, run)
        tried.append({"name": v["name"], "error": e})
        if best is None or e < best[1]:
            best = (r, e, v["name"])
    r = best[0]
    r["chosen"], r["tried"] = best[2], tried
    if opts.get("against_main_line", True):
        # the main line's own choice competes on the same terms: a profile that takes an internal
        # flow (a rate that repeats, times a level) makes the model run away, and the judge sees it
        m = matrix_identify(stocks, run, {**opts, "variants": variants})
        e = min(x["error"] for x in m["tried"])
        if e < best[1]:
            m["tried"] = tried + [{"name": "main line: " + x["name"], "error": x["error"]} for x in m["tried"]]
            m["periods"] = {}
            m["rejected_periods"] = {s: fp["P"] for s, fp in profiles.items()}
            return m
        r["tried"] = tried + [{"name": "main line: " + x["name"], "error": x["error"]} for x in m["tried"]]
    r["periods"] = {s: {"period": fp["P"], "segments": len(fp["segments"]), "slow": len(fp["slow"]) - 1} for s, fp in profiles.items()}
    return r
