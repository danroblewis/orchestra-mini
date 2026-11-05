# Vendored for kalman_lna from sdlab/discovery/strategies/gap_prior/__init__.py at commit ec2f9e8 (imports rewritten to the
# vendored copies; otherwise unchanged). Do not edit: re-vendor instead.
"""gap_prior: lna, with readings the data cannot tell apart written the way SD models write them.

Without noise, a goal gap in b's outside inflow (c - k x a) and b's constant inflow c beside a drain
of b that reads another stock a (b loses k x a) are the same equation for b: the matrix returns the
drain. SD models do not drain a stock by another stock's level alone (an outflow reads its own
stock); a stock's inflow that falls as another stock rises is a goal gap. So, where the data are
noise-free (lna returned the matrix's reading), each such drain beside a single constant inflow is
written as the gap it equals: one flow instead of two, the same behaviour. Likewise a drain of c read
by a stock a that moves things on (a -> b) is, in the averages, a co-flow of that move (each unit
moved uses up per = drain / move of c; lna/coflows.py): where a has such a move, the drain is
written as the co-flow (checked first: a gap needs no move). With counting noise lna's likelihood
tells these apart (its gap swaps and co-flow offers) and its reading stands. A rewrite is kept only
when the judge finds it behaves as the original on the run (it claims the same equations), and
never between stocks whose levels are copies of each other (identical stocks: there the matrix's
cross readings are a choice between collinear columns, and a rewrite behaves differently off the
run).

This is a convention for an unidentifiable choice, not identification; it comes from
docs/research/identifiability.md and judge-ties.md (ties broken by the simpler, conventional model)
and is measured in docs/research/revisit.md.
"""

from __future__ import annotations

import math

import numpy as np

from orchestra.sdlab.discovery.identify import judge
from orchestra.sdlab.discovery.strategies.kalman_lna._vendored.lna import identify as _lna


def _reads(rate: dict) -> dict[str, float] | None:
    """A rate that is k x stocks with no constant: {stock: k}; else None."""
    form = rate.get("form")
    if form == "proportional":
        return {rate["stock"]: float(rate.get("k") or 0)}
    if form == "linear" and not rate.get("c"):
        return {t["stock"]: float(t.get("k") or 0) for t in rate.get("terms") or []}
    return None


def _plain(f: dict) -> bool:
    return not (f.get("inflow") or f.get("byTime") or f.get("effects") or f.get("uses") or f.get("via"))


def one_number(pieces: list[dict] | None, rel: float = 1e-4) -> bool:
    """Inflow pieces that are one number (the matrix can cut a constant into pieces that differ
    only in rounding)."""
    vals = [float(p.get("value") or 0) for p in pieces or []]
    return not vals or max(vals) - min(vals) <= rel * max(1e-9, max(abs(v) for v in vals))


def as_gaps(flows: list[dict], apart=None) -> list[dict]:
    """Each drain of b reading only other stocks, beside b's one constant outside inflow (in one
    piece), folded into that inflow as a goal gap (c - k x a)."""
    out = [dict(f) for f in flows]
    for drain in list(out):
        b = drain.get("from")
        if b is None or drain.get("to") is not None or not _plain(drain) or float(drain.get("ratio") or 1) != 1:
            continue
        reads = _reads(drain.get("rate") or {})
        if not reads or b in reads or any(k <= 0 for k in reads.values()):
            continue
        if apart is not None and not all(apart(b, a) for a in reads):
            continue
        ins = [f for f in out if f.get("from") is None and f.get("to") == b]
        if len(ins) != 1 or (ins[0].get("rate") or {}).get("form") != "constant":
            continue
        inflow = {k: v for k, v in ins[0].items() if not (k == "inflow" and one_number(v))}
        if not _plain(inflow):
            continue
        pcs = ins[0].get("inflow") or []
        w = [max(1, int(p["to"]) - int(p["from"]) + 1) for p in pcs]
        c = sum(float(p["value"]) * k for p, k in zip(pcs, w)) / sum(w) if pcs else float(inflow["rate"].get("value") or 0)
        rate = {"form": "linear", "c": c, "terms": [{"stock": s, "k": -k} for s, k in reads.items()]}
        per_step = float(inflow.get("perStep") or 0) - float(drain.get("perStep") or 0)
        out[out.index(ins[0])] = {**inflow, "rate": rate, "rates": [rate], "perStep": per_step, "note": "goal gap (written as SD writes it)"}
        out.remove(drain)
    return out


def as_coflows(flows: list[dict], apart=None) -> list[dict]:
    """Each drain of c reading one other stock a, where a moves things to a measured stock b != c at
    a plain proportional rate, folded into that move as a co-flow using up c."""
    out = [dict(f) for f in flows]
    for drain in list(out):
        c = drain.get("from")
        if c is None or drain.get("to") is not None or not _plain(drain) or float(drain.get("ratio") or 1) != 1:
            continue
        reads = _reads(drain.get("rate") or {})
        if not reads or len(reads) != 1 or c in reads:
            continue
        (a, k), = reads.items()
        if apart is not None and not apart(c, a):
            continue
        moves = [m for m in out if m.get("from") == a and m.get("to") not in (None, c) and _plain(m) and float(m.get("ratio") or 1) == 1
                 and (m.get("rate") or {}).get("form") == "proportional" and m["rate"].get("stock") == a and float(m["rate"].get("k") or 0) > 0]
        if len(moves) != 1 or k <= 0:
            continue
        move = moves[0]
        per = k / float(move["rate"]["k"])
        out[out.index(move)] = {**move, "uses": [{"stock": c, "per": per}], "note": "co-flow (written as SD writes it)"}
        out.remove(drain)
    return out


def identify(stocks: list[dict], run: dict, opts: dict | None = None) -> dict:
    found = _lna(stocks, run, opts)
    if found.get("method") == "lna":
        return found                    # counting noise: the likelihood chose between them
    return conventions(found, stocks, run, opts)


def apart(run: dict, tol: float = 1e-6):
    """apart(x, y): stock x's levels are not an affine copy of y's on the run (R^2 of x on y and a
    constant below 1 - tol). Where they are (identical stocks), a drain of x read by y is not a
    reading at all but the matrix's choice between collinear columns, and is left alone."""
    levels = {k: np.asarray(v, dtype=float) for k, v in run["levels"].items()}

    def f(x: str, y: str) -> bool:
        a, b = levels.get(x), levels.get(y)
        if a is None or b is None or a.std() == 0 or b.std() == 0:
            return True
        r = np.corrcoef(a, b)[0, 1]
        return bool(r * r < 1 - tol)
    return f


def conventions(found: dict, stocks: list[dict], run: dict, opts: dict | None = None) -> dict:
    """A noise-free model with its drains read by other stocks written as co-flows and goal gaps,
    when that behaves as the original on the run."""
    flows = as_gaps(as_coflows(found.get("flows") or [], apart(run)), apart(run))
    if len(flows) == len(found.get("flows") or []):
        return found
    # the rewrite claims the same equations: it must behave the same on the run (it need not where
    # a floor binds differently, or where the matrix's cross readings are not a gap or a co-flow)
    inj = (opts or {}).get("injections")
    before, after = judge(found["flows"], stocks, run, injections=inj), judge(flows, stocks, run, injections=inj)
    if not math.isfinite(after) or after > before * (1 + 1e-3) + 1e-9:
        return found
    return {**found, "flows": flows, "gap_prior": True}
