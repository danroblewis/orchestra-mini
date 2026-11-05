"""Fitting a flow's equation from its counter's events and the stocks' paths. Adapted from the old
Orchestra's ``sd/discover.py`` (``fit_flow``, ``events_of``), without its driver refutations
(``sd/driverchoice.py``), slow drivers and flags.

Each candidate form is scored with the exact point-process likelihood, each event against the
rate just before it, the rate integrated over the stocks' actual paths:

    log L = Σ count × log rate(t⁻)  −  ∫ rate dt        (rate = c + k × X)

The forms: ``constant`` (rate = c); ``proportional`` (rate = k × X, X the flow's source, or another
stock that held something whenever the flow's events came); ``linear`` (rate = c + k × X). A form
reading another stock (an influence) needs ``EVENTS_PER_PARAM`` events per parameter; with fewer
than ``MIN_EVENTS`` events only the constant is offered. The plain form (constant, or proportional
to the source) with the lowest AIC is chosen unless another beats it by ``AIC_MARGIN``. A line
below zero at levels the stock held too briefly for ``MIN_EVENTS`` events to be due is rejected:
the data never saw that floor."""

from __future__ import annotations

import math

from .balance import MIN_EVENTS
from .path import Path, by_level, nelder_mead

AIC_MARGIN = 2.0
EVENTS_PER_PARAM = 10


def events_of(points: list[tuple[float, float]], start: float, end: float, baseline: float | None) -> list[tuple[float, float]]:
    """A counter's increments in (start, end], as (time, count); a drop is a reset. ``baseline``:
    its value before its first point (0 when it counted from nothing, None when it was counting)."""
    out, prev = [], baseline
    for t, v in points:
        if prev is not None and start < t <= end:
            d = v - prev if v >= prev else v
            if d > 0:
                out.append((t, d))
        prev = v
    return out


def fit_flow(events: list[tuple[float, float]], window: float, paths: dict[str, Path], src: str | None) -> list[dict]:
    """Every candidate form for one flow, the chosen one first. ``window``: the time it was fitted
    on, in the model's time unit; ``paths``: the stocks' paths over it."""
    total = sum(d for _, d in events)
    if total < MIN_EVENTS:
        paths, src = {}, None
    at = {sid: by_level(events, [p.before(t) for t, _ in events]) for sid, p in paths.items()}
    flat = by_level(events, [0.0] * len(events))

    def ll(c: float, k: float, sid: str | None) -> float:
        out = 0.0
        for x, d in at[sid] if sid else flat:
            r = c + k * x
            if r <= 1e-12:
                return -math.inf
            out += d * math.log(r)
        return out - (paths[sid].integral(c, k) if sid else max(0.0, c) * window)

    c0 = total / window if window > 0 else 0.0
    cands = [{"form": "constant", "params": {"c": c0}, "inputs": [], "loglik": ll(c0, 0.0, None) if c0 > 0 else 0.0, "k": 1}]
    for sid, p in paths.items():
        xs = list(p.time_at)
        if not xs or max(xs) - min(xs) < 1e-9:
            continue
        expo = p.integral(0.0, 1.0)
        if expo > 0 and (sid == src or total >= EVENTS_PER_PARAM):
            k = total / expo
            cands.append({"form": "proportional", "params": {"k": k}, "inputs": [sid], "loglik": ll(0.0, k, sid), "k": 1})
        if total < 2 * EVENTS_PER_PARAM:
            continue
        span = max(xs) - min(xs)

        def nll(q, sid=sid):
            v = ll(q[0], q[1], sid)
            return -v if math.isfinite(v) else 1e18
        cc, kk = nelder_mead(nll, [c0, 0.0], [max(c0, 1e-3) * 0.5, max(c0, 1e-3) / span])
        cand = {"form": "linear", "params": {"c": cc, "k": kk}, "inputs": [sid], "loglik": ll(cc, kk, sid), "k": 2}
        level, held = p.below_zero(cc, kk)
        if level is not None and c0 * held < MIN_EVENTS:
            cand["rejected"] = (f"its rate is below zero from level {level:g} of {sid}, which the stock held only long enough "
                                f"for {c0 * held:.1f} events: too few to have seen that nothing flows there")
        cands.append(cand)
    for cand in cands:
        cand["aic"] = 2 * cand["k"] - 2 * cand["loglik"] if math.isfinite(cand["loglik"]) else math.inf
        if not math.isfinite(cand["aic"]):
            cand.setdefault("rejected", "an event came while its rate was zero: the form cannot produce the data")
    ok = [c for c in cands if "rejected" not in c]
    plain = min((c for c in ok if c["k"] == 1 and c["inputs"] in ([], [src])), key=lambda c: c["aic"], default=cands[0])
    rivals = [c for c in ok if c is not plain and c["aic"] < plain["aic"] - AIC_MARGIN]
    best = min(rivals, key=lambda c: c["aic"]) if rivals else plain
    return [best, *sorted((c for c in cands if c is not best), key=lambda c: ("rejected" in c, c["aic"]))]


__all__ = ["AIC_MARGIN", "EVENTS_PER_PARAM", "Path", "events_of", "fit_flow"]
