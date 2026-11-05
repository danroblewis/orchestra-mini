"""kalman_lna: discovery for levels that are reported, not known (docs/research/kalman_lna.md).

The stocks are hidden states that move by the flows, with counting noise as the process noise
(lna's linear noise approximation: each flow a Poisson draw, a move's draw shared by both ends),
and the reported levels are observations of them: with a measurement error (absolute and
relative), at uneven times (each interval spans its steps), as means over a window (weekly
reports), late (a reporting delay, read where announced injections pin it), with gross
errors (left out), and with a stock counted at a scale (a bias, read where a move pins it). The
numbers come from the Kalman filter's likelihood, the flows from lna's atoms chosen by BIC
(run.py, search.py), unmeasured stocks between two measured ones as one more state (hidden.py).

Which reading the data need:
- points at uneven times, or one per window: the filter;
- every step, up to SMALL stocks (or announced injections): the filter's superset is fitted first
  (gross errors left out, a reporting delay read); a measurement error that gains LR_NOISE BIC
  prices sends the data to the filter (not where a stock sits at zero, nor where the atoms leave
  much variance beyond counting noise even with the error read: floors, effects and co-flows are
  outside the atoms, and a measurement error would stand in for them);
- gross errors the filter sees (a standardised innovation beyond 10, where counting noise and a
  misfit of the atoms stay below 6): the filter, which leaves them out;
- otherwise the base strategy's answer, unless the filter's search finds a reporting delay
  (injections), or a scale where the base reads a link beside a move into the same stock.

The base is its own copy of `combined` (as at commit ec2f9e8, in _vendored/). Where the filter is chosen and fails or runs out of
time, the base's answer is returned. Beyond SMALL stocks, with every step reported, the data go to
the base unchanged.
"""

from __future__ import annotations

import numpy as np

# combined and lna (and what they reach) as copies, vendored at commit ec2f9e8 (_vendored/): no other
# strategy is imported (sdlab/CLAUDE.md invariant 11), so changing them never changes this one
from ._vendored.combined import identify as combined
from ._vendored.combined.deadline import outer_left
from .run import BUDGET, FAILURES, Session

CHECK = 30.0          # seconds for the check on data a measurement error may not touch

BASE = "combined"


def _base(name: str):
    if name != BASE:
        raise ValueError(f"kalman_lna's base is its own copy of {BASE}, not {name!r}")
    return combined


def _levels(stocks: list[dict], run: dict) -> tuple[np.ndarray, np.ndarray]:
    ids = [s["id"] for s in stocks]
    t = np.asarray(run["t"], dtype=float)
    Y = np.array([np.asarray(run["levels"].get(i, [np.nan] * len(t)), dtype=float) for i in ids]).T
    return t, Y


def scale_pattern(flows: list[dict]) -> bool:
    """A link into a stock beside a move into it from the stock the link reads: what a stock
    counted at a scale looks like to a method that knows the levels exactly."""
    moves = {(f.get("from"), f.get("to")) for f in flows if f.get("from") and f.get("to")}
    for f in flows:
        if f.get("from") is None and f.get("to"):
            r = f.get("rate") or {}
            reads = {r.get("stock")} | {x.get("stock") for x in r.get("terms") or []}
            if any((a, f["to"]) in moves for a in reads if a):
                return True
    return False


SMALL = 4             # with up to this many stocks, the filter's superset is fitted to test for a
                      # measurement error (beyond, only uneven reports send the data to the filter)
LR_NOISE = 4.0        # twice the log-likelihood the measurement error gains, in BIC prices of one number: 4 is
                      # twice what BIC asks of one number (clean ladder runs reach 2.4 at most)


FLOOR = 0.05          # beyond this share of points at zero, no measurement error is read (floors bind)


MISFIT = 0.3          # nor where, with the error read, the atoms still leave this share of a stock's
                      # step variance unexplained (effects, co-flows, caps: an error would stand in)


CLOSER = 1.25         # the base's answer replaces the filter's only when the filter's multi-step error is this
                      # many times the base's (on the ladder, models that are both sound differ by up to ~1.15)


def _closer(filtered: dict, based: dict, stocks: list[dict], run: dict, opts: dict) -> dict:
    """The filter's answer, or the base's where its flows (numbers refitted for a measurement error) run
    closer to the data over 5, 20 and 100 steps (a filter's model with a stock nobody measures is kept:
    the judge runs on the measured levels only)."""
    from .eiv import _known, judge, refit

    ids = [s["id"] for s in stocks]
    if not _known(filtered.get("flows") or [], set(ids) | set(opts.get("inputs") or {})):
        return filtered
    mine = judge(filtered.get("flows") or [], ids, run, opts)
    theirs = judge(refit(based.get("flows") or [], stocks, run, opts)[0], ids, run, opts)
    if theirs * CLOSER < mine:
        return {**based, "kalman": {**filtered["kalman"], "reading": None, "filter_judged": [mine, theirs]}}
    return {**filtered, "kalman": {**filtered["kalman"], "filter_judged": [mine, theirs]}}


def noisy(ev: dict) -> bool:
    return ev["lr"] > LR_NOISE * ev["price"] and ev["floor"] <= FLOOR and ev["misfit"] <= MISFIT


def _identify(stocks: list[dict], run: dict, opts: dict | None = None) -> dict:
    opts = dict(opts or {})
    base = _base(opts.get("kalman_base", BASE))
    t, Y = _levels(stocks, run)
    uneven = len(t) > 1 and (bool(np.any(np.diff(t) != 1)) or not np.isfinite(Y).all())
    ses = Session(stocks, run, opts)
    info: dict = {}
    def allow(share, most):
        left = outer_left()                    # what the caller's timer leaves now
        ses.allow(min(most, share * left) if left else most)

    def kalman(why):
        allow(0.6, BUDGET)
        try:
            got = ses.finish()
        except FAILURES as e:
            ses.trace.append(f"failed: {type(e).__name__}: {e}")
            return None
        return {**got, "kalman": {"reading": why, **info}}

    prepared = False
    if uneven or len(stocks) <= SMALL or opts.get("injections"):
        allow(0.6 if uneven else 0.25, BUDGET if uneven else CHECK)
        try:
            prepared = ses.prepare()
            if prepared and not uneven:
                info["evidence"] = ses.evidence()
        except FAILURES as e:
            ses.trace.append(f"failed: {type(e).__name__}: {e}")
            prepared = False
    gross = int(ses.gross.sum()) if prepared else 0
    if prepared and (uneven or noisy(info["evidence"]) or gross):
        why = "uneven" if uneven else "meas" if noisy(info["evidence"]) else "outliers"
        got = kalman(why)
        if got is not None and why != "meas":
            return got
        if got is not None:
            # the filter's atoms read no measured input: its model is weighed against the base's, with the
            # base's numbers refitted for the measurement error (eiv.py), on the judge's multi-step error
            # (which a measurement error does not bias); the base's answer is kept only when it runs closer
            r = base(stocks, run, dict(opts))
            return _closer(got, r, stocks, run, opts)
    r = base(stocks, run, opts)
    if prepared and not uneven:
        scaled = scale_pattern(r.get("flows") or [])
        if ses.lag or scaled:
            got = kalman("lag" if ses.lag else "scale")
            if got is not None and (got.get("lag") or got.get("scales")):
                return got
    return {**r, "kalman": {"reading": None, **info, "trace": ses.trace}}


def identify(stocks: list[dict], run: dict, opts: dict | None = None) -> dict:
    """The answer above with its numbers refitted by instrumental variables where the levels carry
    measurement error (eiv.py), then kept inside the announced boundary (boundary.py: the guard and
    the moves where one is declared, each outside flow priced as a boundary claim where none is; the
    searches are inside it too: the filter's atoms, inside.py; the base's candidates, _vendored/)."""
    from . import boundary
    from .eiv import apply

    return boundary.apply(apply(_identify(stocks, run, dict(opts or {})), stocks, run, opts), stocks, run, opts)


# A capacity limit offered to the fit (capmin.py, docs/research/k8s-306-s1.md, 2026-10-21): identify above
# runs with the regime-clipped series MIN(X, theta x Y) as one more measured input where the data show a
# limit, and every term that reads it is written back as MIN(a x X, a theta x Y).
# (k8s-312) capsteps.py: capmin.py with a regime allowed to hold on as few as 5 steps (a short run's rare limit)
from . import capsteps as _capmin  # noqa: E402

_identify_linear = identify


def identify(stocks: list[dict], run: dict, opts: dict | None = None, *args, **kw) -> dict:  # noqa: F811
    """(2026-10-21, k8s-307, docs/research/k8s-307-c1.md) Between the fit and the MIN written back, where the
    levels carry measurement error, the numbers are refitted and the structure revised on the output error
    and judged on a Kalman likelihood (oefit.py, a copy of combined2's: the model run through the whole
    record against the measured levels)."""
    from . import oefit

    from . import tracker

    o, cap = _capmin.offer(stocks, run, opts)
    r = oefit.refit(_identify_linear(stocks, run, o, *args, **kw), stocks, run, o, cap)
    # (2026-10-21, k8s-308, docs/research/fixes/308-kalman_lna.md) a counted stock driven towards a target read
    # off another stock (a controller: rise by the gap to ceil(X / a), fall a step while above ceil(X / b)) replaces
    # the stock's outside flows where it beats them by BIC (tracker.py)
    # (2026-10-21, k8s-309, docs/research/fixes/309-kalman_lna.md) each stock's row read off the data (sparse linear,
    # or linear plus one term floored at 0 on a declared side), and the flows of the stocks whose rows the answer
    # misses rebuilt from the rows (moves paired by unit, the rest outside links) where that wins by BIC (rows.py)
    from . import rows

    r = rows.revise(_capmin.restore(r, cap), stocks, run, opts, cap)
    return tracker.revise(r, stocks, run, opts)
