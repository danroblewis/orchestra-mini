"""A level's path over a flow's window, as the flow fitter reads it (``fit``): the level just before
any instant and how long it held each level, and the small optimiser the fits use. Copied from the
old Orchestra's ``sd/path.py``."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import defaultdict
from dataclasses import dataclass


def nelder_mead(f, x0: list[float], scale: list[float], iters: int = 400) -> list[float]:
    """Minimise f over a few parameters; small and dependency-free."""
    n = len(x0)
    pts = [list(x0)] + [[x0[i] + (scale[i] if i == j else 0.0) for i in range(n)] for j in range(n)]
    vals = [f(p) for p in pts]
    for _ in range(iters):
        order = sorted(range(n + 1), key=lambda i: vals[i])
        pts, vals = [pts[i] for i in order], [vals[i] for i in order]
        cen = [sum(p[i] for p in pts[:-1]) / n for i in range(n)]
        ref = [cen[i] + (cen[i] - pts[-1][i]) for i in range(n)]
        fr = f(ref)
        if fr < vals[0]:
            exp = [cen[i] + 2 * (cen[i] - pts[-1][i]) for i in range(n)]
            fe = f(exp)
            pts[-1], vals[-1] = (exp, fe) if fe < fr else (ref, fr)
        elif fr < vals[-2]:
            pts[-1], vals[-1] = ref, fr
        else:
            con = [cen[i] + 0.5 * (pts[-1][i] - cen[i]) for i in range(n)]
            fc = f(con)
            if fc < vals[-1]:
                pts[-1], vals[-1] = con, fc
            else:
                pts = [pts[0]] + [[pts[0][i] + 0.5 * (p[i] - pts[0][i]) for i in range(n)] for p in pts[1:]]
                vals = [vals[0]] + [f(p) for p in pts[1:]]
        if max(vals) - min(vals) < 1e-9:
            break
    return pts[min(range(n + 1), key=lambda i: vals[i])]


@dataclass
class Path:
    """A stock's level over the window, as the fitter needs it: the level just before any instant,
    and how long it spent at each level (in the model's time unit)."""

    times: list[float]
    values: list[float]
    time_at: dict[float, float]
    live: list[tuple[float, float]]
    unit_s: float

    @classmethod
    def of(cls, points: list[tuple[float, float]], live: list[tuple[float, float]], unit_s: float) -> Path:
        """The path, counting time only inside ``live`` (when every series involved was written)."""
        times, values = [p[0] for p in points], [p[1] for p in points]
        time_at: dict[float, float] = defaultdict(float)
        for start, end in live:
            i = bisect_right(times, start) - 1
            level = values[i] if i >= 0 else (values[0] if values else 0.0)
            t0 = start
            for t, v in points[max(0, i + 1):]:
                if t >= end:
                    break
                time_at[level] += (t - t0) / unit_s
                t0, level = t, v
            time_at[level] += (end - t0) / unit_s
        return cls(times, values, dict(time_at), list(live), unit_s)

    def before(self, t: float) -> float:
        """The level just before ``t``: what an event at ``t`` saw."""
        i = bisect_left(self.times, t) - 1
        return self.values[i] if i >= 0 else (self.values[0] if self.values else 0.0)

    def integral(self, c: float, k: float) -> float:
        """∫ max(0, c + k × level) dt over the window: a rate stops at zero (``fit_flow`` says
        when the data have shown that floor and when they have not)."""
        return sum(d * max(0.0, c + k * x) for x, d in self.time_at.items())

    def expected(self, a: float, b: float, c: float, k: float) -> float:
        """∫ max(0, c + k × level) dt over [a, b) within the live time: the events due between two
        instants, following the level between them."""
        total = 0.0
        for start, end in self.live:
            lo, hi = max(a, start), min(b, end)
            if hi <= lo:
                continue
            i = bisect_right(self.times, lo) - 1
            level, t0 = (self.values[i] if i >= 0 else (self.values[0] if self.values else 0.0)), lo
            for t, v in zip(self.times[i + 1:], self.values[i + 1:]):
                if t >= hi:
                    break
                total += (t - t0) * max(0.0, c + k * level)
                t0, level = t, v
            total += (hi - t0) * max(0.0, c + k * level)
        return total / self.unit_s

    def below_zero(self, c: float, k: float) -> tuple[float | None, float]:
        """The lowest level the stock held at which c + k × level is below zero, and how long it
        held all such levels (in the model's time unit)."""
        neg = [x for x in self.time_at if c + k * x < -1e-9]
        return (min(neg) if neg else None), sum(self.time_at[x] for x in neg)


def by_level(events: list[tuple[float, float]], xs: list[float]) -> list[tuple[float, float]]:
    """The events' counts summed by the level each saw: Σ count × log(c + k × level) over events
    is the same sum over levels, and a stock holds few levels (a count of pods, a queue's length)
    while it sees many events, so the likelihood the fitters evaluate thousands of times costs a
    pass over the levels, not the events."""
    out: dict[float, float] = {}
    for (_, d), x in zip(events, xs, strict=True):
        out[x] = out.get(x, 0.0) + d
    return list(out.items())
