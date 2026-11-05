"""A small PromQL subset, evaluated over ``MetricsStore.dump()``, for stores that have no query
language of their own (the default file store). VictoriaMetrics answers PromQL itself.

Supported: selectors with ``=``, ``!=``, ``=~``, ``!~``; ``rate``, ``increase``, ``delta``,
``*_over_time`` (avg, min, max, sum, count, last) over ``[duration]``; ``sum``, ``avg``, ``min``,
``max``, ``count`` with ``by``/``without``; arithmetic with a number (``* 3600``);
``label_replace(v, "dst", "replacement", "src", "regex")``; and ``a or b`` (every series of ``a``,
and those of ``b`` whose labels no series of ``a`` has), which is how several reads go in one
request (``orchestra.sources.batch``).

Values at each step are the last sample at or before it, carried forward without limit, as the
file store's ``range`` does. ``increase`` is the value at t minus the value at t − d (counter resets
counted), with no extrapolation, so it can differ slightly from Prometheus at window edges."""

from __future__ import annotations

import re
from bisect import bisect_right
from dataclasses import dataclass

AGGS = {"sum", "avg", "min", "max", "count"}
RANGE_FUNCS = {"rate", "increase", "delta", "avg_over_time", "min_over_time", "max_over_time",
               "sum_over_time", "count_over_time", "last_over_time"}
TOKEN = re.compile(r"""\s*(?:(?P<num>\d+(?:\.\d+)?(?:e[+-]?\d+)?)(?![a-zA-Z])|(?P<dur>\[\s*\d+[smhdwy]\s*\])|(?P<str>"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')
    |(?P<id>[a-zA-Z_:][a-zA-Z0-9_:]*)|(?P<op>=~|!~|!=|[=(){},*/+-]))""", re.VERBOSE)
UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800, "y": 31536000}


class PromQLError(ValueError):
    pass


@dataclass
class Selector:
    name: str | None
    matchers: list[tuple[str, str, str]]


def _tokens(q: str) -> list[tuple[str, str]]:
    out, i = [], 0
    while i < len(q):
        if q[i:].strip() == "":
            break
        m = TOKEN.match(q, i)
        if not m or m.end() == i:
            raise PromQLError(f"unexpected {q[i:].strip()[:12]!r} at {i}")
        kind = m.lastgroup
        out.append((kind, m.group(kind)))
        i = m.end()
    return out


class _Parser:
    def __init__(self, q: str):
        self.t = _tokens(q)
        self.i = 0

    def peek(self, k=0):
        return self.t[self.i + k] if self.i + k < len(self.t) else (None, None)

    def take(self, value=None):
        kind, v = self.peek()
        if kind is None or (value is not None and v != value):
            raise PromQLError(f"expected {value!r}, got {v!r}" if value else "unexpected end of query")
        self.i += 1
        return kind, v

    def parse(self):
        e = self.union()
        if self.i != len(self.t):
            raise PromQLError(f"unexpected {self.peek()[1]!r}")
        return e

    def union(self):
        e = self.expr()
        while self.peek() == ("id", "or"):
            self.take()
            e = ("or", e, self.expr())
        return e

    def expr(self):
        e = self.term()
        while self.peek()[1] in ("*", "/", "+", "-"):
            op = self.take()[1]
            kind, v = self.take()
            if kind != "num":
                raise PromQLError("this store supports arithmetic only with a number, e.g. * 3600")
            e = ("arith", op, e, float(v))
        return e

    def _grouping(self):
        v = self.peek()[1]
        if v in ("by", "without"):
            self.take()
            self.take("(")
            labels = []
            while self.peek()[1] != ")":
                labels.append(self.take()[1])
                if self.peek()[1] == ",":
                    self.take()
            self.take(")")
            return (v, labels)
        return None

    def term(self):
        kind, v = self.peek()
        if v == "(":
            self.take()
            e = self.union()
            self.take(")")
            return e
        if kind == "num":
            raise PromQLError("a query must select a metric")
        if kind == "id" and v in AGGS:
            self.take()
            grp = self._grouping()
            self.take("(")
            inner = self.union()
            self.take(")")
            grp = grp or self._grouping()
            return ("agg", v, grp, inner)
        if kind == "id" and v in RANGE_FUNCS:
            self.take()
            self.take("(")
            sel = self.selector()
            kind, d = self.take()
            if kind != "dur":
                raise PromQLError(f"{v}() needs a range, e.g. [1h]")
            self.take(")")
            d = d.strip("[] ")
            return ("range", v, sel, float(d[:-1]) * UNITS[d[-1]])
        if kind == "id" and v == "label_replace" and self.peek(1)[1] == "(":
            self.take()
            self.take("(")
            inner = self.union()
            args = []
            for _ in range(4):
                self.take(",")
                k, a = self.take()
                if k != "str":
                    raise PromQLError("label_replace needs four quoted strings after the vector")
                args.append(a[1:-1].encode().decode("unicode_escape"))
            self.take(")")
            return ("label_replace", inner, *args)
        if kind == "id" and self.peek(1)[1] == "(":
            raise PromQLError(f"{v}() is not supported by this store")
        return ("sel", self.selector())

    def selector(self):
        name = None
        if self.peek()[0] == "id":
            name = self.take()[1]
        matchers = []
        if self.peek()[1] == "{":
            self.take()
            while self.peek()[1] != "}":
                label = self.take()[1]
                op = self.take()[1]
                kind, val = self.take()
                if op not in ("=", "!=", "=~", "!~") or kind != "str":
                    raise PromQLError(f'expected label="value" after {label}')
                matchers.append((label, op, val[1:-1].encode().decode("unicode_escape")))
                if self.peek()[1] == ",":
                    self.take()
            self.take("}")
        if name is None and not matchers:
            raise PromQLError("expected a metric name or {label matchers}")
        return Selector(name, matchers)


def parse(q: str):
    return _Parser(q).parse()


def _match(sel: Selector, name: str, labels: dict) -> bool:
    if sel.name and sel.name != name:
        return False
    for k, op, v in sel.matchers:
        have = name if k == "__name__" else labels.get(k, "")
        ok = (have == v if op in ("=", "!=") else re.fullmatch(v, have) is not None)
        if ok == (op in ("!=", "!~")):
            return False
    return True


def _at(ts: list[float], vs: list[float], t: float) -> float | None:
    i = bisect_right(ts, t)
    return vs[i - 1] if i else None


def _window(fn: str, ts, vs, t, d):
    lo, hi = bisect_right(ts, t - d), bisect_right(ts, t)
    if fn in ("rate", "increase", "delta"):
        start, end = _at(ts, vs, t - d), _at(ts, vs, t)
        if end is None:
            return None
        if start is None:
            if lo == hi:
                return None
            start = vs[lo]
        if fn == "delta":
            return end - start
        inc, prev = 0.0, start
        for v in vs[lo:hi]:
            inc += v - prev if v >= prev else v       # a reset starts the counter again
            prev = v
        return inc / d if fn == "rate" else inc
    w = vs[lo:hi]
    if not w:
        return None
    return {"avg_over_time": lambda: sum(w) / len(w), "min_over_time": lambda: min(w), "max_over_time": lambda: max(w),
            "sum_over_time": lambda: sum(w), "count_over_time": lambda: float(len(w)), "last_over_time": lambda: w[-1]}[fn]()


def _eval(node, dump, steps) -> list[tuple[dict, list[float | None]]]:
    kind = node[0]
    if kind == "sel":
        out = []
        for name, labels, pts in dump:
            if _match(node[1], name, labels):
                ts, vs = [p[0] for p in pts], [p[1] for p in pts]
                out.append(({"__name__": name, **labels}, [_at(ts, vs, t) for t in steps]))
        return out
    if kind == "range":
        _, fn, sel, d = node
        out = []
        for name, labels, pts in dump:
            if _match(sel, name, labels):
                ts, vs = [p[0] for p in pts], [p[1] for p in pts]
                out.append((dict(labels), [_window(fn, ts, vs, t, d) for t in steps]))
        return out
    if kind == "or":
        left = _eval(node[1], dump, steps)
        seen = {tuple(sorted(lab.items())) for lab, _ in left}
        return left + [(lab, v) for lab, v in _eval(node[2], dump, steps) if tuple(sorted(lab.items())) not in seen]
    if kind == "label_replace":
        _, inner, dst, rep, src, rx = node
        out = []
        for lab, vals in _eval(inner, dump, steps):
            m = re.fullmatch(rx, lab.get(src, ""))
            if m:
                lab = dict(lab)
                new = m.expand(re.sub(r"\$(\d+)", r"\\\1", rep))
                if new:
                    lab[dst] = new
                else:
                    lab.pop(dst, None)
            out.append((lab, vals))
        return out
    if kind == "arith":
        _, op, inner, n = node
        f = {"*": lambda v: v * n, "/": lambda v: v / n if n else None, "+": lambda v: v + n, "-": lambda v: v - n}[op]
        return [({k: v for k, v in lab.items() if k != "__name__"}, [None if v is None else f(v) for v in vals])
                for lab, vals in _eval(inner, dump, steps)]
    _, op, grp, inner = node
    groups: dict[tuple, tuple[dict, list[list[float]]]] = {}
    keep_name = grp is not None and grp[0] == "by" and "__name__" in grp[1]     # `by (__name__)`: one group per metric
    for lab, vals in _eval(inner, dump, steps):
        if not keep_name:
            lab = {k: v for k, v in lab.items() if k != "__name__"}
        if grp is None:
            key_lab = {}
        elif grp[0] == "by":
            key_lab = {k: lab[k] for k in grp[1] if k in lab}
        else:
            key_lab = {k: v for k, v in lab.items() if k not in grp[1]}
        g = groups.setdefault(tuple(sorted(key_lab.items())), (key_lab, [[] for _ in steps]))
        for i, v in enumerate(vals):
            if v is not None:
                g[1][i].append(v)
    red = {"sum": sum, "avg": lambda xs: sum(xs) / len(xs), "min": min, "max": max, "count": lambda xs: float(len(xs))}[op]
    return [(lab, [red(xs) if xs else None for xs in cols]) for lab, cols in groups.values()]


def query_range(dump, q: str, start: float, end: float, step: float) -> list[tuple[dict, list[tuple[float, float]]]]:
    """Evaluate ``q`` at every step from ``start`` to ``end``: (labels, [(t, v)]) per result series."""
    if step <= 0 or end < start:
        raise PromQLError("bad range")
    n = int((end - start) // step) + 1
    if n > 11000:
        raise PromQLError("too many points: use a longer step")
    steps = [start + i * step for i in range(n)]
    out = []
    for lab, vals in _eval(parse(q), dump, steps):
        pts = [(t, v) for t, v in zip(steps, vals, strict=True) if v is not None]
        if pts:
            out.append((lab, pts))
    return out


def select(dump, selector: str, start: float, end: float) -> list[tuple[dict, list[tuple[float, float]]]]:
    """The raw samples in [start, end] of every series ``selector`` matches, labels with
    ``__name__``: what a Prometheus server answers to an instant query of ``selector[range]``."""
    p = _Parser(selector)
    sel = p.selector()
    if p.i != len(p.t):
        raise PromQLError(f"unexpected {p.peek()[1]!r}: give a selector")
    out = []
    for name, labels, pts in dump:
        if _match(sel, name, labels):
            got = [(t, v) for t, v in pts if start <= t <= end]
            if got:
                out.append(({"__name__": name, **labels}, got))
    return out
