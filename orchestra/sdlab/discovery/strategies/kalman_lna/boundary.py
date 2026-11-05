# Copied from sdlab/discovery/strategies/matrix/boundary.py at 723f72c (strategies are isolated: copied,
# never imported; HIDDEN_PRICE 0). Imports only the shared simulator. One change in this copy: where a
# boundary is declared, no alternative or move may take away the last outside flow on a declared side
# (`estimate` means the flow exists; kept_sides): without it, pair_moves made a counted chain's declared
# outside -> A and B -> outside into a closed loop B -> A whose numbers fit nearly as well. And a second:
# the guard refits at most GUARD_FITS structures per call; past it, a flow still outside the boundary
# is dropped without a refit (the largest models stay within their time; the answer stays legal).
"""The declared boundary (docs/research/declared-boundary.md): outside flows only where the case says the
system meets its environment.

A case may announce its boundary (Sterman's model-boundary chart): run["boundary"] = {stock: {"in":
[kinds], "out": [kinds]}} with run["boundary_declared"] = True, each kind "measured" (driven by a
measured input series) or "estimate" (exists; its form and rate are to be estimated). A stock not
listed has no outside flow. Nothing else about the truth is announced. Outside flows are a modelling
decision, not something discovery may invent, so:

- declared: an outside flow is allowed only on a declared stock and side, and of a declared kind (a
  "measured" side is fed by a flow that reads a measured input; an "estimate" side by an estimated
  form: a constant, pieces, a drain, a link). Everything else must be a flow between stocks, and
  unexplained wiggle is noise. A flow outside the boundary is never kept: it is dropped, or made a move
  between two stocks, whichever prices best (`guard`).
- not declared (real data, some probes): an outside flow is allowed but is a boundary claim, priced at
  CLAIM_PRICE x log T on top of its numbers (`price_claims`), and listed (`claims`).
- either way, an outflow from A and an inflow into B whose unexplained changes are anti-correlated
  (what one loses the other gains) are tried as the move A -> B (`pair_moves`), kept when it prices
  better.

The price of a structure is BIC on the one-step changes: -2 log L (Gaussian, one variance per stock)
plus log T per number, plus CLAIM_PRICE x log T per boundary claim. A hidden stock is inside the
boundary and pays only for its numbers (HIDDEN_PRICE = 0; the user, 2026-10-20). Numbers are refitted for every
structure compared (`Fit`), by least squares on the one-step changes, so a structure is never judged
with numbers fitted for another.

The module is self-contained (it imports only the shared simulator): strategies are isolated, so
another strategy copies this file into its own package (the "for copiers" section of the doc).
"""

from __future__ import annotations

import copy
import math

import numpy as np

from ...simulate import Model

CLAIM_PRICE = 2.0      # an outside flow with no declared boundary pays this many log T beyond its numbers
HIDDEN_PRICE = 0.0     # a hidden stock is part of the system inside the boundary (the user, 2026-10-20): it pays only for its numbers, no surcharge
ANTI_Z = 3.0           # two stocks' unexplained changes anti-correlated beyond this z: the move is tried
ITERATIONS = 4         # Gauss-Newton steps of a refit (one is exact for linear rates without clipping)
MIN_STEPS = 20         # fewer usable steps: no refit (a flow outside the boundary is then just dropped)
MAX_FITS = 120         # the optional passes (moves, claims) stop after this many refits (deterministic)
GUARD_FITS = 60        # (this copy) the guard's refits per call; past them a flow outside the boundary is dropped




class Boundary:
    """The announced boundary, or none (every side open, each outside flow a claim)."""

    KINDS = ("measured", "estimate")

    def __init__(self, run: dict):
        self.declared = bool(run.get("boundary_declared"))
        raw = run.get("boundary") or {}
        # (2026-10-21) an "input" side: searched as "measured", its flow fixed at 1 x the input (enforce_sides)
        self.original = {s: {side: list(e.get(side) or []) for side in ("in", "out")} for s, e in raw.items()}
        self.fixed = {s: e["in"].count("input") for s, e in self.original.items() if "input" in e["in"]} if self.declared else {}
        self.limit = {(s, side): len(e[side]) for s, e in self.original.items() for side in ("in", "out") if e[side]} if self.declared else {}
        raw = _as_measured(raw)
        self.sides = {s: {side: set(e.get(side) or []) for side in ("in", "out")} for s, e in raw.items()}

    def kinds(self, stock: str, side: str) -> set:
        if not self.declared:
            return set(self.KINDS)
        return self.sides.get(stock, {}).get(side, set())

    def may(self, stock: str, side: str, kind: str | None = None) -> bool:
        k = self.kinds(stock, side)
        # (fix, as 728efec) a side declared "estimate" may hold any outside flow (its form is to be found and
        # may read a measured input); a side declared only "measured" holds only inflows reading inputs
        return bool(k) if kind in (None, "measured") else kind in k

    # what a row regression (one stock's change on the levels, inputs and an outside inflow) may hold
    def pieces_ok(self, stock: str) -> bool:
        """A constant or changing outside term (positive: an inflow, negative: an outflow)."""
        return self.may(stock, "in", "estimate") or self.may(stock, "out", "estimate")

    def inputs_ok(self, stock: str) -> bool:
        """A measured input's column (positive: an inflow reading it, negative: an outflow)."""
        return self.may(stock, "in") or self.may(stock, "out")   # (fix, 728efec) wherever an outside flow is declared

    def links_ok(self, stock: str) -> bool:
        """A column of a stock of another unit (a link from outside, or a drain to outside, reading it)."""
        return self.may(stock, "in", "estimate") or self.may(stock, "out", "estimate")

    def must_tie(self, stock: str) -> bool:
        """Its drain must be all moves: no estimated outflow to outside is declared."""
        return self.declared and not self.may(stock, "out", "estimate")

    def summary(self) -> dict:
        return {"declared": self.declared,
                "sides": {s: {k: sorted(v) for k, v in e.items() if v} for s, e in self.sides.items()}}




def _rate_stocks(f: dict) -> set:
    out = set()
    for r in [g["rate"] for g in f["byTime"]] if f.get("byTime") else [f.get("rate") or {}]:
        out.add(r.get("stock"))
        out |= {x.get("stock") for x in r.get("terms") or []}
    return out - {None}


def kind_of(f: dict, inputs: set) -> str:
    """"measured" for an outside flow that reads a measured input, else "estimate"."""
    # (fix, as 728efec) as the cases declare kinds: "measured" only for an outside inflow whose rate reads
    # only measured inputs; "estimate" for outflows and for inflows that read a stock, a constant or pieces
    if f.get("from") is not None or f.get("inflow"):
        return "estimate"
    reads = _rate_stocks(f) | ({f["input"]} if f.get("input") else set())
    r = f.get("rate") or {}
    const = r.get("form") == "constant" or bool(r.get("c"))
    return "measured" if reads and reads <= set(inputs) and not const else "estimate"


def ends(flows: list[dict], ids: list[str]) -> list[list[tuple[str, str]]]:
    """Per flow, the measured stocks and sides its outside end reaches, as the scorer reads them
    (sdlab/ladder/parts.outside_ends): a flow from outside into a measured stock, or into an unmeasured
    one whose flows lead on to a measured one; the same for flows to outside; [] for a flow between
    stocks."""
    measured = set(ids)
    down: dict = {}
    up: dict = {}
    for f in flows:
        if f.get("from") and f.get("to"):
            down.setdefault(f["from"], set()).add(f["to"])
            up.setdefault(f["to"], set()).add(f["from"])

    def reach(start: str, edges: dict) -> set:
        seen, todo, hit = set(), [start], set()
        while todo:
            x = todo.pop()
            if x in seen:
                continue
            seen.add(x)
            if x in measured:
                hit.add(x)
                continue
            todo += list(edges.get(x) or ())
        return hit

    out = []
    for f in flows:
        a, b = f.get("from"), f.get("to")
        if not a and b:
            out.append(sorted((s, "in") for s in reach(b, down)))
        elif a and not b:
            out.append(sorted((s, "out") for s in reach(a, up)))
        else:
            out.append([])
    return out


def violations(flows: list[dict], ids: list[str], b: Boundary, inputs: set) -> list[int]:
    """The flows outside a declared boundary: an outside end on a stock and side it does not have, or
    of a kind it does not have."""
    if not b.declared:
        return []
    return [i for i, (f, e) in enumerate(zip(flows, ends(flows, ids)))
            if any(not b.may(s, side, kind_of(f, inputs)) for s, side in e)]


def claims(flows: list[dict], ids: list[str], b: Boundary, inputs: set) -> list[dict]:
    """With no boundary declared, every outside flow is a boundary claim: [{stock, side, kind, flow}]."""
    if b.declared:
        return []
    out = []
    for f, e in zip(flows, ends(flows, ids)):
        for s, side in e:
            out.append({"stock": s, "side": side, "kind": kind_of(f, inputs),
                        "flow": f"{f.get('from') or 'outside'} -> {f.get('to') or 'outside'}"})
    return out


def _n_claims(flows: list[dict], ids: list[str], b: Boundary) -> int:
    return 0 if b.declared else sum(1 for e in ends(flows, ids) if e)


def _n_hidden(flows: list[dict], ids: list[str]) -> int:
    return len({x for f in flows for x in (f.get("from"), f.get("to")) if x and x not in ids})


def kept_sides(old: list[dict], new: list[dict], ids: list[str], b: Boundary) -> bool:
    """(this copy) Declared: every declared side that has an outside flow in old still has one in new
    (a declared side's flow exists: it may change form, not vanish). Not declared: True."""
    if not b.declared:
        return True
    have = {x for e in ends(new, ids) for x in e}
    return all(x in have for e in ends(old, ids) for x in e if b.may(*x))




def slots(flows: list[dict]) -> list[tuple[dict, str]]:
    """The numbers of a structure, as (dict, key): a proportional rate's k, a linear rate's constant
    (when not 0) and its terms' k, a constant's value, an outside inflow's piece values; per era where a
    rate follows the eras (eras that share a rate dict share the number)."""
    out, seen = [], set()
    for f in flows:
        if f.get("inflow"):
            out += [(g, "value") for g in f["inflow"] if g.get("value") is not None]
            continue
        for r in ([g["rate"] for g in f["byTime"]] if f.get("byTime") else [f.get("rate") or {}]):
            if id(r) in seen:
                continue
            seen.add(id(r))
            form = r.get("form")
            if form == "proportional" and r.get("k") is not None:
                out.append((r, "k"))
            elif form == "constant" and r.get("value") is not None:
                out.append((r, "value"))
            elif form == "gap":
                out += [(r, k) for k in ("goal", "time") if r.get(k) is not None]
            elif form == "linear":
                if r.get("c"):
                    out.append((r, "c"))
                out += [(x, "k") for x in r.get("terms") or [] if x.get("k") is not None]
    return out


def _quantum(a: np.ndarray) -> float:
    a = np.abs(a[np.isfinite(a)])
    for e in range(0, 9):
        q = 10.0 ** -e
        if np.all(np.abs(a / q - np.round(a / q)) < 1e-6):
            return q
    return 0.0


class Data:
    """The one-step changes a structure is fitted to: levels at each step's start (plus announced
    injections), the change over the step, the step's time (its end, as the matrix reads it), the
    measured inputs; only steps where every stock is measured at both ends."""

    def __init__(self, stocks: list[dict], run: dict, opts: dict):
        self.ids = [s["id"] for s in stocks]
        self.inputs = dict(opts.get("inputs") or {})
        t = np.asarray(run["t"], dtype=float)
        A = np.array([np.asarray(run["levels"].get(i, [np.nan] * len(t)), dtype=float) for i in self.ids])
        J = np.zeros((len(self.ids), len(t) - 1))
        for j in opts.get("injections") or []:
            if j.get("stock") in self.ids:
                J[self.ids.index(j["stock"]), np.nonzero(t[1:] == float(j["t"]))[0]] += float(j["amount"])
        L = A[:, :-1] + J
        dy = A[:, 1:] - L
        ok = np.all(np.isfinite(L), axis=0) & np.all(np.isfinite(dy), axis=0)
        self.L, self.dy, self.tt = L[:, ok], dy[:, ok], t[1:][ok]
        self.T = int(ok.sum())
        self.usable = self.T >= MIN_STEPS and len(self.ids) > 0
        self.log_t = math.log(max(self.T, 2))
        self.model_inputs = {"t": list(run["t"]), "values": self.inputs} if self.inputs else None
        q = _quantum(A)
        self.floor = self.T * q * q / 6                     # below this a stock's rss says nothing (rounding)
        self.fits = 0                                       # structures refitted so far (the passes' budget)

    def change(self, flows: list[dict]) -> np.ndarray:
        return Model(flows, self.ids, inputs=self.model_inputs).change(self.L, self.tt)

    def carried(self, f: dict) -> np.ndarray:
        """What one flow carries in each step (what leaves its source, or arrives in its receiver)."""
        if f.get("from") in self.ids:
            return -Model([{**f, "to": None}], self.ids, inputs=self.model_inputs).change(self.L, self.tt)[self.ids.index(f["from"])]
        if f.get("to") in self.ids:
            return Model([f], self.ids, inputs=self.model_inputs).change(self.L, self.tt)[self.ids.index(f["to"])]
        return np.zeros(self.T)


class Fit:
    """A structure with its numbers refitted by (per-stock weighted) least squares on the one-step
    changes; nll is -2 log L with one variance per stock (over every stock), p the count of numbers.

    free: the stocks whose flows' numbers are refitted (None: every flow's); the other numbers stay as
    given. Two structures compared with the same free stocks are compared fairly, and refitting only
    what a change touches keeps a large model's comparisons cheap. Each number's slope comes from the
    one flow that holds it, on the rows of the stocks that flow touches (what one flow's clipping does
    to another's is left out of the slope, not out of the residuals), so the normal equations are
    built stock by stock."""

    def __init__(self, flows: list[dict], data: Data, free: set | None = None):
        self.flows = copy.deepcopy(flows)
        self.p = 0
        self.ok = data.usable and not _n_hidden(flows, data.ids)
        if not self.ok:
            self.nll = float("inf")
            return
        data.fits += 1
        places, owner, seen = [], [], set()
        for fi, f in enumerate(self.flows):
            for d, k in slots([f]):
                if (id(d), k) not in seen:
                    seen.add((id(d), k))
                    places.append((d, k))
                    owner.append(fi)
        self.p = len(places)
        mine = [j for j in range(len(places)) if free is None or {self.flows[owner[j]].get("from"), self.flows[owner[j]].get("to")} & free]

        def put(th):
            for j, v in zip(mine, th):
                d, k = places[j]
                d[k] = float(v)

        def resid(th):
            put(th)
            return data.dy - data.change(self.flows)

        th = np.array([float(places[j][0][places[j][1]]) for j in mine])
        e = resid(th)
        tiny = 1e-9 * max(1.0, float(np.abs(data.dy).max(initial=0)))
        w = 1.0 / np.maximum(e.std(axis=1), tiny)
        obj = lambda r: float(((r * w[:, None]) ** 2).sum())  # noqa: E731
        cur = obj(e)
        for it in range(ITERATIONS if len(th) else 0):
            put(th)
            alone = {fi: data.change([self.flows[fi]]) for fi in {owner[j] for j in mine}}
            per_stock: dict = {}
            for q, j in enumerate(mine):
                d, k = places[j]
                h = 1e-4 * max(1e-3, abs(th[q]))
                d[k] = float(th[q] + h)
                dj = (data.change([self.flows[owner[j]]]) - alone[owner[j]]) / h
                d[k] = float(th[q])
                for srow in np.nonzero(np.any(dj != 0, axis=1))[0]:
                    per_stock.setdefault(int(srow), []).append((q, dj[srow]))
            G = np.zeros((len(th), len(th)))
            g = np.zeros(len(th))
            for srow, cols in per_stock.items():
                ix = [q for q, _ in cols]
                M = np.array([v for _, v in cols])
                G[np.ix_(ix, ix)] += w[srow] ** 2 * (M @ M.T)
                g[ix] += w[srow] ** 2 * (M @ e[srow])
            G[np.diag_indices_from(G)] += 1e-12 * (np.abs(np.diag(G)) + 1)
            try:
                step = np.linalg.solve(G, g)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(G, g, rcond=None)[0]
            for scale in (1.0, 0.5, 0.25):
                tn = th + scale * step
                en = resid(tn)
                if np.all(np.isfinite(en)) and obj(en) < cur:
                    th, e, cur = tn, en, obj(en)
                    break
            else:
                break
            if it == 1:                                     # re-weight once by the stocks' own spreads
                w = 1.0 / np.maximum(e.std(axis=1), tiny)
                cur = obj(e)
        put(th)
        self.resid = e
        rss = np.maximum((e ** 2).sum(axis=1), max(data.floor, 1e-12))
        self.nll = float(data.T * np.log(rss / data.T).sum())
        # a negative rate where only a positive one means anything (a term of a linear rate may be negative)
        self.negative = any(float(d[k]) < -1e-9 for d, k in places if "form" in d and k in ("k", "value", "time"))


def _key(f: dict) -> str:
    r = f.get("rate") or {}
    return f"{f.get('from')}>{f.get('to')}>{r.get('form')}>{sorted(_rate_stocks(f))}>{bool(f.get('inflow'))}>{bool(f.get('byTime'))}"


def touched(old: list[dict], new: list[dict], ids: list[str]) -> set:
    """The measured stocks at the ends of the flows one structure has and the other has not."""
    a, b = [_key(f) for f in old], [_key(f) for f in new]
    out = set()
    for f, k in [(f, k) for f, k in zip(old, a) if a.count(k) != b.count(k)] + [(f, k) for f, k in zip(new, b) if a.count(k) != b.count(k)]:
        out |= {x for x in (f.get("from"), f.get("to")) if x in ids}
    return out


def price(fit: Fit, data: Data, b: Boundary) -> float:
    """BIC with the boundary's prices: log T per number, CLAIM_PRICE log T per boundary claim (no
    boundary declared), HIDDEN_PRICE log T per hidden stock."""
    if not fit.ok or fit.negative:
        return float("inf")
    extra = CLAIM_PRICE * _n_claims(fit.flows, data.ids, b) + HIDDEN_PRICE * _n_hidden(fit.flows, data.ids)
    return fit.nll + (fit.p + extra) * data.log_t


def _set_rate(f: dict, rate: dict) -> None:
    f["rate"] = rate
    f["rates"] = [rate]


def _joined(flows: list[dict]) -> list[dict]:
    """Flows (copied) with two proportional moves between the same stocks on the same level joined (k
    added)."""
    out: list[dict] = []
    for f in copy.deepcopy(flows):
        r = f.get("rate") or {}
        if f.get("from") and f.get("to") and not f.get("byTime") and not f.get("inflow") and r.get("form") == "proportional":
            g = next((g for g in out if g.get("from") == f["from"] and g.get("to") == f["to"] and not g.get("byTime")
                      and (g.get("rate") or {}).get("form") == "proportional" and g["rate"].get("stock") == r.get("stock")
                      and float(g.get("ratio") or 1) == float(f.get("ratio") or 1)), None)
            if g is not None:
                _set_rate(g, {**g["rate"], "k": float(g["rate"]["k"]) + float(r["k"])})
                continue
        out.append(f)
    return out


def _same_unit(stocks: list[dict], a: str, b: str) -> bool:
    unit = {s["id"]: s.get("unit") or "" for s in stocks}
    return unit.get(a) == unit.get(b)


def alternatives(flows: list[dict], i: int, stocks: list[dict], b: Boundary, data: Data) -> list[tuple[str, list[dict]]]:
    """What flow i may become instead of an outside flow, as (what was done, flows): dropped; an
    outflow from A made a move into another stock B of A's unit (with or without B's estimated outside
    inflows, which the move may replace); an inflow into B replaced by the move from a stock A of B's
    unit that has an outside outflow, or by a move from the stock a link reads; an estimated inflow on a
    side declared measured replaced by an inflow reading each measured input. Final stocks never send."""
    f = flows[i]
    final = {s["id"] for s in stocks if s.get("final")}
    rest = [g for j, g in enumerate(flows) if j != i]
    name = f"{f.get('from') or 'outside'} -> {f.get('to') or 'outside'}"
    out = [(f"dropped {name}", rest)]
    A, B = f.get("from"), f.get("to")
    if A in data.ids and B is None and not f.get("inflow"):
        for R in data.ids:
            if R == A or not _same_unit(stocks, A, R):
                continue
            moved = copy.deepcopy(flows)
            moved[i]["to"] = R
            out.append((f"{name} made the move {A} -> {R}", _joined(moved)))
            for j, g in enumerate(flows):
                if g.get("from") is None and g.get("to") == R and kind_of(g, set(data.inputs)) == "estimate":
                    both = [h for k, h in enumerate(moved) if k != j]
                    out.append((f"{name} and outside -> {R} made the move {A} -> {R}", _joined(both)))
        r = f.get("rate") or {}
        terms = ([{"stock": r.get("stock"), "k": r.get("k")}] if r.get("form") == "proportional"
                 else r.get("terms") if r.get("form") == "linear" and not r.get("c") else None)
        if b.declared and not b.may(A, "out", "estimate") and b.may(A, "in", "estimate") and not f.get("byTime") and terms:
            # an outflow from A where only an inflow is declared: A's estimated inflow falls as what the
            # outflow read rises (a gap inflow, c - k X), the inflow's pieces taken at their mean (only
            # then: with no boundary declared, a stock's inflow and outflow are two claims, not a net one)
            for j, g in enumerate(flows):
                if g.get("from") is not None or g.get("to") != A or g.get("byTime") or kind_of(g, set(data.inputs)) != "estimate":
                    continue
                q = g.get("rate") or {}
                if g.get("inflow"):
                    w = [max(1.0, float(p["to"]) - float(p["from"]) + 1) for p in g["inflow"]]
                    c = sum(float(p["value"]) * x for p, x in zip(g["inflow"], w)) / sum(w)
                elif q.get("form") == "constant":
                    c = float(q.get("value") or 0)
                else:
                    continue
                rate = {"form": "linear", "c": c, "terms": [{"stock": x["stock"], "k": -abs(float(x.get("k") or 0))} for x in terms]}
                gap = {"from": None, "to": A, "ratio": 1, "rate": rate, "rates": [rate]}
                out.append((f"{name} and outside -> {A} made one inflow falling as {', '.join(x['stock'] for x in terms)} rises",
                            [h for k, h in enumerate(flows) if k not in (i, j)] + [gap]))
    if B in data.ids and A is None:
        for j, g in enumerate(flows):
            S = g.get("from")
            # (fix, as d5cd56a / ca00790) S == B is no move: B's own drain "moved" into B would only delete it
            if j == i or S == B or S not in data.ids or S in final or g.get("to") is not None or g.get("inflow") or not _same_unit(stocks, S, B):
                continue
            moved = copy.deepcopy(flows)
            moved[j]["to"] = B
            out.append((f"{name} and {S} -> outside made the move {S} -> {B}", _joined([h for k, h in enumerate(moved) if k != i])))
        r = f.get("rate") or {}
        if not f.get("byTime") and not f.get("inflow") and r.get("form") == "linear" and len(r.get("terms") or []) == 1:
            S = r["terms"][0].get("stock")
            if S in data.ids and S != B and S not in final and _same_unit(stocks, S, B):
                g = copy.deepcopy(f)
                g["from"] = S
                _set_rate(g, {"form": "proportional", "stock": S, "k": abs(float(r["terms"][0].get("k") or 0))})
                g.pop("driver", None)
                out.append((f"{name} (a link on {S}) made the move {S} -> {B}", _joined(rest + [g])))
        if b.may(B, "in", "measured") and kind_of(f, set(data.inputs)) == "estimate":
            for name_in, series in data.inputs.items():
                if name_in in data.ids:
                    continue
                v = np.asarray(series, dtype=float)
                scale = float(np.nanmean(np.abs(v))) or 1.0
                per = float(np.mean(data.carried(f))) if data.usable else 0.0
                rate = {"form": "linear", "c": 0, "terms": [{"stock": name_in, "k": max(1e-6, per / scale)}]}
                g = {"from": None, "to": B, "ratio": 1, "rate": rate, "rates": [rate], "input": name_in}
                out.append((f"{name} made an inflow reading {name_in}", rest + [g]))
    return out




def guard(flows: list[dict], stocks: list[dict], b: Boundary, data: Data, log: list[str]) -> list[dict]:
    """No flow outside a declared boundary survives: each, largest first, becomes its best-priced
    alternative (refitted); without enough data to refit, it is dropped."""
    inputs = set(data.inputs)
    for _ in range(3 * len(flows) + 3):
        bad = violations(flows, data.ids, b, inputs)
        if not bad:
            return flows
        if not data.usable or data.fits >= GUARD_FITS:
            why = "no refit" if not data.usable else "no refit: the guard's budget is spent"
            log += [f"dropped {flows[i].get('from') or 'outside'} -> {flows[i].get('to') or 'outside'} ({why})" for i in bad]
            return [f for j, f in enumerate(flows) if j not in bad]
        i = max(bad, key=lambda j: float(np.abs(data.carried(flows[j])).mean()) if not _n_hidden([flows[j]], data.ids) else 0.0)
        best = None
        alts = alternatives(flows, i, stocks, b, data)
        alts = [a for a in alts if kept_sides(flows, a[1], data.ids, b)] or alts[:1]
        free = set().union(*(touched(flows, cand, data.ids) for _, cand in alts))
        for what, cand in alts:
            fit = Fit(cand, data, free)
            # an alternative that keeps more violations than it removes is worse than none
            pr = price(fit, data, b) if fit.ok else float("inf")
            pr += 1e6 * len(violations(fit.flows, data.ids, b, inputs))
            if best is None or pr < best[0]:
                best = (pr, what, fit.flows if fit.ok else cand)
        log.append(best[1])
        flows = best[2]
    return [f for j, f in enumerate(flows) if j not in violations(flows, data.ids, b, inputs)]


def _unexplained(flows: list[dict], data: Data, without: int) -> np.ndarray:
    """Each stock's change less what every flow but one explains (stocks x steps)."""
    return data.dy - data.change([f for j, f in enumerate(flows) if j != without])


def pair_moves(flows: list[dict], stocks: list[dict], b: Boundary, data: Data, log: list[str]) -> list[dict]:
    """An outflow from A and an inflow into B (both outside flows, estimated) whose unexplained changes
    are anti-correlated beyond ANTI_Z are tried as the move A -> B, replacing both, and kept when it
    prices better; repeated while one is kept."""
    if not data.usable or _n_hidden(flows, data.ids):
        return flows
    inputs = set(data.inputs)
    final = {s["id"] for s in stocks if s.get("final")}
    for _ in range(len(flows)):
        base: dict = {}
        best = None
        left: dict = {}

        def unexplained(k: int) -> np.ndarray:
            if k not in left:
                left[k] = _unexplained(flows, data, k)
            return left[k]

        for i, f in enumerate(flows):
            A = f.get("from")
            if A not in data.ids or A in final or f.get("to") is not None or f.get("inflow") or kind_of(f, inputs) != "estimate":
                continue
            eA = unexplained(i)[data.ids.index(A)]
            for j, g in enumerate(flows):
                B = g.get("to")
                if B not in data.ids or B == A or g.get("from") is not None or kind_of(g, inputs) != "estimate" or not _same_unit(stocks, A, B):
                    continue
                eB = unexplained(j)[data.ids.index(B)]
                if eA.std() == 0 or eB.std() == 0:
                    continue
                z = float(np.corrcoef(eA, eB)[0, 1]) * math.sqrt(data.T)
                if z > -ANTI_Z:
                    continue
                if data.fits >= MAX_FITS:
                    break
                moved = copy.deepcopy(flows)
                moved[i]["to"] = B
                cand = _joined([h for k, h in enumerate(moved) if k != j])
                if not kept_sides(flows, cand, data.ids, b):
                    continue
                free = frozenset(touched(flows, cand, data.ids))
                if free not in base:
                    base[free] = price(Fit(flows, data, set(free)), data, b)
                if not math.isfinite(base[free]):
                    continue
                fit = Fit(cand, data, set(free))
                gain = base[free] - (price(fit, data, b) + 1e6 * len(violations(fit.flows, data.ids, b, inputs)))
                if gain > 0 and (best is None or gain > best[0]):
                    best = (gain, f"{A} -> outside and outside -> {B} made the move {A} -> {B} (z {z:.1f})", fit.flows)
        if best is None:
            return flows
        log.append(best[1])
        flows = best[2]
    return flows


def price_claims(flows: list[dict], stocks: list[dict], b: Boundary, data: Data, log: list[str]) -> list[dict]:
    """No boundary declared: each outside flow is a claim at CLAIM_PRICE log T; one at a time, the
    claim whose best alternative (dropped, made a move) lowers the price most is replaced, while one
    does."""
    if b.declared or not data.usable or _n_hidden(flows, data.ids):
        return flows
    for _ in range(len(flows)):
        base: dict = {}
        best = None
        for i, e in enumerate(ends(flows, data.ids)):
            if not e:
                continue
            for what, cand in alternatives(flows, i, stocks, b, data):
                if data.fits >= MAX_FITS:
                    break
                free = frozenset(touched(flows, cand, data.ids))
                if free not in base:
                    base[free] = price(Fit(flows, data, set(free)), data, b)
                if not math.isfinite(base[free]):
                    continue
                fit = Fit(cand, data, set(free))
                gain = base[free] - price(fit, data, b)
                if gain > 0 and (best is None or gain > best[0]):
                    best = (gain, what + " (claim not earned)", fit.flows)
        if best is None:
            return flows
        log.append(best[1])
        flows = best[2]
    return flows


GAP_FITS = 60          # the gap-form pass stops after this many refits of its own (deterministic)


def _line(y: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    """Least-squares c, k of y = c + k x (k 0 when x does not vary)."""
    if float(np.std(x)) == 0:
        return float(np.mean(y)), 0.0
    k, c = np.polyfit(x, y, 1)
    return float(c), float(k)


IV_Z = 3.0             # the affine form's c and k must each stand beyond this z when instrumented


def _instrumented(y: np.ndarray, x: np.ndarray) -> bool:
    """Whether y = c + k x holds with both c and k nonzero beyond IV_Z when x is instrumented by its value one
    step earlier (two-stage least squares). Measurement error on the levels makes the one-step changes
    fall with the measured level (regression to the mean), which least squares reads as a drain with an
    offset; an instrument that does not share the step's error reads only what the dynamics hold."""
    if len(y) < MIN_STEPS + 1 or float(np.std(x)) == 0:
        return False
    W = np.column_stack([np.ones(len(y) - 1), x[1:]])
    Z = np.column_stack([np.ones(len(y) - 1), x[:-1]])
    ZW = Z.T @ W
    try:
        inv = np.linalg.inv(ZW)
    except np.linalg.LinAlgError:
        return False
    beta = inv @ (Z.T @ y[1:])
    u = y[1:] - W @ beta
    cov = float(u @ u) / max(1, len(u) - 2) * (inv @ (Z.T @ Z) @ inv.T)
    se = np.sqrt(np.maximum(np.diag(cov), 1e-300))
    return bool(np.all(np.abs(beta) > IV_Z * se))


def _simple(f: dict, ids: list[str]) -> bool:
    """A rate the affine form generalises: a constant, pieces, k X, or a linear rate of at most one level
    (with or without effects on it; no eras)."""
    if f.get("byTime"):
        return False
    if f.get("inflow"):
        return True
    r = f.get("rate") or {}
    terms = r.get("terms") or []
    return (r.get("form") == "constant" or (r.get("form") == "proportional" and r.get("stock") in ids)
            or (r.get("form") == "linear" and len(terms) <= 1 and all(x.get("stock") in ids for x in terms)))


def gap_forms(flows: list[dict], stocks: list[dict], b: Boundary, data: Data, log: list[str]) -> list[dict]:
    """On a declared estimate side, the outside flow may have the affine form c + k X (a goal gap, c - k
    X, or a drain with an offset, k X - c; X any measured stock). Per declared estimate side, largest
    first, the side's estimated outside flows of simple forms (a constant, pieces, k X, a one-level
    linear rate) are tried as one flow c + k X for each X whose c and k both hold when X is instrumented
    (`_instrumented`), fitted jointly with every flow of the stock (started from the line through what
    they carried), and the best replaces them when it prices better
    (docs/research/declared-boundary.md, cause 3: 104's 0.1 T - 7, 011's (300 - I) / 5)."""
    if not b.declared or not data.usable or _n_hidden(flows, data.ids):
        return flows
    inputs = set(data.inputs)
    done = data.fits
    sides = []
    for f in flows:
        A, B = f.get("from"), f.get("to")
        if (A is None) == (B is None) or kind_of(f, inputs) != "estimate" or not _simple(f, data.ids):
            continue
        s, side = (B, "in") if A is None else (A, "out")
        if s in data.ids and b.may(s, side, "estimate") and (s, side) not in sides:
            sides.append((s, side))
    size = {k: 0.0 for k in sides}
    for f in flows:
        k = (f.get("to"), "in") if f.get("from") is None else (f.get("from"), "out")
        if k in size:
            size[k] += float(np.abs(data.carried(f)).mean())
    for s, side in sorted(sides, key=lambda k: -size[k]):
        group = [j for j, f in enumerate(flows)
                 if ((f.get("from") is None and f.get("to") == s) if side == "in" else (f.get("from") == s and f.get("to") is None))
                 and kind_of(f, inputs) == "estimate" and _simple(f, data.ids)]
        if not group or (len(group) == 1 and not flows[group[0]].get("inflow")
                         and (flows[group[0]].get("rate") or {}).get("form") == "linear"
                         and (flows[group[0]].get("rate") or {}).get("c") and len(flows[group[0]]["rate"].get("terms") or []) == 1):
            continue                                       # nothing to do, or already c + k X (its numbers kept)
        sign = 1.0 if side == "in" else -1.0
        rest = [f for j, f in enumerate(flows) if j not in group]
        y = sign * (data.dy - data.change(rest))[data.ids.index(s)]          # what the side's flows carry
        free = {s}
        base = price(Fit(flows, data, free), data, b)
        if not math.isfinite(base):
            continue
        best = None
        for X in data.ids:
            if data.fits - done >= GAP_FITS:
                break
            c, k = _line(y, data.L[data.ids.index(X)])
            if k == 0 or c == 0 or not _instrumented(y, data.L[data.ids.index(X)]):
                continue
            g = {key: flows[group[0]][key] for key in ("from", "to", "name") if key in flows[group[0]]}
            g["ratio"] = 1                                 # a new rate: no effects, pieces or driver carried over
            _set_rate(g, {"form": "linear", "c": c, "terms": [{"stock": X, "k": k}]})
            cand = rest + [g]
            fit = Fit(cand, data, free)
            pr = price(fit, data, b) + 1e6 * len(violations(fit.flows, data.ids, b, inputs))
            if pr < base and (best is None or pr < best[0]):
                best = (pr, X, fit.flows)
        if best is not None:
            r = best[2][-1]["rate"]
            log.append(f"{len(group)} outside {'inflow' if side == 'in' else 'outflow'}(s) of {s} made one flow "
                       f"c + k {best[1]} ({r['c']:.4g} + {r['terms'][0]['k']:.4g} {best[1]})")
            flows = best[2]
    return flows


def _missing(flows: list[dict], ids: list[str], b: Boundary, inputs: set) -> list[tuple[str, str]]:
    """The declared sides (stock, "in"/"out") no flow of the model holds (a flow outside the boundary
    does not count)."""
    bad = set(violations(flows, ids, b, inputs))
    held = {e for i, es in enumerate(ends(flows, ids)) if i not in bad for e in es}
    return [(s, side) for s, e in b.sides.items() if s in ids for side in ("in", "out") if e.get(side) and (s, side) not in held]


def fill_declared(flows: list[dict], stocks: list[dict], b: Boundary, data: Data, log: list[str]) -> list[dict]:
    """A declared outside flow is a given of the case (the user, 2026-10-21): the strategy decides its
    form and numbers, never whether it exists. Each declared side the model lacks gets a flow in the
    best-pricing allowed form, its numbers started from what the stock leaves unexplained and refitted
    jointly with the stock's other flows: an estimated inflow a constant, an estimated outflow a drain k X
    (gap_forms may then give either the form c + k X), a measured inflow one reading a measured input.
    The declared flows alone, with fitted numbers, are so the floor of every model."""
    if not b.declared:
        return flows
    inputs = set(data.inputs)
    for s, side in _missing(flows, data.ids, b, inputs):
        row = data.ids.index(s)
        sign = 1.0 if side == "in" else -1.0
        y = sign * (data.dy[row] - data.change(flows)[row]) if data.T else np.zeros(0)
        x = data.L[row] if data.T else np.zeros(0)
        cands = []
        if "estimate" in b.kinds(s, side):
            if side == "in":
                c = max(float(np.mean(y)) if len(y) else 0.0, 1e-6)
                rate = {"form": "constant", "value": c}
                cands.append(("a constant", {"from": None, "to": s, "ratio": 1, "rate": rate, "rates": [rate]}))
            else:
                k = max(float(y @ x / (x @ x)) if len(y) and float(x @ x) > 0 else 0.0, 1e-6)
                rate = {"form": "proportional", "stock": s, "k": k}
                cands.append((f"a drain k {s}", {"from": s, "to": None, "ratio": 1, "rate": rate, "rates": [rate]}))
        if "measured" in b.kinds(s, side) or not cands:
            for name, series in data.inputs.items():
                if name in data.ids:
                    continue
                v = np.asarray(series, dtype=float)
                scale = float(np.nanmean(np.abs(v))) or 1.0
                k = max(float(np.mean(y)) / scale if len(y) else 0.0, 1e-6)
                rate = {"form": "linear", "c": 0, "terms": [{"stock": name, "k": k}]}
                f = {"from": None, "to": s, "ratio": 1, "rate": rate, "rates": [rate], "input": name} if side == "in" else \
                    {"from": s, "to": None, "ratio": 1, "rate": rate, "rates": [rate], "input": name}
                cands.append((f"a flow reading {name}", f))
        best = None
        for what, f in cands:
            cand = flows + [f]
            if violations(cand, data.ids, b, inputs) != violations(flows, data.ids, b, inputs):
                continue                                   # not a form this side allows (this copy's kind reading)
            fit = Fit(cand, data, {s})
            pr = price(fit, data, b) if fit.ok else float("inf")
            got = fit.flows if fit.ok and math.isfinite(pr) else cand       # unfittable or negative: its started numbers
            if best is None or pr < best[0]:
                best = (pr, what, got)
        if best is None:
            continue
        log.append(f"the declared {'inflow into' if side == 'in' else 'outflow from'} {s} was missing: added as {best[1]}")
        flows = best[2]
    return flows


def apply(result: dict, stocks: list[dict], run: dict, opts: dict | None, final: bool = True) -> dict:
    """The hook: a result's flows kept inside the boundary. Declared: the guard, then the moves the
    unexplained changes ask for. Not declared (final only): moves, then each claim priced; the claims
    left are listed in result["boundary_claims"]. What was done is under result["boundary"]."""
    if not isinstance(result, dict) or result.get("flows") is None:
        return result
    b = Boundary(run)
    opts = opts or {}
    data = Data(stocks, run, opts)
    log: list[str] = []
    flows = result["flows"]
    if b.declared:
        flows = guard(flows, stocks, b, data, log)
        flows = fill_declared(flows, stocks, b, data, log)
        if final:
            flows = pair_moves(flows, stocks, b, data, log)
            flows = fill_declared(flows, stocks, b, data, log)   # the moves may have taken a declared side's flow
            flows = gap_forms(flows, stocks, b, data, log)
    elif final:
        flows = pair_moves(flows, stocks, b, data, log)
        flows = price_claims(flows, stocks, b, data, log)
    out = {**result, "flows": flows}
    if final:
        out["boundary"] = {**b.summary(), "changed": log}
        out["boundary_claims"] = claims(flows, data.ids, b, set(data.inputs))
    elif log:
        out["boundary_guard"] = log
    return out


# (2026-10-21, the user) The kind "input": a declared inflow that IS a measured input series, exactly
# 1 x it (no constant, never rescaled or supplemented); and on every declared side at most as many outside
# flows as the side declares. The search reads an "input" side as a "measured" one (Boundary.__init__);
# every call of `apply` then ends with `enforce_sides`, which sets the input's flow at exactly 1 x the
# input and refits the stock's other flows around it (its contribution taken off the stock's change).


def _as_measured(raw: dict) -> dict:
    """The announced boundary with each "input" kind read as "measured" (what the search sees)."""
    return {s: {side: ["measured" if k == "input" else k for k in (e.get(side) or [])] for side in ("in", "out")}
            for s, e in (raw or {}).items()}


def _input_flow(stock: str, name: str) -> dict:
    rate = {"form": "proportional", "stock": name, "k": 1.0}
    return {"from": None, "to": stock, "ratio": 1, "rate": rate, "rates": [rate], "input": name}


def _summable(f: dict) -> bool:
    """A rate that adds to another as numbers do: a constant, k X or a linear rate, nothing else on it."""
    r = f.get("rate") or {}
    return (not f.get("byTime") and not f.get("inflow") and not f.get("effects") and not f.get("uses") and not f.get("via")
            and float(f.get("ratio") or 1) == 1.0 and r.get("form") in ("constant", "proportional", "linear"))


def _sum_rates(group: list[dict]) -> dict:
    """Same-side flows of summable rates as one linear flow (constants added, terms on one level added)."""
    c, terms = 0.0, {}
    for f in group:
        r = f["rate"]
        if r["form"] == "constant":
            c += float(r.get("value") or 0)
        elif r["form"] == "proportional":
            terms[r["stock"]] = terms.get(r["stock"], 0.0) + float(r.get("k") or 0)
        else:
            c += float(r.get("c") or 0)
            for x in r.get("terms") or []:
                terms[x["stock"]] = terms.get(x["stock"], 0.0) + float(x.get("k") or 0)
    rate = {"form": "linear", "c": c, "terms": [{"stock": s, "k": k} for s, k in terms.items()]}
    g = {k: v for k, v in group[0].items() if k in ("from", "to", "name", "id")}
    return {**g, "ratio": 1, "rate": rate, "rates": [rate]}


def _without(f: dict, name: str) -> dict | None:
    """f with its terms reading the input name removed (None when nothing is left of it)."""
    r = f.get("rate") or {}
    if f.get("byTime") or f.get("inflow"):
        return f
    if r.get("form") == "proportional" and r.get("stock") == name:
        return None
    if r.get("form") == "linear" and any(x.get("stock") == name for x in r.get("terms") or []):
        rate = {**r, "terms": [x for x in r["terms"] if x.get("stock") != name]}
        if not rate["terms"] and not rate.get("c"):
            return None
        return {**f, "rate": rate, "rates": [rate]}
    return f


def _around(flows: list[dict], fixed: list[dict], data: Data, free: set) -> Fit | None:
    """flows refitted (the stocks in free) on the stocks' changes less what the fixed flows carry."""
    if not data.usable or not free or _n_hidden(flows, data.ids):
        return None
    rest = copy.copy(data)
    rest.dy = data.dy - data.change(fixed) if fixed else data.dy
    fit = Fit(flows, rest, set(free))
    data.fits = rest.fits
    return fit if fit.ok else None


def enforce_sides(flows: list[dict], stocks: list[dict], b: Boundary, data: Data, log: list[str]) -> list[dict]:
    """Declared only. Each side declared "input" gets exactly 1 x one measured input (the one whose fixed
    flow leaves the best-pricing refit of the rest), and loses every other flow reading an input (on a side
    that declares only inputs, every other flow); the input's terms come off the side's other flows. Then
    every declared side keeps at most as many outside flows as it declares: the excess of summable rates
    is merged into one, the rest dropped smallest first. The touched stocks' other flows are refitted
    around the fixed inflows, which are never refitted or rescaled."""
    if not b.declared or not (b.fixed or b.limit):
        return flows
    inputs = set(data.inputs)
    names = [n for n in data.inputs if n not in data.ids]
    flows = copy.deepcopy(flows)
    touched: set = set()
    fixed: list[dict] = []
    for s, n in sorted(b.fixed.items()):
        if s not in data.ids or not names:
            continue
        only = b.limit.get((s, "in"), 0) == n
        own = "measured" in b.original.get(s, {}).get("in", [])
        e = ends(flows, data.ids)
        hint = [x for f in flows if f.get("from") is None and f.get("to") == s for x in sorted(_rate_stocks(f)) if x in names]
        gone = [i for i, f in enumerate(flows) if (only and (s, "in") in e[i])
                or (not own and f.get("from") is None and f.get("to") == s and kind_of(f, inputs) == "measured")]
        if gone:
            log.append(f"{len(gone)} outside inflow(s) of {s} set apart for its declared input")
        flows = [f for i, f in enumerate(flows) if i not in gone]
        touched.add(s)
        for _ in range(n):
            best = None
            for name in [x for x in names if not any(g["to"] == s and g["rate"]["stock"] == x for g in fixed)]:
                f = _input_flow(s, name)
                cand = [g for g in (_without(h, name) if h.get("from") is None and h.get("to") == s else h for h in flows) if g is not None]
                fit = _around(cand, fixed + [f], data, {s})
                pr = price(fit, data, b) if fit is not None else float("inf")
                key = (pr, hint.index(name) if name in hint else len(hint), names.index(name))
                if best is None or key < best[0]:
                    best = (key, f, cand)
            if best is None:
                break
            fixed.append(best[1])
            flows = best[2]
            log.append(f"the declared input inflow of {s} fixed at exactly 1 x {best[1]['rate']['stock']}")
    for (s, side), n in sorted(b.limit.items()):
        if s not in data.ids:
            continue
        allowed = n - (sum(1 for g in fixed if g["to"] == s) if side == "in" else 0)
        on = [i for i, es in enumerate(ends(flows, data.ids)) if (s, side) in es]
        if len(on) <= allowed:
            continue
        direct = [i for i in on if _summable(flows[i]) and (flows[i].get("to") if side == "in" else flows[i].get("from")) == s]
        if len(direct) >= 2 and allowed >= 1:
            one = _sum_rates([flows[i] for i in direct])
            flows = [f for i, f in enumerate(flows) if i not in direct[1:]]
            flows[direct[0]] = one
            log.append(f"{len(direct)} outside {'inflows to' if side == 'in' else 'outflows from'} {s} merged into one")
            on = [i for i, es in enumerate(ends(flows, data.ids)) if (s, side) in es]
        while len(on) > max(allowed, 0):
            i = min(on, key=lambda j: float(np.abs(data.carried(flows[j])).mean()) if not _n_hidden([flows[j]], data.ids) else 0.0)
            log.append(f"dropped {flows[i].get('from') or 'outside'} -> {flows[i].get('to') or 'outside'} "
                       f"(one outside flow per declared {'inflow' if side == 'in' else 'outflow'} of {s})")
            flows = [f for j, f in enumerate(flows) if j != i]
            on = [j for j, es in enumerate(ends(flows, data.ids)) if (s, side) in es]
        touched.add(s)
    if touched:
        fit = _around(flows, fixed, data, touched)
        if fit is not None and not fit.negative:
            flows = fit.flows
    return flows + fixed


_apply_sides = apply


def apply(result: dict, stocks: list[dict], run: dict, opts: dict | None, final: bool = True, **kw) -> dict:
    """The hook (as before), then enforce_sides: an "input" side's flow exactly 1 x its input, at most one
    outside flow per declared flow; what it did is added to result["boundary"]["changed"] (final) or
    result["boundary_guard"]."""
    out = _apply_sides(result, stocks, run, opts, final, **kw)
    if not isinstance(out, dict) or out.get("flows") is None:
        return out
    b = Boundary(run)
    if not b.declared or not (b.fixed or b.limit):
        return out
    data = kw.get("data") or Data(stocks, run, opts or {})
    log: list[str] = []
    out = {**out, "flows": enforce_sides(out["flows"], stocks, b, data, log)}
    if log and final:
        out["boundary"] = {**(out.get("boundary") or {}), "changed": list((out.get("boundary") or {}).get("changed") or []) + log}
    elif log:
        out["boundary_guard"] = list(out.get("boundary_guard") or []) + log
    return out
