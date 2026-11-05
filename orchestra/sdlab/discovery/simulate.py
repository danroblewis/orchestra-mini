"""Running a model forward from many moments at once, and judging it on the data.

Flows are SD Lab's model flows (dicts): a rate in one of the forms constant, proportional, gap or
linear (every rate is floored at 0, as SD Lab's own equations are); a rate that follows announced
eras (`byTime`: segments {from, to, rate}); or a found outside inflow that changes (`inflow`:
segments {from, to, value}); a flow's rate may be multiplied by effects (curves of other stocks'
levels, as SD Lab's effect_value); and a flow may use up another stock as it moves (`uses`, a co-flow:
`per` of it for each unit moved). A stock's outflows and the flows using it share what it holds, as in
the ground-truth engine (nothing leaves a stock that is not there). A segment holds for times from..to; a time no segment covers takes the
last segment. Levels are floored at 0 after every step.

Mirroring the engine's additive features (2026-10-18; each off by default, so a model without them
steps exactly as before):
  inputs    measured inputs, {"t": [...], "values": {name: [value per step]}}: a rate or effect reads
            an input by name as it reads a stock (the value of the step at or before the time)
  dt        sub-steps: each step is 1/dt sub-steps moving rate x dt (the same times, inputs and
            segments over the step)
  ceilings  {stock: max}: a stock's inflows share the room left under its ceiling, as its outflows
            share what it holds
"""

from __future__ import annotations

import numpy as np


class _Rate:
    """One rate, or one rate per segment (all of one form over the same stocks), evaluated for many
    states at once."""

    def __init__(self, rates: list[dict], index: dict[str, int]):
        self.rates = rates
        self.index = index
        forms = {r.get("form") for r in rates}
        self.form = forms.pop() if len(forms) == 1 else None
        stocks = {self._stocks(r) for r in rates}
        if self.form is None or len(stocks) != 1:
            self.form = "mixed"
            return
        self.stocks = [index.get(s) for s in stocks.pop()]
        num = lambda key: np.array([float(r.get(key) or 0) for r in rates])  # noqa: E731
        if self.form == "constant":
            self.value = num("value")
        elif self.form == "proportional":
            self.k = num("k")
        elif self.form == "gap":
            self.goal, self.time = num("goal"), np.maximum(1e-9, np.array([float(r.get("time") or 1) for r in rates]))
        elif self.form == "linear":
            self.c = num("c")
            self.ks = [np.array([float(r["terms"][i].get("k") or 0) for r in rates]) for i in range(len(self.stocks))]

    @staticmethod
    def _stocks(r: dict) -> tuple:
        if r.get("form") in ("proportional", "gap"):
            return (r.get("stock"),)
        if r.get("form") == "linear":
            return tuple(x.get("stock") for x in r.get("terms") or [])
        return ()

    def _level(self, L: np.ndarray, i: int | None) -> np.ndarray:
        return L[i] if i is not None else np.zeros(L.shape[1])

    def __call__(self, L: np.ndarray, seg: np.ndarray) -> np.ndarray:
        if self.form == "mixed":
            out = np.zeros(L.shape[1])
            for g in np.unique(seg):
                m = seg == g
                out[m] = _Rate([self.rates[g]], self.index)(L[:, m], np.zeros(int(m.sum()), int))
            return out
        if self.form == "constant":
            v = self.value[seg]
        elif self.form == "proportional":
            v = self.k[seg] * self._level(L, self.stocks[0])
        elif self.form == "gap":
            v = (self.goal[seg] - self._level(L, self.stocks[0])) / self.time[seg]
        elif self.form == "linear":
            v = self.c[seg] + sum(k[seg] * self._level(L, i) for k, i in zip(self.ks, self.stocks))
        else:
            v = np.zeros(L.shape[1])
        return np.maximum(0.0, v)


class _Effects:
    """A flow's effects: its rate multiplied by a curve of each named stock's level (SD Lab's
    effect_value: "linear" is level / normal clamped to [min, max]; "table" interpolates between
    points and holds the end values beyond them)."""

    def __init__(self, effects: list[dict], index: dict[str, int]):
        self.parts = []
        for e in effects:
            i = index.get(e.get("stock"))
            if e.get("shape") == "table" and e.get("points"):
                pts = sorted((float(a), float(b)) for a, b in e["points"])
                self.parts.append(("table", i, np.array([a for a, _ in pts]), np.array([b for _, b in pts])))
            else:
                lo = float(e["min"]) if e.get("min") is not None else 0.0
                hi = float(e["max"]) if e.get("max") is not None else 1.0
                self.parts.append(("linear", i, max(1e-9, float(e.get("normal") or 1)), (lo, hi)))

    def __call__(self, L: np.ndarray) -> np.ndarray:
        m = np.ones(L.shape[1])
        for kind, i, a, b in self.parts:
            x = L[i] if i is not None else np.zeros(L.shape[1])
            m = m * (np.maximum(0.0, np.interp(x, a, b)) if kind == "table" else np.minimum(b[1], np.maximum(b[0], x / a)))
        return m


class _Segments:
    """Which segment holds at each time: the first that covers it, else the last."""

    def __init__(self, segs: list[dict]):
        order = sorted(range(len(segs)), key=lambda i: segs[i]["from"])
        self.order = np.array(order)
        self.froms = np.array([segs[i]["from"] for i in order], dtype=float)
        self.tos = np.array([segs[i]["to"] for i in order], dtype=float)
        self.last = len(segs) - 1

    def __call__(self, t: np.ndarray) -> np.ndarray:
        pos = np.searchsorted(self.froms, t, side="right") - 1
        ok = (pos >= 0) & (t <= self.tos[np.clip(pos, 0, None)])
        return np.where(ok, self.order[np.clip(pos, 0, None)], self.last)


class Model:
    """A list of flows, compiled for stepping the stocks `ids` (any other stock reads as 0); with
    inputs, dt and ceilings as the module says."""

    def __init__(self, flows: list[dict], ids: list[str], inputs: dict | None = None, dt: float = 1.0, ceilings: dict | None = None):
        self.ids = ids
        index = {s: i for i, s in enumerate(ids)}
        self.n_sub = max(1, round(1 / float(dt))) if dt and float(dt) < 1 else 1
        self.dt = 1.0 / self.n_sub
        self.inputs = None
        if inputs and inputs.get("values"):
            names = [k for k in inputs["values"] if k not in index]
            self.input_t = np.asarray(inputs["t"], dtype=float)
            self.inputs = np.nan_to_num(np.array([np.asarray(inputs["values"][k], dtype=float) for k in names])) if names else None
            for j, k in enumerate(names):
                index[k] = len(ids) + j
        self.ceilings = [(index[s], float(c)) for s, c in (ceilings or {}).items() if s in index and index[s] < len(ids) and c is not None]
        self.into = {c: [i for i, f in enumerate(flows) if index.get(f.get("to")) == c] for c, _ in self.ceilings}
        self.parts = []
        # co-flows: (flow, used stock, per)
        self.uses = [(i, index[u["stock"]], float(u["per"])) for i, f in enumerate(flows) for u in f.get("uses") or []
                     if u.get("stock") in index and float(u.get("per") or 0) > 0]
        # nothing leaves a stock that is not there: a stock's outflows (and the flows that use it up)
        # share what it holds, stock by stock in order, as the ground-truth engine does
        takers = {c: [(i, float(f.get("ratio") or 1)) for i, f in enumerate(flows) if index.get(f.get("from")) == c]
                  + [(i, per) for i, cc, per in self.uses if cc == c] for c in range(len(ids))}
        self.takers = {c: t for c, t in takers.items() if t}
        self.effects = [(i, _Effects(f["effects"], index)) for i, f in enumerate(flows) if f.get("effects")]
        for f in flows:
            src, dst = index.get(f.get("from")), index.get(f.get("to"))
            ratio = float(f.get("ratio") or 1)
            if f.get("byTime"):
                segs = f["byTime"]
                self.parts.append((src, dst, ratio, _Segments(segs), _Rate([g["rate"] for g in segs], index), None))
            elif f.get("inflow"):
                segs = f["inflow"]
                vals = np.maximum(0.0, np.array([float(g["value"]) for g in segs]))
                self.parts.append((src, dst, ratio, _Segments(segs), None, vals))
            else:
                self.parts.append((src, dst, ratio, None, _Rate([f.get("rate") or {}], index), None))

    def _read(self, L: np.ndarray, t: np.ndarray) -> np.ndarray:
        """The levels with the inputs' values at times t below them (rows in index order)."""
        if self.inputs is None:
            return L
        i = np.clip(np.searchsorted(self.input_t, np.asarray(t, dtype=float), side="right") - 1, 0, self.inputs.shape[1] - 1)
        return np.vstack([L, self.inputs[:, i]])

    def _amounts(self, L: np.ndarray, t: np.ndarray) -> list:
        """Each flow's amount in one step from levels L (stocks x states) at times t (one per state)."""
        if self.inputs is not None or self.n_sub > 1 or self.ceilings:
            return self._amounts_added(L, t)
        m = L.shape[1]
        zero = np.zeros(m, int)
        amounts = []
        for src, dst, ratio, segs, rate, vals in self.parts:
            seg = segs(t) if segs is not None else zero
            amounts.append(vals[seg] if vals is not None else rate(L, seg))
        for i, effect in self.effects:                       # effects: the rate times curves of other levels
            amounts[i] = amounts[i] * effect(L)
        for c, takers in self.takers.items():                 # a flow cannot take more than there is
            want = sum(amounts[i] * w for i, w in takers)
            k = np.where((want > L[c]) & (L[c] > -1), np.maximum(0.0, L[c]) / np.where(want > 0, want, 1), 1.0)
            for i, _ in takers:
                amounts[i] = amounts[i] * k
        return amounts

    def _amounts_added(self, L: np.ndarray, t: np.ndarray) -> list:
        """_amounts with inputs, a sub-step (amounts x dt) and ceilings."""
        S = L.shape[0]
        R = self._read(L, t)
        m = L.shape[1]
        zero = np.zeros(m, int)
        amounts = []
        for src, dst, ratio, segs, rate, vals in self.parts:
            seg = segs(t) if segs is not None else zero
            amounts.append(vals[seg] if vals is not None else rate(R, seg))
        for i, effect in self.effects:
            amounts[i] = amounts[i] * effect(R)
        if self.n_sub > 1:
            amounts = [a * self.dt for a in amounts]
        for c, takers in self.takers.items():
            want = sum(amounts[i] * w for i, w in takers)
            k = np.where((want > L[c]) & (L[c] > -1), np.maximum(0.0, L[c]) / np.where(want > 0, want, 1), 1.0)
            for i, _ in takers:
                amounts[i] = amounts[i] * k
        for c, cap in self.ceilings:                          # nothing enters a full stock
            ins = self.into.get(c) or []
            if not ins or c >= S:
                continue
            want = sum(amounts[i] for i in ins)
            room = np.maximum(0.0, cap - L[c])
            k = np.where(want > room, room / np.where(want > 0, want, 1), 1.0)
            for i in ins:
                amounts[i] = amounts[i] * k
        return amounts

    def _substeps(self, L: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Levels after one step of 1/dt sub-steps (floored at 0 after each)."""
        for _ in range(self.n_sub):
            amounts = self._amounts(L, t)
            nxt = L.copy()
            for (src, dst, ratio, *_), a in zip(self.parts, amounts):
                if src is not None:
                    nxt[src] -= a * ratio
                if dst is not None:
                    nxt[dst] += a
            for i, c, per in self.uses:
                nxt[c] -= amounts[i] * per
            L = np.where(nxt >= 0, nxt, 0.0)
        return L

    def change(self, L: np.ndarray, t: np.ndarray) -> np.ndarray:
        """What the flows add to each stock in one step from levels L (stocks x states) at times t
        (one per state), before levels are floored at 0 (with sub-steps: the change over the step's
        sub-steps, each floored at 0)."""
        if self.n_sub > 1:
            return self._substeps(L, t) - L
        amounts = self._amounts(L, t)
        d = np.zeros_like(L, dtype=float)
        for (src, dst, ratio, *_), a in zip(self.parts, amounts):
            if src is not None:
                d[src] -= a * ratio
            if dst is not None:
                d[dst] += a
        for i, c, per in self.uses:
            d[c] -= amounts[i] * per
        return d

    def step(self, L: np.ndarray, t: np.ndarray) -> np.ndarray:
        """One step from levels L (stocks x states) at times t (one per state)."""
        if self.n_sub > 1:
            return self._substeps(L, t)
        amounts = self._amounts(L, t)
        nxt = L.copy()
        for (src, dst, ratio, *_), a in zip(self.parts, amounts):
            if src is not None:
                nxt[src] -= a * ratio
            if dst is not None:
                nxt[dst] += a
        for i, c, per in self.uses:
            nxt[c] -= amounts[i] * per
        return np.where(nxt >= 0, nxt, 0.0)

    def run(self, L: np.ndarray, times: np.ndarray, injections: list[dict] | None = None) -> np.ndarray:
        """Levels after len(times) steps; times is steps x states. Injections ({t, stock, amount}:
        announced interventions) are added at the start of their step, before its flows, as the
        ground-truth engine adds them (a stock never goes below 0)."""
        index = {s: i for i, s in enumerate(self.ids)}
        pulses = [(float(j["t"]), index[j["stock"]], float(j["amount"])) for j in injections or [] if j.get("stock") in index]
        for t in times:
            if pulses:
                L = L.copy()
                for tj, i, amount in pulses:
                    hit = t == tj
                    if hit.any():
                        L[i, hit] = np.maximum(0.0, L[i, hit] + amount)
                        for c, cap in self.ceilings:
                            if c == i:
                                L[i, hit] = np.minimum(cap, L[i, hit])
            L = self.step(L, t)
        return L


def starts_for(n: int, horizon: int, count: int) -> np.ndarray:
    every = max(1, (n - horizon - 1) // count)
    return np.arange(1, n - horizon, every)


def levels_array(levels: dict, ids: list[str]) -> np.ndarray:
    return np.nan_to_num(np.array([np.asarray(levels.get(i, []), dtype=float) for i in ids]))


def behaviour_error(flows: list[dict], ids: list[str], t: list, levels: dict, horizon: int = 20, starts: int = 80,
                    injections: list[dict] | None = None, inferred: dict | None = None, inputs: dict | None = None) -> float:
    """How far the model's runs from ~`starts` moments in the data end up from what was measured
    `horizon` steps later (its own estimated outside inflows feed it), relative to how far the data
    moved: 0 = perfect, 1 = no better than assuming nothing changes.

    inferred (optional): the model's own unmeasured stocks, {id: estimated level aligned with t};
    each run starts them from that estimate, and only the measured stocks are compared.

    inputs (optional): measured inputs, {name: value per step aligned with t}, that the flows may read."""
    inferred = inferred or {}
    S = len(ids)
    A = levels_array({**levels, **inferred}, ids + list(inferred))
    n = A.shape[1]
    s0 = starts_for(n, horizon, starts)
    if not len(s0):
        return float("inf")
    sd = A[:S].std(axis=1)
    sd[sd == 0] = 1
    tt = np.asarray(t, dtype=float)
    end = Model(flows, ids + list(inferred), inputs={"t": t, "values": inputs} if inputs else None).run(A[:, s0 - 1], tt[s0[None, :] + np.arange(horizon)[:, None]], injections)[:S]
    obs = A[:S, s0 + horizon - 1]
    e = ((end - obs) / sd[:, None]) ** 2
    e[~np.isfinite(e)] = 1e6
    moved = (((obs - A[:S, s0 - 1]) / sd[:, None]) ** 2).sum()
    return float(e.sum() / max(1e-9, moved))
