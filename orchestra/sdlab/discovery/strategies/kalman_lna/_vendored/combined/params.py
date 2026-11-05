# Vendored for kalman_lna from sdlab/discovery/strategies/combined/params.py at commit ec2f9e8 (imports rewritten to the
# vendored copies; otherwise unchanged). Do not edit: re-vendor instead.
"""What a candidate had to write down: the numbers the judge prices.

Close to the ladder's complexity count (flows + numbers), but counting what a fit actually chose:

- 1 per flow (that it is there at all), plus its rate's numbers (constant 1, proportional 1, gap 2,
  linear 1 per stock read and 1 for a constant);
- announced eras (`byTime`): every segment after the first costs its rate's numbers (the places are
  given by the data);
- found outside-inflow pieces (`inflow`): one number per distinct value and one per change place;
  a profile that repeats (the same values at the same spacing) pays its places once per period,
  plus one for the period;
- effects (linear 3, table 2 per point), 1 per co-flow use, 1 for a ratio other than 1, and 1 per
  unmeasured stock (its starting level), a `via` delay 1 more.
"""

from __future__ import annotations


def rate_numbers(rate: dict | None) -> int:
    r = rate or {}
    form = r.get("form")
    if form in ("constant", "proportional"):
        return 1
    if form == "gap":
        return 2
    if form == "linear":
        ks: dict = {}
        for x in r.get("terms") or []:
            ks[x.get("stock")] = ks.get(x.get("stock"), 0.0) + float(x.get("k") or 0)
        return int(bool(float(r.get("c") or 0))) + sum(1 for k in ks.values() if k)
    if form == "available":
        return 0
    return 1


def _period(vals: list, froms: list) -> int | None:
    """The number of pieces in one period when the pieces repeat (values and spacing), else None."""
    n = len(vals)
    for m in range(1, n // 2 + 1):
        if all(vals[i] == vals[i + m] for i in range(n - m)):
            gaps = {froms[i + m] - froms[i] for i in range(1, n - m)}     # the first piece starts with the data
            if len(gaps) <= 1:
                return m
    return None


def inflow_numbers(segs: list[dict]) -> int:
    vals, froms = [], []
    for g in sorted(segs, key=lambda g: g["from"]):
        v = round(float(g.get("value") or 0), 9)
        if vals and v == vals[-1]:
            continue
        vals.append(v)
        froms.append(float(g["from"]))
    if len(vals) <= 1:
        return len(vals)
    m = _period(vals, froms)
    distinct = len(set(vals))
    if m is not None and m < len(vals) - 1:
        return distinct + m + 1
    return distinct + len(vals) - 1


def numbers(flows: list[dict], measured: set) -> int:
    k = 0
    hidden = set()
    for f in flows or []:
        k += 1
        if f.get("inflow"):
            k += inflow_numbers(f["inflow"])
        elif f.get("byTime"):
            rates, last = [], None
            for g in f["byTime"]:
                if g.get("rate") != last:
                    rates.append(g.get("rate"))
                    last = g.get("rate")
            k += sum(rate_numbers(r) for r in rates)
        else:
            k += rate_numbers(f.get("rate"))
        for e in f.get("effects") or []:
            k += 3 if e.get("shape") != "table" else 2 * len(e.get("points") or [])
        k += len(f.get("uses") or [])
        k += 1 if float(f.get("ratio") or 1) != 1 else 0
        if f.get("via"):
            k += 3                                           # the second flow, its delay, the stock
        for s in (f.get("from"), f.get("to")):
            if s and s not in measured:
                hidden.add(s)
    return k + len(hidden)
