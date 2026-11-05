# Vendored for kalman_lna from sdlab/discovery/strategies/lna_products/run.py at commit ec2f9e8 (imports rewritten to the
# vendored copies; otherwise unchanged). Do not edit: re-vendor instead.
"""The products stage as hooks on lna's discover (lna/run.py HOOKS): product columns, a virtual
row a x c / mean(c) for each pair of measured stocks appended to prev (after the stocks and any
measured inputs), so that a move a -> b (or a's outflow) can run at k x a x c / mean(c), a rate
scaled by a linear effect of c. Virtual rows have no flows of their own and no change; lna's search
never reads them (its atoms are lna's), so lna's own reading is unchanged. After lna's search each
kept move and outflow is offered scaled by every other stock, and the search runs again from there:
kept when the BIC gain is MARGIN x log T. No search code is copied."""

from __future__ import annotations

import numpy as np

from orchestra.sdlab.discovery.strategies.kalman_lna._vendored.lna.atoms import Atom
from orchestra.sdlab.discovery.strategies.kalman_lna._vendored.lna.fit import Data, Fitted, OutOfTime, fit
from orchestra.sdlab.discovery.strategies.kalman_lna._vendored.lna.run import SCRATCH, bic, search
from orchestra.sdlab.discovery.strategies.kalman_lna._vendored.lna_levels.run import written


MARGIN = 1.0         # a model with products must beat lna's by this many log T in BIC (the price of two numbers: the
                     # products offered are many, so one that pays only its own price is too often noise)


def product_atoms(S: int, unit: list[str], final: set[int], base: int | None = None) -> tuple[list[Atom], list[tuple[int, int]]]:
    """(atoms, pairs): pairs[v] = (a, c) is virtual row base + v (default S); each pair's moves
    a -> b (b != c, same unit) and a's outflow."""
    base = S if base is None else base
    pairs, atoms = [], []
    for a in range(S):
        if a in final:
            continue
        for c in range(S):
            if c == a:
                continue
            v = base + len(pairs)
            pairs.append((a, c))
            atoms.append(Atom("out", src=a, driver=v))
            for b in range(S):
                if b not in (a, c) and unit[a] == unit[b]:
                    atoms.append(Atom("move", src=a, dst=b, driver=v))
    return atoms, pairs


def with_rows(cur: Fitted, d: Data, n: int) -> tuple[Fitted, Data]:
    """The product columns as rows of the changes too (zero change, no extra variance), as the stage
    was first built: each adds only a constant to the likelihood, but L-BFGS's stopping rule is
    relative to the likelihood's size, so with or without them some fits stop elsewhere."""
    S, T = d.dy.shape
    d2 = Data(d.prev, np.vstack([d.dy, np.zeros((n, T))]), d.cfg, d.delta,
              None if d.free is None else np.concatenate([d.free, np.zeros(n, bool)]), d.deadline)
    tau = None if cur.tau is None else np.concatenate([cur.tau, np.zeros(n)])
    return Fitted(cur.atoms, cur.theta, cur.nll, tau), d2


def stage(state: dict, margin: float = MARGIN, then=None, rows: bool = False) -> dict:
    """{data, after_search, flows} for the products stage (3 to SCRATCH stocks; otherwise nothing);
    `state` collects what the strategy reports (lna_cur, lna_flows, pairs, products, product_gain).
    then(cur, d, ctx): an after_search stage run next on the same data (lna_stack's levels). rows:
    the product columns also as zero rows of the changes there (with_rows; lna_stack, as it was
    first built: without them it lost 101 counts-short; lna_products alone lost nothing without
    them, and with them reads an effect into 014 counts-short)."""

    def data(prev, dy, ctx):
        S = len(ctx["ids"])
        if not 3 <= S <= SCRATCH:
            return prev, dy
        extra, pairs = product_atoms(S, ctx["unit"], ctx["final"], base=prev.shape[0])
        norm = [float(max(1e-9, np.maximum(prev[c], 0).mean())) for _, c in pairs]
        vprev = np.array([np.maximum(prev[a], 0) * np.maximum(prev[c], 0) / n for (a, c), n in zip(pairs, norm)]).reshape(len(pairs), -1)
        ids = ctx["ids"]
        ctx["names"] = (ctx.get("names") or list(ids)) + [f"{ids[a]}*{ids[c]}" for a, c in pairs]
        state.update(extra=extra, pairs=pairs, norm=norm, first=prev.shape[0])
        return np.vstack([prev, vprev]), dy

    def offered(cur, d):
        offer = []
        for a, th in zip(cur.atoms, cur.theta):
            if th <= 0 or a.era is not None or a.kind not in ("move", "out") or a.uses:
                continue
            offer += [x for x in state["extra"] if x.kind == a.kind and x.src == a.src and x.dst == a.dst and x not in offer]
        if not offer:
            return cur
        try:
            g = search(fit(cur.atoms + offer, d, np.concatenate([cur.theta, np.full(len(offer), 1e-3)]), tau=cur.tau), d)
        except OutOfTime:                # past lna's budget: lna's reading stands
            state["products_out_of_time"] = True
            return cur
        state["product_gain"] = gain = float(bic(cur, d) - bic(g, d))
        return g if gain > margin * np.log(d.dy.shape[1]) else cur

    def after_search(cur, d, ctx):
        state.setdefault("lna_cur", cur)
        state.setdefault("base", ctx["base"])
        if not state.get("extra"):
            return then(cur, d, ctx) if then else cur
        S = d.dy.shape[0]
        g, d2 = with_rows(cur, d, len(state["pairs"])) if rows else (cur, d)
        g = offered(g, d2)
        if then:
            g = then(g, d2, ctx)
        return Fitted(g.atoms, g.theta, g.nll, None if g.tau is None else g.tau[:S])

    def flows(flows, cur, d, ctx):
        state.setdefault("lna_flows", written(state["lna_cur"], d, ctx))
        pairs = state.get("pairs")
        if not pairs:
            return flows
        S, lo = len(ctx["ids"]), state["first"]
        state["products"] = [ctx["names"][a.driver] for a, th in zip(cur.atoms, cur.theta) if th > 0 and a.driver is not None and lo <= a.driver < lo + len(pairs)]
        out = with_effects(flows, ctx["ids"], pairs, state["norm"], d.prev[:S])
        if out is None:                  # a product in eras or pieces cannot be written: lna's own reading
            state["products"] = []
            return list(state["lna_flows"])
        return out

    return {"data": data, "after_search": after_search, "flows": flows}


def with_effects(flows: list[dict], ids: list[str], pairs: list, norm: list[float], prev: np.ndarray) -> list[dict] | None:
    """Flows reading a virtual stock a*c -> a's proportional rate with a linear effect of c (normal
    mean(c), up to the highest c seen); beside a plain flow between the same ends, the two are one
    flow with a table effect through the lowest and highest c seen. None: a virtual reading that
    cannot be written (in eras or pieces)."""
    virt = {f"{ids[a]}*{ids[c]}": (a, c, n) for (a, c), n in zip(pairs, norm)}
    plain, eff = [], []
    for f in flows:
        st = (f.get("rate") or {}).get("stock")
        if st in virt:
            if f.get("byTime") or f.get("inflow"):
                return None
            eff.append(f)
        else:
            plain.append(f)
    out = list(plain)
    for f in eff:
        a, c, n = virt[f["rate"]["stock"]]
        k = float(f["rate"]["k"])
        lo, hi = float(np.maximum(prev[c], 0).min()), float(np.maximum(prev[c], 0).max())
        twin = next((g for g in out if g.get("from") == f.get("from") and g.get("to") == f.get("to") and (g.get("rate") or {}).get("form") == "proportional"
                     and g["rate"].get("stock") == ids[a] and not (g.get("byTime") or g.get("uses") or g.get("effects"))), None)
        if twin is None:
            effect = {"stock": ids[c], "shape": "linear", "normal": round(n, 4), "min": 0, "max": round(max(1.0, 1.25 * hi / n), 4)}
            rate = {"form": "proportional", "stock": ids[a], "k": k}
        else:
            k0 = float(twin["rate"]["k"])
            out.remove(twin)
            pts = [[round(x, 4), (k0 + k * x / n) / (k0 + k)] for x in (lo, hi)]
            effect = {"stock": ids[c], "shape": "table", "points": pts}
            rate = {"form": "proportional", "stock": ids[a], "k": k0 + k}
        out.append({**f, "rate": rate, "rates": [rate], "effects": [effect], "driver": ids[a], "uses": f.get("uses") or []})
        if not out[-1]["uses"]:
            out[-1].pop("uses")
    return out
