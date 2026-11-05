"""Identifying a system dynamics model from its stocks' levels: the matrix method.

In a model whose flows are linear in the levels (constant, proportional, gap and linear forms),
every stock's change in a step is a linear function of the levels at the start of the step, plus
what arrives from outside:

    change of s at t  =  u_s(t)  +  sum_j  B[s][j] x level of j at t-1   (+ noise)

So the whole system is the matrix B and the outside inflows u. This module estimates them one stock
at a time, by least squares on single steps, then reads stock-and-flow structure off B (flows.py).

Announced eras: each distinct era label is a configuration of the system. The structure is shared:
one regression per stock, in which a level's number may differ in a configuration (a column "level
of j while configuration c holds" joins the search, and stays only if it pays for itself).
"""

from __future__ import annotations

import math

import numpy as np

from .changes import pelt
from .flows import merge_by_label, read_flows
from .regression import fit_subset, piece_values, row_sums, stepwise
from .shared_noise import analyse, coflows


def configurations(eras: list[dict] | None, t: list) -> tuple[list[str], np.ndarray, list[dict]] | None:
    """The distinct era labels, the configuration in force at each step (index into the labels),
    and the eras as time spans; None without two labels."""
    eras = sorted((e for e in eras or [] if e and e.get("start") is not None), key=lambda e: e["start"])
    name = lambda e: e.get("label") or str(e["start"])  # noqa: E731
    labels = list(dict.fromkeys(name(e) for e in eras))
    if len(labels) < 2:
        return None

    def cfg_at(time):
        c = None
        for e in eras:
            if time >= e["start"]:
                c = name(e)
        return c if c is not None else labels[0]

    T = len(t) - 1
    cfg = np.array([labels.index(cfg_at(t[k + 1])) for k in range(T)], dtype=int)
    spans = [{"from": e["start"], "to": eras[i + 1]["start"] - 1 if i + 1 < len(eras) else t[-1], "label": name(e)} for i, e in enumerate(eras)]
    if spans and spans[0]["from"] > t[0]:
        spans.insert(0, {"from": t[0], "to": spans[0]["from"] - 1, "label": cfg_at(t[0])})
    return labels, cfg, spans


def known_changes(known: dict, ids: list[str], n: int) -> np.ndarray:
    """The known change of each stock at each step (stocks x steps), 0 where nothing is known."""
    out = np.zeros((len(ids), n))
    for k, i in enumerate(ids):
        v = known.get(i)
        if v is not None:
            a = np.nan_to_num(np.asarray(v, dtype=float))[:n]
            out[k, : len(a)] = a
    return out


def identify_one(stocks: list[dict], run: dict, opts: dict) -> dict:
    ids = [s["id"] for s in stocks]
    S = len(ids)
    t = run["t"]
    T = len(t) - 1
    if T < 20:
        return {"flows": [], "note": "Too few steps."}
    lv = np.nan_to_num(np.array([np.asarray(run["levels"].get(i, [0] * len(t)), dtype=float) for i in ids]))
    prev, dy = lv[:, :-1], np.diff(lv, axis=1)
    if opts.get("known"):
        # what is already known (measured flows, the model's own equations) is taken out first, so
        # the rows explain the rest: known[s][i] is the known change of s from step i - 1 to i
        dy = dy - known_changes(opts["known"], ids, len(t))[:, 1:]
    if opts.get("injections"):
        # an announced injection is added at the start of its step, before that step's flows: the
        # step starts from the level plus the injection, and its change less the injection is the flows'
        inj = injected(opts["injections"], ids, t)[:, 1:]
        prev, dy = prev + inj, dy - inj
    if opts.get("dy_adjust") is not None:
        # what each stock's change would have been without what the measurement hides (a strategy's
        # estimate: outflows cut short by an emptied stock); stocks x steps, added to the changes
        dy = dy + opts["dy_adjust"]
    at = lambda k: t[k + 1]  # noqa: E731
    conf = configurations(opts.get("eras"), t) if opts.get("split_eras") else None
    labels, cfg = (conf[0], conf[1]) if conf else ([None], np.zeros(T, dtype=int))
    keep = opts.get("keep")
    if keep is None and opts.get("steps") is not None:
        # (held_out) a mask of the steps to fit on (index k: from t[k] to t[k+1]); the others are held out
        keep = np.flatnonzero(np.asarray(opts["steps"]))
    if keep is not None:
        # only these steps are fitted (the others, e.g. where a used stock ran dry, are left out);
        # an outside inflow's piece spans the steps left out between its kept steps
        keep = np.asarray(keep, dtype=int)
        if len(keep) < 20:
            return {"flows": [], "note": "Too few steps."}
        prev, dy, cfg, T = prev[:, keep], dy[:, keep], cfg[keep], len(keep)
        at = lambda k: t[keep[k] + 1]  # noqa: E731
    L = len(labels)
    final = {s["id"] for s in stocks if s.get("final")}
    unit = {s["id"]: s.get("unit") or "" for s in stocks}
    group = {s["id"]: s.get("group") for s in stocks}
    log_t = math.log(T)
    bic = lambda rss, p: T * math.log(max(rss, 1e-12) / T) + p * log_t  # noqa: E731
    add_gain = opts.get("add_gain", log_t)                                # a level joins a row only if clearly nonzero
    min_piece = opts.get("min_piece", 5)
    # parameters a change in the outside inflow counts for in the BIC comparisons (a constant is 1)
    per_change = {k: opts.get(f"{k}_change_params", 1) for k in ("level", "form")}
    # how changes in the outside inflow are found in a residual series; phase is "fine" (the first
    # pass), "pass" (later passes), "level" (a level tried in place of pieces) or "conserve"
    scan = opts.get("scan") or (lambda r, phase: pelt(r, min_piece, 0.5 if phase == "fine" else 1))  # noqa: E731

    only_rows = opts.get("only_rows") or {}                              # a level that may join only these rows

    # columns a strategy offers beside the levels (each a function of one stock's level, named, with
    # that stock as its base, and the rows that may take it); without them the columns are the levels
    extras = opts.get("extra_columns") or []
    # (hidden_delays) extra regressors after them (opts["extra"]: {"id", "series" aligned with t,
    # "source", "rows"}), e.g. a delayed copy of a stock's level standing for an unmeasured stock
    extra = opts.get("extra") or []
    # (opts "inputs") measured inputs, {name: value per step aligned with t}: each a column after the
    # others, its value during the step (t[k] to t[k + 1]: the value at t[k + 1]); any row may take one
    # (a positive number is an inflow k x input, a negative one an outflow), if it pays for itself
    # (a caller that keeps the raw fit or rows, keep_fit / keep_rows, reads every column as a stock's
    # level: it gets no input columns until it says it can take them, opts["input_columns"])
    use = opts.get("input_columns", not (opts.get("keep_fit") or opts.get("keep_rows")))
    inputs = [(k, np.nan_to_num(np.asarray(v, dtype=float))[1: len(t)]) for k, v in (opts.get("inputs") or {}).items() if k not in ids] if use else []
    if keep is not None:
        inputs = [(k, v[keep]) for k, v in inputs]
    names = ids + [e["id"] for e in extras] + [e["id"] for e in extra] + [k for k, _ in inputs]
    base = list(range(S)) + [ids.index(e["base"]) for e in extras] + [ids.index(e["source"]) for e in extra] + [0] * len(inputs)
    columns = (np.vstack([prev] + [np.asarray(e["x"], dtype=float)[None, :] for e in extras]
                         + [np.asarray(e["series"], dtype=float)[None, :-1] for e in extra]
                         + [v[None, :] for _, v in inputs]) if extras or extra or inputs else prev)
    N = len(names)
    F = S + len(extras)                                                   # the delayed copies start here
    I0 = F + len(extra)                                                   # the inputs start here

    def may_affect(j, s):
        if ids[j] in only_rows and ids[s] not in only_rows[ids[j]]:
            return False
        a, b = group[ids[j]], group[ids[s]]
        return j == s or not a or not b or a == b

    def fit_row(s: int) -> dict:
        sid, y = ids[s], dy[s]
        X = [columns[j] for j in range(N)]
        col = [(j, -1) for j in range(N)]
        allowed = [j for j in range(F) if may_affect(base[j], s) and not (sid in final and base[j] == s)
                   and (j < S or extras[j - S].get("rows") is None or sid in extras[j - S]["rows"])]
        # (opts "extra", hidden_delays) a delayed copy only in the rows it names; a copy that stands
        # in for its source's level in this row ("replaces"): not both
        allowed += [F + e for e, x in enumerate(extra) if sid in x["rows"]]
        gone = {ids.index(x["source"]) for x in extra if sid in x["rows"] and x.get("replaces")}
        allowed = [j for j in allowed if j not in gone] + list(range(I0, N))
        own = s in allowed
        Xa = np.array(X)

        def step(sums, start):
            return stepwise(sums, start, allowed, add_gain, 4 * len(X))

        # fine pieces first (so the levels' numbers are not biased by unexplained changes), then the
        # usual price; the pieces are found in what the levels leave unexplained. Start from every
        # level when there are few (backward elimination), from none when there are many.
        cuts = [0, T]
        chosen = list(allowed) if len(allowed) <= opts.get("backward_max", 0) else []
        no_pieces = False
        # opts["start_with"]: {stock: [columns its row starts from]} (none by default)
        chosen = list(dict.fromkeys(chosen + [names.index(j) for j in (opts.get("start_with") or {}).get(sid, ()) if j in names and names.index(j) in allowed]))
        for p in range(5):
            sums = row_sums(y, Xa, cuts)
            chosen = step(sums, chosen)
            if p == 0 and L > 1:
                # the levels this row uses (and its own) may have a number per configuration; they
                # join before any change in the outside inflow is looked for
                js = list(dict.fromkeys([col[q][0] for q in chosen if col[q][0] < F] + ([s] if own else [])))   # (not delayed copies)
                if opts.get("era_levels") == "allowed":
                    # (an option) every level the row may use, not only those it chose without eras:
                    # a level that barely moves is chosen only once its number may follow the configuration
                    js = list(dict.fromkeys(js + [q for q in allowed if col[q][1] < 0]))
                for j in js:
                    for c in range(L):
                        X.append(np.where(cfg == c, columns[j], 0.0))
                        col.append((j, c))
                        allowed.append(len(X) - 1)
                Xa = np.array(X)
                sums = row_sums(y, Xa, cuts)
                chosen = step(sums, chosen)
            f = fit_subset(sums, chosen)
            res = y - (f.beta @ Xa[chosen] if chosen else 0)
            # opts["steady"]: stocks whose outside inflow may not change (none by default)
            nxt = [0, T] if opts.get("changes") is False or no_pieces or sid in opts.get("steady", ()) else scan(res, "fine" if p == 0 else "pass")
            if p == 0 and opts.get("gate") and len(nxt) > 2:
                # before any piece can bias the levels: a stock gets pieces only if, beside the levels
                # alone, an outside inflow beats none or the gate finds a change in what they leave
                const = bic(f.rss, len(chosen) + 1)
                none_ = bic(fit_subset(row_sums(y, Xa, None), chosen).rss, len(chosen))
                if const >= none_ and len(opts["gate"](res)) <= 2:
                    if opts.get("trace") == sid:
                        print(f"  {sid}: no outside inflow beside the levels alone; no pieces")
                    nxt, no_pieces = [0, T], True
            if opts.get("trace") == sid:
                print(f"  {sid} pass {p}: {len(cuts) - 1} pieces;", ", ".join(f"{names[col[q][0]]}{'@' + str(col[q][1]) if col[q][1] >= 0 else ''} {b:.4f}" for q, b in zip(chosen, f.beta)), f"-> {len(nxt) - 1}")
            if nxt == cuts and p >= 2:
                break
            cuts = nxt

        # a level that changes slowly looks like a series of changes in the outside inflow, and the
        # fine pieces take it before the level is tried: the two levels whose means over the pieces
        # follow the pieces' values best are tried in place of pieces, and kept if the row with it,
        # and the pieces it still needs, is clearly better
        if opts.get("level_for_pieces", True) and opts.get("changes") is not False and not no_pieces and len(cuts) > 2:
            def fit_on(cols, c):
                f = fit_subset(row_sums(y, Xa[cols], c), list(range(len(cols))))
                return f, bic(f.rss, len(cols) + 1 + per_change["level"] * (len(c) - 2))

            def pieces_for(cols, c):
                f, _ = fit_on(cols, c)
                return scan(y - (f.beta @ Xa[cols] if cols else 0), "level")

            cur = fit_on(chosen, cuts)[1]
            for _ in range(3):
                f, _ = fit_on(chosen, cuts)
                pv = piece_values(y, Xa, chosen, f.beta, cuts)
                w = np.array([p_["w"] for p_ in pv], dtype=float)
                v = np.array([p_["v"] for p_ in pv])
                starts = np.array(cuts[:-1])

                def follows(j):
                    m = np.add.reduceat(Xa[j], starts) / w
                    mm, mv = (w * m).sum() / w.sum(), (w * v).sum() / w.sum()
                    sxy, sxx, syy = (w * (m - mm) * (v - mv)).sum(), (w * (m - mm) ** 2).sum(), (w * (v - mv) ** 2).sum()
                    return sxy * sxy / (sxx * syy) if sxx and syy else 0.0

                tries = sorted((j for j in allowed if j not in chosen and col[j][1] < 0), key=lambda j: -follows(j))[:2]
                best = None
                for j in tries:
                    cols = chosen + [j]
                    c = pieces_for(cols, pieces_for(cols, [0, T]))
                    sc = fit_on(cols, c)[1]
                    if sc < cur - log_t and (best is None or sc < best[2]):
                        best = (cols, c, sc)
                if best is None:
                    break
                if opts.get("trace") == sid:
                    print(f"  {sid}: {names[col[best[0][-1]][0]]} in place of pieces: {len(cuts) - 1} -> {len(best[1]) - 1}")
                chosen, cuts, cur = best

        # the outside inflow: none, constant, or changing (only a stock with an inflow on average
        # clearly one way may have it change); the levels chosen alongside the pieces stay
        def try_form(c, extra):
            sums = row_sums(y, Xa, c)
            f = fit_subset(sums, chosen)
            return {"cuts": c, "f": f, "sc": bic(f.rss, len(chosen) + extra)}

        raw = row_sums(y, Xa, None)
        f_none = fit_subset(raw, chosen)
        none = {"cuts": None, "f": f_none, "sc": bic(f_none.rss, len(chosen))}
        constant = try_form([0, T], 1)
        pick = none if none["sc"] <= constant["sc"] else constant
        if len(cuts) > 2:
            pieces = try_form(cuts, 1 + per_change["form"] * (len(cuts) - 2))
            vals = piece_values(y, Xa, chosen, pieces["f"].beta, cuts)
            W = sum(p_["w"] for p_ in vals)
            mean = sum(p_["v"] * p_["w"] for p_ in vals) / W
            sd = math.sqrt(sum(p_["w"] * (p_["v"] - mean) ** 2 for p_ in vals) / W)
            # "clearly one way": by default against the spread of the pieces' values (a few short,
            # large pieces - a stock run dry, a burst - can veto a row whose pieces BIC prefers by
            # far); opts["pieces_gate"] = "noise" tests the mean against its own standard error
            # from the steps' noise instead (2 x sqrt(rss / T) / sqrt(T))
            if opts.get("pieces_gate") == "noise":
                clear = abs(mean) > 2 * math.sqrt(pieces["f"].rss / T) / math.sqrt(T)
            else:
                clear = abs(mean) > 0.5 * sd
            if (pick is constant or clear) and pieces["sc"] < pick["sc"]:
                pick = pieces
        # each configuration's numbers: the shared one plus the configuration's own
        beta = pick["f"].beta
        coef_l: list[dict] = [{} for _ in range(L)]
        for q, b in zip(chosen, beta):
            j, lab = col[q]
            for c in range(L):
                if lab < 0 or lab == c:
                    coef_l[c][names[j]] = coef_l[c].get(names[j], 0.0) + float(b)
        inflow = []
        if pick["cuts"]:
            inflow = [{"from": at(p_["a"]), "to": at(p_["b"] - 1), "value": p_["v"]} for p_ in piece_values(y, Xa, chosen, beta, pick["cuts"])]
        return {"id": sid, "coefL": coef_l, "X": Xa, "col": col, "set": chosen, "beta": beta, "inflow": inflow,
                "cuts": pick["cuts"], "rss": pick["f"].rss, "p": len(chosen) + (len(pick["cuts"]) - 1 if pick["cuts"] else 0)}

    rows = [fit_row(s) for s in range(S)]
    ctx = {"stocks": stocks, "ids": ids, "t": t, "final": final, "unit": unit, "prev": prev, "dy": dy, "cfg": cfg, "L": L,
           "labels": labels, "conf": conf, "bic": bic, "log_t": log_t, "T": T, "at": at,
           "extras": extras, "names": names, "columns": columns, "inputs": [k for k, _ in inputs]}
    # opts["keep_rows"]: the rows as fitted, before conservation, with what finish() needs to assemble
    # a model from them (the row_crossover strategy mixes rows of several candidates)
    kept = [raw_row(r) for r in rows] if opts.get("keep_rows") else None
    out = finish(rows, ctx, opts)
    if kept is not None:
        out["raw_rows"], out["ctx"] = kept, ctx
    return out


def raw_row(row: dict) -> dict:
    """A copy of a fitted row that conservation and re-measuring (which change rows in place) leave
    alone; of its columns only the chosen ones are kept (all conservation reads)."""
    sel = list(row["set"])
    return {**row, "coefL": [dict(c) for c in row["coefL"]], "inflow": [dict(g) for g in row["inflow"]],
            "X": row["X"][sel] if sel else row["X"][:0], "col": [row["col"][q] for q in sel], "set": list(range(len(sel)))}


def finish(rows: list[dict], ctx: dict, opts: dict) -> dict:
    """From fitted rows (one per stock, changed in place) to the model: conservation, the outside
    inflows re-measured where it changed a row, and the structure read off the matrix."""
    stocks, ids, t, prev, dy, cfg, L = ctx["stocks"], ctx["ids"], ctx["t"], ctx["prev"], ctx["dy"], ctx["cfg"], ctx["L"]
    labels, conf = ctx["labels"], ctx["conf"]
    at = ctx.get("at") or (lambda k: t[k + 1])                            # (opts "keep": the kept steps' times)
    extras, names, columns = ctx.get("extras") or [], ctx.get("names") or ids, ctx.get("columns", prev)   # (opts "extra_columns")
    # shared noise (opt-in): which gains are moves and which links, read off the rows' residuals
    noise = analyse(rows, ids, dy, prev, ctx["unit"], ctx["final"], opts.get("noise_z", 3.0), opts.get("tol", 0.003), opts.get("noise_misfit", False), opts.get("noise_swap", False), opts.get("noise_one_sided", False)) if opts.get("shared_noise") else None
    kinds = noise.kinds if noise else None
    conserve(rows, ids, ctx["final"], ctx["unit"], prev, dy, cfg, L, ctx["bic"], ctx["log_t"], opts, kinds)

    # a row whose numbers conservation changed: its outside inflow's pieces are measured again with
    # the numbers it ends with (the pieces were measured beside the row's own, biased, drain)
    for s, row in enumerate(rows):
        if not row.get("conserved") or not row["cuts"] or opts.get("remeasure") is False:
            continue
        res = dy[s].copy()
        for c in range(L):
            m = cfg == c
            for j, k in row["coefL"][c].items():
                res[m] -= k * columns[names.index(j)][m]
        row["inflow"] = [{"from": at(p_["a"]), "to": at(p_["b"] - 1), "value": p_["v"]} for p_ in piece_values(res, prev, [], np.zeros(0), row["cuts"])]

    if extras:
        from .hinges import read_flows_hinged

        per_label = [read_flows_hinged(stocks, [{"id": r["id"], "coef": r["coefL"][c], "inflow": r["inflow"] if c == 0 else []} for r in rows], prev, opts.get("tol", 0.003), extras) for c in range(L)]
    else:
        extra = opts.get("extra") or []          # (hidden_delays) delayed copies, read as unmeasured stocks
        per_label = [read_flows(stocks, [{"id": r["id"], "coef": r["coefL"][c], "inflow": r["inflow"] if c == 0 else []} for r in rows], prev, opts.get("tol", 0.003), kinds,
                                **({"extra": extra, "prevx": columns} if extra else {})) for c in range(L)]
    if noise and opts.get("coflows", True):
        per_label = [coflows(f, noise) for f in per_label]
    for c, fl in enumerate(per_label):                    # (opts "inputs") a row's number on a measured input
        fl += input_flows(rows, c, ctx.get("inputs") or [], columns, names)
    flows = merge_by_label(per_label, labels, conf[2]) if L > 1 else per_label[0]
    matrix = [{k: v for k, v in r.items() if k not in ("X", "col", "set", "beta", "trust_receivers")} for r in rows]
    for r in matrix:
        r["coef"] = r["coefL"][0]
    out = {"flows": flows, "matrix": matrix, "method": "identify", "windows": ctx["T"], "window": 1}
    if L > 1:
        out["eras"] = labels
    if opts.get("keep_fit"):
        # the rows as fitted (columns, chosen set, numbers, pieces), for a strategy that reads the
        # matrix another way (off by default: the result stays as it was)
        out["fit"] = {"rows": rows, "ids": ids, "prev": prev, "dy": dy, "cfg": cfg, "labels": labels,
                      "spans": conf[2] if conf else None, "at": [at(k) for k in range(ctx["T"])]}
    return out


def conserve(rows, ids, final, unit, prev, dy, cfg, L, bic, log_t, opts, kinds=None) -> None:
    """Conservation unless the data clearly says otherwise: a stock's drain is what same-unit stocks
    are seen to receive from it (measured, often more precisely, in their rows). Per configuration,
    tested together: setting the drain to what the receivers take must cost the row no more than
    log T in BIC (refitting the rest), or, for a row with a changing inflow (which cannot separate
    its drain from the changes), be within 30%. A receiver whose gain the shared noise reads as a link
    (`kinds`) takes nothing from the stock; and a stock with such a link has its drain lowered only
    when the refit test passes (the link's gain is no evidence that its drain is too high)."""
    kinds = kinds or {}
    for s, row in enumerate(rows):
        a = row["id"]
        if a in final:
            continue
        drain = [-(c.get(a) or 0.0) for c in row["coefL"]]
        moves = [sum(r["coefL"][c][a] for r in rows if r["id"] != a and (r["coefL"][c].get(a) or 0) > 0 and unit[r["id"]] == unit[a] and kinds.get((a, r["id"])) != "link") for c in range(L)]
        # what receivers take through an unmeasured stock fed by a (opts["extra"] sourced by a)
        via = [e["id"] for e in opts.get("extra") or [] if e["source"] == a]
        if via:
            moves = [m + sum(r["coefL"][c][e] for r in rows for e in via if r["id"] != a and (r["coefL"][c].get(e) or 0) > 0 and unit[r["id"]] == unit[a]) for c, m in enumerate(moves)]
        # a row whose outside inflow changes (in pieces, or one a strategy knows varies: opts
        # "varying_inflow", stock ids) cannot separate its drain from the changes
        changing = (bool(row["cuts"]) and len(row["cuts"]) > 2) or a in (opts.get("varying_inflow") or ())
        linked = any(kinds.get((a, r["id"])) == "link" for r in rows)
        trust = row.get("trust_receivers", opts.get("trust_receivers")) is not False   # a row may carry its own setting
        cand = [c for c in range(L) if moves[c] > 0 and ((changing and moves[c] > drain[c] and trust) or drain[c] > moves[c])]
        if not cand:
            continue
        in_cand = np.zeros(L, bool)
        in_cand[cand] = True
        mv = np.array(moves)
        y2 = dy[s] + np.where(in_cand[cfg], mv[cfg] * prev[s], 0.0)
        others = [(q, i) for i, q in enumerate(row["set"]) if row["col"][q][0] != s]
        Xf = [row["X"][q] for q, _ in others]
        if len(cand) < L:
            Xf.append(np.where(in_cand[cfg], 0.0, prev[s]))
        Xf = np.array(Xf) if Xf else np.zeros((0, len(y2)))
        idx = list(range(len(Xf)))
        if changing:
            fixed_part = sum((row["beta"][i] * row["X"][q] for q, i in others), np.zeros(len(y2)))
            scan = opts.get("scan") or (lambda r, phase: pelt(r, opts.get("min_piece", 5), 1))  # noqa: E731
            cuts2 = scan(y2 - fixed_part, "conserve")
        else:
            cuts2 = row["cuts"]
        sums = row_sums(y2, Xf, cuts2)
        fixed = fit_subset(sums, idx).rss
        p2 = len(idx) + (len(cuts2) - 1 if cuts2 else 0)
        slack = opts.get("slack", 0.3)
        if bic(fixed, p2) <= bic(row["rss"], row["p"]) + log_t or (changing and all(abs(drain[c] - moves[c]) / max(drain[c], moves[c]) < slack and not (linked and drain[c] > moves[c]) for c in cand)):
            for c in cand:
                row["coefL"][c][a] = -moves[c]
            row["conserved"] = True


def input_flows(rows: list[dict], c: int, inputs: list[str], columns: np.ndarray, names: list[str]) -> list[dict]:
    """Flows reading measured inputs, from the rows' numbers on them in configuration c: k > 0 an
    inflow from outside at k x input, k < 0 an outflow to outside at -k x input."""
    out = []
    for r in rows:
        for name in inputs:
            k = r["coefL"][c].get(name) or 0.0
            if not k:
                continue
            rate = {"form": "linear", "c": 0, "terms": [{"stock": name, "k": round(abs(k), 6)}]}
            ends = {"from": None, "to": r["id"]} if k > 0 else {"from": r["id"], "to": None}
            out.append({**ends, "ratio": 1, "rate": rate, "rates": [rate], "perStep": abs(k) * float(columns[names.index(name)].mean()),
                        "nearly": [], "bothWays": False, "input": name})
    return out


def injected(injections: list[dict], ids: list[str], t: list) -> np.ndarray:
    """Announced injections ({t, stock, amount}) as stocks x steps: what was added at the start of
    each step."""
    out = np.zeros((len(ids), len(t)))
    index = {v: i for i, v in enumerate(t)}
    for j in injections:
        if j.get("stock") in ids and j["t"] in index:
            out[ids.index(j["stock"]), index[j["t"]]] += float(j["amount"])
    return out


def known_flows(known: dict, ids: list[str], t: list) -> list[dict]:
    """The known changes as flows the simulator runs: per stock, what arrives (an inflow from
    outside) and what leaves (an outflow to outside), step by step."""
    ch = known_changes(known, ids, len(t))
    out = []
    for k, i in enumerate(ids):
        if not ch[k].any():
            continue
        for sign, ends in ((1, {"from": None, "to": i}), (-1, {"from": i, "to": None})):
            v = np.maximum(0.0, sign * ch[k])
            if v.any():
                out.append({**ends, "ratio": 1, "inflow": [{"from": t[s], "to": t[s], "value": float(v[s])} for s in range(len(t))], "known": True})
    return out
