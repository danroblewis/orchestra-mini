"""Hidden stocks reconstructed from a unit's mass balance.

A delay nobody measures between two measured stocks of one unit (A -> H -> B) is read by the matrix
method as a loss of A to outside and arrivals into B from outside (in pieces fitted to the run). By
conservation, what A sends towards H and what B receives from it differ only by what H holds:

    H(t) = H(0) + alpha x C(t) - U(t),    C = cumsum(A's loss),  U = cumsum(B's unexplained arrivals)

where alpha is the share of A's found loss that really goes to H. With H draining into B at d a step
(a delay of mean 1 / d, Little's law), B's arrivals are u(t) = d x H(t-1), so

    u(t) = d H(0) + d alpha C(t-1) - d U(t-1)

is a linear regression on [1, C, U]: H(0), alpha and d come out of one least-squares fit, and H's
whole level series with them. Nothing is assumed about the delay's shape (a pipeline or two stages
come out as one stock with the mean delay). A false pairing (A's loss and B's arrivals unrelated)
does not balance: H drifts, goes negative, or needs a residence time longer than the run can show,
and is rejected.
"""

from __future__ import annotations

import copy

import numpy as np

from . import balance_filter
from .simulate import Model, levels_array

SINK = "__amount__"


def amounts(flows: list[dict], ids: list[str], t: list, A: np.ndarray) -> np.ndarray:
    """Each flow's amount at every step of the run (flows x steps, step k from level k to k + 1),
    evaluated at the measured levels (nothing caps it: the levels are what they were)."""
    out = np.zeros((len(flows), A.shape[1] - 1))
    if A.shape[1] < 2:
        return out
    L = np.vstack([A[:, :-1], np.zeros((1, A.shape[1] - 1))])
    tt = np.asarray(t, dtype=float)[1:]
    for i, f in enumerate(flows):
        m = Model([{**f, "from": None, "to": SINK, "ratio": 1}], ids + [SINK])
        out[i] = m.step(L.copy(), tt)[-1]
    return out


def net_effect(flows: list[dict], ids: list[str], t: list, A: np.ndarray, stock: str) -> np.ndarray:
    """What the flows add to `stock` at every step, at the measured levels."""
    if not flows:
        return np.zeros(A.shape[1] - 1)
    am = amounts(flows, ids, t, A)
    sign = np.array([(1.0 if f.get("to") == stock else 0.0) - (float(f.get("ratio") or 1) if f.get("from") == stock else 0.0) for f in flows])
    return sign @ am


def reconstruct(F: np.ndarray, u: np.ndarray) -> dict | None:
    """H from the sender's loss F and the receiver's unexplained arrivals u (both per step, aligned):
    {"alpha", "d", "h0", "H": level at every time (len(F) + 1), "rss", "r2"}; None if the fit is
    degenerate."""
    T = len(F)
    if T < 20 or len(u) != T:
        return None
    C = np.concatenate([[0.0], np.cumsum(F)])
    U = np.concatenate([[0.0], np.cumsum(u)])
    X = np.column_stack([np.ones(T), C[:-1], U[:-1]])
    try:
        beta, *_ = np.linalg.lstsq(X, u, rcond=None)
    except np.linalg.LinAlgError:
        return None
    d = -float(beta[2])
    if not np.isfinite(d) or d <= 0:
        return None
    alpha = float(beta[1]) / d
    h0 = float(beta[0]) / d
    H = h0 + alpha * C - U
    res = u - X @ beta
    rss = float(res @ res)
    tss = float(((u - u.mean()) ** 2).sum()) or 1e-12
    return {"alpha": alpha, "d": d, "h0": h0, "H": H, "rss": rss, "r2": 1 - rss / tss}


def drains(flows: list[dict], b: str) -> list[int]:
    """B's own drains: flows out of B at a rate proportional to B's level (to outside or a move)."""
    out = []
    for i, f in enumerate(flows):
        r = f.get("rate") or {}
        if f.get("from") != b or f.get("driver") or f.get("inflow") or f.get("byTime") or f.get("effects") or f.get("uses"):
            continue
        if r.get("form") == "proportional" and r.get("stock") == b:
            out.append(i)
        elif r.get("form") == "linear" and not r.get("c") and [x.get("stock") for x in r.get("terms") or []] == [b]:
            out.append(i)
    return out


def drain_k(f: dict) -> float:
    r = f["rate"]
    return float(r["k"]) if r["form"] == "proportional" else float(r["terms"][0]["k"])


def reconstruct_joint(F: np.ndarray, y: np.ndarray, B: np.ndarray, other: np.ndarray, d0: float, kb0: float) -> dict | None:
    """As reconstruct, with the receiver's own drain kB refitted alongside (its found value is biased
    by the pieces that stood for H): y = B's change less what other kept flows add (per step), B its
    level (len(F) + 1), other those flows' part. With U = cumsum(y + kB x B),

        y(t) = d (H(0) + B(0)) + d alpha C(t-1) + d O(t-1) - (d + kB) B(t-1) - d kB SB(t-1)

    (O = cumsum(other), SB = cumsum(B)) is linear in d (H(0) + B(0)) and d alpha once d and kB are
    given: those two are searched from the simple fit's d and the found kB."""
    from scipy.optimize import minimize

    T = len(F)
    if T < 20 or d0 <= 0:
        return None
    C = np.concatenate([[0.0], np.cumsum(F)])[:-1]
    Oc = np.concatenate([[0.0], np.cumsum(other)])[:-1]
    Bp = B[:-1]
    SB = np.concatenate([[0.0], np.cumsum(Bp)])[:-1]
    X = np.column_stack([np.ones(T), C])

    def inner(p):
        d, kb = p
        r = y - d * Oc + (d + kb) * Bp + d * kb * SB
        beta, *_ = np.linalg.lstsq(X, r, rcond=None)
        res = r - X @ beta
        return float(res @ res), beta

    def cost(p):
        return 1e300 if p[0] <= 1e-4 or p[1] < 0 else inner(p)[0]

    opt = minimize(cost, np.array([d0, max(kb0, 1e-3)]), method="Nelder-Mead", options={"xatol": 1e-6, "fatol": 1e-10, "maxiter": 600})
    d, kb = float(opt.x[0]), float(opt.x[1])
    if d <= 1e-4 or kb < 0:
        return None
    rss, beta = inner((d, kb))
    alpha = float(beta[1]) / d
    h0 = float(beta[0]) / d - float(B[0])
    Cf = np.concatenate([[0.0], np.cumsum(F)])
    U = np.concatenate([[0.0], np.cumsum(y + kb * Bp)])
    H = h0 + alpha * Cf - U
    tss = float(((y - y.mean()) ** 2).sum()) or 1e-12
    return {"alpha": alpha, "d": d, "kb": kb, "h0": h0, "H": H, "rss": rss, "r2": 1 - rss / tss}


def plausible(rec: dict, T: int, max_alpha: float = 1.05, max_residence: float = 0.05, below: float = 0.1) -> str | None:
    """Why a reconstruction is not a hidden stock (None if it may be one): A must send at most all it
    loses (alpha), the delay must be short beside the run (H must turn over at least 20 times in it, or it
    cannot be told from drift), and H must not go below 0 by more than a tenth of its mean."""
    if rec["alpha"] <= 0 or rec["alpha"] > max_alpha:
        return f"alpha {rec['alpha']:.3f}"
    if 1 / rec["d"] > max_residence * T:
        return f"residence {1 / rec['d']:.1f} steps"
    H = rec["H"]
    mean = float(H.mean())
    if mean <= 0 or float(H.min()) < -below * mean:
        return f"level min {float(H.min()):.1f}, mean {mean:.1f}"
    return None


def scale_rate(rate: dict, s: float) -> dict:
    """The rate times s, in its own form."""
    r = copy.deepcopy(rate)
    form = r.get("form")
    if form == "constant":
        r["value"] = float(r.get("value") or 0) * s
    elif form == "proportional":
        r["k"] = float(r.get("k") or 0) * s
    elif form == "linear":
        r["c"] = float(r.get("c") or 0) * s
        for x in r.get("terms") or []:
            x["k"] = float(x.get("k") or 0) * s
    elif form == "gap":
        r["time"] = float(r.get("time") or 1) / max(s, 1e-12)
    return r


def scale_flow(f: dict, s: float, **changes) -> dict:
    g = {k: v for k, v in f.items() if k not in ("rates", "perStep", "nearly")}
    if g.get("byTime"):
        g["byTime"] = [{**seg, "rate": scale_rate(seg["rate"], s)} for seg in g["byTime"]]
    if g.get("rate"):
        g["rate"] = scale_rate(g["rate"], s)
    return {**g, **changes}


def senders(flows: list[dict], a: str, b: str) -> list[int]:
    """A's losses that may really go towards B: to outside, or a move into B (never a link)."""
    return [i for i, f in enumerate(flows) if f.get("from") == a and (f.get("to") is None or f.get("to") == b)
            and not f.get("driver") and not f.get("inflow")]


def arrivals(flows: list[dict], a: str, b: str) -> list[int]:
    """B's found arrivals the hidden stock would replace: from outside (constant or in pieces), and
    links from A (B's arrivals read as following A's level)."""
    return [i for i, f in enumerate(flows) if f.get("to") == b and f.get("from") is None
            and (not f.get("driver") or f.get("driver") == a)]


def pairs(flows: list[dict], stocks: list[dict]) -> list[tuple[str, str]]:
    """Candidate (sender, receiver) pairs: A loses to outside, B (of A's unit and group) gets things
    from outside."""
    unit = {s["id"]: s.get("unit") or "" for s in stocks}
    group = {s["id"]: s.get("group") for s in stocks}
    final = {s["id"] for s in stocks if s.get("final")}
    out = []
    for s_a in stocks:
        a = s_a["id"]
        if a in final or not any(f.get("from") == a and f.get("to") is None and not f.get("inflow") for f in flows):
            continue
        for s_b in stocks:
            b = s_b["id"]
            if b == a or unit[b] != unit[a] or (group[a] and group[b] and group[a] != group[b]):
                continue
            if any(f.get("to") == b and f.get("from") is None and not f.get("driver") for f in flows):
                out.append((a, b))
    return out


def propose(flows: list[dict], stocks: list[dict], run: dict, a: str, b: str, opts: dict | None = None) -> dict | None:
    """The model with A -> H -> B in place of A's loss to outside and B's arrivals, H reconstructed
    from the balance: {"flows", "hidden": H's id, "level": H's reconstructed level, "rec", "why"}
    (flows None with the reason when the pair is not a hidden stock). opts: "mb_fit" ("kalman": the
    filter of balance_filter.py, the default; "joint": the plain balance with B's own drain refitted;
    "simple": the plain balance alone), and the limits of plausible()."""
    opts = opts or {}
    ids = [s["id"] for s in stocks]
    t = run["t"]
    A = levels_array(run["levels"], ids)
    send, recv = senders(flows, a, b), arrivals(flows, a, b)
    if not send or not recv:
        return None
    kept_idx = [i for i in range(len(flows)) if i not in set(send) | set(recv)]
    kept = [flows[i] for i in kept_idx]
    F = amounts([flows[i] for i in send], ids, t, A).sum(axis=0)
    dyb = np.diff(A[ids.index(b)])
    rec = reconstruct(F, dyb - net_effect(kept, ids, t, A, b))
    if rec is None:
        return {"flows": None, "why": "degenerate"}
    scale_b = {}
    dr = [i for i in drains(flows, b) if i in kept_idx]
    kb0 = sum(drain_k(flows[i]) for i in dr)
    how = opts.get("mb_fit", "kalman")
    refit = bool(dr) and kb0 > 0
    other = net_effect([flows[i] for i in kept_idx if i not in dr], ids, t, A, b) if refit else net_effect(kept, ids, t, A, b)
    fitted = None
    if how == "kalman":
        fitted = balance_filter.fit(F, dyb - other, A[ids.index(b)], other, kb0 if refit else 0.0, refit_kb=refit)
    elif how == "joint" and refit:
        fitted = reconstruct_joint(F, dyb - other, A[ids.index(b)], other, rec["d"], kb0)
    if fitted is not None:
        rec = fitted
        if refit:
            scale_b = {i: fitted["kb"] / kb0 for i in dr}
    why = plausible(rec, len(F), **{k: opts[k] for k in ("max_alpha", "max_residence", "below") if k in opts})
    if why:
        return {"flows": None, "why": why, "rec": rec}
    alpha = min(1.0, rec["alpha"])
    h = f"{a}~{b}"
    base = [scale_flow(flows[i], scale_b[i]) if i in scale_b else flows[i] for i in kept_idx]
    into_h = [scale_flow(flows[i], alpha, to=h, hidden=h) for i in send]
    out_h = {"from": h, "to": b, "ratio": 1, "rate": {"form": "proportional", "stock": h, "k": round(rec["d"], 4)}, "hidden": h}
    rest = [scale_flow(flows[i], 1 - alpha) for i in send] if 1 - alpha > 0.01 else []
    out = {"flows": base + into_h + rest + [out_h], "hidden": h, "level": np.maximum(0.0, rec["H"]).tolist(), "rec": rec, "why": None}
    if rest:
        # A loses only what H takes (as conservation reads a drain off what receivers take); A's
        # outside inflow in pieces was measured beside the larger loss, so it is measured again
        out["alt"] = [remeasured(f, (1 - alpha) * F, t) if f.get("to") == a and f.get("from") is None and f.get("inflow") and not f.get("driver") else f
                      for f in base] + into_h + [out_h]
    return out


def remeasured(f: dict, less: np.ndarray, t: list) -> dict:
    """An outside inflow in pieces, each piece lowered by the mean of what its stock no longer loses
    over it (less: per step, step k from time k to k + 1, so it counts at time t[k + 1])."""
    tt = np.asarray(t, dtype=float)[1:]
    pieces = []
    for g in f["inflow"]:
        m = (tt >= g["from"]) & (tt <= g["to"])
        pieces.append({**g, "value": float(g["value"]) - (float(less[m].mean()) if m.any() else 0.0)})
    return {**f, "inflow": pieces}
