# Vendored for kalman_lna from sdlab/discovery/strategies/censored/floor.py at commit ec2f9e8 (imports rewritten to the
# vendored copies; otherwise unchanged). Do not edit: re-vendor instead.
"""A rate floored at zero, found by a threshold search on one row.

SD Lab's engine floors every rate at 0, so a link c + k x_j with k < 0 (or a drain that would go
negative) stops at zero while x_j is past -c / k. On those steps the row's change is the rest of its
flows alone; a straight line through all steps bends toward them. Here the row's change, less the
flows kept from the main line (moves), is fitted as

    a x max(0, s (x_j - theta))  +  b x own level  +  sum_m c_m x level_m  (+ a constant)

for every reader j, side s and threshold theta (a grid over x_j's range, then refined), by least
squares. A hinge whose floor never binds (or always does) is not a floor and is not offered; the
caller also asks it to beat the same row with x_j as a straight line and as a cubic (a smooth bend,
such as an effect or a delay, is not a floor). The
fit is read back as flows: the hinge as a floored linear rate into the stock (a > 0) or out of it
(a < 0), the levels as drains and links, the constant as an outside inflow or outflow.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar


def _fit(y: np.ndarray, cols: list[np.ndarray]) -> tuple[np.ndarray, float]:
    X = np.array(cols).T
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ beta
    return beta, float(r @ r)


def _projector(cols: list[np.ndarray]):
    """Residualising against the columns: v -> v less its least-squares fit on them."""
    if not cols:
        return lambda v: v
    Q, _ = np.linalg.qr(np.array(cols).T)
    return lambda v: v - Q @ (Q.T @ v)


def search(y: np.ndarray, prev: np.ndarray, i: int, others: list[int], grid: int = 48, bind: tuple[float, float] = (0.03, 0.97),
           constant: bool = False, readers: list[int] | None = None) -> list[dict]:
    """The best hinge per (reader, side): {j, s, theta, beta (hinge, own, others..., const?), rss, binds,
    line_rss, smooth_rss}. Each threshold's rss comes from the hinge column with the other columns
    projected out (all thresholds of a grid at once), so the search costs one QR per reader."""
    S, T = prev.shape
    out = []
    for j in range(S) if readers is None else readers:
        x = prev[j]
        if np.ptp(x) <= 0:
            continue
        rest = [prev[i]] + [prev[m] for m in others if m != j and m != i]
        if constant:
            rest.append(np.ones(T))
        proj = _projector(rest)
        yr = proj(y)
        base = float(yr @ yr)
        qs = np.unique(np.quantile(x, np.linspace(bind[0], bind[1], grid)))
        for s in (1.0, -1.0):
            def rss_of(H):
                Hr = proj(H)
                den = (Hr * Hr).sum(axis=0)
                num = (Hr.T @ yr) ** 2
                return np.where(den > 1e-12 * T, base - num / np.where(den > 0, den, 1.0), np.inf)

            H = np.maximum(0.0, s * (x[:, None] - qs[None, :]))
            share = (H == 0).mean(axis=0)
            ok = (share >= bind[0]) & (share <= bind[1]) & H.any(axis=0)
            if not ok.any():
                continue
            r = np.where(ok, rss_of(H), np.inf)
            k = int(np.argmin(r))
            if not np.isfinite(r[k]):
                continue
            lo, hi = qs[max(0, k - 1)], qs[min(len(qs) - 1, k + 1)]
            th = float(qs[k])
            if hi > lo:
                res = minimize_scalar(lambda v: float(rss_of(np.maximum(0.0, s * (x - v))[:, None])[0]), bounds=(lo, hi),
                                      method="bounded", options={"xatol": 1e-6 * max(1.0, abs(th))})
                if res.fun < r[k]:
                    th = float(res.x)
            h = np.maximum(0.0, s * (x - th))
            beta, rss = _fit(y, [h] + rest)
            # the same row with x_j as a straight line (and a constant): what the hinge must beat
            line = _fit(y, [x] + rest + ([] if constant else [np.ones(T)]))[1]
            # and a smooth curve in x_j (a cubic): a floor is a kink, not any bend
            z = (x - x.mean()) / (x.std() or 1.0)
            smooth = _fit(y, [z, z * z, z ** 3] + rest + ([] if constant else [np.ones(T)]))[1]
            out.append({"j": j, "s": s, "theta": th, "beta": beta, "rss": rss, "line_rss": line, "smooth_rss": smooth, "binds": float((h == 0).mean()),
                        "others": [m for m in others if m != j and m != i], "constant": constant})
    return out


def _r4(v: float) -> float:
    a = abs(v)
    return round(float(v), 2 if a >= 10 else 4 if a >= 1 else 6)


def _flow(src, dst, rate: dict, **extra) -> dict:
    return {"from": src, "to": dst, "ratio": 1, "rate": rate, "rates": [rate], "nearly": [], "bothWays": False, **extra}


def as_flows(fit: dict, ids: list[str], tol: float = 1e-9) -> list[dict]:
    """A hinge fit read back as flows of the row's stock (ids[i] is the row)."""
    beta, j, s, th = fit["beta"], fit["j"], fit["s"], fit["theta"]
    i = fit["i"]
    a, own, rest = float(beta[0]), float(beta[1]), [float(b) for b in beta[2:]]
    row, reader = ids[i], ids[j]
    out = []
    k, c = abs(a) * s, -abs(a) * s * th                    # |a| max(0, s (x - th)) = max(0, c + k x)
    rate = {"form": "linear", "c": _r4(c), "terms": [{"stock": reader, "k": _r4(k)}]}
    if a > tol:
        out.append(_flow(None, row, rate, driver=reader, floor=reader, threshold_params=0))
    elif a < -tol:
        out.append(_flow(row, None, rate, driver=reader, floor=reader, threshold_params=0))
    if own < -tol:
        out.append(_flow(row, None, {"form": "proportional", "stock": row, "k": _r4(-own)}))
    elif own > tol:
        out.append(_flow(None, row, {"form": "linear", "c": 0, "terms": [{"stock": row, "k": _r4(own)}]}, driver=row))
    for m, b in zip(fit["others"], rest[: len(fit["others"])]):
        if b > tol:
            out.append(_flow(None, row, {"form": "linear", "c": 0, "terms": [{"stock": ids[m], "k": _r4(b)}]}, driver=ids[m]))
        elif b < -tol:
            out.append(_flow(row, None, {"form": "linear", "c": 0, "terms": [{"stock": ids[m], "k": _r4(-b)}]}, driver=ids[m]))
    if fit["constant"]:
        b = rest[-1]
        if b > tol:
            out.append(_flow(None, row, {"form": "constant", "value": _r4(b)}))
        elif b < -tol:
            out.append(_flow(row, None, {"form": "constant", "value": _r4(-b)}))
    return out
