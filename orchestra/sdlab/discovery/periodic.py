"""Outside inflows as a repeating profile: one period's shape, shared by every period, plus a slow level.

A stock's outside inflow in pieces pays one free value per piece over the whole run, so the pieces can
follow anything, including dynamics the levels should explain. Operational inflows mostly repeat (a
day, a week, a year): a profile of a few phase segments, the same in every period, has a fixed, small
number of parameters, cannot follow what does not repeat, and says what the next period brings.

    find(stocks, run, opts) -> {stock id: {"P", "segments", "series", ...}}

For each stock: candidate periods from the autocorrelation of what its levels leave unexplained
(pooled over stocks), each period's profile found by optimal partitioning of the phase-folded
residual (one change costs a value and a position), a slow component (changes at least one period
apart, at twice the usual price) beside it, and the levels refitted jointly with both. The profile is
kept only if the row with it beats the row with pieces (and with a constant) by BIC, counting the
profile's values, its cut positions and the period.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit

from .changes import pelt
from .regression import Sums, fit_subset, row_sums, stepwise


@njit(cache=True)
def _partition(m: np.ndarray, w: np.ndarray, pen: float, min_len: int) -> np.ndarray:
    """Optimal partitioning of the weighted series m (weights w) into pieces of constant (weighted)
    mean, each piece costing pen; returns the last cut before each end."""
    n = m.shape[0]
    pw = np.zeros(n + 1)
    pm = np.zeros(n + 1)
    pm2 = np.zeros(n + 1)
    for i in range(n):
        pw[i + 1] = pw[i] + w[i]
        pm[i + 1] = pm[i] + w[i] * m[i]
        pm2[i + 1] = pm2[i] + w[i] * m[i] * m[i]
    f = np.empty(n + 1)
    last = np.zeros(n + 1, np.int64)
    f[0] = -pen
    for t in range(1, n + 1):
        best = np.inf
        arg = 0
        for s in range(0, t):
            if t - s < min_len and not (s == 0 and t == n):
                continue
            ww = pw[t] - pw[s]
            if ww <= 0:
                continue
            sm = pm[t] - pm[s]
            v = f[s] + (pm2[t] - pm2[s] - sm * sm / ww) + pen
            if v < best:
                best = v
                arg = s
        if best == np.inf:
            best = 1e300
        f[t] = best
        last[t] = arg
    return last


def fold(u: np.ndarray, P: int) -> tuple[np.ndarray, np.ndarray]:
    """The mean of u at each phase of period P, and how many steps each phase has."""
    ph = np.arange(u.shape[0]) % P
    n = np.bincount(ph, minlength=P).astype(float)
    return np.bincount(ph, weights=u, minlength=P) / np.maximum(n, 1), n


def segment_profile(u: np.ndarray, P: int, sigma2: float, mult: float = 2.0) -> list[tuple[int, int]]:
    """The phase segments [a, b) (phases modulo P, a segment may wrap past P) of the period-P profile
    of u: optimal partitioning of the fold means weighted by their counts, each segment costing mult x
    log T x sigma2 (its value and its position)."""
    m, n = fold(u, P)
    if P == 1:
        return [(0, 1)]
    # start the circle at the largest jump, so that a segment rarely wraps
    r = int(np.argmax(np.abs(m - np.roll(m, 1))))
    mr, nr = np.roll(m, -r), np.roll(n, -r)
    last = _partition(mr, nr, mult * math.log(u.shape[0]) * sigma2, 1)
    cuts, t = [], P
    while t > 0:
        cuts.append(t)
        t = int(last[t])
    cuts = [0] + cuts[::-1]
    return [((a + r) % P, (b + r) % P or P) for a, b in zip(cuts[:-1], cuts[1:])]


def phase_labels(T: int, P: int, segs: list[tuple[int, int]]) -> np.ndarray:
    """Which profile segment each step is in."""
    lab = np.empty(P, dtype=int)
    for i, (a, b) in enumerate(segs):
        if a < b:
            lab[a:b] = i
        else:                                            # wraps past the end of the period
            lab[a:] = i
            lab[:b] = i
    return lab[np.arange(T) % P]


def fold_scores(u: np.ndarray) -> np.ndarray:
    """For every period P (index P, up to T/3): how much more of u the phase-folded means explain
    than chance would, in standard deviations (the fold's sum of squares over the noise variance is
    about chi-squared with P - 1 degrees of freedom without a period). From the autocovariance: the
    sum over a phase is the sum of the autocovariances at multiples of P."""
    T = u.shape[0]
    hi = T // 3
    out = np.zeros(hi + 1)
    if hi < 2:
        return out
    u = u - u.mean()
    s2 = float(np.sum(np.diff(u) ** 2)) / (2 * max(1, T - 1))
    if s2 <= 0:
        return out
    nfft = 1 << (2 * T - 1).bit_length()
    F = np.fft.rfft(u, nfft)
    R = np.fft.irfft(F * np.conj(F), nfft)[:T]
    for P in range(2, hi + 1):
        S = R[0] + 2 * R[P::P].sum()
        ss = S * P / T
        out[P] = (ss / s2 - (P - 1)) / math.sqrt(2 * (P - 1))
    return out


def candidates(series: list[np.ndarray], top: int = 4) -> list[int]:
    """Candidate periods (2 .. T/3): for each series, the sharpest peaks of its fold score (a true
    period explains much more than its neighbours P - 1 and P + 1 do; a slow drift favours every long
    period alike), and the divisors of each that are peaks too (a profile repeats at every multiple of
    its period, and noise decides which multiple ranks highest)."""
    out: list[int] = []
    for u in series:
        z = fold_scores(u)
        hi = z.shape[0] - 1
        if hi < 3:
            continue
        P = np.arange(2, hi)
        sharp = z[P] - 0.5 * (z[P - 1] + z[P + 1])
        ok = (z[P] > 3) & (sharp > 3) & (z[P] >= z[P - 1]) & (z[P] >= z[P + 1])
        peaks = [int(p) for p in P[ok]]
        best = sorted(peaks, key=lambda p: -(z[p] - 0.5 * (z[p - 1] + z[p + 1])))[:top]
        for p in best:
            for q in [p] + [p // k for k in range(2, p // 2 + 1) if p % k == 0]:
                if q in peaks and q not in out:
                    out.append(q)
    return out


class Design:
    """The inflow's columns, held as labels: one column per profile segment (which segment each step
    is in) and one per slow piece after the first (the profile's columns already hold the overall
    level). Least squares on them is done from counts, without forming the columns."""

    def __init__(self, T: int, P: int, segs: list[tuple[int, int]], slow: list[int]):
        self.lp = phase_labels(T, P, segs)
        self.k = len(segs)
        self.ls = np.repeat(np.arange(len(slow) - 1), np.diff(slow))
        self.m = len(slow) - 1
        k, m = self.k, self.m
        n = k + m - 1
        A = np.zeros((n, n))
        A[:k, :k] = np.diag(np.bincount(self.lp, minlength=k).astype(float))
        if m > 1:
            cross = np.bincount(self.lp * m + self.ls, minlength=k * m).reshape(k, m)[:, 1:].astype(float)
            A[:k, k:] = cross
            A[k:, :k] = cross.T
            A[k:, k:] = np.diag(np.bincount(self.ls, minlength=m)[1:].astype(float))
        self.inv = np.linalg.pinv(A)
        self.cols = n

    def _dt(self, a: np.ndarray) -> np.ndarray:
        a2 = np.atleast_2d(a)
        out = np.zeros((a2.shape[0], self.cols))
        for r in range(a2.shape[0]):
            out[r, : self.k] = np.bincount(self.lp, weights=a2[r], minlength=self.k)
            if self.m > 1:
                out[r, self.k:] = np.bincount(self.ls, weights=a2[r], minlength=self.m)[1:]
        return out

    def fitted(self, a: np.ndarray) -> np.ndarray:
        """a's least-squares fit on the columns (same shape as a)."""
        g = self._dt(a) @ self.inv.T
        out = g[:, : self.k][:, self.lp]
        if self.m > 1:
            slow = np.concatenate([np.zeros((g.shape[0], 1)), g[:, self.k:]], axis=1)
            out = out + slow[:, self.ls]
        return out.reshape(a.shape)


def refit(res: np.ndarray, fp: dict) -> np.ndarray:
    """The profile and slow level (fit's segments and slow cuts) measured again on res."""
    return Design(res.shape[0], fp["P"], fp["segments"], fp["slow"]).fitted(res)


def _sums(y, X, D: Design) -> Sums:
    yp, Xp = y - D.fitted(y), X - D.fitted(X)
    return Sums(G=Xp @ Xp.T, b=Xp @ yp, yy=float(yp @ yp), T=y.shape[0], pieces=D.cols)


def fit_periodic(y: np.ndarray, X: np.ndarray, allowed: list[int], start: list[int], P: int, add_gain: float, slow_mult: float = 2.0) -> dict:
    """The row y = levels + profile(period P) + slow component, fitted jointly: alternately the levels
    (stepwise, with the profile and slow pieces projected out) and the profile's segments and the slow
    pieces (in what the levels leave)."""
    T = y.shape[0]
    log_t = math.log(T)
    chosen = list(start)
    segs = [(0, P)]
    slow = [0, T]
    for _ in range(4):
        D = Design(T, P, segs, slow)
        sums = _sums(y, X, D)
        chosen = stepwise(sums, chosen, allowed, add_gain, 4 * X.shape[0])
        f = fit_subset(sums, chosen)
        res = y - (f.beta @ X[chosen] if chosen else 0)
        slow_part = np.zeros(T)
        if len(slow) > 2:
            for a, b in zip(slow[:-1], slow[1:]):
                slow_part[a:b] = res[a:b].mean()
            slow_part -= slow_part.mean()
        sig2 = float(np.sum(np.diff(res) ** 2)) / (2 * max(1, T - 1)) or 1.0
        nsegs = segment_profile(res - slow_part, P, sig2)
        lab2 = phase_labels(T, P, nsegs)
        seg_mean = np.bincount(lab2, weights=res - slow_part) / np.maximum(np.bincount(lab2), 1)
        nslow = pelt(res - seg_mean[lab2], P, slow_mult) if T >= 2 * P else [0, T]
        if nsegs == segs and nslow == slow:
            break
        segs, slow = nsegs, nslow
    D = Design(T, P, segs, slow)
    sums = _sums(y, X, D)
    f = fit_subset(sums, chosen)
    res = y - (f.beta @ X[chosen] if chosen else 0)
    series = D.fitted(res)
    rss = float(np.sum((res - series) ** 2))
    # the profile's values and positions, the slow pieces' values and positions, and the period
    p = len(chosen) + 2 * len(segs) + 2 * (len(slow) - 2) + 1
    return {"P": P, "segments": segs, "slow": slow, "set": chosen, "beta": f.beta, "series": series, "rss": rss,
            "bic": T * math.log(max(rss, 1e-12) / T) + p * log_t, "p": p}


def screen(u: np.ndarray, periods: list[int], keep: int) -> list[int]:
    """The periods whose profile alone (no slow part, the levels as they are) explains u best by BIC."""
    T = u.shape[0]
    sig2 = float(np.sum(np.diff(u) ** 2)) / (2 * max(1, T - 1)) or 1.0
    scored = []
    for P in periods:
        segs = segment_profile(u, P, sig2)
        if len(segs) < 2:
            continue
        lab = phase_labels(T, P, segs)
        mean = np.bincount(lab, weights=u) / np.maximum(np.bincount(lab), 1)
        rss = float(np.sum((u - mean[lab]) ** 2))
        scored.append((T * math.log(max(rss, 1e-12) / T) + (2 * len(segs) + 1) * math.log(T), P))
    return [P for _, P in sorted(scored)[:keep]]


def fit_pieces(y: np.ndarray, X: np.ndarray, allowed: list[int], start: list[int], add_gain: float, min_piece: int = 5) -> dict:
    """The main line's row: levels and inflow pieces, alternately (as matrix.identify_one's passes)."""
    T = y.shape[0]
    cuts, chosen = [0, T], list(start)
    for p in range(5):
        sums = row_sums(y, X, cuts)
        chosen = stepwise(sums, chosen, allowed, add_gain, 4 * X.shape[0])
        f = fit_subset(sums, chosen)
        res = y - (f.beta @ X[chosen] if chosen else 0)
        nxt = pelt(res, min_piece, 0.5 if p == 0 else 1)
        if nxt == cuts and p >= 2:
            break
        cuts = nxt
    f = fit_subset(row_sums(y, X, cuts), chosen)
    const = fit_subset(row_sums(y, X, [0, T]), chosen)
    bic = lambda rss, p: T * math.log(max(rss, 1e-12) / T) + p * math.log(T)  # noqa: E731
    return {"set": chosen, "cuts": cuts, "bic": bic(f.rss, len(chosen) + len(cuts) - 1), "bic_const": bic(const.rss, len(chosen) + 1)}


def find(stocks: list[dict], run: dict, opts: dict | None = None) -> dict:
    """Each stock whose outside inflow is better told as a repeating profile: {id: fit} (fit_periodic's,
    plus the pieces' BIC it beat)."""
    opts = opts or {}
    ids = [s["id"] for s in stocks]
    S = len(ids)
    t = run["t"]
    T = len(t) - 1
    if T < 60:
        return {}
    lv = np.nan_to_num(np.array([np.asarray(run["levels"].get(i, [0] * len(t)), dtype=float) for i in ids]))
    prev, dy = lv[:, :-1], np.diff(lv, axis=1)
    if opts.get("known"):
        from .matrix import known_changes
        dy = dy - known_changes(opts["known"], ids, len(t))[:, 1:]
    log_t = math.log(T)
    add_gain = opts.get("add_gain", log_t)
    final = {s["id"] for s in stocks if s.get("final")}
    group = {s["id"]: s.get("group") for s in stocks}

    def allowed_for(s):
        out = []
        for j in range(S):
            a, b = group[ids[j]], group[ids[s]]
            if (j == s or not a or not b or a == b) and not (ids[s] in final and j == s):
                out.append(j)
        return out

    # the main line's rows (levels and pieces), and what their levels leave unexplained
    base = []
    for s in range(S):
        al = allowed_for(s)
        pc = fit_pieces(dy[s], prev, al, [], add_gain, opts.get("min_piece", 5))
        f = fit_subset(row_sums(dy[s], prev, pc["cuts"]), pc["set"])
        base.append((al, pc, dy[s] - (f.beta @ prev[pc["set"]] if pc["set"] else 0)))
    periods = candidates([b[2] for b in base], opts.get("period_candidates", 4))
    periods = [P for P in periods if P <= T // 3]
    if opts.get("trace_periodic"):
        print("candidate periods", periods)
    out = {}
    for s in range(S):
        al, pc, u = base[s]
        best = None
        for P in screen(u, periods, opts.get("period_screen", 2)):
            fp = fit_periodic(dy[s], prev, al, pc["set"], P, add_gain)
            if len(fp["segments"]) < 2:
                continue
            if best is None or fp["bic"] < best["bic"]:
                best = fp
        if best is None:
            continue
        fp = {**best, "pieces_bic": pc["bic"], "const_bic": pc["bic_const"], "pieces": len(pc["cuts"]) - 1}
        ok = best["bic"] < min(pc["bic"], pc["bic_const"]) - log_t
        if ok:
            out[ids[s]] = fp
        if opts.get("trace_periodic"):
            print(f"  {ids[s]}{'' if ok else ' (not kept)'}: period {fp['P']}, {len(fp['segments'])} segments, {len(fp['slow']) - 1} slow pieces, BIC {fp['bic']:.0f} vs pieces {fp['pieces_bic']:.0f} ({fp['pieces']}) and constant {fp['const_bic']:.0f}")
    # in a system driven by a repeating input, every stock's residual repeats a little (its misfit
    # follows the driver): a profile is kept only where the period explains a good share of what the
    # strongest profile at that period explains (the driven stock, not its echoes downstream)
    ratio = opts.get("source_ratio", 0.25)
    if ratio:
        strongest: dict[int, float] = {}
        for fp in out.values():
            strongest[fp["P"]] = max(strongest.get(fp["P"], 0.0), fp["const_bic"] - fp["bic"])
        for i in [i for i, fp in out.items() if fp["const_bic"] - fp["bic"] < ratio * strongest[fp["P"]]]:
            if opts.get("trace_periodic"):
                print(f"  {i}: an echo of a stronger profile, not kept")
            del out[i]
    return out
