# A copy, one per strategy that uses it (sdlab/CLAUDE.md invariant 11: no strategy imports another);
# tests/test_oefit.py checks that the copies are identical.
"""Output error: where the levels carry measurement error, the found structure's numbers refitted by
running the model through the whole record, and its parts revised on that fit (k8s-306, 2026-10-21;
docs/research/k8s-306-r3-oefit.md).

A one-step fit reads each step's change off the measured levels. Under measurement error the same
error is in the regressors and in the change, so drains come out too slow or too fast, a capacity
limit (MIN) is flattened into a proxy, and the flows that feed counters (stocks that only grow, whose
5% error buries their step changes) are lost. The instrumented refit (eiv.py) corrects the numbers of a
structure only where its instruments are strong, and it cannot change the structure.

Here the model is run from fitted starting levels through the whole record, step by step as the engine
steps (rates floored at 0, a stock's outflows sharing what it holds, announced injections added before
their step), and its levels are compared with the measured levels: the residual is the measurement
error plus the drift of the process noise, and no measured level is a regressor. The numbers (and the
starting levels) are fitted by weighted least squares (the run's derivatives carried with it, _sens), each stock's weight from a noise variance
a + b x level^2 fitted to its own residuals (feasible GLS, two passes; again after each search, which
then runs again from its answer, up to CYCLES times).

On that fit the structure is revised by a beam search (each round keeps the BEAM best new structures,
so a path may pass through a worse one), scored by the weighted squared error plus PRICE x log n per
number (Bayes'; a capacity limit's extra number at MIN_PRICE x log n), over one change at a time:
- drop a part (a term, or a flow between measured stocks);
- write a flow's term on its own source X as a capacity limit a x MIN(X, theta x Y), for each other
  stock Y that is not a counter and each measured input (theta from quantiles of X / Y, then fitted);
- give a flow whose rate reads another measured stock Z (and not its source) Z as its source, or have
  it read its source instead; make a drain into a stock a drain of another stock into it;
- add a move X -> C into a counter C from a stock X that is not a counter (a drain the counters
  receive), or a move between two stocks that move (neither a counter; "moves": beyond its announced
  injections and its noise);
- put back a part of the found structure that a revision dropped;
- give an outside inflow one more term on a measured level that is not a counter's, or an input (a link:
  k8s-307, docs/research/k8s-307-c1.md; a one-step fit on noisy levels picks a noise-free input in place
  of a noisy stock, and the instrumented refit then prunes the row).
A move (a flow between two measured stocks) is added or re-ended only between stocks of the same
announced unit: a flow carries one unit (requests do not pour into millicores; what crosses units is a
link). Without this a big noisy stock with an estimated outside inflow and outflow becomes a hub that
any misfit can be routed through.
The search runs from two seeds and keeps the one with the lower Gaussian -2 log-likelihood (each with
its own noise variances) plus price: the found structure, and the found structure with every such move
added (and every link on each outside inflow) and then thinned by the Wald statistic of its fit (the part furthest below its price dropped, one
fit a step). A strategy whose own fit gave little (an empty reading) is rebuilt from the second. The
search stops at a budget of simulated steps (`budget`; so the answer does not depend on the machine's
speed or load) with the best answer so far; a wall-clock cap (WALL) is only a safety net.
No outside flow is added or removed (the declared boundary), and an outside inflow fixed to a declared
input keeps its number (an outflow that reads an input is the fit's guess, and is revised). The same rule
for every case: nothing reads the case or its truth.

The output error is blind to the process noise, so two things are then read on a Kalman likelihood (an
extended Kalman filter with counting noise in every flow and measurement error a + b x level^2 in every
level, the noise fitted with the numbers; k8s-307):
- the structure: the best CHOICES structures the searches fitted are each fitted on it, and the lowest
  -2 log-likelihood plus log n per number is kept (the output error cannot tell a loop that empties a
  stock each step from a slower drain of a smaller inflow, which give the same levels on average);
- the numbers of the rates that read another stock, and of the drains of the stocks they feed: the
  output error runs such a rate on the other stock's simulated level, which the process noise moves off
  its measured path (an errors-in-variables of the simulation).
A found model with pieces or effects (forms this refit cannot run) is searched from the same flows with
those written as plain rates of the same mean amount, and judged against the found model as the shared
engine runs it; one with eras, co-flows or hidden stocks is left as it is.

It runs only when the data show measurement error (the one-step residuals of the found model are
negatively correlated from one step to the next beyond chance, eiv.white_z), and only on structures
it can run: constants, proportional and linear rates on measured stocks and inputs, and the offered
capacity series (capmin.py), and an expression of constants, terms and MIN(a x X, b x Y). The revised
model replaces the found one only when its output error is lower.
"""

from __future__ import annotations

import copy
import math
import time

import numpy as np
from numba import njit
from scipy.optimize import least_squares, minimize
from scipy.stats import norm

PRICE = 1.0            # a number's price, in log n (n: measured points; Bayes')
ALPHA = 1e-3           # the white-noise test's level (as eiv.py)
ROUNDING = 1e-3        # a stock whose one-step residuals are below this share of its changes (rms) is exact
ROUNDS = 12            # search rounds at most
BEAM = 8               # structures kept each round (the best of the new ones, better or not)
BUDGET = 4e8           # simulated steps the search may spend (the found seed first, then the wide one)
OVERHEAD = 700         # a run's fixed cost, in steps (so the budget is about the same time at any length)
WALL = 180.0           # seconds: a safety net only (the budget above is what normally stops the search)
FOUND_SHARE = 0.6      # the share of the budget the search from the found structure may use
CYCLES = 3             # searches at most, each with the noise read off the model the last one found
PATIENCE = 4           # rounds without a better structure before the search stops
GRID = 12              # theta candidates for a capacity limit
MIN_PRICE = 6.0        # a capacity limit's extra number costs this, in log n (as capmin.py's PRICE)
MIN_SHARE = 0.05       # each regime of a capacity limit holds on at least this share of the steps
NFEV = 60              # function evaluations of a full fit (times the numbers)
ADD = 0.05             # a new move's first rate (a share of its source a step)
MOVES = 10.0           # a stock moves when its level varies this many times more than its noise
POINTS = 3000          # compared levels at most (a long run is compared every k-th step)
QUICK = 10             # function evaluations of a candidate's first fit (its rank in the round)
CAP = "capmin__"       # capmin.py's offered series prefix
JAC_COST = 4.0         # a run with its derivatives costs 1 + (numbers / JAC_COST) runs (measured: 9 runs at 33)
KF_ITER = 60           # iterations of a Kalman fit of a structure's numbers
CHOICES = 4            # the best structures of the searches judged on the Kalman likelihood


@njit(cache=True)
def _sim(n, S, start, inj, U, fsrc, fdst, fc, tf, tv, tk, mf, mx, my, mth, ma):
    """Levels (S x n) of the model run from start: each step adds its injections, then every flow's
    amount (c + sum k v + sum a min(x, theta y), floored at 0; outflows scaled to what the source
    holds), levels floored at 0. Variables v: the levels (0..S-1), then the inputs at the step."""
    F = fsrc.shape[0]
    X = np.zeros((S, n))
    L = start.copy()
    for s in range(S):
        if L[s] < 0:
            L[s] = 0.0
        X[s, 0] = L[s]
    v = np.zeros(S + U.shape[0])
    amt = np.zeros(F)
    for i in range(1, n):
        for s in range(S):
            L[s] = L[s] + inj[s, i]
            if L[s] < 0:
                L[s] = 0.0
            v[s] = L[s]
        for u in range(U.shape[0]):
            v[S + u] = U[u, i]
        for f in range(F):
            amt[f] = fc[f]
        for j in range(tf.shape[0]):
            amt[tf[j]] += tk[j] * v[tv[j]]
        for j in range(mf.shape[0]):
            a, b = v[mx[j]], mth[j] * v[my[j]]
            amt[mf[j]] += ma[j] * (a if a < b else b)
        for f in range(F):
            if amt[f] < 0:
                amt[f] = 0.0
        for s in range(S):
            want = 0.0
            for f in range(F):
                if fsrc[f] == s:
                    want += amt[f]
            if want > L[s] and L[s] > -1:
                k = L[s] / want if want > 0 else 1.0
                for f in range(F):
                    if fsrc[f] == s:
                        amt[f] *= k
        for f in range(F):
            if fsrc[f] >= 0:
                L[fsrc[f]] -= amt[f]
            if fdst[f] >= 0:
                L[fdst[f]] += amt[f]
        for s in range(S):
            if L[s] < 0:
                L[s] = 0.0
            X[s, i] = L[s]
    return X


@njit(cache=True)
def _ekf(n, S, A, inj, U, noise, fsrc, fdst, fc, tf, tv, tk, mf, mx, my, mth, ma):
    """-2 log-likelihood of the measured levels A (S x n, less the first) under the model as an extended
    Kalman filter: counting noise in every flow (variance = its amount, as Poisson counts) and measurement
    error of variance noise[s] + noise[S + s] x level^2 in every level (the same steps as _sim: injections
    first, rates floored at 0, outflows scaled to what the source holds)."""
    F = fsrc.shape[0]
    x = A[:, 0].copy()
    P = np.zeros((S, S))
    for s in range(S):
        P[s, s] = max(noise[s] + noise[S + s] * A[s, 0] ** 2, 1e-9)
    v = np.zeros(S + U.shape[0])
    amt = np.zeros(F)
    dr = np.zeros((F, S))
    Fm = np.zeros((S, S))
    FP = np.zeros((S, S))
    Sm = np.zeros((S, S))
    Lc = np.zeros((S, S))
    K = np.zeros((S, S))
    nu = np.zeros(S)
    w = np.zeros(S)
    out = 0.0
    for i in range(1, n):
        for s in range(S):
            x[s] += inj[s, i]
            v[s] = x[s]
        for u in range(U.shape[0]):
            v[S + u] = U[u, i]
        for f in range(F):
            amt[f] = fc[f]
            for s in range(S):
                dr[f, s] = 0.0
        for j in range(tf.shape[0]):
            amt[tf[j]] += tk[j] * v[tv[j]]
            if tv[j] < S:
                dr[tf[j], tv[j]] += tk[j]
        for j in range(mf.shape[0]):
            a, b = v[mx[j]], mth[j] * v[my[j]]
            if a < b:
                amt[mf[j]] += ma[j] * a
                if mx[j] < S:
                    dr[mf[j], mx[j]] += ma[j]
            else:
                amt[mf[j]] += ma[j] * b
                if my[j] < S:
                    dr[mf[j], my[j]] += ma[j] * mth[j]
        for f in range(F):
            if amt[f] < 0:
                amt[f] = 0.0
                for s in range(S):
                    dr[f, s] = 0.0
        for s in range(S):
            want = 0.0
            for f in range(F):
                if fsrc[f] == s:
                    want += amt[f]
            if want > x[s] and want > 0 and x[s] > -1:
                q = max(0.0, x[s]) / want
                for f in range(F):
                    if fsrc[f] == s:
                        amt[f] *= q
                        for k in range(S):
                            dr[f, k] = 0.0
                        if x[s] > 0:
                            dr[f, s] = amt[f] / x[s]
        for s in range(S):
            for k in range(S):
                Fm[s, k] = 1.0 if s == k else 0.0
        for f in range(F):
            for k in range(S):
                if fsrc[f] >= 0:
                    Fm[fsrc[f], k] -= dr[f, k]
                if fdst[f] >= 0:
                    Fm[fdst[f], k] += dr[f, k]
        for s in range(S):
            for k in range(S):
                acc = 0.0
                for m in range(S):
                    acc += Fm[s, m] * P[m, k]
                FP[s, k] = acc
        for s in range(S):
            for k in range(s + 1):
                acc = 0.0
                for m in range(S):
                    acc += FP[s, m] * Fm[k, m]
                P[s, k] = acc
                P[k, s] = acc
        for f in range(F):
            a, b, q = fsrc[f], fdst[f], amt[f]
            if a >= 0:
                x[a] -= q
                P[a, a] += q
            if b >= 0:
                x[b] += q
                P[b, b] += q
            if a >= 0 and b >= 0:
                P[a, b] -= q
                P[b, a] -= q
        for s in range(S):
            if not np.isfinite(x[s]) or abs(x[s]) > 1e12:
                return 1e300
            nu[s] = A[s, i] - x[s]
            for k in range(S):
                Sm[s, k] = P[s, k]
            Sm[s, s] += max(noise[s] + noise[S + s] * x[s] ** 2, 1e-9)
        for s in range(S):
            for k in range(s + 1):
                acc = Sm[s, k]
                for m in range(k):
                    acc -= Lc[s, m] * Lc[k, m]
                if s == k:
                    if acc <= 0.0:
                        return 1e300
                    Lc[s, s] = math.sqrt(acc)
                else:
                    Lc[s, k] = acc / Lc[k, k]
        for s in range(S):
            out += 2.0 * math.log(Lc[s, s])
        for s in range(S):
            acc = nu[s]
            for m in range(s):
                acc -= Lc[s, m] * w[m]
            w[s] = acc / Lc[s, s]
        for s in range(S):
            out += w[s] * w[s]
        for c in range(S):                       # K' = Sm^-1 P, column by column
            for s in range(S):
                acc = P[s, c]
                for m in range(s):
                    acc -= Lc[s, m] * FP[m, c]
                FP[s, c] = acc / Lc[s, s]
            for s in range(S - 1, -1, -1):
                acc = FP[s, c]
                for m in range(s + 1, S):
                    acc -= Lc[m, s] * K[c, m]
                K[c, s] = acc / Lc[s, s]
        for s in range(S):
            acc = 0.0
            for m in range(S):
                acc += K[s, m] * nu[m]
            w[s] = acc
        for s in range(S):
            x[s] += w[s]
        for s in range(S):
            for k in range(S):
                acc = 0.0
                for m in range(S):
                    acc += K[s, m] * P[m, k]
                FP[s, k] = P[s, k] - acc
        for s in range(S):
            for k in range(s + 1):
                q = 0.5 * (FP[s, k] + FP[k, s])
                P[s, k] = q
                P[k, s] = q
    return out


def _noise0(net: _Net, d: _Data) -> np.ndarray:
    """A first measurement noise per stock, a + b x level^2, from half the squared misfit of the changes
    of the model's run (the Kalman fit then refines it with the numbers)."""
    S = d.S
    e = np.diff(d.A, axis=1) - np.diff(_run(net, d, d.A[:, 0]), axis=1)
    out = np.zeros(2 * S)
    for s in range(S):
        V = np.column_stack([np.ones(d.n - 1), d.A[s, 1:] ** 2])
        ab = np.linalg.lstsq(V, 0.5 * e[s] ** 2, rcond=None)[0]
        out[s] = max(float(ab[0]), 1e-6 * max(1e-9, float(np.var(d.A[s]))), 1e-9)
        out[S + s] = max(float(ab[1]), 1e-10)
    return out


def _kf_fit(net: _Net, d: _Data, idx: list[int] | None = None, noise: np.ndarray | None = None) -> tuple[_Net, float]:
    """The numbers in slots idx (all where None) and the measurement noise fitted on the Kalman likelihood
    of every measured level (_ekf); (net with those numbers, -2 log-likelihood). A fit that ends above its
    start keeps the start."""
    S = d.S
    noise = _noise0(net, d) if noise is None else noise
    pall = net.get()
    idx = list(range(len(pall))) if idx is None else idx
    lo, hi = net.bounds()
    lo, hi, p0 = lo[idx], hi[idx], pall[idx]
    sc = np.maximum(np.abs(p0), 1e-3)
    g = net.copy()

    def f(z):
        p = pall.copy()
        p[idx] = z[:len(p0)] * sc
        g.put(p)
        try:
            val = _ekf(d.n, S, d.A, d.inj, d.U, np.exp(z[len(p0):]), *g.arrays())
        except Exception:                                # noqa: BLE001
            return 1e300
        return float(val) if np.isfinite(val) else 1e300

    z0 = np.concatenate([p0 / sc, np.log(noise)])
    bnds = [(a / s if np.isfinite(a) else None, b / s if np.isfinite(b) else None) for a, b, s in zip(lo, hi, sc)]
    bnds += [(-28.0, 40.0)] * (2 * S)
    z0 = np.array([min(max(z, b[0] if b[0] is not None else -np.inf), b[1] if b[1] is not None else np.inf)
                   for z, b in zip(z0, bnds)])
    f0 = f(z0)
    sol = minimize(f, z0, method="L-BFGS-B", bounds=bnds, options={"maxiter": KF_ITER, "eps": 1e-7})
    z, v = (sol.x, float(sol.fun)) if sol.fun < f0 and np.isfinite(sol.fun) else (z0, f0)
    out = net.copy()
    p = pall.copy()
    p[idx] = z[:len(p0)] * sc
    out.put(p)
    return out, v


def _kf_numbers(net: _Net, d: _Data) -> tuple[_Net, dict]:
    """The numbers of the rates that read another stock (and of the drains of the stocks they feed)
    refitted on the Kalman likelihood of every measured level (counting noise in the flows, measurement
    error in the levels, its variances fitted with them): the output error runs such a rate on the other
    stock's simulated level, which the process noise moves off the measured path, so its number is biased
    (an errors-in-variables of the simulation); the filter reads each step from the levels as measured and
    as predicted, weighted by their noises. The other numbers (a drain on its own source, which the
    output error reads without that bias) are held."""
    S = d.S
    reads = [fi for fi, h in enumerate(net.flows) if not h["fixed"] and h["dst"] >= 0
             and any(v < S and v != h["src"] for v in [v for v, _ in h["terms"]] + [m[k] for m in h["mins"] for k in (0, 1)])]
    rows = {net.flows[fi]["dst"] for fi in reads}
    free_f = set(reads) | {fi for fi, h in enumerate(net.flows) if not h["fixed"] and h["src"] in rows
                           and [v for v, _ in h["terms"]] == [h["src"]] and not h["mins"]}
    idx = [j for j, (fi, _, _) in enumerate(net.slots()) if fi in free_f]
    if not idx:
        return net, {"kf": "no rate reads another stock"}
    out, v = _kf_fit(net, d, idx)
    return out, {"kf": "refitted", "kf_nll": round(v, 1)}


def _kf_choice(cands: list[_Net], d: _Data, price: float) -> tuple[int, list[float]]:
    """Which of the search's best structures the Kalman likelihood (all numbers and the noise fitted)
    plus price per number prefers: the output error, blind to the process noise, cannot tell a loop that
    empties a stock each step from a slower drain of a smaller inflow (the same levels on average); the
    filter's one-step predictions can. (The index, and each one's price.)"""
    prices = []
    for g in cands:
        _, v = _kf_fit(g, d)
        prices.append(v + price * _numbers(g))
    return int(np.argmin(prices)), [round(x, 1) for x in prices]


@njit(cache=True)
def _sens(n, S, start, inj, U, fsrc, fdst, fc, tf, tv, tk, mf, mx, my, mth, ma, qc, qk, qa, qth, P):
    """_sim's levels (S x n) and their derivatives (S x n x P) by the numbers (positions qc, qk, qa, qth;
    -1: held) and the starting levels (the last S positions): the same steps, differentiated (a floor
    at 0 or a MIN's kink taken on the side the step is on)."""
    F = fsrc.shape[0]
    V = S + U.shape[0]
    X = np.zeros((S, n))
    J = np.zeros((S, n, P))
    L = start.copy()
    dL = np.zeros((S, P))
    for s in range(S):
        dL[s, P - S + s] = 1.0
        if L[s] < 0:
            L[s] = 0.0
            dL[s, P - S + s] = 0.0
        X[s, 0] = L[s]
        for q in range(P):
            J[s, 0, q] = dL[s, q]
    v = np.zeros(V)
    dv = np.zeros((V, P))
    amt = np.zeros(F)
    da = np.zeros((F, P))
    for i in range(1, n):
        for s in range(S):
            L[s] = L[s] + inj[s, i]
            if L[s] < 0:
                L[s] = 0.0
                for q in range(P):
                    dL[s, q] = 0.0
            v[s] = L[s]
            for q in range(P):
                dv[s, q] = dL[s, q]
        for u in range(U.shape[0]):
            v[S + u] = U[u, i]
        for f in range(F):
            amt[f] = fc[f]
            for q in range(P):
                da[f, q] = 0.0
            if qc[f] >= 0:
                da[f, qc[f]] = 1.0
        for j in range(tf.shape[0]):
            f = tf[j]
            amt[f] += tk[j] * v[tv[j]]
            if tv[j] < S:
                for q in range(P):
                    da[f, q] += tk[j] * dv[tv[j], q]
            if qk[j] >= 0:
                da[f, qk[j]] += v[tv[j]]
        for j in range(mf.shape[0]):
            f = mf[j]
            a, b = v[mx[j]], mth[j] * v[my[j]]
            if a < b:
                amt[f] += ma[j] * a
                if mx[j] < S:
                    for q in range(P):
                        da[f, q] += ma[j] * dv[mx[j], q]
                if qa[j] >= 0:
                    da[f, qa[j]] += a
            else:
                amt[f] += ma[j] * b
                if my[j] < S:
                    for q in range(P):
                        da[f, q] += ma[j] * mth[j] * dv[my[j], q]
                if qa[j] >= 0:
                    da[f, qa[j]] += b
                if qth[j] >= 0:
                    da[f, qth[j]] += ma[j] * v[my[j]]
        for f in range(F):
            if amt[f] < 0:
                amt[f] = 0.0
                for q in range(P):
                    da[f, q] = 0.0
        for s in range(S):
            want = 0.0
            for f in range(F):
                if fsrc[f] == s:
                    want += amt[f]
            if want > L[s] and L[s] > -1:
                if want > 0:
                    k = L[s] / want
                    for q in range(P):
                        dw = 0.0
                        for f in range(F):
                            if fsrc[f] == s:
                                dw += da[f, q]
                        dk = (dL[s, q] - k * dw) / want
                        for f in range(F):
                            if fsrc[f] == s:
                                da[f, q] = da[f, q] * k + amt[f] * dk
                    for f in range(F):
                        if fsrc[f] == s:
                            amt[f] *= k
        for f in range(F):
            a, b = fsrc[f], fdst[f]
            if a >= 0:
                L[a] -= amt[f]
                for q in range(P):
                    dL[a, q] -= da[f, q]
            if b >= 0:
                L[b] += amt[f]
                for q in range(P):
                    dL[b, q] += da[f, q]
        for s in range(S):
            if L[s] < 0:
                L[s] = 0.0
                for q in range(P):
                    dL[s, q] = 0.0
            X[s, i] = L[s]
            for q in range(P):
                J[s, i, q] = dL[s, q]
    return X, J


class _Net:
    """A structure in the form _sim runs: flows (src, dst, c, terms [(var, k)], mins [(x, y, theta, a)],
    fixed), over measured stocks `ids` and inputs `names`."""

    def __init__(self, flows: list[dict], names: list[str]):
        self.flows, self.names = flows, names
        self.jac = None

    def copy(self) -> _Net:
        return _Net(copy.deepcopy(self.flows), self.names)

    def slots(self) -> list[tuple]:
        """The free numbers as (flow, kind, j): c, a term's k, a MIN's a and theta."""
        out = []
        for fi, f in enumerate(self.flows):
            if f["fixed"]:
                continue
            if f["c"] is not None:
                out.append((fi, "c", 0))
            out += [(fi, "k", j) for j in range(len(f["terms"]))]
            for j in range(len(f["mins"])):
                out += [(fi, "a", j), (fi, "th", j)]
        return out

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """A term on the flow's own source, a MIN's a and theta are not negative (a drain does not fill),
        and such a term and a MIN's a are at most 1 a step (more is what the stock holds, all the same to
        the engine); the rest are free."""
        lo, hi = [], []
        for fi, kind, j in self.slots():
            f = self.flows[fi]
            own = kind in ("a", "th") or (kind == "k" and f["terms"][j][0] == f["src"])
            lo.append(0.0 if own else -np.inf)
            hi.append(1.0 if own and kind != "th" else np.inf)   # a drain takes at most what its source holds a step
        return np.array(lo), np.array(hi)

    def get(self) -> np.ndarray:
        return np.array([self._val(s) for s in self.slots()], dtype=float)

    def _val(self, s):
        fi, kind, j = s
        f = self.flows[fi]
        return f["c"] if kind == "c" else f["terms"][j][1] if kind == "k" else f["mins"][j][3] if kind == "a" else f["mins"][j][2]

    def put(self, p: np.ndarray) -> None:
        for (fi, kind, j), x in zip(self.slots(), p):
            f = self.flows[fi]
            if kind == "c":
                f["c"] = float(x)
            elif kind == "k":
                f["terms"][j] = (f["terms"][j][0], float(x))
            elif kind == "a":
                m = f["mins"][j]
                f["mins"][j] = (m[0], m[1], m[2], float(x))
            else:
                m = f["mins"][j]
                f["mins"][j] = (m[0], m[1], max(float(x), 1e-9), m[3])

    def arrays(self):
        F = self.flows
        fsrc = np.array([f["src"] for f in F], dtype=np.int64)
        fdst = np.array([f["dst"] for f in F], dtype=np.int64)
        fc = np.array([f["c"] or 0.0 for f in F], dtype=float)
        tf = np.array([fi for fi, f in enumerate(F) for _ in f["terms"]], dtype=np.int64)
        tv = np.array([v for f in F for v, _ in f["terms"]], dtype=np.int64)
        tk = np.array([k for f in F for _, k in f["terms"]], dtype=float)
        mf = np.array([fi for fi, f in enumerate(F) for _ in f["mins"]], dtype=np.int64)
        mx = np.array([m[0] for f in F for m in f["mins"]], dtype=np.int64)
        my = np.array([m[1] for f in F for m in f["mins"]], dtype=np.int64)
        mth = np.array([m[2] for f in F for m in f["mins"]], dtype=float)
        ma = np.array([m[3] for f in F for m in f["mins"]], dtype=float)
        return fsrc, fdst, fc, tf, tv, tk, mf, mx, my, mth, ma

    def positions(self):
        """Each number's position among slots() (-1: held): per flow its constant, per term, per MIN its a
        and theta, in arrays()' order."""
        pos = {s: q for q, s in enumerate(self.slots())}
        qc = np.array([pos.get((fi, "c", 0), -1) for fi in range(len(self.flows))], dtype=np.int64)
        qk = np.array([pos.get((fi, "k", j), -1) for fi, f in enumerate(self.flows) for j in range(len(f["terms"]))], dtype=np.int64)
        qa = np.array([pos.get((fi, "a", j), -1) for fi, f in enumerate(self.flows) for j in range(len(f["mins"]))], dtype=np.int64)
        qth = np.array([pos.get((fi, "th", j), -1) for fi, f in enumerate(self.flows) for j in range(len(f["mins"]))], dtype=np.int64)
        return qc, qk, qa, qth


class _Data:
    """The measured levels, injections and inputs; the weights; the steps whose levels are compared
    (every step up to POINTS compared numbers, else every k-th: the run is still stepped in full)."""

    def __init__(self, A: np.ndarray, inj: np.ndarray, U: np.ndarray):
        self.A, self.inj, self.U = A, inj, U
        self.S, self.n = A.shape
        self.w = np.ones_like(A)
        self.keep = np.arange(0, self.n, max(1, math.ceil(A.size / POINTS)))
        self.steps = 0                                   # steps simulated so far, plus OVERHEAD a run (the clock)
        self.wall = time.monotonic() + WALL
        self.units: list = [None] * self.S
        self.tried: list = []                            # (score, key, net) of every structure the beams fitted fully

    def same(self, i: int, j: int) -> bool:
        """May a move run between stocks i and j (outside: -1)? Not between two stocks whose announced
        units differ: a flow carries one unit (requests do not pour into millicores; a reading across
        units is a link, an outside inflow's term)."""
        return i < 0 or j < 0 or not self.units[i] or not self.units[j] or self.units[i] == self.units[j]

    def over(self, limit: float) -> bool:
        """Past the step budget `limit` (or, as a safety net only, the wall-clock cap)?"""
        return self.steps > limit or time.monotonic() > self.wall

    def resid(self, X: np.ndarray) -> np.ndarray:
        k = self.keep
        return ((X[:, k] - self.A[:, k]) * self.w[:, k]).ravel()


def _run(net: _Net, d: _Data, start: np.ndarray) -> np.ndarray:
    d.steps += d.n + OVERHEAD
    return _sim(d.n, d.S, start, d.inj, d.U, *net.arrays())


def _fit(net: _Net, d: _Data, start: np.ndarray, nfev: int | None = None) -> tuple[_Net, np.ndarray, float]:
    """The structure's numbers and starting levels fitted to the measured levels (weighted least squares
    on the output error); (net, start, weighted squared error)."""
    net = net.copy()
    lo, hi = net.bounds()
    p0 = np.concatenate([net.get(), start])
    k = len(p0) - d.S
    lo, hi = np.concatenate([lo, np.zeros(d.S)]), np.concatenate([hi, np.full(d.S, np.inf)])
    p0 = np.clip(p0, lo, hi)

    def res(p):
        net.put(p[:k])
        X = _run(net, d, p[k:])
        return d.resid(X)

    qs, keep = net.positions(), d.keep

    def jac(p):                                          # the run's derivatives, carried with it (_sens)
        net.put(p[:k])
        d.steps += (1 + len(p) / JAC_COST) * (d.n + OVERHEAD)
        _, Jx = _sens(d.n, d.S, p[k:], d.inj, d.U, *net.arrays(), *qs, len(p))
        return (Jx[:, keep, :] * d.w[:, keep, None]).reshape(-1, len(p))

    net.jac = None
    try:
        sol = least_squares(res, p0, jac=jac, bounds=(lo, hi), method="trf", x_scale="jac", max_nfev=nfev or NFEV * len(p0))
        p = sol.x
        net.jac = sol.jac[:, :k]
    except (ValueError, FloatingPointError, np.linalg.LinAlgError):
        p = p0
    r = res(p)
    if not np.all(np.isfinite(r)):
        return net, start, float("inf")
    net.put(p[:k])
    return net, p[k:], float(r @ r)


def _weights(net: _Net, d: _Data, start: np.ndarray) -> np.ndarray:
    """Each point's weight 1 / sd, the variance a + b x level^2 fitted to each stock's squared residuals."""
    X = _run(net, d, start)
    w = np.empty_like(d.A)
    for s in range(d.S):
        r2 = (X[s] - d.A[s]) ** 2
        V = np.column_stack([np.ones(d.n), X[s] ** 2])
        v = np.full(d.n, max(float(np.median(r2)), 1e-12))
        for _ in range(4):
            z = np.minimum(r2, 25 * v)
            ab = np.linalg.lstsq(V / v[:, None], z / v, rcond=None)[0]
            v = np.maximum(max(ab[0], 0.0) + max(ab[1], 0.0) * X[s] ** 2, 1e-6 * max(float(np.mean(z)), 1e-12))
        w[s] = 1 / np.sqrt(v)
    return w


def _parse(flows: list[dict], ids: list[str], names: list[str], cap: dict | None, bd: dict | None = None) -> _Net | None:
    """The found flows as a _Net, or None where one cannot be run here."""
    var = {s: i for i, s in enumerate(ids)} | {u: len(ids) + j for j, u in enumerate(names)}
    out = []
    for f in flows:
        if any(f.get(k) for k in ("byTime", "inflow", "effects", "uses", "via", "hidden")) or float(f.get("ratio") or 1) != 1:
            return None
        src, dst = f.get("from"), f.get("to")
        if (src is not None and src not in ids) or (dst is not None and dst not in ids):
            return None
        r = f.get("rate") or {}
        form = r.get("form")
        if form == "constant":
            c, terms = float(r.get("value") or 0), []
        elif form == "proportional":
            c, terms = None, [(r.get("stock"), float(r.get("k") or 0))]
        elif form == "linear":
            c, terms = float(r.get("c") or 0), [(x.get("stock"), float(x.get("k") or 0)) for x in r.get("terms") or []]
        elif form == "expr":                             # c + k x V + MIN(a x X, b x Y), as written by a MIN's stage
            parts = _expr_parts(r.get("expr") or "")
            if parts is None or any(p[0] == "m" and (p[1] not in var or p[3] not in var or not p[2] > 0) for p in parts):
                return None
            c = sum(p[1] for p in parts if p[0] == "c") if any(p[0] == "c" for p in parts) else None
            terms = [(p[1], p[2]) for p in parts if p[0] == "t"]
            pre = [(var[p[1]], var[p[3]], p[4] / p[2], p[2]) for p in parts if p[0] == "m"]
        else:
            return None
        lin, mins = [], list(pre) if form == "expr" else []
        for name, k in terms:
            if cap and name == cap.get("name"):
                mins.append((var[cap["source"]], var[cap["cap"]], float(cap["theta"]), k))
            elif name in var and not str(name).startswith(CAP):
                lin.append((var[name], k))
            else:
                return None
        out.append({"src": ids.index(src) if src is not None else -1, "dst": ids.index(dst) if dst is not None else -1,
                    "c": c if (c is not None and (c or form == "constant")) else None, "terms": lin, "mins": mins,
                    "fixed": _held(f, bd), "orig": f})
    return _Net(out, names)


def _held(f: dict, bd: dict | None) -> bool:
    """A flow kept as it came: an outside inflow that is a measured input (marked so by the fit), where
    the boundary, if declared, gives its receiver an input inflow. An outflow that reads an input is not
    one: its form is the fit's guess, revised like any other."""
    if not f.get("input") or str(f.get("input")).startswith(CAP) or f.get("from") is not None:
        return False
    return bd is None or "input" in ((bd.get(f.get("to")) or {}).get("in") or [])


_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"


def _expr_parts(expr: str) -> list[tuple] | None:
    """An expression of constants, terms k * V and MIN(a * X, b * Y) joined by +, as
    [("c", c) | ("t", V, k) | ("m", X, a, Y, b)]; None for anything else."""
    import re

    out = []
    for part in [p.strip() for p in re.split(r"\s\+\s", expr)]:
        m = re.fullmatch(rf"min\(\s*({_NUM})\s*\*\s*(\w+)\s*,\s*({_NUM})\s*\*\s*(\w+)\s*\)", part)
        if m:
            out.append(("m", m.group(2), float(m.group(1)), m.group(4), float(m.group(3))))
            continue
        m = re.fullmatch(rf"({_NUM})\s*\*\s*(\w+)", part)
        if m:
            out.append(("t", m.group(2), float(m.group(1))))
            continue
        if re.fullmatch(_NUM, part):
            out.append(("c", float(part)))
            continue
        return None
    return out


def _canonical(net: _Net) -> _Net:
    """The same model with a stock that always empties written as such: where every outflow of a stock
    is a rate on its own level alone and they sum past 1 a step, the engine scales them to what the
    stock holds, so k_i / sum k is what each takes; written so (identical runs)."""
    net = net.copy()
    for s in {f["src"] for f in net.flows if f["src"] >= 0}:
        out = [f for f in net.flows if f["src"] == s]
        if any(f["fixed"] or f["mins"] or f["c"] or len(f["terms"]) != 1 or f["terms"][0][0] != s for f in out):
            continue
        tot = sum(f["terms"][0][1] for f in out)
        if tot > 1:
            for f in out:
                f["terms"][0] = (s, f["terms"][0][1] / tot)
    return net


def _write(net: _Net, ids: list[str]) -> list[dict]:
    """The _Net as flows (a MIN as an expression with an id; the input flow as it came)."""
    net = _canonical(net)
    names = ids + list(net.names)
    out = []
    for n, f in enumerate(net.flows):
        if f["fixed"]:
            out.append(f["orig"])
            continue
        g = {"from": ids[f["src"]] if f["src"] >= 0 else None, "to": ids[f["dst"]] if f["dst"] >= 0 else None, "ratio": 1}
        if f["mins"]:
            parts = [repr(f["c"])] if f["c"] else []
            parts += [f"{k!r} * {names[v]}" for v, k in f["terms"]]
            for x, y, th, a in f["mins"]:
                parts.append(f"min({a!r} * {names[x]}, {a * th!r} * {names[y]})" if a >= 0 else f"{a!r} * min({names[x]}, {th!r} * {names[y]})")
            rate = {"form": "expr", "expr": " + ".join(parts)}
            g["id"] = f"oe{n}"
        elif f["c"] is None and len(f["terms"]) == 1:
            rate = {"form": "proportional", "stock": names[f["terms"][0][0]], "k": f["terms"][0][1]}
        elif not f["terms"]:
            rate = {"form": "constant", "value": f["c"] or 0.0}
        else:
            rate = {"form": "linear", "c": f["c"] or 0.0, "terms": [{"stock": names[v], "k": k} for v, k in f["terms"]]}
        g["rate"], g["rates"] = rate, [rate]
        out.append(g)
    return out


def _sane(net: _Net, d: _Data) -> None:
    """Starting numbers the run can move: a stock whose proportional outflows on itself sum past 1
    a step (a drain capped at what it holds, where the fit has no slope) scaled back to 0.95."""
    for s in range(d.S):
        own = [(fi, j) for fi, f in enumerate(net.flows) if f["src"] == s and not f["fixed"]
               for j, (v, k) in enumerate(f["terms"]) if v == s and k > 0]
        tot = sum(net.flows[fi]["terms"][j][1] for fi, j in own)
        if tot > 1:
            for fi, j in own:
                v, k = net.flows[fi]["terms"][j]
                net.flows[fi]["terms"][j] = (v, k * 0.95 / tot)


def _reads_other(f: dict, S: int) -> int | None:
    """The one measured stock a flow between measured stocks reads, where it is neither of its ends."""
    reads = {v for v, _ in f["terms"] if v < S}
    if f["fixed"] or f["mins"] or f["src"] < 0 or f["dst"] < 0 or len(reads) != 1:
        return None
    z = reads.pop()
    return None if z in (f["src"], f["dst"]) else z


def _merged(net: _Net) -> _Net:
    """Flows between the same ends joined (their constants and terms on the same variable added)."""
    out = []
    for f in net.flows:
        g = next((h for h in out if (h["src"], h["dst"]) == (f["src"], f["dst"]) and not h["fixed"] and not f["fixed"]), None)
        if g is None:
            out.append(f)
            continue
        if f["c"] is not None:
            g["c"] = (g["c"] or 0.0) + f["c"]
        for v, k in f["terms"]:
            j = next((j for j, (w, _) in enumerate(g["terms"]) if w == v), None)
            if j is None:
                g["terms"].append((v, k))
            else:
                g["terms"][j] = (v, g["terms"][j][1] + k)
        g["mins"] += f["mins"]
    net.flows = out
    return net


def _moves(d: _Data, x: int) -> bool:
    """Does stock x move beyond its announced injections and its measurement noise? (The variance of
    its level less the injections' sum against half the variance of its step changes, which white
    measurement error alone makes about 1.)"""
    y = d.A[x] - np.cumsum(d.inj[x])
    dv = float(np.var(np.diff(y)))
    return dv > 0 and float(np.var(y)) > MOVES * dv / 2


def _plus(net: _Net, x: int, y: int) -> _Net:
    g = net.copy()
    g.flows.append({"src": x, "dst": y, "c": None, "terms": [(x, ADD)], "mins": [], "fixed": False, "orig": None})
    return g


def _candidates(net: _Net, d: _Data, final: set, start: np.ndarray, pool: list[tuple] = ()):
    """(label, net, numbers added) for every one-step revision; pool: parts of the found structure,
    (src, dst, variable, k), any of which a revision may put back."""
    S = d.S
    caps = [y for y in range(S) if y not in final and np.ptp(d.A[y]) > 0] + [S + u for u in range(d.U.shape[0]) if np.ptp(d.U[u]) > 0]
    X = _run(net, d, start)
    vals = np.vstack([X, d.U])
    moving = [x for x in range(S) if x not in final and _moves(d, x)]
    for fi, f in enumerate(net.flows):
        if f["fixed"]:
            continue
        inside = f["src"] >= 0 and f["dst"] >= 0
        nparts = len(f["terms"]) + len(f["mins"]) + (f["c"] is not None)
        for j in range(len(f["terms"])):             # drop a term (a flow between measured stocks may lose its last)
            if nparts > 1 or inside:
                g = net.copy()
                del g.flows[fi]["terms"][j]
                if not (g.flows[fi]["terms"] or g.flows[fi]["mins"] or g.flows[fi]["c"] is not None):
                    del g.flows[fi]
                yield f"drop term {j} of flow {fi}", g, -1
        for j in range(len(f["mins"])):
            if nparts > 1 or inside:
                g = net.copy()
                del g.flows[fi]["mins"][j]
                if not (g.flows[fi]["terms"] or g.flows[fi]["mins"] or g.flows[fi]["c"] is not None):
                    del g.flows[fi]
                yield f"drop min {j} of flow {fi}", g, -2
            g = net.copy()                               # the limit as a plain rate on its source
            x, _, _, a = g.flows[fi]["mins"].pop(j)
            g.flows[fi]["terms"].append((x, a))
            yield f"flow {fi}: min {j} as a rate", _merged(g), -1
        if f["c"] is not None and nparts > 1:
            g = net.copy()
            g.flows[fi]["c"] = None
            yield f"drop constant of flow {fi}", g, -1
        src = f["src"]
        empties = sum(k for h in net.flows if h["src"] == src for v, k in h["terms"] if v == src) >= 1
        if src >= 0 and src not in final and not empties:   # a stock that empties each step has no limit to find
            for j, (v, k) in enumerate(f["terms"]):
                if v != src or k <= 0:
                    continue
                for y in caps:
                    if y == src:
                        continue
                    Y = vals[y]
                    ok = Y > 0
                    if ok.sum() < 10:
                        continue
                    tried = []                       # theta from a grid, by the run's error before any fit
                    for th in np.unique(np.quantile(X[src][ok] / Y[ok], np.linspace(MIN_SHARE, 1 - MIN_SHARE, GRID))):
                        g = net.copy()
                        del g.flows[fi]["terms"][j]
                        m = np.minimum(X[src], th * Y)
                        a = k * float(X[src].mean()) / max(float(m.mean()), 1e-12)   # the same mean flow
                        g.flows[fi]["mins"].append((src, y, float(max(th, 1e-9)), a))
                        r = d.resid(_run(g, d, start))
                        if np.all(np.isfinite(r)):
                            tried.append((float(r @ r), float(th), g))
                    if tried:
                        _, th, g = min(tried, key=lambda x: x[0])
                        yield f"flow {fi}: {k:.3g} x MIN(src, {th:.3g} x var {y})", g, 1
        if f["dst"] in final and src >= 0 and not f["mins"] and [v for v, _ in f["terms"]] == [src]:
            for x in range(S):                           # a counter fed from another stock
                if x not in final and x != src and any(h["src"] == x for h in net.flows) and d.same(x, f["dst"]):
                    g = net.copy()
                    grow = max(float(d.A[f["dst"], -1] - d.A[f["dst"], 0]), 0.0) / max(1.0, float(X[x].sum()))
                    g.flows[fi]["src"], g.flows[fi]["terms"] = x, [(x, grow)]
                    yield f"flow {fi}: from var {x}", _merged(g), 0
        own = [v for v, _ in f["terms"]] == [src] and not f["mins"] and f["c"] is None
        if own and inside and f["dst"] not in final and src not in final:
            for x in moving:                             # a drain of another stock into the same receiver
                if x not in (src, f["dst"]) and not any((h["src"], h["dst"]) == (x, f["dst"]) for h in net.flows) and d.same(x, f["dst"]):
                    g = net.copy()
                    k = f["terms"][0][1] * max(1e-9, float(X[src].mean())) / max(1e-9, float(X[x].mean()))
                    g.flows[fi]["src"], g.flows[fi]["terms"] = x, [(x, min(k, 1.0))]
                    yield f"flow {fi}: drains var {x}", g, 0
        if src >= 0 and not f["mins"] and len(f["terms"]) == 1 and f["terms"][0][0] != src:
            z, k = f["terms"][0]                         # an outflow read off its own source instead (of a level or an input)
            g = net.copy()
            g.flows[fi]["terms"] = [(src, min(1.0, abs(k) * max(1e-9, float(np.abs(vals[z]).mean())) / max(1e-9, float(X[src].mean()))))]
            yield f"flow {fi}: reads its source", _merged(g), 0
        z = _reads_other(f, S)
        if z is not None and d.same(z, f["dst"]):
            g = net.copy()
            g.flows[fi]["src"] = z
            yield f"flow {fi}: source var {z}", _merged(g), 0
    others = [fi for fi, f in enumerate(net.flows) if not f["fixed"] and _reads_other(f, S) is not None
              and d.same(_reads_other(f, S), f["dst"])]
    if len(others) > 1:                                  # all of them at once (each alone may not pay)
        g = net.copy()
        for fi in others:
            g.flows[fi]["src"] = _reads_other(g.flows[fi], S)
        yield "every flow reading another stock: that stock as its source", _merged(g), 0
    have = {(f["src"], f["dst"], v) for f in net.flows for v, _ in f["terms"]}
    for src, dst, v, k in pool:                          # a part of the found structure put back
        if (src, dst, v) not in have:
            g = net.copy()
            g.flows.append({"src": src, "dst": dst, "c": None, "terms": [(v, k)], "mins": [], "fixed": False, "orig": None})
            yield f"put back var {v}: var {src} -> var {dst}", _merged(g), 1
    for x in moving:                                     # a move between two stocks that move
        for y in moving:
            if x != y and not any((f["src"], f["dst"]) == (x, y) for f in net.flows) and d.same(x, y):
                yield f"add move var {x} -> var {y}", _plus(net, x, y), 1
    for fi, f in enumerate(net.flows):                   # an outside inflow reads one more level (a link; not a
                                                         # counter's: its running total reads as the run's age)
        if f["fixed"] or f["src"] >= 0 or f["dst"] < 0:
            continue
        read = {v for v, _ in f["terms"]}
        for v in [y for y in range(S) if y != f["dst"] and y not in final and np.ptp(d.A[y]) > 0] + [S + u for u in range(d.U.shape[0]) if np.ptp(d.U[u]) > 0]:
            if v not in read:
                g = net.copy()
                g.flows[fi]["terms"].append((v, 0.0))
                yield f"link var {v} into flow {fi}", g, 1
    for c in final:
        if any(f["dst"] == c for f in net.flows):
            continue
        for x in range(S):
            if x in final or x == c or not any(f["src"] == x for f in net.flows) or not d.same(x, c):
                continue
            grow = max(float(d.A[c, -1] - d.A[c, 0]), 0.0) / max(1.0, float(X[x].sum()))
            g = net.copy()
            g.flows.append({"src": x, "dst": c, "c": None, "terms": [(x, grow)], "mins": [], "fixed": False, "orig": None})
            yield f"move var {x} -> counter {c}", g, 1


def _measurement_error(flows: list[dict], ids: list[str], run: dict, opts: dict) -> bool:
    """Do the found model's one-step residuals show measurement error (eiv.white_z, on the stocks
    whose residuals are more than rounding)?"""
    from ...simulate import Model, levels_array

    A = levels_array(run["levels"], ids)
    t = np.asarray(run["t"], dtype=float)
    inputs = {"t": list(run["t"]), "values": opts["inputs"]} if opts.get("inputs") else None
    try:
        e = (A[:, 1:] - A[:, :-1]) - Model(flows, ids, inputs=inputs).change(A[:, :-1], t[1:])
    except (KeyError, ValueError, IndexError):
        return False
    dy = A[:, 1:] - A[:, :-1]
    z = [x if np.mean(e[k] ** 2) > ROUNDING ** 2 * np.mean(dy[k] ** 2) else 0.0 for k, x in enumerate(white_z(e, dy))]
    return bool(z) and min(z) <= -norm.isf(ALPHA / len(z))


def white_z(e: np.ndarray, dy: np.ndarray) -> list[float]:
    """Each stock's lag-1 correlation of its residuals, as a z-score under white noise (0 where the
    residuals are nothing but rounding); a copy of eiv.white_z, so this module stands alone."""
    out = []
    for s in range(e.shape[0]):
        es = e[s] - e[s].mean()
        v = float(es @ es)
        out.append(float(es[1:] @ es[:-1]) / v * np.sqrt(len(es)) if v > 1e-12 * max(1.0, float(dy[s] @ dy[s])) else 0.0)
    return out


def _key(net: _Net) -> tuple:
    """A structure's signature (which numbers on which ends), for the beam's duplicates."""
    return tuple(sorted((f["src"], f["dst"], f["c"] is not None, tuple(sorted(v for v, _ in f["terms"])),
                         tuple(sorted((m[0], m[1]) for m in f["mins"]))) for f in net.flows))


def _live(net: _Net, d: _Data, start: np.ndarray) -> _Net:
    """The structure without the flows between measured stocks that carry nothing on any step of the
    run (a rate the fit drove below 0 everywhere, floored at 0): the same runs, fewer parts."""
    net = net.copy()
    V = np.vstack([_run(net, d, start), d.U])
    keep = []
    for f in net.flows:
        amt = (f["c"] or 0.0) + sum(k * V[v] for v, k in f["terms"]) + sum(a * np.minimum(V[x], th * V[y]) for x, y, th, a in f["mins"])
        if f["fixed"] or f["src"] < 0 or f["dst"] < 0 or np.any(np.asarray(amt) > 0):
            keep.append(f)
    net.flows = keep
    return net


def _binds(net: _Net, d: _Data, start: np.ndarray) -> bool:
    """Every capacity limit's two regimes each hold on at least MIN_SHARE of the run's steps (a limit
    that never binds, or always does, is a linear rate)."""
    if not any(f["mins"] for f in net.flows):
        return True
    V = np.vstack([_run(net, d, start), d.U])
    for f in net.flows:
        for x, y, th, _ in f["mins"]:
            share = float(np.mean(V[x] <= th * V[y]))
            if not MIN_SHARE <= share <= 1 - MIN_SHARE:
                return False
    return True


def _search(net: _Net, start: np.ndarray, cost: float, d: _Data, final: set, price: float, deadline: float):
    """The structure revised by a beam search over one-step revisions (each round keeps the BEAM best
    new structures, scored by the weighted squared error plus price per number, so a path may pass
    through a worse one), until PATIENCE rounds find nothing better than the best so far or the
    deadline passes; (net, start, cost, revisions) of the best."""
    pool = [(f["src"], f["dst"], v, k) for f in net.flows if not f["fixed"] and f["src"] >= 0 and f["dst"] >= 0
            for v, k in f["terms"]]
    extra = (MIN_PRICE - PRICE) / PRICE * price
    score = lambda c, g: c + price * len(g.slots()) + extra * sum(len(f["mins"]) for f in g.flows)   # noqa: E731
    beam = [(score(cost, net), net, start, cost, [])]
    seen = {_key(net)}
    best, stale = beam[0], 0
    for _ in range(ROUNDS):
        grown = []
        for _, b, st, _, path in beam:
            for label, g, _ in _candidates(b, d, final, st, pool):
                if d.over(deadline):
                    break
                k = _key(g)
                if k in seen:
                    continue
                seen.add(k)
                g2, s2, c2 = _fit(g, d, st, nfev=QUICK)
                if np.isfinite(c2):
                    grown.append((score(c2, g2), g2, s2, c2, path + [label]))
        beam = []
        for _, g, st, _, path in sorted(grown, key=lambda x: x[0]):     # the most promising, fitted fully
            if len(beam) == BEAM or d.over(deadline):
                break
            g2, s2, c2 = _fit(g, d, st)
            if np.isfinite(c2) and _binds(g2, d, s2):
                beam.append((score(c2, g2), g2, s2, c2, path))
                d.tried.append((score(c2, g2), _key(g2), g2))
        if not beam:
            break
        beam.sort(key=lambda x: x[0])
        if beam[0][0] < best[0]:
            best, stale = beam[0], 0
        else:
            stale += 1
        if stale >= PATIENCE or d.over(deadline):
            break
    return best[1], best[2], best[3], best[4]


def refit(result: dict, stocks: list[dict], run: dict, opts: dict | None, cap: dict | None = None, budget: float = BUDGET) -> dict:
    """The result with its flows refitted and revised on the output error (the module's rule), and what
    was done under "oe"."""
    opts = opts or {}
    if not isinstance(result, dict) or not result.get("flows"):
        return result
    ids = [s["id"] for s in stocks]
    final = {i for i, s in enumerate(stocks) if s.get("final")}
    t = np.asarray(run["t"], dtype=float)
    n = len(t)
    if n < 50 or np.any(np.diff(t) != 1):
        return result
    A = np.array([np.asarray(run["levels"].get(i, []), dtype=float) for i in ids])
    if A.ndim != 2 or A.shape[1] != n or not np.all(np.isfinite(A)):
        return result
    names = [k for k in (opts.get("inputs") or {}) if k not in ids and not str(k).startswith(CAP)]
    if not _measurement_error(result["flows"], ids, run, opts):
        return result
    bd = run.get("boundary") if run.get("boundary_declared") else None
    net = _parse(result["flows"], ids, names, cap, bd)
    plain = None
    if net is None:                                      # pieces or an effect: the search starts from a plain rate
        plain = _plain(result["flows"], ids, run, opts)  # of the same mean amount, and is judged against the found
        net = _parse(plain, ids, names, cap, bd) if plain else None   # model as the engine runs it
    if net is None:
        return {**result, "oe": {"skipped": "a form this refit does not run"}}
    U = np.array([np.asarray(opts["inputs"][k], dtype=float)[:n] for k in names]).reshape(len(names), n)
    inj = np.zeros_like(A)
    for j in opts.get("injections") or []:
        if j.get("stock") in ids:
            inj[ids.index(j["stock"]), np.nonzero(t == float(j["t"]))[0]] += float(j["amount"])
    d = _Data(A, inj, U)
    d.units = [s.get("unit") or None for s in stocks]
    d.w = 1.0 / np.maximum(np.std(np.diff(A, axis=1), axis=1, keepdims=True), 1e-9) * np.ones_like(A)
    orig = net.copy()
    _sane(net, d)
    price = PRICE * math.log(d.S * len(d.keep))
    found = _from(net, d, final, price, budget * FOUND_SHARE)
    wide = _from(_superset(net, d, final), d, final, price, budget, thin=True)
    net, start, cost, nll, done, d.w = min(found, wide, key=lambda x: x[3] + price * _numbers(x[0]))
    cands, keys = [], set()                              # the best structures the searches fitted, by their score,
    for g in [found[0], wide[0]] + [x[2] for x in sorted(d.tried, key=lambda x: x[0])]:   # judged on the filter
        if _key(g) not in keys and len(cands) < CHOICES:
            keys.add(_key(g))
            cands.append(g)
    pick, kf_prices = _kf_choice(cands, d, PRICE * math.log(d.S * d.n)) if len(cands) > 1 else (0, [])
    if cands[pick] is not net:
        net, start, cost = _fit(cands[pick], d, start)
        done = done + [f"the filter's choice: structure {pick} of the best {len(cands)}"]
    r0 = d.resid(_run(orig, d, start) if plain is None else _engine_run(result["flows"], ids, start, t, opts))
    found_cost = float(r0 @ r0) if np.all(np.isfinite(r0)) else float("inf")
    info = {"seed": "found" if net is found[0] else "every move", "revised": done,
            "nll": [round(found[3], 1), round(wide[3], 1)], "steps": d.steps}
    if not cost < found_cost or not np.isfinite(cost):
        return {**result, "oe": {"used": False, "cost": [found_cost, cost], **info}}
    net, kf = _kf_numbers(net, d)
    net = _live(net, d, start)                           # a flow the Kalman numbers left carrying nothing goes
    return {**result, "flows": _write(net, ids), "oe": {"used": True, "cost": [round(found_cost, 1), round(cost, 1)], **info, **kf,
                                                          "kf_prices": kf_prices}}


def _plain(flows: list[dict], ids: list[str], run: dict, opts: dict) -> list[dict] | None:
    """The flows with each one in pieces (an unannounced change of its amount, or pieces by time where no
    eras are announced) or under an effect written as a plain rate of the same mean amount over the run:
    a share of its source, or a constant from outside; None where another form (announced eras, co-flows,
    hidden stocks, a ratio) is in the way."""
    from ...simulate import Model, levels_array

    A = levels_array(run["levels"], ids)
    t = np.asarray(run["t"], dtype=float)
    inputs = {"t": list(run["t"]), "values": opts["inputs"]} if opts.get("inputs") else None
    out = []
    for f in flows:
        if any(f.get(k) for k in ("uses", "via", "hidden")) or float(f.get("ratio") or 1) != 1 or (f.get("byTime") and opts.get("eras")):
            return None
        if not (f.get("inflow") or f.get("effects") or f.get("byTime")):
            out.append(f)
            continue
        src, dst = f.get("from"), f.get("to")
        try:
            if src in ids:
                amt = -Model([{**f, "to": None}], ids, inputs=inputs).change(A[:, :-1], t[1:])[ids.index(src)]
            elif dst in ids:
                amt = Model([f], ids, inputs=inputs).change(A[:, :-1], t[1:])[ids.index(dst)]
            else:
                return None
        except (KeyError, ValueError, IndexError):
            return None
        m = float(np.mean(amt)) if np.all(np.isfinite(amt)) else 0.0
        g = {k: v for k, v in f.items() if k not in ("inflow", "effects", "driver", "perStep", "changes", "rates", "byTime")}
        if src in ids:
            rate = {"form": "proportional", "stock": src, "k": min(1.0, max(0.0, m / max(1e-9, float(np.mean(A[ids.index(src)])))))}
        else:
            rate = {"form": "constant", "value": m}
        out.append({**g, "rate": rate, "rates": [rate]})
    return out


def _engine_run(flows: list[dict], ids: list[str], start: np.ndarray, t: np.ndarray, opts: dict) -> np.ndarray:
    """Levels (stocks x steps) of the found flows run by the shared engine from `start`, the announced
    injections added before their step's flows (as _sim steps), for a found model _sim cannot run."""
    from ...simulate import Model

    inputs = {"t": list(t), "values": opts["inputs"]} if opts.get("inputs") else None
    try:
        model = Model(flows, ids, inputs=inputs)
    except (KeyError, ValueError, IndexError):
        return np.full((len(ids), len(t)), np.nan)
    inj = {}
    for j in opts.get("injections") or []:
        if j.get("stock") in ids:
            inj.setdefault(float(j["t"]), []).append((ids.index(j["stock"]), float(j["amount"])))
    X = np.zeros((len(ids), len(t)))
    L = np.maximum(np.asarray(start, float), 0.0)[:, None]
    X[:, 0] = L[:, 0]
    for i in range(1, len(t)):
        L = L.copy()
        for s, a in inj.get(float(t[i]), []):
            L[s, 0] = max(0.0, L[s, 0] + a)
        L = model.step(L, np.array([t[i]]))
        X[:, i] = L[:, 0]
    return X


def _numbers(net: _Net) -> float:
    """A structure's numbers, a capacity limit's extra number at its own price (in PRICE units)."""
    return len(net.slots()) + (MIN_PRICE - PRICE) / PRICE * sum(len(f["mins"]) for f in net.flows)


def _from(net: _Net, d: _Data, final: set, price: float, deadline: float, thin: bool = False) -> tuple:
    """The search from one seed (thin: pruned by _thin first), with the noise read again off each
    revised model; (net, start, its weighted squared error, its Gaussian -2 log-likelihood with those
    variances, revisions, weights)."""
    start = d.A[:, 0].copy()
    net, start, _ = _fit(net, d, start)
    if thin:
        d.w = _weights(net, d, start)
        net, start = _thin(net, d, start, price, deadline)
    done: list[str] = []
    for _ in range(CYCLES):
        for _ in range(2):
            d.w = _weights(net, d, start)
            net, start, cost = _fit(net, d, start)
        before = _key(net)
        net, start, cost, path = _search(net, start, cost, d, final, price, deadline)
        done += path
        if _key(net) == before or d.over(deadline):
            break
    net = _live(net, d, start)
    d.w = _weights(net, d, start)
    net, start, cost = _fit(net, d, start)
    return net, start, cost, cost - 2 * float(np.log(d.w[:, d.keep]).sum()), done, d.w.copy()


def _thin(net: _Net, d: _Data, start: np.ndarray, price: float, deadline: float) -> tuple[_Net, np.ndarray]:
    """A wide structure pruned by the Wald statistic of its fit: the part (a term, a constant, a capacity
    limit) whose numbers fall furthest below the price per number dropped, refitted, until every part
    pays (one fit a step, where the search would fit every drop)."""
    net, start, _ = _fit(net, d, start)
    while not d.over(deadline) and net.jac is not None:
        slots = net.slots()
        th = net.get()
        info = net.jac.T @ net.jac
        parts: dict = {}
        for j, (fi, kind, i) in enumerate(slots):
            parts.setdefault((fi, "m" if kind in ("a", "th") else kind, i), []).append(j)
        worst = None
        for (fi, kind, i), js in parts.items():
            if kind == "th":
                continue
            js = [j for j in js if slots[j][1] != "th"]
            try:
                wald = float(th[js] @ (np.linalg.inv(np.linalg.pinv(info)[np.ix_(js, js)])) @ th[js])
            except np.linalg.LinAlgError:
                wald = 0.0
            short = wald - price * len(js)
            if short < 0 and (worst is None or short < worst[0]):
                worst = (short, fi, kind, i)
        if worst is None:
            break
        _, fi, kind, i = worst
        g = net.copy()
        f = g.flows[fi]
        if kind == "c":
            f["c"] = None
        elif kind == "k":
            del f["terms"][i]
        else:
            del f["mins"][i]
        if not (f["terms"] or f["mins"] or f["c"] is not None) and f["src"] >= 0 and f["dst"] >= 0:
            del g.flows[fi]
        elif not (f["terms"] or f["mins"] or f["c"] is not None):
            break                                        # an outside flow keeps its last part (the boundary)
        net, start, _ = _fit(g, d, start)
    return net, start


def _superset(net: _Net, d: _Data, final: set) -> _Net:
    """The found structure with every move between two stocks that move and every move from a stock
    that moves into each counter (a wide seed, which the search prunes)."""
    g = net.copy()
    S = d.S
    moving = [x for x in range(S) if x not in final and _moves(d, x)]
    for x in moving:
        for y in moving + sorted(final):
            if x != y and not any((f["src"], f["dst"]) == (x, y) for f in g.flows) and d.same(x, y):
                k = ADD if y not in final else max(float(d.A[y, -1] - d.A[y, 0]), 0.0) / max(1.0, float(d.A[x].sum())) / max(1, len(moving))
                g.flows.append({"src": x, "dst": y, "c": None, "terms": [(x, k)], "mins": [], "fixed": False, "orig": None})
    for f in g.flows:                                    # every outside inflow reading every other level
        if f["fixed"] or f["src"] >= 0 or f["dst"] < 0:
            continue
        read = {v for v, _ in f["terms"]}
        f["terms"] += [(v, 0.0) for v in [y for y in range(S) if y != f["dst"] and y not in final and np.ptp(d.A[y]) > 0]
                       + [S + u for u in range(d.U.shape[0]) if np.ptp(d.U[u]) > 0] if v not in read]
    _sane(g, d)
    return g
