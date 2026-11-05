"""The search on noisy data: the reporting read (span, lag, outliers), the superset fitted by the
Kalman likelihood, the atoms chosen, the flows written."""

from __future__ import annotations

import math
import time

import numpy as np

from ...matrix import configurations
from ._vendored.lna.atoms import Atom, restricted, superset
from ._vendored.lna.flows import to_flows
from . import search as S
from .model import Fitted, Layout, OutOfTime, Problem
from .hidden import hidden
from .boundary import Boundary
from .inside import Allow
from .robust import FRAC, RATIO, gross_errors

BUDGET = 100.0        # seconds for the whole search (opts "kalman_budget"); past it, None
MAX_LAG = 8           # reporting delays tried where announced injections pin one
Z_GROSS = 10.0       # beyond this, a value is a gross error for the base too (it gets its neighbours' mean)
Z_OUT = 5.0           # a reported value whose standardised innovation passes this is a gross error


def buckets(cfg: np.ndarray, L: int, least: int = 50, share: float = 0.05, most: int = 4) -> list[int]:
    """lna's: each era label's own numbers only if it holds long enough (else one shared set)."""
    n = np.bincount(cfg, minlength=L)
    long = [c for c in range(L) if n[c] >= max(least, share * len(cfg))]
    if len(long) > most:
        long = sorted(sorted(long, key=lambda c: -n[c])[: most - 1])
    return [long.index(c) if c in long else len(long) for c in range(L)]


def spans_to_try(t: np.ndarray) -> list[int]:
    """1 (each point a level at its time); and, when every point is the same L > 1 steps apart, L
    (each point the mean over the L steps ending there: weekly averages)."""
    dt = np.diff(t)
    if len(dt) and dt.min() == dt.max() and dt[0] > 1:
        return [int(dt[0]), 1]
    return [1]


def candidate_atoms(stocks, base: dict | None, lay: Layout) -> list[Atom]:
    ids = [s["id"] for s in stocks]
    n = len(ids)
    unit = [s.get("unit") or "" for s in stocks]
    final = {k for k, s in enumerate(stocks) if s.get("final")}
    group = [s.get("group") for s in stocks]
    if n <= 4:
        return superset(n, unit, final, group, gaps=False)
    index = {v: i for i, v in enumerate(ids)}
    coef = [{index[j]: k for j, k in (row.get("coef") or {}).items() if j in index and abs(k) > 1e-9} for row in (base or {}).get("matrix") or []]
    if len(coef) != n:
        coef = [{s: -0.1} for s in range(n)]
    return restricted(n, coef, unit, final)


SCREEN_T = 2.0       # with 3 or more stocks, a flow between two stocks is offered only where the
                     # regression of one's changes on every level shows the other (|t| above this)


def screened(atoms: list[Atom], pr: Problem) -> list[Atom]:
    """The atoms between stocks whose coupling shows in the means: each stock's step changes on
    the first reference path regressed on every level (robust standard errors); an atom whose
    driver is another stock is kept only where that coefficient's |t| passes SCREEN_T."""
    ref = pr.zref[:, 1:]
    dy = np.diff(ref, axis=1)
    X = np.vstack([ref[:, :-1], np.ones(dy.shape[1])]).T
    XtX = np.linalg.pinv(X.T @ X)
    sig = np.zeros((pr.n, pr.n), dtype=bool)
    for s in range(pr.n):
        b = XtX @ X.T @ dy[s]
        e = dy[s] - X @ b
        cov = XtX @ (X.T * e ** 2) @ X @ XtX
        se = np.sqrt(np.maximum(np.diag(cov), 1e-30))
        sig[s] = np.abs(b[: pr.n] / se[: pr.n]) > SCREEN_T
    out = []
    for a in atoms:
        if a.driver is None or a.driver >= pr.n:
            out.append(a)
            continue
        hit = [j for j in (a.src, a.dst) if j is not None and j != a.driver]
        if all(sig[j, a.driver] for j in hit):
            out.append(a)
    return out


def _search(pr: Problem, cur):
    cur = S.prune(pr, cur)
    for _ in range(3):
        before = list(cur.atoms)
        cur = S.prune(pr, S.pieces(pr, S.prune(pr, S.swaps(pr, cur))))
        if cur.atoms == before:
            break
    return S.merge_eras(pr, cur)


class Session:
    """One run of the search, in stages the caller can stop between: prepare() (how the points
    report the stocks, a reporting delay, gross errors, the superset fitted), evidence() (whether
    the levels carry a measurement error), finish() (the atoms chosen, the flows). A stage that runs
    past the budget raises OutOfTime; a numerical failure raises one of FAILURES."""

    def __init__(self, stocks: list[dict], run: dict, opts: dict, base: dict | None = None, trace: list | None = None):
        self.stocks, self.opts, self.base = stocks, opts, base
        self.trace = trace if trace is not None else []
        self.deadline = time.monotonic() + float(opts.get("kalman_budget", BUDGET))
        self.clock = time.monotonic()
        self.ids = [s["id"] for s in stocks]
        self.t = np.asarray(run["t"], dtype=np.int64)
        self.Y = np.array([np.asarray(run["levels"].get(i, [np.nan] * len(self.t)), dtype=float) for i in self.ids]).T
        self.seen = np.isfinite(self.Y)
        self.inj = opts.get("injections")
        conf = configurations(opts.get("eras"), list(map(int, self.t)))
        bucket = None
        if conf:
            bucket = buckets(conf[1], len(conf[0]))
            if max(bucket) < 1:
                conf, bucket = None, None
        self.conf, self.bucket = conf, bucket
        self.pr = self.cur = self.full = None
        # the declared boundary (boundary.py, inside.py): outside atoms only where it allows them
        self.allow_atom = Allow(Boundary(run), self.ids)

    def allow(self, seconds: float) -> None:
        """From now on, at most this many seconds (every Problem made reads the new deadline)."""
        self.deadline = time.monotonic() + seconds
        for pr in (self.pr,):
            if pr is not None:
                pr.deadline = self.deadline

    def stage(self, name):
        self.trace.append(f"{name}: {time.monotonic() - self.clock:.1f}s")

    def problem(self, span, lag, mask):
        lay = Layout(self.ids, self.t, self.Y, span=span, lag=lag, injections=self.inj, eras=None, mask=mask.copy())
        lay.allow = self.allow_atom             # read by search.swaps (a swap may not leave the boundary)
        cfg = lay.era_cfg(self.conf)
        if cfg is not None:
            cfg = np.asarray(self.bucket)[cfg]
        return Problem(lay, cfg, self.deadline)

    def first_fit(self, pr):
        cur = S.refit(pr, self.atoms0, None, trial=True)
        pr.refresh(cur)
        cur = S.refit(pr, self.atoms0, cur)
        return S.drop_zeros(pr, cur)

    def prepare(self) -> bool:
        """The superset fitted to the reports as they are read best; False if the run is too short."""
        n, t = len(self.ids), self.t
        if len(t) < 12 or n == 0:
            return False
        mask = self.seen & ~gross_errors(t, self.Y, self.seen, self.inj)
        atoms0 = self.allow_atom.keep(candidate_atoms(self.stocks, self.base, None))
        if n >= 3:
            atoms0 = screened(atoms0, self.problem(spans_to_try(t)[0], 0, mask))
        if self.conf:
            atoms0 = [a.with_(era=c) for a in atoms0 for c in range(max(self.bucket) + 1)]
        self.atoms0 = atoms0
        # how the points report the stocks: a level at their time, or the mean over the window ending there
        best = None
        for span in spans_to_try(t):
            pr = self.problem(span, 0, mask)
            cur = self.first_fit(pr)
            self.trace.append(f"span {span}: nll {cur.nll:.1f}")
            if best is None or cur.nll < best[1].nll:
                best = (pr, cur)
        pr, cur = best
        self.stage("first fit")
        # a reporting delay, where announced injections pin it (eras, noise-free, misled it: 019)
        if self.inj:
            best_lag = (pr, cur)
            for lag in range(1, MAX_LAG + 1):
                p2 = self.problem(pr.lay.span, lag, pr.lay.mask)
                p2.refresh(cur)
                c2 = S.refit(p2, cur.atoms, cur)
                self.trace.append(f"lag {lag}: nll {c2.nll:.1f}")
                if c2.nll < best_lag[1].nll - S.price(pr):          # a delay is one more number
                    best_lag = (p2, c2)
            if best_lag[0] is not pr:
                # the atoms were chosen from the superset at lag 0: fitted again from the superset at this lag
                pr = best_lag[0]
                cur = self.first_fit(pr)
        # gross errors the filter sees (each value's standardised innovation)
        self.gross = np.zeros_like(self.seen)
        for _ in range(3):
            sm = pr.smoothed(cur)
            z = np.nan_to_num(np.abs(sm[2]))
            # a gross error is far off in its noise and also large against the level itself (a
            # model that misfits noise-free data can put a value many standard errors off)
            fit = pr._report(sm[0]) * (cur.scale if cur.scale is not None else 1.0)
            far = np.abs(self.Y - fit) > FRAC * (np.abs(fit) + 10.0)
            # and back straight away: off from both its neighbours, in the same direction, by RATIO
            # times their own difference (a pulse that decays leaves its neighbours far apart)
            Y = self.Y
            back = np.zeros_like(far)
            a, b = Y[1:-1] - Y[:-2], Y[1:-1] - Y[2:]
            back[1:-1] = (np.sign(a) == np.sign(b)) & (np.minimum(np.abs(a), np.abs(b)) > RATIO * np.abs(Y[2:] - Y[:-2]))
            hit = self.seen & (z > Z_GROSS) & far & back
            # and alone: a run of values off together is a misfit of the model (a transient it does
            # not follow), not a few independent gross errors
            nb = np.zeros_like(hit)
            nb[1:] |= hit[:-1]
            nb[:-1] |= hit[1:]
            self.gross |= hit & ~nb
            bad = pr.lay.mask & (z > Z_OUT)
            if not bad.any():
                break
            pr.lay.mask &= ~bad
            self.trace.append(f"outliers: {int(bad.sum())} more left out")
            cur = S.refit(pr, cur.atoms, cur)
        self.stage("lag and outliers")
        cur = S.free_variance(pr, cur)
        self.pr, self.cur = pr, cur
        return True

    @property
    def lag(self) -> int:
        return self.pr.lay.lag if self.pr is not None else 0

    def evidence(self) -> dict:
        """The measurement error's evidence in the superset: twice the log-likelihood it gains over
        reading every level as exact (to its rounding), and its size."""
        pr, cur = self.pr, self.cur
        keep = pr.noise
        pr.noise = ()
        exact = S.refit(pr, cur.atoms, Fitted(cur.atoms, cur.theta, cur.nll, cur.tau, 0.0, 0.0, cur.scale))
        pr.noise = keep
        lr = max(0.0, 2 * (exact.nll - cur.nll))
        # what the flows leave beyond counting noise once the error is read (a misfit of the atoms:
        # effects, co-flows, caps), as a share of each stock's step variance
        misfit = float(np.max(np.where(pr.free_tau[: pr.nm], cur.tau[: pr.nm], 0.0) / pr.step_variance(cur)[: pr.nm]))
        share = float(np.max(pr.meas_variance(cur) / pr.step_variance(cur)[: pr.nm]))
        # a stock at (or next to) zero: where a floor binds, the linear flows misfit at the step's
        # scale, and an error in the levels is the only thing the model has to explain it with
        floor = float(np.mean(np.any(self.seen & (self.Y <= 1.0), axis=1)))
        return {"lr": lr, "price": S.price(pr), "abs": math.sqrt(max(cur.alpha, 0.0)), "rel": math.sqrt(max(cur.rho2, 0.0)),
                "share": share, "floor": floor, "misfit": misfit, "points": int(pr.lay.mask.any(axis=1).sum())}

    def finish(self) -> dict:
        pr, cur = self.pr, self.cur
        cur = S.noise_free(pr, cur)
        pr.refresh(cur)
        cur = S.refit(pr, cur.atoms, cur)
        full = cur
        self.stage("noise")
        # changes in the outside inflows first, beside every flow (so that a random walk, or levels, do
        # not stand in for them); and the search without them: the lower BIC wins
        cur = _search(pr, S.pieces(pr, full, mult=0.5))
        self.stage("search with pieces first")
        other = _search(pr, full)
        if S.bic(pr, other) < S.bic(pr, cur):
            cur = other
        self.stage("search")
        pr, cur = hidden(pr, cur, _search)
        self.stage("hidden")
        cur = S.scales(pr, cur, lambda c: S.prune(pr, S.swaps(pr, S.prune(pr, c))))
        self.stage("scales")
        pr.refresh(cur)
        cur = S.refit(pr, cur.atoms, cur)
        self.pr, self.cur = pr, cur
        return self.result(full)

    def result(self, full) -> dict:
        pr, cur, ids, conf, bucket = self.pr, self.cur, self.ids, self.conf, self.bucket
        self.trace.extend(pr.notes)
        self.trace.append(f"chosen: {len(cur.atoms)} atoms, nll {cur.nll:.1f}, alpha {cur.alpha:.3g}, rho2 {cur.rho2:.3g}")
        # the flows (lna's writer): steps are grid steps 1..K-1, written as indices 0..K-2
        lay = pr.lay
        atoms = [a.with_(span=(a.span[0] - 1, a.span[1] - 1)) if a.span else a for a in cur.atoms]
        prev = pr.zref[:, 1:]
        at = lambda k: int(lay.t0 + k + 1)  # noqa: E731
        flows = to_flows(atoms, cur.theta, pr.names, prev, at, *((conf[0], conf[2], bucket) if conf else ()))
        inferred = None
        if pr.hidden:
            xs = pr.smoothed(cur)[0]
            k = lay.t - lay.t0
            inferred = {h: np.maximum(0.0, xs[pr.nm + j, k]).round(3).tolist() for j, h in enumerate(pr.hidden)}
        meas_sd = None
        if cur.alpha > 0 or cur.rho2 > 0:
            meas_sd = {"abs": math.sqrt(cur.alpha), "rel": math.sqrt(cur.rho2)}
        return {"flows": flows, "method": "kalman_lna", "nll": cur.nll, "span": lay.span, "lag": lay.lag,
                "outliers": int((self.seen & ~lay.mask).sum()), "measurement_error": meas_sd,
                "scales": {ids[s]: float(cur.scale[s]) for s in pr.scaled},
                "extra_variance": {ids[s]: float(cur.tau[s]) for s in range(len(ids)) if pr.free_tau[s]},
                "atoms": [{"kind": a.kind, "src": a.src, "dst": a.dst, "driver": a.driver, "theta": float(th)} for a, th in zip(cur.atoms, cur.theta)],
                "superset_nll": full.nll, "trace": self.trace, **({"inferred_levels": inferred, "hidden": pr.hidden} if inferred else {})}


FAILURES = (OutOfTime, np.linalg.LinAlgError, FloatingPointError, ValueError, ZeroDivisionError)


def discover(stocks: list[dict], run: dict, opts: dict, base: dict | None = None, trace: list | None = None) -> dict | None:
    """The whole search in one call; None when it cannot be made (too short, out of time, a
    numerical failure)."""
    ses = Session(stocks, run, opts, base, trace)
    try:
        if not ses.prepare():
            return None
        return ses.finish()
    except FAILURES as e:
        ses.trace.append(f"failed: {type(e).__name__}: {e}")
        return None
