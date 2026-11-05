"""The run laid on a grid of every step, and a set of atoms (lna's flow atoms) fitted to it by the
Kalman likelihood: the numbers, the variance beyond counting noise, the measurement error and the
scales of stocks counted with a bias.

Layout: the reported times t (uneven when points are missing, every `span` steps for weekly
reports) on the grid of every step from t0 - span + 1 to the last t; a point reports the level at
its time (span 1) or the mean over the `span` steps ending there. Injections are added at the start
of their step, `lag` steps after their announced time (a reporting delay shows them late); eras
change `lag` steps late too.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from ._vendored.lna.atoms import Atom, stoich
from .kf import nll_grad, smooth

EPS = 0.05            # a pooled inflow's variance is never below this (lna's)
SCALE = (0.25, 4.0)   # the range a stock's reporting scale may take


class OutOfTime(Exception):
    pass


@dataclass
class Layout:
    ids: list[str]
    t: np.ndarray                 # reported times
    Y: np.ndarray                 # reported points x measured stocks
    span: int = 1
    lag: int = 0
    injections: list | None = None
    eras: list | None = None
    mask: np.ndarray | None = None        # points x stocks: seen (False: an outlier left out)
    t0: int = field(init=False)
    K: int = field(init=False)

    def __post_init__(self):
        self.t = np.asarray(self.t, dtype=np.int64)
        self.Y = np.asarray(self.Y, dtype=float)
        self.t0 = int(self.t[0]) - (self.span - 1)
        self.K = int(self.t[-1]) - self.t0 + 1
        if self.mask is None:
            self.mask = np.isfinite(self.Y)
        self.Y = np.nan_to_num(self.Y)
        self.obs_at = np.full(self.K, -1, dtype=np.int64)
        self.obs_at[self.t - self.t0] = np.arange(len(self.t))
        self.reset = np.array([k % self.span == 0 for k in range(self.K)], dtype=np.bool_)
        n = len(self.ids)
        self.inj = np.zeros((self.K, n))
        for j in self.injections or []:
            if j.get("stock") in self.ids:
                k = int(j["t"]) + self.lag - self.t0
                if 1 <= k < self.K:
                    self.inj[k, self.ids.index(j["stock"])] += float(j["amount"])
        self.meas = np.arange(n, dtype=np.int64)

    @property
    def grid(self) -> np.ndarray:
        """The time each grid point stands for."""
        return self.t0 + np.arange(self.K)

    def era_cfg(self, conf) -> np.ndarray | None:
        """Each grid step's configuration (an era label's index), `lag` steps late; None without eras."""
        if not conf:
            return None
        labels, _, spans = conf
        true_time = self.grid - self.lag
        cfg = np.zeros(self.K, dtype=np.int64)
        for sp in spans:
            cfg[true_time >= sp["from"]] = labels.index(sp["label"])
        return cfg

    def interpolated(self) -> np.ndarray:
        """A first reference path (stocks x grid): the reported levels interpolated over every step,
        robustly smoothed a little (a median of 5)."""
        out = np.zeros((len(self.ids), self.K))
        g = np.arange(self.K)
        centre = self.t - self.t0 - (self.span - 1) / 2.0
        for r in range(len(self.ids)):
            m = self.mask[:, r]
            y = self.Y[m, r]
            if len(y) >= 5:
                y = np.array([np.median(y[max(0, i - 2): i + 3]) for i in range(len(y))])
            out[r] = np.interp(g, centre[m], y)
        return np.maximum(out, 0.0)


@dataclass
class Fitted:
    atoms: list[Atom]
    theta: np.ndarray
    nll: float
    tau: np.ndarray
    alpha: float = 0.0
    rho2: float = 0.0
    scale: np.ndarray | None = None
    info: dict = field(default_factory=dict)

    def numbers(self, free_tau: np.ndarray, noise: tuple, scaled: tuple) -> int:
        """Counted numbers: atoms (a co-flow's per too), freed taus, the noise numbers, the scales."""
        return sum(1 + len(a.uses) for a in self.atoms) + int(np.sum(free_tau)) + len(noise) + len(scaled)


class Problem:
    """The data on the grid and everything fixed while numbers are fitted: the era configuration,
    the reference path the counting noise is evaluated at, which numbers are free."""

    def __init__(self, lay: Layout, cfg: np.ndarray | None, deadline: float | None = None, hidden: list[str] | None = None,
                 hidden_ref: np.ndarray | None = None):
        """hidden: names of unmeasured stocks, states after the measured ones (hidden_ref: their
        first reference path, hidden x grid)."""
        self.lay = lay
        self.cfg = cfg
        self.nm = len(lay.ids)
        self.hidden = list(hidden or [])
        self.names = list(lay.ids) + self.hidden
        self.n = len(self.names)
        self.deadline = deadline
        self.inj = np.zeros((lay.K, self.n))
        self.inj[:, : self.nm] = lay.inj
        ref = lay.interpolated()
        if self.hidden:
            hr = np.zeros((len(self.hidden), lay.K)) if hidden_ref is None else np.asarray(hidden_ref, dtype=float)
            ref = np.vstack([ref, hr])
        self.set_reference(ref)
        y0 = lay.Y[0]
        self.m0rep = np.concatenate([np.where(lay.mask[0], y0, ref[: self.nm, 0]), ref[self.nm:, 0]])
        self.P0rep = (0.5 * np.abs(self.m0rep) + 10.0) ** 2
        self.P0rep[self.nm:] = (2.0 * np.abs(self.m0rep[self.nm:]) + 10.0) ** 2
        self.free_tau = np.arange(self.n) < self.nm
        self.noise: tuple = ("alpha", "rho2")         # measurement-error numbers fitted
        self.scaled: tuple = ()                        # measured rows whose scale is fitted
        sp = max(1.0, float(np.mean(np.abs(lay.Y))))
        self.level = sp
        self.evals = 0
        self.method = "scoring"
        self.notes: list[str] = []

    def set_reference(self, ref: np.ndarray, rep: np.ndarray | None = None) -> None:
        """ref: stocks x grid, the levels the counting noise is evaluated at (z at step k is the
        driver's level at k-1, plus the step's injection); rep: the reported points' reference
        levels (points x stocks) for the relative error."""
        self.ref = np.asarray(ref, dtype=float)
        z = np.zeros_like(ref)
        z[:, 1:] = ref[:, :-1] + self.inj[1:].T
        z[:, 0] = ref[:, 0]
        self.zref = np.ascontiguousarray(np.maximum(z, 0.0))
        if rep is None:
            rep = self._report(ref)
        self.yref2 = np.ascontiguousarray(rep ** 2)

    def _report(self, ref: np.ndarray) -> np.ndarray:
        lay = self.lay
        ref = ref[: self.nm]
        k = lay.t - lay.t0
        if lay.span > 1:
            c = np.cumsum(np.concatenate([np.zeros((ref.shape[0], 1)), ref], axis=1), axis=1)
            return ((c[:, k + 1] - c[:, k + 1 - lay.span]) / lay.span).T
        return ref[:, k].T

    # ---- arrays for the filter ----
    def arrays(self, atoms: list[Atom]):
        n, K = self.n, self.lay.K
        V = np.ascontiguousarray(stoich(atoms, n).T) if atoms else np.zeros((0, n))
        drv = np.array([-1 if a.driver is None else a.driver for a in atoms], dtype=np.int64)
        pool = np.array([a.dst if a.kind in ("in", "gap") else -1 for a in atoms], dtype=np.int64)
        sign = np.array([-1.0 if a.kind == "gap" else 1.0 for a in atoms])
        act = np.ones((len(atoms), K))
        for f, a in enumerate(atoms):
            if a.era is not None and self.cfg is not None:
                act[f] = self.cfg == a.era
            if a.span is not None:
                m = np.zeros(K)
                m[a.span[0]:a.span[1]] = 1.0
                act[f] *= m
        act[:, 0] = 0.0
        return V, drv, pool, sign, np.ascontiguousarray(act)

    def params(self, atoms: list[Atom]):
        """(ptype, pidx, unit) of every free number."""
        pt, pi, unit = [], [], []
        means = self.zref[:, 1:].mean(axis=1)
        for f, a in enumerate(atoms):
            pt.append(0)
            pi.append(f)
            unit.append(1.0 if a.driver is None else 1.0 / max(1e-6, float(means[a.driver])))
        for s in range(self.n):
            if self.free_tau[s]:
                pt.append(1)
                pi.append(s)
                unit.append(max(1.0, float(means[s])))
        if "alpha" in self.noise:
            pt.append(2)
            pi.append(0)
            unit.append(self.level)
        if "rho2" in self.noise:
            pt.append(3)
            pi.append(0)
            unit.append(1e-3)
        for r in self.scaled:
            pt.append(4)
            pi.append(r)
            unit.append(1.0)
        return np.array(pt, dtype=np.int64), np.array(pi, dtype=np.int64), np.array(unit)

    def unpack(self, atoms, x, pt, pi, unit, base: Fitted | None = None):
        v = x * unit
        theta = np.zeros(len(atoms))
        tau = np.zeros(self.n)
        alpha, rho2 = 0.0, 0.0
        scale = np.ones(self.nm) if base is None or base.scale is None else base.scale.copy()
        for k, (a, b) in enumerate(zip(pt, pi)):
            if a == 0:
                theta[b] = v[k]
            elif a == 1:
                tau[b] = v[k]
            elif a == 2:
                alpha = v[k]
            elif a == 3:
                rho2 = v[k]
            else:
                scale[b] = v[k]
        return theta, tau, alpha, rho2, scale

    def evaluate(self, atoms, theta, tau, alpha, rho2, scale, pt=None, pi=None, grad=False, info=False):
        lay = self.lay
        V, drv, pool, sign, act = self.arrays(atoms)
        if pt is None:
            pt, pi = np.zeros(0, np.int64), np.zeros(0, np.int64)
        self.evals += 1
        return nll_grad(np.asarray(theta, float), np.asarray(tau, float), float(alpha), float(rho2), np.asarray(scale, float),
                        V, drv, pool, sign, act, self.zref, self.inj, EPS, lay.obs_at, lay.Y, lay.mask, lay.meas, lay.span,
                        lay.reset, self.yref2, self.m0rep, self.P0rep, pt, pi, grad or info, info)

    def fit(self, atoms: list[Atom], start: Fitted | None = None, theta0=None, maxiter: int = 300, ftol: float = 1e-9) -> Fitted:
        """Every free number by maximum likelihood (L-BFGS-B with the filter's gradient)."""
        if self.deadline is not None and time.monotonic() > self.deadline:
            raise OutOfTime()
        pt, pi, unit = self.params(atoms)
        x0 = np.zeros(len(pt))
        for k, (a, b) in enumerate(zip(pt, pi)):
            if a == 0:
                v = theta0[b] if theta0 is not None else 0.01 / unit[k]
            elif a == 1:
                v = start.tau[b] if start is not None else 0.1 * unit[k]
            elif a == 2:
                v = start.alpha if start is not None else 0.5
            elif a == 3:
                v = start.rho2 if start is not None else 1e-4
            else:
                v = start.scale[self.lay.meas[b]] if start is not None and start.scale is not None else 1.0
            x0[k] = v / unit[k]
        x0 = np.where(pt == 4, x0, np.maximum(x0, 1e-4))
        base = start
        bounds = [(SCALE[0], SCALE[1]) if a == 4 else (0.0, None) for a in pt]

        def f(x):
            th, ta, al, rh, sc = self.unpack(atoms, x, pt, pi, unit, base)
            v, g, _ = self.evaluate(atoms, th, ta, al, rh, sc, pt, pi, grad=True)
            if not math.isfinite(v):
                return 1e300, np.zeros_like(x)
            return v, g * unit

        if self.method == "scoring":
            xs = self._scoring(atoms, x0, pt, pi, unit, base, bounds, maxiter=max(8, maxiter // 10), ftol=ftol)
        else:
            xs = minimize(f, x0, jac=True, method="L-BFGS-B", bounds=bounds, options={"maxiter": maxiter, "ftol": ftol, "gtol": 1e-6}).x
        th, ta, al, rh, sc = self.unpack(atoms, xs, pt, pi, unit, base)
        th[th * np.maximum(1.0, 1.0 / unit[: len(atoms)]) < 1e-7] = 0.0
        val = self.evaluate(atoms, th, ta, al, rh, sc)[0]
        return Fitted(list(atoms), th, float(val), ta, float(al), float(rh), sc)

    def _scoring(self, atoms, x, pt, pi, unit, base, bounds, maxiter: int, ftol: float) -> np.ndarray:
        """Fisher scoring with bounds (a projected, damped Newton step on the free numbers, the step
        halved until the fit improves): the filter gives the information with the gradient."""
        lb = np.array([b[0] for b in bounds])
        ub = np.array([np.inf if b[1] is None else b[1] for b in bounds])
        x = np.clip(x, lb, ub)
        lam = 1e-3

        def ev(x, info):
            th, ta, al, rh, sc = self.unpack(atoms, x, pt, pi, unit, base)
            v, g, fim = self.evaluate(atoms, th, ta, al, rh, sc, pt, pi, grad=True, info=info)
            return v, g * unit, fim * np.outer(unit, unit) if info else None

        v, g, fim = ev(x, True)
        if not math.isfinite(v):
            return x
        for _ in range(maxiter):
            if self.deadline is not None and time.monotonic() > self.deadline:
                raise OutOfTime()
            act = ((x <= lb + 1e-12) & (g > 0)) | ((x >= ub - 1e-12) & (g < 0))
            fr = ~act
            if not fr.any():
                break
            Iff = fim[np.ix_(fr, fr)]
            d = np.diag(Iff).copy()
            d[d <= 0] = 1.0
            improved = False
            for _half in range(12):
                try:
                    step = np.linalg.solve(Iff + lam * np.diag(d), -g[fr])
                except np.linalg.LinAlgError:
                    lam *= 10
                    continue
                xn = x.copy()
                xn[fr] = x[fr] + step
                xn = np.clip(xn, lb, ub)
                th, ta, al, rh, sc = self.unpack(atoms, xn, pt, pi, unit, base)
                vn = self.evaluate(atoms, th, ta, al, rh, sc)[0]
                if math.isfinite(vn) and vn < v:
                    improved = True
                    break
                lam *= 8
            if not improved:
                break
            lam = max(lam / 4, 1e-6)
            done = v - vn <= ftol * max(1.0, abs(v))
            x = xn
            v, g, fim = ev(x, True)
            if done:
                break
        return x

    def step_variance(self, fit: Fitted) -> np.ndarray:
        """Each stock's mean one-step variance from counting noise and tau (lna's LNA terms)."""
        V, drv, pool, sign, act = self.arrays(fit.atoms)
        z = np.where(drv[:, None] >= 0, self.zref[np.maximum(drv, 0)], 1.0)
        q = fit.theta[:, None] * act * z                                   # atoms x grid
        out = ((V ** 2).T @ np.where(pool[:, None] < 0, q, 0.0))           # stocks x grid
        net = np.zeros_like(out)
        for f in np.flatnonzero(pool >= 0):
            net[pool[f]] += sign[f] * q[f]
        out = out + np.maximum(np.abs(net), EPS) + fit.tau[:, None]
        return out[:, 1:].mean(axis=1)

    def meas_variance(self, fit: Fitted) -> np.ndarray:
        """Each measured stock's mean measurement-error variance over its reported points."""
        return fit.alpha + fit.rho2 * self.yref2.mean(axis=0)

    def information(self, fit: Fitted):
        pt, pi, unit = self.params(fit.atoms)
        _, _, fim = self.evaluate(fit.atoms, fit.theta, fit.tau, fit.alpha, fit.rho2, fit.scale, pt, pi, info=True)
        return pt, pi, fim

    def smoothed(self, fit: Fitted):
        lay = self.lay
        V, drv, pool, sign, act = self.arrays(fit.atoms)
        sc = fit.scale if fit.scale is not None else np.ones(self.nm)
        return smooth(fit.theta, fit.tau, fit.alpha, fit.rho2, sc, V, drv, pool, sign, act, self.zref, self.inj, EPS,
                      lay.obs_at, lay.Y, lay.mask, lay.meas, lay.span, lay.reset, self.yref2, self.m0rep, self.P0rep)

    def refresh(self, fit: Fitted) -> None:
        """The reference path from the smoother of a fitted model."""
        xs = self.smoothed(fit)[0]
        sc = fit.scale if fit.scale is not None else np.ones(self.nm)
        self.set_reference(np.maximum(xs, 0.0), self._report(np.maximum(xs, 0.0)) * sc[None, :])
