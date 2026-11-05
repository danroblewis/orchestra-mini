# Vendored for kalman_lna from sdlab/discovery/strategies/lna/fit.py at commit ec2f9e8 (imports rewritten to the
# vendored copies; otherwise unchanged). Do not edit: re-vendor instead.
"""Fitting atoms' numbers by the linear noise approximation's likelihood, and choosing which atoms
stay (backward elimination by BIC)."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
from scipy.optimize import lsq_linear, minimize

from orchestra.sdlab.discovery.strategies.kalman_lna._vendored.lna.atoms import Atom, features, pooling, stoich
from orchestra.sdlab.discovery.strategies.kalman_lna._vendored.lna.likelihood import ROUNDING, fisher, nll


TRIAL = 60         # iterations for a trial fit (a removal, a swap, a ratio tried), warm-started
FTOL = 1e-8        # relative, for a fit that is kept (a loose fit can stop in a wrong reading)
FTOL_TRIAL = 1e-6  # relative, for a trial: about 0.01 in a likelihood of 10^4, well inside the BIC price (> 2)


@dataclass
class Data:
    prev: np.ndarray              # stocks x steps: levels at the start of each step
    dy: np.ndarray                # stocks x steps: each step's change
    cfg: np.ndarray | None        # the configuration (era) in force at each step
    delta: float = ROUNDING
    free: np.ndarray | None = None   # the stocks whose variance beyond counting noise is fitted (others: 0)
    deadline: float | None = None    # time.monotonic() past which fitting stops (OutOfTime)


class OutOfTime(Exception):
    """The search ran past its budget."""


@dataclass
class Fitted:
    atoms: list[Atom]
    theta: np.ndarray
    nll: float
    tau: np.ndarray | None = None          # each stock's variance beyond counting noise

    @property
    def p(self) -> int:
        return int((self.theta > 0).sum())


def mean_start(atoms: list[Atom], d: Data) -> np.ndarray:
    """Non-negative least squares on the means, each row weighted by its spread."""
    S, T = d.dy.shape
    V, Z = stoich(atoms, S), features(atoms, d.prev, d.cfg)
    w = 1 / np.sqrt(np.maximum(d.dy.var(axis=1), 1e-9))
    D = (V[:, :, None] * Z[None, :, :]) * w[:, None, None]          # S x F x T
    A = D.transpose(0, 2, 1).reshape(S * T, len(atoms))
    b = (d.dy * w[:, None]).reshape(-1)
    sc = np.maximum(np.abs(A).max(axis=0), 1e-12)
    r = lsq_linear(A / sc, b, bounds=(0, np.inf), method="bvls" if len(atoms) < 60 else "trf")
    return r.x / sc


def fit(atoms: list[Atom], d: Data, start: np.ndarray | None = None, maxiter: int = 400, tau: np.ndarray | None = None, loose: bool = False) -> Fitted:
    """The atoms' numbers (>= 0) and each stock's extra variance, by maximum likelihood."""
    if d.deadline is not None and time.monotonic() > d.deadline:
        raise OutOfTime()
    S, T = d.dy.shape
    V, Z = stoich(atoms, S), features(atoms, d.prev, d.cfg)
    scale = np.maximum(Z.mean(axis=1), 1e-9) if atoms else np.zeros(0)   # theta x scale: the mean amount per step
    if atoms:
        x0 = (mean_start(atoms, d) if start is None else np.asarray(start, float)) * scale
        x0 = np.maximum(x0, 1e-3)
    else:
        x0 = np.zeros(0)
    vs = np.maximum(d.dy.var(axis=1), 1.0)                            # tau's scale: each stock's spread
    free = np.ones(S, bool) if d.free is None else d.free
    t0 = np.where(free, (np.zeros(S) if tau is None else np.asarray(tau, float)) / vs, 0.0)
    F = len(atoms)
    # an inflow's pieces are kept apart from the other atoms: one series per stock, built from the
    # pieces' numbers, with each piece's gradient summed over its steps
    sp = np.array([a.span is not None and a.kind == "in" and a.era is None for a in atoms], dtype=bool)
    dn = ~sp
    pool, sign = pooling([a for a, k in zip(atoms, dn) if k])
    Vd, Zd = V[:, dn], Z[dn]
    sp_stock = np.array([a.dst for a, k in zip(atoms, sp) if k], dtype=int)
    sp_a = np.array([a.span[0] for a, k in zip(atoms, sp) if k], dtype=int)
    sp_b = np.array([a.span[1] for a, k in zip(atoms, sp) if k], dtype=int)

    def f(x):
        th = x[:F] / scale
        if sp.any():
            inc = np.zeros((S, T + 1))
            np.add.at(inc, (sp_stock, sp_a), th[sp])
            np.add.at(inc, (sp_stock, sp_b), -th[sp])
            v, gd, gin = nll(th[dn], Vd, Zd, d.dy, d.delta, pool=pool, sign=sign, tau=x[F:] * vs, inflow=np.cumsum(inc, axis=1)[:, :T])
            G = np.concatenate([np.zeros((S, 1)), np.cumsum(gin, axis=1)], axis=1)
            g = np.empty(F + S)
            g[:F][dn] = gd[:-S]
            g[:F][sp] = G[sp_stock, sp_b] - G[sp_stock, sp_a]
            g[F:] = gd[-S:]
        else:
            v, g = nll(th, V, Z, d.dy, d.delta, pool=pool, sign=sign, tau=x[F:] * vs)
        if not np.isfinite(v):
            return 1e300, np.zeros_like(x)
        return v, np.concatenate([g[:F] / scale, g[F:] * vs])

    r = minimize(f, np.concatenate([x0, t0]), jac=True, method="L-BFGS-B", bounds=[(0, None)] * F + [(0, None) if f else (0, 0) for f in free], options={"maxiter": maxiter, "ftol": FTOL_TRIAL if loose or maxiter <= TRIAL else FTOL, "gtol": 1e-5})
    th = r.x[:F] / scale
    th[r.x[:F] < 1e-6] = 0.0
    return Fitted(list(atoms), th, float(r.fun), r.x[F:] * vs)


def extra_variance(cur: Fitted, d: Data, price: float | None = None) -> np.ndarray:
    """Which stocks need variance beyond counting noise: those where fixing it at 0 costs more than
    the BIC price (so a model that fits as counting noise predicts reads the noise at full strength)."""
    S, T = d.dy.shape
    price = 0.5 * math.log(T) if price is None else price
    free = np.ones(S, bool)
    for s in np.argsort(cur.tau if cur.tau is not None else np.zeros(S)):
        trial = free.copy()
        trial[s] = False
        g = fit(cur.atoms, Data(d.prev, d.dy, d.cfg, d.delta, trial, d.deadline), cur.theta, maxiter=TRIAL, tau=cur.tau)
        if g.nll - cur.nll < price:
            free, cur = trial, polish(g, Data(d.prev, d.dy, d.cfg, d.delta, trial, d.deadline))
    return free


def polish(g: Fitted, d: Data) -> Fitted:
    """A trial that is kept, fitted to full precision."""
    return fit(g.atoms, d, g.theta, tau=g.tau)


def numbers(atoms: list[Atom], idx: list[int]) -> int:
    """The numbers these atoms carry: one each, and a co-flow's per."""
    return sum(1 + len(atoms[i].uses) for i in idx)


def wald(cur: Fitted, d: Data) -> dict:
    """Each flow's (key's) approximate cost of removal, from the Fisher information: half the
    squared distance of its numbers from 0 in their standard errors. An inflow in pieces is not
    weighed here (its pieces are chosen apart)."""
    dense = [i for i, a in enumerate(cur.atoms) if a.span is None]
    if not dense:
        return {}
    atoms = [cur.atoms[i] for i in dense]
    V, Z = stoich(atoms, d.dy.shape[0]), features(atoms, d.prev, d.cfg)
    pool, sign = pooling(atoms)
    info = fisher(cur.theta[dense], V, Z, d.dy, d.delta, pool=pool, sign=sign, tau=cur.tau)
    cov = np.linalg.pinv(info + 1e-9 * np.eye(len(info)) * max(1e-12, np.abs(np.diag(info)).max()))
    pos = {i: k for k, i in enumerate(dense)}
    out = {}
    for key in dict.fromkeys(a.key() for a in cur.atoms):
        idx = [i for i, a in enumerate(cur.atoms) if a.key() == key]
        if any(i not in pos for i in idx):
            continue
        th = cur.theta[idx]
        c = cov[np.ix_([pos[i] for i in idx], [pos[i] for i in idx])]
        try:
            w = 0.5 * float(th @ np.linalg.solve(c, th))
        except np.linalg.LinAlgError:
            w = 0.0
        out[key] = (w / numbers(cur.atoms, idx), idx)
    return out


def prune(fitted: Fitted, d: Data, price: float | None = None, tries: int = 3) -> Fitted:
    """Flows whose removal costs less than the BIC price (log T / 2 in the negative log-likelihood,
    per number) go, one at a time, the cheapest first. A flow is all the atoms with one key (its
    numbers per era, its inflow's pieces): they go together. Atoms at zero go at once. The cheapest
    are found by the Fisher information (Wald), and only the few cheapest are refitted to check."""
    T = d.dy.shape[1]
    price = 0.5 * math.log(T) if price is None else price
    cur = fitted
    while True:
        live = [i for i, th in enumerate(cur.theta) if th > 0]
        if len(live) < len(cur.atoms):
            cur = fit([cur.atoms[i] for i in live], d, cur.theta[live], tau=cur.tau)
            continue
        if not cur.atoms:
            return cur
        order = sorted(wald(cur, d).items(), key=lambda kv: kv[1][0])
        done = False
        for key, (w, idx) in order[:tries]:
            if w > 4 * price:
                break
            rest = [j for j in range(len(cur.atoms)) if j not in idx]
            g = fit([cur.atoms[j] for j in rest], d, cur.theta[rest], maxiter=TRIAL, tau=cur.tau)
            if (g.nll - cur.nll) / numbers(cur.atoms, idx) < price:
                cur, done = polish(g, d), True
                break
        if not done:
            return cur
