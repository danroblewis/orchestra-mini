# Vendored for kalman_lna from sdlab/discovery/strategies/lna/likelihood.py at commit ec2f9e8 (imports rewritten to the
# vendored copies; otherwise unchanged). Do not edit: re-vendor instead.
"""The linear noise approximation's likelihood of a run's per-step changes.

Each flow f moves q_f(t) = theta_f x z_f(t) in step t (z_f: 1 for an outside inflow, the driver's
level for a proportional flow, masked to the steps where the flow's number holds). Counting noise
draws each flow's amount around q_f on its own (Poisson: variance = mean), and every stock the flow
touches sees the same draw (v_f: -1 at the source, +1 at the destination, -per at a stock a co-flow
uses up). So a step's change is, to second order,

    dy_t ~ Normal(V q_t, V diag(q_t) V' + delta I)

(the compartmental analogue's Cov(changes) = S diag(q) S'; delta is the levels' rounding to whole
things). A move and a link with the same mean differ here: a move's draw leaves its source in the
same step it arrives, a link's does not.
"""

from __future__ import annotations

import math

import numpy as np

from orchestra.sdlab.discovery.strategies.kalman_lna._vendored.lna.small import inv_logdet

ROUNDING = 1 / 6        # the variance of a change between two levels each rounded to a whole thing


def _onehot(idx: np.ndarray, S: int) -> np.ndarray:
    out = np.zeros((S, len(idx)))
    out[idx, np.arange(len(idx))] = 1.0
    return out


def nll(theta: np.ndarray, V: np.ndarray, Z: np.ndarray, Y: np.ndarray, delta: float = ROUNDING, grad: bool = True,
        pool: np.ndarray | None = None, sign: np.ndarray | None = None, eps: float = 0.05, tau: np.ndarray | None = None,
        inflow: np.ndarray | None = None):
    """The negative log-likelihood (without the constant) and its gradient in theta.
    V: stocks x flows; Z: flows x steps; Y: stocks x steps.

    pool (optional, per atom): the stock whose outside inflow the atom is part of (-1: none). An
    outside inflow is one draw, whatever its equation: its constant (sign +1) and the terms that
    lower it as a level rises (a gap, sign -1) add up to one rate, whose variance is that rate
    (its size, kept above eps).

    tau (optional, per stock): variance beyond counting noise (what the flows do not explain: a
    misfit, an unmodelled input), so that no reading buys variance with flows that cancel. With tau,
    the gradient is returned for (theta, tau).

    inflow (optional, stocks x steps): outside inflows given as series (an inflow's pieces, whose
    atoms the caller keeps apart); the gradient in each stock's inflow at each step is then returned
    too (stocks x steps), for the caller to sum over each piece."""
    S, T = Y.shape
    F = len(theta)
    pool = np.full(F, -1) if pool is None else pool
    sign = np.ones(F) if sign is None else sign
    q = np.maximum(theta, 0.0)[:, None] * Z                       # F x T
    r = Y - V @ q                                                   # S x T
    if inflow is not None:
        r = r - inflow
    rest = pool < 0
    Vr = V[:, rest]
    K = (Vr[:, None, :] * Vr[None, :, :]).reshape(S * S, -1)        # S^2 x F: v_f v_f' flattened
    sig = (K @ q[rest]).T.reshape(T, S, S)
    pooled = ~rest
    net = _onehot(pool[pooled], S) @ (sign[pooled, None] * q[pooled]) if pooled.any() else np.zeros((S, T))
    if inflow is not None:
        net = net + inflow
    # a rate below zero is not a draw of nothing: what a gap takes beyond the inflow is noise again
    live = np.abs(net) > eps
    sig[:, np.arange(S), np.arange(S)] += np.where(live, np.abs(net), eps).T + delta + (0.0 if tau is None else np.maximum(tau, 0.0))
    W, logdet, a = inv_logdet(sig, np.ascontiguousarray(r))         # T x S x S, sum, T x S
    if not np.isfinite(logdet):
        return (math.inf, np.zeros(F + (0 if tau is None else S))) + ((np.zeros((S, T)),) if inflow is not None else ()) if grad else math.inf
    val = 0.5 * float(logdet + np.einsum("it,ti->", r, a))
    if not grad:
        return val
    Wd = W[:, np.arange(S), np.arange(S)].T                         # S x T
    va = V.T @ a.T                                                  # F x T: v_f . a_t (the mean's part)
    g = np.empty(F)
    dq = K.T @ W.reshape(T, S * S).T                                # v_f' W_t v_f (F x T)
    g[rest] = (Z[rest] * (-va[rest] + 0.5 * (dq - va[rest] ** 2))).sum(axis=1)
    if pooled.any():
        p = pool[pooled]
        dvar = sign[pooled, None] * (live * np.sign(net))[p] * (Wd[p] - a.T[p] ** 2)
        g[pooled] = (Z[pooled] * (-va[pooled] + 0.5 * dvar)).sum(axis=1)
    if tau is not None:
        g = np.concatenate([g, 0.5 * (Wd - a.T ** 2).sum(axis=1)])
    if inflow is not None:
        return val, g, -a.T + 0.5 * (live * np.sign(net)) * (Wd - a.T ** 2)
    return val, g


def gaussian_nll(Y: np.ndarray, mu: np.ndarray) -> float:
    """Homoscedastic, independent rows (for the noise-free check): 0.5 T sum_s log(rss_s / T)."""
    T = Y.shape[1]
    rss = ((Y - mu) ** 2).sum(axis=1)
    return 0.5 * T * float(np.log(np.maximum(rss / T, 1e-300)).sum())


def fisher(theta: np.ndarray, V: np.ndarray, Z: np.ndarray, Y: np.ndarray, delta: float = ROUNDING,
           pool: np.ndarray | None = None, sign: np.ndarray | None = None, eps: float = 0.05, tau: np.ndarray | None = None) -> np.ndarray:
    """The Fisher information of theta (flows x flows): sum over steps of z_a z_b (v_a' W v_b
    + 1/2 c_a c_b (u_a' W u_b)^2), with W the step's inverse covariance, v the atom's mean direction,
    u its variance direction (v, or the pooled stock for an outside inflow's parts) and c its sign
    there."""
    S, T = Y.shape
    F = len(theta)
    pool = np.full(F, -1) if pool is None else pool
    sign = np.ones(F) if sign is None else sign
    q = np.maximum(theta, 0.0)[:, None] * Z
    rest = pool < 0
    sig = np.einsum("if,ft,jf->tij", V[:, rest], q[rest], V[:, rest], optimize=True)
    net = _onehot(pool[~rest], S) @ (sign[~rest, None] * q[~rest]) if (~rest).any() else np.zeros((S, T))
    live = np.abs(net) > eps
    sig[:, np.arange(S), np.arange(S)] += np.where(live, np.abs(net), eps).T + delta + (0.0 if tau is None else np.maximum(tau, 0.0))
    W = np.linalg.inv(sig)
    U = V.copy()
    for f in np.flatnonzero(~rest):
        U[:, f] = 0.0
        U[pool[f], f] = 1.0
    c = np.ones((F, T))
    if (~rest).any():
        c[~rest] = sign[~rest, None] * (live * np.sign(net))[pool[~rest]]
    WV = np.einsum("tij,jf->tif", W, V, optimize=True)
    mean = np.einsum("ia,tib->tab", V, WV, optimize=True)                  # T x F x F
    WU = np.einsum("tij,jf->tif", W, U, optimize=True)
    var = np.einsum("ia,tib->tab", U, WU, optimize=True) ** 2
    zc = (Z * c).T                                                          # T x F
    return np.einsum("ta,tb,tab->ab", Z.T, Z.T, mean, optimize=True) + 0.5 * np.einsum("ta,tb,tab->ab", zc, zc, var, optimize=True)
