"""The Kalman filter of the stock-flow state-space model, with the gradient of its negative
log-likelihood by forward sensitivities, and the Rauch-Tung-Striebel smoother.

The model, one step k (from grid time k-1 to k) at a time:

    x_k = A_k (x_(k-1) + inj_k) + u_k + w_k,      w_k ~ N(0, Q_k)
    A_k = I + sum_f theta_f act_fk v_f e_drv(f)'   (a flow driven by a level)
    u_k = sum_f theta_f act_fk v_f                  (an outside inflow)
    Q_k = sum_f theta_f act_fk z_fk v_f v_f'  (+ pooled inflows' |net| + tau)   (counting noise, LNA)

with z_fk a reference level of the flow's driver at the step's start (held fixed while the numbers
are fitted; the caller refreshes it from the smoother). A reported point i at grid time k is

    y_ir = s_r x_(k, meas_r) + v_ir                              (a level at one time), or
    y_ir = s_r mean(x_(k-span+1 .. k), meas_r) + v_ir            (a mean over the reporting window)
    v_ir ~ N(0, 1/12 + alpha + rho2 yref_ir^2)                   (rounding, absolute and relative error)

s_r a scale (a stock counted with a bias), masked points (outliers) not seen. The window mean is
followed by an accumulator beside the state (D = 2n): c_k = x_k + (c_(k-1) unless k starts a window).

Parameters p (ptype, pidx): 0 an atom's theta (pidx: atom), 1 tau (pidx: stock), 2 alpha, 3 rho2,
4 a scale (pidx: measured row). The gradient is returned for every parameter listed.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit


@njit(cache=True)
def _inv_sym(S, mo):
    """Inverse and log-determinant of the leading mo x mo block (Cholesky); logdet inf if not PD."""
    L = np.zeros((mo, mo))
    for i in range(mo):
        for j in range(i + 1):
            v = S[i, j]
            for k in range(j):
                v -= L[i, k] * L[j, k]
            if i == j:
                if v <= 0.0:
                    return np.zeros((mo, mo)), math.inf
                L[i, i] = math.sqrt(v)
            else:
                L[i, j] = v / L[j, j]
    ld = 0.0
    for i in range(mo):
        ld += 2.0 * math.log(L[i, i])
    Li = np.zeros((mo, mo))
    for i in range(mo):
        Li[i, i] = 1.0 / L[i, i]
        for j in range(i):
            v = 0.0
            for k in range(j, i):
                v -= L[i, k] * Li[k, j]
            Li[i, j] = v / L[i, i]
    W = np.zeros((mo, mo))
    for i in range(mo):
        for j in range(i + 1):
            v = 0.0
            for k in range(i, mo):
                v += Li[k, i] * Li[k, j]
            W[i, j] = v
            W[j, i] = v
    return W, ld


@njit(cache=True)
def _step_system(k, n, theta, tau, V, drv, pool, sign, act, zref, eps, A, u, Q, net):
    """A_k, u_k, Q_k (written in place) and each pooled stock's net inflow rate."""
    F = theta.shape[0]
    for i in range(n):
        u[i] = 0.0
        net[i] = 0.0
        for j in range(n):
            A[i, j] = 1.0 if i == j else 0.0
            Q[i, j] = 0.0
    for f in range(F):
        a = act[f, k]
        if a == 0.0 or theta[f] <= 0.0:
            continue
        th = theta[f] * a
        if drv[f] >= 0:
            z = zref[drv[f], k]
            if z < 0.0:
                z = 0.0
            for i in range(n):
                if V[f, i] != 0.0:
                    A[i, drv[f]] += th * V[f, i]
        else:
            z = 1.0
            for i in range(n):
                u[i] += th * V[f, i]
        if pool[f] >= 0:
            net[pool[f]] += sign[f] * th * z
        else:
            q = th * z
            for i in range(n):
                if V[f, i] != 0.0:
                    for j in range(n):
                        Q[i, j] += q * V[f, i] * V[f, j]
    for i in range(n):
        an = abs(net[i])
        Q[i, i] += (an if an > eps else eps) + tau[i] + 1e-9


@njit(cache=True)
def nll_grad(theta, tau, alpha, rho2, scale, V, drv, pool, sign, act, zref, inj, eps,
             obs_at, Y, mask, meas, span, reset, yref2, m0rep, P0rep, ptype, pidx, want_grad, want_info):
    """The negative log-likelihood (without 2 pi), its gradient in the listed parameters and, with
    want_info, their Fisher information (sum over the points of de' W de + tr(W dS W dS) / 2).
    Written without temporaries (every buffer allocated once): the sensitivities dominate the cost."""
    K = obs_at.shape[0]
    F, n = V.shape
    nm = meas.shape[0]
    aug = span > 1
    D = 2 * n if aug else n
    P = ptype.shape[0] if (want_grad or want_info) else 0
    hs = np.ones(n)
    for r in range(nm):
        hs[meas[r]] = scale[r]
    m = np.zeros(D)
    Pm = np.zeros((D, D))
    for i in range(n):
        m[i] = m0rep[i] / hs[i]
        Pm[i, i] = P0rep[i] / (hs[i] * hs[i])
        if aug:
            m[n + i] = m[i]
            Pm[n + i, n + i] = Pm[i, i]
            Pm[i, n + i] = Pm[i, i]
            Pm[n + i, i] = Pm[i, i]
    dm = np.zeros((P, D))
    dP = np.zeros((P, D, D))
    for p in range(P):
        if ptype[p] == 4:
            j = meas[pidx[p]]
            s = hs[j]
            dm[p, j] = -m0rep[j] / (s * s)
            dP[p, j, j] = -2.0 * P0rep[j] / (s * s * s)
            if aug:
                dm[p, n + j] = dm[p, j]
                dP[p, n + j, n + j] = dP[p, j, j]
                dP[p, j, n + j] = dP[p, j, j]
                dP[p, n + j, j] = dP[p, j, j]
    A = np.zeros((n, n))
    u = np.zeros(n)
    Q = np.zeros((n, n))
    net = np.zeros(n)
    T = np.zeros((D, D))
    pre = np.zeros(D)
    mnew = np.zeros(D)
    PT = np.zeros((D, D))
    Pnew = np.zeros((D, D))
    tmp = np.zeros((D, D))
    dcol = np.zeros(D)
    dmn = np.zeros(D)
    dQd = np.zeros((n, n))
    total = 0.0
    g = np.zeros(P)
    info = np.zeros((P if want_info else 0, P if want_info else 0))
    rows = np.zeros(nm, np.int64)
    col = np.zeros(nm, np.int64)
    hv = np.zeros(nm)
    hval = np.zeros(nm)
    e = np.zeros(nm)
    Sk = np.zeros((nm, nm))
    PH = np.zeros((D, nm))
    Kg = np.zeros((D, nm))
    KS = np.zeros((D, nm))
    We = np.zeros(nm)
    de = np.zeros(nm)
    dS = np.zeros((nm, nm))
    dK = np.zeros((D, nm))
    X = np.zeros((D, nm))
    KdS = np.zeros((D, nm))
    de_all = np.zeros((P if want_info else 0, nm))
    dS_all = np.zeros((P if want_info else 0, nm, nm))
    WdS = np.zeros((P if want_info else 0, nm, nm))
    Wde = np.zeros((P if want_info else 0, nm))
    for k in range(K):
        if k > 0:
            _step_system(k, n, theta, tau, V, drv, pool, sign, act, zref, eps, A, u, Q, net)
            for i in range(D):
                for j in range(D):
                    T[i, j] = 0.0
            for i in range(n):
                for j in range(n):
                    T[i, j] = A[i, j]
                    if aug:
                        T[n + i, j] = A[i, j]
                if aug and not reset[k]:
                    T[n + i, n + i] = 1.0
            for i in range(D):
                pre[i] = m[i]
            for i in range(n):
                pre[i] += inj[k, i]
            for i in range(D):
                v = 0.0
                for j in range(D):
                    v += T[i, j] * pre[j]
                mnew[i] = v
            for i in range(n):
                mnew[i] += u[i]
                if aug:
                    mnew[n + i] += u[i]
            # PT = Pm T'
            for i in range(D):
                for j in range(D):
                    v = 0.0
                    for c2 in range(D):
                        v += Pm[i, c2] * T[j, c2]
                    PT[i, j] = v
            for i in range(D):
                for j in range(D):
                    v = 0.0
                    for c2 in range(D):
                        v += T[i, c2] * PT[c2, j]
                    Pnew[i, j] = v
            for i in range(n):
                for j in range(n):
                    Pnew[i, j] += Q[i, j]
                    if aug:
                        Pnew[n + i, j] += Q[i, j]
                        Pnew[i, n + j] += Q[i, j]
                        Pnew[n + i, n + j] += Q[i, j]
            for p in range(P):
                tp = ptype[p]
                j0 = -1
                for i in range(D):
                    dcol[i] = 0.0
                for i in range(n):
                    for j in range(n):
                        dQd[i, j] = 0.0
                has_q = False
                ducon = False
                if tp == 0:
                    f = pidx[p]
                    a = act[f, k]
                    if a != 0.0:
                        if drv[f] >= 0:
                            z = zref[drv[f], k]
                            if z < 0.0:
                                z = 0.0
                            j0 = drv[f]
                        else:
                            z = 1.0
                            ducon = True
                        for i in range(n):
                            dcol[i] = a * V[f, i]
                            if aug:
                                dcol[n + i] = a * V[f, i]
                        if pool[f] >= 0:
                            s0 = pool[f]
                            if abs(net[s0]) > eps:
                                dQd[s0, s0] = sign[f] * a * z * (1.0 if net[s0] > 0 else -1.0)
                                has_q = True
                        else:
                            for i in range(n):
                                if V[f, i] != 0.0:
                                    for j in range(n):
                                        dQd[i, j] = a * z * V[f, i] * V[f, j]
                            has_q = True
                elif tp == 1:
                    dQd[pidx[p], pidx[p]] = 1.0
                    has_q = True
                # dm' = dT pre + T dm + du ; dT = dcol e_j0'
                for i in range(D):
                    v = 0.0
                    for j in range(D):
                        v += T[i, j] * dm[p, j]
                    if j0 >= 0:
                        v += dcol[i] * pre[j0]
                    elif ducon:
                        v += dcol[i]
                    dmn[i] = v
                for i in range(D):
                    dm[p, i] = dmn[i]
                # dP' = T dP T' + dT P T' + T P dT' + dQ
                for i in range(D):
                    for j in range(D):
                        v = 0.0
                        for c2 in range(D):
                            v += dP[p, i, c2] * T[j, c2]
                        tmp[i, j] = v
                for i in range(D):
                    for j in range(i, D):
                        v = 0.0
                        for c2 in range(D):
                            v += T[i, c2] * tmp[c2, j]
                        if j0 >= 0:
                            v += dcol[i] * PT[j0, j] + dcol[j] * PT[j0, i]
                        dP[p, i, j] = v
                        dP[p, j, i] = v
                if has_q:
                    for i in range(n):
                        for j in range(n):
                            q = dQd[i, j]
                            if q != 0.0:
                                dP[p, i, j] += q
                                if aug:
                                    dP[p, n + i, j] += q
                                    dP[p, i, n + j] += q
                                    dP[p, n + i, n + j] += q
            for i in range(D):
                m[i] = mnew[i]
                for j in range(D):
                    Pm[i, j] = 0.5 * (Pnew[i, j] + Pnew[j, i])
        i_obs = obs_at[k]
        if i_obs < 0:
            continue
        mo = 0
        for r in range(nm):
            if mask[i_obs, r]:
                rows[mo] = r
                mo += 1
        if mo == 0:
            continue
        for a_ in range(mo):
            r = rows[a_]
            j = meas[r]
            if aug:
                col[a_] = n + j
                hv[a_] = 1.0 / span
            else:
                col[a_] = j
                hv[a_] = 1.0
            hval[a_] = scale[r] * hv[a_]
        for i in range(D):
            for a_ in range(mo):
                PH[i, a_] = Pm[i, col[a_]] * hval[a_]
        for a_ in range(mo):
            for b_ in range(mo):
                Sk[a_, b_] = hval[a_] * PH[col[a_], b_]
            r = rows[a_]
            e[a_] = Y[i_obs, r] - hval[a_] * m[col[a_]]
            Sk[a_, a_] += 1.0 / 12.0 + alpha + rho2 * yref2[i_obs, r]
        W, ld = _inv_sym(Sk, mo)
        if not math.isfinite(ld):
            return math.inf, g, info
        quad = 0.0
        for a_ in range(mo):
            v = 0.0
            for b_ in range(mo):
                v += W[a_, b_] * e[b_]
            We[a_] = v
            quad += e[a_] * v
        total += 0.5 * (ld + quad)
        for i in range(D):
            for a_ in range(mo):
                v = 0.0
                for b_ in range(mo):
                    v += PH[i, b_] * W[b_, a_]
                Kg[i, a_] = v
        for i in range(D):
            for a_ in range(mo):
                v = 0.0
                for b_ in range(mo):
                    v += Kg[i, b_] * Sk[b_, a_]
                KS[i, a_] = v
        for p in range(P):
            tp = ptype[p]
            astar = -1
            if tp == 4:
                for a_ in range(mo):
                    if rows[a_] == pidx[p]:
                        astar = a_
            for a_ in range(mo):
                de[a_] = -hval[a_] * dm[p, col[a_]]
                if a_ == astar:
                    de[a_] -= hv[a_] * m[col[a_]]
            for a_ in range(mo):
                for b_ in range(mo):
                    v = hval[a_] * hval[b_] * dP[p, col[a_], col[b_]]
                    if a_ == astar:
                        v += hv[a_] * PH[col[a_], b_]
                    if b_ == astar:
                        v += hv[b_] * PH[col[b_], a_]
                    dS[a_, b_] = v
                if tp == 2:
                    dS[a_, a_] += 1.0
                elif tp == 3:
                    dS[a_, a_] += yref2[i_obs, rows[a_]]
            gp = 0.0
            for a_ in range(mo):
                for b_ in range(mo):
                    gp += 0.5 * W[a_, b_] * dS[b_, a_] - 0.5 * We[a_] * dS[a_, b_] * We[b_]
                gp += We[a_] * de[a_]
            g[p] += gp
            if want_info:
                for a_ in range(mo):
                    de_all[p, a_] = de[a_]
                    for b_ in range(mo):
                        dS_all[p, a_, b_] = dS[a_, b_]
            # KdS = K dS; X = dP H' + P dH' - K dS ; dK = X W
            for i in range(D):
                for a_ in range(mo):
                    v = 0.0
                    for b_ in range(mo):
                        v += Kg[i, b_] * dS[b_, a_]
                    KdS[i, a_] = v
            for i in range(D):
                for a_ in range(mo):
                    v = dP[p, i, col[a_]] * hval[a_] - KdS[i, a_]
                    if a_ == astar:
                        v += Pm[i, col[a_]] * hv[a_]
                    X[i, a_] = v
            for i in range(D):
                for a_ in range(mo):
                    v = 0.0
                    for b_ in range(mo):
                        v += X[i, b_] * W[b_, a_]
                    dK[i, a_] = v
            # dm += dK e + K de
            for i in range(D):
                v = 0.0
                for a_ in range(mo):
                    v += dK[i, a_] * e[a_] + Kg[i, a_] * de[a_]
                dm[p, i] += v
            # dP -= dK (KS)' + KS dK' + K dS K'
            for i in range(D):
                for j in range(i, D):
                    v = 0.0
                    for a_ in range(mo):
                        v += dK[i, a_] * KS[j, a_] + KS[i, a_] * dK[j, a_] + KdS[i, a_] * Kg[j, a_]
                    dP[p, i, j] -= v
                    if j != i:
                        dP[p, j, i] -= v
        if want_info and P > 0:
            # WdS_p = W dS_p and W de_q, once per point; then each pair in mo^2
            for p in range(P):
                for a_ in range(mo):
                    for b_ in range(mo):
                        v = 0.0
                        for c_ in range(mo):
                            v += W[a_, c_] * dS_all[p, c_, b_]
                        WdS[p, a_, b_] = v
                    v = 0.0
                    for b_ in range(mo):
                        v += W[a_, b_] * de_all[p, b_]
                    Wde[p, a_] = v
            for p in range(P):
                for q in range(p + 1):
                    v = 0.0
                    for a_ in range(mo):
                        v += de_all[p, a_] * Wde[q, a_]
                        for b_ in range(mo):
                            v += 0.5 * WdS[p, a_, b_] * WdS[q, b_, a_]
                    info[p, q] += v
                    if q != p:
                        info[q, p] += v
        # the update
        for i in range(D):
            v = 0.0
            for a_ in range(mo):
                v += Kg[i, a_] * e[a_]
            m[i] += v
        for i in range(D):
            for j in range(i, D):
                v = 0.0
                for a_ in range(mo):
                    v += KS[i, a_] * Kg[j, a_]
                x = 0.5 * ((Pm[i, j] - v) + (Pm[j, i] - v))
                Pm[i, j] = x
                Pm[j, i] = x
    return total, g, info


@njit(cache=True)
def smooth(theta, tau, alpha, rho2, scale, V, drv, pool, sign, act, zref, inj, eps,
           obs_at, Y, mask, meas, span, reset, yref2, m0rep, P0rep):
    """The smoothed state (stocks only, true units) at every grid time, its variance, the
    standardised innovation of every reported value (a value left out is not used, but is scored), and the smoothed mean of
    each step's process noise w_k with its smoothed variance (stocks x grid)."""
    K = obs_at.shape[0]
    F, n = V.shape
    nm = meas.shape[0]
    aug = span > 1
    D = 2 * n if aug else n
    M = Y.shape[0]
    hs = np.ones(n)
    for r in range(nm):
        hs[meas[r]] = scale[r]
    mf = np.zeros((K, D))
    Pf = np.zeros((K, D, D))
    mp = np.zeros((K, D))
    Pp = np.zeros((K, D, D))
    Ts = np.zeros((K, D, D))
    us = np.zeros((K, D))
    Qs = np.zeros((K, D, D))
    z = np.full((M, nm), np.nan)
    m = np.zeros(D)
    Pm = np.zeros((D, D))
    for i in range(n):
        m[i] = m0rep[i] / hs[i]
        Pm[i, i] = P0rep[i] / (hs[i] * hs[i])
        if aug:
            m[n + i] = m[i]
            Pm[n + i, n + i] = Pm[i, i]
            Pm[i, n + i] = Pm[i, i]
            Pm[n + i, i] = Pm[i, i]
    A = np.zeros((n, n))
    u = np.zeros(n)
    Q = np.zeros((n, n))
    net = np.zeros(n)
    for k in range(K):
        if k > 0:
            _step_system(k, n, theta, tau, V, drv, pool, sign, act, zref, eps, A, u, Q, net)
            T = np.zeros((D, D))
            for i in range(n):
                for j in range(n):
                    T[i, j] = A[i, j]
                    if aug:
                        T[n + i, j] = A[i, j]
                if aug and not reset[k]:
                    T[n + i, n + i] = 1.0
            pre = m.copy()
            for i in range(n):
                pre[i] += inj[k, i]
            dd = np.zeros(D)
            for i in range(n):
                dd[i] = u[i]
                if aug:
                    dd[n + i] = u[i]
            QQ = np.zeros((D, D))
            for i in range(n):
                for j in range(n):
                    QQ[i, j] = Q[i, j]
                    if aug:
                        QQ[n + i, j] = Q[i, j]
                        QQ[i, n + j] = Q[i, j]
                        QQ[n + i, n + j] = Q[i, j]
            m = T @ pre + dd
            Pm = T @ Pm @ T.T + QQ
            Pm = 0.5 * (Pm + Pm.T)
            Ts[k] = T
            us[k] = dd
            Qs[k] = QQ
            for i in range(n):          # the injection, as part of the offset (for the smoother)
                for j in range(D):
                    us[k, j] += T[j, i] * inj[k, i]
        mp[k] = m
        Pp[k] = Pm
        i_obs = obs_at[k]
        if i_obs >= 0:
            # every reported value's standardised innovation, those left out too (not used to update)
            for r in range(nm):
                j = meas[r]
                hcol = n + j if aug else j
                hval = scale[r] / span if aug else scale[r]
                var = hval * hval * Pm[hcol, hcol] + 1.0 / 12.0 + alpha + rho2 * yref2[i_obs, r]
                if var > 0.0:
                    z[i_obs, r] = (Y[i_obs, r] - hval * m[hcol]) / math.sqrt(var)
            rows = np.zeros(nm, np.int64)
            mo = 0
            for r in range(nm):
                if mask[i_obs, r]:
                    rows[mo] = r
                    mo += 1
            if mo > 0:
                H = np.zeros((mo, D))
                for a_ in range(mo):
                    r = rows[a_]
                    j = meas[r]
                    if aug:
                        H[a_, n + j] = scale[r] / span
                    else:
                        H[a_, j] = scale[r]
                PH = Pm @ H.T
                Sk = H @ PH
                e = np.zeros(mo)
                yp = H @ m
                for a_ in range(mo):
                    r = rows[a_]
                    e[a_] = Y[i_obs, r] - yp[a_]
                    Sk[a_, a_] += 1.0 / 12.0 + alpha + rho2 * yref2[i_obs, r]
                W, ld = _inv_sym(Sk, mo)
                if math.isfinite(ld):
                    Kg = PH @ W
                    m = m + Kg @ e
                    Pm = Pm - Kg @ Sk @ Kg.T
                    Pm = 0.5 * (Pm + Pm.T)
            # for a robust check of each value on its own: the innovation is standardised by its own
            # variance (not decorrelated), so one gross error shows in its own row
        mf[k] = m
        Pf[k] = Pm
    # RTS smoother
    ms = mf.copy()
    Ps = Pf.copy()
    wm = np.zeros((n, K))
    wv = np.zeros((n, K))
    for k in range(K - 2, -1, -1):
        Ppi = np.linalg.pinv(Pp[k + 1])
        G = Pf[k] @ Ts[k + 1].T @ Ppi
        ms[k] = mf[k] + G @ (ms[k + 1] - mp[k + 1])
        Ps[k] = Pf[k] + G @ (Ps[k + 1] - Pp[k + 1]) @ G.T
        # w_(k+1) = x_(k+1) - T x_k - d: its smoothed mean and variance (stocks only)
        wmean = ms[k + 1] - Ts[k + 1] @ ms[k] - us[k + 1]
        C = Ps[k + 1] @ G.T                               # Cov(x_(k+1), x_k | all)
        Wv = Ps[k + 1] - Ts[k + 1] @ C.T - C @ Ts[k + 1].T + Ts[k + 1] @ Ps[k] @ Ts[k + 1].T
        for i in range(n):
            wm[i, k + 1] = wmean[i]
            wv[i, k + 1] = Wv[i, i]
    xs = np.zeros((n, K))
    xv = np.zeros((n, K))
    for k in range(K):
        for i in range(n):
            xs[i, k] = ms[k, i]
            xv[i, k] = Ps[k, i, i]
    return xs, xv, z, wm, wv
