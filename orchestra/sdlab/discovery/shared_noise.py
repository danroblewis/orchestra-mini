"""Shared noise: which stocks see the same draws, and at what ratio.

A flow's amount each step is one random draw (counting noise), and every stock the flow touches
sees that same draw. So the rows' unexplained changes (residuals, after the levels and the outside
inflow's pieces are taken out) carry structure the average rates cannot:

  - a move A -> B: A's and B's residuals share the draw with opposite signs, so
    cov(res_A, res_B) = -Var(draw), and the draw counts in both stocks' variances;
  - a link A => B (B's arrivals follow A's level, nothing leaves A): drawn on its own, so the
    residuals do not covary, and A's variance does not see it;
  - a co-flow (A -> B also uses up `per` of C): C's residual carries -per x the draw, so
    cov(res_C, res_B) = per x cov(res_A, res_B).

Counting noise has a variance equal to its mean (up to a dispersion fitted to the data). So every
reading of the matrix's gains (each one a move or a link) predicts the whole covariance matrix of the
residuals: sum over flows of dispersion x mean amount x v v', where v is -1 at the flow's source and
+1 at its destination. A move and a link in opposite directions between the same two stocks give the
same covariance, but not the same variances: a move out of A beyond A's own drain shows up in A's
variance. The reading that predicts the observed covariances best (each entry weighted by its own
sampling error) is kept, and a gain is decided only where flipping it costs clearly more than chance.
Optional guards (tried, see the experiment log): flips measured against the best reading's own
misfit (`misfit`), a move-one-way/link-the-other pair also tested reversed (`swap`), and variances
allowed to exceed the prediction (`one_sided`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .regression import demean


@dataclass
class Noise:
    ids: list[str]
    res: np.ndarray                      # stocks x steps
    means: np.ndarray                    # mean level of each stock at the start of a step
    dispersion: float = 1.0
    kinds: dict = field(default_factory=dict)   # (a, s) -> "move" | "link"
    tests: dict = field(default_factory=dict)   # (a, s) -> {"move": bool, "cost": cost of flipping it}

    def cov(self, i: int, j: int) -> tuple[float, float]:
        """The residuals' covariance and its standard error (robust to unequal variances)."""
        p = self.res[i] * self.res[j]
        return float(p.mean()), float(np.sqrt(max(p.var(), 1e-12) / len(p)))


def residuals(rows: list[dict], dy: np.ndarray) -> np.ndarray:
    out = []
    for s, row in enumerate(rows):
        res = dy[s] - (row["beta"] @ row["X"][row["set"]] if row["set"] else 0)
        out.append(demean(res, row["cuts"]) if row["cuts"] else res - res.mean())
    return np.array(out)


class _Budget:
    """The covariance a reading predicts: fixed flows (links into other units, outside inflows,
    growth) plus, per candidate gain, a move or a link; each source's drain beyond its moves goes
    outside."""

    def __init__(self, S: int, drains: np.ndarray, cands: list[tuple[int, int, float]], fixed: np.ndarray):
        self.S, self.drains, self.cands, self.fixed = S, drains, cands, fixed

    def predict(self, moves: np.ndarray) -> np.ndarray:
        P = self.fixed.copy()
        out = self.drains.copy()
        for (a, s, mu), m in zip(self.cands, moves):
            P[s, s] += mu
            if m:
                P[a, a] += mu
                P[a, s] -= mu
                P[s, a] -= mu
                out[a] -= mu
        P[np.diag_indices(self.S)] += np.maximum(0.0, out)
        return P


def _cost(obs: np.ndarray, w: np.ndarray, pred: np.ndarray, one_sided: bool = False) -> tuple[float, float]:
    """Weighted squared error with the dispersion fitted (clipped to a plausible range). One-sided:
    a stock's variance may exceed what the reading predicts (misfit and unmodelled flows only add
    to it), but not fall short of it."""
    phi = float(np.clip((w * obs * pred).sum() / max(1e-12, (w * pred * pred).sum()), 0.25, 4.0))
    err = obs - phi * pred
    if one_sided:
        d = np.diag_indices(len(obs))
        err[d] = np.minimum(0.0, err[d])
    return float((w * err**2).sum()), phi


def analyse(rows: list[dict], ids: list[str], dy: np.ndarray, prev: np.ndarray, unit: dict, final: set, z: float = 3.0, tol: float = 0.003, misfit: bool = False, swap: bool = False, one_sided: bool = False) -> Noise:
    S = len(ids)
    res = residuals(rows, dy)
    means = prev.mean(axis=1)
    noise = Noise(ids, res, means)
    index = {s: i for i, s in enumerate(ids)}
    # observed covariances, their sampling errors, the upper triangle only
    obs, w = np.zeros((S, S)), np.zeros((S, S))
    for i in range(S):
        for j in range(i, S):
            c, se = noise.cov(i, j)
            obs[i, j], w[i, j] = c, 1 / se**2
    drains, fixed, cands = np.zeros(S), np.zeros((S, S)), []
    for s, row in enumerate(rows):
        coef = row["coefL"][0]
        for j_id, k in coef.items():
            j = index[j_id]
            mu = abs(k) * float(means[j])
            if j == s:
                if k < 0 and ids[s] not in final:
                    drains[s] += mu
                else:
                    fixed[s, s] += mu
            elif k > tol and unit[ids[s]] == unit[j_id] and j_id not in final:
                cands.append((j, s, mu))
            else:
                fixed[s, s] += mu
        if row["inflow"]:
            fixed[s, s] += abs(float(np.mean([p["value"] for p in row["inflow"]])))
    if not cands:
        return noise
    budget = _Budget(S, drains, cands, fixed)
    tri = lambda P: np.triu(P)  # noqa: E731

    def cost(m):
        return _cost(obs, w, tri(budget.predict(m)), one_sided)

    best = None
    for start in (np.ones(len(cands), bool), np.zeros(len(cands), bool)):
        m, c = start.copy(), cost(start)[0]
        for _ in range(4 * len(cands)):
            flips = []
            for q in range(len(cands)):
                m2 = m.copy()
                m2[q] = not m2[q]
                flips.append(cost(m2)[0])
            q = int(np.argmin(flips))
            if flips[q] >= c:
                break
            m[q], c = not m[q], flips[q]
        if best is None or c < best[1]:
            best = (m, c)
    m, c = best
    noise.dispersion = cost(m)[1]
    # the best reading's own misfit per covariance entry (1 when the residuals are only counting
    # noise): a flip must cost more than chance at that scale
    scale = max(1.0, c / (S * (S + 1) / 2)) if misfit else 1.0
    where = {(a, s): q for q, (a, s, _mu) in enumerate(cands)}
    for q, (a, s, _mu) in enumerate(cands):
        m2 = m.copy()
        m2[q] = not m2[q]
        flip = cost(m2)[0] - c
        # a pair with a gain each way, read as a move one way and a link the other: their shared
        # draws give the same covariance whichever way the move runs, so the reverse reading
        # (both flipped together) must cost clearly more too
        r = where.get((s, a))
        if swap and r is not None and m[r] != m[q]:
            m2[r] = not m2[r]
            flip = min(flip, cost(m2)[0] - c)
        noise.tests[(ids[a], ids[s])] = {"move": bool(m[q]), "cost": flip, "scale": scale}
        if flip > z * z * scale:
            noise.kinds[(ids[a], ids[s])] = "move" if m[q] else "link"
    return noise


def coflows(flows: list[dict], noise: Noise, z: float = 3.0, agree: float = 2.0) -> list[dict]:
    """A stock C that drains faster when A is high (C loses q x level of A), where A has a flow
    A -> B whose draws C's residual mirrors at a steady ratio, becomes a `uses` of that flow:
    each unit moved uses up `per` of C, the ratio seen in the noise (cov(C, B) / cov(A, B)), which
    must agree with the averages' (q / the flow's k) within a factor `agree`; C must also share the
    draw with A, the same way round (both lose it: cov(C, A) = per x Var(draw), significantly)."""
    index = {s: i for i, s in enumerate(noise.ids)}
    out = list(flows)
    for f in flows:
        if f.get("to") or not f.get("driver") or f.get("note") != f"{f.get('from')} drains faster when {f['driver']} is high":
            continue
        c, a = f["from"], f["driver"]
        q = f["rate"]["terms"][0]["k"]
        best = None
        for g in out:
            if g is f or g.get("from") != a or not g.get("to") or g.get("driver") or g["rate"]["form"] != "proportional" or g["rate"].get("stock") != a:
                continue
            b, k = g["to"], g["rate"]["k"]
            ccb, se = noise.cov(index[c], index[b])
            cab, _ = noise.cov(index[a], index[b])
            cca, se_a = noise.cov(index[c], index[a])
            if ccb > -z * se or cab >= 0 or cca < z * se_a:
                continue
            per_avg, per_noise, per_a = q / k, ccb / cab, -cca / cab
            if not (1 / agree <= per_noise / per_avg <= agree and 1 / agree <= per_a / per_noise <= agree):
                continue
            if best is None or ccb / se < best[2]:
                best = (g, per_noise, ccb / se)
        if best is None:
            continue
        g, per, _ = best
        uses = [*(g.get("uses") or []), {"stock": c, "per": round(per, 4 if per < 1 else 3)}]
        out = [{**h, "uses": uses} if h is g else h for h in out if h is not f]
    return out
