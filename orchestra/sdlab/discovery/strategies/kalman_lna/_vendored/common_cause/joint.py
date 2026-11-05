# Vendored for kalman_lna from sdlab/discovery/strategies/common_cause/joint.py at commit ec2f9e8 (imports rewritten to the
# vendored copies; otherwise unchanged). Do not edit: re-vendor instead.
"""(earned_pieces, vendored at a3a5e01) Conservation decided by all the rows at once, with counting noise as evidence.

The matrix fits each stock's row alone and then sets a drain to what the receivers are seen to take
unless the source's row clearly objects (matrix.conserve). Two things are lost that way:

- the receivers' numbers are measured too: when the source's row and the receivers' rows disagree,
  the shared number (the move) should come from both, each by its precision;
- whole things move (counting noise), so a row's residual variance is the stock's gross throughput:
  everything in plus everything out. An exit beside a move changes the throughput the source's row
  claims, and that is seen in the residuals even when the levels barely move (a stock that sits at its
  equilibrium, where an inflow and a drain trade off one for one).

Here every row keeps the columns and outside-inflow pieces it was fitted with; the numbers are
fitted jointly by weighted least squares, each step weighted by 1 / its throughput, and with a set
of ties: for a tied source a (in each configuration), a's own number is minus the sum of what its
same-unit receivers gain from a (no exit to outside). Each source is tied or freed by BIC on the
counting likelihood: residuals Gaussian with variance phi x gross, phi shared by the stocks of a
unit (the noise process is the same), so -2 log L = sum log(phi g) + e^2 / (phi g).
"""

from __future__ import annotations

import math

import numpy as np

from orchestra.sdlab.discovery.changes import pelt

FLOOR = 0.5          # the least throughput a step can have (half a thing): counting.gross's floor


def _weighted_demean(a: np.ndarray, w: np.ndarray, cuts: list[int] | None) -> np.ndarray:
    """a (columns x steps, or steps) less each piece's weighted mean; unchanged without pieces."""
    if not cuts:
        return a
    out = np.array(a, dtype=float, copy=True)
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        ww = w[lo:hi]
        m = (out[..., lo:hi] * ww).sum(-1) / ww.sum()
        out[..., lo:hi] -= m[..., None] if out.ndim > 1 else m
    return out


def _piece_means(r: np.ndarray, w: np.ndarray, cuts: list[int] | None) -> np.ndarray:
    """Each step's outside inflow: its piece's weighted mean of r (0 without pieces)."""
    u = np.zeros_like(r)
    if cuts:
        for lo, hi in zip(cuts[:-1], cuts[1:]):
            u[lo:hi] = (r[lo:hi] * w[lo:hi]).sum() / w[lo:hi].sum()
    return u


class System:
    """The rows (as matrix.identify_one fitted them: columns X, chosen set, col -> (stock index,
    configuration or -1), cuts) and the system they share (ids, units, configurations)."""

    def __init__(self, rows: list[dict], ids: list[str], unit: dict, final: set, dy: np.ndarray, cfg: np.ndarray, L: int,
                 counting: bool = False, min_piece: int = 5, scale: str = "unit"):
        self.min_piece, self.scale = min_piece, scale     # scale: one phi per unit's rows, or per row
        self.counting = counting          # False: stacked least squares (one variance for a unit's rows)
        self.rows, self.ids, self.unit, self.final, self.dy, self.cfg, self.L = rows, ids, unit, final, dy, cfg, L
        self.S, self.T = len(ids), dy.shape[1]
        # every free number: (row index, column index q in the row's X)
        self.free = [(s, q) for s, r in enumerate(rows) for q in r["set"]]

    def configs(self, a: int) -> list[int]:
        """The configurations in which source a can be tied: its own level is in its row and some
        same-unit row gains from it."""
        if self.ids[a] in self.final:
            return []
        return [c for c in range(self.L) if self._reading(a, a, c) and any(self._reading(s, a, c) for s in self._same_unit(a))]

    def ties(self) -> list[int]:
        """The sources a tie can be made for (in every configuration it can be made in at once)."""
        return [a for a in range(self.S) if self.configs(a)]

    def changing(self, a: int) -> bool:
        """Whether row a's outside inflow is in pieces."""
        cuts = self.rows[a]["cuts"]
        return bool(cuts) and len(cuts) > 2

    def own(self, fit: dict, a: int, c: int) -> float:
        """a's own number in configuration c, in a fit (minus its drain)."""
        r = self.rows[a]
        return float(sum(fit["beta"][a][r["set"].index(q)] for q in self._reading(a, a, c)))

    def gain(self, fit: dict, a: int, c: int) -> float:
        """What a's same-unit receivers gain per unit of a's level in configuration c, in a fit."""
        total = 0.0
        for s in self._same_unit(a):
            r = self.rows[s]
            for q in self._reading(s, a, c):
                total += float(fit["beta"][s][r["set"].index(q)])
        return total

    def _same_unit(self, a: int) -> list[int]:
        return [s for s in range(self.S) if s != a and self.unit[self.ids[s]] == self.unit[self.ids[a]]]

    def _reading(self, s: int, j: int, c: int) -> list[int]:
        """Row s's chosen columns that read stock j's level in configuration c."""
        r = self.rows[s]
        return [q for q in r["set"] if r["col"][q][0] == j and r["col"][q][1] in (-1, c)]

    def fit(self, tied: set, honest: bool = False, iters: int = 3) -> dict:
        """Numbers for every row, by weighted least squares (weights 1 / throughput, iterated), with
        a tied source's own number (in a configuration) minus what its same-unit receivers gain from
        it; the residuals, throughputs and -2 log L of the counting likelihood.

        honest: every changing row's pieces are found again beside its numbers at the textbook
        price (2 x log n a change, and 2 parameters each in the count): cheap pieces can stand in
        for a drain (or co-adapt with a biased one), so they must not decide a tie."""
        mult, per_change = (2, 2) if honest else (1, 1)
        index = {p: i for i, p in enumerate(self.free)}
        P = len(self.free)
        A = []
        for a, c in sorted((a, c) for a in tied for c in self.configs(a)):
            row = np.zeros(P)
            for q in self._reading(a, a, c):
                row[index[(a, q)]] += 1.0
            for s in self._same_unit(a):
                for q in self._reading(s, a, c):
                    row[index[(s, q)]] += 1.0
            A.append(row)
        A = np.array(A) if A else np.zeros((0, P))
        # a tied source whose outside inflow changes cannot tell its drain from the changes (pieces
        # bias it): the receivers alone measure the tied number, and the source's other numbers are
        # then fitted with it held
        held = {a for a in tied if self.changing(a)}
        cuts = [r["cuts"] for r in self.rows]
        w = [np.ones(self.T) for _ in self.rows]
        theta = np.zeros(P)
        for _ in range(iters):
            G = np.zeros((P, P))
            b = np.zeros(P)
            for s, r in enumerate(self.rows):
                cols = r["set"]
                if not cols or s in held:
                    continue
                idx = [index[(s, q)] for q in cols]
                X = _weighted_demean(r["X"][cols], w[s], cuts[s])
                y = _weighted_demean(self.dy[s], w[s], cuts[s])
                Xw = X * w[s]
                G[np.ix_(idx, idx)] += Xw @ X.T
                b[idx] += Xw @ y
            G[np.diag_indices_from(G)] += 1e-9 * (np.abs(np.diag(G)) + 1)
            K = np.block([[G, A.T], [A, np.zeros((len(A), len(A)))]])
            rhs = np.concatenate([b, np.zeros(len(A))])
            try:
                theta = np.linalg.solve(K, rhs)[:P]
            except np.linalg.LinAlgError:
                theta = np.linalg.lstsq(K, rhs, rcond=None)[0][:P]
            for a in held:
                r = self.rows[a]
                own = [q for q in r["set"] if r["col"][q][0] == a]
                rest = [q for q in r["set"] if r["col"][q][0] != a]
                y = self.dy[a] - sum((theta[index[(a, q)]] * r["X"][q] for q in own), np.zeros(self.T))
                # the pieces are found again with the drain held (found beside the row's own,
                # biased drain, they would favour it), then the row's other numbers refitted
                for _ in range(2):
                    left = y - sum((theta[index[(a, q)]] * r["X"][q] for q in rest), np.zeros(self.T))
                    cuts[a] = pelt(left, self.min_piece, mult)
                    if not rest:
                        break
                    X = _weighted_demean(r["X"][rest], w[a], cuts[a])
                    Xw = X * w[a]
                    Gr = Xw @ X.T
                    Gr[np.diag_indices_from(Gr)] += 1e-9 * (np.abs(np.diag(Gr)) + 1)
                    sol = np.linalg.lstsq(Gr, Xw @ _weighted_demean(y, w[a], cuts[a]), rcond=None)[0]
                    for q, v in zip(rest, sol):
                        theta[index[(a, q)]] = v
            if honest:
                for s, r in enumerate(self.rows):
                    if s in held or not self.changing(s):
                        continue
                    cols = r["set"]
                    left = self.dy[s] - (np.array([theta[index[(s, q)]] for q in cols]) @ r["X"][cols] if cols else 0.0)
                    cuts[s] = pelt(left, self.min_piece, mult)
            out = self._evaluate(theta, index, w, cuts)
            w = [1.0 / g for g in out["gross"]]
        out["P"] = P - len(A) + sum(1 + per_change * (len(c) - 2) if c else 0 for c in cuts)
        out["cuts"] = cuts
        out["tied"] = set(tied)
        out["theta"] = theta
        return out

    def _evaluate(self, theta, index, w, cuts) -> dict:
        betas, res, gross, inflow = [], [], [], []
        for s, r in enumerate(self.rows):
            cols = r["set"]
            beta = np.array([theta[index[(s, q)]] for q in cols]) if cols else np.zeros(0)
            fitted = beta @ r["X"][cols] if cols else np.zeros(self.T)
            u = _piece_means(self.dy[s] - fitted, w[s], cuts[s])
            e = self.dy[s] - fitted - u
            g = np.abs(u) + (np.abs(beta) @ np.abs(r["X"][cols]) if cols else 0.0)
            betas.append(beta)
            res.append(e)
            gross.append(np.maximum(FLOOR, g) if self.counting else np.ones(self.T))
            inflow.append(u)
        # phi per unit (or per row), then -2 log L
        nll = 0.0
        groups = [[s] for s in range(self.S)] if self.scale == "row" else \
            [[s for s in range(self.S) if self.unit[self.ids[s]] == un] for un in dict.fromkeys(self.unit[i] for i in self.ids)]
        for ss in groups:
            z = np.concatenate([res[s] ** 2 / gross[s] for s in ss])
            lg = sum(float(np.log(gross[s]).sum()) for s in ss)
            phi = max(float(z.mean()), 1e-300)
            nll += lg + z.size * (math.log(phi) + 1)
        return {"beta": betas, "res": res, "gross": gross, "inflow": inflow, "nll": nll}


def decide(system: System, log_t: float, trace: bool = False) -> dict:
    """Tie or free each source (in all its configurations at once), one at a time while BIC on the
    counting likelihood improves, starting from every source free; every decision is made with
    changing rows' pieces found at the textbook price (System.fit, honest). The model is then
    fitted with the chosen ties and the cheap pieces: a changing row keeps the matrix's pieces when
    free, and gets pieces found with its drain held when tied."""
    cands = system.ties()
    tied: set = set()
    best = system.fit(tied, honest=True)
    score = best["nll"] + best["P"] * log_t
    while True:
        step = None
        for t in cands:
            trial = tied ^ {t}
            f = system.fit(trial, honest=True)
            sc = f["nll"] + f["P"] * log_t
            if trace:
                print(f"    tie {system.ids[t]} {'on' if t in trial else 'off'}: {sc:.2f} vs {score:.2f}")
            if sc < score - 1e-9 and (step is None or sc < step[1]):
                step = (trial, sc, f)
        if step is None:
            break
        tied, score, best = step
    return system.fit(tied)
