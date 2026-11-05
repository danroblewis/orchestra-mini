# Vendored for kalman_lna from sdlab/discovery/strategies/common_cause/__init__.py at commit ec2f9e8 (imports rewritten to the
# vendored copies; otherwise unchanged). Do not edit: re-vendor instead.
"""Independent systems under a common cause (see docs/research/common_cause.md).

earned_pieces (the matrix's candidates, rows reselected, conservation decided jointly), with:

- exact.py: noise-free data fitted with a floor at float precision;
- places.py: every row offered pieces at the change places other rows earned, priced as values
  only (a common cause), so a cross-link must explain what they do not.
"""

from __future__ import annotations

from orchestra.sdlab.discovery.identify import VARIANTS, judge
from orchestra.sdlab.discovery.matrix import identify_one
from orchestra.sdlab.discovery.changes import pelt as _pelt
from orchestra.sdlab.discovery.strategies.kalman_lna._vendored.common_cause.exact import floored, make_pelt, make_scan, scale_of
from orchestra.sdlab.discovery.strategies.kalman_lna._vendored.common_cause.refit import refit


def identify(stocks: list[dict], run: dict, opts: dict | None = None) -> dict:
    opts = dict(opts or {})
    ids = [s["id"] for s in stocks]
    scale = scale_of(run, ids)
    exact = opts.pop("exact", True)
    if exact and "scan" not in opts:
        opts["scan"] = make_scan(scale, opts.get("min_piece", 5))
    has_eras = len({e.get("label") or e.get("start") for e in opts.get("eras") or [] if e}) >= 2
    best, tried = None, []
    with floored(scale) if exact else _null():
        for v in opts.pop("variants", None) or VARIANTS:
            if v.get("split_eras") and not has_eras:
                continue
            r = identify_one(stocks, run, {**opts, **v, "keep_fit": True, "keep_rows": True})
            if r.get("flows") is not None and r.get("fit"):
                r = refit(stocks, r, opts, make_pelt(scale) if exact else _pelt)
            e = judge(r["flows"], stocks, run, injections=opts.get("injections"))
            tried.append({"name": v["name"], "error": e})
            if best is None or e < best[1]:
                best = (r, e, v["name"])
    r = best[0]
    for k in ("fit", "ctx", "raw_rows"):
        r.pop(k, None)
    r["chosen"], r["tried"] = best[2], tried
    return r


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False
