"""The model as ``orchestra query model`` prints it: each stock with its balance, each flow with its ends,
its equation and the machines that may move it."""

from __future__ import annotations

import time

from .model import Model


def equation(f, unit: str) -> str:
    e = f.current()
    if e is None:
        return "not fitted"
    p = e.params
    x = e.inputs[0] if e.inputs else ""
    form = {"constant": f"{p.get('c', 0):.4g}", "proportional": f"{p.get('k', 0):.4g} × {x}",
            "linear": f"{p.get('c', 0):.4g} + {p.get('k', 0):.4g} × {x}"}.get(e.form, e.form)
    return f"rate = {form} per {unit} ({e.form})"


def text(model: Model, at: float) -> str:
    d = model.discovery
    lines = [f"model {model.name}, discovered {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(at))} "
             f"(version {d.get('version')}, strategy {d.get('strategy')}): {len(model.stocks)} stocks, {len(model.flows)} flows"]
    for s in model.stocks:
        lines.append(f"stock {s.id}: {s.name}  (balance R² {s.evidence.get('r2')})")
    for f in model.flows:
        lines.append(f"flow {f.id}: {f.src or 'outside'} -> {f.dst or 'outside'}  {equation(f, model.time_unit)}")
        hs = f.evidence.get("machines", {}).get("hypotheses", [])
        for h in hs:
            lines.append(f"    moved by {h['machine']}? confidence {h['confidence']:.2f}: {h['why']}")
        if not hs and f.evidence.get("machines"):
            lines.append(f"    {f.evidence['machines'].get('why')}")
    if errors := d.get("store_errors"):
        lines += [f"store {k}: {v}" for k, v in errors.items()]
    lines.append(f"left out: {len(d.get('left_out') or {})} series (orchestra query model --json says why)")
    return "\n".join(lines)
