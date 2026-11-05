"""The system dynamics model: stocks, flows, their equations, and the machines that may move them.
Copied from the old Orchestra's ``sd/model.py``, without eras and the declared layer.

- A **stock** is a measured level, bound to a series.
- A **flow** moves material between stocks (or from or to outside, ``None``), bound to the counter
  that counts it when one does; its **equation** records its ``origin`` (``fitted`` here, with its
  method, window and quality).
- A **machine** (a process, a source on the bus) is associated with flows only as hypotheses,
  inferred from timing (``attribute``), recorded in ``flow.evidence["machines"]``; never an equation.

The JSON form names a flow's ends ``from`` and ``to``."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class SeriesRef:
    name: str
    labels: dict[str, str] = field(default_factory=dict)
    source: str = "own"                 # the project's metrics store, or a store's name
    query: str | None = None            # how a store's series was read


@dataclass
class Equation:
    form: str                           # constant | proportional | linear
    params: dict[str, float]            # constant: {c}; proportional: {k}; linear: {c, k}; per the model's time unit
    origin: str = "fitted"
    method: str = "poisson-mle"
    window: tuple[float, float] | None = None
    quality: dict[str, float] = field(default_factory=dict)   # loglik, aic, events
    inputs: list[str] = field(default_factory=list)            # the stock ids the form reads
    spec: dict[str, Any] = field(default_factory=dict)         # the whole rate as SD Lab writes it, when it found it


@dataclass
class Stock:
    id: str
    name: str
    series: SeriesRef | None = None
    evidence: dict[str, Any] = field(default_factory=dict)    # the balance: its fit, the counters and their signs


@dataclass
class Flow:
    id: str
    name: str
    src: str | None
    dst: str | None
    counter: SeriesRef | None = None
    equations: list[Equation] = field(default_factory=list)   # the last is current
    evidence: dict[str, Any] = field(default_factory=dict)    # machines: the hypotheses (attribute)

    def current(self) -> Equation | None:
        return self.equations[-1] if self.equations else None


@dataclass
class Machine:
    id: str
    attributed: list[str] = field(default_factory=list)       # flow ids it may move: hypotheses


@dataclass
class Model:
    id: str
    name: str
    time_unit: str = "minute"
    stocks: list[Stock] = field(default_factory=list)
    flows: list[Flow] = field(default_factory=list)
    machines: list[Machine] = field(default_factory=list)
    discovery: dict[str, Any] = field(default_factory=dict)   # window, step, what was left out and why

    def stock(self, sid: str) -> Stock:
        return next(s for s in self.stocks if s.id == sid)

    def flow(self, fid: str) -> Flow:
        return next(f for f in self.flows if f.id == fid)

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        for f in d["flows"]:
            f["from"], f["to"] = f.pop("src"), f.pop("dst")
        return d

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> Model:
        def ref(r):
            return SeriesRef(**r) if r else None
        return cls(
            id=d["id"], name=d["name"], time_unit=d.get("time_unit", "minute"),
            stocks=[Stock(s["id"], s["name"], ref(s.get("series")), dict(s.get("evidence") or {})) for s in d.get("stocks", [])],
            flows=[Flow(f["id"], f["name"], f.get("from"), f.get("to"), ref(f.get("counter")),
                        [Equation(**{**e, "window": tuple(e["window"]) if e.get("window") else None}) for e in f.get("equations", [])],
                        dict(f.get("evidence") or {})) for f in d.get("flows", [])],
            machines=[Machine(m["id"], list(m.get("attributed", []))) for m in d.get("machines", [])],
            discovery=dict(d.get("discovery") or {}))

    def check(self) -> list[str]:
        """Structural problems: ids used twice, flows naming unknown stocks or connecting nothing."""
        ids = {s.id for s in self.stocks}
        fids = {f.id for f in self.flows}
        errs = [f"stock id {sid} is used {n} times" for sid, n in Counter(s.id for s in self.stocks).items() if n > 1]
        errs += [f"flow id {fid} is used {n} times" for fid, n in Counter(f.id for f in self.flows).items() if n > 1]
        errs += [f"flow {f.id}: unknown stock {end}" for f in self.flows for end in (f.src, f.dst) if end and end not in ids]
        errs += [f"machine {m.id}: attributed to unknown flow {a}" for m in self.machines for a in m.attributed if a not in fids]
        errs += [f"flow {f.id} connects nothing" for f in self.flows if not f.src and not f.dst]
        return errs


def structure(model: Model) -> list:
    """What makes a new version: the stocks, the flows' ends and their current forms and inputs."""
    def form(f: Flow):
        e = f.current()
        return (e.form, tuple(e.inputs)) if e else None
    return [sorted(s.id for s in model.stocks), sorted((f.id, f.src or "", f.dst or "", repr(form(f))) for f in model.flows)]
