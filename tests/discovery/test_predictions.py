"""Prediction judging (discovery/predictions.py): a prediction on the bus is recorded once, judged
when its time comes against the metrics, and its judgement published as orchestra.prediction.judged."""

from __future__ import annotations

import time

from orchestra import metrics
from orchestra.discovery import predictions
from orchestra.discovery.model import Equation, Flow, Model, SeriesRef, Stock
from orchestra.events import Events, utc_time
from orchestra.lines import Publisher
from orchestra.metrics import Point

T0 = time.time() // 60 * 60 - 3600


def model() -> Model:
    return Model("t", "t", "minute", [Stock("queue", "queue", SeriesRef("queue_depth", {}))],
                 [Flow("served", "served", "queue", None, SeriesRef("served_total", {}), [Equation("constant", {"c": 1.0})])])


def made(ev: Events, id: str, **data) -> None:
    ev.bus.append({"specversion": "1.0", "id": id, "type": predictions.MADE, "source": "planner", "subject": "task/1",
                   "time": utc_time(T0 + 600), "data": data})


def test_predictions_are_judged_when_due(make_project):
    p = make_project()
    store = metrics.for_project(p)
    store.write([Point("queue_depth", 10, T0, {}), Point("queue_depth", 4, T0 + 1200, {})]
                + [Point("served_total", i, T0 + 30 * i, {}) for i in range(41)])      # 2 a minute throughout
    ev = Events.for_project(p)
    made(ev, "p1", stock="queue", expect="down", within="10m")
    made(ev, "p2", stock="queue", expect="up", by=utc_time(T0 + 1200))
    made(ev, "p3", flow="served", value=2.0, within="10m")
    made(ev, "p4", flow="served", expect="same", within="10m")
    made(ev, "p5", stock="nowhere", expect="up", within="10m")
    made(ev, "p6", stock="queue", expect="sideways", within="10m")
    made(ev, "p7", stock="queue", expect="down", within="10h")
    pub = Publisher(ev)

    judged = predictions.sync(p, ev, pub, model(), now=T0 + 1300)
    verdicts = {j["data"]["prediction"]: j["data"]["verdict"] for j in judged}
    assert verdicts == {"p1": "met", "p2": "missed", "p3": "met", "p4": "met", "p5": "unjudged"}
    assert all(j["type"] == predictions.JUDGED and j["subject"] == "task/1" for j in judged)
    on_bus = [e for e in ev.all(0) if e["type"] == predictions.JUDGED]
    assert len(on_bus) == 5
    rec = p.state_dir / "model" / "predictions.json"
    assert rec.exists()
    assert predictions.sync(p, ev, pub, model(), now=T0 + 1300) == [], "each is judged once"
    import json
    got = json.loads(rec.read_text())["predictions"]
    assert got["p6"]["status"] == "invalid" and got["p7"]["status"] == "waiting"
