"""Discovery on a synthetic history (tests/discovery/synthetic.py): it finds the one stock and its
in- and outflows, fits them, guesses the worker behind the outflow, and saves a version only when
the structure changes."""

from __future__ import annotations

import time

import pytest
from discovery.synthetic import ARRIVALS, SERVICE, history

from orchestra import metrics
from orchestra.discovery import run_once, show
from orchestra.discovery.store import ModelStore

END = time.time() // 60 * 60


@pytest.fixture
def project(make_project):
    p = make_project({"discovery": {"window": "60m", "every": "5m"}})
    metrics.for_project(p).write(history(END))
    return p


def test_it_finds_the_stock_and_its_flows(project):
    m = run_once(project, now=END)
    assert [s.series.name for s in m.stocks] == ["queue_depth"]
    queue = m.stocks[0].id
    by = {f.counter.name: f for f in m.flows}
    assert set(by) == {"arrivals_total", "served_total"}
    assert (by["arrivals_total"].src, by["arrivals_total"].dst) == (None, queue)
    assert (by["served_total"].src, by["served_total"].dst) == (queue, None)
    assert m.stocks[0].evidence["r2"] > 0.99
    arr = by["arrivals_total"].current()
    assert arr.form == "constant" and arr.params["c"] == pytest.approx(ARRIVALS, rel=0.35)
    srv = by["served_total"].current()
    assert srv.form == "proportional" and srv.inputs == [queue] and srv.params["k"] == pytest.approx(SERVICE, rel=0.35)
    assert any("room_temperature" in k and "input" in why for k, why in m.discovery["left_out"].items())
    assert not m.check()


def test_it_guesses_the_worker_behind_the_outflow(project):
    m = run_once(project, now=END)
    by = {f.counter.name: f for f in m.flows}
    hs = by["served_total"].evidence["machines"]["hypotheses"]
    assert hs[0]["machine"] == "worker" and hs[0]["confidence"] >= 0.7
    assert "chatter" not in {h["machine"] for f in m.flows for h in f.evidence["machines"]["hypotheses"]}
    assert not by["arrivals_total"].evidence["machines"]["hypotheses"]
    assert [x.id for x in m.machines] == ["worker"] and m.machines[0].attributed == [by["served_total"].id]


def test_a_version_is_saved_only_when_the_structure_changes(project, capsys):
    run_once(project, now=END)
    run_once(project, now=END + 1)
    store = ModelStore.for_project(project)
    assert [v["version"] for v in store.versions()] == [1]
    assert show(project, []) == 0
    out = capsys.readouterr().out
    assert "1 stocks, 2 flows" in out and "moved by worker?" in out and "proportional" in out
    assert show(project, ["--versions"]) == 0 and "v1" in capsys.readouterr().out


def test_no_model_yet(make_project, capsys):
    p = make_project()
    assert show(p, []) == 1 and "no model yet" in capsys.readouterr().out


def test_discovery_reads_a_store_through_its_adapter(make_project, tmp_path, monkeypatch):
    """The same history, in another store only: the series come through the engine and the fake
    adapter (tests/fake_adapter.py), within the store's match."""
    import json
    import sys
    from pathlib import Path

    from orchestra.stores import adapter
    merged: dict = {}
    for p in history(END):
        if not p.series.startswith("orchestra_"):
            merged.setdefault((p.series, tuple(sorted(p.labels.items()))), [p.series, {**p.labels, "env": "prod"}, []])[2].append([p.ts, p.value])
    (tmp_path / "data.json").write_text(json.dumps(list(merged.values())))
    monkeypatch.setenv("FAKE_ADAPTER_DATA", str(tmp_path / "data.json"))
    monkeypatch.setenv("FAKE_ADAPTER_LOG", str(tmp_path / "asked.jsonl"))
    fake = Path(__file__).resolve().parents[1] / "fake_adapter.py"
    p = make_project({"discovery": {"window": "60m"}, "stores": [{"name": "far", "command": f"{sys.executable} {fake}",
                                                                   "match": '{env="prod"}', "rate": "100/s"}]})
    try:
        m = run_once(p, now=END)
    finally:
        adapter.stop_all()
    assert [s.series.name for s in m.stocks] == ["queue_depth"] and m.stocks[0].series.source == "far"
    assert {f.counter.name for f in m.flows} == {"arrivals_total", "served_total"}
    asked = [json.loads(x) for x in (tmp_path / "asked.jsonl").read_text().splitlines()]
    assert asked[0] == {"op": "series", "match": '{env="prod"}'} and all(r["op"] in ("series", "range") for r in asked)
