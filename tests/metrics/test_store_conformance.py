"""One conformance suite for every metrics store. The file store always runs; VictoriaMetrics
runs when ORCHESTRA_TEST_VM_URL is set (e.g. http://127.0.0.1:8428; see tests/bus/README.md for
the demo servers). Each test uses its own project label and deletes its series afterwards."""

from __future__ import annotations

import logging
import os
import time
import uuid

import pytest

from orchestra.metrics import FUTURE_SKEW_S, Point, open_metrics

BACKENDS = {"file": "file", "victoriametrics": os.environ.get("ORCHESTRA_TEST_VM_URL")}


@pytest.fixture(params=list(BACKENDS))
def store(request, tmp_path):
    url = BACKENDS[request.param]
    if not url:
        pytest.skip(f"set the URL for the {request.param} store to run it")
    s = open_metrics(url, file_path=tmp_path / "metrics.jsonl")
    project = f"test-{uuid.uuid4().hex[:10]}"
    s.project = project
    yield s
    s.delete("", {"project": project})
    s.close()


def P(store, series, value, ts, **labels):
    return Point(series, value, ts, {"project": store.project, **labels})


def test_write_and_read_raw(store):
    store.write([P(store, "orchestra_stock", v, 1000 + 10 * i, stock="OPEN") for i, v in enumerate([3, 4, 2, 5])])
    assert store.raw("orchestra_stock", {"project": store.project, "stock": "OPEN"}) == [(1000, 3), (1010, 4), (1020, 2), (1030, 5)]


def test_backfill_out_of_order(store):
    store.write([P(store, "orchestra_stock", 5, 2000, stock="DONE")])
    store.write([P(store, "orchestra_stock", 1, 1000, stock="DONE")])
    assert store.raw("orchestra_stock", {"project": store.project, "stock": "DONE"}) == [(1000, 1), (2000, 5)]


def test_range_carries_the_last_value_forward(store):
    store.write([P(store, "orchestra_stock", 3, 1000, stock="OPEN"), P(store, "orchestra_stock", 7, 1025, stock="OPEN")])
    got = store.range("orchestra_stock", {"project": store.project, "stock": "OPEN"}, 990, 1040, 10)
    assert got == [(1000, 3), (1010, 3), (1020, 3), (1030, 7), (1040, 7)], "steps before the first point are left out"


def test_latest_and_labels_keep_series_apart(store):
    store.write([P(store, "orchestra_stock", 1, 1000, stock="OPEN"), P(store, "orchestra_stock", 9, 1000, stock="DONE"),
                 P(store, "orchestra_stock", 2, 1100, stock="OPEN")])
    assert store.latest("orchestra_stock", {"project": store.project, "stock": "OPEN"}) == (1100, 2)
    assert store.latest("orchestra_stock", {"project": store.project, "stock": "DONE"}) == (1000, 9)
    names = [(n, l["stock"]) for n, l in store.series() if l.get("project") == store.project]
    assert names == [("orchestra_stock", "DONE"), ("orchestra_stock", "OPEN")]


def test_series_lists_every_name_a_machine_writes_and_filters_by_labels(store):
    store.write([P(store, "disk_free_bytes", 5e9, 1000, machine="telemetry"), P(store, "orchestra_stock", 1, 1000, stock="OPEN")])
    assert [n for n, _ in store.series("", {"project": store.project})] == ["disk_free_bytes", "orchestra_stock"]
    assert store.series("", {"project": store.project, "machine": "telemetry"}) == [
        ("disk_free_bytes", {"project": store.project, "machine": "telemetry"})]
    assert [n for n, _, _ in store.dump("", {"project": store.project})] == ["disk_free_bytes", "orchestra_stock"]


def test_delete_by_prefix_and_label(store):
    store.write([P(store, "orchestra_stock", 1, 1000, stock="OPEN"), P(store, "orchestra_flow_total", 4, 1000, flow="a>b")])
    store.delete("orchestra_flow", {"project": store.project})
    assert store.raw("orchestra_flow_total", {"project": store.project, "flow": "a>b"}) == []
    assert store.raw("orchestra_stock", {"project": store.project, "stock": "OPEN"}) == [(1000, 1)]


def test_dump_returns_every_matching_series_in_one_pass(store):
    store.write([P(store, "orchestra_stock", 1, 1000, stock="OPEN"), P(store, "orchestra_stock", 2, 1010, stock="OPEN"),
                 P(store, "orchestra_flow_total", 4, 1000, flow="a>b")])
    got = store.dump("orchestra_", {"project": store.project})
    assert [(n, l.get("stock") or l.get("flow"), pts) for n, l, pts in got] == [
        ("orchestra_flow_total", "a>b", [(1000, 4)]), ("orchestra_stock", "OPEN", [(1000, 1), (1010, 2)])]


def test_a_point_stamped_in_the_future_is_refused_and_logged(store, caplog):
    now = time.time()
    with caplog.at_level(logging.WARNING, logger="orchestra.metrics"):
        store.write([P(store, "caiso_demand_mw", 1, now - 30, machine="telemetry"),
                     P(store, "caiso_demand_mw", 2, now + FUTURE_SKEW_S / 2, machine="telemetry"),     # a clock a little ahead
                     P(store, "caiso_demand_mw", 3, now + 13 * 3600, machine="telemetry")])            # stamped tonight
    pts = store.raw("caiso_demand_mw", {"project": store.project, "machine": "telemetry"})
    assert [v for _, v in pts] == [1, 2]
    assert "refused 1 point(s)" in caplog.text and "caiso_demand_mw" in caplog.text


def test_recent_reads_only_series_with_a_point_since_and_their_last_points(store):
    """``recent`` (docs/proposals/instruments-on-machines.md §2.2): bounded by ``since``, the last
    ``keep`` points in order, ``=`` and ``=~`` matchers, the name by prefix."""
    store.write([P(store, "pg_up", v, 1000 + 60 * i, server="db-1", job="postgres") for i, v in enumerate([1, 1, 0, 1, 1])]
                + [P(store, "pg_up", 1, 1000, server="db-2", job="postgres")]                  # nothing since 1100
                + [P(store, "pg_connections", 7, 1200, server="db-3", job="postgres")]
                + [P(store, "build_seconds", 30, 1200, job="deploy-web"), P(store, "build_seconds", 40, 1210, job="deploy-api"),
                   P(store, "build_seconds", 50, 1220, job="redeploy-web")])
    mine = [("project", "=", store.project)]
    got = store.recent("pg_", mine, since=1100, keep=3)
    assert [(n, lab["server"], pts) for n, lab, pts in got] == [
        ("pg_connections", "db-3", [(1200, 7)]), ("pg_up", "db-1", [(1120, 0), (1180, 1), (1240, 1)])]
    assert store.recent("pg_up", {"project": store.project, "server": "db-1"}, since=0, keep=1) == [
        ("pg_up", {"project": store.project, "server": "db-1", "job": "postgres"}, [(1240, 1)])]
    deploys = store.recent("build_", [*mine, ("job", "=~", "deploy-.*")], since=0, keep=5)
    assert sorted(lab["job"] for _, lab, _ in deploys) == ["deploy-api", "deploy-web"], "=~ matches the whole value"
    assert store.recent("pg_", [*mine, ("job", "=", "postgres")], since=1300) == []
    assert [n for n, _, _ in store.recent("", mine, since=1215)] == ["build_seconds", "pg_up"]


def test_query_range_is_scoped_to_the_labels_and_refuses_a_bad_query(store):
    from orchestra.metrics import MetricsError
    now = time.time()
    t0 = now - 600
    store.write([P(store, "conf_disk_bytes", 10 + i, t0 + 60 * i, disk="a") for i in range(10)]
                + [P(store, "conf_disk_bytes", 100.0, t0 + 60 * i, disk="b") for i in range(10)]
                + [Point("conf_disk_bytes", 999.0, t0 + 60 * i, {"project": "someone-else", "disk": "a"}) for i in range(10)])
    try:
        out = store.query_range("sum by (__name__) (conf_disk_bytes)", t0 + 300, t0 + 540, 60, {"project": store.project})
        assert len(out) == 1
        labels, pts = out[0]
        assert labels.get("__name__") == "conf_disk_bytes"
        assert [v for _, v in pts] == [115.0, 116.0, 117.0, 118.0, 119.0]
        with pytest.raises(MetricsError):
            store.query_range("sum by (", t0, now, 60, {"project": store.project})
    finally:
        store.delete("conf_disk_bytes", {"project": "someone-else"})
