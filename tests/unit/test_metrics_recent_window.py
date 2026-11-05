"""``VictoriaMetrics.recent`` asks the server for nothing older than ``since`` (the export's
``start``), and quotes its matchers (instruments-on-machines.md §6, test 4)."""

from __future__ import annotations

import json

import pytest

from orchestra.metrics import MetricsError, matchers_of, matches
from orchestra.metrics.victoria import VictoriaMetrics


def test_recent_exports_from_since_with_its_matchers(monkeypatch):
    vm = VictoriaMetrics("http://127.0.0.1:1")
    asked = []

    def req(path, params=None, body=None):
        asked.append((path, params))
        row = {"metric": {"__name__": "pg_up", "server": "db-1"}, "values": [1, 0, 1], "timestamps": [1000_000, 1060_000, 1120_000]}
        return (json.dumps(row) + "\n").encode()
    monkeypatch.setattr(vm, "_req", req)
    got = vm.recent("pg_", [("server", "=~", 'db-"1"|db-2'), ("job", "=", "postgres")], since=1050, keep=5)
    assert got == [("pg_up", {"server": "db-1"}, [(1060, 0), (1120, 1)])]
    path, params = asked[0]
    assert path == "/api/v1/export" and params["start"] == 1050
    assert params["match[]"] == '{__name__=~"pg_.*",server=~"db-\\"1\\"|db-2",job="postgres"}'


def test_matchers_are_checked_and_a_missing_label_is_empty():
    with pytest.raises(MetricsError):
        matchers_of([("job", "!=", "x")])
    with pytest.raises(MetricsError):
        matchers_of([("job", "=~", "(")])
    assert matches({"job": "a"}, matchers_of([("svc", "=", "")]))
    assert not matches({"job": "deploy-web-2"}, matchers_of([("job", "=~", "deploy-web")]))
