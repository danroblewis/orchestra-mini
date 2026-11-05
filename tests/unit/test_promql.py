import pytest

from orchestra.metrics.promql import PromQLError, query_range

DUMP = [
    ("orchestra_flow_total", {"project": "p", "flow": "a"}, [(0, 0), (600, 1), (1200, 3), (1800, 3), (2400, 6)]),
    ("orchestra_flow_total", {"project": "p", "flow": "b"}, [(0, 10), (1200, 2), (2400, 4)]),   # reset at 1200
    ("orchestra_stock", {"project": "p", "stock": "OPEN"}, [(0, 5), (1500, 7)]),
]


def run(q, start=0, end=2400, step=600):
    return {",".join(f"{k}={v}" for k, v in sorted(lab.items())): pts for lab, pts in query_range(DUMP, q, start, end, step)}


def test_selector_carries_the_last_value_forward():
    out = run('orchestra_stock{stock="OPEN"}')
    assert out == {"__name__=orchestra_stock,project=p,stock=OPEN": [(0, 5), (600, 5), (1200, 5), (1800, 7), (2400, 7)]}


def test_matchers():
    assert set(run('orchestra_flow_total{flow!="a"}')) == {"__name__=orchestra_flow_total,flow=b,project=p"}
    assert len(run('orchestra_flow_total{flow=~"a|b"}')) == 2
    assert run('{__name__="orchestra_stock"}')


def test_increase_and_rate_count_resets():
    out = run("increase(orchestra_flow_total[20m])", start=1200, end=2400)
    assert out["flow=a,project=p"] == [(1200, 3), (1800, 2), (2400, 3)]
    assert out["flow=b,project=p"][0] == (1200, 2)        # 10 → 2 is a reset: the counter counted 2
    rate = run("rate(orchestra_flow_total{flow=\"a\"}[20m]) * 3600", start=2400, end=2400)
    assert rate["flow=a,project=p"] == [(2400, 9.0)]


def test_aggregation():
    out = run("sum(orchestra_flow_total)", start=2400, end=2400)
    assert out == {"": [(2400, 10)]}
    out = run("sum by (flow) (orchestra_flow_total)", start=2400, end=2400)
    assert out == {"flow=a": [(2400, 6)], "flow=b": [(2400, 4)]}
    out = run("max without (flow) (orchestra_flow_total)", start=2400, end=2400)
    assert out == {"project=p": [(2400, 6)]}


def test_over_time():
    assert run("max_over_time(orchestra_stock[1h])", start=2400, end=2400) == {"project=p,stock=OPEN": [(2400, 7)]}


@pytest.mark.parametrize("q", ["sum(((", "histogram_quantile(0.9, x)", "orchestra_stock * orchestra_stock", "rate(orchestra_stock)", "3"])
def test_unsupported_or_broken_queries_explain_themselves(q):
    with pytest.raises(PromQLError):
        run(q)


def test_sum_by_name_keeps_one_series_per_metric():
    out = run('sum by (__name__) ({project="p"})', start=2400, end=2400)
    assert out == {"__name__=orchestra_flow_total": [(2400, 10)], "__name__=orchestra_stock": [(2400, 7)]}
    assert run('sum without (flow, stock) ({project="p"})', start=2400, end=2400) == {"project=p": [(2400, 17)]}


def test_the_file_store_answers_query_range_for_one_project(tmp_path):
    from orchestra.metrics import MetricsError, Point
    from orchestra.metrics.file import FileMetrics

    store = FileMetrics(tmp_path / "m.jsonl")
    store.write([Point(n, v, t, {"project": p}) for n, v, p in (("a_total", 1.0, "p"), ("b_level", 5.0, "p"), ("a_total", 9.0, "q"))
                 for t in (0, 600)])
    out = {lab["__name__"]: pts for lab, pts in store.query_range('sum by (__name__) ({__name__=~".+"})', 0, 600, 600, {"project": "p"})}
    assert out == {"a_total": [(0, 1.0), (600, 1.0)], "b_level": [(0, 5.0), (600, 5.0)]}
    with pytest.raises(MetricsError):
        store.query_range("sum by (", 0, 600, 600, {"project": "p"})
