"""The Prometheus text exposition parser (orchestra.exposition.py)."""

import pytest

from orchestra.exposition import Sample, parse, sample


def test_gauges_counters_help_and_type():
    p = parse("# HELP disk_free_bytes Free space.\n"
              "# TYPE disk_free_bytes gauge\n"
              "disk_free_bytes 1.5e+11\n"
              "\n"
              "# TYPE files_written_total counter\n"
              "files_written_total 42\n")
    assert p.samples == [Sample("disk_free_bytes", {}, 1.5e11), Sample("files_written_total", {}, 42.0)]
    assert p.bad == [] and p.left_out == []


def test_labels_with_escapes_spaces_and_a_trailing_comma():
    s = sample(r'path_bytes{path="C:\\dir",note="say \"hi\"\nthere", empty="" ,} 7')
    assert s.labels == {"path": "C:\\dir", "note": 'say "hi"\nthere', "empty": ""}
    assert s.value == 7.0
    assert sample('up{} 1').labels == {}
    assert sample('x{a="}{,="} 2').labels == {"a": "}{,="}


def test_timestamps_are_milliseconds():
    assert sample("x 1 1761500000123").ts_ms == 1761500000123
    assert sample("x -2.5e-3 -5").ts_ms == -5
    assert sample("x 1").ts_ms is None


def test_comments_and_eof_are_ignored():
    p = parse("# a comment\n#no space\n# EOF\n  \nx 1\n")
    assert [s.name for s in p.samples] == ["x"] and p.bad == []


@pytest.mark.parametrize("line", [
    "1x 3",                       # a name cannot start with a digit
    "x",                          # no value
    "x one",                      # not a number
    "x 1 2 3",                    # too much
    "x 1 12.5",                   # a timestamp is whole milliseconds
    'x{a=1} 1',                   # an unquoted value
    'x{a="1} 1',                  # an unclosed value
    'x{a="1" b="2"} 1',           # no comma
    'x{a="1",a="2"} 1',           # a label twice
    r'x{a="\t"} 1',               # an escape the format does not have
    'x{1a="1"} 1',                # a bad label name
    "x-y 1",                      # a bad name
])
def test_garbage_lines_are_skipped_and_reported(line):
    p = parse(f"good 1\n{line}\nalso_good 2\n")
    assert [s.name for s in p.samples] == ["good", "also_good"]
    assert len(p.bad) == 1 and p.bad[0][0] == 2


def test_histograms_and_summaries_keep_only_count_and_sum():
    p = parse("# TYPE req_seconds histogram\n"
              'req_seconds_bucket{le="0.1"} 3\n'
              'req_seconds_bucket{le="+Inf"} 5\n'
              "req_seconds_sum 0.9\n"
              "req_seconds_count 5\n"
              "# TYPE rpc_seconds summary\n"
              'rpc_seconds{quantile="0.5"} 0.2\n'
              "rpc_seconds_sum 4\n"
              "rpc_seconds_count 9\n")
    assert [s.name for s in p.samples] == ["req_seconds_sum", "req_seconds_count", "rpc_seconds_sum", "rpc_seconds_count"]
    assert p.left_out == ["req_seconds_bucket", "rpc_seconds"]


def test_values_that_are_not_numbers_are_left_out():
    p = parse("a NaN\nb +Inf\nc -Inf\nd 3\n")
    assert [s.name for s in p.samples] == ["d"]
    assert p.left_out == ["a", "b", "c"] and p.bad == []


def test_untyped_and_colons():
    assert parse("job:rate:5m{job=\"x\"} 0.5\n").samples == [Sample("job:rate:5m", {"job": "x"}, 0.5)]
