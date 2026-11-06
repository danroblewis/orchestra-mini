"""``orchestra query`` (SPEC §5): ``facts``, ``views``, ``check``, ``metrics``, ``model``, ``runs``, or
PromQL. Every fact they print came through the one read function (``facts.read``)."""

from __future__ import annotations

import argparse
import json
import sys
import time

from . import hooks, metrics, runs
from .events import Events
from .facts import read
from .views import Builder, Views, projections, said

GONE = {"events": "facts", "items": "views"}


def trace(facts: list[dict], fact_id: str) -> list[dict]:
    """The fact, what caused it (its ``causedby``, and that one's, up), and what it caused (down),
    in log order."""
    by = {f.get("id"): f for f in facts}
    if fact_id not in by:
        return []
    keep, at = {fact_id}, by[fact_id]
    while (up := at.get("causedby")) in by and up not in keep:
        keep.add(up)
        at = by[up]
    children: dict = {}
    for f in facts:
        children.setdefault(f.get("causedby"), []).append(f.get("id"))
    todo = [fact_id]
    while todo:
        for c in children.get(todo.pop(), []):
            if c not in keep:
                keep.add(c)
                todo.append(c)
    return [f for f in facts if f.get("id") in keep]


def q_facts(project, rest) -> int:
    import fnmatch
    a = argparse.ArgumentParser(prog="orchestra query facts")
    a.add_argument("-n", type=int, default=20, help="the last N (default 20)")
    a.add_argument("--type", help="only this type (a glob may)")
    a.add_argument("--subject")
    a.add_argument("--trace", metavar="ID", help="the fact, what caused it and what it caused (every one)")
    o = a.parse_args(rest)
    found = read(Events.for_project(project))
    if o.trace:
        found = trace(found, o.trace)
        if not found:
            print(f"no fact {o.trace}", file=sys.stderr)
            return 1
    found = [f for f in found if (not o.type or fnmatch.fnmatchcase(str(f.get("type")), o.type))
             and (not o.subject or f.get("subject") == o.subject)]
    for f in found if o.trace else found[-o.n:]:
        print(json.dumps(f))
    return 0


def _summary(v: dict, views: dict) -> str:
    f = v.get("fields") or {}
    waits = said(views, "waiting_on")
    return (f"{v.get('last')}  {v.get('kind') or '?'}  {v['subject']}  {v.get('type')}" + (f"  {f['title']}" if f.get("title") else "")
            + ("  done" if said(views, "done") else "") + (f"  waits on {waits}" if waits else ""))


def q_views(project, rest) -> int:
    a = argparse.ArgumentParser(prog="orchestra query views",
                                usage="orchestra query views [PROJECTION] [show SUBJECT] [--input SUBJECT] [--rebuild]")
    a.add_argument("words", nargs="*", help="a projection's name, then show SUBJECT")
    a.add_argument("--input", metavar="SUBJECT", help="print exactly what the projection is given for the subject")
    a.add_argument("--rebuild", action="store_true", help="build the views again from the whole log")
    a.add_argument("--kind", help="only this kind (github.pr, jira.issue)")
    a.add_argument("--all", action="store_true", help="done ones too")
    a.add_argument("-n", type=int, default=20)
    a.add_argument("--page", type=int, default=0)
    o = a.parse_args(rest)
    words = list(o.words)
    name = words.pop(0) if words and words[0] != "show" else None
    subject = None
    if words:
        if words[0] != "show" or len(words) != 2:
            a.error("after the projection: show SUBJECT")
        subject = words[1]
    known = {p.name: p for p in projections(project)}
    if name is not None and name not in known:
        print(f"no projection {name} (there are: {', '.join(known)})", file=sys.stderr)
        return 1
    if o.input is not None:
        if name is None:
            a.error("--input needs a projection: orchestra query views NAME --input SUBJECT")
        from .views.builder import given
        for f in given(known[name], read(Events.for_project(project)), o.input):
            print(json.dumps(f))
        return 0
    builder = Builder(project)
    if o.rebuild:
        builder.build(rebuild={name} if name else True)
    views = Views(project, builder, build=False if o.rebuild else None)
    if subject is not None:
        found = views.of(subject)
        got = found.get(name) if name else found or None
        if got is None:
            print(f"no view of {subject}" + (f" in {name}" if name else ""), file=sys.stderr)
            return 1
        print(json.dumps(got, indent=2))
        return 0
    if name is not None:
        for v in views.all(name)[o.page * o.n:(o.page + 1) * o.n]:
            print(json.dumps(v))
        return 0
    for v in views.latest(closed=o.all, kind=o.kind)[o.page * o.n:(o.page + 1) * o.n]:
        print(_summary(v, views.of_cached(v["subject"])))
    return 0


def q_check(project, rest) -> int:
    from .logcheck import report
    found = report(project)
    for line in found:
        print(line)
    if not found:
        print("nothing to report")
    return 1 if found else 0


def q_metrics(project, rest) -> int:
    for name, labels, pts in metrics.for_project(project).recent(rest[0] if rest else "", {"project": project.name}):
        lab = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()) if k != "project")
        print(f"{name}{{{lab}}} {pts[-1][1]:g}")
    return 0


def q_model(project, rest) -> int:
    d = hooks.hook("discovery", "show")
    if d is None:
        print("discovery is not installed", file=sys.stderr)
        return 2
    return d.show(project, rest)


def q_runs(project, rest) -> int:
    a = argparse.ArgumentParser(prog="orchestra query runs")
    a.add_argument("machine", nargs="?")
    a.add_argument("-n", type=int, default=20)
    o = a.parse_args(rest)
    for r in runs.read(project, o.machine, o.n):
        print("  ".join(filter(None, [
            r.get("start"), r.get("machine"), f"exit {r.get('exit')}", f"{r.get('facts', r.get('events', 0))} facts",
            f"{r['points']} points" if r.get("points") else "", f"{r['held']} held" if r.get("held") else "",
            f"${r['cost_usd']:.4f}" if r.get("cost_usd") else "", f"for {r['subject']}" if r.get("subject") else "", r.get("note")])))
    return 0


def q_promql(project, promql: str, rest) -> int:
    a = argparse.ArgumentParser(prog="orchestra query '<promql>'")
    a.add_argument("--store", help="ask this store (through its adapter) instead of the metrics store")
    o = a.parse_args(rest)
    if o.store:
        from . import stores
        return stores.query(project, promql, o.store)
    end = time.time()
    try:
        rows = metrics.for_project(project).query_range(promql, end - 3600, end, 60, {"project": project.name})
    except metrics.MetricsError as e:
        print(e, file=sys.stderr)
        return 2
    for labels, pts in rows:
        if pts:
            print(f"{json.dumps(labels)} {pts[-1][1]:g}")
    return 0


QUERIES = {"facts": q_facts, "views": q_views, "check": q_check, "metrics": q_metrics, "model": q_model, "runs": q_runs}


def query(project, what: str, rest: list[str]) -> int:
    if what in GONE:
        print(f"orchestra query {what}: now orchestra query {GONE[what]} (SPEC §5)", file=sys.stderr)
        return 2
    if what in QUERIES:
        return QUERIES[what](project, rest)
    return q_promql(project, what, rest)
