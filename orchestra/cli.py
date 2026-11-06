"""The ``orchestra`` command (SPEC §5): four commands, ``run``, ``chat``, ``emit`` and ``query``,
and the built-in programs a machine's ``command`` names, ``agent`` and ``decide``."""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

from . import __version__, hooks, metrics, runs
from .config import ConfigError
from .events import Events
from .items import Items
from .lines import Publisher, event
from .project import load, repo_root

PROGRAMS = """programs for machines (named in a machine's command):
  orchestra agent <prompt file> [--model M] [--per-day-usd N]
      one Claude Code turn per event; a continuous machine's agent keeps one session
  orchestra decide "<question>" --answers a,b,c [--inputs f,g] [--min-confidence 0.6] [--model M]
      one answer per event: publishes decision.made and <machine>.<answer>"""


def cmd_run(args) -> int:
    from .check import problems
    from .runner import Runner
    try:
        project = load(args.repo)
    except ConfigError as e:
        print(f"config: {e}\nnot started: fix it, or ask orchestra chat to", file=sys.stderr)
        return 1
    project.state_dir.mkdir(parents=True, exist_ok=True)
    found = problems(project)
    for level, text in found:
        print(f"{level}: {text}", file=sys.stderr)
    if any(level == "error" for level, _ in found):
        print("not started: fix the errors above, or ask orchestra chat to", file=sys.stderr)
        return 1
    runner = Runner(project)
    if args.web:
        web = hooks.hook("web", "serve")
        if web is None:
            print("the pages are not installed; running without them", file=sys.stderr)
        else:
            threading.Thread(target=web.serve, args=(project, runner.stop, args.listen), daemon=True).start()
    runner.run(args.seconds)
    return 0


def cmd_chat(args) -> int:
    """The Supervisor; with no config yet, or one that doesn't load, briefed to set the project up."""
    from . import setup
    from .config import Config
    from .project import Project
    brief = "chat.md"
    try:
        project = load(args.repo)
        if not project.files:
            brief = "setup.md"
    except ConfigError as e:
        root = (args.repo or repo_root()).resolve()
        project, brief = Project(root.name, root, Config(), []), "setup.md"
        print(f"orchestra chat: the config does not load yet ({e}); setting it up", file=sys.stderr)
    return setup.run(project, " ".join(args.message) or None, print_only=args.print_prompt, brief=brief, resume=args.resume)


def cmd_emit(args) -> int:
    """One event, or one metric line, from the argument; its source is ``ORCHESTRA_SOURCE`` (a
    machine's run sets it) unless the line names one."""
    import uuid

    from .output import metric, remember_kinds
    try:
        raw = json.loads(args.json)
    except ValueError:
        raw = None
    if not isinstance(raw, dict):
        print("not a JSON object", file=sys.stderr)
        return 2
    source = os.environ.get("ORCHESTRA_SOURCE") or "cli"
    project = load(args.repo)
    if "metric" in raw and "value" in raw:
        point, why = metric(raw, os.environ.get("ORCHESTRA_MACHINE") or source, time.time())
        if point is None:
            print(why, file=sys.stderr)
            return 2
        if raw.get("kind"):
            remember_kinds(project, {point.series: raw["kind"]})
        metrics.for_project(project).write([point])
        print(f"{point.series} {point.value:g}")
        return 0
    ev, why = event(raw, source, uuid.uuid4().hex)
    if ev is None:
        print(why, file=sys.stderr)
        return 2
    fresh = Publisher(Events.for_project(project)).publish(ev)
    print(ev["id"] + ("" if fresh else " (duplicate: already on the bus)"))
    return 0


def q_events(project, rest) -> int:
    a = argparse.ArgumentParser(prog="orchestra query events")
    a.add_argument("-n", type=int, default=20)
    a.add_argument("--type")
    a.add_argument("--subject")
    o = a.parse_args(rest)
    found = [e for e in Events.for_project(project).all(0) if (not o.type or e.get("type") == o.type)
             and (not o.subject or e.get("subject") == o.subject)]
    for e in found[-o.n:]:
        print(json.dumps(e))
    return 0


def q_items(project, rest) -> int:
    a = argparse.ArgumentParser(prog="orchestra query items")
    a.add_argument("show", nargs="?", choices=["show"])
    a.add_argument("subject", nargs="?")
    a.add_argument("--kind", help="only this kind (github.pr, jira.issue)")
    a.add_argument("--all", action="store_true", help="closed ones too")
    a.add_argument("-n", type=int, default=20)
    a.add_argument("--page", type=int, default=0)
    o = a.parse_args(rest)
    items = Items(project, Events.for_project(project))
    if o.show:
        item = items.get(o.subject or "")
        if item is None:
            print(f"no item {o.subject}", file=sys.stderr)
            return 1
        print(json.dumps(item, indent=2))
        return 0
    for i in items.list(closed=o.all, kind=o.kind)[o.page * o.n:(o.page + 1) * o.n]:
        f = i.get("fields") or {}
        print(f"{i['last']}  {i.get('kind') or '?'}  {i['subject']}  {i['type']}" + (f"  {f['title']}" if f.get("title") else "")
              + ("  done" if i.get("done") else "") + (f"  waits on {f['waiting_on']}" if f.get("waiting_on") else ""))
    return 0


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
            r.get("start"), r.get("machine"), f"exit {r.get('exit')}", f"{r.get('events', 0)} events",
            f"{r['points']} points" if r.get("points") else "", f"${r['cost_usd']:.4f}" if r.get("cost_usd") else "",
            f"for {r['subject']}" if r.get("subject") else "", r.get("note")])))
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


QUERIES = {"events": q_events, "items": q_items, "metrics": q_metrics, "model": q_model, "runs": q_runs}


def cmd_query(args) -> int:
    project = load(args.repo)
    if args.what in QUERIES:
        return QUERIES[args.what](project, args.rest)
    return q_promql(project, args.what, args.rest)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="orchestra", description="Runs commands, reads their stdout, keeps the bus.",
                                epilog=PROGRAMS, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=__version__)
    p.add_argument("--repo", type=Path, help="the repository (default: found from here)")
    sub = p.add_subparsers(dest="cmd", required=True, title="commands", metavar="{run,chat,emit,query}")
    s = sub.add_parser("run", help="check the config, then run every machine; --web also serves the pages")
    s.add_argument("--web", action="store_true", help="also serve the pages")
    s.add_argument("--listen", default="127.0.0.1:8765", help="where the pages listen")
    s.add_argument("--seconds", type=float, help="stop after this long")
    s.set_defaults(fn=cmd_run)
    s = sub.add_parser("chat", help="the Supervisor: sets Orchestra up, answers about it, changes it, clears its data")
    s.add_argument("message", nargs="*", help="what you want (else it waits for you)")
    s.add_argument("--continue", dest="resume", action="store_true", help="carry on the last session in this folder")
    s.add_argument("--print-prompt", action="store_true", help="print the brief and stop")
    s.set_defaults(fn=cmd_chat)
    s = sub.add_parser("emit", help="publish one event, or write one metric line, as ORCHESTRA_SOURCE")
    s.add_argument("json")
    s.set_defaults(fn=cmd_emit)
    s = sub.add_parser("query", help="events | items | metrics | model | runs | '<promql>' [--store NAME]",
                       description="orchestra query events [-n N] [--type T] [--subject S]\n"
                                   "orchestra query items [--kind K] [--all] [show SUBJECT]\n"
                                   "orchestra query metrics [prefix]\norchestra query model\n"
                                   "orchestra query runs [machine] [-n N]\norchestra query '<promql>' [--store NAME]",
                       formatter_class=argparse.RawDescriptionHelpFormatter)
    s.add_argument("what", help="events, items, metrics, model, runs, or a PromQL query")
    s.add_argument("rest", nargs=argparse.REMAINDER)
    s.set_defaults(fn=cmd_query)
    return p


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] in (["agent"], ["decide"]):           # programs for machines, apart from the four commands
        from . import agent, decide
        return (agent if argv[0] == "agent" else decide).main(argv[1:])
    args = parser().parse_args(argv)
    try:
        return args.fn(args)
    except ConfigError as e:
        print(f"config: {e}", file=sys.stderr)
        return 1
