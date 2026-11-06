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

from . import __version__, hooks, metrics
from .config import ConfigError, OldConfig
from .lines import DUPLICATE, REPEAT, Publisher, fact
from .project import load, repo_root

PROGRAMS = """programs for machines (named in a machine's command):
  orchestra agent <prompt file> [--model M] [--per-day-usd N]
      one Claude Code turn per fact; a continuous machine's agent keeps one session
  orchestra decide "<question>" --answers a,b,c [--inputs proj:field,field] [--min-confidence 0.6] [--model M]
      one answer per fact: publishes decision.made and <machine>.<answer>"""


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
    """The Supervisor; with no config yet, or one that doesn't load, briefed to set the project up;
    with one written for an earlier Orchestra, briefed to rewrite it."""
    from . import setup
    from .config import Config
    from .project import Project
    brief, migrate = "chat.md", None
    try:
        project = load(args.repo)
        if not project.files:
            brief = "setup.md"
    except ConfigError as e:
        root = (args.repo or repo_root()).resolve()
        project = Project(root.name, root, Config(), [])
        if isinstance(e, OldConfig):
            migrate = str(e)
            print(f"orchestra chat: the config was written for an earlier Orchestra ({e}); the Supervisor rewrites it",
                  file=sys.stderr)
        else:
            brief = "setup.md"
            print(f"orchestra chat: the config does not load yet ({e}); setting it up", file=sys.stderr)
    return setup.run(project, " ".join(args.message) or None, print_only=args.print_prompt, brief=brief, resume=args.resume,
                     migrate=migrate)


def cmd_emit(args) -> int:
    """One fact, or one metric line, from the argument; its source is ``ORCHESTRA_SOURCE`` (a
    machine's run sets it) unless the line names one, and its cause ``ORCHESTRA_CAUSEDBY`` and
    ``ORCHESTRA_DEPTH`` (a once run sets them; depth 0 without)."""
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
    ev, why = fact(raw, source, uuid.uuid4().hex)
    if ev is None:
        print(why, file=sys.stderr)
        return 2
    try:
        depth = int(os.environ.get("ORCHESTRA_DEPTH") or 0)
    except ValueError:
        depth = 0
    if os.environ.get("ORCHESTRA_CAUSEDBY") and not ev.get("causedby"):
        ev["causedby"] = os.environ["ORCHESTRA_CAUSEDBY"]
    ev["depth"] = depth if ev.get("causedby") else 0
    pub = Publisher.for_project(project)
    done = pub.publish(ev)
    print(ev["id"] + {DUPLICATE: " (duplicate: already on the bus)", REPEAT: " (a repeat: it says nothing new)"}.get(done, ""))
    return 0


def cmd_query(args) -> int:
    from .queries import query
    return query(load(args.repo), args.what, args.rest)


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
    s = sub.add_parser("emit", help="publish one fact, or write one metric line, as ORCHESTRA_SOURCE")
    s.add_argument("json")
    s.set_defaults(fn=cmd_emit)
    s = sub.add_parser("query", help="facts | views | check | metrics | model | runs | '<promql>' [--store NAME]",
                       description="orchestra query facts [-n N] [--type T] [--subject S] [--trace ID]\n"
                                   "orchestra query views [PROJECTION] [show SUBJECT] [--input SUBJECT] [--rebuild]\n"
                                   "orchestra query check\n"
                                   "orchestra query metrics [prefix]\norchestra query model\n"
                                   "orchestra query runs [machine] [-n N]\norchestra query '<promql>' [--store NAME]",
                       formatter_class=argparse.RawDescriptionHelpFormatter)
    s.add_argument("what", help="facts, views, check, metrics, model, runs, or a PromQL query")
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
