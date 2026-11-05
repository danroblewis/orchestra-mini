"""The ``orchestra`` command (SPEC §7)."""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time

from . import __version__, hooks, metrics, runs
from .config import ConfigError
from .events import Events
from .items import Items
from .lines import Publisher, event
from .project import load, repo_root


def _project(args):
    return load(args.repo)


def cmd_init(args) -> int:
    root = args.repo or repo_root()
    if args.config:
        path = root / "orchestra.yaml"
        if path.exists():
            print(f"{path} exists")
        else:
            path.write_text("# Orchestra's config: see SPEC.md §1.\n")
            print(f"wrote {path}")
    project = load(root)
    project.state_dir.mkdir(parents=True, exist_ok=True)
    print(f"state directory {project.state_dir}")
    return 0


# Orchestra's own data, the only things reset removes. Anything else in the state directory (the
# overlay, scripts and prompts kept beside it, a cursor a script keeps) is the owner's and is never
# touched (the owner, 2026-11-05: an earlier reset removed everything but a keep-list, and with it
# the scripts the setup agent had put there).
DATA = ("events.jsonl", "metrics.jsonl", "items.json", "cursor.json", "health.json", "endpoint.json", "web.json",
        "model", "runs", "logs", "stores")


def cmd_reset(args) -> int:
    """Clear the project's data (the bus, metrics, items, runs, logs, the model), keeping its
    config overlay and tokens. Refused while ``orchestra run`` runs."""
    import shutil
    project = _project(args)
    state = project.state_dir
    try:
        pid = json.loads((state / "endpoint.json").read_text()).get("pid")
        if pid:
            os.kill(int(pid), 0)
            print(f"orchestra run is running (pid {pid}): stop it first", file=sys.stderr)
            return 1
    except (FileNotFoundError, ValueError, ProcessLookupError, PermissionError, TypeError):
        pass
    gone = sorted(p for p in (state / name for name in DATA) if p.exists()) if state.exists() else []
    if not gone:
        print("nothing to clear")
        return 0
    if not args.yes:
        print("this removes, in " + str(state) + ":\n  " + "\n  ".join(p.name for p in gone)
              + "\nrun again with --yes to clear it")
        return 1
    for p in gone:
        shutil.rmtree(p) if p.is_dir() else p.unlink()
    print(f"cleared Orchestra's data in {state}; nothing else there was touched. "
          "Your scripts' own cursors still remember what they relayed.")
    return 0


def cmd_setup(args) -> int:
    """Works on a broken config too: fixing it is part of the job."""
    from . import setup
    from .config import Config
    from .project import Project, repo_root
    try:
        project = _project(args)
    except ConfigError as e:
        root = args.repo or repo_root()
        project = Project(root.name, root, Config(), [])
        print(f"orchestra setup: the config does not load yet ({e}); the setup agent will fix it", file=sys.stderr)
    return setup.run(project, " ".join(args.message) or None, print_only=args.print_prompt)


def cmd_run(args) -> int:
    from .runner import Runner
    project = _project(args)
    runner = Runner(project)
    if args.web:
        web = hooks.hook("web", "serve")
        if web is None:
            print("the pages are not installed; running without them", file=sys.stderr)
        else:
            threading.Thread(target=web.serve, args=(project, runner.stop, args.listen), daemon=True).start()
    runner.run(args.seconds)
    return 0


def cmd_check(args) -> int:
    from .check import lines
    try:
        project = _project(args)
    except ConfigError as e:
        print(f"config: {e}")
        return 1
    print(f"config: ok ({', '.join(map(str, project.files)) or 'no file: every default'})")
    bad = 0
    for ok, text in lines(project):
        print(("ok    " if ok else "FAIL  ") + text)
        bad += not ok
    return 1 if bad else 0


def cmd_events(args) -> int:
    ev = Events.for_project(_project(args))
    found = [e for e in ev.all(0) if (not args.type or e.get("type") == args.type)
             and (not args.subject or e.get("subject") == args.subject)]
    for e in found[-args.n:]:
        print(json.dumps(e))
    return 0


def cmd_emit(args) -> int:
    try:
        raw = json.loads(args.json)
    except ValueError:
        print("not JSON", file=sys.stderr)
        return 2
    import uuid
    ev, why = event(raw, os.environ.get("ORCHESTRA_PROCESS") or "cli", uuid.uuid4().hex) if isinstance(raw, dict) else (None, "not an object")
    if ev is None:
        print(why, file=sys.stderr)
        return 2
    fresh = Publisher(Events.for_project(_project(args))).publish(ev)
    print(ev["id"] + ("" if fresh else " (duplicate: already on the bus)"))
    return 0


def cmd_metrics(args) -> int:
    project = _project(args)
    store = metrics.for_project(project)
    for name, labels, pts in store.recent(args.prefix or "", {"project": project.name}):
        lab = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()) if k != "project")
        print(f"{name}{{{lab}}} {pts[-1][1]:g}")
    return 0


def cmd_query(args) -> int:
    project = _project(args)
    if args.store:
        stores = hooks.hook("stores", "query")
        if stores is None:
            print("reading stores is not installed", file=sys.stderr)
            return 2
        return stores.query(project, args.promql, args.store)
    store = metrics.for_project(project)
    end = time.time()
    try:
        rows = store.query_range(args.promql, end - 3600, end, 60, {"project": project.name})
    except metrics.MetricsError as e:
        print(e, file=sys.stderr)
        return 2
    for labels, pts in rows:
        if pts:
            print(f"{json.dumps(labels)} {pts[-1][1]:g}")
    return 0


def cmd_items(args) -> int:
    project = _project(args)
    items = Items(project, Events.for_project(project))
    if args.show:
        item = items.get(args.show)
        if item is None:
            print(f"no item {args.show}", file=sys.stderr)
            return 1
        print(json.dumps(item, indent=2))
        return 0
    found = items.list(closed=args.all, kind=args.kind)
    for i in found[args.page * args.n:(args.page + 1) * args.n]:
        waits = (i.get("fields") or {}).get("waiting_on")
        title = (i.get("fields") or {}).get("title")
        print(f"{i['last']}  {i.get('kind') or '?'}  {i['subject']}  {i['type']}" + (f"  {title}" if title else "")
              + ("  done" if i.get("done") else "")
              + (f"  waits on {waits}" if waits else ""))
    return 0


def cmd_runs(args) -> int:
    for r in runs.read(_project(args), args.process, args.n):
        print(json.dumps(r))
    return 0


def cmd_tool(args) -> int:
    from .tools import ToolError, call, help_text, parse_args
    project = _project(args)
    tool = project.config.tool(args.name)
    if tool is None:
        print(f"no tool {args.name}; tools: {', '.join(t.name for t in project.config.tools) or 'none'}", file=sys.stderr)
        return 2
    if "--help" in args.args or "-h" in args.args:
        print(help_text(tool, project.repo))
        return 0
    source = os.environ.get("ORCHESTRA_PROCESS") or tool.name
    caller = project.config.process(os.environ.get("ORCHESTRA_PROCESS", ""))
    if caller and caller.tools is not None and tool.name not in caller.tools:
        print(f"{caller.name} may not call {tool.name}; its tools: {', '.join(caller.tools) or 'none'}", file=sys.stderr)
        return 2
    try:
        return call(project, tool, parse_args(args.args), Publisher(Events.for_project(project)), source)
    except ToolError as e:
        print(f"{tool.name}: {e}", file=sys.stderr)
        return 2


def cmd_endpoint(args) -> int:
    from .endpoint import recorded, token
    project = _project(args)
    rec = recorded(project)
    if not rec:
        print("no endpoint: orchestra run serves it", file=sys.stderr)
        return 1
    print(f"{rec['url']}/events\nAuthorization: Bearer {token(project)}")
    return 0


def cmd_model(args) -> int:
    d = hooks.hook("discovery", "show")
    if d is None:
        print("discovery is not installed", file=sys.stderr)
        return 2
    return d.show(_project(args), args.rest)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="orchestra", description="Runs commands, reads their stdout, keeps the bus.")
    p.add_argument("--version", action="version", version=__version__)
    p.add_argument("--repo", type=lambda s: __import__("pathlib").Path(s), help="the repository (default: found from here)")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("init", help="make the state directory")
    s.add_argument("--config", action="store_true", help="and an empty orchestra.yaml")
    s.set_defaults(fn=cmd_init)
    s = sub.add_parser("reset", help="clear the data (bus, metrics, items, runs, logs, model), keeping the config")
    s.add_argument("--yes", action="store_true", help="do it (without it, say what would go)")
    s.set_defaults(fn=cmd_reset)
    s = sub.add_parser("setup", help="a Claude Code session that sets this project up: the config, its scripts and prompts")
    s.add_argument("message", nargs="*", help="what you want, to start with (else it asks)")
    s.add_argument("--print-prompt", action="store_true", help="print the setup agent's brief and stop")
    s.set_defaults(fn=cmd_setup)
    s = sub.add_parser("run", help="run everything")
    s.add_argument("--web", action="store_true", help="also serve the pages")
    s.add_argument("--listen", default="127.0.0.1:8765", help="where the pages listen")
    s.add_argument("--seconds", type=float, help="stop after this long")
    s.set_defaults(fn=cmd_run)
    sub.add_parser("check", help="validate the config and say each part's health").set_defaults(fn=cmd_check)
    s = sub.add_parser("events", help="the latest events")
    s.add_argument("--type")
    s.add_argument("--subject")
    s.add_argument("-n", type=int, default=20)
    s.set_defaults(fn=cmd_events)
    s = sub.add_parser("emit", help="publish one event")
    s.add_argument("json")
    s.set_defaults(fn=cmd_emit)
    s = sub.add_parser("metrics", help="the series and their latest values")
    s.add_argument("prefix", nargs="?")
    s.set_defaults(fn=cmd_metrics)
    s = sub.add_parser("query", help="PromQL over the metrics store, or a store")
    s.add_argument("promql")
    s.add_argument("--store")
    s.set_defaults(fn=cmd_query)
    s = sub.add_parser("items", help="open items; items show SUBJECT")
    s.add_argument("show", nargs="?", help="show")
    s.add_argument("subject", nargs="?")
    s.add_argument("--all", action="store_true", help="closed ones too")
    s.add_argument("--kind", help="only this kind (github.pr, jira.issue)")
    s.add_argument("-n", type=int, default=20)
    s.add_argument("--page", type=int, default=0)
    s.set_defaults(fn=cmd_items)
    s = sub.add_parser("runs", help="process runs")
    s.add_argument("process", nargs="?")
    s.add_argument("-n", type=int, default=20)
    s.set_defaults(fn=cmd_runs)
    s = sub.add_parser("tool", help="call a tool: orchestra tool NAME key=value ...", add_help=False)
    s.add_argument("name")
    s.add_argument("args", nargs=argparse.REMAINDER)
    s.set_defaults(fn=cmd_tool)
    sub.add_parser("endpoint", help="the events endpoint and its token").set_defaults(fn=cmd_endpoint)
    s = sub.add_parser("model", help="the system dynamics model")
    s.add_argument("rest", nargs=argparse.REMAINDER)
    s.set_defaults(fn=cmd_model)
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.cmd == "items":
        if args.show not in (None, "show") or (args.show == "show" and not args.subject):
            print("orchestra items [show SUBJECT]", file=sys.stderr)
            return 2
        args.show = args.subject
    try:
        return args.fn(args)
    except ConfigError as e:
        print(f"config: {e}", file=sys.stderr)
        return 1
