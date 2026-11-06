"""Clearing a project's data (SPEC §5), something ``orchestra chat`` does when the owner asks:
``python -m orchestra.reset [--repo R] [--yes]``. It removes exactly ``DATA`` from the state
directory (the bus and metrics files, items, runs, logs, the model, the stores' request logs) and
keeps everything else there: the config overlay, the tokens, and the scripts and prompts kept
beside them. Without ``--yes`` it says what would go. Refused while ``orchestra run`` runs."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

# Orchestra's own data, the only things reset removes. Anything else in the state directory (the
# overlay, scripts and prompts kept beside it, a cursor a script keeps) is the owner's and is never
# touched (the owner, 2026-11-05: an earlier reset removed everything but a keep-list, and with it
# the scripts the setup agent had put there).
DATA = ("events.jsonl", "metrics.jsonl", "items.json", "cursor.json", "kinds.json", "endpoint.json", "web.json",
        "model", "runs", "logs", "stores")


def running(state: Path) -> int | None:
    """The pid of the ``orchestra run`` serving this project, or None."""
    try:
        pid = int(json.loads((state / "endpoint.json").read_text()).get("pid"))
        os.kill(pid, 0)
        return pid
    except (FileNotFoundError, ValueError, ProcessLookupError, PermissionError, TypeError):
        return None


def clear(state: Path, yes: bool) -> int:
    if pid := running(state):
        print(f"orchestra run is running (pid {pid}): stop it first", file=sys.stderr)
        return 1
    gone = sorted(p for p in (state / name for name in DATA) if p.exists()) if state.exists() else []
    if not gone:
        print("nothing to clear")
        return 0
    if not yes:
        print(f"this removes, in {state}:\n  " + "\n  ".join(p.name for p in gone) + "\nrun again with --yes to clear it")
        return 1
    for p in gone:
        shutil.rmtree(p) if p.is_dir() else p.unlink()
    print(f"cleared Orchestra's data in {state}; nothing else there was touched. "
          "Your scripts' own cursors still remember what they relayed.")
    return 0


def main(argv: list[str] | None = None) -> int:
    from .config import home
    from .config.load import find, read
    from .project import repo_root
    a = argparse.ArgumentParser(prog="python -m orchestra.reset", description="Clear the project's data, keeping its config.")
    a.add_argument("--repo", type=Path, help="the repository (default: found from here)")
    a.add_argument("--yes", action="store_true", help="do it (without it, say what would go)")
    args = a.parse_args(argv)
    root = (args.repo or repo_root()).resolve()
    try:                                      # the project's name only: a config that doesn't load can still be cleared
        name = read(find(root)).get("project")
    except Exception:
        name = None
    return clear(home() / (name if isinstance(name, str) and name else root.name), args.yes)


if __name__ == "__main__":
    sys.exit(main())
