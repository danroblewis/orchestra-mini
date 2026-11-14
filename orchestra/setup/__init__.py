"""``orchestra chat``: the Supervisor, a Claude Code session in the repository (the owner, 2026-11-05:
"Orchestra is supposed to provide a setup agent that does this for me. I should be able to write the
file by hand, read it, maintain it, but I don't want to write the file or the scripts.")

It runs ``claude`` interactively with a brief and the spec appended to its system prompt: ``chat.md``
with what is happening now (and, for a config written for an earlier Orchestra, what to rewrite:
``MIGRATE``), and what is already here: the config's path and text, what ``orchestra run`` would say of it, the
environment variables that look like credentials (by name only), ``gh``'s login state, and how to
run Orchestra's own commands (``orchestra``, and the reset, ``<python> -m orchestra.reset``)."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
MIGRATE = """# Rewrite the config first

The config was written for an earlier Orchestra, and `orchestra run` refuses it:

    {why}

Rewrite it before anything else, and say what you changed. Each `items:` reducer becomes a
projection under `projections:` (SPEC §6) with a `name`, the same `match` and its `command`. A
projection is given every fact of the subjects being built, in log order, not time order: if the
reducer relied on time order, sort its stdin by `time`. It prints one JSON line per subject,
`{{"subject": ..., ...}}`, not one object for one item. Test it on recorded facts
(`orchestra query views NAME --input SUBJECT | <command>`), then run `orchestra query check`.
"""
CREDENTIAL_NAME = re.compile(r"(TOKEN|KEY|SECRET|PASSWORD|PAT|_URL|_HOST|_USER)$|^(GH|GITHUB|JIRA|SLACK|ANTHROPIC|AWS|GOOGLE)_")


def _run(cmd: list[str], cwd: Path, timeout: float = 20) -> str:
    try:
        r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout, check=False)
        return (r.stdout + r.stderr).strip()
    except (OSError, subprocess.TimeoutExpired) as e:
        return f"(could not run {cmd[0]}: {e})"


def checked(project) -> str:
    """What ``orchestra run`` would say of the config before starting, without starting anything."""
    from ..check import problems
    from ..config import ConfigError
    from ..project import load
    try:
        fresh = load(project.repo)
    except ConfigError as e:
        return f"config: {e}"
    found = problems(fresh, stores=False)
    return "\n".join(f"{level}: {text}" for level, text in found) or "config: ok" + ("" if fresh.files else " (no file yet)")


def context(project) -> str:
    root = project.repo
    configs = [p for p in (root / "orchestra.yaml", root / "orchestra.yml", root / "orchestra.json",
                           project.state_dir / "orchestra.yaml", project.state_dir / "orchestra.yml",
                           project.state_dir / "orchestra.json") if p.exists()]
    shown = "\n\n".join(f"### {p}\n```\n{p.read_text(encoding='utf-8')[:6000]}\n```" for p in configs) or "none yet"
    names = sorted(k for k in os.environ if CREDENTIAL_NAME.search(k))
    check = checked(project)
    gh = _run(["gh", "auth", "status"], root) if shutil.which("gh") else "gh is not installed"
    remote = _run(["git", "remote", "-v"], root)
    return (f"# This project now\n\n- repository: {root}\n- state directory: {project.state_dir} "
            f"(the overlay goes here as orchestra.yaml)\n- git remotes:\n```\n{remote}\n```\n"
            f"- environment variables that may be credentials (names only): {', '.join(names) or 'none'}\n"
            f"- clearing the data (only when the owner asks): `{sys.executable} -m orchestra.reset --yes`\n"
            f"- gh:\n```\n{gh[:1500]}\n```\n\n## Config files\n\n{shown}\n\n"
            f"## What orchestra run checks before it starts\n```\n{check[:3000]}\n```\n")


def now(project) -> str:
    """What is happening: open items, the latest runs and facts, in a few lines."""
    root = project.repo
    o = [sys.executable, "-m", "orchestra", "--repo", str(root)]
    parts = [("open items", o + ["query", "views", "-n", "15"]), ("latest runs", o + ["query", "runs"]),
             ("latest facts", o + ["query", "facts", "-n", "15"])]
    return "# What is happening now\n\n" + "\n".join(f"## {t}\n```\n{_run(cmd, root)[:2500]}\n```" for t, cmd in parts)


def prompt(project, migrate: str | None = None) -> str:
    pieces = [(HERE / "chat.md").read_text(encoding="utf-8"), (HERE / "code.md").read_text(encoding="utf-8"), context(project)]
    if migrate:
        pieces.insert(1, MIGRATE.format(why=migrate))
    else:
        pieces.append(now(project))
    return "\n\n".join(pieces + ["# SPEC.md\n\n" + (HERE / "SPEC.md").read_text(encoding="utf-8")])


def run(project, message: str | None, print_only: bool = False, binary: str = "claude",
        resume: bool = False, migrate: str | None = None) -> int:
    """Claude Code, interactive, briefed; ``resume`` carries on the last session in this folder."""
    if not shutil.which(binary):
        print(f"orchestra needs Claude Code ({binary}) on PATH: https://docs.claude.com/en/docs/claude-code", file=sys.stderr)
        return 2
    text = prompt(project, migrate)
    if print_only:
        print(text)
        return 0
    first = message or ("Rewrite the config for this Orchestra." if migrate else None)
    cmd = [binary, *(["--continue"] if resume else []), "--append-system-prompt", text, *([first] if first else [])]
    return subprocess.call(cmd, cwd=project.repo)
