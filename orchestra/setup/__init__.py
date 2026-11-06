"""``orchestra chat``: the Supervisor, a Claude Code session in the repository (the owner, 2026-11-05:
"Orchestra is supposed to provide a setup agent that does this for me. I should be able to write the
file by hand, read it, maintain it, but I don't want to write the file or the scripts.")

It runs ``claude`` interactively with a brief and the spec appended to its system prompt: ``setup.md``
when there is no config yet or it doesn't load, else ``chat.md`` with what is happening now. Both get
what is already here: the config's path and text, what ``orchestra run`` would say of it, the
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
    """What is happening: open items by kind, the latest runs and events, in a few lines."""
    root = project.repo
    o = [sys.executable, "-m", "orchestra", "--repo", str(root)]
    parts = [("open items", o + ["query", "items", "-n", "15"]), ("latest runs", o + ["query", "runs"]),
             ("latest events", o + ["query", "events", "-n", "15"])]
    return "# What is happening now\n\n" + "\n".join(f"## {t}\n```\n{_run(cmd, root)[:2500]}\n```" for t, cmd in parts)


def prompt(project, brief: str = "setup.md") -> str:
    pieces = [(HERE / brief).read_text(encoding="utf-8"), context(project)]
    if brief == "chat.md":
        pieces.append(now(project))
    return "\n\n".join(pieces + ["# SPEC.md\n\n" + (HERE / "SPEC.md").read_text(encoding="utf-8")])


def run(project, message: str | None, print_only: bool = False, binary: str = "claude", brief: str = "setup.md",
        resume: bool = False) -> int:
    """Claude Code, interactive, briefed; ``resume`` carries on the last session in this folder."""
    if not shutil.which(binary):
        print(f"orchestra needs Claude Code ({binary}) on PATH: https://docs.claude.com/en/docs/claude-code", file=sys.stderr)
        return 2
    text = prompt(project, brief)
    if print_only:
        print(text)
        return 0
    first = message or ("Set Orchestra up for this project. Start by asking me what I want it to watch and do."
                        if brief == "setup.md" else None)
    cmd = [binary, *(["--continue"] if resume else []), "--append-system-prompt", text, *([first] if first else [])]
    return subprocess.call(cmd, cwd=project.repo)
