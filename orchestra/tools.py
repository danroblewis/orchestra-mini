"""Tools (SPEC §4): commands an agent may call, ``orchestra tool <name> key=value …``. Each
``{name}`` placeholder is filled from the call's arguments as part of one argument, never through
a shell. A tool's help is its ``help``, or the first 20 lines of its command's ``--help`` (the
command's words before the first placeholder)."""

from __future__ import annotations

import re
import subprocess

from .command import argv, run

PLACEHOLDER = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


class ToolError(Exception):
    pass


def placeholders(command: str) -> list[str]:
    seen: list[str] = []
    for word in argv(command):
        for name in PLACEHOLDER.findall(word):
            if name not in seen:
                seen.append(name)
    return seen


def fill(command: str, args: dict[str, str]) -> list[str]:
    """The command's arguments with every placeholder filled; each word stays one argument."""
    need = placeholders(command)
    if missing := [n for n in need if n not in args]:
        raise ToolError(f"missing {', '.join(f'{n}=' for n in missing)}")
    if extra := [k for k in args if k not in need]:
        raise ToolError(f"unknown argument {', '.join(extra)}; this tool takes {', '.join(need) or 'none'}")
    return [PLACEHOLDER.sub(lambda m: args[m.group(1)], w) for w in argv(command)]


def parse_args(words: list[str]) -> dict[str, str]:
    out = {}
    for w in words:
        k, sep, v = w.partition("=")
        if not sep or not k:
            raise ToolError(f"{w!r}: arguments are key=value")
        out[k] = v
    return out


def help_text(tool, cwd) -> str:
    if tool.help:
        return tool.help.strip()
    words = argv(tool.command)
    head = words[:next((i for i, w in enumerate(words) if PLACEHOLDER.search(w)), len(words))]
    try:
        out = subprocess.run(head + ["--help"], cwd=cwd, capture_output=True, text=True, timeout=10)
        text = out.stdout or out.stderr
    except (OSError, subprocess.TimeoutExpired):
        text = ""
    return "\n".join(text.splitlines()[:20]).strip() or "(no help)"


def describe(project, names: list[str] | None) -> str:
    """The tools section of an agent's prompt."""
    tools = [t for t in project.config.tools if names is None or t.name in names]
    if not tools:
        return "You have no tools."
    parts = ["Call a tool by running `orchestra tool <name> key=value ...` in your shell."]
    for t in tools:
        args = " ".join(f"{n}=..." for n in placeholders(t.command))
        parts.append(f"### {t.name}\n`orchestra tool {t.name} {args}`\n{help_text(t, project.repo)}")
    return "\n\n".join(parts)


def call(project, tool, args: dict[str, str], publisher, source: str, out=print) -> int:
    """Run the tool, print its stdout as it comes, publish its event lines; its exit code."""
    words = fill(tool.command, args)
    log = project.path("logs", "tools", f"{tool.name}.log")
    res = run(words, project.repo, log, timeout=600, on_line=lambda line: out(line, end=""))
    publisher.lines(res.lines, source)
    if res.error:
        raise ToolError(res.error)
    return 124 if res.timed_out else (res.rc or 0)
