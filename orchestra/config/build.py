"""Building the schema's dataclasses from the merged layers, refusing what does not fit: an unknown
key (with its path, so a typo never silently does nothing), a missing required field, a wrong
type, a bad duration, a bad ``_env`` name, and a literal secret (secrets are environment
variables only: a field ending ``_env`` names one)."""

from __future__ import annotations

import dataclasses
import re
import types
import typing
from typing import Any

from . import schema


class ConfigError(Exception):
    pass


UNITS = {"ms": 0.001, "s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}
SECRET_KEYS = re.compile(r"(^|_)(token|password|passwd|secret|api_key|apikey)$")
SECRET_VALUES = re.compile(r"(sk-ant-|sk-[A-Za-z0-9]{20}|ghp_|github_pat_|xox[abprs]-|AKIA[0-9A-Z]{16}|Bearer\s+\S)")
CREDENTIAL_URL = re.compile(r"^[a-z][a-z0-9+.-]*://[^/@\s]*:[^/@\s]+@")
ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def parse_duration(value: Any, default: float | None = None, where: str = "") -> float | None:
    """Seconds: a number is seconds, a string a number with a unit (ms, s, m, h, d or w; none is s)."""
    if value is None:
        return default
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*(ms|s|m|h|d|w)?\s*", str(value))
    if not m:
        raise ConfigError(f"{where or 'duration'}: bad duration {value!r}: a number of seconds, "
                          "or with a unit: ms, s, m, h, d or w")
    return float(m.group(1)) * UNITS[m.group(2) or "s"]


def _check_secret(path: str, key: str, value: Any) -> None:
    if SECRET_KEYS.search(key):
        raise ConfigError(f"{path}: secrets are environment variables only: use {key}_env: NAME")
    if isinstance(value, str) and (SECRET_VALUES.search(value) or CREDENTIAL_URL.match(value)):
        raise ConfigError(f"{path}: looks like a literal secret; name an environment variable in a field ending _env")


def _value(tp: Any, value: Any, path: str, meta: dict) -> Any:
    if meta.get("duration"):
        return parse_duration(value, None, path)
    origin = typing.get_origin(tp)
    if origin in (typing.Union, types.UnionType):
        if value is None:
            return None
        tp = next(a for a in typing.get_args(tp) if a is not type(None))
        origin = typing.get_origin(tp)
    if meta.get("names") and isinstance(value, str):
        value = [value]
    if dataclasses.is_dataclass(tp):
        return build(tp, value, path)
    if origin is list:
        if not isinstance(value, list):
            raise ConfigError(f"{path}: expected a list, not {value!r}")
        (item,) = typing.get_args(tp)
        return [_value(item, v, f"{path}[{i}]", {}) for i, v in enumerate(value)]
    if origin is dict:
        if not isinstance(value, dict):
            raise ConfigError(f"{path}: expected a map, not {value!r}")
        return {str(k): _value(str, v, f"{path}.{k}", {}) for k, v in value.items()}
    if tp is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{path}: expected true or false, not {value!r}")
        return value
    if tp in (int, float):
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ConfigError(f"{path}: expected a number, not {value!r}")
        return tp(value)
    if tp is str:
        if not isinstance(value, str | int | float) or isinstance(value, bool):
            raise ConfigError(f"{path}: expected a string, not {value!r}")
        return str(value)
    raise ConfigError(f"{path}: cannot read {tp}")    # pragma: no cover - a schema mistake


def build(cls: type, raw: Any, path: str = "") -> Any:
    """An instance of the dataclass ``cls`` from the map ``raw``; ``path`` names it in errors."""
    if not isinstance(raw, dict):
        raise ConfigError(f"{path or 'the config'}: expected a map, not {raw!r}")
    hints = typing.get_type_hints(cls)
    fields = {f.name: f for f in dataclasses.fields(cls)}

    def where(k):
        return f"{path}.{k}" if path else k
    for k, v in raw.items():
        _check_secret(where(k), str(k), v)
        if k not in fields:
            known = ", ".join(sorted(fields))
            raise ConfigError(f"{where(k)}: unknown key (known here: {known})")
    kw = {}
    for name, f in fields.items():
        if name in raw:
            kw[name] = _value(hints[name], raw[name], where(name), dict(f.metadata))
            if name.endswith("_env") and kw[name] is not None and not ENV_NAME.fullmatch(kw[name]):
                raise ConfigError(f"{where(name)}: {kw[name]!r} is not an environment variable's name")
        elif f.metadata.get("required"):
            raise ConfigError(f"{where(name)}: required")
    return cls(**kw)


def label(entry: Any, i: int) -> str:
    return entry.get("name") if isinstance(entry, dict) and entry.get("name") else str(i)


LISTS = {"stores": schema.Store, "tools": schema.Tool, "machines": schema.Machine, "items": schema.Reducer}
RENAMED = ("processes", "metrics", "events")      # one section now: machines (the owner, 2026-11-05)


def build_config(raw: dict) -> schema.Config:
    """The config from the merged layers, with each named entry's path given by its name."""
    if not isinstance(raw, dict):
        raise ConfigError(f"the config must be a map, not {raw!r}")
    for key in RENAMED:
        if key in raw:
            raise ConfigError(f"{key}: is now machines: (SPEC §1)")
    for key in LISTS:
        if key in raw and not isinstance(raw[key], list):
            raise ConfigError(f"{key}: expected a list")
    cfg = build(schema.Config, {k: v for k, v in raw.items() if k not in LISTS})
    for key, c in LISTS.items():
        setattr(cfg, key, [build(c, e, f"{key}[{label(e, i)}]") for i, e in enumerate(raw.get(key) or [])])
    check(cfg)
    return cfg


def check(cfg: schema.Config) -> None:
    """What the types alone don't say: names unique, kinds and modes known, each kind's fields given."""
    for key in ("stores", "tools", "machines"):
        names = [e.name for e in getattr(cfg, key)]
        if dup := sorted({n for n in names if names.count(n) > 1}):
            raise ConfigError(f"{key}: the name {dup[0]!r} is given twice")
    tools = {t.name for t in cfg.tools}
    for m in cfg.machines:
        at = f"machines[{m.name}]"
        if m.kind not in schema.KINDS:
            raise ConfigError(f"{at}.kind: {m.kind!r} is not one of {', '.join(schema.KINDS)}")
        if m.mode not in schema.MODES:
            raise ConfigError(f"{at}.mode: {m.mode!r} is not one of {', '.join(schema.MODES)}")
        if m.kind == "agent" and not m.prompt:
            raise ConfigError(f"{at}.prompt: an agent needs its prompt file")
        if m.kind == "program" and not m.command:
            raise ConfigError(f"{at}.command: a program needs its command")
        if m.kind == "decision" and (not m.question or not m.answers):
            raise ConfigError(f"{at}: a decision needs a question and its answers")
        if m.mode == "continuous" and m.kind == "decision":
            raise ConfigError(f"{at}.mode: a decision is one answer per event, never continuous")
        if m.mode == "continuous" and m.every:
            raise ConfigError(f"{at}.every: a continuous machine is kept running, never scheduled")
        if m.mode == "once" and not (m.on or m.every):
            raise ConfigError(f"{at}: say when it runs: on, every, or mode: continuous")
        for t in m.tools or []:
            if t not in tools:
                raise ConfigError(f"{at}.tools: no tool named {t!r}")
    for s in cfg.stores:
        if re.fullmatch(r"\{\s*(__name__\s*=~\s*[\"']\.[*+][\"']\s*)?\}", s.match.strip()):
            raise ConfigError(f"stores[{s.name}].match: a match-everything selector is refused")
    if cfg.agent.harness != "claude_code":
        raise ConfigError(f"agent.harness: {cfg.agent.harness!r} is not here yet; use claude_code")
