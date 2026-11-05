"""Reading the config's layers (SPEC §1): ``orchestra.yaml`` (or ``.yml``, ``.json``) at the
repository root, then the private overlay ``<home>/<project>/orchestra.yaml`` (or ``.json``),
where home is ``ORCHESTRA_HOME`` or ``~/.orchestra``. JSON is YAML, so one reader does both."""

from __future__ import annotations

import os
import re
from pathlib import Path

import yaml

from .build import ConfigError
from .merge import drop_disabled, merge

NAMES = ("orchestra.yaml", "orchestra.yml", "orchestra.json")


class Loader(yaml.SafeLoader):
    """YAML 1.2's booleans: only true and false, so ``on:`` is the key ``on``, not True."""


Loader.yaml_implicit_resolvers = {k: [r for r in v if r[0] != "tag:yaml.org,2002:bool"]
                                  for k, v in yaml.SafeLoader.yaml_implicit_resolvers.items()}
Loader.add_implicit_resolver("tag:yaml.org,2002:bool", re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$"), list("tTfF"))


def home() -> Path:
    return Path(os.environ.get("ORCHESTRA_HOME") or Path.home() / ".orchestra").expanduser()


def find(folder: Path) -> Path | None:
    found = [folder / n for n in NAMES if (folder / n).is_file()]
    if len(found) > 1:
        raise ConfigError(f"{folder}: more than one config file ({', '.join(p.name for p in found)}); keep one")
    return found[0] if found else None


def read(path: Path | None) -> dict:
    if path is None:
        return {}
    try:
        raw = yaml.load(path.read_text(encoding="utf-8"), Loader=Loader)
    except yaml.YAMLError as e:
        raise ConfigError(f"{path}: {e}") from None
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: the config must be a map")
    return raw


def layers(repo: Path) -> tuple[str, list[Path], dict]:
    """The project's name, the files read, and the merged raw config."""
    repo_file = find(repo)
    raw = read(repo_file)
    name = raw.get("project") if isinstance(raw.get("project"), str) else None
    name = name or repo.resolve().name
    overlay = find(home() / name)
    merged = drop_disabled(merge(raw, read(overlay)))
    return name, [p for p in (repo_file, overlay) if p], merged
