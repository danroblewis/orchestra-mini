"""The seams for what is not in the runtime (part 2): a module is used when it is there.

- ``orchestra.discovery``: ``start(project, stop)``, called by ``orchestra run`` (it runs its own
  thread every ``discovery.every`` and returns); ``show(project, args) -> int``, ``orchestra query model``.
- ``orchestra.web``: ``serve(project, stop, listen)``, called by ``orchestra run --web`` (it blocks
  until ``stop``, in a thread of its own).
"""

from __future__ import annotations

import importlib
import importlib.util
from types import ModuleType


def hook(name: str, attr: str) -> ModuleType | None:
    """``orchestra.<name>`` when it is installed and has ``attr``; an import error inside it is raised."""
    if importlib.util.find_spec(f"orchestra.{name}") is None:
        return None
    mod = importlib.import_module(f"orchestra.{name}")
    return mod if hasattr(mod, attr) else None
