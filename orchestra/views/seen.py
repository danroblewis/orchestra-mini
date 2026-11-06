"""The views a once run saw (SPEC §3): ``{projection: view}`` for its subject, written to
``runs/<id>.views.json`` and named in ``ORCHESTRA_VIEWS_FILE``, so "why did the agent do that" has
the exact views it was given. The newest ``KEEP`` are kept."""

from __future__ import annotations

import json

KEEP = 500


def write(project, run_id: str, views: dict) -> str:
    path = project.path("runs", f"{run_id}.views.json")
    path.write_text(json.dumps(views, indent=1))
    old = sorted(path.parent.glob("*.views.json"))           # run ids begin with their time
    for p in old[:-KEEP]:
        p.unlink(missing_ok=True)
    return str(path)


def legacy_item(views: dict | None) -> str:
    """``ORCHESTRA_ITEM`` for one release: ``latest`` without ``by_type``, with ``events`` (its
    count of facts); empty past 32 KB."""
    base = (views or {}).get("latest")
    if not base:
        return ""
    text = json.dumps({k: v for k, v in base.items() if k not in ("by_type", "facts", "behind")} | {"events": base.get("facts", 0)})
    return "" if len(text) > 32_000 else text
