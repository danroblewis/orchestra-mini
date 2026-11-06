"""What each page shows, as plain data (the pages render it; ``/api/<page>`` returns it as JSON).
Every read is of files and the configured bus and stores; nothing here changes anything."""

from __future__ import annotations

import fnmatch
import threading
import time

from .. import metrics, runs
from ..check import failed, how
from ..events import Events
from ..items import Items
from ..machines import machines as all_machines

PROBE_S = 3.0           # each probe on the Core page answers within this, or is shown as not answering


def probe(fn, *args):
    """``fn(*args)`` under ``PROBE_S``: (result, None) or (None, why), so one broken part never takes
    a page down."""
    box: dict = {}

    def go():
        try:
            box["ok"] = fn(*args)
        except Exception as e:
            box["err"] = f"{type(e).__name__}: {e}"
    t = threading.Thread(target=go, daemon=True)
    t.start()
    t.join(PROBE_S)
    if t.is_alive():
        return None, f"no answer within {PROBE_S:g} s"
    return box.get("ok"), box.get("err")


def _tail(path, n: int = 5) -> list[str]:
    try:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()[-n:]
    except OSError:
        return []


SHOWN_FIELDS = 4             # a kind's table shows its most common fields, at most this many beside the title
PER_KIND = 10                # the overview shows this many of each kind; the kind's own page shows them all


def _columns(found: list[dict]) -> list[str]:
    """The fields most of a kind's items have (besides title and waiting_on), commonest first."""
    seen: dict[str, int] = {}
    for i in found:
        for k, v in (i.get("fields") or {}).items():
            if k not in ("title", "waiting_on") and not isinstance(v, (dict, list)):
                seen[k] = seen.get(k, 0) + 1
    return [k for k, c in sorted(seen.items(), key=lambda kv: (-kv[1], kv[0])) if c * 2 >= len(found)][:SHOWN_FIELDS]


def _row(i: dict, cols: list[str]) -> dict:
    f = i.get("fields") or {}
    return {k: i.get(k) for k in ("subject", "kind", "type", "first", "last", "done")} | {
        "waiting_on": f.get("waiting_on"), "title": f.get("title"), "values": [f.get(c) for c in cols],
        "events": len(i.get("events") or [])}


def items(project, page: int = 0, n: int = 25, closed: bool = False, kind: str | None = None) -> dict:
    """One table per kind (the overview: the newest of each), or one kind's items, paged."""
    store = Items(project, Events.for_project(project))
    found = store.list(closed=closed, kind=kind)
    groups = []
    order: list[str] = []
    by: dict[str, list[dict]] = {}
    for i in found:
        k = i.get("kind") or "?"
        if k not in by:
            order.append(k)
            by[k] = []
        by[k].append(i)
    for k in order:
        cols = _columns(by[k])
        rows = by[k][page * n:(page + 1) * n] if kind else by[k][:PER_KIND]
        groups.append({"kind": k, "total": len(by[k]), "columns": cols, "items": [_row(i, cols) for i in rows]})
    return {"groups": groups, "page": page, "n": n, "total": len(found), "closed": closed, "kind": kind, "kinds": store.kinds()}


def item(project, subject: str) -> dict | None:
    return Items(project, Events.for_project(project)).get(subject)


def quiet(r: dict) -> bool:
    """A collector's scheduled run that went well: its runs say it, the dashboard doesn't."""
    return not r.get("event") and not r.get("continuous") and not r.get("events") and not r.get("cost_usd") and not failed(r)


def dashboard(project) -> dict:
    found = Items(project, Events.for_project(project)).list()
    owner = [i for i in found if (i.get("fields") or {}).get("waiting_on") == "owner"]
    spent = {m.name: x for m in project.config.machines if (x := runs.spent_today(project, m.name))}
    return {"waiting": [{"subject": i["subject"], "type": i["type"], "last": i["last"],
                         "title": (i.get("fields") or {}).get("title")} for i in owner],
            "open": len(found), "recent_open": [{"subject": i["subject"], "type": i["type"], "last": i["last"],
                                                  "waiting_on": (i.get("fields") or {}).get("waiting_on")} for i in found[:10]],
            "runs": [r for r in runs.read(project, None, 500) if not quiet(r)][:10], "spent_today": round(sum(spent.values()), 4), "spent_by": spent}


def machines(project) -> dict:
    found = all_machines(project)
    types = sorted({t for m in found for t in m["publishes"]})
    edges = []
    for m in found:
        for t in m["publishes"]:
            edges.append({"from": m["name"], "type": t})
        for pat in m["listens_for"]:
            for t in [t for t in types if fnmatch.fnmatchcase(t, pat)] or [pat]:
                edges.append({"type": t, "to": m["name"]})
    for m in found:
        if m["command"]:
            m["recent_runs"] = runs.read(project, m["name"], 5)
    return {"machines": found, "edges": edges}


def dynamics(project) -> dict:
    from ..discovery.store import ModelStore
    store = ModelStore.for_project(project)
    got = store.latest_discovery()
    preds = (store.read_record("predictions") or {}).get("predictions", {})
    return {"at": got[0] if got else None, "model": got[1].to_json() if got else None, "versions": store.versions(),
            "predictions": [{"id": k, **{x: v.get(x) for x in ("source", "subject", "made", "due", "status", "why", "verdict", "data")}}
                            for k, v in sorted(preds.items(), key=lambda kv: -(kv[1].get("made") or 0))][:20]}


def core(project) -> dict:
    out: dict = {"parts": []}

    def bus():
        ev = Events.for_project(project)
        return f"{ev.bus.name}: {ev.end()} events"

    def store():
        s = metrics.for_project(project)
        return f"{s.name}: {len(s.series('', {'project': project.name}))} series"
    for name, fn in (("bus", bus), ("metrics store", store)):
        said, err = probe(fn)
        out["parts"].append({"part": name, "ok": err is None, "said": said or err, "errors": []})
    from .. import stores
    from ..stores import reqlog
    for s in project.config.stores:
        lines, err = probe(stores.check, project, s)
        errors = [f"{time.strftime('%H:%M:%S', time.localtime(x['t']))} {x['error']}" for x in reqlog.last(project, s.name, 50) if x.get("error")]
        out["parts"].append({"part": f"store {s.name}", "ok": err is None and all(ok for ok, _ in lines or []),
                             "said": err or "; ".join(line for _, line in lines or []), "errors": errors[-5:]})
    for m in project.config.machines:
        recent = runs.read(project, m.name, 20)
        last = recent[0] if recent else None
        said = f"{how(m)}: " + (f"last run {last['start']} exit {last.get('exit')}" if last else "no runs yet")
        errors = [f"{r['start']} exit {r.get('exit')} {r.get('note') or ''}".strip() for r in recent if failed(r)][:5]
        if failed(last):                            # and the end of its log
            errors += _tail(project.state_dir / "logs" / f"{m.name}.log")
        out["parts"].append({"part": f"machine {m.name}", "ok": not failed(last), "said": said, "errors": errors})
    errors = [x for x in _tail(project.state_dir / "logs" / "discovery.log", 20) if "Error" in x][-5:]
    out["parts"].append({"part": "discovery", "ok": not errors, "said": f"every {project.config.discovery.every:g} s over "
                         f"{project.config.discovery.window:g} s", "errors": errors})
    return out
