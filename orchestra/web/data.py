"""What each page shows, as plain data (the pages render it; ``/api/<page>`` returns it as JSON).
Every read is of files and the configured bus and stores; nothing here changes anything."""

from __future__ import annotations

import fnmatch
import json
import threading
import time

from .. import hooks, metrics, runs
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


def _health(project) -> dict:
    try:
        return json.loads((project.state_dir / "health.json").read_text())
    except (OSError, ValueError):
        return {}


def items(project, page: int = 0, n: int = 25, closed: bool = False) -> dict:
    found = Items(project, Events.for_project(project)).list(closed=closed)
    rows = [{k: i.get(k) for k in ("subject", "type", "first", "last", "done")} | {
        "waiting_on": (i.get("fields") or {}).get("waiting_on"), "title": (i.get("fields") or {}).get("title"),
        "events": len(i.get("events") or [])} for i in found[page * n:(page + 1) * n]]
    return {"items": rows, "page": page, "n": n, "total": len(found), "closed": closed}


def item(project, subject: str) -> dict | None:
    return Items(project, Events.for_project(project)).get(subject)


def dashboard(project) -> dict:
    found = Items(project, Events.for_project(project)).list()
    owner = [i for i in found if (i.get("fields") or {}).get("waiting_on") == "owner"]
    spent = {p.name: runs.spent_today(project, p.name) for p in project.config.processes if p.kind != "program"}
    return {"waiting": [{"subject": i["subject"], "type": i["type"], "last": i["last"],
                         "title": (i.get("fields") or {}).get("title")} for i in owner],
            "open": len(found), "recent_open": [{"subject": i["subject"], "type": i["type"], "last": i["last"],
                                                  "waiting_on": (i.get("fields") or {}).get("waiting_on")} for i in found[:10]],
            "runs": runs.read(project, None, 10), "spent_today": round(sum(spent.values()), 4), "spent_by": spent}


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
        if m["process"]:
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
    checker = hooks.hook("stores", "check")
    for s in project.config.stores:
        lines, err = probe(checker.check, project, s) if checker else ([(True, "configured (reading stores is not installed)")], None)
        from ..stores import reqlog
        errors = [f"{time.strftime('%H:%M:%S', time.localtime(x['t']))} {x['error']}" for x in reqlog.last(project, s.name, 50) if x.get("error")]
        out["parts"].append({"part": f"store {s.name}", "ok": err is None and all(ok for ok, _ in lines or []),
                             "said": err or "; ".join(line for _, line in lines or []), "errors": errors[-5:]})
    health = _health(project)
    for c in [*project.config.events, *project.config.metrics]:
        h = health.get(c.name)
        said = f"last round {h['at']}: " + ("ok" if h.get("ok") else f"failed: {h.get('error')}") if h else "not run yet"
        out["parts"].append({"part": f"command {c.name}", "ok": not h or bool(h.get("ok")), "said": said,
                             "errors": _tail(project.state_dir / "logs" / f"{c.name}.log")})
    for p in project.config.processes:
        last = runs.read(project, p.name, 1)
        bad = [r for r in runs.read(project, p.name, 20) if r.get("exit") not in (0, None) or r.get("note")]
        out["parts"].append({"part": f"process {p.name}", "ok": not last or last[0].get("exit") in (0, None),
                             "said": f"last run {last[0]['start']} exit {last[0].get('exit')}" if last else "no runs yet",
                             "errors": [f"{r['start']} exit {r.get('exit')} {r.get('note') or ''}".strip() for r in bad[:5]]})
    errors = [x for x in _tail(project.state_dir / "logs" / "discovery.log", 20) if "Error" in x][-5:]
    out["parts"].append({"part": "discovery", "ok": not errors, "said": f"every {project.config.discovery.every:g} s over "
                         f"{project.config.discovery.window:g} s", "errors": errors})
    return out
