"""The pages, as HTML: plain markup, one stylesheet, and ``graph.js`` for the two drawings (the
model on Dynamics, the event map on Machines), which read the page's embedded JSON."""

from __future__ import annotations

import json
import time
from html import escape
from urllib.parse import quote

PAGES = [("/", "Dashboard"), ("/items", "Items"), ("/machines", "Machines"), ("/dynamics", "Dynamics"), ("/core", "Core")]


def e(x) -> str:
    return escape("" if x is None else str(x))


def when(t) -> str:
    if t is None:
        return ""
    if isinstance(t, int | float):
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t))
    return e(str(t).replace("T", " ").replace(".000Z", "Z"))


def table(heads: list[str], rows: list[list[str]], empty: str = "nothing yet") -> str:
    if not rows:
        return f'<p class="quiet">{e(empty)}</p>'
    h = "".join(f"<th>{e(x)}</th>" for x in heads)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><thead><tr>{h}</tr></thead><tbody>{body}</tbody></table>"


def layout(project, path: str, title: str, body: str, data: dict | None = None) -> str:
    nav = "".join(f'<a href="{href}"{" class=on" if href == path else ""}>{name}</a>' for href, name in PAGES)
    embed = f'<script id="data" type="application/json">{json.dumps(data).replace("</", "<\\/")}</script>' if data is not None else ""
    return f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{e(title)} · {e(project.name)}</title><link rel="stylesheet" href="/static/style.css"></head>
<body><header><b>{e(project.name)}</b><nav>{nav}</nav><form method="post" action="/logout"><button>log out</button></form></header>
<main><h1>{e(title)}</h1>{body}</main>{embed}<script src="/static/graph.js"></script></body></html>"""


def item_link(subject: str) -> str:
    return f'<a href="/items?show={quote(subject, safe="")}">{e(subject)}</a>'


def dashboard(d: dict) -> str:
    waiting = table(["item", "title", "latest", "when"], [[item_link(i["subject"]), e(i["title"]), e(i["type"]), when(i["last"])]
                                                           for i in d["waiting"]], "nothing waits on you")
    opened = table(["item", "latest", "when", "waits on"], [[item_link(i["subject"]), e(i["type"]), when(i["last"]), e(i["waiting_on"])]
                                                            for i in d["recent_open"]], "no open items")
    spent = ", ".join(f"{e(k)} ${v:.2f}" for k, v in d["spent_by"].items() if v) or "nothing"
    return (f'<section><h2>Waiting on you</h2>{waiting}</section>'
            f'<section><h2>Open items ({d["open"]})</h2>{opened}<p><a href="/items">all items</a></p></section>'
            f'<section><h2>Recent runs</h2>{runs_table(d["runs"])}</section>'
            f'<section><h2>Spend today</h2><p class="big">${d["spent_today"]:.2f}</p><p class="quiet">{spent}</p></section>')


def runs_table(rs: list[dict]) -> str:
    return table(["machine", "started", "ended", "exit", "events", "points", "cost", "note"],
                 [[e(r.get("machine")), when(r.get("start")), when(r.get("end")), e(r.get("exit")), e(r.get("events")),
                   e(r.get("points") or ""), f"${r['cost_usd']:.4f}" if r.get("cost_usd") else "", e(r.get("note"))]
                  for r in rs], "no runs yet")


def _value(v) -> str:
    return "" if v is None else e(v if isinstance(v, str) else json.dumps(v))


def _kind_table(g: dict, closed: bool) -> str:
    rows = [[item_link(i["subject"]), e(i["title"])] + [_value(v) for v in i["values"]]
            + [e(i["type"]), when(i["last"]), e(i["waiting_on"]), "closed" if i["done"] else "open"] for i in g["items"]]
    return table(["item", "title", *g["columns"], "latest", "when", "waits on", "state"], rows, "none")


def items(d: dict) -> str:
    kind = d.get("kind")
    q = ("&closed=1" if d["closed"] else "") + (f"&kind={quote(kind)}" if kind else "")
    kinds = " · ".join(['<a href="/items">all</a>' if kind else "<b>all</b>"]
                       + [(f"<b>{e(k)} ({n})</b>" if k == kind else f'<a href="/items?kind={quote(k)}">{e(k)} ({n})</a>')
                          for k, n in (d.get("kinds") or {}).items()])
    pages = []
    if d["page"] > 0:
        pages.append(f'<a href="/items?page={d["page"] - 1}{q}">newer</a>')
    if (d["page"] + 1) * d["n"] < d["total"]:
        pages.append(f'<a href="/items?page={d["page"] + 1}{q}">older</a>')
    base = f"/items?kind={quote(kind)}&" if kind else "/items?"
    toggle = f'<a href="{base}">hide closed</a>' if d["closed"] else f'<a href="{base}closed=1">show closed</a>'
    closed = "&closed=1" if d["closed"] else ""
    if kind:                                    # one kind: its table, paged
        g = d["groups"][0] if d["groups"] else {"kind": kind, "total": 0, "columns": [], "items": []}
        shown = f'{d["page"] * d["n"] + 1 if g["items"] else 0}-{d["page"] * d["n"] + len(g["items"])} of {d["total"]}'
        body = f'<p>{shown} {"items" if d["closed"] else "open items"} · {toggle}</p>' + _kind_table(g, d["closed"])
    else:                                       # every kind: a table each, the newest of each
        body = f'<p>{d["total"]} {"items" if d["closed"] else "open items"} · {toggle}</p>' + ("".join(
            f'<section><h2><a href="/items?kind={quote(g["kind"])}{closed}">{e(g["kind"])}</a> '
            f'<span class="muted">({g["total"]}{", the newest " + str(len(g["items"])) if g["total"] > len(g["items"]) else ""})</span></h2>'
            + _kind_table(g, d["closed"]) + "</section>" for g in d["groups"]) or "<p>no items</p>")
        pages = []
    return (f'<p class="kinds">{kinds}</p>' + body
            + f'<p class="pager">{" · ".join(pages)}</p>')


def item(i: dict) -> str:
    fields = table(["field", "value"], [[e(k), f"<code>{e(json.dumps(v))}</code>"] for k, v in sorted(i["fields"].items())], "no fields")
    hist = table(["when", "type", "source", "data"], [[when(ev.get("time")), e(ev.get("type")), e(ev.get("source")),
                                                       f"<code>{e(json.dumps(ev.get('data')))}</code>"] for ev in i["events"]])
    waits = f'<p>waits on <b>{e(i["fields"]["waiting_on"])}</b></p>' if i["fields"].get("waiting_on") else ""
    return (f'<p><b>{e(i.get("kind") or "")}</b> · {"closed" if i["done"] else "open"} · latest {e(i["type"])} at {when(i["last"])}</p>{waits}'
            f'<section><h2>Fields</h2>{fields}</section><section><h2>History</h2>{hist}</section><p><a href="/items">all items</a></p>')


def machines(d: dict) -> str:
    rows = []
    for m in d["machines"]:
        kind = m["kind"] or "observed"           # heard on the bus, not run by Orchestra (a person, another program)
        how = ", ".join(filter(None, ["continuous" if m.get("mode") == "continuous" else "",
                                      f"on {', '.join(m['listens_for'])}" if m["listens_for"] else "",
                                      f"every {m['every']:g} s" if m.get("every") else ""]))
        pubs = "<br>".join(f"{e(t)} ×{n}" for t, n in m["publishes"].items())
        rows.append([f'<b>{e(m["name"])}</b>' + (f'<br><span class="quiet">{e(m["label"])}</span>' if m["label"] else ""),
                     e(kind), e(how), pubs, when(m["last_seen"]), e(m["runs"]) if m["kind"] else ""])
    runs = "".join(f'<h3>{e(m["name"])}</h3>{runs_table(m["recent_runs"])}' for m in d["machines"] if m.get("recent_runs"))
    return ('<section><h2>Event map</h2><div id="map" class="graph"></div></section>'
            + "<section><h2>Machines</h2>" + table(["machine", "kind", "runs when", "publishes", "last heard", "runs"], rows,
                                                   "no machines yet: none configured and nothing on the bus") + "</section>"
            + f"<section><h2>Runs</h2>{runs or '<p class=quiet>no runs yet</p>'}</section>")


def equation(f: dict, unit: str) -> str:
    eqs = f.get("equations") or []
    if not eqs:
        return "not fitted"
    q, p = eqs[-1], eqs[-1]["params"]
    x = (q.get("inputs") or [""])[0]
    form = {"constant": f"{p.get('c', 0):.3g}", "proportional": f"{p.get('k', 0):.3g} × {x}",
            "linear": f"{p.get('c', 0):.3g} + {p.get('k', 0):.3g} × {x}"}.get(q["form"], q["form"])
    return e(f"{form} per {unit}")


def dynamics(d: dict) -> str:
    m = d["model"]
    if m is None:
        return '<p class="quiet">no model yet: discovery runs every discovery.every while orchestra run runs</p>'
    disc = m.get("discovery") or {}
    stocks = table(["stock", "series", "balance R²"], [[f'<b>{e(s["id"])}</b>', e(s["name"]), e(s["evidence"].get("r2"))] for s in m["stocks"]],
                   "no stocks: no level is balanced by counters yet")
    flows = []
    for f in m["flows"]:
        hs = (f.get("evidence") or {}).get("machines") or {}
        who = "<br>".join(f'{e(h["machine"])} ({h["confidence"]:.2f}): <span class="quiet">{e(h["why"])}</span>' for h in hs.get("hypotheses", []))
        flows.append([f'<b>{e(f["id"])}</b>', e(f["from"] or "outside"), e(f["to"] or "outside"), equation(f, m["time_unit"]),
                      who or f'<span class="quiet">{e(hs.get("why"))}</span>'])
    left = table(["series", "why"], [[e(k), e(v)] for k, v in sorted((disc.get("left_out") or {}).items())], "nothing")
    versions = table(["version", "saved", "stocks", "flows", "why"], [[f"v{v['version']}", when(v["saved_at"]), e(v["stocks"]), e(v["flows"]),
                                                                       e(v["note"])] for v in reversed(d["versions"])])
    preds = table(["made", "by", "about", "status", "verdict"], [[when(p["made"]), e(p["source"]),
                  e(json.dumps(p["data"])), e(p["status"] + (f": {p['why']}" if p.get("why") else "")),
                  e((p.get("verdict") or {}).get("verdict"))] for p in d["predictions"]], "no predictions")
    errs = "".join(f'<p class="bad">store {e(k)}: {e(v)}</p>' for k, v in (disc.get("store_errors") or {}).items())
    return (f'<p>discovered {when(d["at"])} · version {e(disc.get("version"))} · strategy {e(disc.get("strategy"))} · '
            f'{len(m["stocks"])} stocks, {len(m["flows"])} flows</p>{errs}'
            '<section><div id="model" class="graph"></div></section>'
            f'<section><h2>Stocks</h2>{stocks}</section>'
            f'<section><h2>Flows</h2>{table(["flow", "from", "to", "rate", "moved by (guessed from timing)"], flows, "no flows")}</section>'
            f'<section><h2>Predictions</h2>{preds}</section><section><h2>Versions</h2>{versions}</section>'
            f'<details><summary>Left out ({len(disc.get("left_out") or {})})</summary>{left}</details>')


def core(d: dict) -> str:
    rows = [[f'<span class="{"ok" if p["ok"] else "bad"}">{"ok" if p["ok"] else "failing"}</span>', f'<b>{e(p["part"])}</b>', e(p["said"]),
             "<br>".join(f"<code>{e(x)}</code>" for x in p["errors"])] for p in d["parts"]]
    return table(["", "part", "health", "recent errors"], rows)


def login(nxt: str, error: str) -> str:
    err = f'<p class="bad">{e(error)}</p>' if error else ""
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Log in</title><link rel="stylesheet" href="/static/style.css"></head>
<body><main class="login"><h1>Orchestra</h1>{err}<form method="post" action="/login" id="f">
<input type="hidden" name="next" value="{e(nxt)}"><label>Token <input type="password" name="token" id="t" autofocus></label>
<button>Log in</button></form><p class="quiet">orchestra run --web prints a link that logs you in.</p></main>
<script>const m=location.hash.match(/token=([^&]+)/);if(m){{history.replaceState(null,"",location.pathname+location.search);
document.getElementById("t").value=decodeURIComponent(m[1]);document.getElementById("f").submit();}}</script></body></html>"""
