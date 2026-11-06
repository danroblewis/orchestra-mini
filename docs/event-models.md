# Event models compared: a Jira integration six ways

One running example, done each way, so the models can be compared side by side.

**The example.** Our team's Jira tickets come into Orchestra. One processor reacts: when a ticket
moves to *In Review* and has no reviewer, it picks one from a rota and records *on our side* that a
review is due (it never writes to Jira). We want, for every model:

- the **collector**: reads Jira, writes lines to stdout;
- the **processor**: reads lines on stdin, writes lines to stdout;
- the **stdin and stdout** each sees;
- what **Orchestra** must do;
- what it costs.

The same Jira history is used throughout:

```
ABC-123  created by alice                      2026-11-05 09:00
ABC-123  status To Do -> In Progress  by alice 2026-11-05 09:30
ABC-123  labels + "billing"            by bob  2026-11-05 09:40
ABC-123  comment c501 "is this safe?"  by bob  2026-11-05 10:00
ABC-123  status In Progress -> In Review by alice 2026-11-05 10:05
```

Times compare as strings because every time here is ISO 8601 in UTC, as Jira Data Center returns
them with `+0000`; a real collector would parse them. One more real-world limit: a search with
`expand=changelog` returns at most the latest 100 histories per issue; a collector backfilling a
long-lived ticket reads `/rest/api/2/issue/<key>/changelog` (Cloud) or the issue with
`expand=changelog` (Data Center) for the rest.

The Jira API used by every collector is the same: `GET /rest/api/2/search?jql=project=ABC AND
updated >= "<cursor>"&expand=changelog&fields=summary,status,assignee,labels,comment`. It returns
each ticket's current fields and its changelog (`histories[]`, each with `created`, `author` and
`items[]` of `{field, fromString, toString}`).

A shared helper for the collectors (any language works; this is Python, no Orchestra library).
`pick_from_rota()` in the processors stands for whatever business rule picks a reviewer.

```python
# jira.py: the only code that talks to Jira
import json, os, urllib.request, urllib.parse

BASE, TOKEN = os.environ["JIRA_URL"], os.environ["JIRA_TOKEN"]

def search(jql):
    start = 0
    while True:
        q = urllib.parse.urlencode({"jql": jql, "expand": "changelog", "startAt": start, "maxResults": 50,
                                    "fields": "summary,status,assignee,labels,comment,created,creator"})
        req = urllib.request.Request(f"{BASE}/rest/api/2/search?{q}", headers={"Authorization": f"Bearer {TOKEN}"})
        page = json.load(urllib.request.urlopen(req))
        yield from page["issues"]
        start += len(page["issues"])
        if start >= page["total"] or not page["issues"]:
            return

from datetime import datetime, timedelta, timezone

def cursor(path, new=None):          # the collector's own memory: the time it last read up to (ISO, UTC)
    if new is not None:
        open(path, "w").write(new)
    try:
        return open(path).read().strip()
    except FileNotFoundError:
        return (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%S.000+0000")

def jql_time(iso):                   # Jira's JQL takes "yyyy-MM-dd HH:mm"
    return iso[:10] + " " + iso[11:16]

def next_cursor(overlap_minutes=10):  # now, less an overlap; repeated facts are dropped by their ids
    return (datetime.now(timezone.utc) - timedelta(minutes=overlap_minutes)).strftime("%Y-%m-%dT%H:%M:%S.000+0000")

def emit(line):
    print(json.dumps(line), flush=True)
```

---

## Model A: event notification (ids only)

Events say *that* something happened to *what*; they carry no state. Whoever cares looks the thing
up.

**Collector** (one event per changed ticket per poll; it does not even need the changelog):

```python
import jira
since = jira.cursor(".cursor")
for issue in jira.search(f'project = ABC AND updated >= "{jira.jql_time(since)}"'):
    jira.emit({"type": "jira.issue.changed", "subject": issue["key"], "time": issue["fields"]["updated"]})
jira.cursor(".cursor", new=jira.next_cursor())
```

**stdout of the collector:**

```jsonl
{"type": "jira.issue.changed", "subject": "ABC-123", "time": "2026-11-05T10:05:00Z"}
```

**Processor** (`on: jira.issue.changed`). It must fetch the ticket, from Jira or from a store, and
remember on its own side what it already did:

```python
import json, os, sys, jira
event = json.loads(sys.stdin.readline())
issue = next(jira.search(f'key = {event["subject"]}'))          # back to Jira for the state
status = issue["fields"]["status"]["name"]
path = f"state/{event['subject']}.json"                         # our own data: its problem
mine = json.load(open(path)) if os.path.exists(path) else {}
if status == "In Review" and not mine.get("reviewer"):
    reviewer = pick_from_rota()
    mine["reviewer"] = reviewer
    json.dump(mine, open(f"state/{event['subject']}.json", "w"))
    jira.emit({"type": "review.assigned", "subject": event["subject"], "data": {"reviewer": reviewer}})
```

**Orchestra does:** stores and routes events. Nothing else; there is no state to fold.

**Costs.**
- Every processor calls Jira for every event: slow, rate-limited, and it only ever sees *now*.
- "Did it move to In Review in *this* change, or was it already there?" is unanswerable. The
  processor re-assigns, or keeps its own memory.
- No history, so no flows: "tickets entering review per hour" can't be computed.
- Our own data (the reviewer) lives in each processor's private files.

The simplest collector, and everything else gets harder.

---

## Model B: event-carried state transfer (full snapshots)

Every event carries the whole entity as it is now. The latest snapshot is the state.

**Collector:**

```python
import jira
since = jira.cursor(".cursor")
for issue in jira.search(f'project = ABC AND updated >= "{jira.jql_time(since)}"'):
    f = issue["fields"]
    jira.emit({"type": "jira.issue.snapshot", "subject": issue["key"], "time": f["updated"],
               "data": {"summary": f["summary"], "status": f["status"]["name"],
                        "assignee": (f["assignee"] or {}).get("name"), "labels": f["labels"],
                        "comments": [{"id": c["id"], "by": c["author"]["name"], "text": c["body"]}
                                     for c in f["comment"]["comments"]]}})
jira.cursor(".cursor", new=jira.next_cursor())
```

**stdout of the collector** (one line per poll that saw it changed):

```jsonl
{"type": "jira.issue.snapshot", "subject": "ABC-123", "time": "2026-11-05T10:05:00Z",
 "data": {"summary": "Fix retries", "status": "In Review", "assignee": "alice", "labels": ["billing"],
          "comments": [{"id": "c501", "by": "bob", "text": "is this safe?"}]}}
```

**Processor:**

```python
import json, sys
ev = json.loads(sys.stdin.readline())
s = ev["data"]
if s["status"] == "In Review" and not ev.get("ours", {}).get("reviewer"):   # where does "ours" come from? see costs
    print(json.dumps({"type": "review.assigned", "subject": ev["subject"], "data": {"reviewer": pick_from_rota()}}))
```

**Orchestra does:** keeps the latest snapshot per subject (fold = "replace").

**Costs.**
- **Intermediate states are lost.** Polls every 5 minutes: if a ticket went To Do → In Progress → In
  Review between polls, we see one jump. Flows between states are wrong.
- **"Did it just change?"** The processor gets a whole snapshot and must compare it with the
  previous one to know whether the status *changed* to In Review.
- **Lists are fine here** (the whole list is replaced each time), but the log is mostly repeated data.
- **Two writers conflict.** If a processor also writes a snapshot of the same subject (with our
  `reviewer`), the next Jira snapshot replaces it. Our data needs a separate subject.

Simple to write, wrong about history, wasteful.

---

## Model C: event sourcing with domain facts (operation-based, last-writer-wins map)

Each event is a *fact*: a past-tense name for what happened, at the time it happened, carrying the
change as a JSON Merge Patch (RFC 7396: objects merge key by key, `null` deletes). Collections are
maps keyed by id, never arrays. The state is the fold of the patches; each leaf keeps the time of
the fact that set it, so applying facts in any order gives the same state (an operation-based
CRDT: the operations are the patches, the merge is last-writer-wins per leaf, ties by event id).

**Collector** (an anti-corruption layer: translates Jira's changelog into *our* facts; Jira's
fields under a `jira` namespace so they never collide with ours):

```python
import jira

FACTS = {"status": "status_changed", "assignee": "assigned", "labels": "labelled"}

since = jira.cursor(".cursor")
for issue in jira.search(f'project = ABC AND updated >= "{jira.jql_time(since)}"'):
    key, f = issue["key"], issue["fields"]
    if f["created"] >= since:                         # created within this window
        jira.emit({"type": "jira.issue.created", "subject": key, "id": f"{key}:created", "time": f["created"],
                   "source": f"person.jira.{f['creator']['name']}",
                   "data": {"jira": {"summary": f["summary"], "status": "To Do", "url": f"{jira.BASE}/browse/{key}"}}})
    for h in issue["changelog"]["histories"]:
        if h["created"] < since:
            continue
        for item in h["items"]:
            fact = FACTS.get(item["field"])
            if not fact:
                continue                              # a field we don't track: deliberately not translated
            if item["field"] == "labels":             # Jira gives the whole new list: turn it into a keyed set
                old, new = set(item["fromString"].split()), set(item["toString"].split())
                patch = {"labels": {**{l: True for l in new - old}, **{l: None for l in old - new}}}
            else:
                patch = {item["field"]: item["toString"]}
            jira.emit({"type": f"jira.issue.{fact}", "subject": key, "id": f"{key}:{h['id']}:{item['field']}",
                       "time": h["created"], "source": f"person.jira.{h['author']['name']}", "data": {"jira": patch}})
    for c in f["comment"]["comments"]:
        if c["created"] >= since:
            jira.emit({"type": "jira.issue.commented", "subject": key, "id": f"{key}:comment:{c['id']}",
                       "time": c["created"], "source": f"person.jira.{c['author']['name']}",
                       "data": {"jira": {"comments": {c["id"]: {"by": c["author"]["name"], "text": c["body"]}}}}})
jira.cursor(".cursor", new=jira.next_cursor())        # overlap; ids make repeats harmless
```

**stdout of the collector** (the history above, one fact per change):

```jsonl
{"type": "jira.issue.created",        "subject": "ABC-123", "id": "ABC-123:created",            "time": "2026-11-05T09:00:00Z", "source": "person.jira.alice", "data": {"jira": {"summary": "Fix retries", "status": "To Do", "url": "https://jira.example/browse/ABC-123"}}}
{"type": "jira.issue.status_changed", "subject": "ABC-123", "id": "ABC-123:9001:status",        "time": "2026-11-05T09:30:00Z", "source": "person.jira.alice", "data": {"jira": {"status": "In Progress"}}}
{"type": "jira.issue.labelled",       "subject": "ABC-123", "id": "ABC-123:9002:labels",        "time": "2026-11-05T09:40:00Z", "source": "person.jira.bob",   "data": {"jira": {"labels": {"billing": true}}}}
{"type": "jira.issue.commented",      "subject": "ABC-123", "id": "ABC-123:comment:c501",       "time": "2026-11-05T10:00:00Z", "source": "person.jira.bob",   "data": {"jira": {"comments": {"c501": {"by": "bob", "text": "is this safe?"}}}}}
{"type": "jira.issue.status_changed", "subject": "ABC-123", "id": "ABC-123:9003:status",        "time": "2026-11-05T10:05:00Z", "source": "person.jira.alice", "data": {"jira": {"status": "In Review"}}}
```

**The folded entity** (what Orchestra keeps):

```json
{"subject": "ABC-123", "kind": "jira.issue", "version": 5,
 "state": {"jira": {"summary": "Fix retries", "status": "In Review", "url": "…",
                    "labels": {"billing": true}, "comments": {"c501": {"by": "bob", "text": "is this safe?"}}}}}
```

**Processor, written against the raw contract** (`on: jira.issue.status_changed`; it gets the
fact and fetches the entity itself):

```python
import json, subprocess, sys
ev = json.loads(sys.stdin.readline())
if ev["data"]["jira"].get("status") != "In Review":
    sys.exit(0)                                        # the fact says exactly what changed: no comparing
entity = json.loads(subprocess.check_output(["orchestra", "query", "items", "show", ev["subject"]]))
if not entity["state"].get("mine", {}).get("reviewer"):
    print(json.dumps({"type": "review.assigned", "subject": ev["subject"],
                      "data": {"mine": {"reviewer": pick_from_rota(), "review_due": "2026-11-06"}}}))
```

**Its stdout:**

```jsonl
{"type": "review.assigned", "subject": "ABC-123", "data": {"mine": {"reviewer": "carol", "review_due": "2026-11-06"}}}
```

The entity now holds `jira.*` (from the collector) and `mine.*` (from us) side by side; neither
writer can overwrite the other.

**Orchestra does:**
- stores facts; drops a repeated `id`;
- folds patches per subject (last-writer-wins per leaf by event time, ties by id; `null` deletes;
  maps merge, arrays replace whole);
- serves the folded entity (`orchestra query items show`);
- projects entities and facts into series: open tickets by `jira.status` (stocks), status
  changes per interval (flows), time in status (durations).

**Costs.**
- **The collector is the hardest piece**: it must translate Jira's changelog into facts, handle
  lists as keyed maps, and pick ids that are stable across re-reads. Roughly 40 lines here.
- **A fact type it doesn't translate is silently missing** (a priority change, a resolution). A
  reconciler is needed eventually (see Model F).
- **Processors that need context must query** (one extra call), unless Orchestra hands the entity in.

Exact history, exact flows, our data beside theirs, small events. The processor is simple *because*
the fact names what changed.

---

## Model D: state-based CRDT (whole state with per-field clocks)

The other main CRDT family. Instead of operations, each event carries a *whole state*, where every
field carries its own timestamp. Merging two states takes, per field, the value with the later
timestamp. Merge is commutative, associative and idempotent, so any two copies converge.

**Collector** (rebuilds the full state from the changelog, stamping each field with when it last
changed):

```python
import jira

def stamped_state(issue):
    f, st = issue["fields"], {}
    st["summary"] = {"v": f["summary"], "t": f["created"]}
    for h in sorted(issue["changelog"]["histories"], key=lambda h: h["created"]):
        for item in h["items"]:
            if item["field"] in ("status", "assignee"):
                st[item["field"]] = {"v": item["toString"], "t": h["created"]}
            if item["field"] == "labels":
                for l in item["toString"].split():
                    st.setdefault("labels", {})[l] = {"v": True, "t": h["created"]}
                for l in set(item["fromString"].split()) - set(item["toString"].split()):
                    st.setdefault("labels", {})[l] = {"v": None, "t": h["created"]}     # tombstone
    for c in f["comment"]["comments"]:
        st.setdefault("comments", {})[c["id"]] = {"v": {"by": c["author"]["name"], "text": c["body"]}, "t": c["created"]}
    st.setdefault("status", {"v": "To Do", "t": f["created"]})
    return st

since = jira.cursor(".cursor")
for issue in jira.search(f'project = ABC AND updated >= "{jira.jql_time(since)}"'):
    jira.emit({"type": "jira.issue.state", "subject": issue["key"], "time": issue["fields"]["updated"],
               "data": {"jira": stamped_state(issue)}})
jira.cursor(".cursor", new=jira.next_cursor())
```

**stdout of the collector:**

```jsonl
{"type": "jira.issue.state", "subject": "ABC-123", "time": "2026-11-05T10:05:00Z",
 "data": {"jira": {"summary": {"v": "Fix retries", "t": "2026-11-05T09:00:00Z"},
                   "status": {"v": "In Review", "t": "2026-11-05T10:05:00Z"},
                   "labels": {"billing": {"v": true, "t": "2026-11-05T09:40:00Z"}},
                   "comments": {"c501": {"v": {"by": "bob", "text": "is this safe?"}, "t": "2026-11-05T10:00:00Z"}}}}}
```

**Processor:** gets the merged state, but no fact. To know that the status *just became* In
Review, it compares `status.t` with the last time it looked:

```python
import json, sys
ev = json.loads(sys.stdin.readline())
status = ev["data"]["jira"]["status"]
seen = load_last_seen(ev["subject"])                     # its own memory again
if status["v"] == "In Review" and status["t"] > seen and not ours_reviewer(ev["subject"]):
    print(json.dumps({"type": "review.assigned", "subject": ev["subject"],
                      "data": {"mine": {"reviewer": {"v": pick_from_rota(), "t": now()}}}}))
save_last_seen(ev["subject"], status["t"])
```

**Orchestra does:** merges each incoming state into the stored one, per field by `t`; stores the
merged state.

**Costs.**
- **Robust against missed polls and duplicates**: any later state contains everything; re-sending
  is harmless; merge order doesn't matter.
- **No facts.** The log is a series of whole states, so "what happened" is recovered by comparing
  timestamps. Triggering on a specific change needs each processor to remember what it saw.
- **Every value is wrapped** in `{"v", "t"}`; processors must read and write that format.
- **Big events**, as in Model B.

This is what you'd choose if many replicas edit the same documents offline and must converge (the
classic CRDT use case). For a single collector per system and processors that react to changes,
it's the wrong trade.

---

## Model E: richer CRDT types (sets, counters, conflict-revealing registers)

Model C used one CRDT (a last-writer-wins map). The literature has others, each solving a case
last-writer-wins handles badly. In Model C's event shape, they'd appear as typed patch operations.

**Observed-remove set (OR-set):** "add wins over a concurrent remove". Each add carries a unique tag;
a remove removes only the tags it has seen.

```jsonl
{"type": "jira.issue.labelled",   "subject": "ABC-123", "data": {"jira": {"labels": {"$add": {"billing": "tag-9002"}}}}}
{"type": "jira.issue.unlabelled", "subject": "ABC-123", "data": {"jira": {"labels": {"$remove": {"billing": ["tag-9002"]}}}}}
```

**Counter (PN-counter):** concurrent increments add up rather than overwrite.

```jsonl
{"type": "review.round_started", "subject": "ABC-123", "data": {"mine": {"review_rounds": {"$inc": 1}}}}
```

**Multi-value register:** keeps *all* concurrent values instead of picking one, so a conflict is
visible (needs version vectors, not timestamps):

```jsonl
{"type": "review.assigned", "subject": "ABC-123", "data": {"mine": {"reviewer": {"$set": "carol", "$after": ["v4"]}}}}
{"type": "review.assigned", "subject": "ABC-123", "data": {"mine": {"reviewer": {"$set": "dave",  "$after": ["v4"]}}}}
```

→ state `{"mine": {"reviewer": {"$conflict": ["carol", "dave"]}}}`, which can itself trigger a processor.

**Orchestra does:** implements each type's merge (OR-set tags, counter sums, version vectors for
MV-registers).

**Costs.** Each type is correct for its case and adds rules every writer must follow. With one
collector per system and processors that rarely write the same field, the cases they fix (two
writers adding and removing the same label at once, two processors assigning a reviewer at once)
almost never happen. Worth adding *one type at a time*, only when a real conflict shows up. The
counter is the likeliest to be needed first.

---

## Model F: Model C, with Orchestra hydrating processors

Model C's log, with the processor contract made easier: Orchestra hands each processor the fact
**and** the folded entity (and entities it references), and accepts back either facts with patches
or a whole new state, which it diffs against the version it handed in.

**The collector** is exactly Model C's.

**stdin of the processor** (one line):

```json
{"event":   {"type": "jira.issue.status_changed", "subject": "ABC-123", "id": "ABC-123:9003:status", "time": "2026-11-05T10:05:00Z", "data": {"jira": {"status": "In Review"}}},
 "entity":  {"subject": "ABC-123", "kind": "jira.issue", "version": 5,
             "state": {"jira": {"summary": "Fix retries", "status": "In Review", "labels": {"billing": true},
                                "comments": {"c501": {"by": "bob", "text": "is this safe?"}}}}},
 "related": {}}
```

**Processor**, the shortest of all:

```python
import json, sys
msg = json.loads(sys.stdin.readline())
ev, state = msg["event"], msg["entity"]["state"]
if ev["data"]["jira"].get("status") == "In Review" and not state.get("mine", {}).get("reviewer"):
    state.setdefault("mine", {}).update({"reviewer": pick_from_rota(), "review_due": "2026-11-06"})
    print(json.dumps({"type": "review.assigned", "subject": ev["subject"], "based_on": msg["entity"]["version"],
                      "state": state}))         # the whole entity back; Orchestra works out the change
```

**Its stdout:**

```jsonl
{"type": "review.assigned", "subject": "ABC-123", "based_on": 5, "state": {"jira": {"…": "…unchanged…"}, "mine": {"reviewer": "carol", "review_due": "2026-11-06"}}}
```

**What Orchestra appends to the log** (the diff against version 5):

```jsonl
{"type": "review.assigned", "subject": "ABC-123", "id": "…", "time": "2026-11-05T10:05:03Z", "source": "assign_reviewer",
 "data": {"mine": {"reviewer": "carol", "review_due": "2026-11-06"}}}
```

A processor that knows the change can still print a patch directly (`"data"` instead of
`"state"`). One that changes nothing prints nothing.

**Orchestra does:** everything in Model C, plus:
- builds each processor's stdin: the fact, the entity at its current version, and the entities its
  state references (`related`, e.g. a PR's `jira.key`);
- diffs a returned `state` against `based_on` and appends only what changed. If the entity moved on
  meanwhile, only this processor's changed fields are written; nothing it didn't touch is lost;
- refuses a returned `state` that changes another namespace's fields (`jira.*` from a non-Jira
  processor), so the boundary holds.

**A reconciler**, the other Orchestra-side piece this model makes natural: a scheduled processor
that fetches each open ticket's current Jira state, compares it with the entity's `jira.*`, and
emits the difference as one fact, so a skipped changelog type never leaves the state wrong:

```jsonl
{"type": "jira.issue.reconciled", "subject": "ABC-123", "data": {"jira": {"priority": "High"}}}
```

**Costs.** The collector is still Model C's (the real work). Orchestra gains two small features:
hydration and the diff of returned state. Processors become ordinary programs over a whole entity,
as you'd write against a database row.

---

## Side by side

| | A notification | B snapshot | C facts | D state CRDT | E typed CRDTs | F facts + hydration |
|---|---|---|---|---|---|---|
| Event size | tiny | whole entity | the change | whole entity + clocks | the change | the change |
| History of what happened | none | lossy | exact | by comparing clocks | exact | exact |
| Flows between states | no | wrong between polls | exact | approximate | exact | exact |
| Trigger on a specific change | no | must compare | yes, by type | must compare | yes | yes |
| Our data beside theirs | private files | separate subject | namespaces | namespaces | namespaces | namespaces, enforced |
| Collector | 5 lines | 10 lines | ~40 lines | ~35 lines | ~45 lines | ~40 lines (C's) |
| Processor | queries Jira, keeps memory | compares snapshots | reads the fact, queries for context | keeps memory | reads the fact | reads the fact and the entity, returns state |
| Orchestra | route | keep latest | fold patches | merge states | per-type merges | fold, hydrate, diff, guard namespaces |
| Converges with duplicates and late events | n/a | no | yes | yes | yes | yes |

## Which one

**Model F.** The log is Model C's (facts with patches: exact history and flows, small events, a
simple operation-based CRDT that converges), and the processor contract is the one you described
wanting: whole entities in, whole entities (or facts) out. The only hard code is the collector,
which translates another system's history into our facts, once per system. Model E's types are
added one at a time when a real conflict appears; Model D is for many writers editing offline,
which isn't our situation; Models A and B lose the history the system dynamics model needs.

**What Orchestra does in Model F** (and nothing else):
1. append facts, dropping repeated ids;
2. fold each subject's facts (last-writer-wins per leaf, ties by id, `null` deletes, maps merge);
3. hydrate a processor's stdin with the fact, the entity and its referenced entities;
4. turn a processor's returned state into a fact by diffing against `based_on`, refusing changes
   outside the processor's namespace;
5. project entities and facts into stocks, flows and durations in the metrics store.

**What the AI writing an integration does:**
1. the collector: translate the system's changelog into named facts with patches, keyed maps for
   collections, stable ids, its own namespace;
2. a reconciler for that system;
3. processors: business logic over whole entities, printing facts or new state.
