# Why CRDTs, for a fleet of agents

The goal is better coordination among many agents (AI agents, decision agents and programs)
working on the same things at once. This document argues that the thing they work on should be a
**CRDT whose operations are named facts**, says where those facts come from (mostly not from the
agents themselves), says exactly what that buys over the alternatives, answers the objection that it
duplicates the system dynamics model, and says what it costs.

## 1. The coordination problems, as they happen today

Fifty processors, interleaved, each running from seconds to thirty minutes, on shared items
(tickets, PRs, incidents, tasks). These failures happen with any shared-snapshot design:

1. **Stale context.** An agent reads at 10:00, reasons until 10:04, and writes as if 10:00 were
   still true. Your outage status update is this.
2. **Lost updates.** Two agents read version 41, both write a whole new version; the second
   silently erases the first's work.
3. **Edit conflicts.** An agent's partial edit names text that changed since it read it (the Edit
   tool's retry loop), or it rewrites a whole document to change one line.
4. **Duplicate work.** Two agents pick up the same job because nothing tells either the other has.
5. **Unvalidated writes.** An agent writes something impossible (approves a closed ticket), and it
   becomes the truth.
6. **No attribution or undo.** After a bad run, nobody can say which agent changed what, or take
   back exactly its changes and nothing else.
7. **Context bloat.** To be current, every agent re-reads everything, every time.

A lock or a central scheduler fixes some of these by making agents wait for each other, which
throws away the reason for having fifty of them. The question is whether something outside the
agents can make concurrent work merge correctly without making them wait.

## 2. The key observation: long-running agents are disconnected replicas

CRDTs exist for replicas that edit the same data while disconnected, then sync, and must agree
without a coordinator. An agent that reads an item and then thinks for four minutes *is* a
disconnected replica: it holds a copy, edits it on its own, and syncs back late. Fifty of them are
fifty replicas, each four minutes stale, each with its own edits. That is the CRDT problem exactly,
and it is why snapshots, locks and "last write wins on the whole object" keep failing for agents.

This is not an analogy. An AI harness already keeps each agent's **session log**: JSON lines, each
an immutable entry with a `uuid` and a `parentUuid` (a message, a tool request, the tool's result).
That is a replica's local operation log with causal links, and a set of immutable entries joined by
parent pointers is the simplest CRDT there is: two copies merge by union on `uuid`, in any order,
however late. Fifty agents' session logs merge into one log with no coordination at all.

What a CRDT guarantees, in agent terms: **every agent's changes, applied in any order, any number of
times, however late, produce the same item, with every agent's work in it.** No agent waits, no
agent referees, no write is lost.

## 3. Where the facts come from

Asking agents to emit facts themselves (a `rename_function` tool instead of Edit, a
`report_work` call at the end of each turn) puts the burden on the one participant least reliable
at formal output. Agents should work the way they already work. The facts are derived, at two
levels:

- **Raw events: the session log, wrapped.** Each entry goes into Orchestra's log as it is, inside
  our envelope (`type: agent.session.entry`, `subject`: the session, `data`: the entry, with large
  tool results truncated to a hash and a pointer to the original file). They merge by union and
  never conflict, because nothing in them is shared.
- **Named facts: classified from windows of that log.** A window (one turn, or one tool request
  with its result and the reasoning before it) holds the request that started the work, the agent's
  reasoning, the exact tool inputs and the results, test output included: intent and outcome
  together. A cheap classifier (`orchestra decide`, with the fact catalogue as its closed list of
  answers) names the fact; a program fills in its changeset; a check confirms it against an outcome
  (the test failed before and passes after). The named fact points back to its evidence:

```jsonl
{"type": "agent.session.entry", "subject": "session/7f3a", "source": "agent/worker-3", "id": "e41", "data": {"uuid": "e41", "parentUuid": "e40", "type": "assistant", "message": {"content": [{"type": "tool_use", "name": "Edit", "input": {"file_path": "src/inventory.py", "old_string": "…", "new_string": "…"}}]}}}
{"type": "agent.session.entry", "subject": "session/7f3a", "source": "agent/worker-3", "id": "e42", "data": {"uuid": "e42", "parentUuid": "e41", "type": "user", "toolUseResult": {"sha256": "c09e…", "truncated": true}}}
{"type": "code.test_fixed", "subject": "demo/tests/test_inventory.py::test_reserve", "source": "classifier/v3", "caused_by": ["e38", "e41", "e42", "e45"], "data": {"by": "agent/worker-3", "confidence": 0.94, "checked": "failed before, passes after", "changed": ["demo/src/inventory.py::reserve"]}}
```

Two rules make derived facts safe in a CRDT:

1. **A classification is recorded once and never rerun.** Replicas must apply the same operations,
   and a model is not deterministic, so the label is itself the fact, with the classifier's version
   and confidence. Replicas copy the label; they never reclassify. A better classifier later emits
   correcting facts that supersede the old ones; it never rewrites them.
2. **Facts that replay and facts that describe are different.** A `function_renamed` produced by a
   rename tool (or recognised exactly by a syntax-tree diff) can be applied again on top of someone
   else's change: a rename and a change to the same function's body commute, where text patches
   conflict. A `test_fixed` classified from a session describes an edit made as text, and git still
   merges that text. So for code, derived facts mostly describe, and the merge of the code stays in
   git unless a fact replays. The CRDT's own territory is the shared record that is not code:
   items, claims, findings, statuses, the model.

Agents can still emit facts directly when they have a natural tool for it (a decision agent's
answer is already a fact), and programs always can. Derivation is for the work that happens in free
form.

## 4. What it looks like

The AI writing an integration writes a **model**: the item's fields, each with a CRDT type from a
short menu, and the named facts that change it. The same catalogue is the classifier's list of
answers. Orchestra derives the merge from the types; nobody writes merge code.

```python
from orchestra_model import Entity, Fact, Register, Set, Map, Counter, Log, Choice

class Incident(Entity):                         # kind: incident, subject: incident/2026-11-05-payments
    severity:  Choice[Literal["SEV3", "SEV2", "SEV1"]]   # one value; concurrent different values are kept and flagged
    status:    Register[Literal["investigating", "mitigated", "resolved"]]   # latest wins
    commander: Register[str | None]
    responders: Set[str]                        # add and remove commute; a concurrent add beats a remove
    timeline:  Log[TimelineEntry]               # append-only, ordered by event time
    findings:  Map[str, Finding]                # keyed parts: each agent writes its own keys
    pages_sent: Counter                         # concurrent increments add up
    draft_update: Register[str]                 # the status update, kept current

class SeverityRaised(Fact[Incident]):
    to: Literal["SEV2", "SEV1"]
    requires = "status != 'resolved'"           # a transition rule, checked when the fact arrives
    def ops(self): return {"severity": self.to}

class FindingAdded(Fact[Incident]):
    id: str; text: str; evidence: list[str]
    def ops(self): return {"findings": {self.id: Finding(text=self.text, evidence=self.evidence)}}

class ResponderJoined(Fact[Incident]):
    who: str
    def ops(self): return {"responders": {"add": self.who}}
```

Every field type merges on its own:

| Type | Concurrent writes merge as | Typical use |
|---|---|---|
| `Register` | latest by event time wins (ties by id) | status, last seen, current draft |
| `Choice` | all concurrent values kept, flagged as a conflict until one fact picks | severity, owner, decisions |
| `Set` | adds and removes commute; add wins over a concurrent remove | responders, labels, reviewers |
| `Map` | per key, each value by its own type | findings, comments, sections of a document |
| `Log` | append-only, ordered by time | timeline, notes, observations |
| `Counter` | increments add | pages sent, retries, rounds |
| `Text` (later, only if needed) | character-level merge | a shared prose document |

Facts are the only way to change an item, whoever produces them: a program printing them, a
decision agent answering, or the classifier reading an agent's session. Each carries the version it
was based on:

```jsonl
{"type": "incident.severity_raised", "subject": "incident/2026-11-05-payments", "base": 41, "data": {"to": "SEV1"}}
{"type": "incident.finding_added",   "subject": "incident/2026-11-05-payments", "base": 41, "data": {"id": "f3", "text": "errors began at the 10:02 deploy", "evidence": ["deploy d881", "error_rate 0.4% -> 6%"]}}
```

For a derived fact, `base` comes from the session log, not from the agent: the version it read is
in its Read result.

## 5. When a fact happens: branches

A fact belongs to the stream it happened on. An agent working on a branch produces facts about that
branch ("removed `legacy.py` on `fix-17`"); they are true, and they are kept, but they never change
main's items. Main's items fold only main's facts.

The merge is the fact that carries work across: at merge time, the net change that actually landed
(main's before and after, so a rebase or a conflict resolution is accounted for, and a file deleted
and restored three turns later is not) is classified once, and those facts are applied to main,
each caused by the merge. A branch that is never merged stays a record of an attempt: useful for
calibration (how often an agent's work lands), for spotting wasted work, and for telling the next
agent what was tried. So there are two moments to classify: cheaply, per window on the branch, as a
running account others can watch; and once, at merge, for the facts that count.

## 6. Each coordination problem, solved

**Stale context (1).** A running agent's stdin keeps receiving the facts that touch its item while
it works ("since you read v41: ResponderJoined carol, FindingAdded f2, TimelineEntry x3"). It
updates its working copy incrementally: deltas, not a re-read. Its own late writes still merge,
because CRDT operations don't depend on being fresh. And staleness is detectable without the agent's
help: its session log says which version it read, and Orchestra knows the current one.

**Lost updates (2).** There are no whole-object writes, only operations on typed fields. Two
agents' operations on different fields, different keys, a set or a log never overwrite each other.
On a single value, `Register` keeps the latest and `Choice` keeps both and says so.

**Edit conflicts (3).** On items, agents' work lands as operations addressed to structure ("set
`findings.f3`", "append to `timeline`") against the version read, not as quoted text; nothing has
to match. A shared prose document is a `Map` of sections, so two agents rewriting different sections
never collide, and the same section becomes a `Choice` conflict someone resolves instead of a garbled
merge. On code, replayable facts (renames, added parameters) rebase onto each other structurally;
the rest is git's merge, as now.

**Duplicate work (4).** A claim is an operation too: `Map[job, Claim]` where a claim has a holder
and a lease. Two agents claiming the same job at once resolve deterministically (earliest event
wins, ties by id); the loser sees on its stdin that it lost and stops. That's work allocation with
no scheduler.

**Unvalidated writes (5).** Each fact's `requires` is checked against the state it lands on, and a
derived fact is also checked against its outcome before it is recorded. `SeverityRaised` on a
resolved incident, or a `test_fixed` whose test still fails, is rejected, recorded as rejected (with
who caused it), and the agent is told. The facts and their rules are a state machine shared by every
writer, enforced outside all of them.

**Attribution and undo (6).** Every fact names its source, and a derived fact names the session
entries it came from, so any change traces to the agent's exact turn and reasoning. Undoing a bad
run is emitting the inverse of exactly the facts caused by its session (`Set` remove for its adds,
the previous `Register` value), which the CRDT merges like anything else, and which leaves the rest
of the fleet's concurrent work intact.

**Context bloat (7).** An agent needs the current state once and deltas after; the facts are small
because they say only what changed, and they say it by name.

## 7. Why not the alternatives

| | Shared snapshots (prompts 1 and 2) | Locks or a scheduler | Central log, last write wins per field | CRDT with named facts |
|---|---|---|---|---|
| 50 interleaved writers | overwrite each other | wait for each other | mostly fine; single values silently lost | merge; conflicts on single values kept and shown |
| Agents 4 minutes stale | their writes are wrong | they held a lock for 4 minutes | late writes land, and may undo newer ones | late writes merge by type; deltas keep them current |
| Concurrent edits to collections | lost | serialized | the whole collection replaced | sets, maps and logs merge |
| Impossible transitions | become truth | caught only if the lock holder checks | become truth | rejected by the fact's rule |
| Undo one agent's work | restore an old snapshot, losing others' work | same | hard | inverse facts, merged |
| Several Orchestras (work and home laptops, a teammate's) | can't sync | needs a shared server | needs a shared server | sync logs both ways; they converge |

The last row is the reason that needs a CRDT rather than a central sequencer: as soon as more than
one Orchestra holds the same items, there is no hub to order everything, and only CRDT semantics
make them agree. Session logs and recorded classifications sync the same way: by union.

## 8. The objection: "if the AI writes the facts, it's writing the system dynamics model, so why estimate?"

It isn't. The model file declares **what can happen**; the system dynamics model is about **how
fast, how much, after how long and because of what**. These are different things, and only the
first can be written down:

| The model file gives | Only data gives |
|---|---|
| the states an item can be in (stocks) | how many items are in each state over time |
| the transitions that can happen (possible flows) | each transition's rate, and what that rate depends on |
| who may cause each transition | how long items stay in each state, and how that stretches under load |
| | delays, and feedback: does a long review queue cause faster, sloppier reviews and more reopens later? |
| | the links to the observed system: does a deploy-fixing flow move the error rate, and with what lag? |
| | hidden states nobody declared (a growing backlog no one tracks as an item) |

So the declared facts are the model's **structure**, and the estimation gives its **parameters and
couplings**. This is grey-box system identification, the classical setting for Kalman filtering:
the structure is known, the parameters and some states are hidden, and the measurements are noisy.
It is also why the task queue was found so well and the kube simulation wasn't: the queue's
structure was given by its events, and the filter only had to fit the rates. Linear noise
approximation fits naturally on top: the facts are counts of discrete transitions, exactly what LNA
models.

Nobody has to be begged to write flows, and with derived facts nobody has to be begged to emit them
either: the agents work as they do, the classifier names what they did, and the counts of those
facts are the flows, merged facts on main for the system, branch facts for the agents themselves
(attempts, abandoned branches, rework).

## 9. Costs, and how to keep them small

- **The model must be written.** One small file per kind of item, by the AI, reviewable by you.
  The type menu is short (seven types) and closed; no custom merge code. The same file is the
  classifier's list of answers.
- **Orchestra must implement the types once**: a few hundred lines for Register, Choice, Set, Map,
  Log and Counter (Text is a library, pycrdt, if ever needed), plus validation of `requires` and the
  base-version bookkeeping.
- **Facts must be named well, and now that is measurable.** A bad model (a fact called `updated`
  that sets everything) brings back the snapshot problems. With a classifier, a missing name shows
  up directly as a high rate of `other`, on the windows that need one.
- **A wrong classification is a wrong fact that merges cleanly.** Hence a confidence threshold
  (below it, the window stays raw events), a check against an outcome wherever one exists, and the
  raw events kept so facts can be derived again and corrected.
- **Session logs are large and hold everything an agent read.** Keep their structure, truncate big
  tool results to a hash and a pointer, and decide per project what may be ingested at all.
- **Language-neutral.** The model exports a JSON Schema per fact, with each field's CRDT type as an
  annotation, so Orchestra (and processors in other languages) work from the schema, not from
  Python.
- **CRDT merge types are for shared writes.** Defining the facts is always worth it, for a source
  you only watch too: they are the vocabulary, the classifier's answers and the model's structure.
  The typed merge matters only where several writers touch the same item.

## 10. What would be built, in order

1. The type menu and generic merge in Orchestra, driven by a model's JSON Schema.
2. Facts with `base` versions; rejection by `requires`, recorded as events.
3. Session logs ingested as wrapped entries, and a classifier processor that turns windows into
   named facts with `caused_by`, its version and confidence, checked against outcomes.
4. Delivery of an item's new facts to running processors working on it (deltas on stdin).
5. Branch streams, and classification of the net change at merge.
6. Claims with leases, as a `Map` type, for work allocation without a scheduler.
7. The setup brief asks for the model first, then the collector and processors that import it.
8. Later: inverse facts for undo; syncing logs between Orchestras; replayable code facts; `Text`.

Steps 1 to 4 are the coordination core; the rest follows when a real case asks for it.
