# Why CRDTs, for a fleet of agents

The goal is better coordination among many agents (AI agents, decision agents and programs)
working on the same things at once. This document argues that the thing they work on should be a
**CRDT whose operations are named facts**, says exactly what that buys over the alternatives,
answers the objection that it duplicates the system dynamics model, and says what it costs.

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
without a coordinator. That sounds like distributed databases and offline document editors, not
like agents talking to one hub. But an agent that reads an item and then thinks for four minutes
*is* a disconnected replica: it holds a copy, edits it on its own, and syncs back late. Fifty of
them are fifty replicas, each four minutes stale, each with its own edits. That is the CRDT problem
exactly, and it is why snapshots, locks and "last write wins on the whole object" keep failing for
agents: they are the wrong tools for disconnected replicas.

What a CRDT guarantees, in agent terms: **every agent's changes, applied in any order, any number of
times, however late, produce the same item, with every agent's work in it.** No agent waits, no
agent referees, no write is lost.

## 3. What it looks like

The AI writing an integration writes a **model**: the item's fields, each with a CRDT type from a
short menu, and the named facts that change it. Orchestra derives the merge from the types; nobody
writes merge code.

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

Facts are the only way to change an item. Each line a processor prints is one fact, with the
version it was based on:

```jsonl
{"type": "incident.severity_raised", "subject": "incident/2026-11-05-payments", "base": 41, "data": {"to": "SEV1"}}
{"type": "incident.finding_added",   "subject": "incident/2026-11-05-payments", "base": 41, "data": {"id": "f3", "text": "errors began at the 10:02 deploy", "evidence": ["deploy d881", "error_rate 0.4% -> 6%"]}}
```

## 4. Each coordination problem, solved

**Stale context (1).** A running agent's stdin keeps receiving the facts that touch its item while
it works ("since you read v41: ResponderJoined carol, FindingAdded f2, TimelineEntry x3"). It
updates its working copy incrementally: deltas, not a re-read. And its own late writes still merge,
because CRDT operations don't depend on being fresh. The status update stops being a four-minute-old
snapshot: an agent keeps `draft_update` current as facts arrive, and when you need it, it's seconds
old.

**Lost updates (2).** There are no whole-object writes, only operations on typed fields. Two
agents' operations on different fields, different keys, a set or a log never overwrite each other.
On a single value, `Register` keeps the latest and `Choice` keeps both and says so.

**Edit conflicts (3).** Agents edit by addressing structure ("set `findings.f3`", "append to
`timeline`") against the version they read, not by quoting text. Orchestra places the operation on
the current version; nothing has to match. A shared prose document is a `Map` of sections, so two
agents rewriting different sections never collide, and the same section becomes a `Choice` conflict
someone resolves instead of a garbled merge.

**Duplicate work (4).** A claim is an operation too: `Map[job, Claim]` where a claim has a holder
and a lease. Two agents claiming the same job at once resolve deterministically (earliest event
wins, ties by id); the loser sees on its stdin that it lost and stops. That's work allocation with
no scheduler, the stigmergy the notes in `~/thoughts.md` §7 describe.

**Unvalidated writes (5).** Each fact's `requires` is checked against the state it lands on.
`SeverityRaised` on a resolved incident is rejected, recorded as rejected (with who sent it), and
the agent is told. The facts and their rules are a state machine shared by every writer, enforced
outside all of them.

**Attribution and undo (6).** Every fact names its source. Undoing a bad run is emitting the
inverse of exactly its facts (`Set` remove for its adds, the previous `Register` value), which the
CRDT merges like anything else, and which leaves the rest of the fleet's concurrent work intact.

**Context bloat (7).** An agent needs the current state once and deltas after; the facts are small
because they say only what changed, and they say it by name.

## 5. Why not the alternatives

| | Shared snapshots (prompts 1 and 2) | Locks or a scheduler | Central log, last write wins per field | CRDT with named facts |
|---|---|---|---|---|
| 50 interleaved writers | overwrite each other | wait for each other | mostly fine; single values silently lost | merge; conflicts on single values kept and shown |
| Agents 4 minutes stale | their writes are wrong | they held a lock for 4 minutes | late writes land, and may undo newer ones | late writes merge by type; deltas keep them current |
| Concurrent edits to collections | lost | serialized | the whole collection replaced | sets, maps and logs merge |
| Impossible transitions | become truth | caught only if the lock holder checks | become truth | rejected by the fact's rule |
| Undo one agent's work | restore an old snapshot, losing others' work | same | hard | inverse facts, merged |
| Several Orchestras (work and home laptops, a teammate's) | can't sync | needs a shared server | needs a shared server | sync logs both ways; they converge |

The last row is the reason that needs a CRDT rather than a central sequencer: as soon as more than
one Orchestra holds the same items (your work laptop and home laptop, or two people's), there is no
hub to order everything, and only CRDT semantics make them agree.

## 6. The objection: "if the AI writes the facts, it's writing the system dynamics model, so why estimate?"

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
structure was given by its events, and the filter only had to fit the rates. The facts don't replace
the estimators; they hand them the structure, which is the hard part of identification. Linear noise
approximation fits naturally on top: the facts are counts of discrete transitions, exactly what LNA
models.

Nobody has to be begged to write flows. The AI writes the facts because the agents need them to
coordinate (sections 3 and 4); the flows are a by-product, counted for free.

## 7. Costs, and how to keep them small

- **The model must be written.** One small file per kind of item, by the AI, reviewable by you.
  The type menu is short (seven types) and closed; no custom merge code.
- **Orchestra must implement the types once**: about a few hundred lines for Register, Choice, Set,
  Map, Log and Counter (Text is a library, pycrdt, if ever needed), plus validation of `requires`
  and the base-version bookkeeping.
- **Facts must be named well.** A bad model (a fact called `updated` that sets everything) brings
  back the snapshot problems. The model is where that is visible and fixable.
- **Language-neutral.** The model exports a JSON Schema per fact, with each field's CRDT type as an
  annotation, so Orchestra (and processors in other languages) work from the schema, not from
  Python.
- **Not for everything.** A source you only watch and never write, trigger from or merge stays a
  metric collector or a plain event script. CRDT items are for the shared work: your tickets, PRs,
  incidents, tasks, and the documents agents write together.

## 8. What would be built, in order

1. The type menu and generic merge in Orchestra, driven by a model's JSON Schema.
2. Facts with `base` versions; rejection by `requires`, recorded as events.
3. Delivery of an item's new facts to running processors working on it (deltas on stdin).
4. Claims with leases, as a `Map` type, for work allocation without a scheduler.
5. The setup brief asks for the model first, then the collector and processors that import it.
6. Later: inverse facts for undo; syncing logs between Orchestras; `Text`.

Steps 1 to 3 are the coordination core; the rest follows when a real case asks for it.
