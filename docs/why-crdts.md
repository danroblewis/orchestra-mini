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
   still true. A status draft that misses the mitigation that landed while it was written is this.
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

### What already works, for code

For code, practitioners already handle most of this without any new layer: one worktree and branch
per agent; a planner that splits work so tasks touch different files; an orchestrator that assigns
each task to one agent; a reviewer; a merge queue that runs CI on the rebased change. Stale context
is a rebase and CI, lost updates are git's job, duplicate work is the assignment, unvalidated writes
are CI, attribution is commits, undo is a revert. This document does not propose replacing that.

What it is about is the rest: the shared records that are not code and have several writers at
once (an incident's findings, timeline and status during an outage; a ticket's fields while a
triage agent, a fixer and a person all touch it; a document several agents draft together; the
claims on jobs), and the facts about the work itself, which the git workflow does not record at
all. How often several writers really touch one record at once is not yet measured; it is the
first thing to measure before building more than section 10's first steps.

## 2. The key observation: long-running agents are disconnected replicas

CRDTs exist for replicas that edit the same data while disconnected, then sync, and must agree
without a coordinator. An agent that reads an item and then thinks for four minutes *is* a
disconnected replica: it holds a copy, edits it on its own, and syncs back late. Fifty of them are
fifty replicas, each four minutes stale, each with its own edits. That is the CRDT problem exactly,
and it is why snapshots, locks and "last write wins on the whole object" keep failing for agents.

This is close to literal, with one honest gap. An AI harness already keeps each agent's **session log**: JSON lines, each
an immutable entry with a `uuid` and a `parentUuid` (a message, a tool request, the tool's result).
That is a replica's local operation log with causal links, and a set of immutable entries joined by
parent pointers is the simplest CRDT there is: two copies merge by union on `uuid`, in any order,
however late. Fifty agents' session logs merge into one log with no coordination at all. The gap:
a CRDT replica applies operations to its copy, while an agent holds a prose rendering of the item
and its operations are reconstructed afterwards (section 3); and `parentUuid` gives causality
*within* a session, not between agents. Causality between agents has to be recorded separately, as
the facts each agent had read (`deps`, section 4).

What a CRDT guarantees, in agent terms: **every agent's changes, applied in any order, any number of
times, however late, produce the same item, with every agent's work in it.** No agent waits, no
agent referees, no write is lost. What it does not guarantee is that the converged item is the one
anyone *intended* (intention preservation): convergence is the easy half, and the field types and
conflict rules in sections 4 and 6 are where intent is protected or lost.

### The objection from event sourcing: connected and slow is not disconnected

An event-sourcing practitioner answers that a four-minute agent is not disconnected; it is
connected and slow, which is what optimistic concurrency was built for. Each subject's stream lives
in one place; an append carries the version the writer read; a stale append fails, and the
command is decided again against the new state, by a program, not by rerunning the agent. Most
commands still hold on the new state; only those the new state rejects go back to the agent.

That is right, and it changes the design: **every subject has a home Orchestra**, which appends its
facts and checks `deps` against the subject's current heads. When they differ, the home re-decides
the fact against the new state using the field types as its **conflict table**: two adds to a `Set`
never conflict, two writes to a `Register` do, a `Log` append never does. The typed fields are a
compact way to write the "which event types conflict with which" table that event stores already
use; nothing is merged blindly. What must hold at the instant of acceptance (claims, terminal
transitions like `resolved`) is checked there, with real exclusion.

The CRDT part is what is left when there is no home to ask: a laptop offline, two Orchestras
replicating each other's subjects, an agent's branch. There the same field types give the merge,
and the rules in section 6 say what can and cannot be guaranteed without coordination. So the
design is an event store per subject for decisions, and CRDT merge for replication and for working
offline, with one set of types serving both.

## 3. Where the facts come from

Asking agents to emit facts themselves (a `rename_function` tool instead of Edit, a
`report_work` call at the end of each turn) puts the burden on the one participant least reliable
at formal output, and the lab measured it. Forced (only semantic tools and one reporting tool
available), agents used them every time. Given optional `record_*` fact tools (one per fact name,
generated from the catalogue) beside Edit and Bash (strategy F): unprompted, 0 calls in 8 tasks;
told once to record, calls in 7 of 8, covering 10 of the 26 changes the detector found; told to
record after every change, calls in 5 of 8, covering 8 of 26. Across all runs it never once
recorded the intent (`test_fixed`), even on the two fix-a-test tasks: it records the code change and
stops. Its records carry no link to the edits, so they join the detector's facts only by function
name. What they do add is a reason per change, which no detector produces. An agent records some of
its work when asked, never all of it, and never the part that matters most.
Agents should work the way they already work. The facts are derived, at two
levels:

- **Evidence: the session log, kept beside the event log, not in it.** Session entries are
  high-volume and hold everything an agent read, secrets and personal data included; they are not
  domain events, and mixing them into the domain log is how an event-sourced monolith starts. They
  go into an **evidence store**: content-addressed, encrypted with one key per session (deleting the
  key erases the session everywhere it was copied, which is crypto-shredding), with its own
  retention. They merge by union and never conflict, because nothing in them is shared. The event
  log refers to them by hash.
- **Named facts: derived from the log by programs first, a classifier second.** The harness
  records the exact before and after of every tool call (a hook that diffs the working tree, so an
  edit made through the shell is caught too); a syntax-tree detector turns each diff into change
  facts (`function_renamed`, `parameter_added`, `function_body_changed`, `test_added`); a checker
  turns intent into work facts only against an outcome (the test failed on the baseline and passes
  after). A cheap classifier over a window of the session (the request, the agent's reasoning, the
  tool calls and results) adds what programs cannot: the *why*, and the windows no catalogue entry
  fits. Detector and classifier output is recorded as what it is, an observation
  (`classification.recorded`, with its label, deriver, version and confidence), not as a domain
  fact. It becomes a domain fact only when it is checked (and, for a branch, at merge), submitted as
  a command the subject's home accepts or rejects. The named fact points back to its evidence:

  Measured (`~/agent-facts-lab`: eight small tasks on a toy Python module, one run each, one model,
  and the same author wrote the expected facts and the detector, so read these as a feasibility
  check, not a benchmark; Claude Code on Sonnet, $1.18 in all): the
  detector over harness diffs recorded every catalogued change fact (precision and recall 1.00 on
  seven tasks, 0.88/0.75 on the mixed one), including edits the agent made with `sed`. Haiku
  classifying session windows alone was much worse (recall 0 to 0.75 by function; 13 of 14 change
  facts by file, 1 of 8 work facts): an Edit window has no line context, so the function is lost, and
  one window cannot show "failed before". Once it named a fact the catalogue lacked
  (`parameter_renamed`, from the agent's own words).

  A real classification model does much better on the change layer (strategy G: TypeSafe's Jev, a
  hosted classifier that answers multiple-choice questions with probabilities and writes no text,
  over 154 windows labelled by the detector): 0.94 accuracy on the windows that change code, 98%
  right among the 86% of answers given at 0.95 confidence or more, the same answer on every rerun,
  67 ms and $0.00004 a call (Haiku: 2.5 s and seven times the price). It names the type only; which
  function, and the changeset, still come from the detector. On intent it fails like everything
  else: "does this window fix a failing test?" is about 0.6, because one window cannot show the
  failure before. A small local classifier (sentence embeddings and logistic regression) did not
  work on so few examples (0.43); it would need on the order of a hundred labelled windows per fact
  type, which the detector can label for free as work accumulates.

  So: the detector for change facts where a syntax tree exists; a real classifier for the change
  type where none does (other languages, shell output, non-code work) and as a cross-check, with
  0.95 as its threshold; the checker for intent; agents' own records as optional notes of why.
  Because a classifier like Jev is deterministic, its labels can even be recomputed on any replica,
  which softens rule 1 below for that kind of deriver.

```jsonl
{"type": "classification.recorded", "subject": "demo/src/inventory.py::reserve", "source": "detector/ast@3", "id": "c7…", "caused_by": ["evidence:sha256:41e0…", "evidence:sha256:42b9…"], "data": {"label": "code.function_body_changed", "lines_changed": 2}}
{"type": "work.test_fixed", "subject": "demo/tests/test_inventory.py::test_reserve", "source": "checker", "id": "5d1a…", "caused_by": ["c7…", "evidence:sha256:38aa…"], "data": {"by": "agent/worker-3", "failed_before": true, "passes_after": true, "why": "reserve compared the quantity to on_hand instead of on_hand - reserved"}}
```

Two rules make derived facts safe in a CRDT:

1. **A classification is recorded once and never rerun.** Replicas must apply the same operations,
   and a model is not deterministic, so the label is itself the fact, with the classifier's version
   and confidence. Replicas copy the label; they never reclassify. A better classifier later records
   new classifications that supersede the old ones; it never rewrites them. A classification's id is
   content-addressed, `hash(label, subject, caused_by)`, without the deriver's version: two
   Orchestras, or two versions of the classifier, that agree produce the same id, and union
   deduplicates it; where they disagree, the newer supersedes the older on the same evidence, and
   everything counted from it (the system dynamics model's flows included) is a projection,
   rebuilt. Every other fact carries a globally unique id (a UUID, not a local sequence).
2. **Facts that replay and facts that describe are different.** A `function_renamed` produced by a
   rename tool can be applied again on top of someone else's change, but only if it is stored as an
   *intent* ("rename symbol S to N", re-executed on the new base, with preconditions such as "N is
   free"), not as the change set it made at its base. The lab's one pair (a rename and a change to
   the same function's body) commuted only through a hand-written rename map, an operational
   transform written for that pair; a body that mentions the renamed function breaks it, and the
   text baseline was `git merge-file`, not a structured merge tool (Mergiraf, Spork, jDime), which
   would likely have merged the pair too. And `function_body_changed` holding a whole new body is a
   register write: two agents' body edits become a conflict, worse than git's line merge. Replay
   also depends on semantic tools, which today reformat code a little and lack operations (renaming
   a parameter, removing a test). A `test_fixed` derived from a session describes an edit made as text, and git still
   merges that text. So for code, derived facts mostly describe, and the merge of the code stays in
   git unless a fact replays. The CRDT's own territory is the shared record that is not code:
   items, claims, findings, statuses, the model.

Agents can still write directly when they have a natural tool for it (a decision agent's answer,
a claim), and programs always can. Strictly, what an agent writes is a **command**, an intent, and
the subject's home is the command handler that turns it into a fact or rejects it; merge-time
classification is command handling too. A rejected command changed nothing, so it goes to a command
audit log, not into the item's fold. Derivation is for the work that happens in free form.

## 4. What it looks like

The AI writing an integration writes a **model**: the item's fields, each with a CRDT type from a
short menu, and the named facts that change it. The same catalogue is the classifier's list of
answers. Orchestra derives the fold, the conflict table and the merge from the types; nobody
writes merge code. `ops()` below lives in the model file, not in the stored events: the events say
what happened, and the model says what that means for the item, so changing a field's type is
rebuilding a projection, not migrating events. Underneath, every
type is the same thing: a deterministic fold over the set of facts on a subject, in the order of
(hybrid logical clock, id). The clock is stamped once, by the first Orchestra to ingest the fact,
and never restamped; it respects causality (a fact always sorts after the facts in its `deps`),
which producers' wall clocks do not.

```python
from orchestra_model import Entity, Fact, Owned, Register, Set, Map, Counter, Log, Choice

class Incident(Entity):                         # kind: incident, subject: incident/2026-11-05-payments
    roles:     Map[str, Owned[str, "commander"]]                     # commander, ops, comms, scribe
    severity:  Owned[Literal["SEV3", "SEV2", "SEV1"], "commander"]   # one value at every moment; others propose
    status:    Owned[Literal["investigating", "mitigated", "resolved"], "commander"]
    responders: Set[str]                        # add and remove commute; a concurrent add beats a remove
    timeline:  Log[TimelineEntry]               # append-only, by the evidence's wall-clock time, correctable
    findings:  Map[str, Finding]                # keyed parts: each agent writes its own keys
    pages_sent: Counter                         # concurrent increments add up
    draft_update: Owned[str, "comms"]           # the status update; flagged if a fact landed after its deps

class Finding(Entity):
    text: Register[str]
    status: Register[Literal["hypothesis", "supported", "refuted"]]   # a derived fact never sets it above hypothesis
    evidence_kind: Register[Literal["correlation", "test"]]
    evidence: Set[str]
    verified_by: Register[str | None]

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
| `Owned` | the owning role's write wins and closes any open conflict; anyone else's write is a proposal shown to the owner | severity, status, the commander, the public status update |
| `Register` | latest by clock wins; but a write whose `deps` don't include the current value becomes a conflict (the stale-write guard) | a finding's text, last seen |
| `Choice` | all causally concurrent values kept, flagged until a fact that has seen all of them picks; every conflict has an owner and a default (the owner's last write) | decisions with no natural owner |
| `Set` | each add is tagged by its fact id; a remove names the tags it saw, so a concurrent add survives | responders, labels, reviewers |
| `Map` | per key, each value by its own type | findings, comments, sections of a document |
| `Log` | append-only, ordered by time | timeline, notes, observations |
| `Counter` | increments add | pages sent, retries, rounds |
| `Text` (later, only if needed) | character-level merge | a shared prose document |

Authority comes before merge. In incident and ticket work, the fields that matter most (severity,
status, who leads, what is said publicly) have an owner, and an outage needs one value for each at
every moment: a severity with two values open is worse than a wrong one. So those fields are
`Owned`: the role decides, agents propose. Merge types are for what has no owner (findings,
responders, the timeline, counts). A finding that merely correlates with a deploy ("errors began at
the 10:02 deploy") is the classic anchoring error, so findings carry their status and the kind of
their evidence, and the commander sees correlation-only findings for what they are.

Code facts need the same table, pair by pair:

| | rename S | add parameter to S | change S's body | add call to S |
|---|---|---|---|---|
| **rename S** | conflict if the names differ | transform (apply to the new name) | transform; conflict if the body mentions S | transform: rewrite the call, or conflict if it was written after with the old name |
| **add parameter to S** | | conflict unless identical | commutes, unless the body uses the parameter | transform: the call needs the argument, or uses the default |
| **change S's body** | | | conflict (a register) | commutes |

Each new code fact type needs a transform against every other: the table grows as the square of
the catalogue, and operational transformation's history (published transforms later shown to
break its convergence property TP2, as Imine and others found) says each cell needs a proof or a
property test (random pairs and triples on real repositories, applied in every order, comparing
trees and test results).

The stale-write guard matters most for fields agents write. Plain last-writer-wins lets an agent
four minutes stale, writing `status: investigating` with a fresh clock, beat `resolved`; with the
guard, that write didn't see `resolved`, so it becomes a conflict to resolve instead. Fields only
programs write (telemetry, last seen) can stay plain last-writer-wins.

Facts are the only way to change an item, whoever produces them: a program printing them, a
decision agent answering, or the detector, checker and classifier reading an agent's session. Each
carries `deps`: the heads of the subject it had read, as fact ids. A scalar version number would not
do: each Orchestra's "version 41" is a different thing, and "concurrent" has to mean causally
concurrent, which only the facts read can say.

```jsonl
{"type": "incident.severity_raised", "subject": "incident/2026-11-05-payments", "id": "0b6e…", "deps": ["9a1f…"], "data": {"to": "SEV1"}}
{"type": "incident.finding_added",   "subject": "incident/2026-11-05-payments", "id": "4c2d…", "deps": ["9a1f…"], "data": {"id": "f3", "text": "errors began at the 10:02 deploy", "status": "hypothesis", "evidence_kind": "correlation", "evidence": ["deploy d881", "error_rate 0.4% -> 6%"]}}
```

For a derived fact, `deps` come from the session log, not from the agent: what it read is in its
Read results.

## 5. When a fact happens: branches

A fact belongs to the stream it happened on. An agent working on a branch produces facts about that
branch ("removed `legacy.py` on `fix-17`"); they are true, and they are kept, but they never change
main's items. Main's items fold only main's facts.

The merge is the fact that carries work across: at merge time, the net change that actually landed
(main's before and after, so a rebase or a conflict resolution is accounted for, and a file deleted
and restored three turns later is not) is classified once, and those facts are applied to main,
each caused by the merge. These are two different models, and the document should not pretend
otherwise: a CRDT merge is a union of operations, while classifying the net diff throws the
operations away and re-derives them from state. Where they meet, the branch's own facts from
semantic tools are authoritative (a recorded `rename_symbol` is ground truth; the detector is a
heuristic), so the merge also runs an **explanation check**: replay the branch's facts on the merge
base, diff the result against the merged tree, and record what is left as `unexplained`, credited to
whoever resolved the merge, not to the branch. That keeps a conflict resolution from being credited
to the agent whose branch it was, and it catches the detector's mistakes. A branch that is never merged stays a record of an attempt: useful for
calibration (how often an agent's work lands), for spotting wasted work, and for telling the next
agent what was tried. So there are two moments to classify: cheaply, per window on the branch, as a
running account others can watch; and once, at merge, for the facts that count.

### One agent, several items

An agent's run produces one item for certain, its session, and facts on any number of others. That
is normal: any replica writes many keys. What CRDTs don't give by default is **atomicity across
items**: another replica may see "ticket A closed" before "ticket B opened as its replacement",
because each item converges on its own. Two standard answers exist, and Orchestra's shape already
takes the first:

- **One document, many objects.** In Automerge and Yjs one document holds many objects, and one
  change may touch several of them atomically. Orchestra is one log with items as projections by
  subject; the facts of one turn can form a transaction, applied together or not at all. A shared
  id is not enough: the transaction must list its members, and a receiver holds them back until it
  has every one.
- **Transactions over separate CRDTs.** AntidoteDB (Cure) gives transactions across CRDT objects
  with atomic visibility and causal snapshots, without coordination ("causal consistency with
  atomic visibility"). It makes writes visible together; it does not validate them together, so it
  does not rescue rules like `requires` (section 6).

Event sourcing adds the nuance that one event should not belong to two aggregates: so one fact per
item, linked by a shared cause. The session is the evidence; every fact on another item has
`caused_by` pointing into it. The session is the source of those changes, not their container.

### Forks and merges

A CRDT's history is a DAG of changes, like git's, and a version is a set of heads:

- **Forking with history is cheap.** Copy the changes; Automerge's `fork()` does it, and can fork
  from a past version (the document as it was at given heads). Yjs has no fork as such: with
  garbage collection off, a snapshot can be restored into a new document.
- **Merging a fork back is a union of facts** and never stops on a conflict: concurrent writes to
  one value are kept side by side for something to resolve. The union is of facts, not of what is
  accepted: the home still re-decides and admission still rejects (sections 2 and 6).
- **Merging two CRDTs that never shared history** works mechanically (the merge is defined on any
  two states), with one catch: **independent creation**. If two documents each created "the items
  list", the merge sees two competing objects under one key, and one wins. The cure is naming
  things by stable ids, which Orchestra's subjects already are (`owner/repo/issues/42`): two
  Orchestras that each saw the issue merge their facts onto the same item.

**Code subjects are names, not identities.** `inventory.py::available` stops existing when it is
renamed, which is the independent-creation problem again, inside code. Code facts need a
symbol-lineage id (rename and move facts are edges between identities; Unison's content-addressed
definitions are the extreme version), and they are keyed by tree hashes (or `git patch-id`), not
commit ids, so they survive rebase and squash.

For code conflicts the right model already exists: Pijul (after Mimram and Di Giusto's patch
theory) records a conflict as a state, which a later patch depending on both sides resolves;
`Choice` is the same idea for records. Darcs' `replace` patch is the original replayable rename,
and its weaknesses carry over (lexical, not scope-aware, and a precondition that the new name is
absent), as do Darcs 2's exponential merges when conflicts of conflicts accumulate.

The closest prior work is Ink & Switch's: Upwelling (drafts for writers over a CRDT, merged when
ready) and Patchwork (branches, diffs, history and merges for any Automerge document). Their main
finding is the same as section 3's: merging the data is solved; showing what changed, and deciding
when a branch's changes count, is the hard part. That is merge-time classification.

| CRDT idea | Orchestra |
|---|---|
| a replica's change log | an agent's session log, in the evidence store |
| fork at some heads | the agent reads an item; its Read results record the heads it forked from (`deps`) |
| working on a fork | the agent working on a branch, its facts on the branch's stream |
| merge, keeping conflicts (a union of operations) | the branch merged: its net change re-derived from state, checked against the branch's facts |
| an abandoned fork | an unmerged branch: an attempt, kept, never on main |
| atomic visibility | a turn's facts sharing a transaction id |
| merging unrelated documents | two Orchestras syncing, safe because subjects are stable ids |

So an agent run is a fork of the items it touches plus one new item, its session: the harness is a
machine that produces the session item outright, and proposes changes to other items through a
fork that reaches main only by a merge.

## 6. Each coordination problem, solved

**Stale context (1).** An agent's loop does not read anything mid-turn, and models anchor on what
they read first, so pushing every new fact at a running agent is noise at best. Two things work
instead:

- **The home rejects stale commands at submit.** The agent's `deps` say what it had read; if the
  subject moved on in a way the conflict table says matters, the command is re-decided or sent back.
- **A relevance-gated notice, at the agent's next write or submit.** A hook intersects what the agent
  read and edited (from its session log and the detector's facts) with the facts that landed since
  its `deps`, and a cheap yes/no classifier asks "does this invalidate the plan?" Only a yes
  interrupts, and only at that boundary.

`deps` are sound only if the instrumented read is the only way in: an agent that also reads the
item with `gh` or `curl` has read things no `deps` record.

**Lost updates (2).** There are no whole-object writes, only operations on typed fields. Two
agents' operations on different fields, different keys, a set or a log never overwrite each other.
On a single value, `Register` keeps the latest and `Choice` keeps both and says so.

**Edit conflicts (3).** On items, agents' work lands as operations addressed to structure ("set
`findings.f3`", "append to `timeline`") against the version read, not as quoted text; nothing has
to match. A shared prose document is a `Map` of sections, so two agents rewriting different sections
never collide, and the same section becomes a `Choice` conflict someone resolves instead of a garbled
merge. On code, replayable facts (renames, added parameters) rebase onto each other structurally;
the rest is git's merge, as now.

**Duplicate work (4).** Claims are taken by the dispatcher before a session starts, not by agents
mid-run: an agent does not stop because a message says it lost, and a thirty-minute session outlives
a lease nobody renews, so the harness enforces both (it suspends or stops the session when the claim
is lost or the lease lapses). Merged as a CRDT (`Map[job, Claim]`, earliest clock wins), a claim
*reduces* duplicate work and cannot prevent it: a CRDT cannot give mutual exclusion,
and a claim with an earlier clock that arrives late (a laptop that was offline) takes the job after
the holder has worked on it. So claims are decided at the job's home: one claim stream per job, the
first append accepted (it expects the stream not to exist yet), and a lease kept by a process
manager whose timeout releases it. That is real exclusion, the standard reservation pattern. Only
offline, with no home to ask, are claims advisory.

A related trap: when one Orchestra syncs another's facts, facts matching its machines' `on:`
triggers would start its agents again. Triggers fire only on facts appended live at that
Orchestra, never on synced or replayed ones, with (machine, triggering fact id) as an idempotency
key.

**Unvalidated writes (5).** "Checked against the state it lands on" sounds right and is wrong:
which state a fact lands on depends on delivery order, so two Orchestras would accept and reject
the same fact and diverge for good. An invariant like "never after resolved" is not monotone, and
by the CALM theorem (and Bailis's I-confluence) it cannot be kept without coordination. So the
check splits in two:

- **Admission, against the fact's own causal past.** `requires` is evaluated on the state the
  writer had read (its `deps`). That is deterministic and final on every replica: `SeverityRaised`
  written by someone who had already seen `resolved` is rejected everywhere, recorded as rejected,
  and the writer is told. A derived fact is also checked against its outcome (the test really
  passes) before it is recorded.
- **People are never rejected.** A command from a person is accepted even when its `deps` are
  stale: the fact is annotated `stale_against: [...]` and the role owner is told. A commander on a
  bridge call must never be told to re-read and resubmit. Rejection is for machines.
- **At the home, against current state.** For a subject whose home is reachable, this is the
  ordinary aggregate check for machines' commands: the home rejects a `SeverityRaised` on a resolved incident outright, as
  an event store would. The two cases below are what remains when facts are written without the
  home, offline or on another Orchestra.
- **Violations, on the merged state.** A `SeverityRaised` concurrent with `resolved` passes
  admission on both sides, and the merged incident breaks the rule. That is not rejected; it is a
  derived view, a flagged violation computed from the merged state, which a later fact resolves
  (reopen, or withdraw). Transitions that must be final at once are accepted only by the home.

The facts and their rules are still a state machine shared by every writer, enforced outside all
of them; it is just honest about which rules hold without coordination.

**Actions with side effects.** A page, a rollback or a failover is not a record and is not undone
by removing facts. It is its own entity, with a lifecycle: proposed → approved (by the owning role)
→ executed → verified (against a metric, with the lag the model expects). Each has an idempotency
key per target (`rollback:payments-api:d881`), decided at the home, so two agents or two people
cannot run the same rollback; destructive actions need a second approver. Claims (4) cover actions
on targets as well as jobs, because the dangerous duplicate is not two agents on one job but two
responders on one failover. And every incident has a kill switch for its agents.

**Attribution and undo (6).** Every fact names its source, and a derived fact names the session
entries it came from, so any change traces to the agent's exact turn and reasoning. Undoing a bad
run targets exactly the facts caused by its session, by id: remove *its* `Set` tags, not the
values; drop its `Log` entries and `Map` keys; and for a `Register`, put back the value it displaced
only if nothing has written since, otherwise raise a conflict. A blind "write the previous value"
would clobber the fleet's later work; undo by id leaves it intact. It undoes the record, not the
world: a pushed branch, a posted comment or a page sent are actions, undone (if at all) through
their own lifecycle above. For a postmortem the record answers the real question, "who knew what,
when, and acted on it": every fact's `deps` say what its writer had seen.

**Context bloat (7).** Mostly this is the codebase an agent reads, which facts do not shrink. What
they do shrink is handing work over: the next agent, or a reviewer, gets the previous attempt's
named facts, their why and the checker's verdicts, not the transcript.

## 7. Why not the alternatives

| | Shared snapshots (prompts 1 and 2) | Locks or a scheduler | Last write wins per field (with a hybrid clock) | Typed CRDT with named facts |
|---|---|---|---|---|
| 50 interleaved writers | overwrite each other | wait for each other | mostly fine; single values silently lost | merge; conflicts on single values kept and shown |
| Agents 4 minutes stale | their writes are wrong | they held a lock for 4 minutes | a stale write with a fresh clock beats newer work | the stale-write guard makes it a conflict; stale commands are rejected at submit |
| Concurrent edits to collections | lost | serialized | the whole collection replaced | sets, maps and logs merge |
| Impossible transitions | become truth | caught only if the lock holder checks | become truth | rejected at admission if the writer saw the state; otherwise flagged |
| Undo one agent's work | restore an old snapshot, losing others' work | same | hard | its facts removed by id |
| Several Orchestras (work and home laptops, a teammate's) | can't sync | needs a shared server | syncs: it is a CRDT too | syncs; the same merge everywhere |

Per-field last-writer-wins with a hybrid clock is itself a CRDT (an LWW map) and syncs between
Orchestras without a server, so syncing is not what sets the typed version apart. What does is the
rest of the table: stale writes made into conflicts instead of wins, collections that merge,
admission against what the writer saw, and undo by id. Session logs and recorded classifications
sync the same way, by union.

## 7b. What the facts are for, beyond merging

Most of the value may be in uses that need no merge at all:

- **Scope fences.** A task names the subjects it may touch; a change fact outside them pauses the
  agent. Deterministic and cheap.
- **Tripwires on the session log.** Cheap classifiers that spot looping, repeated failing edits, or
  "done" claimed with no test run, and stop or redirect the agent. Acted on, not recorded as facts.
- **Fact-assisted rebase.** On a git conflict, invert the agent's replayable facts, merge the text,
  and replay them on the new main, handing back only the remaining hunks. This is RefMerge (Ellis,
  Nadi and Dig, 2022; also MolhadoRef, IntelliMerge), whose evaluation found it removes some
  conflicts and introduces others. Failed preconditions (the name is taken, a concurrent new call
  uses the old name, two renames of one symbol) are conflicts, never silent results, and CI runs on
  the replayed result: equal trees in either order is convergence, not correct code. Measure first
  what share of agent-branch conflicts a structured merge tool leaves that replay would fix; if it is
  small, code facts are for prevention (scope fences, scheduling), not merging.
- **Scheduling.** Predict which symbols a task will touch from similar past tasks' facts, and run
  overlapping tasks one after another instead of in parallel.
- **Review.** The reviewer reads the fact chain (named facts, why, checker verdicts) before the
  diff, and checks the unverified claims first.
- **The incident scribe.** A timeline built from facts that already exist (deploys, alerts, pages,
  actions, findings, severity changes), each entry linked to its evidence, in the evidence's
  wall-clock time.
- **Duplicate investigations.** Intersect running sessions' reads (the same dashboards, log queries,
  hypotheses) and tell the commander "agents 2 and 5 and a person are all on the connection pool".
- **Stale drafts.** Flag the status draft when a mitigation fact has landed since its `deps`, before
  it is published.
- **Judged mitigations.** Before an action, record a prediction ("the rollback brings errors under
  1% within 5 minutes"); judge it against the metrics; a mitigation that does not hold does not
  close the incident. The model's expected recovery with no action is the counterfactual a
  postmortem needs, since postmortems, like regressions, credit the last action taken.
- **Growing the catalogue.** Every `other` and every claim rejected for want of a fact type goes on
  a queue of proposed new facts.

## 8. The objection: "if the AI writes the facts, it's writing the system dynamics model, so why estimate?"

It isn't, but the reason is narrower than "facts give the structure, data gives the numbers". The
model file gives the **stoichiometry**: which transitions exist, and which state each one moves an
item from and to. It also gives part of the **measurement map**: which facts count which
transitions. It does not give the **rate laws**: what each transition's rate depends on (load, the
length of another queue, the error rate, time of day). In system dynamics those dependencies *are*
structure, so estimation is still structure selection over rate laws, plus parameter estimation;
the facts narrow the search, they don't settle it.

| The model file gives | Only data gives (all of it structure, not just numbers) |
|---|---|
| the states an item can be in | how many items are in each state over time |
| the transitions that can happen | each transition's rate law: what it depends on, and how strongly |
| who may cause each transition | how long items stay in each state, and how that stretches under load |
| which facts measure which transition | delays, and feedback (does a long review queue cause sloppier reviews and more reopens?) |
| | the links to the observed system (does a deploy-fixing flow move the error rate, with what lag?) |
| | hidden states nobody declared, where some measured flow depends on them (Orchestra's observability check tests exactly this) |

Two corrections to what this document used to say. The task queue was not "found so well because
its structure was given": its stocks balance its counters exactly, an accounting identity, and no
rate law was identified. The kube simulation failed for recorded, unrelated reasons (loops through
auxiliaries the loop finder could not represent, a step size longer than the autoscaler's settling
time, counter-reading bugs, an inference window that dropped most series). And the linear noise
approximation does not "fit naturally": it needs a Markov jump process with large counts and
unimodal fluctuations, and this system has tens of items, batched arrivals and heavy-tailed
durations.

### Fact counts are measurements of flows, not the flows

Counts of facts are noisy, delayed, biased measurements:

- **Dated by evidence time.** A fact is dated by when the work happened (its evidence), not when it
  merged or was ingested; otherwise a pull request that lands twelve facts at once looks like an
  impulse.
- **Corrected for the deriver's errors.** Each deriver version has a confusion matrix, estimated from
  a random audited sample; counts are corrected for it before they reach the filter. Orchestra has
  seen this fail: discovery once credited an agent that closed items wrongly, because wrong closes
  look like resolutions in the counts.
- **Revisions are revisions.** A superseded classification changes past counts; the filter treats it
  as revised data, and a new deriver version as a known change in the measurement map, not as
  something that happened in the system.
- **Main is a survivor population.** Merged work and branch work are different populations; neither
  stands for the other.

The better model uses what the fact log uniquely has, the item. Each item's facts are an exact,
censored trajectory through its states, so transition **hazards** can be fitted item by item (with
the metrics filter's estimated state as covariates), the aggregate stock-and-flow equations derived
from them, and the result checked against the counts and against the metrics series (data
reconciliation, as in process engineering: when the balances disagree, a deriver is wrong or a flow
is unseen). Counts enter the filter as count observations of the integrated flow (Poisson or
negative binomial), not as Gaussian noise.

### Actions as inputs: identification in closed loop

Agents are dispatched in response to the state (on a surprise, a spike, a failing check), so their
actions are chosen by the disturbance. Regressing outcomes on such actions credits agents with
regression to the mean: one is sent at a spike, the spike decays anyway, and the agent looks
effective. The classifier makes it worse when it names facts from the agent's own words, so the
sensor's errors correlate with the actor. So:

- Never estimate an action's effect by regressing on dispatches triggered by the state.
- Log each dispatch's probability, and randomize a fraction of near-threshold dispatches and their
  timing; the threshold itself then gives a regression-discontinuity design, and the randomization
  an instrument.
- Validate on held-out windows (innovations white, normalized innovation squared in range) and on
  predictions made before randomized interventions.
- Report uncertainty: parameter covariance, and the parameter combinations the data cannot tell
  apart.

The conflict and rejection rates are signals too. Stale-command rejections should run at about
1 − exp(−λ·τ), the rate of others' writes times how stale a session gets: a free consistency check
on session lengths and write rates, and a measure of how coherent the fleet is.

Nobody has to be begged to write flows, or to emit facts: the agents work as they do, programs and
a classifier name what they did, and the corrected counts measure the flows (merged facts for the
system, branch facts for the agents themselves: attempts, abandoned branches, rework).

## 9. Costs, and how to keep them small

- **The model must be written.** One small file per kind of item, by the AI, reviewable by you.
  The type menu is short (seven types) and closed; no custom merge code. The same file is the
  classifier's list of answers.
- **Orchestra must implement the types once**: a few hundred lines for Register, Choice, Set, Map,
  Log and Counter (Text is a library, pycrdt, if ever needed), plus validation of `requires` and the
  `deps` and clock bookkeeping.
- **Facts must be named well, and now that is measurable.** A bad model (a fact called `updated`
  that sets everything) brings back the snapshot problems. A missing name shows up directly: as a
  rate of `other`, and as checked claims rejected for want of a fact (in the lab, a correct
  parameter rename was recorded as a removal plus an addition, and the agent's true claim
  rejected).
- **A wrong classification is a wrong fact that merges cleanly.** Hence a confidence threshold
  (below it, the window stays raw events), a check against an outcome wherever one exists, and the
  raw events kept so facts can be derived again and corrected.
- **Session logs are large.** Keep their structure, truncate big tool results to a hash and a
  pointer, and decide per project what may be ingested at all.
- **Bookkeeping the types need.** Removes leave tombstones (a `Set` must remember removed tags);
  they can be dropped only once every replica has seen them (causal stability), which with laptops
  that go offline may be never, so the log grows. Snapshots of the fold keep reads fast.
- **The model will change.** A field may move from `Register` to `Choice`, or split in two; old
  facts must still fold. Because events never carry merge operations, that is upcasting at read
  time and rebuilding the folds. Only two Orchestras running different model versions at once need
  more (lenses between versions, as in Ink & Switch's Cambria).
- **One file does four jobs.** The model is the event schema, the classifier's labels, the conflict
  table and the system dynamics model's stoichiometry; a change to one moves all four. Version it, and
  treat a change as a migration of each.
- **Session logs and privacy.** They hold everything an agent read. Immutable, replicated logs make
  deletion impossible unless planned for: hence the separate evidence store, one key per session,
  and crypto-shredding.
- **Orchestra is one more thing that can fail in an outage.** An incident's home must not share
  failure domains with the systems it records (the same cloud, database or network); when it is
  unreachable, writes append locally and merge later, and the incident channel stays the record of
  last resort. And responders under stress will not learn seven field types: the page shows them a
  timeline, owners and proposals, never heads or violations.
- **`source` is self-declared.** Within one Orchestra that is fine; facts that cross between
  Orchestras need signatures, or a peer can write as anyone.
- **Language-neutral.** The model exports a JSON Schema per fact, with each field's CRDT type as an
  annotation, so Orchestra (and processors in other languages) work from the schema, not from
  Python.
- **CRDT merge types are for shared writes.** Defining the facts is always worth it, for a source
  you only watch too: they are the vocabulary, the classifier's answers and the model's structure.
  The typed merge matters only where several writers touch the same item.

## 10. What would be built, in order

1. The type menu and generic merge in Orchestra, driven by a model's JSON Schema.
2. A home per subject: appends checked against current heads, re-decided by the conflict table,
   rejections to a command audit log; claim streams with leases. Facts with globally unique (or
   content-addressed) ids, `deps` and a hybrid clock.
3. Session logs in an evidence store (per-session keys); harness diffs per tool call; a syntax-tree detector
   for change facts, a checker for work facts, and a classifier for the why and the gaps, each fact
   with `caused_by`.
4. Relevance-gated notices to running agents at their next write or submit; deltas on stdin for continuous programs, which can use them.
5. Branch streams, and classification of the net change at merge.
6. Claims with leases, as a `Map` type, for work allocation without a scheduler.
7. The setup brief asks for the model first, then the collector and processors that import it.
8. Later: syncing between Orchestras (CRDT merge, admission against the causal past, violations as
   a view, triggers only on live appends); undo by id; replayable code facts; `Text`.

Before any of it: measure how often several writers touch one non-code record at once, on real
incident channels and tickets (an operator's guess: near zero, apart from severity, status and the
update draft, which authority settles, not merge). Steps 1 to
4 are the coordination core; section 7b's uses need only step 3 and can come first; the rest follows
when a real case asks for it.
