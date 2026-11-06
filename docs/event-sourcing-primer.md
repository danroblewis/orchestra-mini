# Event sourcing: a primer for the practitioner's vocabulary

For an SRE who has met the term, read some of it, and wants what people who have built these
systems take for granted: the definitions, the standard arguments, the standard mistakes, and the
words they use without explaining. Sources are listed at the end; quotes are attributed where the
wording matters.

Contents:

1. The core idea
2. Events, commands, queries
3. The event store, streams and expected-version concurrency
4. Aggregates and consistency boundaries
5. Work across aggregates: process managers and sagas
6. Rehydration, folding and snapshots
7. Projections, read models and CQRS
8. Fowler's four event-driven patterns
9. Designing events
10. Schema evolution, and what to do when an event is wrong
11. Delivery: idempotency, ordering, the outbox
12. Privacy and the right to be forgotten
13. Time: temporal queries, audit, replay
14. When not to use it, and how it goes wrong
15. Relatives: CRDTs and git
16. Common mistakes
17. Glossary
18. What's on the test
19. Mapping to Orchestra
20. Sources

---

## 1. The core idea

Most systems store **current state** and overwrite it: `UPDATE accounts SET balance = 110`. The
history of how the balance got there is lost, or kept on the side in an audit table that may or may
not agree with the real data.

**Event sourcing** stores the **changes** instead, as an append-only sequence of events, and treats
that sequence as the source of truth. Current state is computed from it.

```jsonl
{"type": "AccountOpened",  "stream": "account-42", "version": 1, "data": {"owner": "c-7", "currency": "EUR"}}
{"type": "MoneyDeposited", "stream": "account-42", "version": 2, "data": {"amount": 100}}
{"type": "MoneyWithdrawn", "stream": "account-42", "version": 3, "data": {"amount": 30}}
{"type": "MoneyDeposited", "stream": "account-42", "version": 4, "data": {"amount": 40}}
```

Balance is 110, but no row says 110. It is a **left fold** over the events:
`state = events.reduce(apply, initial)`. Greg Young's short form, widely quoted: *current state is
a left fold of previous behaviours*.

The definitions everyone agrees on:

- **"Capture all changes to an application state as a sequence of events"** (Fowler, 2005). The
  log is not a by-product; it is the record.
- **The events are the source of truth; any stored state is a cache.** Delete every table and
  rebuild them from the log, and you have lost nothing. If you could not do that, it is not event
  sourcing (it is a database with an event log attached).
- **Events are immutable facts in the past tense.** Something happened; it cannot un-happen. You
  correct the record by adding a new fact, as an accountant does ("accountants don't use
  erasers", a line Young uses constantly).
- **State is derived by replay, and different consumers may derive different states from the same
  events.** A balance, a monthly statement and a fraud score are three folds over one log.

What it is *not*: a message bus, a Kafka topic, CQRS, microservices, or asynchrony. All of those
travel with it in talks, and all are separate decisions (section 8).

## 2. Events, commands, queries

Three message kinds, and practitioners are strict about them:

| | Command | Event | Query |
|---|---|---|---|
| Means | "please do X" | "X happened" | "tell me Y" |
| Tense | imperative: `WithdrawMoney` | past: `MoneyWithdrawn` | question: `GetBalance` |
| Can be rejected? | yes, that's the point | no, it already happened | n/a |
| Receivers | exactly one handler | zero to many subscribers | one |
| Changes state? | maybe, by producing events | it *is* the change | never |

A command is validated against current state; if it passes the business rules, the handler emits
one or more events; the events are stored; nothing else is. The command itself is usually not
stored (some systems keep it for audit, but it is not the truth: a rejected command changed
nothing).

**Naming.** Events are named in the past tense in the language of the business (DDD's
"ubiquitous language"): `OrderPlaced`, `PaymentDeclined`, `ShipmentDispatched`, `CustomerRelocated`.
Not `OrderUpdated`, not `StatusChanged`, not `SetAddress`. The test: would a domain expert say this
sentence about their work? "The customer moved" is a fact a business cares about; "the address
field changed" is a database operation.

**Passive-aggressive commands** (Fowler's phrase): an event that is really an order to one
particular receiver, which must act on it and whose sender is waiting for the result
(`InvoiceNeedsSending`). If the sender cares what happens, it is a command; model it as one.

## 3. The event store, streams and expected-version concurrency

An **event store** is a database specialised for this: append events, read them back in order.
The operations that matter:

- `append(stream, events, expected_version)`
- `read(stream, from_version)` (forward, sometimes backward)
- `read_all(from_position)` and **subscriptions** (catch-up or live) over everything

A **stream** is an ordered sequence of events with an id, usually one per entity or aggregate
(`account-42`, `order-9001`). Each event in a stream has a **stream version** (also "revision",
"sequence number"), 0 or 1 based, gapless and strictly increasing. Across the whole store there is
usually also a **global position** (EventStoreDB/KurrentDB calls the all-stream `$all`). A
**category** is the set of streams sharing a prefix (`account-*`), and projections often subscribe
by category.

Event stores are built to hold millions of small streams cheaply. That is the first practical
difference from Kafka (section 11): a topic or partition per entity does not scale that way.

### Expected-version (optimistic) concurrency

The one concurrency mechanism an event-sourced write model needs:

1. Load `account-42`: four events, version 4. Balance 110.
2. Decide: `WithdrawMoney(100)` is allowed. Produce `MoneyWithdrawn(100)`.
3. Append with `expected_version = 4`.

If someone else appended event 5 in between (another withdrawal of 50), the store **rejects** the
append (`WrongExpectedVersion` in EventStoreDB, a concurrency exception in Marten and Axon). The
handler reloads (now version 5, balance 60), re-runs the decision (now rejected: insufficient
funds), and either retries or reports. No locks are held while deciding; conflicts are detected at
write time, per stream.

Common expected-version values: a specific number; "no stream" (the stream must not exist yet: use
it for creation, so two `AccountOpened` cannot race); "stream exists"; and "any" (no check, which
means you are opting out of the invariant protection and had better know why).

This is the thing that makes the decision correct: the events were derived from exactly the state
they are being appended to. Without it, two writers each see version 4, each decide, both append,
and the invariant (no overdraft) is broken. That lost-update race is the classic bug of event
sourcing on a log without per-stream conditional writes.

Some stores also make appends **idempotent by event id**: retrying the same append with the same
event ids after a timeout does not duplicate them.

## 4. Aggregates and consistency boundaries

From Domain-Driven Design (Evans, then Vernon's *Implementing Domain-Driven Design* and his
"Effective Aggregate Design" papers). An **aggregate** is a cluster of objects treated as one unit
for changes, with one **aggregate root** that is the only entry point. Its job is to protect
**invariants**: rules that must hold *immediately and always* ("balance never negative", "an order
has at most one shipping address", "seats sold ≤ capacity").

The aggregate is the **consistency boundary**: inside, rules hold transactionally; outside, things
are eventually consistent. In an event-sourced system that becomes concrete:

- **One stream per aggregate.** The aggregate's state is the fold of its stream; its invariants are
  checked against that state; its new events are appended with that stream's expected version.
  The stream is the unit of ordering and of concurrency control.
- **One command changes one aggregate, in one transaction.** Vernon's rules of thumb:
  1. Model true invariants in consistency boundaries.
  2. Design small aggregates.
  3. Reference other aggregates by identity only (an `order` holds `customer_id`, not a customer).
  4. Use eventual consistency outside the boundary.

**Why one event shouldn't change two aggregates.** If `TransferMade` lived in both `account-1` and
`account-2`, you would need an atomic append to two streams (which most stores don't offer, and
which kills scaling because the two are now coupled), and the event would be versioned and
folded by two models that will evolve separately. The standard shape is two facts in two streams:
`MoneyWithdrawn` on `account-1` (with a `transfer_id`), then `MoneyDeposited` on `account-2`,
coordinated by a process (section 5). Between the two, money is "in flight", and the business
usually already has a name for that state.

**Sizing.** Too big (an aggregate for "the whole catalogue") and every write conflicts with every
other write and streams grow without bound. Too small and invariants that span them can't be
enforced. The question to ask is: *what must be true at the instant a command is accepted, and what
can be true a second later?* Only the first belongs inside one aggregate. Business people tolerate
much more eventual consistency than engineers assume (airlines overbook; banks reconcile at night).

A modern functional framing (Jérémie Chassaing's **Decider**) strips aggregates to three functions:
`initialState`, `evolve(state, event) -> state` (the fold step), and
`decide(command, state) -> events` (pure: no I/O). Most experienced people now think of aggregates
this way even in object-oriented code.

## 5. Work across aggregates: process managers and sagas

Anything that spans aggregates, or spans time, is a **process**: "when the order is placed, reserve
stock, then take payment; if payment fails, release the stock; if nothing happens in 30 minutes,
cancel." Two patterns, whose names are used loosely:

- **Process manager** (from *Enterprise Integration Patterns*): a stateful component, itself often
  event sourced, that listens for events, keeps track of where the process is, and **sends
  commands** to aggregates. It holds the workflow's state and decides the next step.
- **Saga** (Garcia-Molina and Salem, 1987, from databases): a long-running transaction broken into
  local steps, each with a **compensating action** to undo it on failure. In the CQRS community
  "saga" has come to mean almost any process manager; purists keep the distinction (saga = sequence
  plus compensations, often without central state; process manager = central state machine).

**Choreography vs orchestration.** Choreography: each service reacts to others' events, with no
coordinator; easy to start, hard to see ("where is the order workflow defined?" "nowhere"). Young,
in 2016, named the absence of process managers as a major problem of real systems: with services
subscribed directly to each other's events, nobody can say what the system does. Orchestration:
a process manager owns the flow explicitly.

Process managers need care around time (**timeouts** are scheduled messages, or "passage of time"
events like `DayEnded`, Verraes's pattern), idempotency (they receive duplicates), and versioning
(a process in flight when you deploy a new version must still finish: Young's book has a chapter on
"take over" strategies).

## 6. Rehydration, folding and snapshots

**Rehydration** (also "hydration", "loading the aggregate"): read the stream, fold its events into
state, then decide. The fold step is usually a method per event type, `apply(MoneyDeposited)`, or
`when`/`evolve`. Two rules for it:

- **`apply` must not fail and must not decide.** The decision was made when the event was written;
  folding just records its consequence. Validation lives in the command handler. An `apply` that
  throws on old data means you can no longer load that aggregate.
- **`apply` must be deterministic and side-effect free.** No clock reads, no calls out, no emails.
  Replay runs it again, possibly thousands of times.

**Snapshots** store the folded state at version N, so loading is "read snapshot, then fold events
after N". They are a **performance optimisation for the write model**, never a source of truth: you
can always delete them and rebuild. The mainstream view, which Oskar Dudycz states plainly:
snapshots are a **last resort**, and needing them often means the stream is modelled too long.
Folding a few hundred events is microseconds of CPU; the cost is I/O, and stores read short streams
fast.

The alternative is to keep streams short by modelling lifecycles the business already has:
**closing the books**. A cashier's shift, a billing period, an accounting year: at the end, write a
summary event (`ShiftEnded` with the totals), and start a new stream carrying the opening balance.
That is a real domain fact, not a technical snapshot, and it also bounds versioning pain (old
streams that are closed never need to be upcast for new logic).

Snapshot trade-offs when you do use them: they must be versioned and invalidated whenever the fold
logic changes (a snapshot is the output of old code); they tempt people to query them as read
models (don't); strategies are every N events, on a particular event type, or on a schedule.

## 7. Projections, read models and CQRS

The event stream is excellent for one question ("what is the state of aggregate X?") and terrible
for all others ("which accounts are overdrawn?", "orders by customer this month"). So you build
**read models** (also "views", "query models", "materialized views"): tables, documents or search
indexes shaped for each question. The code that builds them from events is a **projection**
(or "projector", "denormalizer", "event handler"). Vocabulary varies: Marten calls the folded
document an "aggregate projection"; EventStoreDB has server-side "projections" written in
JavaScript; Axon has "event processors".

A projection subscribes to events (from one category, or `$all`), updates its store, and
**checkpoints** the position it has processed, so it can resume after restarts.

```jsonl
{"type": "OrderPlaced",   "stream": "order-9001", "data": {"customer": "c-7", "total": 54.00}}
{"type": "OrderShipped",  "stream": "order-9001", "data": {"carrier": "DHL"}}
```

A "customer's open orders" projection inserts a row on `OrderPlaced` and deletes it on
`OrderShipped`. A "revenue per day" projection adds `total` to a daily bucket. Same events, two
read models, neither aware of the other.

Key properties:

- **Read models are disposable.** Rebuild one by resetting its checkpoint and **replaying** from the
  start. That is how you fix a projection bug, add a new view over past data ("we can answer
  questions we didn't know we'd ask"), or change a schema: build the new one alongside (a
  **blue/green projection**), switch reads over, drop the old.
- **Read models are eventually consistent** when updated asynchronously: a user who just placed an
  order may not see it in "my orders" for a few hundred milliseconds. Standard responses: return
  the new state or version from the command so the UI can show it; have the UI wait until the
  projection's checkpoint passes the written position ("read your writes" by version); or update
  that projection **inline**, in the same transaction as the append (Marten offers inline, async
  and live projections; "live" folds on demand and stores nothing).
- **Projections must be idempotent** or exactly-once by checkpointing in the same transaction as
  the view update, because they will see events again after crashes and during rebuilds.
- **Command handlers must not read from projections to make decisions.** The read model is stale;
  decisions use the aggregate's own stream (Udi Dahan is emphatic on this).

### CQRS, and why it is a separate idea

**CQRS** (Command Query Responsibility Segregation, Greg Young, about 2010, building on Bertrand
Meyer's CQS) means: use **different models for writing and reading**. That is all. You can do
CQRS with one relational database (a normalised write schema and a few denormalised read views)
and no events at all; Fowler: CQRS "isn't really about events".

The relationship is one-directional. **Event sourcing practically requires CQRS** (the event log is
a write model you cannot query, so you need separate read models), but **CQRS does not require
event sourcing**. Young's 2010 post listing what CQRS is not includes "nor is it using event
sourcing"; he called it a stepping stone toward event sourcing, not an end.

Udi Dahan's framing adds the "why": CQRS is for **collaborative** domains, where many actors change
the same data and whatever a user sees is already stale; commands should capture **intent**
(`MakeCustomerPreferred`) rather than diffs of a form; and a valid command can still be rejected by
business rules. For single-user, form-over-data parts of a system, plain CRUD is better.

## 8. Fowler's four event-driven patterns

Martin Fowler's 2017 "What do you mean by 'Event-Driven'?" exists because people use "event-driven"
for four different things and blame one for the costs of another.

| Pattern | What it is | Gains | Costs |
|---|---|---|---|
| **Event notification** | "something happened" with little data (often an id); receivers call back if they need more | low coupling; sender doesn't know receivers | the overall flow is implicit ("hard to see... not explicit in any program text"); callbacks add load |
| **Event-carried state transfer** | the event carries the data receivers need, so they keep a local copy and never call back | resilience (works when source is down), lower latency, less load on source | duplicated data, eventual consistency, receivers must maintain state |
| **Event sourcing** | the events are the system of record; state is derived | audit, history, replay, temporal queries, alternative histories | replay and schema evolution are hard; external interactions during replay |
| **CQRS** | separate read and write models | each model fit for its job; scale reads independently | two models to keep in sync; added complexity |

Why they get confused: they co-occur (an event-sourced service often publishes notifications and
uses CQRS), they share the word "event", and vendors sell all four under one name. The practical
consequence: Kafka-based **event streaming** between services (notification or state transfer) is
not event sourcing, even if events are retained forever; event sourcing is a choice *inside* one
service about how it stores its own state, and doesn't have to be asynchronous at all.

## 9. Designing events

### Fat vs thin

- **Thin** (notification): `{"type": "OrderPlaced", "data": {"order_id": "9001"}}`. Consumers fetch
  details, coupling them to the source's API and to timing (by the time they fetch, it may have
  changed).
- **Fat** (state transfer): the event carries what consumers need. Fewer calls, more coupling to the
  event's shape, larger payloads.

Inside an event-sourced aggregate, events carry **what the fold needs and what the business would
want to know later**, including derived values that were true at decision time (the exchange rate
used, the price computed), because recomputing them later with new code or new data gives a
different answer.

### CRUD events: the anti-pattern

```jsonl
{"type": "CustomerUpdated", "data": {"name": "Ana Silva", "address": "...", "email": "...", "tier": "gold"}}
```

This is **CRUD sourcing** ("property sourcing"): an event per row update, carrying a changeset or
the whole record. It loses the reason (did she move, or was a typo fixed? Was she upgraded, or did a
migration set the tier?), forces every consumer to diff to find out what happened, bloats streams,
and gives you all the cost of event sourcing and none of the benefit. Compare:

```jsonl
{"type": "CustomerRelocated",  "data": {"customer": "c-7", "new_address": {"...": "..."}}}
{"type": "CustomerTierUpgraded", "data": {"customer": "c-7", "from": "silver", "to": "gold", "reason": "annual_spend"}}
```

Events **stating facts** say what happened in business terms; events **carrying changesets** say
which fields changed. The second is sometimes right (a genuinely data-entry-shaped part of a
domain, or integrating with a CRUD source you don't control), but if all your events look like it,
you probably wanted a database with an audit log.

Young's related advice: **avoid "And"** in event names (`OrderPlacedAndPaid` is two facts), and
don't make one event mean different things depending on its fields.

### Domain events vs integration events; internal vs external

- **Domain events** (internal, private): fine-grained, in the service's own language, shaped by its
  aggregates, free to change with refactoring. They are the event-sourced service's storage format.
- **Integration events** (external, public): a deliberate, stable, versioned **contract** for other
  services, usually coarser and enriched (Verraes's **summary event**: one `OrderCompleted` with
  everything the outside needs instead of the twenty internal steps).

Publishing your internal event stream directly is the most common architectural mistake in the
area (Chris Kiehl's critique and Dudycz both make it): every consumer couples to your storage
schema, you can't refactor without a cross-team negotiation, and nobody can reason about your
service as a black box. Translate internal events into public ones at the boundary (often via the
outbox, section 11). Young's book has a chapter on this: internal and external models change at
different rates and granularities.

### Envelope and metadata

Practitioners separate the **payload** (`data`) from **metadata**: event id (for dedup), type,
schema version, timestamp, stream and version, and **correlation id** (the whole conversation) and
**causation id** (the message that directly caused this one). Correlation and causation ids are how
you trace a workflow through process managers; CloudEvents standardises a similar envelope for
interop.

## 10. Schema evolution, and what to do when an event is wrong

### Why events are immutable

Events have already been consumed: projections built from them, other services acted on them,
auditors read them. Rewriting one silently invalidates all of that, and destroys the audit
property that justified the design. Young's book opens on this ("Why can't I update an event?"),
with WORM storage and fraud as part of the answer. Note the subtlety practitioners make: **the
facts are immutable; their stored representation may sometimes be migrated** (copy-transform,
below), as a deliberate, logged operation.

### Versioning techniques (roughly from least to most invasive)

- **Additive change with a tolerant reader / weak schema.** Add optional fields; consumers ignore
  unknown fields and default missing ones. Map by field name rather than by strict type. Young's
  book recommends this as the default: most changes need nothing more.
- **New event type** (`OrderPlaced_v2`, or a better name). Old events keep their type; code handles
  both. Sometimes **double write** both versions during a transition for consumers.
- **Upcasting.** Transform old events into the current shape **at read time**, before the fold or
  projection sees them: `v1 -> v2 -> v3`, one small function per step (Axon's upcaster chain;
  Marten has the same). The stored events are untouched. Upcasters must be pure and cheap, since
  they run on every read.
- **Copy-and-replace / copy-transform.** Write a new stream (or a new store) by reading the old one
  and transforming, then switch over. Used when stream boundaries were wrong (one aggregate should
  have been two), or for "versioning bankruptcy" when the accumulated upcasters become unbearable.
  Heavy, but honest: the old store can be kept for audit.
- **Semantic changes are the real danger.** A field whose *meaning* changes (amount now includes
  tax) is not a schema change your serializer will notice. Give it a new name or a new event.

Rule of thumb: you will get event design wrong; plan for versioning from day one (version in the
metadata, tolerant readers), and keep streams short so old shapes die out.

### When an event is wrong: compensating events

The business or the code recorded something false: a deposit of 1000 that was meant to be 100.
You don't edit it; you append a correcting fact, as an accountant writes a reversing entry:

```jsonl
{"type": "MoneyDeposited",       "stream": "account-42", "version": 5, "data": {"amount": 1000, "ref": "dep-881"}}
{"type": "DepositCorrected",     "stream": "account-42", "version": 6, "data": {"ref": "dep-881", "amount": -900, "reason": "keying error"}}
```

Better names make the correction a domain concept (`DepositReversed`, `ChargeRefunded`). The
history now says what actually happened, including the mistake, which is what an auditor wants.
Downstream consumers see the correction like any other event. When a *bug* wrote nonsense events
at scale, the escape hatches are compensating events generated by a script, or copy-transform;
Young's chapter on this is titled "Whoops, I Did It Again", with a section "But I Really Screwed
Up".

## 11. Delivery: idempotency, ordering, the outbox

These are messaging concerns that event sourcing inherits whenever events leave the store.

**At-least-once is the norm.** Exactly-once delivery across a network does not exist; what exists is
at-least-once delivery plus **idempotent processing** ("effectively once"). Every subscriber,
projection and process manager will see some events twice (after a crash before checkpointing, on
reconnect, on rebuild).

**Deduplication** techniques: keep the processed event ids (or the last processed stream version
per stream, which is cheaper and works because streams are ordered); store the checkpoint in the
same transaction as the side effect; make the effect naturally idempotent (`SET` rather than
`INCREMENT`, upserts keyed by event id).

**Ordering guarantees.** The guarantee you can rely on is **within one stream**: events in
`account-42` are totally ordered by version. Across streams, a store may offer a **global order**
(a commit position over `$all`) for subscriptions, but consumers should not depend on causal order
between unrelated aggregates unless the store promises it, and once events cross a broker they
are typically ordered only per partition (Kafka: per partition, so per key if you key by stream id).
Projections that join two streams must tolerate one arriving before the other.

**Kafka is not an event store** (the standard discussion): Kafka lacks conditional appends per
entity (no expected-version check, so no lost-update protection for aggregates), cannot cheaply
read "all events of entity X" (you'd scan a partition, or need a topic per entity, which doesn't
scale to millions), and its retention and compaction model is for moving data, not keeping the
record. People build event sourcing on it with a single writer per key or external locking, and
many regret it. The usual architecture: an event store (or a Postgres table) for the write side,
and Kafka for distributing integration events to other services.

**The transactional outbox.** The **dual-write problem**: a service that writes its database and
then publishes to a broker can crash between the two, so one happens without the other. The fix:
write the outgoing messages to an **outbox table in the same transaction** as the state change,
and have a **relay** (polling, or change-data-capture tailing the transaction log) publish them.
Messages go out if and only if the transaction committed, at least once, so consumers dedup. In a
pure event-sourced system the event store *is* the outbox: subscribers read committed events
directly. The mirror pattern on the consumer side is the **inbox** (record received message ids to
dedupe).

## 12. Privacy and the right to be forgotten

An immutable log and the GDPR's right to erasure are in direct tension. The standard answers:

- **Keep personal data out of events** where you can. Store a reference (`customer: "c-7"`), keep
  the PII in a separate mutable store, delete it there. This is the best answer and the most often
  skipped.
- **Forgettable payloads** (Verraes): the event holds a pointer to a separate payload store, which
  can be purged; you accept that the event store is no longer the whole truth.
- **Crypto-shredding** (Verraes): encrypt each person's personal fields with a per-person key held
  in a key store; to "forget" them, delete the key. Events remain; the data is unreadable.
  Caveats practitioners raise: under the GDPR encrypted personal data is still personal data, some
  regulators (Germany is often cited) don't accept it as erasure, it only works if consumers never
  stored the decrypted data, and key management becomes critical infrastructure.
- **Rewriting streams** (copy-transform without the person's data) is the blunt last resort, and
  some stores support truncating or deleting streams for exactly this.

Plus: projections and integration events copied the data elsewhere, and all of them need their own
erasure path. Design this in before go-live; retrofitting it means touching every historical event.

## 13. Time: temporal queries, audit, replay

The benefits people actually cite, with the honest qualifications:

- **Audit**: every change, by whom, when, and *why* (if events are named well). Regulated industries
  (finance, healthcare, gambling) are the clearest fit. Qualification: an audit log of low-level
  technical events is noise until you project it into something an auditor reads.
- **Temporal queries**: the state of X at time T is the fold of its events up to T. "What was the
  balance on 31 March?" is a replay with a cut-off.
- **"What did we know when?" (bitemporality)**: distinguish **when something happened** (business
  or effective time: the salary raise effective 1 January) from **when the system learned it**
  (recorded time: entered 15 February). Events can carry an `effective_at` separate from their
  recorded time, letting you answer both "what was true on 10 January?" and "what did we believe on
  10 January?". Fowler's retroactive event and Snodgrass's bitemporal databases are the references.
- **Debugging by replay**: copy the production stream of a misbehaving aggregate, fold it locally
  with the code, step through. Tests become **given events, when command, then events (or an
  error)**, which is the standard testing style.
- **New read models over old data**, and **what-if analysis** (replay history through changed rules).

The big replay caveat (Fowler): replay must not re-trigger side effects. Projections can be replayed
freely; anything that sends email, charges cards or calls APIs must be disabled or wrapped in a
gateway that is silent during replay. And code that *queried* external systems while handling
events will get different answers on replay unless the answers were recorded in the events, which
is a strong argument for putting the inputs to a decision in the event.

## 14. When not to use it, and how it goes wrong

Young, in 2016, called the event-sourced **whole system** the worst thing he had seen in a decade
of practice: "an event sourced monolith". The mainstream position: event sourcing is a tactical
choice for **some** bounded contexts, the ones where history, audit, temporal questions or complex
behaviour matter, not a top-level architecture.

Bad fits:

- CRUD domains: data entry, settings, catalogues, content. If your events are all `XUpdated`, stop.
- Teams without the time to learn it, under delivery pressure.
- Where you need ad-hoc queries over current state everywhere and can't tolerate stale reads.
- Domains with strong cross-entity invariants (uniqueness of usernames across all users is the
  classic awkward case: you need a reservation aggregate or a side table).
- Heavy PII with strict erasure obligations, unless designed for from the start.

The usual failure stories (Chris Kiehl's "Don't Let the Internet Dupe You, Event Sourcing is Hard"
is the canonical read):

- Other services subscribed directly to the raw event stream: boundaries dissolve, every event
  change becomes a cross-team migration, nobody can trace the flow.
- Months spent building infrastructure (store, subscriptions, replay, snapshots, versioning) before
  shipping a feature, and a lot of boilerplate per feature.
- UI impedance mismatch: forms submit blobs, the backend wants small intention-revealing commands.
- Requirements changed, old events no longer meant anything useful, and the promised historical
  fidelity eroded into migrations.
- Projection lag surprising users (just-created thing 404s, deleted thing lingers) and the
  workarounds creeping into every screen.
- Events designed as CRUD changesets, so all the cost and none of the value.
- Kafka as the event store, then discovering lost updates and no per-entity reads.

Typical costs, stated plainly: two models (write and read) per feature; eventual consistency in the
UI; versioning forever; operational work on subscriptions, checkpoints and rebuilds (rebuilding a
large projection can take hours, so plan capacity for it); a learning curve for every new hire;
and storage that only grows.

## 15. Relatives: CRDTs and git

All three are **logs of immutable changes, with state derived from them**. They differ in who
orders the log and how conflicts are handled:

| | Event sourcing | CRDTs | git |
|---|---|---|---|
| Unit | domain event | operation or state delta | commit (a snapshot, plus parent pointers) |
| Ordering | total order **per stream**, by a central store | none needed: operations commute, or are ordered by causality (vector clocks) | a DAG of commits; branches are concurrent histories |
| Conflicts | **prevented** at write time by expected version; the loser retries against new state | **impossible by construction**: merge is defined for every concurrent pair | **detected** at merge time; a human resolves them |
| Coordinator | yes, per stream (the store) | none; replicas sync peer to peer, offline for any time | none for commits; a shared remote by convention |
| Invariants | strong, within an aggregate | only those the data types guarantee (no "balance ≥ 0" across replicas) | none; tests and review |
| History | immutable; corrections are new events | immutable ops; tombstones for deletes | rewritable locally (rebase), immutable once shared by convention |

The key trade: event sourcing's per-stream ordering lets an aggregate **reject** a command
against definitely-current state, so it can enforce invariants like "no overdraft". CRDTs give
up rejection to gain availability without coordination: every replica accepts writes and they
converge, but an invariant that needs global knowledge cannot be enforced (two replicas can each
sell the last seat). git sits between: concurrent histories are allowed, then merged, and conflicts
are surfaced to a person. A common combination is CRDT-style merges for collaborative data
plus a coordinated stream for the few decisions that must not be double-made.

## 16. Common mistakes

1. Event sourcing the whole system instead of the few contexts that benefit.
2. CRUD events (`CustomerUpdated` with a changeset) instead of named facts.
3. Exposing internal domain events as other teams' integration contract.
4. No expected-version check (or "any" everywhere): lost updates under concurrency.
5. Using Kafka (or any log without per-stream conditional appends and per-stream reads) as the
   event store.
6. Command handlers deciding from read models instead of the aggregate's own stream.
7. Fold (`apply`) code that validates, throws, reads the clock or calls out.
8. Side effects re-fired during replay.
9. Non-idempotent projections and subscribers; assuming exactly-once delivery.
10. Assuming global ordering across streams.
11. Long-lived, ever-growing streams, then snapshots to hide it, instead of closing the books.
12. One event changing two aggregates; giant aggregates that serialise every write.
13. Editing or deleting stored events to fix mistakes, instead of compensating events.
14. No versioning plan; semantic changes under an unchanged field name.
15. Personal data embedded in events with no erasure strategy.
16. Treating CQRS, event sourcing, messaging and async as one decision.

## 17. Glossary

- **Aggregate / aggregate root**: a consistency boundary around one entity cluster; all changes go
  through the root; in ES, one stream.
- **Append-only**: events are only added, never changed or removed (barring deliberate migration).
- **At-least-once delivery**: every message arrives one or more times; requires idempotent handlers.
- **Bounded context**: a part of a system with its own model and language (DDD); the scope inside
  which events are internal.
- **Bitemporal**: tracking both effective (business) time and recorded (system) time.
- **Catch-up subscription**: reads history from a checkpoint, then switches to live events.
- **Category / category stream**: all streams of one type (`order-*`).
- **Causation id / correlation id**: the message that directly caused this one / the id shared by a
  whole workflow.
- **Checkpoint**: the last position a subscriber has processed.
- **Closing the books**: ending a stream with a summary event at a business period's end and
  starting a new one.
- **Command**: a request to change state; may be rejected; one handler.
- **Command handler**: loads the aggregate, decides, appends events.
- **Compensating event**: a new event that corrects or reverses an earlier one.
- **Competing consumers**: several instances share one subscription's load.
- **Copy-and-replace / copy-transform**: migrating events into a new stream or store with a
  transformation.
- **CQRS**: separate models for commands (writes) and queries (reads).
- **Crypto-shredding**: encrypt personal data with a per-subject key; delete the key to forget.
- **Decider**: `initialState`, `decide(command, state) -> events`, `evolve(state, event) -> state`.
- **Domain event**: a fact in the language of the domain, internal to a context.
- **Dual write**: writing to two systems without a shared transaction; the bug the outbox fixes.
- **Event**: an immutable, past-tense fact.
- **Event-carried state transfer**: events carrying enough data for receivers to keep local copies.
- **Event modeling / EventStorming**: workshop techniques for discovering events, commands and
  read models with domain experts (Adam Dymitruk; Alberto Brandolini).
- **Event notification**: an event announcing a change, with little data.
- **Event store**: a database of streams of events with conditional append and ordered reads.
- **Event streaming**: moving events between systems (Kafka, Kinesis); not the same as ES.
- **Expected version / optimistic concurrency**: append succeeds only if the stream is at the
  version the writer read.
- **Fold / left fold / reduce**: computing state by applying events in order to an initial state.
- **Forgettable payload**: personal data held outside the event, referenced, purgeable.
- **Global position**: an event's position across all streams (`$all`).
- **Idempotency**: processing a message twice has the same effect as once.
- **Inbox**: consumer-side record of processed message ids, for deduplication.
- **Inline / async / live projection**: updated in the append transaction / in the background /
  computed on read (Marten's terms).
- **Integration (public) event**: a stable contract for other contexts.
- **Invariant**: a rule that must always hold, enforced within an aggregate.
- **Outbox**: messages written in the same transaction as state, published by a relay.
- **Passage-of-time event**: an event that marks time (`DayEnded`) so processes can react to it.
- **Process manager**: stateful coordinator that reacts to events and sends commands.
- **Projection / projector**: code that builds a read model from events.
- **Read model / view / query model**: a store shaped for queries, derived from events.
- **Rehydration**: loading an aggregate by folding its stream.
- **Replay / rebuild**: running events again through a projection or fold from the start.
- **Saga**: a long-running transaction as steps with compensations; often used for process manager.
- **Snapshot**: stored state at a stream version, to shorten rehydration.
- **Stream**: an ordered sequence of events with one id; usually one per aggregate.
- **Stream version / revision**: an event's position within its stream.
- **Summary event**: one coarse event summarising a finished process for outside consumers.
- **Tolerant reader / weak schema**: consumers ignore unknown fields and default missing ones.
- **Upcasting**: transforming old event versions into the current shape at read time.

## 18. What's on the test

1. **What is event sourcing, in one sentence?** Storing every state change as an immutable event and
   deriving current state by folding them; the events are the source of truth.
2. **Event vs command?** A command asks and can be rejected, with one handler; an event states a fact
   that already happened, has any number of subscribers, and cannot be rejected.
3. **How should events be named?** Past tense, in the domain's language, stating what happened in
   business terms: `OrderShipped`, not `UpdateOrder` or `OrderStatusChanged`.
4. **What is a stream?** An ordered sequence of events with one id, typically one per aggregate; the
   unit of ordering and concurrency.
5. **How do you prevent two concurrent commands from breaking an invariant?** Optimistic concurrency:
   append with the expected stream version; on conflict, reload, re-decide, retry.
6. **What is an aggregate for?** Enforcing invariants transactionally within a consistency boundary.
7. **Why not change two aggregates with one event or transaction?** It couples them, needs
   multi-stream atomicity, hurts scaling; use eventual consistency via a process manager.
8. **Process manager vs saga?** A process manager is a stateful coordinator sending commands; a saga
   is a sequence of local steps with compensations. Often used interchangeably.
9. **How do you get current state?** Rehydrate: read the stream and fold events with `apply`/`evolve`.
10. **When do you need snapshots?** Rarely; when a stream is long enough that loading is measurably
    slow. First try shorter streams (closing the books). Snapshots are a disposable cache.
11. **Should `apply` validate?** No. Validation is in the command handler; `apply` records an
    already-made decision and must never fail or have side effects.
12. **What is a projection?** Code that consumes events and builds a read model for queries.
13. **How do you fix a buggy projection?** Fix the code, reset its checkpoint, replay (often into a new
    store side by side, then switch).
14. **Are read models consistent with the write side?** Usually eventually; handle with
    read-your-writes by version, returning state from commands, or inline projections.
15. **Is CQRS the same as event sourcing?** No. CQRS is separate read and write models; you can do it
    without events. Event sourcing almost always needs CQRS, not the reverse.
16. **Name Fowler's four event-driven patterns.** Event notification, event-carried state transfer,
    event sourcing, CQRS.
17. **What is wrong with `CustomerUpdated` carrying the whole record?** It's CRUD sourcing: loses
    intent, forces consumers to diff, all the cost of ES without the value.
18. **Domain vs integration events?** Internal, fine-grained, free to change vs public, stable,
    versioned contract, often a summary. Don't publish internal events directly.
19. **How do you change an event's schema?** Additive changes with tolerant readers; new event types;
    upcasting at read time; copy-transform as last resort.
20. **An event was recorded wrong. What do you do?** Append a compensating event; don't edit history.
21. **What delivery guarantee do you design for?** At-least-once; make handlers idempotent (dedupe by
    event id or by last processed stream version).
22. **What ordering can you rely on?** Total order within a stream; across streams only what the store
    explicitly guarantees (a global position), and only per partition once on a broker.
23. **Why is Kafka not an event store?** No per-key conditional append (expected version), no cheap
    per-entity reads, topic-per-entity doesn't scale; it's for streaming, not system of record.
24. **What problem does the outbox solve?** The dual write: committing state and publishing a message
    atomically, by writing both in one transaction and relaying later.
25. **How do you handle GDPR erasure?** Keep PII out of events; forgettable payloads; crypto-shredding
    (with its legal caveats); rewrite as last resort; also erase from projections.
26. **What breaks on replay?** Side effects (emails, payments) and external queries; disable or
    gateway them, and record decision inputs in events.
27. **What is a temporal query?** State as of time T, by folding events up to T; bitemporal adds "as
    known at time T".
28. **How do you test an event-sourced aggregate?** Given past events, when a command, then expect
    these events (or this error).
29. **When should you not use event sourcing?** CRUD domains, whole-system adoption, teams without
    time to learn it, heavy cross-entity invariants, strict erasure without design.
30. **What do correlation and causation ids give you?** Tracing a workflow end to end and knowing which
    message directly caused each event.

## 19. Mapping to Orchestra

Orchestra (from `README.md` and `docs/why-crdts.md`) has an append-only event log of lines
`{"type", "subject", "source", "id", "time", "kind", "data"}`, **items** folded per `subject`, and
**machines**: processes that read events on stdin and write events on stdout.

| Event-sourcing term | Orchestra term | Notes |
|---|---|---|
| event store | the event log | append-only, the record everything else derives from |
| event | an event line | `type` and `data` as usual; the envelope is CloudEvents-like |
| stream | all events with one `subject` | there is no separate stream id or per-stream version |
| aggregate state / fold | an **item** | the generic fold: latest value per `data` key |
| custom `apply` | a reducer program for chosen subjects | |
| event id / dedup | `id`, "drops an id it has seen" | idempotent appends by id, as many stores do |
| metadata: who | `source` | close to an actor or causation field |
| projection / read model | items, metrics, the derived machine list, the model | |
| subscriber / reactor / process manager | a **machine** with `on:` | reacts to event types, writes events |
| scheduled process / passage of time | a machine with `every:` | |
| command | none as a separate kind | machines are triggered by events and append events |

Where it matches standard practice:

- An append-only log as the source of truth, with everything else (items, the list of machines and
  people, the model) derived from it and rebuildable.
- State as a fold over events per entity; a pluggable fold (reducer) where the generic one isn't
  enough.
- Deduplication by event id, which is the standard idempotent-append protection.
- A standard envelope separating metadata from a payload the core does not interpret.
- Consumers as independent subscribers, like projections and process managers; past-tense,
  dotted event types (`github.check.failed`, `deploy.release.done`) in the domain's language.
- `why-crdts.md` argues for **named facts** over whole-object writes, which is exactly the
  standard argument against CRUD events, and for correcting a wrong classification by appending
  a superseding fact rather than rewriting it, which is the compensating-event rule.

Where it departs:

- **No per-stream expected-version check.** An append is accepted whatever the subject's state; two
  machines that read the same item and both write cannot be told apart from sequential writes. So
  there is no aggregate in the DDD sense: nothing rejects an event against current state, and
  invariants are not enforced at write time. `why-crdts.md` proposes a `base` version and
  `requires` rules as future work; that is the gap it describes.
- **No commands.** In standard practice a command is validated and may be rejected, and only the
  resulting events are stored. In Orchestra, what a machine writes is the event; an event type can
  also act as a request to another machine (`ci_triage.real` triggering `fixer`), which is close to
  what Fowler calls a passive-aggressive command.
- **The generic fold is last-write-wins per field, ordered by event time**, not by append order
  within a stream. Standard folds apply events in stream order and are written per event type;
  Orchestra's is closer to a CRDT register map (or to CRUD-style changesets, if events carry whole
  records) and depends on producers' clocks.
- **Done-ness from type naming** (`.completed`, `.closed`, `.cancelled`) is a convention in the
  fold, where an event-sourced system would decide it in the aggregate's own `apply`.
- **No distinction between internal and integration events.** Every event is on one log, visible to
  every machine; that is deliberate (one system, one log) but it is the shape the critiques warn
  about for services owned by different teams.
- **Machines are external processes with side effects**, and replay is not a concept for them:
  re-running history would re-run agents. Folding items and deriving the model are replayable; a
  machine's work is not, which matches the standard "gateway side effects during replay" caveat.
- **Ordering** is a single log rather than per-stream versions; the `time` field is the producer's
  claim and the fold uses it, where standard practice orders by the store's own sequence.
- **Toward CRDTs rather than an event store.** `why-crdts.md` chooses convergence without a
  coordinator (several Orchestras syncing by union) over per-stream serialisation; per section 15,
  that gives availability and mergeable concurrent work, and gives up rejecting a command against
  definitely-current state.

## 20. Sources

- Martin Fowler, "Event Sourcing" (2005): https://martinfowler.com/eaaDev/EventSourcing.html
- Martin Fowler, "What do you mean by 'Event-Driven'?" (2017):
  https://martinfowler.com/articles/201701-event-driven.html
- Martin Fowler, "CQRS": https://martinfowler.com/bliki/CQRS.html
- Greg Young, *Versioning in an Event Sourced System* (Leanpub; chapter list consulted):
  https://leanpub.com/esversioning/read
- Greg Young, 2010 post on what CQRS is not (reproduced):
  https://gist.github.com/meigwilym/025f08208b5640ad26bc410c8a83b10f
- InfoQ, "Greg Young on event sourcing anti-patterns" (DDD Europe 2016):
  https://www.infoq.com/news/2016/04/event-sourcing-anti-pattern
- Transcript of Greg Young's Code on the Beach 2014 talk (Kurrent blog):
  https://kurrent.io/blog/transcript-of-greg-youngs-talk-at-code-on-the-beach-2014-cqrs-and-event-sourcing
- Udi Dahan, "Clarified CQRS" (2009): https://udidahan.com/2009/12/09/clarified-cqrs/
- Vaughn Vernon, "Effective Aggregate Design", parts I and II:
  https://dddcommunity.org/wp-content/uploads/files/pdf_articles/Vernon_2011_1.pdf and
  https://www.dddcommunity.org/wp-content/uploads/files/pdf_articles/Vernon_2011_2.pdf;
  summary at InfoQ: https://www.infoq.com/news/2014/12/aggregates-ddd ; and
  *Implementing Domain-Driven Design* (Addison-Wesley, 2013), chapter 10, Aggregates.
- Kurrent (EventStoreDB), "What is event sourcing?":
  https://kurrentdb.kurrent.io/blog/what-is-event-sourcing
- Oskar Dudycz, "Snapshots in Event Sourcing" (Kurrent blog):
  https://kurrentdb.kurrent.io/blog/snapshots-in-event-sourcing
- Oskar Dudycz, "Event Streaming is not Event Sourcing!":
  https://event-driven.io/en/event_streaming_is_not_event_sourcing/
- Oskar Dudycz, "Internal and external events":
  https://event-driven.io/en/internal_external_events/
- Oskar Dudycz, "Don't let Event-Driven Architecture buzzwords fool you":
  https://event-driven.io/en/dont_let_event_driven_architecture_buzzwords_fool_you/
- Mathias Verraes, "Eventsourcing Patterns: Crypto-Shredding":
  https://verraes.net/2019/05/eventsourcing-patterns-throw-away-the-key/
- Mathias Verraes, "Eventsourcing Patterns: Forgettable Payloads":
  https://verraes.net/2019/05/eventsourcing-patterns-forgettable-payloads/
- Mathias Verraes, "Patterns for Decoupling in Distributed Systems: Summary Event":
  https://verraes.net/2019/05/patterns-for-decoupling-distsys-summary-event/
- Jérémie Chassaing, "Functional Event Sourcing Decider":
  https://thinkbeforecoding.com/post/2021/12/17/functional-event-sourcing-decider
- Marten event store documentation: https://martendb.io/events/
- Axon Framework reference, event versioning and upcasting:
  https://docs.axoniq.io/axon-framework-reference/4.10/events/event-versioning/
- Chris Kiehl, "Don't Let the Internet Dupe You, Event Sourcing is Hard":
  https://chriskiehl.com/article/event-sourcing-is-hard
- Conduktor, "Event sourcing with Kafka: patterns and pitfalls":
  https://conduktor.io/blog/event-sourcing-kafka-patterns-pitfalls
- "Event Sourcing: why Kafka is not suitable as an event store" (ITNEXT):
  https://itnext.io/event-sourcing-why-kafka-is-not-suitable-as-an-event-store-796e5d9ab63c
- Chris Richardson, "Pattern: Transactional outbox":
  https://microservices.io/patterns/data/transactional-outbox.html
- Discussion of crypto-shredding and GDPR compliance:
  https://world.hey.com/otar/challenges-of-building-gdpr-compliant-event-sourced-system-0db251d4
