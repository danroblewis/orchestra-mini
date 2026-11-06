# Facts first: machines are fact-in, fact-out stream processors

Status: proposal, 2026-11-06. Changes README contracts 1, 3 and 4, SPEC §2, §3, §5 and §6, and
`orchestra/setup/setup.md`.

## The change in one paragraph

Orchestra's log holds **facts**: past-tense statements a source makes about a subject ("alice moved
ABC-123 from In Progress to In Review at 10:05"). A machine is a fact-in, fact-out stream
processor: it reads facts on stdin and writes facts (and measurements) on stdout. An event is one
kind of fact, the commonest; the fact is the base. What a team looks at (a ticket's current
state, a review queue, time in status) is a **projection**: a program that reads facts and prints
**views**. Projections are the one exception to facts in, facts out: views are rebuilt and never
logged, so the log holds only facts. Facts are designed first and kept forever; projections are
written after there are facts to look at, can be many, and are changed and rebuilt freely.

## Why

- **The fold forces facts into patches.** Contract 3 keeps the latest value of each `data` key, so
  a fact is only useful if it is shaped like a field update. A faithful Jira fact,
  `{"from": "In Progress", "to": "In Review", "by": "alice"}`, folds into `fields.from` and
  `fields.to`, which is nonsense. Producers therefore write conclusions (`{"status": "In Review"}`)
  and drop what the source said (who, from what, on which commit), and the setup agent decides
  each kind's fields before it has seen any data.
- **What was not recorded can never be projected.** If the approval fact has no commit SHA, no
  later view can tell a stale approval from a current one.
- **Changing a view should not change history.** With one fold per subject, a new way of looking
  at tickets means changing what producers emit. With projections, it means a new projection
  file, rebuilt from the log.
- **Facts merge; views decide.** Two sets of facts with stable ids merge by union. Views do not:
  `latest`, `done` and any queue a subject leaves can be changed by a late fact. Facts put that
  decision in the projection, in one place that can be changed later (`docs/why-crdts.md` §5).

The cost: a projection is a program the owner must be able to read. Most subjects need only the
built-in `latest`, and a projection is written only when a view needs more.

## Words

| Word | Means |
|---|---|
| **fact** | One line with a `type`: something a `source` asserts about a `subject`, true as of its `time`. Never edited: a correction is a new fact. |
| **event** | A fact that something happened (`jira.issue.status_changed`). Most facts are events. |
| **observation** | A fact that something was so when looked at (`jira.issue.observed`, a poller's snapshot). |
| **decision** | A fact that a machine or person judged something (`ci_triage.real`). |
| **repeat** | An observation whose `data` (canonical JSON) equals the previous fact with the same subject, type and source, in log order. It says nothing new (item 2). |
| **subject** | The thing a fact is about, named as its own system names it (`ABC-123`, `acme/api/pull/3`). A machine's decision about a pull request has the pull request as its subject. Other things a fact concerns (the reviewer, the commit, a ticket a pull request mentions) are named in `data` by their ids. |
| **`time`, `seq`** | `time`: when the source says the fact was true, by its clock. `seq`: the fact's position in the log, when Orchestra recorded it. |
| **projection** | A named program from facts to views. The built-in one is `latest`. |
| **key** | What a view is about: the subject, or for a `whole: true` projection the `key` it prints. |
| **view** | One projection's output for one key. What the pages, `query` and machines read. |
| **machine** | A process that reads facts and writes facts and measurements; a **collector** is one with `every` that reads a source. |

Event, observation and decision are naming conventions in the type, not a field. The subject is
the one modelling choice made before data exists; the ids in `data` are what let a `whole`
projection join subjects later.

## What changes

1. **Lines and the envelope (contract 1, SPEC §2).** A line with `type` is a fact, with the same
   fields as today (`type`, `subject`, `source`, `id`, `time`, `kind`, `data`; only `type`
   required) and one optional new key, `causedby` (item 5). The bus keeps the CloudEvents
   envelope, adds the extensions `causedby` and `depth` (item 5), and stamps `recordedtime`
   itself. Dedup by `id` is unchanged.

2. **Repeats.** Every path by which facts reach a projection, a machine or a query goes through one
   function, which skips repeats and stamps `seq`. Publishing also skips a repeat when it can tell
   and counts it as `repeats` on the run record, so the log grows only on change; that is an
   optimisation, best effort across processes, and correctness doesn't depend on it. A poller can
   therefore print the whole current state every time and keep no state of its own (setup step 3
   says how). A subject known only by observations never gets a `.closed` type; a projection
   decides when it is `done`.

3. **Projections replace the fold (contract 3, SPEC §6).**
   ```yaml
   projections:
     - name: ticket                     # Where is each ticket, and who has it?
       match: "ABC-*"                   # subject glob, or a list of them
       command: .orchestra/projections/ticket.py
     - name: review                     # Which pull requests wait on a review, since when?
       match: "acme/api/pull/*"
       command: .orchestra/projections/review.py
     - name: ticket_prs                 # Which tickets have an approved pull request?
       match: ["ABC-*", "acme/api/pull/*"]
       types: ["jira.issue.*", "github.pr.*"]   # optional: only these reach it, and it says what it reads
       whole: true                      # reads every matching fact on each build
       command: .orchestra/projections/ticket_prs.py
   ```
   - **How it runs.** Each build is one run of the command. Its stdin is every fact of the subjects
     being built (all matching subjects the first time, then only those with new facts), in log
     order, each with its `seq`; it prints one JSON line per subject, `{"subject": …, …}`. A
     projection that needs the source's order sorts by `time`. A board (the review queue) is views
     filtered. A `whole: true` projection gets every matching fact on each build and prints one
     line per key it chooses (`{"key": …}`: a ticket, a person), and its build replaces its views as
     a set, so a key it no longer prints is gone, not an error. That is the whole history per
     build, fine to about 10^5 facts. A projection is any command, so `duckdb -jsonlines -c
     "<query over the facts>"` is one too, often clearer for a join or a count; a fold in time
     order is clearer as a short program.
   - **It reads only its stdin** (no network, no environment) and prints times, not ages
     (`in_review_since`, not "2 days"), so it can be rerun at will and gives the same views.
   - **`latest`, built in,** covers every subject with no entry, in Orchestra's own process:
     `fields` (today's fold, the latest value of each `data` key), `by_type` (the latest fact's
     `data` of each type, so `{"from", "to", "by"}` stays together under
     `jira.issue.status_changed`), `type`, `first`, `last`, `kind` and `done` (by the latest type's
     verb). It orders by `time`, ties by `seq`, as today's fold does, so a month of history
     backfilled after today's snapshot doesn't become the current state.
   - **One builder, one lock.** Views are built by one function, under a file lock in the state
     directory, each run of a projection under a 60 s timeout that kills its process group. Each
     projection's views are kept in their own file (`views/<name>.json`, replaced whole) with
     `built_to` (the last `seq` built) and the projection's hash: its entry (`match`, `types`,
     `whole`, `command`), every file in the folder of the script `command` names (none for an
     inline command), and Orchestra's version. A changed hash rebuilds the projection from the log;
     `orchestra query views --rebuild` forces it. The runner calls the builder after each batch of
     facts; a page or `query` calls it only when no runner is alive, and otherwise reads the files.
     Every view carries `behind`: how many matching facts have a `seq` past `built_to` (for a
     per-subject view, its subject's facts only). Routing facts to machines never waits for a
     projection, and a rebuild triggers no machine.
   - **A failure says so.** A subject the program printed no line for, or a run that failed or
     timed out, keeps its last good view with `error` and `error_at`, never `latest` in its place;
     each projection's last build, time and errors show on the Core page.

4. **Machines read views (contract 4, SPEC §3).** A once run gets its fact on stdin and in
   `ORCHESTRA_FACT`, and `ORCHESTRA_VIEWS_FILE`: `{projection: view}` for its subject, `latest`
   included, and a `whole` projection's view when its key is the subject. Before the run starts the
   builder brings its subject's per-subject views to at least its fact; `whole` views are read as
   built, with `behind`. The file is kept as `runs/<id>.views.json` (the newest 500), so "why did
   the agent do that" has the exact views it saw. A continuous machine reads views with
   `orchestra query views`.
   - `orchestra agent` shows the model the fact, the subject's last 10 facts one per line, each
     view other than `latest` in full up to 2 KB (else named only, 8 KB in all), and from `latest`
     only `fields` and `last`, and names the file and `orchestra query facts --subject S` for the
     rest.
   - `orchestra decide --inputs ticket:status` names a projection's field; a bare `status` is
     `latest`'s `fields.status`.

5. **Causation.** Every fact a run writes is stamped in the envelope, never in `data`, with
   `causedby` (the triggering fact's id) and `depth` (its cause's plus one). A once run's lines are
   stamped as Orchestra reads them, and `orchestra emit` in it reads `ORCHESTRA_CAUSEDBY` and
   `ORCHESTRA_DEPTH`. A continuous machine's lines are stamped with the oldest fact it was given
   and has not answered (exact for `orchestra agent`, which takes one at a time); a line may name
   `causedby` itself. Facts from the endpoint and from scheduled runs have depth 0. A fact with
   `depth` over 8 is recorded but **held**: no machine gets it, and the run record and the Core page
   say so. That stops loops among machines (A to B to A). A loop through an outside system (a fixer
   pushes, the GitHub poller sees a failed check, triage wakes the fixer) starts again at depth 0
   and is bounded only by `--per-day-usd`.

6. **Commands and the check (SPEC §5).** `orchestra query facts` (was `events`; `--subject`,
   `--type`, `--trace ID`, which walks `causedby` both ways). `orchestra query views [PROJECTION]
   [show SUBJECT] [--input SUBJECT] [--rebuild]` (was `items`); `--input` prints exactly what the
   projection is given (every matching fact for a `whole` one), so `orchestra query views review
   --input acme/api/pull/3 | review.py` is its test. `orchestra query check` reports what can be
   checked from the log alone:
   - a source whose facts' `time` always equals `recordedtime` (it stamps now, not the source's
     time);
   - `data` over 4 KB;
   - a type whose keys stopped appearing or newly appeared (the source changed its shape);
   - a collector silent for more than three of its `every`, or whose runs keep adding facts where
     nothing changed;
   - each projection run twice over its facts: a missing or `error` view, or different bytes the
     second time.

   `orchestra emit` emits a fact. The bus counter keeps its name, `orchestra_events_total`, and
   counts facts: renaming it would leave discovery two copies of every flow over its window.

7. **Pages.** The Items page becomes **Subjects**: each subject's facts and its views. The
   Dashboard and Subjects page take `waiting_on` and `done` from the subject's first view, in
   config order, that has the key at its top level, falling back to `latest` (`fields.waiting_on`,
   `done`).

8. **The setup agent designs facts first** (`orchestra/setup/setup.md`):
   1. Ask the owner for 2 to 4 questions Orchestra must answer, in their words. The questions
      choose which sources and which records to collect, never which fields.
   2. For each source, find its change history (Jira's changelog, GitHub's timeline and reviews,
      git's log); prefer history to current state. Write a collector with one fact per change
      record: `id` the source's own id for it (`jira:ABC-123:history:9912`), so rerunning adds
      nothing; `subject` the source's own thing; `time` the source's time for the change, not now;
      `data` the record as the source gives it, its field names and nesting kept (a Jira history
      with its list of changed items, each with both sides), leaving out bodies, HTML and anything
      over 4 KB, and saying so in `truncated`. Renaming and flattening belong in projections,
      shared through a helper module beside them (`.orchestra/projections/lib.py`).
   3. A source with no history: print its current state as `<system>.<noun>.observed`, with no
      `id` (an id from its content would drop a return to an earlier state) and without fields
      that change on every look (`updated`, `lastViewed`); Orchestra keeps only the changes.
   4. Run the collector twice. Read `orchestra query facts --subject S` for three real subjects
      and check that every field a question needs is there.
   5. Only now write the projections, one per question, the question as its first comment, each
      tested on recorded facts (`orchestra query views NAME --input S | <command>`). Prefer a new
      projection to a changed fact; when a fact's shape must change, give it a new type (`….v2`).
   6. Done means `orchestra query check` reports nothing; quote it in the report. Translator
      machines (two sources saying one thing in different words) and rule machines (a fact
      written when "an approval is of the current head" breaks) are not written during setup;
      name them in the report when a question needs one.

9. **Old names are refused with the new one**, as the config already does for earlier fields:
   `items:` says `projections:`, and `orchestra query events` and `orchestra query items` say
   `facts` and `views`. `orchestra run` refuses an old config; `orchestra chat` then briefs the
   Supervisor, not a new setup, to rewrite the reducer as a projection (it gets log order now, not
   time order: sort by `time` if it relied on that). For one release `ORCHESTRA_EVENT` is still set,
   and `ORCHESTRA_ITEM` holds `latest` without `by_type`, plus `events` (the count), empty past
   32 KB: it was the one fold, in the environment, where a large view fails to start the process.
   `reset` removes `views/` and the old `items.json`.

## What does not change

The bus backends, dedup by `id`, the metrics store, stores and adapters, discovery, machine
scheduling, the endpoint (it takes a fact as JSON), the six-contract shape.

## Considered and not done

- **Keep "event" and only add named reducers.** Most of the value at a tenth of the diff, but it
  keeps the word that invites patches, commands and notifications onto the log, and the owner's
  direction is that machines are fact-in, fact-out.
- **A `set` field carrying the producer's conclusion beside the fact.** It asks the collector to
  decide the view's fields at collection time, which is what this change undoes; `latest`'s
  `by_type` keeps each fact's `data` together beside `fields` instead.
- **One view printed for everything.** Boards are per-subject views filtered; questions across
  subjects are `whole: true` projections keyed by what they choose.
- **Projections as machines writing their views as facts.** Derived state in a log kept forever
  makes every rebuild a migration.

## Later, when a flow needs it

A retraction fact; revisions for a fixed collector's corrections (`…@r2`, applied in the one read
function); `read_at` stamps of the views a fact's writer read; conditional appends (`expect_seq`;
a fact id like `fix:<check-id>` already stops a second fixer); view changes as facts (`emit:
true`); clock-reading projections (`as_of`); per-projection data tests (`.expect.yaml`: not null,
unique, accepted values, relationships) and fixtures; `.unobserved` facts for subjects a complete
scan no longer returns; a shadow build with `query views --diff` before a changed projection
replaces its views; a SQLite index of the facts for `--subject` and `--trace`; folding a subject
from a saved state (today a changed subject is refolded from its first fact); typed fact schemas;
Datalog-style rules.

## How we will know it worked

A demo project, with questions written down and kept from the setup agent before it starts:
`orchestra chat` sets it up from nothing, `orchestra query check` reports nothing, and each kept
question (at least one spanning two sources, and one needing a field no asked question used) is
answered by a change only to `.orchestra/projections/` and the `projections:` list, checked
against the source itself. The run is repeated from nothing after each round of fixes, since an
agent's setup differs between runs.

## History

Written 2026-11-06 and revised after ten critical reviews, in order: an event-sourcing veteran
(view consistency, purity, a small fact shape), a deductive-database researcher (two times,
causation, chained observations, `by_type`), a stream-processing engineer (log order, build off the
dispatch path, views in a file, depth), a DDD practitioner (questions first, the subject as the one
up-front choice), a minimalist (cut half of what the first four added), an agent-harness engineer
(questions choose records, not fields; Orchestra skips repeats; `query check`), an SRE (one
builder, last good view on failure, envelope stamps, the views a run saw), an analytics engineer
(source names kept, schema drift, `whole` projections), an implementer (one read function, the
lock, the counter kept), and a technical editor.
