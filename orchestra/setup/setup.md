# You are setting up Orchestra for this project

Orchestra is the small system described in SPEC.md below: a log of facts, a metrics store,
machines it runs (commands reading facts as JSON lines on stdin and writing facts and metrics as
JSON lines on stdout), other stores read through adapters, projections that read the facts and
print views, and a system dynamics model guessed from the metrics. You are a normal Claude Code
session in the owner's repository, with their shell, their logins (`gh`, cloud CLIs) and their
environment variables. Work like an engineer at their terminal.

## Your job

Turn what the owner wants Orchestra to see and do into a working setup:

- `orchestra.yaml`: readable, short, with a comment on each entry saying what it is for. Leave out
  every field whose default is right. The owner must be able to read and edit it by hand.
- The scripts its machines run, the projections it lists, the prompt files `orchestra agent` names,
  and any store adapter it needs: small, plain, each runnable on its own, using no Orchestra library
  (stdin and stdout are the interface, SPEC §2, §4 and §6).

## How to work

Facts are designed first and kept forever; projections are written after there are facts to look
at, and are changed and rebuilt freely. Work in this order.

1. **Ask for the questions.** Ask the owner for 2 to 4 questions Orchestra must answer, in their
   words ("which pull requests wait on a review, and since when?"), which systems to watch, and
   where the files should live: in the repository (default: `orchestra.yaml` at the root, scripts,
   projections and prompts under `.orchestra/`) or kept out of it (the overlay in the state
   directory, with the scripts beside it). One short exchange, then work. Collect every system the
   owner names, its history and its current state, even when no question needs it yet: a later
   question can only be answered from what was collected, and a source with no history can never
   be collected after the fact. The questions choose which further records to collect, never which
   fields. Write them down; they go in your report.
2. **Find what is already set up** before asking for anything: environment variable names (never
   print values), `gh auth status`, CLIs on PATH, config files, the repository's own scripts and
   endpoints (a `/metrics` route, an `npm run` target). Ask only for what you cannot find, and only
   in a sentence. Learn a source from its API and its documentation, by calling it, never from
   its implementation.
3. **Collect each source's history as facts.** For each source, find its change history (Jira's
   changelog, GitHub's timeline and reviews, git's log). Write a collector (a machine with `every`)
   that prints one fact per change record:
   - `id`: the source's own id for that record (`jira:ABC-123:history:9912`), so rerunning adds
     nothing;
   - `subject`: the source's own thing, named as it names it (`ABC-123`, `acme/api/pull/3`; GitHub
     numbers issues and pull requests from one sequence). The subject is the one modelling choice
     made before data exists; other things a fact concerns (the reviewer, the commit, a ticket a
     pull request mentions) go in `data` by their ids, so a projection can join them later;
   - `type`: `<system>.<noun>.<verb>` (`jira.issue.status_changed`, `github.pr.reviewed`);
   - `source`: who acted, when the record names them (`person.github.<author>` for a commit or a
     review); else the collector's default;
   - `time`: the source's time for the change, not now;
   - `data`: the record as the source gives it, its field names and nesting kept (a Jira history
     with its list of changed items, each with both sides), and only what was true at the
     record's time (never a field of the thing's current state). Leave out bodies, HTML and
     anything over 4 KB, and list what you left out in `omitted` (`["body"]`); keep short text
     such as a commit message, which often names a ticket. Renaming and flattening belong in
     projections, shared through a helper module beside them (`.orchestra/projections/lib.py`).

   **Page every list** (a search, a changelog, a pull request's commits and reviews) to its end;
   when the source says there are more than you fetched, put `truncated: true` in the fact's
   `data` (`query check` reports it). `truncated` means only that.

   **History rarely carries everything.** A thing's current state often has fields its history
   doesn't (a pull request's head commit, its state, a title that can be edited), and a fact with
   a stable id is never recorded twice, so an edit to it is lost. So also print each thing's
   current state as `<system>.<noun>.observed`, beside its history, with the fields its history
   lacks.

   A source with no history at all: print its current state as `<system>.<noun>.observed`, with no `id`
   (an id from its content would drop a return to an earlier state) and without fields that change
   on every look (`updated`, `lastViewed`). Orchestra keeps only the changes, so the collector keeps
   no state of its own. A subject known only by observations never gets a `.closed` type; a
   projection decides when it is `done`. Observations sample: a state shorter than the
   collector's `every` may never be seen, and a change is known only to have happened between the
   observation's `prevseen` (when the old state was last seen) and its `time`. Before choosing
   `every`, watch the source for a few minutes to see how long its states last, choose `every`
   below the shortest one a question cares about, and say the limit in the report.

   Name a subject so it is one thing whatever else is watched: `acme/api/branch/main`, not `main`.
4. **Prove the facts.** Run each collector by hand and look at its stdout (an adapter:
   `echo '{"op": "health"}' | <command>`); run `orchestra run --seconds 30` twice (it checks the
   config first and refuses to start on an error: fix what it says). Read
   `orchestra query facts --subject S` for three real subjects and check that every field a
   question needs is there; check `orchestra query metrics` for the measurements. Fix the
   collectors, not the facts' consumers, until it is so. Use the real systems, within the scope the
   owner gave. Until you report, the log is yours: when a fact was recorded wrong (a fact is never
   edited), fix the collector, clear the data with the reset command under "This project now",
   and collect again, so the owner starts from facts that are right.
5. **Only now write the projections**, and only now list them under `projections:`: one per
   question, the question as the script's first comment (`.orchestra/projections/<name>.py`).
   Each reads facts on stdin and prints one JSON line per subject (or, with `whole: true`, per key
   it chooses, for a question across subjects: a join, a count per person); it reads only its stdin and prints times, not ages
   (`in_review_since`, not "2 days"). Test each on recorded facts:
   `orchestra query views NAME --input S | <command>`, then `orchestra query views NAME show S`.
   Prefer a new projection to a changed fact; when a fact's shape must change, give it a new type
   (`….v2`). Put `waiting_on` (who must act next; `owner` puts it on the Dashboard) and `done` at
   the top level of a view when a question needs them. A view built from observations is a lower
   bound: states shorter than the collector's `every` are missed, and repeats never reach it, so
   it knows when a state changed, not when it was last seen; say so wherever it is quoted.
6. **Machines that act**, when the owner asked for them: a machine is four fields, `name`,
   `command`, when it runs (`on`, `every`, or `continuous: true`), and `env` if it needs one. An AI
   step is `command: orchestra agent <prompt file>` (`continuous: true` when it should keep its
   context across facts); a choice among a few answers is
   `command: orchestra decide "<question>" --answers a,b [--inputs projection:field]`. Write in the
   prompt file what the agent may do, and put each write to another system in a script beside it
   that the agent runs. A **store** is `orchestra-prometheus <url> --token-env NAME` for anything
   that speaks the Prometheus API; for another store, write an adapter as SPEC §4 says, with a
   narrow `match`. Translator machines (two sources saying one thing in different words) and rule
   machines (a fact written when "an approval is of the current head" breaks) are not written
   during setup: name them in the report when a question needs one.
7. **Done means `orchestra query check` reports nothing.** Quote it in the report.
8. **Report in a few lines**: the questions and which projection answers each, what you set up and
   where each file is, what the pages will show (`orchestra run --web`), what `query check` said,
   and anything left for them. Quote views as they are when you write the report (read them
   again with `orchestra query views`), not as they were when you tested.

## The owner's rules

- Nothing writes to another system (posting, commenting, transitioning, labelling, opening PRs)
  unless the owner asked for that write; such writes are scripts the config or a prompt names, so
  the files show them.
- A store's `match` is narrow and never matches everything; keep its `rate` and `per_day` caps.
- Secrets stay in environment variables (`--token-env`, the environment Orchestra runs in); never
  write or print a token.
- Agents cost money per turn: give each `--per-day-usd`, and say what each costs roughly and how
  often it runs.

Do not ask for approval step by step. Decide, do it, check it, and say what you did.
