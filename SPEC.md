# Orchestra, minimal

Orchestra is an append-only **log of facts**, read through **projections** into **views**, and a
**metrics store**, readable together with remote stores through **adapters**. It runs **machines**
on a schedule, on facts, or continuously: a machine is a fact-in, fact-out stream processor that
reads facts as JSON lines on stdin and writes facts and metrics as JSON lines on stdout, and
Orchestra feeds its stdin and routes its stdout. An event is one kind of fact, the commonest.
From what it records, Orchestra knows who acts and discovers a **system dynamics model** from the
metrics. Everything a team does lives in its own scripts and prompts; Orchestra only runs them,
records what they say, and shows it. The six contracts are listed at the top of README.md; this
file states each exactly.

The owner, 2026-11-05: "This should be so damn simple that someone could read the whole config and
even edit it by hand." "Most of our interface is 'run a command' and 'parse the stdout'."

## 1. The config file

One file, YAML or JSON (the same schema), in layers merged in this order, the later winning:

1. `orchestra.yaml` (or `.yml`, `.json`) at the repository root, optional, shared by the team;
2. the private overlay, `~/.orchestra/<project>/orchestra.yaml` (or `.json`), the owner's own.

Maps merge key by key; `stores`, `machines` and `projections` merge by `name`, an overlay entry replacing the
repository's of that name whole, and `{name: x, enabled: false}` removing it. Unknown keys are
refused with the key's path, and the fields of earlier versions (`items`, `kind`, `mode`, `prompt`,
`tools`, a store's `url`) are refused with what says them now. **Secrets are environment variables
only**: a literal secret in the file is refused. Durations are `30s`, `5m`, `2h`, `1d`.

```yaml
project: payments                  # optional: the name pages show (default: the repository's folder)
bus: file                          # default; or redis://… amqp://… kafka://…
metrics_store: file                # default; or http://host:8428 (VictoriaMetrics)

stores:                            # other stores, read in place through an adapter (section 4)
  - name: chronosphere
    command: orchestra-prometheus https://tenant.chronosphere.io/data/metrics --token-env CHRONOSPHERE_TOKEN
    match: '{__name__=~"payments_.*", env="prod"}'    # required: every query stays inside it
    kinds: {"*_inflight": level, "*_bucket": ignore}  # flow | level | ignore (default: by the name)
    names: {"payments_requests_total": "{service} requests"}   # renaming rules: glob -> template
    split: [service]               # one series per value of these labels
    per_day: 2000                  # requests a day at most (default 5000); rate: 1/s by default

machines:                          # what Orchestra runs (section 3)
  - name: github                   # pull requests, reviews and checks as facts
    command: .orchestra/github.py
    every: 2m
  - name: ci_triage                # a red check: caused by the change, or flaky?
    command: orchestra decide "Is this failure caused by the change?" --answers real,flaky --inputs title,log_tail
    on: [github.check.failed]
  - name: fixer                    # a real failure: an agent pushes a fix to the pull request
    command: orchestra agent .orchestra/prompts/fixer.md --model sonnet --per-day-usd 15
    on: [ci_triage.real]
  - name: reviewer                 # one session that keeps its context across comments
    command: orchestra agent .orchestra/prompts/reviewer.md --per-day-usd 10
    on: [github.pr.review_requested, github.pr.commented]
    continuous: true
  - name: deploys                  # a program kept running: follows the deploy log
    command: .orchestra/deploy_tail.py
    continuous: true
    env: {DEPLOY_ENV: prod}

projections:                       # what is looked at, built from the facts (section 6)
  - name: review                   # Which pull requests wait on a review, since when?
    match: "acme/api/pull/*"
    command: .orchestra/review.py
```

Also optional: `discovery: {every: 5m, window: 60m}` (section 7).

## 2. Lines

A machine writes one JSON object per line on stdout. A line with `type` is a **fact**, a line
with `metric` and `value` a **measurement**, a line with `cost_usd` a **run line**, and anything
else goes to the machine's log (as does its stderr).

```json
{"type": "github.pr.opened", "subject": "acme/api/pull/12", "source": "person.github.alice",
 "id": "gh:acme/api/pull/12:opened", "time": "2026-11-05T10:00:00Z", "data": {"title": "Fix retries"}}
{"metric": "queue_depth", "value": 12, "labels": {"queue": "triage"}, "kind": "gauge"}
{"cost_usd": 0.12, "note": "optional: a word about the run"}
```

A fact is something a `source` asserts about a `subject`, true as of its `time`; it is never
edited (a correction is a new fact). Its type's verb says which: an **event**, that something
happened (`jira.issue.status_changed`); an **observation**, that something was so when looked at
(`jira.issue.observed`); a **decision**, that a machine or person judged something
(`ci_triage.real`). The keys: `type` (required, `<system>.<noun>.<verb>`); `subject`, the **item**
it is about (a ticket, a pull request, a task: whatever the facts about one id fold up into), named
as its own system names it (`ABC-123`, `acme/api/pull/3`; other things it concerns are
named in `data` by their ids); `source`, who asserts it (default the machine; people are
`person.<system>.<account>`); `id`, stable per change (a fact whose id the bus has is dropped;
default a hash of the line); `time`, when the source says it was true (RFC 3339, default now);
`kind`, what the item is (default the type less its verb: `github.pr`); `data`, the record as
the source gives it (Orchestra reads it only for `latest`); `causedby`, the id of the fact that
caused it, when the line knows better than Orchestra (section 3).
`{"type": "orchestra.actor", "source": …, "data": {"name": …}}` names an actor for the pages.

A **repeat** is an observation whose `data` (canonical JSON) equals the previous fact's with the
same item, type and source, in log order. Publishing skips one when it can tell (counted as
`repeats` on the run record), and every path by which facts reach a projection, a machine or a
query skips any that got past, and gives each fact its `seq`, its position in the log. So a poller
may print the whole current state on every run and keep no state of its own. A repeat still says
when the state was last seen unchanged: the next observation that differs carries `prevseen`, the
time of the last repeat skipped before it (kept in the state directory across restarts), so the
change happened between `prevseen` and its `time`. In `data`, `omitted` lists fields a collector
left out on purpose (a body, HTML); `truncated: true` says a list was not read to its end, and
`orchestra query check` reports it.

A measurement: `metric` (required, named for what it measures), `value` (required, a number),
`labels`, `time` (RFC 3339 or Unix seconds; default now), `kind` (`counter` or `gauge`; default
`_total` and `_count` are counters). Every point is labelled `machine` and `project`; a series
named `orchestra_*` is refused. A once run whose first line is Prometheus text is read as
Prometheus text, written when the run succeeds.

A run line says what the run cost (`cost_usd`), with an optional `note` and `skipped: true` for a
run that did no work. Programs elsewhere may `POST` a fact's JSON to the facts endpoint
(`/events`), which `orchestra run` prints with its token when it starts. The bus keeps every fact
in the CloudEvents envelope (`specversion`, `id`, `source`, `type`, `time`, `subject`, `data`) with
the extensions `causedby` and `depth` (section 3) and `recordedtime`, when the bus recorded it.

## 3. Machines

A machine has four fields: `name`, `command` (run without a shell, in the repository), when it
runs, and an optional `env` map added to its environment.

- **once** (the default): one run per fact of the types in `on` (globs allowed), the fact (with its
  `seq`) as one line on stdin and stdin then closed; with `every`, a run on that schedule with
  nothing on stdin (a tick is skipped while the last one runs). Both may be given. One run at a
  time, facts waiting in order; a run is stopped after an hour.
- **`continuous: true`**: started when `orchestra run` starts and restarted when it exits; the
  facts in `on` arrive on its open stdin as they come, one line each. It reads views with
  `orchestra query views`.

A machine never gets its own facts. Every run gets `ORCHESTRA_MACHINE` and `ORCHESTRA_SOURCE`
(its name, what `orchestra emit` publishes as), `ORCHESTRA_CONTINUOUS` (`1` when continuous),
`ORCHESTRA_PROJECT`, `ORCHESTRA_REPO` and `ORCHESTRA_RUN`, in the owner's environment, unstripped.
A once run for a fact also gets `ORCHESTRA_FACT` (the fact) and `ORCHESTRA_VIEWS_FILE`, a file of
`{projection: view}` for its item (`latest` included, and a whole projection's view when its key
is the item), kept as `runs/<id>.views.json` (the newest 500) so the views a run saw stay
readable; before the run starts its item's per-item views are built to at least its fact,
and whole views are read as built, with `behind`. For one release `ORCHESTRA_EVENT` is still set,
and `ORCHESTRA_ITEM` holds `latest` without `by_type`, plus `events` (the count), empty past 32 KB.

**Causation.** Every fact a run writes is stamped in the envelope, never in `data`, with
`causedby` (the triggering fact's id) and `depth` (its cause's plus one). A once run's lines are
stamped as Orchestra reads them, and `orchestra emit` in it reads `ORCHESTRA_CAUSEDBY` and
`ORCHESTRA_DEPTH`. A continuous machine's lines are stamped with the oldest fact it was given and
has not answered (exact for `orchestra agent`, which takes one at a time); a line may name
`causedby` itself. Facts from the endpoint and from scheduled runs have depth 0. A fact with
`depth` over 8 is recorded but **held**: no machine gets it, and the run record and the Core page
say so. That stops loops among machines; a loop through an outside system starts again at depth 0
and is bounded only by `--per-day-usd`.

Every run is recorded (`orchestra query runs`): start, end, exit code, facts, points, cost, note,
log, and `repeats`, `restated` (facts the log already had) and `held` when there were any. A
continuous machine's process is one record, and each run line it prints ends a **turn**, a record
of its own for the oldest fact it was given and not yet answered.

Two programs come with Orchestra, named in `command`; they read stdin and write lines like any:

- `orchestra agent <prompt file> [--model M] [--per-day-usd N]`: Claude Code (`claude -p`, with
  the owner's login, shell and credentials, `--dangerously-skip-permissions`) on the prompt file
  and each fact with its item: the item's last 10 facts, one per line, each view other than
  `latest` in full up to 2 KB (else named only, 8 KB in all), and `latest`'s `fields` and `last`,
  with the views file and `orchestra query facts --subject S` named for the rest. Once: one turn
  per fact. Continuous: one stream-json session, each fact a user message, so it keeps its
  context. The JSON lines in what it says are its facts and metrics; it may also run
  `orchestra emit '<json>'`. Each turn ends with a run line of its cost; past `--per-day-usd`
  spent today a turn is not started (`skipped`). What it may do is written in its prompt file, or
  in scripts beside it that it runs.
- `orchestra decide "<question>" --answers a,b,c [--inputs proj:field,field] [--min-confidence 0.6]
  [--model M]`: one model call per fact with the question, the fact and the `inputs` of its
  item's views (`ticket:status` is the `ticket` projection's `status`, a bare `status` is
  `latest`'s `fields.status`; default all of `latest`'s fields); it prints `decision.made`
  (`answer`, `confidence`, a one-line `reason`) and `<machine>.<answer>` (so `on: [ci_triage.real]`
  reads plainly), and a run line. Below `--min-confidence` the answer is `unsure`.

The Machines page shows every machine with its runs, and every `source` heard on the bus that
Orchestra doesn't run (people, other programs).

## 4. Stores

A store is read in place, never copied or written: its `command` is an **adapter**, a program
Orchestra keeps running and asks one request at a time. A store has `name`, `command`, `match`
(required; a match-everything selector is refused) and optionally `names`, `kinds`, `split`,
`rate` (default `1/s`), `per_day` (default 5000) and `env`.

Orchestra does everything around every adapter: each query is conjoined with `match` and needs a
selector of its own, reaches back a day at most and selects 500 series at most (counted with a
`series` request first); requests wait for `rate`, stop at `per_day` until 00:00 UTC, and are
logged (`<state>/stores/<name>.requests.jsonl`); an adapter that gives no answer in 60 s is killed,
and one that has exited is started again on the next request. `kinds` (`flow` read as a counter,
`level` as a gauge, `ignore`), `names` (`{label}` and `{metric}` in the template) and `split` are
applied to the answers. `orchestra-prometheus <url> [--token-env NAME] [--header NAME]` is the
adapter for any Prometheus HTTP API, a path prefix in the URL allowed.

### Writing a store adapter

An adapter reads one JSON request per line on stdin and answers each on stdout, in order, flushing
after each answer. Its stderr goes to `<state>/logs/stores/<name>.log`. It asks the store what it is
told; it never widens a query (Orchestra has already bounded it).

| request | answer lines |
|---|---|
| `{"op": "series", "match": "<selector>"}` | `{"metric": "<name>", "labels": {"<label>": "<value>", …}}`, one per series the selector selects |
| `{"op": "range", "query": "<selector or PromQL>", "start": <unix s>, "end": <unix s>, "step": <s>}` | `{"metric": "<name or empty>", "labels": {…}, "points": [[<unix s>, <number>], …]}`, one per result series |
| `{"op": "health"}` | `{"ok": true}`, or `{"ok": false, "error": "<why>"}` |

Every answer ends with the line `{"end": true}`. A request that fails is answered with one
`{"error": "<why, never a credential>"}` line, then `{"end": true}`. `labels` exclude the name;
`points` are in time order, numbers only (leave out NaN). The adapter takes its credentials from
environment variables named on its command line (`--token-env NAME`), never from the command line
itself, and keeps running until its stdin closes. Try it by hand:
`echo '{"op": "health"}' | <command>`.

## 5. Commands and pages

- `orchestra run [--web]`: makes the state directory (`~/.orchestra/<project>/`), loads and checks
  the config (refusing to start on an error: a config that doesn't load, a command not found, an
  `orchestra agent` prompt file missing; saying warnings: a store not answering), prints the facts
  endpoint and its token, and runs every machine; `--web` also serves the pages and prints their
  login link.
- `orchestra chat [message] [--continue]`: the Supervisor, a Claude Code session in the repository
  briefed with this spec and what is set up. It answers about Orchestra and what is happening,
  changes it when asked (the config, its scripts, projections and prompts, proved by running
  them), and clears the data when asked
  (`python -m orchestra.reset --yes`: the bus, metrics, views, runs, logs and model; never the
  config, tokens or scripts); with a config written for an earlier Orchestra (`items:`), it
  rewrites it first. With no config it sets nothing up of its own accord. `--continue` carries on
  the last session.
- `orchestra emit '<json>'`: one fact or metric line, as `ORCHESTRA_SOURCE` (default `cli`).
- `orchestra query facts [-n N] [--type T] [--subject S] [--trace ID]` (`--trace` walks `causedby`
  both ways); `query views [PROJECTION] [show SUBJECT] [--input SUBJECT] [--rebuild]` (`--input`
  prints exactly what the projection is given, every matching fact for a whole one, so
  `orchestra query views review --input acme/api/pull/3 | review.py` is its test); `query check`;
  `query metrics [prefix]`, `query model`, `query runs [machine]`, and `query '<promql>'
  [--store NAME]` (the metrics store, or a store through its adapter). `query events` and
  `query items` are refused with the new names.
- `orchestra query check` reports what can be checked from the log alone: a source whose facts'
  `time` always equals `recordedtime` (it stamps now, not the source's time); `data` over 4 KB; a
  type whose keys stopped appearing or newly appeared; a collector silent for more than three of
  its `every`, or whose runs keep adding facts where nothing changed; each projection run twice
  over its facts: a missing or `error` view, or different bytes the second time.

The bus counter keeps its name, `orchestra_events_total`, and counts facts.

Pages, read only: **Dashboard** (what waits on the owner, open items, recent runs, spend),
**Items** (each item's facts and views), **Machines** (every machine and source, what each
publishes and listens for, runs), **Dynamics** (the model), **Core** (bus, metrics store, stores,
machines, each projection's last build, held facts: health and recent errors). The Dashboard and
Items take `waiting_on` and `done` from an item's first view, in config order, that has the key
at its top level, else from `latest` (`fields.waiting_on`, `done`).

## 6. Projections and views

What a team looks at (a ticket's state, a review queue, time in status) is a **view**, printed by a
**projection**: a program that reads facts on stdin and prints views. Views are rebuilt, never
logged, so the log holds only facts.

```yaml
projections:
  - name: ticket                     # Where is each ticket, and who has it?
    match: "ABC-*"                   # subject glob, or a list of them
    command: .orchestra/ticket.py
  - name: ticket_prs                 # Which tickets have an approved pull request?
    match: ["ABC-*", "acme/api/pull/*"]
    types: ["jira.issue.*", "github.pr.*"]   # optional: only these reach it
    whole: true                      # reads every matching fact on each build
    command: .orchestra/ticket_prs.py
```

Each build is one run of the command. Its stdin is every fact of the items being built (all
matching items the first time, then only those with new facts), in log order, each with its
`seq`; it prints one JSON line per item, `{"subject": …, …}`. One that needs the source's order
sorts by `time`. A `whole: true` projection gets every matching fact on each build and prints one
line per key it chooses (`{"key": …}`); its build replaces its views as a set, so a key it no
longer prints is gone. A projection reads only its stdin (it is run with no environment but `PATH`,
`HOME` and the locale) and prints times, not ages, so it can be rerun at will. Any command will do
(`duckdb -jsonlines -c "<query>"` too).

**`latest`**, built in, covers every item, in Orchestra's own process: `fields`, every key of
every fact's `data`, the latest fact winning, `null` removing a key; `by_type`, the latest fact's
`data` of each type; `type`, the latest fact's; `first` and `last`, the times of the first and latest
facts; `kind` (section 2); `facts`, how many; `done`, true once the latest type's verb is
`completed`, `closed`, `cancelled` or `merged`. It orders by `time`, ties by `seq`, so history backfilled after
today's snapshot doesn't become the current state.

**One builder, one lock.** Views are built by one function, under a file lock in the state directory,
each run of a projection under a 60 s timeout that kills its process group. Each projection's views
are kept in `views/<name>.json` with `built_to` (the last `seq` built) and the projection's hash: its
entry, the script `command` names and the modules beside it that it imports (none for an inline
command), and Orchestra's version. A changed hash rebuilds it from the log; `orchestra query views --rebuild`
forces it. `orchestra run` builds after each batch of facts; a page or `query` builds only when no
runner is alive. Every view carries `behind`: how many matching facts have a `seq` past `built_to`
(for a per-item view, its item's only). Routing facts to machines never waits for a build, and
a build triggers no machine. An item the program printed no line for, or a run that failed or
timed out, keeps its last good view with `error` and `error_at`, never `latest` in its place.

## 7. The model

Discovery runs every `discovery.every` (default 5m) over `discovery.window` (default 60m) of the
metrics: the store's own series, every store's series within its `match` (through its adapter), and
Orchestra's count of the facts on the bus, `orchestra_events_total{type, source}`. It finds stocks and flows,
fits their equations, saves a version when the structure changes, and guesses which machines move
each flow from timing. A **prediction** is a fact `orchestra.prediction.made` (`data`: the stock
or flow, the expected direction or value, by when); Orchestra judges it when its time comes and
publishes `orchestra.prediction.judged`.

## 8. What is not here

No plugins, no tools section, no machine kinds, no declared model, no consent or confirmations, no
workflows engine, no planners, no outbox, no TOML. A team's way of working is which machine listens
for which facts, in the config file, and the scripts, projections and prompts it names. Anything more is added
only when a real flow needs it.
