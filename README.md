# Orchestra

A dumb system: an event bus, a metrics store, machines it runs (commands that read events as JSON
lines on stdin and write events and metrics as JSON lines on stdout), other stores read through
adapters, items folded from events, and a system dynamics model guessed from the metrics. The whole
design is [SPEC.md](SPEC.md).

## Install

```bash
gh auth setup-git      # the repository is private
uv tool install --force 'orchestra @ git+https://github.com/danroblewis/orchestra-mini@main'
# with the Kalman discovery extra:
uv tool install --force 'orchestra[sd] @ git+https://github.com/danroblewis/orchestra-mini@main'
```

## Use

```bash
cd <your repository>
orchestra chat         # the Supervisor: with no config it sets the project up (or: orchestra chat "watch our Jira");
                       # then it answers about what is happening, changes things, clears the data; --continue resumes
orchestra run --web    # checks the config, runs every machine, serves the pages; Ctrl-C stops it
orchestra emit '{"type": "deploy.release.done", "subject": "v1.4"}'
orchestra query items  # also: events, metrics, model, runs, or '<promql>' [--store NAME]
```

A machine is four fields, so the config stays readable by hand:

```yaml
machines:
  - name: ci_triage
    command: orchestra decide "Is this failure caused by the change?" --answers real,flaky
    on: [github.check.failed]
  - name: fixer
    command: orchestra agent .orchestra/prompts/fixer.md --per-day-usd 15
    on: [ci_triage.real]
```

`orchestra agent` and `orchestra decide` are programs that come with Orchestra; any other command
works the same way. An agent runs Claude Code with your own login and environment.

## Develop

```bash
uv sync --extra dev             # --extra sd for the Kalman discovery
uv run python -m pytest -q
uv run ruff check
```
