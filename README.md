# Orchestra

A dumb system: an event bus, a metrics store, commands whose stdout it reads, items folded from
events, processes it starts when the events they name arrive, and a system dynamics model guessed
from the metrics. The whole design is [SPEC.md](SPEC.md), one page.

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
orchestra init         # the state directory, ~/.orchestra/<project>/
orchestra setup        # a Claude Code session that asks what you want, then writes and tests
                       # orchestra.yaml, its scripts and its prompts; or: orchestra setup "watch our Jira"
orchestra run --web    # everything, and the pages; Ctrl-C stops it
orchestra chat         # talk to the Supervisor: what's happening, and changes; --continue resumes
orchestra reset        # clear the data and start over (keeps the config); --yes to do it
```

The config stays short and readable, so you can edit it by hand too. Secrets come from environment variables named in the config (`token_env: JIRA_TOKEN`). An agent
process runs Claude Code with your own login and environment.

## Read

```bash
orchestra events -n 20          # the latest events
orchestra items                 # open items; orchestra items show <subject>
orchestra runs                  # process runs
orchestra metrics               # series in the metrics store
orchestra model                 # the discovered model
```

## Develop

```bash
uv sync --extra dev             # --extra sd for the Kalman discovery
uv run python -m pytest -q
uv run ruff check
```
