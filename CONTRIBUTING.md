# Contributing

Thanks for helping improve wakewordkit! This project uses
[uv](https://docs.astral.sh/uv/) and targets Python 3.14+.

## Setup

```bash
uv sync --dev          # core + dev tools
uv sync --extra mic    # add microphone support to try the examples
```

## Checks

Run these before opening a pull request — CI runs the same commands:

```bash
uv run ruff check .
uv run ruff format .
uv run pytest
```

Optionally install the pre-commit hooks to run linting and formatting on every
commit:

```bash
uvx pre-commit install
```

## Tests

Tests cover logic, not model behavior — the openWakeWord model is faked so they
stay fast and deterministic (see `tests/conftest.py`). New logic should come
with a test; please keep them focused rather than chasing full line coverage.

## Pull requests

Keep changes small and focused, make sure the checks above pass, and describe
what changed and why.
