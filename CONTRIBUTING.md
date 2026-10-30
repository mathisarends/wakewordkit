# Contributing

WakewordKit uses [uv](https://docs.astral.sh/uv/) and supports Python 3.12+.

## Setup

```bash
uv sync --locked --dev
uv sync --locked --dev --extra examples  # optional hardware/provider examples
```

## Checks

CI runs these on Linux and Windows with Python 3.12, 3.13 and 3.14:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest
uv run python tools/wheel_smoke.py
```

Use `uv run ruff format .` to apply formatting. Ruff is pinned to the same version
in the dev group and pre-commit hooks. Optionally run `uvx pre-commit install`.
The package version has one source: `wakewordkit.__version__`, read by Hatch.

## Tests and contracts

Unit tests use a fake backend that requires real 1,280-sample model blocks.
`uv run pytest -m integration` runs real inference and checks frame partitioning,
backend/VAD reset against fresh instances and model compatibility.
`uv run pytest -m "not integration"` skips those integration tests.
Resampling tests exercise the actual filter, passband and alias rejection.
Concurrency/device tests use controlled worker barriers and mocked hardware.

`tools/wheel_smoke.py` builds a wheel, installs it with its dependencies into a
fresh temporary environment and runs inference outside the source tree. It
checks model assets and package version consistency, then deletes the temporary
environment. It requires uv and package-index access on first use.

Strict mypy covers the package and a public API consumer. Update the consumer
when changing public types. Preserve the single ordered stream contract and
add a focused regression test for behavior changes.

See [docs/EVALUATION.md](docs/EVALUATION.md) before making recognition-quality or
hardware-performance claims. Real positive and negative recordings are required;
passing synthetic/silence engineering tests alone cannot establish those claims.

See [docs/RELEASING.md](docs/RELEASING.md) for isolated release preparation and publication.
