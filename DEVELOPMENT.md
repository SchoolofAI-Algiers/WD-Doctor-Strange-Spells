# Development Setup Guide

## Quick Start

```bash
# 1. Install dependencies + pre-commit hooks
pip install -e ".[dev]"
pre-commit install

# 2. On every commit: auto-formats, lints, type-checks
# 3. On push: runs unit tests (pre-push hook)
```

## Available Commands

| Command         | Description                                      |
| --------------- | ------------------------------------------------ |
| `make test`   | Run unit tests (`pytest tests/unit -v`)        |
| `make lint`   | Lint + typecheck (`ruff check . && mypy src/`) |
| `make format` | Auto-format code (`ruff format .`)             |
| `make check`  | Lint + format check (CI gate)                    |
| `make run`    | Run the application (`python -m src.main`)     |

## Pre-commit Hooks

Runs automatically on `git commit`:

- **Ruff:** lint + auto-fix + format
- **MyPy:** strict type checking
- **Whitespace:** trim trailing, ensure newline at EOF
- **YAML/TOML:** validate config files
- **Large files:** prevent >500KB commits
- **Merge conflicts:** detect unresolved conflicts

Runs on `git push`:

- **Unit tests:** `pytest tests/unit -v`

## Manual Override

```bash
# Skip hooks (emergency only)
git commit --no-verify

# Run all hooks manually
pre-commit run --all-files
```

## IDE Setup (VS Code)

Install extensions: **Ruff**, **MyPy**, **Python**. Settings auto-configured via `pyproject.toml`.
