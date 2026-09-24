# Contributing Guidelines

## Branch Naming Conventions

### Format
```
<type>/<short-description>
```

### Types
| Type | Purpose | Example |
|------|---------|---------|
| `feat/` | New feature | `feat/palm-orientation-calculation` |
| `fix/` | Bug fix | `fix/landmark-flicker-multi-hand` |
| `refactor/` | Code restructuring | `refactor/separate-detection-render` |
| `perf/` | Performance optimization | `perf/cache-spell-assets` |
| `docs/` | Documentation only | `docs/update-subsystem-breakdown` |
| `test/` | Test additions/changes | `test/geometry-calculator-unit` |
| `chore/` | Maintenance, config, deps | `chore/update-mediapipe-version` |
| `asset/` | Asset additions | `asset/add-glow-particle-texture` |

### Rules
- Use **kebab-case** for descriptions
- Keep descriptions short (3-5 words max)
- One branch per logical change
- Delete branch after merge

---

## Commit Message Format (Conventional Commits)

### Format
```
<type>(<scope>): <short summary>

<body (optional)>

<footer (optional)>
```

### Types
| Type | Description |
|------|-------------|
| `feat` | New feature |
| `fix` | Bug fix |
| `refactor` | Code change that neither fixes nor adds features |
| `perf` | Performance improvement |
| `docs` | Documentation only |
| `test` | Adding/modifying tests |
| `chore` | Maintenance, dependencies, build config |
| `style` | Formatting, linting (no logic change) |
| `asset` | Adding/modifying assets |

### Scopes (use lowercase, singular)
`capture`, `tracking`, `geometry`, `gestures`, `motion`, `rendering`, `pipeline`, `assets`, `docs`, `tests`, `ci`, `config`

### Examples
```bash
# Feature
feat(geometry): add palm orientation calculation using MCP landmarks

# Fix with body
fix(tracking): resolve landmark jitter on low-light frames

Add temporal smoothing to landmark positions using exponential moving average.
Reduces flicker by 60% in testing.

Closes #12

# Refactor
refactor(pipeline): separate detection and rendering into independent loops

# Perf
perf(rendering): cache spell assets at startup instead of per-frame load

# Docs
docs(subsystems): add motion analysis subsystem breakdown

# Test
test(geometry): add unit tests for palm center calculation
```

### Rules
- **Summary line**: Max 72 chars, imperative mood ("add" not "adds")
- **Body**: Wrap at 72 chars, explain *what* and *why*, not *how*
- **Footer**: Reference issues (`Closes #123`, `Refs #456`)
- **One logical change per commit** — split if needed

---

## Pull Request Process

### Before Opening PR
- [ ] Branch follows naming convention
- [ ] Commits follow Conventional Commits
- [ ] Code passes lint: `make lint` (or `ruff check .`)
- [ ] Code passes typecheck: `make typecheck` (or `mypy src/`)
- [ ] Tests pass: `make test` (or `pytest`)
- [ ] No debug prints, commented code, or `TODO` without issue reference

### PR Title Format
Same as commit summary: `<type>(<scope>): <summary>`

### PR Description Template
```markdown
## Summary
Brief description of changes.

## Motivation
Why is this needed? Link issue if applicable.

## Changes
- Bullet list of key changes
- Include subsystem(s) affected

## Testing
- How was this tested?
- Any manual verification steps?

## Screenshots/Videos (if visual)
- Before/after or demo

## Checklist
- [ ] Lint passes
- [ ] Typecheck passes
- [ ] Tests pass
- [ ] No breaking changes (or documented in footer)
- [ ] Documentation updated if needed
```

### Review Requirements
- **Minimum 1 approval** from team member
- **All CI checks passing**
- **No unresolved review comments**

### Merge Strategy
- **Squash and merge** for feature/fix branches
- **Rebase and merge** for long-running refactor branches (with maintainer approval)
- Delete source branch after merge

---

## Code Style

### Python
- **Formatter**: `ruff format` (or `black`)
- **Linter**: `ruff check` (or `flake8` + `isort`)
- **Type Checker**: `mypy --strict`
- **Line Length**: 100 chars
- **Target**: Python 3.10+

### Import Order (ruff default)
1. Standard library
2. Third-party
3. Local (`src.`)

### Naming
| Element | Convention |
|---------|------------|
| Modules, packages | `snake_case` |
| Classes | `PascalCase` |
| Functions, methods, variables | `snake_case` |
| Constants | `UPPER_SNAKE_CASE` |
| Private (module/class) | `_leading_underscore` |

### Docstrings
- **Google style** for public APIs
- Minimum: module, class, public function signatures

---

## Testing

### Structure
```
tests/
├── unit/           # Pure unit tests (no CV, no I/O)
│   ├── test_geometry.py
│   ├── test_gestures.py
│   └── ...
├── integration/    # Multi-component tests
│   └── test_pipeline.py
└── fixtures/       # Test images, landmarks, expected outputs
```

### Requirements
- **Unit tests**: Required for all new logic in `geometry/`, `gestures/`, `motion/`, `rendering/`
- **Coverage target**: ≥80% for new code
- **Run locally**: `pytest tests/unit -v`
- **CI runs**: Unit + integration on every PR

---

## Performance Standards

| Metric | Target |
|--------|--------|
| FPS (detection) | ≥30 on mid-range laptop |
| FPS (full pipeline) | ≥24 |
| Latency (frame-to-display) | <100ms |
| Memory | <500MB steady state |

Profile before optimizing. Use `make profile` (or `python -m cProfile -o profile.stats -m src.main`).

---

## Issue Labels

| Label | Purpose |
|-------|---------|
| `bug` | Something broken |
| `enhancement` | New feature or improvement |
| `performance` | Perf-related |
| `documentation` | Docs only |
| `good first issue` | Beginner-friendly |
| `help wanted` | Needs contributor |
| `blocked` | Waiting on dependency |
| `wontfix` | Closed without fix |

---

## Getting Help

- **Architecture questions**: Open a `discussion` or tag `@maintainer` in PR
- **CV/Math questions**: Check `docs/subsystems/` first
- **MediaPipe issues**: Check [MediaPipe FAQ](https://github.com/google/mediapipe/blob/master/docs/faq.md)

---

## Attribution for Assets

When adding new assets, update [`ASSET_ATTRIBUTION.md`](ASSET_ATTRIBUTION.md) with:
- Source URL
- License
- Author
- Any modifications made