# Agent instructions

## Project Overview

khalib is a Python library for binary and multiclass classifier probability
calibration using Khiops' MODL regularized histograms.

## Architecture

- `src/khalib/main.py` — Library implementation: `Histogram`, `KhalibClassifier`,
  `calibrate_binary`, `calibration_error`, and `build_reliability_diagram`.
- `src/khalib/__init__.py` — Public API exports.
- `src/tests/test_all.py` — Tests, with fixtures and reference data in `src/tests/data/`.
- `docs/` — Sphinx documentation.

## Toolchain

- Python >= 3.11, < 3.15; setuptools and uv.
- Runtime: khiops >= 11.0.1.0, numpy >= 2.0.0, scikit-learn >= 1.5.0,
  matplotlib >= 3.9.0.
- Checks: pytest and Ruff (via prek); docs: Sphinx + MyST.

## Development Commands

- **Run tests**: `uv run pytest src/tests`
- **Run fast tests**: `uv run pytest src/tests -m "not slow"`
- **Lint**: `uvx prek run --all-files`
- **Build docs**: `cd docs && uv run sphinx-build --nitpicky . _build/html`
- **Task runner**: `just test`, `just docs`, `just prek` (alias: `just pk`)

## Release and CI Practices

- Publishing is intentionally manual and tag-only for both PyPI and docs.
  Do not add automatic release triggers.
- Distribute source packages only; do not add wheels.
- Pin GitHub Actions to full commit SHAs with readable version comments.
  Dependabot checks weekly with a 30-day version-update cooldown.
- Give CI jobs explicit least-privilege token permissions; restrict write
  permissions to publishing jobs.
- Follow Keep a Changelog conventions. Focus release notes on user-facing
  changes and keep CHANGELOG.md lines within 100 characters.
- Keep pyproject.toml and uv.lock versions synchronized without unrelated
  dependency updates.
- Commit only requested changes using the configured Git identity.
  Do not add Copilot co-author trailers.

## Working Practices

- Always use `uv`, including for ad hoc scripts: `uv run --no-project python`,
  never bare `python` or `python3`.
- Use repository-pinned tooling; run targeted checks with
  `uvx prek run --files ...`.
- Test incrementally: start with affected tests, then run the non-slow suite
  after shared-validation changes. Use `MPLBACKEND=Agg` for headless plotting.
- Audit related surfaces together. For validation bugs, check histogram
  factories, scalar/vector lookup, binary calibration, ECE, plotting, and
  classifier integration before proposing separate fixes.
- Bound conclusions to the code and behavior actually examined.
- Scope history rewrites from the branch's merge-base, not an arbitrary recent
  commit. Inspect the complete commit range, preserve a backup, and verify exact
  tree identity afterward.
- Keep research artifacts untouched. Store temporary scripts and validation
  worktrees outside the repository, and remove only artifacts created for the
  task.

## Coding Style

- Follow existing scikit-learn API patterns and NumPy-style docstrings.
- Use modern type hints: `list[float]`, `X | None`.
- Structure code into a few coherent paragraphs, with short, high-level comment
  headers for nontrivial logic. Do not narrate individual statements.
- Separate paragraphs with one blank line, but do not add one immediately after
  a docstring or between a control statement and its body.
- Standalone returns, loop-variable assignments, and trivial methods need no
  header comment.

## Test Design

- Use pytest fixtures and parametrization; mark slow tests with `@pytest.mark.slow`.
- Optimize for readability, not line count. Keep small repetitions when a helper
  or fixture would obscure intent; stop refactoring when the benefit is marginal.
- Make cases declarative. Prefer named parameter cases and explicit expected
  values over string-dispatch branches and conditional expectations.
- Preserve coverage deliberately. Retain input representations, dtypes, boundary
  cases, error assertions, and state-preservation guarantees, not merely the test
  count.
- Control fixture cost and state. Build expensive datasets lazily; share
  deterministic, read-only data; create fresh estimators per case and copy arrays
  before mutation.
- Reduce Cartesian products carefully. Separate input-validation coverage from
  mode coverage only when evidence shows the combinations exercise the same
  validation path.

## Domain and Integration

- **Histogram**: Discretizes scores into bins with optional per-class frequencies
  and probabilities. Khiops intervals are left-open and right-closed.
- **ECE (Expected Calibration Error)**: Measures agreement between scores and
  observed class frequencies.
- **MODL**: Khiops' regularized histogram method, which automatically estimates
  bin boundaries without requiring a number of bins or cross-validation.
- **Dirac heuristic**: Detects point-mass score distributions for visualization.
- Khiops runs as an external subprocess; `single_core_khiops_runner()` manages
  its execution context.
- `Histogram.target_probas` is derived from target frequencies divided by bin
  frequency. For histograms with target statistics, nonempty bins sum to one;
  empty bins contain zero probabilities.
- Supervised histogram extraction expects `target_values` in the Khiops
  preparation report.
- `KhalibClassifier` follows the scikit-learn estimator interface through
  `ClassifierMixin`, `MetaEstimatorMixin`, and `BaseEstimator`.
