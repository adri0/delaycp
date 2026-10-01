# AGENTS.md

I'm building `delaycp`, a Python package for online conformal prediction on
streaming binary classification with delayed, class-imbalanced labels (fraud,
chargebacks, credit default). It wraps any model that outputs p = P(fraud).

## Project setup

- Python >= 3.11, src layout (`src/delaycp/`), runtime dependency: numpy only.
- Dev tools: pytest, hypothesis, ruff, mypy (strict on `src/`). Full type
  hints, NumPy-style docstrings, no global state, deterministic given a seed.
- Run tests, mypy, and ruff via `uv run` (not `uvx`) so they see the
  project's installed dependencies (e.g. numpy) instead of an isolated env:
  `uv run pytest`, `uv run mypy src`, `uv run ruff check`.

## Domain semantics

- Labels: `0` = legit, `1` = fraud. Inputs are `p_fraud` floats in `[0, 1]`.
- Nonconformity score: `s(x, y) = 1 - p_hat(y | x)`, so `s(x, 1) = 1 - p` and
  `s(x, 0) = p`.
- Each class has its own threshold `q_y` on its score:
  - "fraud" is in the set iff `1 - p <= q_1`, i.e. `p >= t_low := 1 - q_1`.
  - "legit" is in the set iff `p <= q_0 =: t_high`.
- Actions: set `{0}` -> APPROVE, `{1}` -> DECLINE, `{0,1}` -> REVIEW. An empty
  set (`t_low > t_high` and `p` between them) -> REVIEW (conservative).
- Miscoverage for a labelled example uses the threshold **in effect when the
  prediction was made**, not the current one. This matters because labels are
  delayed.

## Key references

- Gibbs & Candes 2021 (ACI)
- Angelopoulos, Candes & Tibshirani 2023, "Conformal PID Control for Time
  Series Prediction" (reference code:
  github.com/aangelopoulos/conformal-time-series, `core/methods.py`)
- El Halabi & Brandt 2026, arXiv 2609.07251 (ACI under delayed feedback)

## Working on this repo

Write code plus tests. Keep each public class small and documented. If
something in the task is ambiguous or seems mathematically wrong, say so
before implementing rather than guessing silently.
