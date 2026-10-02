# delaycp

[![CI](https://github.com/adri0/delaycp/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/adri0/delaycp/actions/workflows/ci.yml)
[![Tested on Python 3.11 | 3.12 | 3.13](https://img.shields.io/badge/tested%20on-3.11%20%7C%203.12%20%7C%203.13-blue?logo=python&logoColor=white)](https://github.com/adri0/delaycp/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/github/license/adri0/delaycp)](LICENSE)
[![Status: pre-alpha](https://img.shields.io/badge/status-pre--alpha-orange)](#delaycp)
[![Dependencies: numpy](https://img.shields.io/badge/dependencies-numpy-013243?logo=numpy)](pyproject.toml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with mypy (strict)](https://img.shields.io/badge/mypy-strict-2a6db2)](https://mypy-lang.org/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

**Status: pre-alpha.** APIs are unstable and may change without notice.

`delaycp` wraps any probabilistic classifier and produces prediction sets with coverage guarantees that hold in production conditions: labels that arrive days or months late, rare positive classes, and distributions that drift or are adversarial. Think of credit card chargebacks, transaction alerts where the true label might be very delayed, or similar situations.

It implements conformal PID control (Angelopoulos, Candès & Tibshirani, 2023) with per-class (Mondrian) thresholds, delay-aware updates following El Halabi & Brandt (2026), and optional label-shift reweighting (Podkopaev & Ramdas, 2021). Transactions whose prediction set contains both classes are flagged as uncertain and can be sent to manual review, with tools to tune coverage against review budget. The package includes metrics for per-class and rolling coverage and simulators that add realistic chargeback delays to public datasets such as IEEE-CIS and BAF. It follows scikit-learn and River conventions and needs only NumPy.

## Installation

### Using `uv`

```bash
git clone https://github.com/adri0/delaycp.git
cd delaycp
uv sync --extra dev
```

### Using `pip`

```bash
git clone https://github.com/adri0/delaycp.git
cd delaycp
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Development

```bash
ruff check .
mypy src
pytest
```
