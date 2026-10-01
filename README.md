# delaycp

**Status: pre-alpha.** APIs are unstable and may change without notice.

`delaycp` wraps any probabilistic classifier and produces prediction sets with coverage guarantees that hold in production conditions: labels that arrive days or months late, rare positive classes, and distributions that drift or are adversarial. It implements conformal PID control (Angelopoulos, Candès & Tibshirani, 2023) with per-class (Mondrian) thresholds, delay-aware updates following El Halabi & Brandt (2026), and optional label-shift reweighting (Podkopaev & Ramdas, 2021). Transactions whose prediction set contains both classes are flagged as uncertain and can be sent to manual review, with tools to tune coverage against review budget. The package includes metrics for per-class and rolling coverage and simulators that add realistic chargeback delays to public datasets such as IEEE-CIS and BAF. It follows scikit-learn and River conventions and needs only NumPy.

## Installation

### Using `uv` (recommended)

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
