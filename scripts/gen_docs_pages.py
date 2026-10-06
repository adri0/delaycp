"""Regenerate the docs pages that are built from other files.

Usage, from the repository root::

    uv run --extra docs python -m scripts.gen_docs_pages

Writes:

- ``docs/examples/ieee_cis_chargebacks.md`` (plus its figures) from the
  committed outputs of ``examples/ieee_cis_chargebacks.ipynb``. The notebook
  is not executed; re-run it first if the example changed.
- ``docs/validation.md`` from ``python -m benchmarks.scenario_report``
  (about half a minute). Skip with ``--no-report``.
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

from nbconvert import MarkdownExporter

from benchmarks.scenario_report import report

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "examples" / "ieee_cis_chargebacks.ipynb"
EXAMPLES_DIR = ROOT / "docs" / "examples"
BLOB = "https://github.com/adri0/delaycp/blob/main/examples/"


def notebook_page() -> None:
    stem = NOTEBOOK.stem
    body, resources = MarkdownExporter().from_filename(
        str(NOTEBOOK), resources={"output_files_dir": f"{stem}_files"}
    )
    shutil.rmtree(EXAMPLES_DIR / f"{stem}_files", ignore_errors=True)
    for name, data in resources["outputs"].items():
        path = EXAMPLES_DIR / name  # name already includes output_files_dir
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    body = body.replace("![png](", "![Figure](")
    # Links relative to examples/ point at files that are not part of the site.
    body = re.sub(r"\]\((?!https?://|#)([^)]+\.(?:py|ipynb))\)", rf"]({BLOB}\1)", body)
    note = (
        '!!! note "Generated from a notebook"\n'
        f"    This page is [`examples/{NOTEBOOK.name}`]({BLOB}{NOTEBOOK.name}) with the "
        "outputs committed alongside it. The data is not redistributed; download it "
        "from Kaggle to run the notebook yourself.\n\n"
    )
    title, _, rest = body.lstrip().partition("\n")
    (EXAMPLES_DIR / f"{stem}.md").write_text(f"{title}\n\n{note}{rest.lstrip()}", "utf-8")


def validation_page() -> None:
    (ROOT / "docs" / "validation.md").write_text(report(), "utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--no-report", action="store_true", help="skip the validation report")
    args = parser.parse_args()
    notebook_page()
    if not args.no_report:
        validation_page()


if __name__ == "__main__":
    main()
