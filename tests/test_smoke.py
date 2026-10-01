"""Smoke test: the package imports and exposes a version string."""

import delaycp


def test_version_is_a_string() -> None:
    assert isinstance(delaycp.__version__, str)
    assert delaycp.__version__
