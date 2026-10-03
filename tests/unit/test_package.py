"""Smoke tests: the package imports and exposes a version."""

import codetrack


def test_version_is_string() -> None:
    assert isinstance(codetrack.__version__, str)


def test_version_is_semver_like() -> None:
    parts = codetrack.__version__.split(".")
    assert len(parts) == 3 and all(p.isdigit() for p in parts)
