"""Shared guard for tests that read the manuscript sources.

The manuscript lives under `development/`, which is gitignored: a public clone has
the code, the tests and the artifacts, but not the paper. Those tests must skip
there rather than error on a missing path, so the suite a reader runs is honest
about what it can and cannot check.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DEV = ROOT / "development"
DOCS = DEV / "docs"

requires_dev_tree = pytest.mark.skipif(
    not DOCS.exists(),
    reason="manuscript sources live under development/, which a public clone does not carry")
