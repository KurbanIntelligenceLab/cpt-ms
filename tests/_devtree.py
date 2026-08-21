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


DATA = ROOT / "data"

requires_mp20 = pytest.mark.skipif(
    not (DATA / "mp20_train.csv").exists(),
    reason="MP-20 CSVs are ~58MB and not tracked; fetch them into data/ from the CDVAE repo")


def requires(module: str) -> pytest.MarkDecorator:
    """Skip when an optional dependency is absent.

    A missing extra is an install state, not a failed claim. Without this a reader
    who ran a bare `uv sync` sees red and reasonably concludes the results do not
    reproduce, when nothing has been measured at all.
    """
    return pytest.mark.skipif(
        __import__("importlib.util", fromlist=["util"]).find_spec(module) is None,
        reason=f"{module} is not installed; it comes with the 'crystal' extra")
