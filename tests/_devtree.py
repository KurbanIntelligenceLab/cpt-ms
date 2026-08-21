"""Shared guard for tests that read the manuscript sources."""

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
    """Skip when an optional dependency is absent."""
    return pytest.mark.skipif(
        __import__("importlib.util", fromlist=["util"]).find_spec(module) is None,
        reason=f"{module} is not installed; it comes with the 'crystal' extra")
