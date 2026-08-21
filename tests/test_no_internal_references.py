"""The public tree must not name internal workstreams, decision points, or private paths."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

PATTERNS = {
    "workstream code": re.compile(r"\bWS[-_ ]?\d[a-z]?\b"),
    "checkpoint code": re.compile(r"\bCPR?\d+[a-z]?\b"),
    "arm code": re.compile(r"(?<![A-Za-z0-9_])A[1-7](?![A-Za-z0-9]|_TO_)"),
    "private run notes": re.compile(r"(?<!development/)\breports/"),
    "internal planning doc": re.compile(r"\bresearch plan\b|\bstanding note\b"),
}

# scripts/common/arms.py is the sanctioned place for the historical arm codes: it exists
# precisely so artifacts stamped with them stay readable. This test is the other one.
ALLOWED_FILES = {"scripts/common/arms.py", "tests/test_no_internal_references.py"}

# README.md documents the legacy codes on purpose, in the same spirit as arms.py: a reader
# opening an older artifact needs somewhere that says what A1..A7 meant.
ALLOWED_PATTERNS_BY_FILE = {"README.md": {"arm code"}}

# Payload values inside already-emitted artifacts are covered by payload_sha256 and cannot
# be re-stamped without the checkpoints that produced them, which are not distributed.
# equivalence.json records where its pre-registered margin was read from, which is a
# factual provenance record rather than a leak.
ALLOWED_ARTIFACTS = {"equivalence.json", "additive_control.json",
                     "serialization_expressibility.json", "training_summary.json"}


def _tracked(prefix: str) -> list[Path]:
    out = subprocess.run(["git", "ls-files", prefix], cwd=ROOT,
                         capture_output=True, text=True).stdout.split()
    return [ROOT / p for p in out]


def _sources() -> list[Path]:
    files = _tracked("scripts") + _tracked("tests") + [ROOT / "README.md"]
    return [f for f in files if f.suffix in {".py", ".md"}
            and str(f.relative_to(ROOT)) not in ALLOWED_FILES]


@pytest.mark.parametrize("path", _sources(), ids=lambda p: str(p.relative_to(ROOT)))
def test_source_names_nothing_internal(path: Path) -> None:
    rel = str(path.relative_to(ROOT))
    exempt = ALLOWED_PATTERNS_BY_FILE.get(rel, set())
    text = path.read_text()
    found = [(label, m.group(0), text.count("\n", 0, m.start()) + 1)
             for label, pat in PATTERNS.items() if label not in exempt
             for m in pat.finditer(text)]
    assert not found, (
        f"{path.relative_to(ROOT)} names internal project vocabulary a reader cannot "
        f"resolve: {found}. Say what the thing is instead of citing where it was tracked.")


@pytest.mark.parametrize("path", _tracked("results"), ids=lambda p: p.name)
def test_artifact_names_nothing_internal(path: Path) -> None:
    if path.name in ALLOWED_ARTIFACTS:
        pytest.skip(f"{path.name} carries codes inside a checksummed payload; see ALLOWED_ARTIFACTS")
    text = path.read_text()
    found = [(label, m.group(0)) for label, pat in PATTERNS.items()
             if label != "arm code"
             for m in pat.finditer(text)]
    assert not found, (
        f"{path.name} carries internal vocabulary: {found}. Regenerate it by rerunning "
        f"the script in its 'script' field.")
