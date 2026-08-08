"""Reports must not carry counts that contradict the repository.

A hand-typed test count in a standing note drifts the moment a test is added,
and it drifts silently: nothing fails, the note simply becomes wrong. This
collects the real counts and checks every claim a report makes about them.

The same class of defect — a narrative number stored beside the thing it
describes, with no mechanism tying them together — was already fixed once in the
determinability artifact by deriving its verdicts. This closes it for the prose
reports, which cannot derive their text but can at least be checked.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
REPORTS = REPO / "reports"

WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20,
}


def _counts() -> dict[str, int]:
    """Tests per file, collected from pytest rather than assumed."""
    out = subprocess.run(
        [sys.executable, "-m", "pytest", str(REPO / "tests"), "-q", "--collect-only"],
        capture_output=True, text=True, cwd=REPO,
        env={"PYTHONPATH": str(REPO), "PATH": "/usr/bin:/bin"},
    )
    per: dict[str, int] = {}
    for line in out.stdout.splitlines():
        if "::" in line:
            f = line.split("::", 1)[0].strip()
            per[Path(f).name] = per.get(Path(f).name, 0) + 1
    if not per:
        pytest.skip("could not collect tests in this environment")
    return per


def _as_int(token: str) -> int | None:
    token = token.strip().lower().replace(",", "")
    if token.isdigit():
        return int(token)
    return WORDS.get(token)


@pytest.fixture(scope="module")
def counts():
    return _counts()


def test_per_file_counts_in_reports_are_current(counts):
    """Any '<N> tests in `tests/<file>`' claim must match that file."""
    pattern = re.compile(
        r"([A-Za-z]+|\d+)\s+tests?\s+in\s+`tests/([A-Za-z0-9_]+\.py)`", re.I
    )
    problems = []
    for md in sorted(REPORTS.glob("*.md")):
        text = md.read_text()
        for m in pattern.finditer(text):
            claimed, fname = _as_int(m.group(1)), m.group(2)
            if claimed is None or fname not in counts:
                continue
            if claimed != counts[fname]:
                problems.append(f"{md.name}: claims {claimed} for {fname}, actual {counts[fname]}")
    assert not problems, "; ".join(problems)


def test_suite_totals_in_reports_are_current(counts):
    """Any 'all N tests' / 'suite of N' claim must match the whole suite."""
    total = sum(counts.values())
    pattern = re.compile(
        r"(?:all|suite of)\s+([A-Za-z]+|\d+)\s+tests|"
        r"([A-Za-z]+|\d+)\s+tests\s+in\s+the\s+suite",
        re.I,
    )
    problems = []
    for md in sorted(REPORTS.glob("*.md")):
        for m in pattern.finditer(md.read_text()):
            claimed = _as_int(m.group(1) or m.group(2) or "")
            if claimed is not None and claimed != total:
                problems.append(f"{md.name}: claims {claimed}, actual {total}")
    assert not problems, "; ".join(problems)
