"""Reports must not carry counts that contradict the repository."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
REPORTS = REPO / "development" / "reports"

WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20,
}


def _counts() -> dict[str, int]:
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
    """Any '<N> [tests] in `tests/<file>`' claim must match that file."""
    pattern = re.compile(
        r"(\d+|[A-Za-z]+)\s+(?:tests?\s+)?in\s+`tests/([A-Za-z0-9_]+\.py)`", re.I
    )
    problems, checked = [], 0
    for md in sorted(REPORTS.glob("*.md")):
        text = md.read_text()
        for m in pattern.finditer(text):
            claimed, fname = _as_int(m.group(1)), m.group(2)
            if claimed is None or fname not in counts:
                continue
            checked += 1
            if claimed != counts[fname]:
                problems.append(f"{md.name}: claims {claimed} for {fname}, actual {counts[fname]}")
    assert not problems, "; ".join(problems)
    assert checked >= 15, f"only {checked} per-file claims parsed; the check is not binding"


def test_multi_file_count_claims_are_current(counts):
    """'<N> tests across `a.py` and `b.py`' must equal the sum of those files."""
    pattern = re.compile(
        r"([A-Za-z]+|\d+)\s+tests?\s+across\s+((?:`tests/[A-Za-z0-9_]+\.py`"
        r"(?:\s*\(\d+\))?[\s,]*(?:and\s*)?)+)",
        re.I,
    )
    fname_re = re.compile(r"`tests/([A-Za-z0-9_]+\.py)`")
    problems, checked = [], 0
    for md in sorted(REPORTS.glob("*.md")):
        for m in pattern.finditer(md.read_text()):
            claimed = _as_int(m.group(1))
            files = fname_re.findall(m.group(2))
            if claimed is None or len(files) < 2:
                continue
            if any(f not in counts for f in files):
                continue
            total = sum(counts[f] for f in files)
            checked += 1
            if claimed != total:
                problems.append(
                    f"{md.name}: claims {claimed} across {files}, actual {total}"
                )
    assert not problems, "; ".join(problems)
    assert checked >= 1, "no multi-file count claim parsed; the check is not binding"


def test_token_position_claims_match_the_artifact():
    """Reports must not carry token ranges that contradict their own source."""
    import json

    art = REPO / "results" / "family_determinability.json"
    note = REPORTS / "family_determinability_notes.md"
    if not art.exists() or not note.exists():
        pytest.skip("determinability artifact or note not present")

    rows = json.loads(art.read_text())["payload"]["rows"]

    def observed(backbone: str, family: str):
        sel = [r for r in rows if r["backbone"] == backbone and r["family"] == family]
        pos = sorted({r["first_determined_position"] for r in sel})
        tok = sorted({r["n_tokens"] for r in sel})
        return (pos[0], pos[-1]), (tok[0], tok[-1])

    labels = {"B1 (CIF chars)": "B1", "B1ref": "B1ref", "B2 (fields)": "B2"}
    fams = {"(i) charge": "i_charge", "(ii) density": "ii_density"}

    pattern = re.compile(
        r"\|\s*(?P<label>[^|]+?)\s*\|\s*(?P<family>\(i+\)[^|]+?)\s*\|"
        r"[^|]*\|[^|]*\|\s*token\s*(?P<a>\d+)(?:[\u2013-](?P<b>\d+))?"
        r"\s*of\s*(?P<c>\d+)[\u2013-](?P<d>\d+)"
    )
    checked = 0
    for m in pattern.finditer(note.read_text()):
        bb = labels.get(m.group("label").strip())
        fam = fams.get(m.group("family").strip())
        if bb is None or fam is None:
            continue
        pos, tok = observed(bb, fam)
        a = int(m.group("a"))
        b = int(m.group("b")) if m.group("b") else a
        assert (a, b) == pos, f"{bb}/{fam}: note says det {a}-{b}, artifact {pos}"
        assert (int(m.group("c")), int(m.group("d"))) == tok, (
            f"{bb}/{fam}: note says {m.group('c')}-{m.group('d')} tokens, artifact {tok}"
        )
        checked += 1
    assert checked >= 4, f"only {checked} position claims parsed; the check is not binding"

    all_tokens = sorted({r["n_tokens"] for r in rows})
    valid_spans = {
        (min(r["n_tokens"] for r in rows if r["backbone"] == bb),
         max(r["n_tokens"] for r in rows if r["backbone"] == bb))
        for bb in {r["backbone"] for r in rows}
    }
    prose = re.compile(
        r"token\s*(?P<a>\d+)(?:[\u2013-](?P<b>\d+))?\s*of\s*(?P<c>\d+)[\u2013-](?P<d>\d+)"
    )
    prose_checked = 0
    for m in prose.finditer(note.read_text()):
        span = (int(m.group("c")), int(m.group("d")))
        assert span in valid_spans, (
            f"token span {span[0]}-{span[1]} matches no backbone's measured "
            f"n_tokens range; measured spans are {sorted(valid_spans)}"
        )
        a = int(m.group("a"))
        assert min(all_tokens) <= a <= max(all_tokens) or a <= span[1], (
            f"determined position {a} lies outside every measured token range"
        )
        prose_checked += 1
    assert prose_checked >= checked, "prose scan must cover at least the table rows"


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
