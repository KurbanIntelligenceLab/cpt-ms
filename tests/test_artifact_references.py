"""An artifact that names another artifact must name one that exists."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

REF = re.compile(r"results/[A-Za-z0-9_.*{}-]+\.json")

KNOWN_STALE = {"smc_tuned.json", "smc_tuned_smoke.json"}


def _artifacts() -> list[Path]:
    return sorted(p for p in RESULTS.glob("*.json") if p.name != "run_ledger.jsonl")


@pytest.mark.parametrize("artifact", _artifacts(), ids=lambda p: p.name)
def test_referenced_artifacts_exist(artifact: Path) -> None:
    if artifact.name in KNOWN_STALE:
        pytest.skip(f"{artifact.name} needs a checkpoint to regenerate; see KNOWN_STALE")
    dangling = sorted(
        ref for ref in set(REF.findall(artifact.read_text()))
        if "*" not in ref and "{" not in ref and not (ROOT / ref).exists()
    )
    assert not dangling, (
        f"{artifact.name} names artifacts that do not exist: {dangling}. "
        f"Regenerate it by rerunning the script in its 'script' field.")


@pytest.mark.parametrize("artifact", _artifacts(), ids=lambda p: p.name)
def test_script_field_names_a_file_that_exists(artifact: Path) -> None:
    """The 'script' stamp is a rerun instruction, so it has to still point somewhere."""
    record = json.loads(artifact.read_text())
    if "schema_version" not in record:
        pytest.skip(f"{artifact.name} predates the schema and carries no 'script' field")
    script = record.get("script")
    assert script, f"{artifact.name} carries no 'script' provenance field"
    if script.startswith("development/"):
        pytest.skip(f"{artifact.name} was emitted from development/, which is not distributed")
    assert (ROOT / script).exists(), (
        f"{artifact.name} was emitted by {script}, which no longer exists. Point it at the "
        f"module that replaced it; a rerun instruction nobody can follow is not provenance.")
