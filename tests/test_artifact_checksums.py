"""Every artifact's payload must still hash to the checksum stamped beside it."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# Records written before payload_sha256 entered the schema. They carry no checksum to
# check, which is a fact about their age, not a mismatch.
NO_CHECKSUM = {
    "hessian_speedup_sweep.json", "retarget_t10_s1.json", "retarget_t10_s2.json",
    "scoping_recomputed.json", "steering_frontier_rows.json", "training_summary.json",
}

# These two do not hash to their stamped checksum and did not before this test existed.
# Their payloads were altered after emission, so the provenance stamp no longer certifies
# the content. Regenerating them needs a checkpoint that is not distributed; until then the
# mismatch is recorded here rather than hidden.
KNOWN_MISMATCH = {"overhead_benchmark.json", "smc_steering.json"}


def _tracked() -> list[Path]:
    out = subprocess.run(["git", "ls-files", "results"], cwd=ROOT,
                         capture_output=True, text=True).stdout.split()
    return [ROOT / p for p in out if p.endswith(".json")]


@pytest.mark.parametrize("path", _tracked(), ids=lambda p: p.name)
def test_payload_matches_its_checksum(path: Path) -> None:
    if path.name in NO_CHECKSUM:
        pytest.skip(f"{path.name} predates the payload_sha256 field")
    if path.name in KNOWN_MISMATCH:
        pytest.xfail(f"{path.name} has a stale checksum; see KNOWN_MISMATCH")
    record = json.loads(path.read_text())
    blob = json.dumps(record["payload"], sort_keys=True, separators=(",", ":"), default=str)
    assert hashlib.sha256(blob.encode()).hexdigest() == record["payload_sha256"], (
        f"{path.name} does not hash to its stamped payload_sha256: the payload was changed "
        f"after emission. Rerun the script in its 'script' field rather than editing it.")
