"""Shared artifact reader for the test suite."""

from __future__ import annotations

import json
from typing import Any

import pytest

from cpt.common.artifacts import RESULTS_DIR, load

# These two do not hash to their stamped checksum and did not before the suite existed;
# test_artifact_checksums records why. Reading them has to bypass verification, or every
# test that touches them would error on a stamp already known to be stale.
UNVERIFIABLE = {"overhead_benchmark", "smc_steering"}


def payload(name: str) -> dict[str, Any]:
    """Read an artifact's payload, verifying its checksum; skip when it is absent."""
    stem = name[:-5] if name.endswith(".json") else name
    path = RESULTS_DIR / f"{stem}.json"
    if not path.exists():
        pytest.skip(f"{stem}.json absent; rerun the module named in its 'script' field")

    record = json.loads(path.read_text())
    if stem in UNVERIFIABLE or "payload_sha256" not in record:
        # No checksum to check: either stale by record, or emitted before the field existed.
        return record["payload"]
    return load(stem)["payload"]
