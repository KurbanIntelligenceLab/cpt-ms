"""Artifact emission convention.

Every script that produces a number writes it through `emit`, which stamps the
payload with the emitting script's identity, a hash of its configuration, the
seeds used, and a content checksum. Reports are built from these files; nothing
quantitative is ever typed by hand.

This is the day-one logging convention referred to in the reporting protocol.
The hash-enforced report builder (due CP3) consumes exactly this schema.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1.0"
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"


def _git_commit() -> str:
    """Repo commit, or an explicit sentinel when version control is unavailable.

    Never fabricate a hash: a missing commit is recorded as such so the report
    builder can flag artifacts that predate version control.
    """
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0 and out.stdout.strip():
            dirty = subprocess.run(
                ["git", "-C", str(REPO_ROOT), "status", "--porcelain"],
                capture_output=True, text=True, timeout=10,
            ).stdout.strip()
            return out.stdout.strip() + ("-dirty" if dirty else "")
    except Exception:
        pass
    return "UNVERSIONED"


def config_hash(config: dict[str, Any]) -> str:
    """Stable hash of a configuration dict, used to key smoke tests to runs."""
    blob = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def emit(
    name: str,
    payload: dict[str, Any],
    *,
    config: dict[str, Any],
    seeds: list[int],
    script: str,
    results_dir: Path | None = None,
) -> Path:
    """Write a results artifact and return its path.

    `name` becomes `results/<name>.json`. The payload is wrapped in a record
    carrying provenance; the content checksum covers the payload only, so
    re-running with identical inputs produces an identical checksum even though
    the timestamp differs.
    """
    out_dir = Path(results_dir) if results_dir is not None else RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    payload_blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    record = {
        "schema_version": SCHEMA_VERSION,
        "artifact_name": name,
        "script": script,
        "git_commit": _git_commit(),
        "config": config,
        "config_hash": config_hash(config),
        "seeds": seeds,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "payload_sha256": hashlib.sha256(payload_blob.encode()).hexdigest(),
        "payload": payload,
    }

    path = out_dir / f"{name}.json"
    path.write_text(json.dumps(record, indent=2, sort_keys=False, default=str))
    return path


def load(name: str, results_dir: Path | None = None) -> dict[str, Any]:
    """Read an artifact and verify its payload checksum."""
    out_dir = Path(results_dir) if results_dir is not None else RESULTS_DIR
    record = json.loads((out_dir / f"{name}.json").read_text())
    blob = json.dumps(record["payload"], sort_keys=True, separators=(",", ":"), default=str)
    actual = hashlib.sha256(blob.encode()).hexdigest()
    if actual != record["payload_sha256"]:
        raise ValueError(
            f"artifact {name}: payload checksum mismatch "
            f"(recorded {record['payload_sha256'][:12]}, actual {actual[:12]})"
        )
    return record
