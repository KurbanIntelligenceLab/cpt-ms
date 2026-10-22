"""Artifact emission convention."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1.0"
REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = REPO_ROOT / "results"


def _git_commit() -> str:
    try:
        head = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        if head.returncode != 0 or not head.stdout.strip():
            return "UNVERSIONED"
        dirty = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "status", "--porcelain",
             "--", "src", "tests"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        return head.stdout.strip() + ("-dirty" if dirty else "")
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
    """Write a results artifact and return its path."""
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
        "created_utc": datetime.now(UTC).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "payload_sha256": hashlib.sha256(payload_blob.encode()).hexdigest(),
        "payload": payload,
    }

    path = out_dir / f"{name}.json"
    path.write_text(json.dumps(record, indent=2, sort_keys=False, default=str))
    return path


def assert_fresh(name: str, max_age_s: float = 3600.0,
                 results_dir: Path | None = None) -> None:
    """Fail if an artifact on disk predates the current run."""
    out_dir = Path(results_dir) if results_dir is not None else RESULTS_DIR
    path = out_dir / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"artifact {name} was never emitted at {path}")
    age = datetime.now(UTC).timestamp() - path.stat().st_mtime
    if age > max_age_s:
        raise ValueError(
            f"artifact {name} is {age / 60:.1f} min old (limit {max_age_s / 60:.1f}); "
            "a run that was expected to re-emit it may have failed silently"
        )


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
