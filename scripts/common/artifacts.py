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
    """Commit of the SOURCE tree, or a sentinel when version control is absent.

    Never fabricate a hash: a missing commit is recorded as such so the report
    builder can flag artifacts that predate version control.

    Dirtiness is judged over source only — `scripts/` and `tests/` — and
    deliberately excludes `results/`, `reports/`, and `figures/`. The reason is
    ordering, not convenience. An artifact records the code that produced it, and
    it necessarily exists BEFORE the commit that contains it, so its own presence
    in the working tree must not count against the provenance of the code. When
    several scripts are re-emitted in sequence the first one's output would
    otherwise dirty the tree for every script after it, stamping them `-dirty`
    against a commit whose source was in fact pristine — a misleading record,
    since nothing about the code differed between the runs.

    What `-dirty` means here is therefore precise and useful: the code that
    produced this artifact is not exactly the code at the recorded commit.
    """
    try:
        head = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        if head.returncode != 0 or not head.stdout.strip():
            return "UNVERSIONED"
        dirty = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "status", "--porcelain",
             "--", "scripts", "tests"],
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


def assert_fresh(name: str, max_age_s: float = 3600.0,
                 results_dir: Path | None = None) -> None:
    """Fail if an artifact on disk predates the current run.

    Guards the failure mode where a measurement script raises, its stderr is
    suppressed, and the PREVIOUS run's artifact is still on disk — so a reader
    sees a plausible file and no error. Call this after a run that was expected to
    re-emit, or in a report builder before consuming an artifact.
    """
    out_dir = Path(results_dir) if results_dir is not None else RESULTS_DIR
    path = out_dir / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"artifact {name} was never emitted at {path}")
    age = datetime.now(timezone.utc).timestamp() - path.stat().st_mtime
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
