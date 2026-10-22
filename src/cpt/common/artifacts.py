from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections.abc import Mapping
from pathlib import Path

SCHEMA_VERSION = "2.0"
RESULTS_DIR = Path(os.environ.get("CPT_RESULTS_DIR", "results")).expanduser()


def json_value(value: object) -> object:
    """Encode nonfinite numbers explicitly and keep JSON null for unavailable values."""
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite": "nan" if math.isnan(value) else ("+inf" if value > 0 else "-inf")}
    if isinstance(value, Mapping):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, Path):
        return value.name
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported record value: {type(value).__name__}")


def file_sha256(path: str | Path) -> str:
    """Hash a file without loading its full contents into memory."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical(value: object) -> bytes:
    return json.dumps(json_value(value), sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def config_hash(config: Mapping[str, object]) -> str:
    return hashlib.sha256(canonical(config)).hexdigest()[:16]


def emit(name: str, payload: Mapping[str, object], *, config: Mapping[str, object],
         seeds: list[int], script: str, results_dir: Path = RESULTS_DIR) -> Path:
    """Write a new record without overwriting supplied evidence."""
    if not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError("Record name must be a filename stem")
    results_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "schema_version": SCHEMA_VERSION,
        "artifact_name": name,
        "script": script,
        "config": json_value(config),
        "config_hash": config_hash(config),
        "seeds": seeds,
        "payload_sha256": hashlib.sha256(canonical(payload)).hexdigest(),
        "payload": json_value(payload),
    }
    path = results_dir / f"{name}.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return path


def load(name: str, results_dir: Path = RESULTS_DIR) -> dict[str, object]:
    """Read a record and check payload and configuration integrity."""
    record = json.loads((results_dir / f"{name}.json").read_text())
    if not isinstance(record, dict):
        raise ValueError("Record must be an object")
    actual = hashlib.sha256(canonical(record["payload"])).hexdigest()
    if actual != record["payload_sha256"]:
        raise ValueError(f"{name}: payload checksum mismatch")
    if config_hash(record["config"]) != record["config_hash"]:
        raise ValueError(f"{name}: configuration checksum mismatch")
    return record


def assert_fresh(name: str, max_age_s: float = 3600.0,
                 results_dir: Path = RESULTS_DIR) -> bool:
    """Check that an output was written within the requested interval."""
    age = time.time() - (results_dir / f"{name}.json").stat().st_mtime
    if age > max_age_s:
        raise ValueError(f"{name}: output is older than {max_age_s:g} seconds")
    return True
